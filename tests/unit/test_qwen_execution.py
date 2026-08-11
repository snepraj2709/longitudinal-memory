from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from evaluation.openai_client import OpenAIResponseMetadata
from evaluation.qwen_execution import (
    ExecutionJob,
    QwenExecutionError,
    TransportRetryLedger,
    _answer_max_output_tokens,
    _evidence_index,
    _normalize_qwen_answer_response,
    build_extraction_jobs,
    derive_b7_records,
    execute_jobs,
    extraction_rows_from_records,
    write_derived_b7,
)
from evaluation.qwen_materialization import ContextPackage
from evaluation.vllm_client import VLLMResponseError, VLLMTransportError


ROOT = Path(__file__).resolve().parents[2]


def _output(answer: str = "Known fact.") -> dict[str, object]:
    return {
        "status": "answered",
        "answer": answer,
        "confidence": 0.8,
        "statements": [answer],
        "citations": [{"source_id": "source_1", "message_id": "message_1", "quote": "Known fact"}],
        "unresolved_parts": [],
        "abstention_reason": None,
    }


def _job(position: int, validator=None) -> ExecutionJob:
    return ExecutionJob(
        request_id=f"request_{position}",
        position=position,
        split="development",
        task="qa",
        record_id=f"case_{position}",
        user_id="user_001",
        baseline_id="B6",
        context_sha256="a" * 64,
        context_count=1,
        system_prompt="system",
        user_prompt=f"user {position}",
        response_format={"type": "json_object"},
        max_output_tokens=100,
        validator=validator or (lambda raw, _metadata: json.loads(raw)),
    )


class _Client:
    def __init__(self, call, active=None) -> None:
        self.call = call
        self.active = active

    def complete_with_metadata(self, **_kwargs):
        if self.active is not None:
            with self.active["lock"]:
                self.active["current"] += 1
                self.active["maximum"] = max(self.active["maximum"], self.active["current"])
            time.sleep(0.01)
        try:
            return self.call()
        finally:
            if self.active is not None:
                with self.active["lock"]:
                    self.active["current"] -= 1


def _success():
    raw = json.dumps(_output())
    metadata = OpenAIResponseMetadata("response_1", "qwen35-27b-fp8-v2", 10, 5, 15)
    return raw, metadata


class QwenExecutionTests(unittest.TestCase):
    def test_extraction_job_counts_match_the_frozen_splits(self) -> None:
        development = build_extraction_jobs(ROOT, "development")
        frozen = build_extraction_jobs(ROOT, "test")
        self.assertEqual(len(development), 20)
        self.assertEqual(len(frozen), 80)
        self.assertTrue(all(job.task == "extraction" for job in development + frozen))

    def test_extraction_validator_coerces_qwen_string_booleans(self) -> None:
        job = next(
            item
            for item in build_extraction_jobs(ROOT, "development", series_id="qwen3-8b-vllm-dev-v1")
            if item.record_id == "scaled_user_001_conversation_002"
        )
        raw = json.dumps({
            "claims": [{
                "claim_id": "claim_001",
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "feels_exhausted",
                "object": "true",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": "2026-05-18T09:30:00+00:00",
                "valid_to": None,
                "confidence": 1,
                "evidence": [{
                    "source_id": "scaled_user_001_conversation_002",
                    "message_id": "scaled_user_001_conversation_002_message_001",
                    "quote": "This month I feel exhausted after the launch.",
                }],
            }],
        })

        output = job.validator(raw, OpenAIResponseMetadata("response", "qwen3-8b-vllm", 1, 1, 2))

        self.assertIs(output["claims"][0]["object"], True)

    def test_answer_validator_normalizes_qwen_abstentions(self) -> None:
        normalized = _normalize_qwen_answer_response({
            "status": "abstained",
            "answer": "",
            "confidence": 0,
            "statements": [],
            "citations": [],
            "unresolved_parts": ["Which university did Asha attend?"],
            "abstention_reason": "The context records are empty.",
        }, "qa")

        self.assertEqual(normalized["status"], "abstained")
        self.assertEqual(normalized["answer"], "The context records are empty.")
        self.assertEqual(normalized["unresolved_parts"], [])

    def test_answer_validator_normalizes_qwen_citation_substrings(self) -> None:
        exact_quote = (
            "I need to correct the start date. I began on 2026-01-11, not "
            "2026-02-03. My current goal is still to build reliable health tools."
        )
        normalized = _normalize_qwen_answer_response(
            {
                "status": "answered",
                "answer": "Asha began on 2026-01-11.",
                "confidence": 0.95,
                "statements": ["Asha began on 2026-01-11."],
                "citations": [{
                    "source_id": "scaled_user_001_conversation_003",
                    "message_id": "scaled_user_001_conversation_003_message_001",
                    "quote": "I need to correct the start date. I began on 2026-01-11, not 2026-02-03.",
                }],
                "unresolved_parts": ["Extra unresolved text."],
                "abstention_reason": "",
            },
            "qa",
            {("scaled_user_001_conversation_003", "scaled_user_001_conversation_003_message_001", exact_quote): object()},
        )

        self.assertIsNone(normalized["abstention_reason"])
        self.assertEqual(normalized["unresolved_parts"], [])
        self.assertEqual(normalized["citations"][0]["quote"], exact_quote)

    def test_answer_validator_normalizes_qwen_calendar_message_ids(self) -> None:
        evidence = _evidence_index([{
            "record_kind": "source",
            "source_id": "scaled_user_001_calendar_002",
            "content": "Care Map review is scheduled for 2026-08-11.",
            "evidence": [],
        }])

        normalized = _normalize_qwen_answer_response(
            {
                "status": "answered",
                "answer": "The Care Map review is scheduled for 2026-08-11.",
                "confidence": 1,
                "statements": ["The Care Map review is scheduled for 2026-08-11."],
                "citations": [{
                    "source_id": "scaled_user_001_calendar_002",
                    "message_id": "scaled_user_001_calendar_002_message_001",
                    "quote": "Care Map review is scheduled for 2026-08-11.",
                }],
                "unresolved_parts": [],
                "abstention_reason": "",
            },
            "qa",
            evidence,
        )

        self.assertIsNone(normalized["citations"][0]["message_id"])
        self.assertEqual(
            normalized["citations"][0]["quote"],
            "Care Map review is scheduled for 2026-08-11.",
        )

    def test_summary_answers_have_larger_output_budget(self) -> None:
        self.assertEqual(_answer_max_output_tokens("summary"), 1600)
        self.assertEqual(_answer_max_output_tokens("qa"), 1000)
        self.assertEqual(_answer_max_output_tokens("interactive"), 1000)

    def test_concurrent_atomic_checkpoints_are_composed_in_plan_order(self) -> None:
        active = {"current": 0, "maximum": 0, "lock": threading.Lock()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            manifest = execute_jobs(
                tuple(_job(index) for index in range(1, 10)),
                output_dir=output,
                client_factory=lambda _job: _Client(_success, active),
            )
            rows = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([row["position"] for row in rows], list(range(1, 10)))
            self.assertGreater(active["maximum"], 1)
            self.assertEqual(len(list((output / "requests").glob("*.json"))), 9)
            self.assertEqual(manifest["successful_count"], 9)
            self.assertEqual(manifest["worker_count"], 8)

    def test_single_worker_execution_is_allowed_for_small_vllm(self) -> None:
        active = {"current": 0, "maximum": 0, "lock": threading.Lock()}
        with tempfile.TemporaryDirectory() as directory:
            manifest = execute_jobs(
                tuple(_job(index) for index in range(1, 4)),
                output_dir=Path(directory) / "batch",
                client_factory=lambda _job: _Client(_success, active),
                workers=1,
            )
        self.assertEqual(active["maximum"], 1)
        self.assertEqual(manifest["successful_count"], 3)
        self.assertEqual(manifest["worker_count"], 1)

    def test_resume_does_not_repeat_terminal_requests(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            factory = lambda job: _Client(lambda: (calls.append(job.request_id), _success())[1])
            execute_jobs((_job(1), _job(2)), output_dir=output, client_factory=factory)
            execute_jobs((_job(1), _job(2)), output_dir=output, client_factory=factory)
            self.assertEqual(calls, ["request_1", "request_2"])

    def test_only_no_body_transport_failure_retries_once(self) -> None:
        attempts = {"request_1": 0, "request_2": 0}

        def factory(job):
            def call():
                attempts[job.request_id] += 1
                if attempts[job.request_id] == 1:
                    raise VLLMTransportError("no body")
                return _success()
            return _Client(call)

        with tempfile.TemporaryDirectory() as directory:
            manifest = execute_jobs(
                (_job(1), _job(2)),
                output_dir=Path(directory) / "batch",
                client_factory=factory,
                retry_ledger=TransportRetryLedger(1),
            )
            self.assertEqual(sum(attempts.values()), 3)
            self.assertEqual(manifest["transport_retry_count"], 1)
            self.assertEqual(manifest["failure_count"], 1)

    def test_invalid_output_is_not_retried_or_persisted_raw(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            execute_jobs(
                (_job(1, validator=lambda _raw, _meta: (_ for _ in ()).throw(ValueError("invalid secret raw"))),),
                output_dir=output,
                client_factory=lambda _job: _Client(lambda: (calls.append(1), _success())[1]),
            )
            row = json.loads(next((output / "requests").glob("*.json")).read_text())
            self.assertEqual(calls, [1])
            self.assertEqual(row["failure_stage"], "validation")
            self.assertNotIn("invalid secret raw", json.dumps(row))

    def test_http_response_error_is_not_retried(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            execute_jobs(
                (_job(1),),
                output_dir=output,
                client_factory=lambda _job: _Client(
                    lambda: (calls.append(1), (_ for _ in ()).throw(
                        VLLMResponseError("bad request", status_code=400)
                    ))[1]
                ),
            )
            row = json.loads(next((output / "requests").glob("*.json")).read_text())
            self.assertEqual(calls, [1])
            self.assertEqual(row["failure_stage"], "provider")
            self.assertEqual(row["failure_code"], "http_400")

    def test_checkpoint_request_hash_mismatch_stops_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            execute_jobs((_job(1),), output_dir=output, client_factory=lambda _job: _Client(_success))
            changed = _job(1)
            object.__setattr__(changed, "user_prompt", "changed")
            with self.assertRaisesRegex(QwenExecutionError, "request hash"):
                execute_jobs((changed,), output_dir=output, client_factory=lambda _job: _Client(_success))

    def test_b7_reuses_b6_context_and_underlying_answer_hash_without_call(self) -> None:
        context = ContextPackage(
            "qwen35-27b-fp8-v2", "development", "qa", "case_1", "user_001",
            "2026-12-01T12:00:00+00:00", "B7", False, "exact_b6", ({
                "lifecycle_statuses": ["current"],
                "evidence": [{"source_id": "source_1", "message_id": "message_1", "quote": "Known fact"}],
            },), "a" * 64,
        )
        underlying = _output()
        from evaluation.qwen_execution import _sha
        rows = derive_b7_records((context,), ({
            "status": "succeeded", "task": "qa", "record_id": "case_1", "baseline_id": "B6",
            "context_sha256": "a" * 64, "output": underlying,
            "underlying_answer_sha256": _sha(underlying),
        },))
        self.assertEqual(rows[0]["provider_attempt_count"], 0)
        self.assertEqual(rows[0]["underlying_answer_sha256"], _sha(underlying))
        self.assertEqual(rows[0]["output"], underlying)
        self.assertEqual(rows[0]["gate_action"], "allowed")
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_derived_b7(Path(directory) / "b7", rows)
            self.assertEqual(manifest["provider_request_count"], 0)

    def test_extraction_records_adapt_to_materializer_contract(self) -> None:
        rows = extraction_rows_from_records(({
            "position": 1,
            "task": "extraction",
            "baseline_id": None,
            "status": "succeeded",
            "record_id": "source_1",
            "user_id": "user_001",
            "request_sha256": "a" * 64,
            "output": {"claims": [{"claim_id": "claim_1"}]},
        },))
        self.assertEqual(rows[0]["claims"], [{"claim_id": "claim_1"}])
        self.assertTrue(rows[0]["valid"])


if __name__ == "__main__":
    unittest.main()
