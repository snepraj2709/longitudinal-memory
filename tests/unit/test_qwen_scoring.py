from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_execution import _sha
from evaluation.qwen_materialization import ContextPackage
from evaluation.qwen_scoring import (
    QwenScoringError,
    _validate_judge_output,
    build_judge_jobs,
    score_records,
)
from evaluation.qwen_benchmark import select_runtime


def _answer(status="answered"):
    if status == "abstained":
        return {"status": status, "answer": "Not known.", "confidence": 0, "statements": [], "citations": [], "unresolved_parts": [], "abstention_reason": "missing"}
    return {
        "status": status, "answer": "Mumbai", "confidence": 0.9,
        "statements": ["Mumbai"],
        "citations": [{"source_id": "source_1", "message_id": "message_1", "quote": "Mumbai"}],
        "unresolved_parts": [], "abstention_reason": None,
    }


class QwenScoringTests(unittest.TestCase):
    def test_development_judge_pack_has_112_blinded_batches(self) -> None:
        runtime = select_runtime(Path(__file__).resolve().parents[2], "development")
        task_rows = {task: runtime[task] for task in ("qa", "summary", "interactive")}
        predictions = []
        for baseline_number in range(8):
            baseline = f"B{baseline_number}"
            for task, cases in task_rows.items():
                for case in cases:
                    output = {
                        "status": "abstained",
                        {"qa": "answer", "summary": "summary", "interactive": "response"}[task]: "Not enough information.",
                        "confidence": 0,
                        "statements": [], "citations": [], "unresolved_parts": [],
                        "abstention_reason": "missing",
                    }
                    predictions.append({
                        "baseline_id": baseline, "task": task,
                        "record_id": case["case_id"], "user_id": case["user_id"],
                        "status": "succeeded", "output": output,
                        "context_sha256": "a" * 64,
                        "underlying_answer_sha256": _sha(output) if baseline in {"B6", "B7"} else None,
                    })
        predictions.sort(key=lambda row: (int(row["baseline_id"][1:]), ("qa", "summary", "interactive").index(row["task"]), row["record_id"]))
        payload = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in predictions
        )
        references = {"claims": []}
        for task, cases in task_rows.items():
            references[task] = []
            for case in cases:
                row = {**case, "should_abstain": False}
                if task == "qa":
                    row.update(reference_answer="Reference", acceptable_answers=[])
                elif task == "summary":
                    row.update(reference_summary="Reference", gold_event_ids=[])
                else:
                    row.update(expected_behaviours=[])
                references[task].append(row)
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "predictions"
            release.mkdir()
            (release / "predictions.jsonl").write_bytes(payload)
            (release / "failures.jsonl").write_bytes(b"")
            (release / "manifest.json").write_text(json.dumps({
                "schema_version": "qwen_logical_prediction_seal_v2",
                "series_id": "qwen35-27b-fp8-v2", "split": "development",
                "status": "sealed", "failure_count": 0,
                "predictions_sha256": sha256(payload).hexdigest(),
                "failures_sha256": sha256(b"").hexdigest(),
            }))
            jobs = build_judge_jobs(release, references)
        self.assertEqual(len(jobs), 112)
        self.assertTrue(all(job.context_count <= 10 for job in jobs))
        self.assertTrue(all('"baseline_id"' not in job.user_prompt for job in jobs))
        self.assertTrue(all("qwen35" not in job.user_prompt.casefold() for job in jobs))

    def test_deterministic_scorecard_separates_failures_and_has_no_composite(self) -> None:
        output = _answer()
        predictions = []
        contexts = []
        for index in range(8):
            baseline = f"B{index}"
            underlying = _sha(output) if baseline in {"B6", "B7"} else None
            predictions.append({
                "baseline_id": baseline, "task": "qa", "record_id": "case_1",
                "user_id": "user_001", "status": "succeeded", "output": output,
                "context_sha256": "a" * 64, "underlying_answer_sha256": underlying,
            })
            contexts.append(ContextPackage(
                "qwen35-27b-fp8-v2", "development", "qa", "case_1", "user_001",
                "2026-12-01T00:00:00+00:00", baseline, baseline != "B7", "test", ({
                    "lifecycle_statuses": ["current"], "claim_ids": ["claim_1"],
                    "evidence": [{"source_id": "source_1", "message_id": "message_1", "quote": "Mumbai"}],
                },), "a" * 64,
            ))
        references = {
            "claims": [{
                "claim_id": "claim_1", "user_id": "user_001", "subject_id": "user_001",
                "speaker_id": "user_001", "predicate": "lives_in", "object": "Mumbai",
                "polarity": "positive", "epistemic_status": "asserted", "valid_from": None,
                "valid_to": None, "time_precision": "unknown", "status": "current",
                "evidence": [{"source_id": "source_1", "message_id": "message_1", "quote": "Mumbai"}],
            }],
            "qa": [{
                "case_id": "case_1", "reference_answer": "Mumbai", "acceptable_answers": [],
                "should_abstain": False, "required_claim_ids": ["claim_1"],
                "evidence": [{"source_id": "source_1", "message_id": "message_1", "quote": "Mumbai"}],
            }],
            "summary": [], "interactive": [],
        }
        rows = score_records(
            split="development", predictions=predictions, contexts=contexts,
            extraction_records=[], references=references,
            execution_metadata={"gpu_cost_inr": 10.0},
        )
        qa_b7 = [row for row in rows if row.baseline_id == "B7" and row.task == "qa"]
        self.assertTrue(any(row.metric == "strict_correctness" and row.value == 1 for row in qa_b7))
        self.assertTrue(any(row.metric == "gpu_cost_inr" and row.value == 10 for row in rows))
        self.assertFalse(any(row.metric == "composite" for row in rows))

    def test_judge_validation_requires_exact_candidate_coverage(self) -> None:
        valid = {"judgments": [{"candidate_id": "a", "label": "correct", "reason_codes": []}]}
        self.assertEqual(_validate_judge_output(valid, ("a",)), valid)
        with self.assertRaisesRegex(QwenScoringError, "coverage"):
            _validate_judge_output(valid, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
