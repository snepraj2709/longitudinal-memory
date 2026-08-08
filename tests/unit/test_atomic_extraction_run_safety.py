from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIRateLimitMetadata, OpenAIResponseMetadata
from extraction.run_atomic import (
    ATOMIC_CASE_REFS,
    AtomicPipelineError,
    dry_run_atomic,
    execute_atomic_pipeline,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-4.1-2025-04-14"


class SafetyFakeClient:
    def __init__(
        self,
        *,
        fail_positions: set[int] | None = None,
        invalid_positions: set[int] | None = None,
        interrupt_positions: set[int] | None = None,
        returned_model: str = MODEL,
    ) -> None:
        self.model = MODEL
        self.fail_positions = fail_positions or set()
        self.invalid_positions = invalid_positions or set()
        self.interrupt_positions = interrupt_positions or set()
        self.returned_model = returned_model
        self.calls: list[str] = []
        self.restored: list[OpenAIResponseMetadata] = []

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        source_id = json.loads(user_prompt.split("\n", 1)[1])["source_id"]
        self.calls.append(source_id)
        position = len(self.calls)
        if position in self.interrupt_positions:
            raise KeyboardInterrupt
        if position in self.fail_positions:
            raise RuntimeError("sk-proj-" + "x" * 48)
        response = "sensitive-invalid-provider-output" if position in self.invalid_positions else '{"claims":[]}'
        return response, OpenAIResponseMetadata(
            response_id=f"resp_safety_{source_id}",
            returned_model=self.returned_model,
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            request_id=f"req_safety_{source_id}",
            rate_limits=OpenAIRateLimitMetadata(
                remaining_requests=99,
                remaining_tokens=900_000,
                reset_requests_seconds=1.0,
                reset_tokens_seconds=2.0,
            ),
            pacing_delay_seconds=0.25,
        )

    def restore_pacing_metadata(self, metadata: OpenAIResponseMetadata) -> None:
        self.restored.append(metadata)


class AtomicExtractionRunSafetyTests(unittest.TestCase):
    def execute(
        self,
        output_dir: Path,
        client: SafetyFakeClient,
        *,
        resume: bool = False,
        max_cost_usd: str | None = None,
    ) -> dict[str, object]:
        times = iter(
            [
                datetime.fromisoformat("2026-08-08T10:00:00+00:00"),
                datetime.fromisoformat("2026-08-08T10:01:00+00:00"),
            ]
        )
        return execute_atomic_pipeline(
            client=client,
            requested_model=MODEL,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            now=lambda: next(times),
            resume=resume,
            max_cost_usd=max_cost_usd,
        )

    def test_dry_run_is_source_only_and_does_not_parse_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "unused"
            with patch("extraction.run_atomic.load_atomic_gold") as gold_loader:
                from io import StringIO

                dry_run = dry_run_atomic(repo_root=REPO_ROOT, stdout=StringIO())
            gold_loader.assert_not_called()
            self.assertFalse(output_dir.exists())
        self.assertEqual(dry_run.record["provider_calls"], 0)
        self.assertEqual(dry_run.record["output_writes"], 0)
        self.assertTrue(dry_run.record["source_only_inputs"])
        self.assertFalse(dry_run.record["oracle_or_gold_fields_in_prompts"])
        self.assertEqual(dry_run.record["estimated_input_tokens_upper_bound"], 158_584)
        self.assertEqual(dry_run.record["expected_output_tokens"], 12_000)
        self.assertEqual(dry_run.record["expected_cost_usd"], "0.413168")
        self.assertEqual(dry_run.record["maximum_cost_usd"], "1.274336")

    def test_prompt_leakage_guard_rejects_scorer_only_fields(self) -> None:
        from extraction.run_atomic import _assert_source_only_prompts

        for field in ("expected_claims", "gold_answer", "oracle_truth", "scorer_only"):
            with self.subTest(field=field), self.assertRaisesRegex(
                AtomicPipelineError, "scorer-only"
            ):
                _assert_source_only_prompts((f"source data plus {field}",))

    def test_frozen_config_rejects_prompt_and_model_drift(self) -> None:
        original = json.loads(
            (REPO_ROOT / "configs/extraction/atomic_extraction_run_v2.json").read_text()
        )
        for field, value in (("prompt_sha256", "0" * 64), ("requested_model", "gpt-4.1")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config = dict(original)
                config[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(AtomicPipelineError):
                    dry_run_atomic(repo_root=REPO_ROOT, config_path=path)

    def test_execute_preflight_rejects_drift_before_reading_provider_access(self) -> None:
        with (
            patch("extraction.run_atomic.load_env_value") as access_loader,
            patch("extraction.run_atomic.OpenAIResponsesClient") as client_type,
            patch("sys.stderr", new=io.StringIO()),
        ):
            status = main(["--execute", "--model", "gpt-4.1"])
        self.assertEqual(status, 1)
        access_loader.assert_not_called()
        client_type.assert_not_called()

    def test_cost_cap_stops_before_the_first_provider_call(self) -> None:
        client = SafetyFakeClient()
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(Path(directory) / "run", client, max_cost_usd="0.000001")
        self.assertEqual(client.calls, [])
        self.assertEqual(run["run_status"], "stopped_safety_cap")
        self.assertEqual(run["provider_requests_attempted"], 0)
        self.assertEqual(run["attempts"][0]["failure_stage"], "safety_cap")

    def test_request_retry_and_token_ceilings_stop_before_a_call(self) -> None:
        from io import StringIO
        from extraction.run_atomic import AtomicCaseExecution, _request_limit_failure

        plan = dry_run_atomic(repo_root=REPO_ROOT, stdout=StringIO()).plan

        def execution(input_tokens: int, output_tokens: int) -> AtomicCaseExecution:
            return AtomicCaseExecution(
                position=1,
                case_id=ATOMIC_CASE_REFS[0][0],
                source_id=ATOMIC_CASE_REFS[0][1],
                attempted=True,
                passed=False,
                claims=(),
                provider_metadata=None,
                failure_stage="provider",
                error="The provider request failed.",
                budget_input_tokens=input_tokens,
                budget_output_tokens=output_tokens,
            )

        request_limit = _request_limit_failure(
            plan,
            [execution(0, 0)] * plan.config.maximum_request_attempts,
            (),
            2,
            1,
            plan.config.hard_cost_cap_usd,
        )
        input_limit = _request_limit_failure(
            plan,
            [execution(plan.config.maximum_input_tokens, 0)],
            (),
            2,
            1,
            plan.config.hard_cost_cap_usd,
        )
        output_limit = _request_limit_failure(
            plan,
            [execution(0, plan.config.maximum_output_tokens)],
            (),
            2,
            1,
            plan.config.hard_cost_cap_usd,
        )
        prior = [
            {
                "position": 1,
                "budget_input_tokens": 0,
                "budget_output_tokens": 0,
            }
            for _ in range(plan.config.maximum_retry_requests + 1)
        ]
        retry_limit = _request_limit_failure(
            plan,
            (),
            prior,
            1,
            1,
            plan.config.hard_cost_cap_usd,
        )

        self.assertIn("request-count", request_limit)
        self.assertIn("input-token", input_limit)
        self.assertIn("output-token", output_limit)
        self.assertIn("retry allowance", retry_limit)

    def test_retry_allowance_stops_before_an_extra_provider_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.execute(output_dir, SafetyFakeClient(fail_positions={1}))
            for _ in range(10):
                run = self.execute(
                    output_dir,
                    SafetyFakeClient(fail_positions={1}),
                    resume=True,
                )
            self.assertEqual(run["provider_requests_attempted"], 11)

            blocked = SafetyFakeClient()
            stopped = self.execute(output_dir, blocked, resume=True)

        self.assertEqual(blocked.calls, [])
        self.assertEqual(stopped["run_status"], "stopped_safety_cap")
        self.assertIn("retry allowance", stopped["attempts"][0]["error"])

    def test_each_response_is_checkpointed_with_full_provider_metadata(self) -> None:
        client = SafetyFakeClient()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            import extraction.run_atomic as runner

            writer = runner._write_checkpoint
            with patch("extraction.run_atomic._write_checkpoint", wraps=writer) as checkpoint:
                run = self.execute(output_dir, client)
            self.assertGreaterEqual(checkpoint.call_count, 12)
            persisted = json.loads((output_dir / "run.json").read_text())
            for filename, digest in run["output_file_sha256"].items():
                import hashlib

                self.assertEqual(
                    hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
                    digest,
                )
        metadata = persisted["attempts"][0]["provider_metadata"]
        self.assertEqual(metadata["response_id"], "resp_safety_cal_001")
        self.assertEqual(metadata["requested_model"], MODEL)
        self.assertEqual(metadata["returned_model"], MODEL)
        self.assertEqual(metadata["request_id"], "req_safety_cal_001")
        self.assertEqual(metadata["input_tokens"], 100)
        self.assertEqual(metadata["output_tokens"], 10)
        self.assertEqual(metadata["total_tokens"], 110)
        self.assertEqual(metadata["rate_limits"]["remaining_requests"], 99)
        self.assertEqual(metadata["pacing_delay_seconds"], 0.25)
        self.assertEqual(run["run_status"], "completed")

    def test_interrupted_run_resumes_only_missing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            interrupted = SafetyFakeClient(interrupt_positions={5})
            with self.assertRaises(KeyboardInterrupt):
                self.execute(output_dir, interrupted)
            checkpoint = json.loads((output_dir / "run.json").read_text())
            self.assertEqual(checkpoint["run_status"], "running")
            self.assertEqual(checkpoint["successful_cases"], 4)
            self.assertEqual(checkpoint["provider_requests_attempted"], 5)
            self.assertEqual(
                checkpoint["attempts"][-1]["failure_stage"], "provider_pending"
            )

            resumed = SafetyFakeClient()
            completed = self.execute(output_dir, resumed, resume=True)

        self.assertEqual(
            resumed.calls,
            [source_id for _, source_id in ATOMIC_CASE_REFS[4:]],
        )
        self.assertEqual(completed["successful_cases"], 10)
        self.assertEqual(completed["provider_requests_attempted"], 11)

    def test_model_snapshot_mismatch_is_checkpointed_and_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            client = SafetyFakeClient(returned_model="gpt-4.1")
            run = self.execute(output_dir, client)
            self.assertEqual(run["run_status"], "failed_model_mismatch")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(
                run["attempts"][0]["failure_stage"], "provider_model_mismatch"
            )
            self.assertEqual(
                run["attempts"][0]["provider_metadata"]["returned_model"],
                "gpt-4.1",
            )
            with self.assertRaisesRegex(AtomicPipelineError, "provider-failed"):
                self.execute(output_dir, SafetyFakeClient(), resume=True)

    def test_resume_retries_provider_failure_without_replaying_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            first = SafetyFakeClient(fail_positions={4})
            failed = self.execute(output_dir, first)
            first_predictions = (output_dir / "predictions.jsonl").read_bytes()

            resumed = SafetyFakeClient()
            completed = self.execute(output_dir, resumed, resume=True)

            self.assertEqual(failed["successful_cases"], 3)
            self.assertEqual(resumed.calls, [source_id for _, source_id in ATOMIC_CASE_REFS[3:]])
            self.assertEqual(resumed.restored[0].response_id, "resp_safety_conv_003")
            self.assertEqual(completed["run_status"], "completed")
            self.assertEqual(completed["resume_count"], 1)
            self.assertEqual(completed["provider_requests_attempted"], 11)
            self.assertTrue((output_dir / "predictions.jsonl").read_bytes().startswith(first_predictions))
            self.assertEqual(
                [item["case_id"] for item in completed["attempts"]],
                [case_id for case_id, _ in ATOMIC_CASE_REFS],
            )

    def test_resume_never_retries_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.execute(output_dir, SafetyFakeClient(invalid_positions={4}))
            resumed = SafetyFakeClient()
            with self.assertRaisesRegex(AtomicPipelineError, "provider-failed"):
                self.execute(output_dir, resumed, resume=True)
        self.assertEqual(resumed.calls, [])

    def test_resume_rejects_completed_reordered_and_incompatible_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed_dir = Path(directory) / "complete"
            self.execute(completed_dir, SafetyFakeClient())
            with self.assertRaises(AtomicPipelineError):
                self.execute(completed_dir, SafetyFakeClient(), resume=True)

            failed_dir = Path(directory) / "failed"
            self.execute(failed_dir, SafetyFakeClient(fail_positions={4}))
            run_path = failed_dir / "run.json"
            base = json.loads(run_path.read_text())
            for mutation in ("reordered", "incompatible"):
                with self.subTest(mutation=mutation):
                    changed = json.loads(json.dumps(base))
                    if mutation == "reordered":
                        changed["attempts"][0], changed["attempts"][1] = (
                            changed["attempts"][1], changed["attempts"][0]
                        )
                    else:
                        changed["configuration_sha256"] = "0" * 64
                    run_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(AtomicPipelineError):
                        self.execute(failed_dir, SafetyFakeClient(), resume=True)
                    run_path.write_text(json.dumps(base), encoding="utf-8")

    def test_resume_rejects_predicate_registry_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "failed"
            self.execute(output_dir, SafetyFakeClient(fail_positions={4}))
            run_path = output_dir / "run.json"
            checkpoint = json.loads(run_path.read_text())
            checkpoint["predicate_registry_sha256"] = "0" * 64
            run_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaises(AtomicPipelineError):
                self.execute(output_dir, SafetyFakeClient(), resume=True)

    def test_resume_rejects_stale_scores_hash_mismatch_and_bad_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for mutation in ("stale_scores", "hash_mismatch", "bad_counters"):
                with self.subTest(mutation=mutation):
                    output_dir = Path(directory) / mutation
                    self.execute(output_dir, SafetyFakeClient(fail_positions={4}))
                    if mutation == "stale_scores":
                        (output_dir / "scores.json").write_text("{}\n", encoding="utf-8")
                    elif mutation == "hash_mismatch":
                        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write("{}\n")
                    else:
                        run_path = output_dir / "run.json"
                        record = json.loads(run_path.read_text())
                        record["successful_cases"] = 99
                        run_path.write_text(json.dumps(record), encoding="utf-8")
                    with self.assertRaises(AtomicPipelineError):
                        self.execute(output_dir, SafetyFakeClient(), resume=True)

    def test_gold_hash_is_rechecked_after_scoring_before_scores_are_written(self) -> None:
        import extraction.run_atomic as runner

        real_hash = runner._file_sha256
        gold_path = (REPO_ROOT / "data/phase3/atomic_extraction_gold.jsonl").resolve()
        gold_hash_calls = 0

        def drifting_hash(path: Path) -> str:
            nonlocal gold_hash_calls
            if Path(path).resolve() == gold_path:
                gold_hash_calls += 1
                if gold_hash_calls >= 3:
                    return "0" * 64
            return real_hash(path)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with patch("extraction.run_atomic._file_sha256", side_effect=drifting_hash):
                with self.assertRaisesRegex(AtomicPipelineError, "during scoring"):
                    self.execute(output_dir, SafetyFakeClient())
            self.assertFalse((output_dir / "scores.json").exists())
            self.assertFalse((output_dir / "case_scores.jsonl").exists())

    def test_later_provider_failure_keeps_completed_predictions_and_sanitizes_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            run = self.execute(output_dir, SafetyFakeClient(fail_positions={8}))
            predictions = (output_dir / "predictions.jsonl").read_text().splitlines()
            artifact_text = "\n".join(
                path.read_text(encoding="utf-8") for path in output_dir.iterdir()
            )
        self.assertEqual(run["successful_cases"], 7)
        self.assertEqual(len(predictions), 7)
        self.assertNotIn("sk-proj-", artifact_text)
        self.assertNotIn("sensitive", artifact_text)


if __name__ == "__main__":
    unittest.main()
