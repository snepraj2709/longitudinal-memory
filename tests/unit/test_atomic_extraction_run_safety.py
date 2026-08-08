from __future__ import annotations

from datetime import datetime
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
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-4.1-2025-04-14"


class SafetyFakeClient:
    def __init__(
        self,
        *,
        fail_positions: set[int] | None = None,
        invalid_positions: set[int] | None = None,
    ) -> None:
        self.model = MODEL
        self.fail_positions = fail_positions or set()
        self.invalid_positions = invalid_positions or set()
        self.calls: list[str] = []
        self.restored: list[OpenAIResponseMetadata] = []

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        source_id = json.loads(user_prompt.split("\n", 1)[1])["source_id"]
        self.calls.append(source_id)
        position = len(self.calls)
        if position in self.fail_positions:
            raise RuntimeError("sk-proj-" + "x" * 48)
        response = "sensitive-invalid-provider-output" if position in self.invalid_positions else '{"claims":[]}'
        return response, OpenAIResponseMetadata(
            response_id=f"resp_safety_{source_id}",
            returned_model=MODEL,
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

    def test_frozen_config_rejects_prompt_and_model_drift(self) -> None:
        original = json.loads(
            (REPO_ROOT / "configs/extraction/atomic_extraction_run_v1.json").read_text()
        )
        for field, value in (("prompt_sha256", "0" * 64), ("requested_model", "gpt-4.1")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config = dict(original)
                config[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(AtomicPipelineError):
                    dry_run_atomic(repo_root=REPO_ROOT, config_path=path)

    def test_cost_cap_stops_before_the_first_provider_call(self) -> None:
        client = SafetyFakeClient()
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(Path(directory) / "run", client, max_cost_usd="0.000001")
        self.assertEqual(client.calls, [])
        self.assertEqual(run["run_status"], "stopped_safety_cap")
        self.assertEqual(run["provider_requests_attempted"], 0)
        self.assertEqual(run["attempts"][0]["failure_stage"], "safety_cap")

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
            with self.assertRaisesRegex(AtomicPipelineError, "provider-failed"):
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
