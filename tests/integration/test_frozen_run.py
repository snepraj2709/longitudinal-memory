from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.frozen_run import (
    MODEL,
    FrozenRunError,
    run_extraction_batch,
    verify_extraction_batch,
)
from evaluation.openai_client import OpenAIResponseMetadata


ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def complete_with_metadata(self, *, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("synthetic provider failure")
        metadata = OpenAIResponseMetadata(
            response_id=f"resp_{self.calls}", returned_model=MODEL,
            input_tokens=100 + self.calls, output_tokens=5,
            total_tokens=105 + self.calls, request_id=f"req_{self.calls}",
            pacing_delay_seconds=0.0,
        )
        return '{"claims":[]}', metadata


class FrozenRunIntegrationTests(unittest.TestCase):
    def test_fake_100_source_run_checkpoints_and_deep_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            client = FakeClient()
            result = run_extraction_batch(
                repo_root=ROOT, output_dir=output, client=client,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual((result["successful_count"], client.calls), (100, 100))
            self.assertEqual(result["failure_count"], 0)
            self.assertEqual((output / "failures.jsonl").read_bytes(), b"")
            verified = verify_extraction_batch(repo_root=ROOT, output_dir=output)
            self.assertEqual(verified["provider_request_count"], 100)
            with self.assertRaises(FrozenRunError):
                run_extraction_batch(repo_root=ROOT, output_dir=output, client=client)

    def test_provider_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            client = FakeClient(fail_at=2)
            result = run_extraction_batch(
                repo_root=ROOT, output_dir=output, client=client,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            self.assertEqual(result["status"], "failed_provider")
            self.assertEqual((result["successful_count"], result["failure_count"]), (1, 1))
            self.assertEqual(client.calls, 2)
            failure = json.loads((output / "failures.jsonl").read_text())
            self.assertEqual(failure["stage"], "provider")
            with self.assertRaises(FrozenRunError):
                run_extraction_batch(
                    repo_root=ROOT, output_dir=output, client=client, resume=True,
                )

    def test_runtime_read_closure_excludes_scorer_only_inputs(self):
        opened: list[str] = []
        original = Path.read_bytes

        def guarded(path: Path):
            value = path.resolve().as_posix()
            opened.append(value)
            if any(token in value for token in ("/gold/", "/oracle/", "/review_queues/")):
                raise AssertionError(f"prohibited scorer path opened: {value}")
            return original(path)

        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "read_bytes", guarded):
            run_extraction_batch(
                repo_root=ROOT, output_dir=Path(directory) / "batch", client=FakeClient(),
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
        self.assertTrue(any("runtime/sources.jsonl" in path for path in opened))

    def test_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            run_extraction_batch(
                repo_root=ROOT, output_dir=output, client=FakeClient(),
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            prediction = output / "predictions.jsonl"
            prediction.write_bytes(prediction.read_bytes() + b" ")
            with self.assertRaises(FrozenRunError):
                verify_extraction_batch(repo_root=ROOT, output_dir=output)


if __name__ == "__main__":
    unittest.main()
