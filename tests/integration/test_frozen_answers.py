from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_answers import MODEL, run_answer_batch, verify_answer_batch
from evaluation.openai_client import OpenAIResponseMetadata


ROOT = Path(__file__).resolve().parents[2]


class FakeFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        parent = self

        class Client:
            def complete_with_metadata(self, *, system_prompt: str, user_prompt: str):
                parent.calls += 1
                task = "interactive" if "initial_user_message" in user_prompt else (
                    "summary" if "instruction" in user_prompt else "qa"
                )
                body = {"interactive": "response", "summary": "summary", "qa": "answer"}[task]
                raw = json.dumps({
                    "status": "abstained", body: "Not enough reliable memory.",
                    "confidence": 0, "statements": [], "citations": [],
                    "unresolved_parts": [], "abstention_reason": "insufficient_evidence",
                })
                usage = OpenAIResponseMetadata(
                    response_id=f"resp_{parent.calls}", returned_model=MODEL,
                    input_tokens=200, output_tokens=20, total_tokens=220,
                    request_id=f"req_{parent.calls}", pacing_delay_seconds=0.0,
                )
                return raw, usage

        return Client()


class FrozenAnswerIntegrationTests(unittest.TestCase):
    def test_provider_batch_is_checkpointed_and_deep_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            factory = FakeFactory()
            result = run_answer_batch(
                repo_root=ROOT, baseline_id="B0", task="interactive",
                output_dir=output, client_factory=factory, require_prior=False,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            self.assertEqual((result["status"], factory.calls), ("completed", 20))
            self.assertEqual(result["failure_count"], 0)
            verified = verify_answer_batch(
                repo_root=ROOT, baseline_id="B0", task="interactive", output_dir=output,
            )
            self.assertEqual(verified["provider_request_count"], 20)

    def test_b7_gate_makes_no_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            factory = FakeFactory()
            result = run_answer_batch(
                repo_root=ROOT, baseline_id="B7", task="interactive",
                output_dir=output, client_factory=factory, require_prior=False,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
            )
            self.assertEqual((result["status"], factory.calls), ("completed", 0))
            self.assertEqual(result["deterministic_gate_count"], 20)
            self.assertEqual(result["provider_request_count"], 0)
            verify_answer_batch(
                repo_root=ROOT, baseline_id="B7", task="interactive", output_dir=output,
            )


if __name__ == "__main__":
    unittest.main()
