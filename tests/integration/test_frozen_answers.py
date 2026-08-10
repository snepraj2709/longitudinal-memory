from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_answers import MODEL, run_answer_batch, verify_answer_batch
from evaluation.openai_client import OpenAIResponseError, OpenAIResponseMetadata


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


class InvalidOutputFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        parent = self

        class Client:
            def complete_with_metadata(self, *, system_prompt: str, user_prompt: str):
                parent.calls += 1
                raw = json.dumps({
                    "status": "abstained", "response": "Not enough reliable memory.",
                    "confidence": 0.5, "statements": [], "citations": [],
                    "unresolved_parts": [], "abstention_reason": "insufficient_evidence",
                })
                usage = OpenAIResponseMetadata(
                    response_id=f"resp_invalid_{parent.calls}", returned_model=MODEL,
                    input_tokens=100, output_tokens=10, total_tokens=110,
                    request_id=f"req_invalid_{parent.calls}", pacing_delay_seconds=0.0,
                )
                return raw, usage

        return Client()


class RateLimitFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        parent = self

        class Client:
            def complete_with_metadata(self, *, system_prompt: str, user_prompt: str):
                parent.calls += 1
                raise OpenAIResponseError("OpenAI API returned HTTP 429: limited")

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

    def test_validation_failures_retain_billed_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            factory = InvalidOutputFactory()
            result = run_answer_batch(
                repo_root=ROOT, baseline_id="B0", task="interactive",
                output_dir=output, client_factory=factory, require_prior=False,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            self.assertEqual((result["status"], factory.calls), ("failed", 4))
            self.assertEqual((result["failure_count"], result["provider_request_count"]), (4, 4))
            self.assertEqual((result["input_token_count"], result["output_token_count"]), (400, 40))
            self.assertEqual(result["incremental_cost_usd"], "0.0011200")

    def test_resume_skips_failed_positions_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            failed_factory = RateLimitFactory()
            first = run_answer_batch(
                repo_root=ROOT, baseline_id="B0", task="interactive",
                output_dir=output, client_factory=failed_factory, require_prior=False,
                clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
            )
            self.assertEqual((first["failure_count"], failed_factory.calls), (4, 4))

            resumed_factory = FakeFactory()
            resumed = run_answer_batch(
                repo_root=ROOT, baseline_id="B0", task="interactive",
                output_dir=output, client_factory=resumed_factory, require_prior=False,
                resume=True, clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
                monotonic=lambda: 1.0,
            )
            self.assertEqual(resumed_factory.calls, 16)
            self.assertEqual((resumed["successful_count"], resumed["failure_count"]), (16, 4))
            self.assertEqual((resumed["provider_request_count"], resumed["remaining_count"]), (20, 0))
            self.assertEqual(resumed["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
