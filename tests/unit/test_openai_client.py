from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.openai_client import (
    OpenAIRateLimitMetadata,
    OpenAIResponseError,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    _parse_reset_duration,
    load_env_value,
)


class OpenAIResponsesClientTests(unittest.TestCase):
    def completed_response(self, text: str = '{"status":"answered"}') -> dict[str, object]:
        return {
            "id": "resp_test",
            "status": "completed",
            "model": "gpt-4.1-2025-04-14",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        }

    def test_sends_approved_settings_and_returns_raw_output(self) -> None:
        payloads: list[dict[str, object]] = []

        def transport(payload: object) -> dict[str, object]:
            payloads.append(dict(payload))
            return self.completed_response()

        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=transport,
        )

        raw = client.complete(system_prompt="system", user_prompt="user")

        self.assertEqual(raw, '{"status":"answered"}')
        self.assertEqual(
            payloads,
            [
                {
                    "model": "gpt-4.1-2025-04-14",
                    "instructions": "system",
                    "input": "user",
                    "temperature": 0,
                    "max_output_tokens": 1000,
                    "store": False,
                    "text": {"format": {"type": "json_object"}},
                }
            ],
        )
        self.assertEqual(client.response_metadata[0].response_id, "resp_test")
        self.assertEqual(client.response_metadata[0].total_tokens, 120)

    def test_rejects_a_different_returned_model(self) -> None:
        response = self.completed_response()
        response["model"] = "gpt-4.1"
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=lambda payload: response,
        )

        with self.assertRaisesRegex(OpenAIResponseError, "expected"):
            client.complete(system_prompt="system", user_prompt="user")

    def test_rejects_missing_output_text(self) -> None:
        response = self.completed_response()
        response["output"] = []
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=lambda payload: response,
        )

        with self.assertRaisesRegex(OpenAIResponseError, "exactly one output_text"):
            client.complete(system_prompt="system", user_prompt="user")

    def test_paces_next_request_from_rate_limit_metadata(self) -> None:
        sleeps: list[float] = []
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=lambda payload: self.completed_response(),
            sleep=sleeps.append,
            minimum_request_interval_seconds=0,
        )
        client.restore_pacing_metadata(
            OpenAIResponseMetadata(
                response_id="resp_previous",
                returned_model="gpt-4.1-2025-04-14",
                input_tokens=7000,
                output_tokens=100,
                total_tokens=7100,
                rate_limits=OpenAIRateLimitMetadata(
                    remaining_requests=10,
                    remaining_tokens=5000,
                    reset_tokens_seconds=6.0,
                    remaining_project_tokens=4000,
                    reset_project_tokens_seconds=3.0,
                ),
            )
        )

        _, metadata = client.complete_with_metadata(
            system_prompt="system", user_prompt="user"
        )

        self.assertEqual(sleeps, [6.25])
        self.assertEqual(metadata.pacing_delay_seconds, 6.25)

    def test_does_not_sleep_when_reported_capacity_is_sufficient(self) -> None:
        sleeps: list[float] = []
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=lambda payload: self.completed_response(),
            sleep=sleeps.append,
            minimum_request_interval_seconds=0,
        )
        client.restore_pacing_metadata(
            OpenAIResponseMetadata(
                response_id="resp_previous",
                returned_model="gpt-4.1-2025-04-14",
                input_tokens=7000,
                output_tokens=100,
                total_tokens=7100,
                rate_limits=OpenAIRateLimitMetadata(
                    remaining_requests=10,
                    remaining_tokens=10000,
                    reset_tokens_seconds=6.0,
                ),
            )
        )

        client.complete(system_prompt="system", user_prompt="user")

        self.assertEqual(sleeps, [])

    def test_minimum_interval_applies_after_a_provider_error(self) -> None:
        sleeps: list[float] = []
        calls = 0

        def transport(payload: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OpenAIResponseError("rate limited")
            return self.completed_response()

        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-4.1-2025-04-14",
            temperature=0,
            max_output_tokens=1000,
            transport=transport,
            sleep=sleeps.append,
            minimum_request_interval_seconds=15,
        )

        with self.assertRaisesRegex(OpenAIResponseError, "rate limited"):
            client.complete(system_prompt="system", user_prompt="first")
        client.complete(system_prompt="system", user_prompt="second")

        self.assertEqual(sleeps, [15])

    def test_parses_provider_reset_duration_formats(self) -> None:
        self.assertEqual(_parse_reset_duration("250ms"), 0.25)
        self.assertEqual(_parse_reset_duration("6m0s"), 360.0)
        self.assertEqual(_parse_reset_duration("1h2m3.5s"), 3723.5)
        self.assertIsNone(_parse_reset_duration("later"))

    def test_env_loader_reads_only_requested_value_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "OTHER=value\nOPENAI_API_KEY='secret-value'\n", encoding="utf-8"
            )
            self.assertEqual(load_env_value(path, "OPENAI_API_KEY"), "secret-value")

            path.write_text(
                "OPENAI_API_KEY=first\nOPENAI_API_KEY=second\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(OpenAIResponseError, "more than once"):
                load_env_value(path, "OPENAI_API_KEY")


if __name__ == "__main__":
    unittest.main()
