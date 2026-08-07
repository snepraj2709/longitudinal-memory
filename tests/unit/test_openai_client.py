from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.openai_client import (
    OpenAIResponseError,
    OpenAIResponsesClient,
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
