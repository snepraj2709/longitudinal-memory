from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from evaluation.qwen_preflight import VLLMTokenizer


class _Response:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class QwenPreflightTests(unittest.TestCase):
    def test_tokenizer_sends_authentication_and_non_thinking_messages(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
            requests.append((request, timeout))
            return _Response({"count": 41})

        tokenizer = VLLMTokenizer(
            base_url="http://127.0.0.1:6006", model="qwen35-27b-fp8-v1",
            api_key="private-key", timeout_seconds=9,
        )
        with patch("evaluation.qwen_preflight.urlopen", side_effect=fake_urlopen):
            result = tokenizer.count([{"role": "user", "content": "hello"}])

        request, timeout = requests[0]
        body = json.loads(request.data)
        self.assertEqual(result.count, 41)
        self.assertEqual(timeout, 9)
        self.assertEqual(request.full_url, "http://127.0.0.1:6006/tokenize")
        self.assertEqual(request.headers["Authorization"], "Bearer private-key")
        self.assertEqual(body["model"], "qwen35-27b-fp8-v1")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(body["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
