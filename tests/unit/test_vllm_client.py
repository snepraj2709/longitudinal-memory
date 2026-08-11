from __future__ import annotations

import http.server
import json
import socket
import threading
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIResponseMetadata
from evaluation.vllm_client import (
    VLLMClient,
    VLLMModelMismatchError,
    VLLMResponseError,
)

MODEL = "qwen35-27b-fp8-v1"


def completion_response(
    text: str = '{"status":"answered"}',
    model: str = MODEL,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> dict[str, object]:
    return {
        "id": "chatcmpl_fake_vllm",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeVLLM/1.0"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.server.lock:
            self.server.records.append(
                {"path": self.path, "headers": dict(self.headers), "body": json.loads(body)}
            )
        encoded = json.dumps(self.server.payload).encode("utf-8")
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-request-id", "req_fake_vllm")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class _FakeVLLMServer(http.server.ThreadingHTTPServer):
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        # HTTPServer.__init__ calls socket.getfqdn() for a reverse DNS lookup,
        # which can stall for tens of seconds on some machines. server_name is
        # unused here, so resolve it locally and instantly.
        with patch("socket.getfqdn", return_value="localhost"):
            super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.status = status
        self.payload = payload
        self.records: list[dict[str, object]] = []
        self.lock = threading.Lock()
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def start(self) -> "_FakeVLLMServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self._thread.join(timeout=5)
        self.server_close()


class VLLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.servers: list[_FakeVLLMServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()

    def fake_server(
        self, status: int = 200, payload: dict[str, object] | None = None
    ) -> _FakeVLLMServer:
        server = _FakeVLLMServer(status, payload if payload is not None else completion_response())
        self.servers.append(server)
        return server.start()

    def client(self, server: _FakeVLLMServer, **kwargs: object) -> VLLMClient:
        return VLLMClient(
            base_url=server.base_url,
            model=MODEL,
            temperature=0.0,
            max_output_tokens=1000,
            **kwargs,
        )

    def test_real_fake_server_round_trip_sends_approved_payload(self) -> None:
        server = self.fake_server()

        client = self.client(server, api_key="test-key")
        raw, metadata = client.complete_with_metadata(
            system_prompt="system", user_prompt="user"
        )

        self.assertEqual(raw, '{"status":"answered"}')
        self.assertIsInstance(metadata, OpenAIResponseMetadata)
        self.assertEqual(metadata.response_id, "chatcmpl_fake_vllm")
        self.assertEqual(metadata.returned_model, MODEL)
        self.assertEqual(metadata.input_tokens, 100)
        self.assertEqual(metadata.output_tokens, 20)
        self.assertEqual(metadata.total_tokens, 120)
        self.assertEqual(metadata.request_id, "req_fake_vllm")

        self.assertEqual(len(server.records), 1)
        record = server.records[0]
        self.assertEqual(record["path"], "/v1/chat/completions")
        self.assertEqual(record["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            record["body"],
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "temperature": 0.0,
                "max_tokens": 1000,
                "stream": False,
                "response_format": {"type": "json_object"},
            },
        )
        self.assertEqual(metadata.pacing_delay_seconds, 0.0)

    def test_omits_authorization_header_without_api_key(self) -> None:
        server = self.fake_server()

        self.client(server).complete(system_prompt="system", user_prompt="user")

        self.assertNotIn("Authorization", server.records[0]["headers"])

    def test_complete_returns_raw_text_only(self) -> None:
        server = self.fake_server()

        raw = self.client(server).complete(system_prompt="system", user_prompt="user")

        self.assertEqual(raw, '{"status":"answered"}')

    def test_response_format_is_deep_copied_and_customizable(self) -> None:
        server = self.fake_server()
        text_format = {"type": "json_schema", "name": "frozen_answer_v1", "strict": True}
        client = self.client(server, text_format=text_format)

        client.complete(system_prompt="system", user_prompt="user")
        text_format["name"] = "mutated_after_construction"

        self.assertEqual(server.records[0]["body"]["response_format"], {
            "type": "json_schema", "name": "frozen_answer_v1", "strict": True,
        })

    def test_sends_qwen_non_thinking_sampling_options(self) -> None:
        server = self.fake_server()
        options = {
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        client = self.client(server, payload_options=options)

        client.complete(system_prompt="system", user_prompt="user")
        options["chat_template_kwargs"]["enable_thinking"] = True

        body = server.records[0]["body"]
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["presence_penalty"], 1.5)
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_rejects_a_different_returned_model(self) -> None:
        server = self.fake_server(payload=completion_response(model="other-model"))

        with self.assertRaisesRegex(VLLMModelMismatchError, "expected"):
            self.client(server).complete(system_prompt="system", user_prompt="user")

    def test_rejects_missing_content_string(self) -> None:
        payload = completion_response()
        payload["choices"] = [{"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}]
        server = self.fake_server(payload=payload)

        with self.assertRaisesRegex(VLLMResponseError, "exactly one content"):
            self.client(server).complete(system_prompt="system", user_prompt="user")

    def test_rejects_missing_token_usage(self) -> None:
        payload = completion_response()
        payload["usage"] = {"prompt_tokens": 100}
        server = self.fake_server(payload=payload)

        with self.assertRaisesRegex(VLLMResponseError, "token usage is missing"):
            self.client(server).complete(system_prompt="system", user_prompt="user")

    def test_rejects_multiple_choices(self) -> None:
        payload = completion_response()
        payload["choices"].append({"index": 1, "message": {"role": "assistant", "content": "x"}})
        server = self.fake_server(payload=payload)

        with self.assertRaisesRegex(VLLMResponseError, "exactly one choice"):
            self.client(server).complete(system_prompt="system", user_prompt="user")

    def test_http_error_is_sanitized(self) -> None:
        server = self.fake_server(
            status=400,
            payload={
                "error": {
                    "message": "Input must contain the word JSON.",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": None,
                }
            },
        )

        with self.assertRaisesRegex(VLLMResponseError, "vLLM API returned HTTP 400") as raised:
            self.client(server).complete(system_prompt="system", user_prompt="user")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.provider_code, "invalid_request_error")
        self.assertEqual(raised.exception.provider_param, "input")
        self.assertEqual(raised.exception.provider_reason, "json_instruction_missing")

    def test_paces_second_request_by_minimum_interval(self) -> None:
        server = self.fake_server()
        sleeps: list[float] = []

        client = self.client(
            server,
            sleep=sleeps.append,
            minimum_request_interval_seconds=5.0,
        )
        client.complete(system_prompt="system", user_prompt="first")
        _, second = client.complete_with_metadata(
            system_prompt="system", user_prompt="second"
        )

        self.assertEqual(sleeps, [5.0])
        self.assertEqual(second.pacing_delay_seconds, 5.0)

    def test_rejects_invalid_constructor_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_url"):
            VLLMClient(base_url="", model=MODEL, temperature=0.0, max_output_tokens=1000)
        with self.assertRaisesRegex(ValueError, "model"):
            VLLMClient(base_url="http://127.0.0.1:8000", model="", temperature=0.0, max_output_tokens=1000)
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            VLLMClient(base_url="http://127.0.0.1:8000", model=MODEL, temperature=0.0, max_output_tokens=0)
        with self.assertRaisesRegex(ValueError, "minimum_request_interval_seconds"):
            VLLMClient(
                base_url="http://127.0.0.1:8000", model=MODEL, temperature=0.0,
                max_output_tokens=1000, minimum_request_interval_seconds=-1,
            )
        with self.assertRaisesRegex(ValueError, "protected fields"):
            VLLMClient(
                base_url="http://127.0.0.1:8000", model=MODEL, temperature=0.0,
                max_output_tokens=1000, payload_options={"model": "other"},
            )


if __name__ == "__main__":
    unittest.main()
