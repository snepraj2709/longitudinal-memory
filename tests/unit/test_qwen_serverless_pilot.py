from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIResponseMetadata
from evaluation.qwen_execution import ExecutionJob, TransportRetryLedger
from evaluation.qwen_serverless_pilot import (
    PAID_SERVERLESS_CONFIRMATION,
    QwenServerlessPilotError,
    _execute_serverless_jobs,
    load_serverless_pilot_config,
    run_serverless_pilot,
)
from evaluation.vllm_client import VLLMResponseError


ROOT = Path(__file__).resolve().parents[2]


def _job(position: int) -> ExecutionJob:
    return ExecutionJob(
        request_id=f"request_{position}",
        position=position,
        split="development",
        task="qa",
        record_id=f"case_{position}",
        user_id="user_001",
        baseline_id="B6",
        context_sha256="a" * 64,
        context_count=1,
        system_prompt="system",
        user_prompt="user",
        response_format={"type": "json_object"},
        max_output_tokens=100,
        validator=lambda raw, _metadata: json.loads(raw),
    )


def _success():
    raw = json.dumps({
        "status": "answered",
        "answer": "Known fact.",
        "confidence": 0.8,
        "statements": ["Known fact."],
        "citations": [{"source_id": "source_1", "message_id": "message_1", "quote": "Known fact"}],
        "unresolved_parts": [],
        "abstention_reason": None,
    })
    return raw, OpenAIResponseMetadata("response_1", "qwen3-8b-vllm", 10, 5, 15)


class _Client:
    def __init__(self, call) -> None:
        self._call = call

    def complete_with_metadata(self, **_kwargs):
        return self._call()


class QwenServerlessPilotTests(unittest.TestCase):
    def test_config_preserves_vllm_safety_contract(self) -> None:
        config = load_serverless_pilot_config(ROOT)
        self.assertEqual(config["compatibility_request_count"], 12)
        self.assertEqual(config["concurrency"], 1)
        self.assertEqual(config["context_length"], 8192)
        self.assertEqual(config["temperature"], 0)
        self.assertEqual(config["pilot_id"], "qwen3-8b-vllm-pilot-v1")
        self.assertTrue(config["delete_after_pilot"])

    def test_runner_builds_twelve_jobs_and_uses_exact_deployed_model(self) -> None:
        manifest = {
            "schema_version": "qwen_vllm_compatibility_batch_v1",
            "status": "completed",
            "planned_request_count": 12,
            "provider_request_count": 12,
            "successful_count": 12,
            "failure_count": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch("evaluation.qwen_serverless_pilot._client_factory", return_value="factory") as factory:
                with patch("evaluation.qwen_serverless_pilot._execute_serverless_jobs", return_value=manifest) as execute:
                    receipt = run_serverless_pilot(
                        repo_root=ROOT,
                        base_url="https://in2.example/openai/deploy_123/v1",
                        model="qwen3-8b-vllm",
                        api_key="secret",
                        deployment_id="deploy_123",
                        gpu="L4",
                        output_root=Path(directory),
                        paid_serverless_confirmation=PAID_SERVERLESS_CONFIRMATION,
                    )
        factory.assert_called_once_with(
            base_url="https://in2.example/openai/deploy_123/v1",
            model="qwen3-8b-vllm",
            api_key="secret",
            temperature=0.0,
        )
        jobs = execute.call_args.args[0]
        self.assertEqual(len(jobs), 12)
        self.assertEqual(execute.call_args.kwargs["client_factory"], "factory")
        self.assertEqual(execute.call_args.kwargs["retryable_response_statuses"], (408, 429, 502, 503, 504))
        self.assertFalse(receipt["api_key_persisted"])
        self.assertEqual(receipt["series_id"], "qwen3-8b-vllm-pilot-v1")
        self.assertEqual(receipt["deployment"]["model"], "qwen3-8b-vllm")
        self.assertEqual(receipt["deployment"]["concurrency"], 1)
        self.assertEqual(receipt["deployment"]["context_length"], 8192)
        self.assertEqual(receipt["deployment"]["temperature"], 0)

    def test_runner_rejects_unapproved_path_and_missing_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QwenServerlessPilotError, "paid vLLM requests"):
                run_serverless_pilot(
                    repo_root=ROOT,
                    base_url="https://in2.example/openai/deploy_123/v1",
                    model="qwen3-8b-vllm",
                    api_key="secret",
                    deployment_id="deploy_123",
                    gpu="L4",
                    output_root=Path(directory),
                    paid_serverless_confirmation="missing",
                )
            with self.assertRaisesRegex(QwenServerlessPilotError, "approved vLLM path"):
                run_serverless_pilot(
                    repo_root=ROOT,
                    base_url="https://in2.example/openai/deploy_123/v1",
                    model="qwen3-8b-vllm",
                    api_key="secret",
                    deployment_id="deploy_123",
                    gpu="A100-80GB",
                    output_root=Path(directory),
                    paid_serverless_confirmation=PAID_SERVERLESS_CONFIRMATION,
                )

    def test_vllm_executor_retries_configured_transient_http_status_once(self) -> None:
        attempts = {"count": 0}

        def call():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise VLLMResponseError("busy", status_code=503)
            return _success()

        with tempfile.TemporaryDirectory() as directory:
            manifest = _execute_serverless_jobs(
                (_job(1),),
                output_dir=Path(directory),
                client_factory=lambda _job: _Client(call),
                retry_ledger=TransportRetryLedger(1),
                retryable_response_statuses=(408, 429, 502, 503, 504),
            )
            row = json.loads(next((Path(directory) / "requests").glob("*.json")).read_text())
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(manifest["provider_request_count"], 2)
        self.assertEqual(manifest["transport_retry_count"], 1)
        self.assertEqual(row["status"], "succeeded")

    def test_vllm_executor_does_not_retry_bad_request(self) -> None:
        attempts = {"count": 0}

        def call():
            attempts["count"] += 1
            raise VLLMResponseError("bad request", status_code=400)

        with tempfile.TemporaryDirectory() as directory:
            manifest = _execute_serverless_jobs(
                (_job(1),),
                output_dir=Path(directory),
                client_factory=lambda _job: _Client(call),
                retry_ledger=TransportRetryLedger(1),
                retryable_response_statuses=(408, 429, 502, 503, 504),
            )
            row = json.loads(next((Path(directory) / "requests").glob("*.json")).read_text())
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(manifest["provider_request_count"], 1)
        self.assertEqual(row["failure_stage"], "provider")
        self.assertEqual(row["failure_code"], "http_400")


if __name__ == "__main__":
    unittest.main()
