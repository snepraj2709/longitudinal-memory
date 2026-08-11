from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIResponseMetadata
from evaluation.qwen3_development import (
    PAID_RUN_CONFIRMATION,
    Qwen3DevelopmentError,
    _load_gold,
    load_config,
    run_development,
)
from evaluation.qwen_execution import ExecutionJob


ROOT = Path(__file__).resolve().parents[2]


def _job(position: int, series_id: str = "qwen3-8b-vllm-dev-v1") -> ExecutionJob:
    return ExecutionJob(
        request_id=f"request_{position}",
        position=position,
        split="development",
        task="qa",
        record_id=f"record_{position}",
        user_id="user_001",
        baseline_id=None,
        context_sha256=None,
        context_count=0,
        system_prompt="system",
        user_prompt="user",
        response_format={"type": "json_object"},
        max_output_tokens=10,
        validator=lambda raw, _metadata: json.loads(raw),
        series_id=series_id,
    )


class Qwen3DevelopmentTests(unittest.TestCase):
    def test_config_pins_development_contract(self) -> None:
        config = load_config(ROOT)
        self.assertEqual(config["series_id"], "qwen3-8b-vllm-dev-v1")
        self.assertEqual(config["model"]["hugging_face_id"], "Qwen/Qwen3-8B")
        self.assertEqual(config["model"]["model_alias"], "qwen3-8b-vllm")
        self.assertEqual(config["runtime"]["client_concurrency"], 1)
        self.assertEqual(config["runtime"]["temperature"], 0)
        self.assertTrue(config["workload"]["retry_validation_failures"])

    def test_gold_loader_filters_to_development_split(self) -> None:
        references = _load_gold(ROOT)
        for task in ("qa", "summary", "interactive"):
            self.assertTrue(references[task])
            self.assertEqual({row["split"] for row in references[task]}, {"development"})
            self.assertLess(len(references[task]), 500)
        self.assertEqual({row["user_id"] for row in references["claims"]}, {"user_001", "user_002"})

    def test_model_alias_must_match_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Qwen3DevelopmentError, "model alias"):
                run_development(
                    repo_root=ROOT,
                    output_root=Path(directory) / "run",
                    base_url="https://example.test",
                    model="wrong-model",
                    api_key="secret",
                    database_url="postgresql://example",
                    hourly_rate_inr=1,
                    setup_seconds=0,
                    paid_run_confirmation=PAID_RUN_CONFIRMATION,
                )

    def test_runner_uses_qwen3_series_temperature_zero_and_workers_one(self) -> None:
        extraction = {
            "successful_count": 20,
            "provider_request_count": 20,
            "transport_retry_count": 0,
            "input_tokens": 100,
            "output_tokens": 20,
        }
        answers = {
            "successful_count": 798,
            "provider_request_count": 798,
            "transport_retry_count": 0,
            "input_tokens": 200,
            "output_tokens": 40,
        }
        judge = {
            "provider_request_count": 112,
            "transport_retry_count": 0,
            "input_tokens": 30,
            "output_tokens": 10,
        }
        scorecard = {"execution_failure_count": 0, "series_id": "qwen3-8b-vllm-dev-v1"}
        metadata = OpenAIResponseMetadata("response", "qwen3-8b-vllm", 1, 1, 2)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch("evaluation.qwen3_development.build_extraction_jobs", return_value=(_job(1),)) as extraction_jobs,
                patch("evaluation.qwen3_development.build_answer_jobs", return_value=(_job(1),)) as answer_jobs,
                patch("evaluation.qwen3_development.execute_jobs", side_effect=[extraction, answers]) as execute,
                patch("evaluation.qwen3_development._jsonl", return_value=[{
                    "task": "extraction",
                    "baseline_id": None,
                    "status": "succeeded",
                    "position": 1,
                    "record_id": "source_1",
                    "user_id": "user_001",
                    "request_sha256": "a" * 64,
                    "output": {"claims": []},
                }]),
                patch("evaluation.qwen3_development._load_contexts", return_value=()),
                patch("evaluation.qwen3_development._materialize", return_value={"context_count": 912, "failure_count": 0}),
                patch("evaluation.qwen3_development.derive_b7_records", return_value=({"status": "succeeded"},)),
                patch("evaluation.qwen3_development.write_derived_b7", return_value={"failure_count": 0}),
                patch("evaluation.qwen3_development.seal_logical_predictions", return_value={"failure_count": 0}),
                patch("evaluation.qwen3_development.run_judge", return_value=judge) as run_judge,
                patch("evaluation.qwen3_development.score_sealed_release", return_value=scorecard) as score,
                patch("evaluation.qwen_execution.VLLMClient.complete_with_metadata", return_value=("{}", metadata)),
            ):
                result = run_development(
                    repo_root=ROOT,
                    output_root=output,
                    base_url="https://example.test/v1",
                    model="qwen3-8b-vllm",
                    api_key="secret",
                    database_url="postgresql://example",
                    hourly_rate_inr=1,
                    setup_seconds=0,
                    paid_run_confirmation=PAID_RUN_CONFIRMATION,
                )

        extraction_jobs.assert_called_once()
        self.assertEqual(extraction_jobs.call_args.kwargs["series_id"], "qwen3-8b-vllm-dev-v1")
        self.assertEqual(answer_jobs.call_args.kwargs["series_id"], "qwen3-8b-vllm-dev-v1")
        self.assertEqual(execute.call_args_list[0].kwargs["workers"], 1)
        self.assertEqual(execute.call_args_list[1].kwargs["workers"], 1)
        self.assertEqual(
            execute.call_args_list[1].kwargs["retryable_http_statuses"],
            (408, 429, 502, 503, 504, 520),
        )
        self.assertTrue(execute.call_args_list[1].kwargs["retry_validation_failures"])
        self.assertEqual(run_judge.call_args.kwargs["workers"], 1)
        self.assertEqual(
            run_judge.call_args.kwargs["retryable_http_statuses"],
            (408, 429, 502, 503, 504, 520),
        )
        self.assertEqual(run_judge.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(score.call_args.kwargs["series_id"], "qwen3-8b-vllm-dev-v1")
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
