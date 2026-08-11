from __future__ import annotations

from decimal import Decimal
import io
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from evaluation.openai_client import OpenAIResponseMetadata
from extraction.run_phase4_fallback import (
    CUMULATIVE_HARD_MAXIMUM_USD,
    FALLBACK_CONFIG_PATH,
    FALLBACK_OUTPUT_ROOT,
    NEW_HARD_MAXIMUM_USD,
    dry_run_fallback,
    execute_fallback,
    prepare_fallback,
)
from extraction.run_phase4_input import Step35RunError
from extraction.run_safety import additive_cost_text, cost_text


REPO_ROOT = Path(__file__).resolve().parents[2]


class _EmptyClaimsClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_metadata(self, *, system_prompt: str, user_prompt: str):
        self.calls += 1
        return (
            '{"claims":[]}',
            OpenAIResponseMetadata(
                response_id=f"response_{self.calls}",
                returned_model="gpt-4.1-2025-04-14",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                request_id=f"request_{self.calls}",
            ),
        )


class Phase4FallbackRunnerTest(unittest.TestCase):
    def _repo(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / "results/phase3").mkdir(parents=True)
        (root / "configs").symlink_to(REPO_ROOT / "configs", target_is_directory=True)
        (root / "data").symlink_to(REPO_ROOT / "data", target_is_directory=True)
        (root / "results/phase3/atomic-extraction-step35-model-qualification-v2").symlink_to(
            REPO_ROOT / "results/phase3/atomic-extraction-step35-model-qualification-v2",
            target_is_directory=True,
        )
        return root

    def test_dry_run_reuses_four_and_reserves_sixteen_exact_requests(self) -> None:
        output = REPO_ROOT / FALLBACK_OUTPUT_ROOT
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.is_file()
        }
        record = dry_run_fallback(REPO_ROOT, FALLBACK_CONFIG_PATH, stdout=io.StringIO())
        self.assertEqual(record["provider_calls"], 0)
        self.assertEqual(record["output_writes"], 0)
        self.assertEqual(record["reused_prediction_count"], 4)
        self.assertEqual(record["new_request_attempt_count"], 16)
        self.assertEqual(record["retry_request_count"], 0)
        self.assertEqual(
            record["current_hard_maximum_cost_usd"],
            additive_cost_text(NEW_HARD_MAXIMUM_USD),
        )
        self.assertEqual(
            record["cumulative_hard_maximum_cost_usd"],
            additive_cost_text(CUMULATIVE_HARD_MAXIMUM_USD),
        )
        self.assertEqual(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir()
                if path.is_file()
            },
            before,
        )

    def test_reuse_is_revalidated_and_remaining_order_is_original_order(self) -> None:
        plan, reused, remaining = prepare_fallback(REPO_ROOT, FALLBACK_CONFIG_PATH)
        self.assertEqual([item["position"] for item in reused], [1, 4, 6, 7])
        self.assertEqual(
            [plan.sources[position - 1].source_id for position in remaining],
            [
                source.source_id
                for position, source in enumerate(plan.sources, 1)
                if position not in {1, 4, 6, 7}
            ],
        )
        for item in reused:
            self.assertEqual(len(item["provider_metadata"]["charged_cost_usd"].split(".")[1]), 7)

    def test_new_costs_keep_seven_decimals_without_changing_legacy_formatter(self) -> None:
        value = Decimal("0.019646") + Decimal("0.0501504")
        self.assertEqual(additive_cost_text(value), "0.0697964")
        self.assertEqual(cost_text(value), "0.069797")

    def test_execution_writes_no_gold_until_all_twenty_predictions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._repo(temporary)
            client = _EmptyClaimsClient()

            def guarded_gold(*args, **kwargs):
                predictions = (root / FALLBACK_OUTPUT_ROOT / "predictions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(len(predictions), 20)
                return ()

            with mock.patch(
                "extraction.run_phase4_fallback.load_scaled_development_gold",
                side_effect=guarded_gold,
            ), mock.patch(
                "extraction.run_phase4_fallback.score_scaled_development",
                return_value={"case_results": []},
            ):
                result = execute_fallback(
                    repo_root=root,
                    client=client,
                    config_path=FALLBACK_CONFIG_PATH,
                )
            self.assertTrue(result["generation_complete"])
            self.assertEqual(client.calls, 16)
            predictions = [
                json.loads(line)
                for line in (root / FALLBACK_OUTPUT_ROOT / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([item["position"] for item in predictions], list(range(1, 21)))

    def test_arbitrary_fallback_config_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(Step35RunError, "not allowlisted"):
            prepare_fallback(REPO_ROOT, Path("configs/extraction/not-fallback.json"))

    def test_completed_resume_verifies_and_preserves_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._repo(temporary)
            shutil.copytree(
                REPO_ROOT / FALLBACK_OUTPUT_ROOT,
                root / FALLBACK_OUTPUT_ROOT,
            )
            output = root / FALLBACK_OUTPUT_ROOT
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir()
                if path.is_file()
            }
            client = _EmptyClaimsClient()

            result = execute_fallback(
                repo_root=root,
                client=client,
                resume=True,
                config_path=FALLBACK_CONFIG_PATH,
            )

            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir()
                if path.is_file()
            }
            self.assertTrue(result["generation_complete"])
            self.assertEqual(result["prediction_count"], 20)
            self.assertEqual(client.calls, 0)
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
