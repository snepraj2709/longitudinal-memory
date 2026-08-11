from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from demo.bundle import build_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[2]


class DemoBundleTests(unittest.TestCase):
    def test_bundle_is_byte_stable_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_bundle(ROOT, first)
            write_bundle(ROOT, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            lowered = first.read_text(encoding="utf-8").lower()
            for forbidden in (
                "oracle_fact", "api_key", "openai_api_key", "reviewer_identifier",
                "eval_answer", "failure_analysis", "manual_review", "gold",
            ):
                self.assertNotIn(forbidden, lowered)

    def test_bundle_contains_four_guided_cases_and_separate_series(self) -> None:
        bundle = build_bundle(ROOT)
        self.assertEqual(len(bundle["cases"]), 4)
        self.assertEqual(
            [run["series_id"] for run in bundle["runs"]],
            ["openai-gpt41-v1", "qwen35-27b-fp8-v1", "qwen3-8b-vllm-dev-v1"],
        )
        self.assertEqual(bundle["runs"][0]["status"], "interrupted_not_scored")
        self.assertEqual(bundle["runs"][2]["status"], "completed")
        qwen_rows = [
            row for row in bundle["scorecards"]
            if row["series_id"] == "qwen3-8b-vllm-dev-v1"
        ]
        self.assertEqual([row["baseline_id"] for row in qwen_rows], [f"B{index}" for index in range(8)])
        self.assertTrue(all(row["status"] == "development_scored" for row in qwen_rows))


if __name__ == "__main__":
    unittest.main()
