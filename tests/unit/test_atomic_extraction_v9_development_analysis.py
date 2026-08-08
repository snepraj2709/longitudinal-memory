from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from extraction.development_analysis import DevelopmentAnalysisError, METRICS
from extraction.v9_development_analysis import run_v9_development_analysis


REPO_ROOT = Path(__file__).resolve().parents[2]


class AtomicExtractionV9DevelopmentAnalysisTests(unittest.TestCase):
    def test_analysis_reconciles_metrics_and_rejects_regression(self) -> None:
        fixed = datetime.fromisoformat("2026-08-09T04:00:00+00:00")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first) / "analysis"
            second_dir = Path(second) / "analysis"
            first_manifest = run_v9_development_analysis(
                repo_root=REPO_ROOT, output_dir=first_dir, now=lambda: fixed
            )
            second_manifest = run_v9_development_analysis(
                repo_root=REPO_ROOT, output_dir=second_dir, now=lambda: fixed
            )
            summary = json.loads((first_dir / "summary.json").read_text())
            comparisons = [
                json.loads(line)
                for line in (first_dir / "metric_comparison.jsonl").read_text().splitlines()
            ]

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(summary["decision"], "tune_another_candidate")
        self.assertFalse(summary["acceptance_checks"]["claim_f1_not_below_v2"])
        self.assertFalse(
            summary["acceptance_checks"]["unsupported_memory_rate_not_above_v2"]
        )
        self.assertEqual(len(comparisons), len(METRICS) * 3)
        self.assertEqual(summary["unsupported_claim_review"]["count"], 6)
        self.assertEqual(summary["v9_full_run"]["successful_case_count"], 10)
        self.assertEqual(first_manifest["review_status"], "awaiting_sneha_review")

    def test_analysis_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "analysis"
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(DevelopmentAnalysisError, "refusing to overwrite"):
                run_v9_development_analysis(repo_root=REPO_ROOT, output_dir=output_dir)

    def test_analysis_has_no_model_or_api_client_dependency(self) -> None:
        module = (REPO_ROOT / "src/extraction/v9_development_analysis.py").read_text()
        self.assertNotIn("OpenAIResponsesClient", module)
        self.assertNotIn("load_env_value", module)


if __name__ == "__main__":
    unittest.main()
