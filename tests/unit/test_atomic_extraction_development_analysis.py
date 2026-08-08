from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from extraction.development_analysis import (
    DevelopmentAnalysisError,
    run_development_analysis,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class AtomicExtractionDevelopmentAnalysisTests(unittest.TestCase):
    def test_analysis_reconciles_smoke_and_preserves_failed_full_run(self) -> None:
        fixed = datetime.fromisoformat("2026-08-08T12:00:00+00:00")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first) / "analysis"
            second_dir = Path(second) / "analysis"
            first_manifest = run_development_analysis(
                repo_root=REPO_ROOT,
                output_dir=first_dir,
                now=lambda: fixed,
            )
            second_manifest = run_development_analysis(
                repo_root=REPO_ROOT,
                output_dir=second_dir,
                now=lambda: fixed,
            )
            summary = json.loads((first_dir / "summary.json").read_text())

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (first_dir / "summary.json").read_bytes(),
                (second_dir / "summary.json").read_bytes(),
            )
            self.assertEqual(summary["candidate_status"], "blocked_failed_validation")
            self.assertEqual(summary["full_run"]["metrics"], None)
            self.assertEqual(summary["full_run"]["failure_stage"], "validation")
            self.assertEqual(
                len(summary["smoke_comparison"]["case_ids"]), 3
            )
            self.assertIn("by_source_type", summary["smoke_failure_analysis"])
            self.assertIn("by_predicate_family", summary["smoke_failure_analysis"])
            self.assertIn("by_category", summary["smoke_failure_analysis"])
            self.assertEqual(
                summary["smoke_failure_analysis"]["category_counts"][
                    "unsupported_claim"
                ],
                summary["smoke_comparison"]["candidate_total_unsupported_claims"],
            )

    def test_analysis_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "analysis"
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(
                DevelopmentAnalysisError, "refusing to overwrite"
            ):
                run_development_analysis(
                    repo_root=REPO_ROOT,
                    output_dir=output_dir,
                )

    def test_analysis_has_no_model_or_api_client_dependency(self) -> None:
        module = (
            REPO_ROOT / "src/extraction/development_analysis.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("OpenAIResponsesClient", module)
        self.assertNotIn("load_env_value", module)


if __name__ == "__main__":
    unittest.main()
