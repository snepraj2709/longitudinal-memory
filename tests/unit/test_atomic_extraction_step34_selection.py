from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from extraction.failure_analysis import AtomicFailureAnalysisError
from extraction.step34_selection import freeze_step34_selection


REPO_ROOT = Path(__file__).resolve().parents[2]


class AtomicExtractionStep34SelectionTests(unittest.TestCase):
    def test_freezes_approved_safe_selection_and_hashes(self) -> None:
        fixed = datetime.fromisoformat("2026-08-09T05:00:00+00:00")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first) / "selection"
            second_dir = Path(second) / "selection"
            first_manifest = freeze_step34_selection(
                repo_root=REPO_ROOT, output_dir=first_dir, now=lambda: fixed
            )
            second_manifest = freeze_step34_selection(
                repo_root=REPO_ROOT, output_dir=second_dir, now=lambda: fixed
            )
            decision = json.loads((first_dir / "decision.json").read_text())
            for name, expected in first_manifest["output_file_sha256"].items():
                actual = hashlib.sha256((first_dir / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["release_status"], "step_3_4_complete")
        self.assertEqual(first_manifest["review_status"], "approved_by_sneha")
        self.assertFalse(first_manifest["scope"]["frozen_test_user_records_accessed"])
        self.assertFalse(first_manifest["scope"]["api_calls_made_by_selection"])
        self.assertEqual(decision["selected_runtime_prompt_version"], "atomic-extraction-v3")
        self.assertEqual(decision["quality_evidence_prompt_version"], "atomic-extraction-v2")
        self.assertEqual(
            decision["measured_results"]["v2_safety_baseline"]["total_unsupported_claims"],
            0,
        )
        self.assertTrue(decision["phase_3_5"]["ready_to_begin"])
        self.assertFalse(decision["phase_3_5"]["started"])

    def test_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "selection"
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(AtomicFailureAnalysisError, "refusing to overwrite"):
                freeze_step34_selection(repo_root=REPO_ROOT, output_dir=output_dir)

    def test_selection_has_no_model_api_or_gold_loader_dependency(self) -> None:
        module = (REPO_ROOT / "src/extraction/step34_selection.py").read_text()
        self.assertNotIn("OpenAIResponsesClient", module)
        self.assertNotIn("load_env_value", module)
        self.assertNotIn("load_atomic_gold", module)


if __name__ == "__main__":
    unittest.main()
