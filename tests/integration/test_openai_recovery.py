from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from evaluation.frozen_run_contracts import FrozenRunError
from evaluation.frozen_run import run_extraction_batch
from evaluation.openai_recovery import MANIFEST_PATH, verify_interrupted_openai_history


ROOT = Path(__file__).resolve().parents[2]


class InterruptedOpenAIRecoveryTests(unittest.TestCase):
    def test_recovery_manifest_binds_failed_attempts_and_empty_predictions(self):
        manifest = verify_interrupted_openai_history(repo_root=ROOT)
        attempts = manifest["attempts"]
        self.assertEqual([item["accepted_prediction_count"] for item in attempts], [100, 0, 0])
        self.assertEqual([item["provider_request_count"] for item in attempts], [100, 20, 4])
        self.assertFalse(manifest["recovery_policy"]["openai_execution_authorized"])
        self.assertTrue(manifest["recovery_policy"]["qwen_regenerates_extraction"])

    def test_completed_extraction_resume_does_not_replay_or_change_artifacts(self):
        source = ROOT / "results/evaluation/frozen-run-v1/batches/batch_01_extraction_all_sources"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            shutil.copytree(source, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}

            class FailIfCalled:
                def complete_with_metadata(self, **_kwargs):
                    raise AssertionError("completed extraction was replayed")

            result = run_extraction_batch(
                repo_root=ROOT,
                output_dir=output,
                client=FailIfCalled(),
                resume=True,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_changed_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "manifest.json"
            changed.write_bytes((ROOT / MANIFEST_PATH).read_bytes() + b"\n")
            with self.assertRaisesRegex(FrozenRunError, "recovery manifest changed"):
                verify_interrupted_openai_history(repo_root=ROOT, manifest_path=changed)


if __name__ == "__main__":
    unittest.main()
