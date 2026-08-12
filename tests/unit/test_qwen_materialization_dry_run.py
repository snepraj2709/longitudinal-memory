from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_materialization_dry_run import QwenMaterializationDryRunError, run_dry_run


ROOT = Path(__file__).resolve().parents[2]


class QwenMaterializationDryRunTests(unittest.TestCase):
    def test_database_reset_requires_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QwenMaterializationDryRunError, "allow-reset-db"):
                run_dry_run(
                    repo_root=ROOT,
                    database_url="postgresql://example",
                    output_dir=Path(directory) / "dry-run",
                    allow_reset_db=False,
                )


if __name__ == "__main__":
    unittest.main()
