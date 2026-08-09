from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from storage.contracts import MemoryUser
from storage.migrations import apply_migrations
from storage.repository import StorageRepository
from summaries.grounded_evaluation import (
    ARTIFACT_NAMES,
    RESULT_ROOT,
    GroundedSummaryEvaluationError,
    execute_grounded_evaluation,
    verify_grounded_release,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RELEASE_FILES = (*ARTIFACT_NAMES, "manifest.json")


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for grounded-summary evaluation",
)
class GroundedSummaryEvaluationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._reset_database()

    def tearDown(self) -> None:
        self._reset_database()

    @staticmethod
    def _factory():
        return psycopg.connect(DATABASE_URL, autocommit=True)

    @staticmethod
    def _reset_database() -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    def _run(self, output: Path):
        return execute_grounded_evaluation(
            self._factory,
            output,
            repo_root=ROOT,
        )

    def test_release_is_byte_identical_on_two_clean_databases(self) -> None:
        checked = ROOT / RESULT_ROOT
        self.assertTrue(checked.is_dir())
        verify_grounded_release(repo_root=ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_checks = self._run(first)
            self._reset_database()
            second_checks = self._run(second)
            self.assertEqual(first_checks, second_checks)
            for name in RELEASE_FILES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual((first / name).read_bytes(), (checked / name).read_bytes())

    def test_exact_release_counts_and_structural_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checks = self._run(Path(temporary) / "release")
        self.assertEqual(
            (
                checks.session_count,
                checks.summary_count,
                checks.empty_session_count,
                checks.failure_count,
                checks.statement_count,
            ),
            (20, 17, 3, 0, 34),
        )
        self.assertEqual(checks.provenance_integrity["value"], 1.0)
        self.assertEqual(checks.membership_integrity["value"], 1.0)
        self.assertEqual(checks.eligible_claim_accounting["value"], 1.0)
        self.assertEqual(
            (
                checks.cross_user_count,
                checks.stale_reference_count,
                checks.duplicate_statement_count,
                checks.unsupported_statement_count,
                checks.model_calls,
            ),
            (0, 0, 0, 0, 0),
        )
        self.assertTrue(checks.deletion_recompute_pass)
        self.assertFalse(checks.model_fallback)

    def test_nonempty_result_directory_is_refused_without_database_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            output.mkdir()
            (output / "existing").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(GroundedSummaryEvaluationError, "must be empty"):
                self._run(output)

    def test_dirty_database_is_refused(self) -> None:
        with self._factory() as connection:
            apply_migrations(connection, ROOT / "migrations")
            StorageRepository(connection).insert_user(
                MemoryUser("user_dirty", datetime.now(timezone.utc))
            )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(GroundedSummaryEvaluationError, "must be clean"):
                self._run(Path(temporary) / "release")


if __name__ == "__main__":
    unittest.main()
