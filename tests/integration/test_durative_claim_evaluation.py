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
from summaries.durative_evaluation import (
    ARTIFACT_NAMES,
    RESULT_ROOT,
    DurativeEvaluationError,
    execute_durative_evaluation,
    verify_durative_release,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RELEASE_FILES = (*ARTIFACT_NAMES, "manifest.json")


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for durative evaluation",
)
class DurativeClaimEvaluationIntegrationTests(unittest.TestCase):
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
        return execute_durative_evaluation(self._factory, output, repo_root=ROOT)

    def test_two_clean_database_runs_are_byte_identical_to_checked_release(self) -> None:
        checked = ROOT / RESULT_ROOT
        self.assertTrue(checked.is_dir())
        verify_durative_release(repo_root=ROOT)
        self.assertTrue((checked / "rejections.jsonl").is_file())
        self.assertFalse((checked / "decisions.jsonl").exists())
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_score = self._run(first)
            self._reset_database()
            second_score = self._run(second)
            self.assertEqual(first_score, second_score)
            for name in RELEASE_FILES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual((first / name).read_bytes(), (checked / name).read_bytes())

    def test_exact_rejection_accounting_provenance_and_zero_model_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            score = self._run(Path(temporary) / "release")
        self.assertEqual(
            (
                score.input_claim_count,
                score.durative_proposition_count,
                score.decision_count,
                score.accepted_count,
                score.rejected_count,
                score.derived_claim_count,
                score.failure_count,
                score.unsupported_predicate_count,
            ),
            (33, 26, 26, 0, 26, 0, 0, 7),
        )
        self.assertEqual(score.rejection_counts, {"counterevidence": 26})
        self.assertEqual(score.decision_accounting["value"], 1.0)
        self.assertEqual(score.input_claim_accounting["value"], 1.0)
        self.assertEqual(score.provenance_integrity["value"], 1.0)
        self.assertEqual(
            (
                score.cross_user_count,
                score.duplicate_decision_count,
                score.duplicate_episode_count,
                score.stale_reference_count,
                score.unsupported_decision_count,
                score.model_calls,
                score.retry_count,
                score.cost_usd,
            ),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
        self.assertTrue(score.idempotency_replay_pass)
        self.assertTrue(score.deletion_recompute_pass)

    def test_persisted_runs_cover_every_proposition_without_derived_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self._run(Path(temporary) / "release")
        with self._factory() as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM durative_inference_runs").fetchone()[0],
                2,
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT count(DISTINCT user_id), sum(decision_count)
                    FROM durative_inference_runs
                    """
                ).fetchone(),
                (2, 26),
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT count(*) FROM durative_inference_decisions
                    WHERE decision_status = 'rejected'
                    """
                ).fetchone()[0],
                26,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT count(*) FROM claims WHERE memory_kind = 'durative'"
                ).fetchone()[0],
                0,
            )

    def test_nonempty_result_directory_is_refused_before_database_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            output.mkdir()
            (output / "occupied").write_text("present", encoding="utf-8")
            with self.assertRaisesRegex(DurativeEvaluationError, "must be empty"):
                self._run(output)

    def test_dirty_database_is_refused(self) -> None:
        with self._factory() as connection:
            apply_migrations(connection, ROOT / "migrations")
            StorageRepository(connection).insert_user(
                MemoryUser("dirty_user", datetime.now(timezone.utc))
            )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(DurativeEvaluationError, "must be clean"):
                self._run(Path(temporary) / "release")


if __name__ == "__main__":
    unittest.main()
