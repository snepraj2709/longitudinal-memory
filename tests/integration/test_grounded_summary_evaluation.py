from __future__ import annotations

import os
import hashlib
import json
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
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RELEASE_FILES = (*ARTIFACT_NAMES, "manifest.json")
FROZEN_RELEASE_MANIFEST_SHA256 = "ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761"
AUTHORIZED_STEP63_DRIFT = {
    "Makefile",
    "src/summaries/grounded_evaluation.py",
    "tests/integration/test_grounded_summary_persistence.py",
    "tests/integration/test_grounded_summary_evaluation.py",
    "tests/unit/test_grounded_summary_evaluation.py",
}
AUTHORIZED_STEP63_PREDECESSOR_HASH_DRIFT = {
    "Makefile",
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "tests/integration/test_belief_resolution.py",
    "tests/integration/test_conflict_relations.py",
    "tests/integration/test_phase4_storage.py",
    "tests/integration/test_phase5_conflict_evaluation.py",
    "tests/integration/test_temporal_service.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
}
FRESH_ONLY_PREDECESSOR_DRIFT = {
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
}


def verify_frozen_release(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != FROZEN_RELEASE_MANIFEST_SHA256:
        raise AssertionError("frozen grounded-summary manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise AssertionError("frozen grounded-summary artifact changed")
    drift = {
        relative
        for relative, expected in manifest["implementation_hashes"].items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected
    }
    if drift != AUTHORIZED_STEP63_DRIFT:
        raise AssertionError("grounded-summary compatibility drift changed")


def verify_fresh_manifest_adapter(fresh_path: Path, frozen_path: Path) -> None:
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(fresh))

    implementation_drift = {
        path
        for path, frozen_hash in frozen["implementation_hashes"].items()
        if fresh["implementation_hashes"][path] != frozen_hash
    }
    if implementation_drift != AUTHORIZED_STEP63_DRIFT:
        raise AssertionError("fresh grounded implementation drift changed")
    for path in implementation_drift:
        if fresh["implementation_hashes"][path] != hashlib.sha256(
            (ROOT / path).read_bytes()
        ).hexdigest():
            raise AssertionError("fresh grounded implementation hash is stale")
        normalized["implementation_hashes"][path] = frozen["implementation_hashes"][path]

    fresh_predecessor = {
        item["path"]: item for item in fresh["predecessor"]["authorized_drift"]
    }
    frozen_predecessor = {
        item["path"]: item for item in frozen["predecessor"]["authorized_drift"]
    }
    if set(fresh_predecessor) - set(frozen_predecessor) != FRESH_ONLY_PREDECESSOR_DRIFT:
        raise AssertionError("fresh grounded predecessor drift additions changed")
    predecessor_hash_drift = {
        path
        for path, item in frozen_predecessor.items()
        if fresh_predecessor[path]["step6_2_sha256"] != item["step6_2_sha256"]
    }
    if predecessor_hash_drift | FRESH_ONLY_PREDECESSOR_DRIFT != AUTHORIZED_STEP63_PREDECESSOR_HASH_DRIFT:
        raise AssertionError("fresh grounded predecessor hash drift changed")
    normalized_predecessor = {
        item["path"]: item
        for item in normalized["predecessor"]["authorized_drift"]
    }
    for path in FRESH_ONLY_PREDECESSOR_DRIFT:
        if fresh_predecessor[path]["step6_2_sha256"] != hashlib.sha256(
            (ROOT / path).read_bytes()
        ).hexdigest():
            raise AssertionError("fresh grounded predecessor addition hash is stale")
        del normalized_predecessor[path]
    for path in predecessor_hash_drift:
        if fresh_predecessor[path]["step6_2_sha256"] != hashlib.sha256(
            (ROOT / path).read_bytes()
        ).hexdigest():
            raise AssertionError("fresh grounded predecessor hash is stale")
        normalized_predecessor[path]["step6_2_sha256"] = frozen_predecessor[path][
            "step6_2_sha256"
        ]
    normalized["predecessor"]["authorized_drift"] = [
        normalized_predecessor[path] for path in sorted(normalized_predecessor)
    ]
    normalized["predecessor"]["unchanged_file_count"] += len(FRESH_ONLY_PREDECESSOR_DRIFT)
    if normalized != frozen:
        raise AssertionError("fresh grounded manifest changed outside approved hashes")


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
        verify_frozen_release(checked)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_checks = self._run(first)
            self._reset_database()
            second_checks = self._run(second)
            self.assertEqual(first_checks, second_checks)
            for name in RELEASE_FILES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            for name in ARTIFACT_NAMES:
                self.assertEqual((first / name).read_bytes(), (checked / name).read_bytes())
            verify_fresh_manifest_adapter(first / "manifest.json", checked / "manifest.json")
            verify_fresh_manifest_adapter(second / "manifest.json", checked / "manifest.json")

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
