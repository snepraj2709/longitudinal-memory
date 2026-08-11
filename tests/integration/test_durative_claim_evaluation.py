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
from summaries.durative_evaluation import (
    ARTIFACT_NAMES,
    RESULT_ROOT,
    DurativeEvaluationError,
    execute_durative_evaluation,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RELEASE_FILES = (*ARTIFACT_NAMES, "manifest.json")
FROZEN_MANIFEST_SHA256 = "1d3f1c78d95bd96399224581bec21143c4b52562517d4779b74850e42d26fdbb"
AUTHORIZED_IMPLEMENTATION_DRIFT = {
    "Makefile",
    "src/summaries/durative_evaluation.py",
    "tests/integration/test_durative_claim_persistence.py",
    "tests/integration/test_durative_claim_evaluation.py",
    "tests/unit/test_durative_evaluation.py",
}
AUTHORIZED_PREDECESSOR_HASH_DRIFT = {
    "Makefile",
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "src/summaries/grounded_evaluation.py",
    "tests/integration/test_belief_resolution.py",
    "tests/integration/test_conflict_relations.py",
    "tests/integration/test_grounded_summary_evaluation.py",
    "tests/integration/test_grounded_summary_persistence.py",
    "tests/integration/test_phase4_storage.py",
    "tests/integration/test_phase5_conflict_evaluation.py",
    "tests/integration/test_temporal_service.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
    "tests/unit/test_grounded_summary_evaluation.py",
}
FRESH_ONLY_PREDECESSOR_DRIFT = {
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "src/summaries/grounded_evaluation.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
}


def verify_frozen_release(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise AssertionError("frozen durative manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise AssertionError("frozen durative artifact changed")
    drift = {
        relative
        for relative, expected in manifest["implementation_hashes"].items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected
    }
    if drift != AUTHORIZED_IMPLEMENTATION_DRIFT:
        raise AssertionError("durative compatibility drift changed")
    return manifest


def verify_fresh_manifest_adapter(fresh_path: Path, frozen: dict[str, object]) -> None:
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(fresh))
    implementation_drift = {
        path
        for path, frozen_hash in frozen["implementation_hashes"].items()
        if fresh["implementation_hashes"][path] != frozen_hash
    }
    if implementation_drift != AUTHORIZED_IMPLEMENTATION_DRIFT:
        raise AssertionError("fresh durative implementation drift changed")
    for path in implementation_drift:
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if fresh["implementation_hashes"][path] != current:
            raise AssertionError("fresh durative implementation hash is stale")
        normalized["implementation_hashes"][path] = frozen["implementation_hashes"][path]

    fresh_predecessor = {
        item["path"]: item for item in fresh["predecessor"]["authorized_drift"]
    }
    frozen_predecessor = {
        item["path"]: item for item in frozen["predecessor"]["authorized_drift"]
    }
    if set(fresh_predecessor) - set(frozen_predecessor) != FRESH_ONLY_PREDECESSOR_DRIFT:
        raise AssertionError("fresh durative predecessor drift additions changed")
    predecessor_drift = {
        path
        for path, item in frozen_predecessor.items()
        if fresh_predecessor[path]["step6_3_sha256"] != item["step6_3_sha256"]
    }
    if predecessor_drift | FRESH_ONLY_PREDECESSOR_DRIFT != AUTHORIZED_PREDECESSOR_HASH_DRIFT:
        raise AssertionError("fresh durative predecessor drift changed")
    normalized_predecessor = {
        item["path"]: item for item in normalized["predecessor"]["authorized_drift"]
    }
    for path in FRESH_ONLY_PREDECESSOR_DRIFT:
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if fresh_predecessor[path]["step6_3_sha256"] != current:
            raise AssertionError("fresh durative predecessor addition hash is stale")
        del normalized_predecessor[path]
    for path in predecessor_drift:
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if fresh_predecessor[path]["step6_3_sha256"] != current:
            raise AssertionError("fresh durative predecessor hash is stale")
        normalized_predecessor[path]["step6_3_sha256"] = frozen_predecessor[path][
            "step6_3_sha256"
        ]
    normalized["predecessor"]["authorized_drift"] = [
        normalized_predecessor[path] for path in sorted(normalized_predecessor)
    ]
    normalized["predecessor"]["unchanged_file_count"] += len(FRESH_ONLY_PREDECESSOR_DRIFT)
    if normalized != frozen:
        raise AssertionError("fresh durative manifest changed outside approved hashes")


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
        frozen_manifest = verify_frozen_release(checked)
        self.assertTrue((checked / "rejections.jsonl").is_file())
        self.assertFalse((checked / "decisions.jsonl").exists())
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_score = self._run(first)
            self._reset_database()
            second_score = self._run(second)
            self.assertEqual(first_score, second_score)
            for name in ARTIFACT_NAMES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual((first / name).read_bytes(), (checked / name).read_bytes())
            verify_fresh_manifest_adapter(first / "manifest.json", frozen_manifest)
            verify_fresh_manifest_adapter(second / "manifest.json", frozen_manifest)

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
