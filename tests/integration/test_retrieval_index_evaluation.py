from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from retrieval.index_evaluation import (
    ARTIFACT_NAMES,
    EXPECTED,
    RetrievalIndexEvaluationError,
    execute_index_evaluation,
    verify_index_release,
)
import retrieval.index_evaluation as index_evaluation


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for retrieval evaluation tests",
)
class RetrievalIndexEvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admin = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.admin.close()

    def setUp(self) -> None:
        self._clean_database()

    def tearDown(self) -> None:
        self._clean_database()

    def _clean_database(self) -> None:
        self.admin.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.admin.execute("CREATE SCHEMA public")

    def _factory(self):
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def _run(self, output: Path):
        return execute_index_evaluation(self._factory, output, repo_root=ROOT)

    def test_clean_development_build_has_exact_counts_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checks = self._run(Path(directory) / "release")
        self.assertEqual(checks.user_count, EXPECTED["user_count"])
        self.assertEqual(checks.atomic_record_count, 33)
        self.assertEqual(checks.session_record_count, 17)
        self.assertEqual(checks.durative_record_count, 0)
        self.assertEqual(checks.record_count, 50)
        self.assertEqual(checks.run_count, 2)
        self.assertEqual(checks.failure_count, 0)
        self.assertTrue(checks.exact_claim_lineage)
        self.assertTrue(checks.exact_source_lineage)
        self.assertTrue(checks.exact_relation_lineage)
        self.assertEqual(checks.relation_line_count, 0)
        self.assertTrue(checks.replay_pass)
        self.assertTrue(checks.deletion_pass)
        self.assertEqual(
            self.admin.execute(
                "SELECT count(*) FROM retrieval_index_runs WHERE status = 'succeeded'"
            ).fetchone()[0],
            2,
        )
        self.assertEqual(
            self.admin.execute("SELECT count(*) FROM retrieval_index_records").fetchone()[0],
            50,
        )

    def test_vectors_fts_and_user_ownership_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checks = self._run(Path(directory) / "release")
        self.assertEqual(checks.vector_dimension, 256)
        self.assertEqual(checks.normalized_vector_count, 50)
        self.assertEqual(checks.fts_document_count, 50)
        self.assertEqual(checks.cross_user_count, 0)
        self.assertEqual(checks.restricted_count, 0)
        self.assertEqual(checks.partial_count, 0)
        ownership = self.admin.execute(
            """
            SELECT count(*) FROM retrieval_index_records AS record
            JOIN retrieval_index_claim_links AS claim
              ON claim.index_record_id = record.index_record_id
            WHERE claim.user_id <> record.user_id
            """
        ).fetchone()[0]
        self.assertEqual(ownership, 0)

    def test_two_clean_database_releases_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            self._run(first)
            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(first.iterdir())
            }
            self._clean_database()
            self._run(second)
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(second.iterdir())
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(set(first_hashes), {*ARTIFACT_NAMES, "manifest.json"})

    def test_release_self_verifies_and_refuses_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._run(output)
            verify_index_release(output, repo_root=ROOT)
            self._clean_database()
            with self.assertRaisesRegex(RetrievalIndexEvaluationError, "must be empty"):
                self._run(output)

    def test_full_execution_reads_only_approved_runtime_inputs(self) -> None:
        manifest = json.loads(
            (ROOT / index_evaluation.DATASET_MANIFEST).read_text(encoding="utf-8")
        )
        approved = {
            index_evaluation.DATASET_MANIFEST.as_posix(),
            index_evaluation.CONFIG_PATH.as_posix(),
            *index_evaluation.IMPLEMENTATION_PATHS,
            index_evaluation.STEP63_MANIFEST_PATH.as_posix(),
            index_evaluation.STEP64_RESULT_MANIFEST_PATH.as_posix(),
            index_evaluation.STEP64_CHECKPOINT_MANIFEST_PATH.as_posix(),
            index_evaluation.STEP64_CHECKPOINT_PREFLIGHT_PATH.as_posix(),
        }
        for binding in manifest["inputs"].values():
            approved.update(
                value
                for name, value in binding.items()
                if name.endswith("_path")
            )
        approved_prefixes = (
            "migrations/",
            "configs/summaries/",
            "data/scaled-v1/runtime/",
            "data/summaries/grounded-summary-development-v1/",
            "results/phase3/phase4-input-development-gpt41-fallback-v1/",
            "results/phase4/step4.3-temporal-lifecycle-v1/",
            "results/conflicts/",
            "results/summaries/grounded-summary-development-v1/",
            "results/summaries/sessionization-development-v1/",
            "results/summaries/durative-claim-development-v1/",
        )
        authority_paths = {
            index_evaluation.STEP64_RESULT_MANIFEST_PATH.as_posix(),
            index_evaluation.STEP64_CHECKPOINT_MANIFEST_PATH.as_posix(),
            index_evaluation.STEP64_CHECKPOINT_PREFLIGHT_PATH.as_posix(),
        }
        original_open = Path.open
        opened = set()

        def guarded_open(path: Path, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "r")
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return original_open(path, *args, **kwargs)
            if "r" in mode or "+" in mode:
                forbidden = (
                    "/gold/" in f"/{relative}"
                    or "oracle" in relative
                    or "review_queue" in relative
                    or "test_user" in relative
                    or "summary_quality_scorer" in relative
                    or "summary_quality_evaluation" in relative
                    or "test_summary_quality_evaluation" in relative
                )
                if forbidden and relative not in authority_paths:
                    raise AssertionError(f"forbidden runtime read: {relative}")
                if relative not in approved and not relative.startswith(approved_prefixes):
                    raise AssertionError(f"unapproved runtime read: {relative}")
                opened.add(relative)
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "open", guarded_open):
                checks = self._run(Path(directory) / "release")
        self.assertEqual(checks.record_count, 50)
        self.assertTrue(authority_paths.issubset(opened))


if __name__ == "__main__":
    unittest.main()
