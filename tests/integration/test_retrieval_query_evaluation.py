from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from retrieval.query_evaluation import (
    FINAL_ARTIFACTS,
    QueryEvaluationError,
    execute_query_evaluation,
    execute_query_runtime,
    score_query_release,
    verify_query_release,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
CHECKED = ROOT / "results/retrieval/query-planning-development-v1"
RUNTIME_FILES = (
    "predictions.jsonl",
    "filter-decisions.jsonl",
    "failures.jsonl",
    "runtime-checkpoint.json",
)


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for query evaluation tests",
)
class RetrievalQueryEvaluationIntegrationTests(unittest.TestCase):
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

    def _runtime(self, output: Path):
        return execute_query_runtime(self._factory, output, repo_root=ROOT)

    def _full(self, output: Path):
        return execute_query_evaluation(self._factory, output, repo_root=ROOT)

    def test_runtime_executes_before_reference_and_never_opens_scorer_paths(self) -> None:
        original_open = Path.open
        forbidden_opened = []

        def guarded_open(path: Path, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "r")
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return original_open(path, *args, **kwargs)
            if "r" in mode or "+" in mode:
                forbidden = any(
                    token in relative
                    for token in (
                        "reference.jsonl",
                        "/gold/",
                        "benchmark_qa",
                        "oracle",
                        "review_queue",
                        "test_user",
                    )
                )
                if forbidden:
                    forbidden_opened.append(relative)
                    raise AssertionError(f"forbidden runtime read: {relative}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "open", guarded_open):
                predictions = self._runtime(Path(directory) / "runtime")
        self.assertEqual(len(predictions), 24)
        self.assertEqual(forbidden_opened, [])

    def test_two_clean_runtime_runs_are_byte_identical_and_match_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            self._runtime(first)
            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(first.iterdir())
            }
            self._clean_database()
            self._runtime(second)
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(second.iterdir())
            }
        checked_hashes = {
            name: hashlib.sha256((CHECKED / name).read_bytes()).hexdigest()
            for name in RUNTIME_FILES
        }
        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(first_hashes, checked_hashes)

    def test_full_release_has_exact_metrics_and_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            checks = self._full(output)
            verify_query_release(output, repo_root=ROOT)
            self.assertEqual(checks.label_accuracy.numerator, 24)
            self.assertEqual(checks.plan_expectation_accuracy.numerator, 24)
            self.assertEqual(checks.eligibility_expectation_accuracy.numerator, 24)
            self.assertEqual(checks.failure_count, 0)
            self.assertEqual(checks.cross_user_count, 0)
            self.assertEqual(checks.restricted_leakage_count, 0)
            self.assertEqual(checks.post_cutoff_leakage_count, 0)
            for name in RUNTIME_FILES:
                self.assertEqual((output / name).read_bytes(), (CHECKED / name).read_bytes())

    def test_two_scorer_runs_over_one_checkpoint_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outputs = []
            for name in ("first", "second"):
                output = base / name
                output.mkdir()
                for runtime_name in RUNTIME_FILES:
                    shutil.copyfile(CHECKED / runtime_name, output / runtime_name)
                score_query_release(output, repo_root=ROOT)
                outputs.append(
                    {
                        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(output.iterdir())
                    }
                )
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(set(outputs[0]), {*FINAL_ARTIFACTS, "manifest.json"})

    def test_checked_release_self_verifies_and_immutable_outputs_refuse_replay(self) -> None:
        verify_query_release(CHECKED, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._runtime(output)
            self._clean_database()
            with self.assertRaisesRegex(QueryEvaluationError, "must be empty"):
                self._runtime(output)


if __name__ == "__main__":
    unittest.main()
