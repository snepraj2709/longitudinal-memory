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

from retrieval.baseline_evaluation import (
    BaselineEvaluationError,
    PREDECESSOR_DRIFT,
    PROTECTED_HASHES,
    execute_baseline_evaluation,
    verify_baseline_release,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
CHECKED = ROOT / "results/retrieval/baseline-execution-development-v1"


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for baseline evaluation tests",
)
class RetrievalBaselineEvaluationIntegrationTests(unittest.TestCase):
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

    def _execute(self, output: Path):
        return execute_baseline_evaluation(self._factory, output, repo_root=ROOT)

    def test_clean_release_has_8_queries_24_type_pure_results_and_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            checks = self._execute(output)
            verify_baseline_release(output, repo_root=ROOT)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(checks.query_count, 8)
        self.assertEqual(checks.result_count, 24)
        self.assertEqual((checks.b2_result_count, checks.b3_result_count, checks.b4_result_count), (8, 8, 8))
        self.assertEqual(checks.failure_count, 0)
        self.assertEqual(checks.observed_channel_count, checks.expected_channel_count)
        self.assertTrue(checks.type_purity)
        self.assertTrue(checks.contiguous_ranks)
        self.assertTrue(checks.complete_lineage)
        self.assertGreater(checks.pre_filter_rejection_count, 0)
        self.assertGreater(checks.post_rank_rejection_count, 0)
        self.assertEqual(len(rows), 24)
        for row in rows:
            baseline = row["result"]["baseline_id"]
            allowed = {"B2": {"atomic"}, "B3": {"session"}, "B4": {"atomic", "session"}}[baseline]
            self.assertTrue(all(item["record_kind"] in allowed for item in row["result"]["accepted"]))
            self.assertEqual(
                [item["rank"] for item in row["result"]["accepted"]],
                list(range(1, len(row["result"]["accepted"]) + 1)),
            )
            for item in row["result"]["accepted"]:
                self.assertTrue(item["claim_ids"])
                self.assertTrue(item["claim_version_ids"])
                self.assertTrue(item["source_ids"])
                self.assertTrue(item["span_ids"])

    def test_two_clean_database_releases_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self._execute(first)
            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(first.iterdir())
            }
            self._clean_database()
            self._execute(second)
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(second.iterdir())
            }
        self.assertEqual(first_hashes, second_hashes)

    def test_full_execution_read_trap_forbids_relevance_gold_review_and_test_users(self) -> None:
        original_open = Path.open
        forbidden_opened = []

        def guarded_open(path: Path, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "r")
            try:
                relative = path.resolve().relative_to(ROOT).as_posix().lower()
            except ValueError:
                return original_open(path, *args, **kwargs)
            if "r" in mode or "+" in mode:
                if any(
                    token in relative
                    for token in (
                        "retrieval-relevance",
                        "reference.jsonl",
                        "benchmark_qa",
                        "/gold/",
                        "oracle",
                        "review_queue",
                        "test_user",
                    )
                ):
                    forbidden_opened.append(relative)
                    raise AssertionError(f"forbidden runtime read: {relative}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "open", guarded_open):
                self._execute(Path(directory) / "release")
        self.assertEqual(forbidden_opened, [])

    def test_checked_release_self_verifies_and_refuses_nonempty_output(self) -> None:
        verify_baseline_release(CHECKED, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            self._clean_database()
            with self.assertRaisesRegex(BaselineEvaluationError, "must be empty"):
                self._execute(output)

    def test_release_has_no_relevance_artifact_metrics_or_model_usage(self) -> None:
        verify_baseline_release(CHECKED, repo_root=ROOT)
        self.assertEqual(
            {path.name for path in CHECKED.iterdir()},
            {
                "results.jsonl",
                "failures.jsonl",
                "checks.json",
                "run.json",
                "findings.md",
                "runtime-checkpoint.json",
                "manifest.json",
            },
        )
        manifest = json.loads((CHECKED / "manifest.json").read_text())
        checkpoint = json.loads((CHECKED / "runtime-checkpoint.json").read_text())
        self.assertFalse(checkpoint["relevance_opened"])
        self.assertFalse(checkpoint["retrieval_metrics_computed"])
        self.assertEqual(manifest["model_usage"]["provider_requests"], 0)
        self.assertEqual(manifest["model_usage"]["incremental_cost_usd"], 0)
        self.assertEqual(manifest["predecessor_drift"], list(PREDECESSOR_DRIFT))
        self.assertEqual(
            manifest["protected_hash_audit"]["runtime_verified_hashes"],
            dict(sorted(PROTECTED_HASHES.items())),
        )
        serialized_checks = json.dumps(manifest["checks"], sort_keys=True).lower()
        for forbidden in ("recall@", "ndcg", "mrr", "latency"):
            self.assertNotIn(forbidden, serialized_checks)


if __name__ == "__main__":
    unittest.main()
