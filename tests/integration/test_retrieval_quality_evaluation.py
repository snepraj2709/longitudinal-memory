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

from retrieval.quality_runtime import (
    OUTPUT_ROOT,
    QualityRuntimeError,
    execute_quality_runtime,
    load_quality_runtime_config,
    verify_quality_runtime_checkpoint,
)
from retrieval.quality_evaluation import (
    DATA_MANIFEST,
    RESULT_ROOT,
    execute_quality_evaluation,
    load_relevance_cases,
    verify_quality_release,
)
from retrieval.query_contracts import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
CHECKED = ROOT / OUTPUT_ROOT


class RetrievalQualityRuntimeStaticTests(unittest.TestCase):
    def test_config_manifest_and_runtime_import_boundary_are_frozen(self) -> None:
        config = load_quality_runtime_config(ROOT / "configs/retrieval/quality_evaluation_v1.json")
        self.assertEqual(config.latency_sample_count, 240)
        self.assertEqual(config.baselines, ("B2", "B3", "B4"))
        manifest = json.loads(
            (
                ROOT
                / "data/retrieval/retrieval-quality-development-v1/runtime/manifest.json"
            ).read_text()
        )
        self.assertEqual(manifest["starting_commit"], "d849ea1140f97066edb408acd8704268655c7abe")
        self.assertEqual(manifest["latency_sample_count"], 240)
        preflight = json.loads((CHECKED / "checkpoint_preflight.json").read_text())
        self.assertEqual(preflight["absent_paths"], manifest["required_absent_paths"])
        self.assertTrue(preflight["all_required_paths_absent"])
        source = (ROOT / "src/retrieval/quality_runtime.py").read_text().lower()
        for forbidden in (
            "from .quality_contracts",
            "from .quality_evaluation",
            "reference.jsonl",
            "/gold/",
            "oracle/",
            "review_queue",
            "user_003",
        ):
            self.assertNotIn(forbidden, source)

    def test_nonempty_output_is_refused_before_database_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            output.mkdir()
            (output / "existing").write_text("keep")
            with self.assertRaisesRegex(QualityRuntimeError, "must be empty"):
                execute_quality_runtime(lambda: None, output, repo_root=ROOT)


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for quality runtime tests",
)
class RetrievalQualityRuntimeIntegrationTests(unittest.TestCase):
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

    def test_runtime_freezes_240_samples_with_exact_result_digests_and_read_trap(self) -> None:
        original_open = Path.open
        original_exists = Path.exists
        forbidden_opened: list[str] = []
        absent = set(
            json.loads(
                (ROOT / "data/retrieval/retrieval-quality-development-v1/runtime/manifest.json").read_text()
            )["required_absent_paths"]
        )

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
                        "/gold/",
                        "reference.jsonl",
                        "oracle",
                        "review_queue",
                        "quality_contracts.py",
                        "quality_evaluation.py",
                        "retrieval-quality-development-v1/manifest.json",
                        "results/retrieval/retrieval-quality-development-v1",
                        "user_003",
                        "user_004",
                        "user_005",
                        "user_006",
                        "user_007",
                        "user_008",
                        "user_009",
                        "user_010",
                    )
                ):
                    forbidden_opened.append(relative)
                    raise AssertionError(f"forbidden runtime read: {relative}")
            return original_open(path, *args, **kwargs)

        def checkpoint_boundary_exists(path: Path) -> bool:
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return original_exists(path)
            if relative in absent:
                return False
            return original_exists(path)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            with patch.object(Path, "exists", checkpoint_boundary_exists), patch.object(Path, "open", guarded_open):
                samples = execute_quality_runtime(self._factory, output, repo_root=ROOT)
            verify_quality_runtime_checkpoint(output, repo_root=ROOT)
            checkpoint = json.loads((output / "checkpoint_manifest.json").read_text())
            preflight = json.loads((output / "checkpoint_preflight.json").read_text())
            payload = b"".join(path.read_bytes() for path in sorted(output.iterdir()))
        self.assertEqual(forbidden_opened, [])
        self.assertEqual(len(samples), 240)
        self.assertEqual(len({(item.case_id, item.baseline_id) for item in samples}), 24)
        self.assertEqual({item.trial for item in samples}, set(range(1, 11)))
        self.assertTrue(all(item.elapsed_ns > 0 for item in samples))
        expected_rows = [
            json.loads(line)
            for line in (
                ROOT / "results/retrieval/baseline-execution-development-v1/results.jsonl"
            ).read_text().splitlines()
        ]
        expected = {
            (row["case_id"], row["result"]["baseline_id"]): hashlib.sha256(
                canonical_json_bytes(row["result"])
            ).hexdigest()
            for row in expected_rows
        }
        self.assertTrue(
            all(expected[(item.case_id, item.baseline_id)] == item.result_sha256 for item in samples)
        )
        self.assertEqual(checkpoint["result_digest_match_count"], 240)
        self.assertEqual(checkpoint["failure_count"], 0)
        self.assertTrue(preflight["all_required_paths_absent"])
        self.assertFalse(preflight["relevance_opened"])
        for forbidden in (b"postgresql://", b"storage_test", b"password", b"api_key"):
            self.assertNotIn(forbidden, payload.lower())

    def test_checked_runtime_checkpoint_self_verifies(self) -> None:
        verify_quality_runtime_checkpoint(CHECKED, repo_root=ROOT)


class RetrievalQualityScorerIntegrationTests(unittest.TestCase):
    def test_scorer_runs_twice_byte_identically_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            execute_quality_evaluation(first, repo_root=ROOT)
            execute_quality_evaluation(second, repo_root=ROOT)
            first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
            second_bytes = {path.name: path.read_bytes() for path in second.iterdir()}
            self.assertEqual(first_bytes, second_bytes)
            with self.assertRaisesRegex(Exception, "must be empty"):
                execute_quality_evaluation(first, repo_root=ROOT)

    def test_gold_opens_only_after_runtime_checkpoint_verification(self) -> None:
        order: list[str] = []
        from retrieval import quality_evaluation as module
        original_verify = module.verify_quality_runtime_checkpoint
        original_load = module.load_relevance_cases

        def verified(*args, **kwargs):
            order.append("checkpoint")
            return original_verify(*args, **kwargs)

        def loaded(*args, **kwargs):
            order.append("gold")
            return original_load(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(module, "verify_quality_runtime_checkpoint", verified), patch.object(module, "load_relevance_cases", loaded):
            execute_quality_evaluation(Path(directory) / "result", repo_root=ROOT)
        self.assertEqual(order, ["checkpoint", "gold", "checkpoint", "gold"])

    def test_scorer_has_no_database_search_or_prohibited_runtime_import(self) -> None:
        source = (ROOT / "src/retrieval/quality_evaluation.py").read_text().lower()
        for forbidden in (
            "psycopg", "retrievalsearchrepository", "execute_index_evaluation",
            "oracle", "review_queue", "user_003", "openai_api_key",
        ):
            self.assertNotIn(forbidden, source)

    def test_checked_gold_and_review_cover_the_exact_candidate_matrix(self) -> None:
        cases = load_relevance_cases(ROOT / "data/retrieval/retrieval-quality-development-v1/gold/relevance.jsonl")
        review = json.loads((ROOT / "data/retrieval/retrieval-quality-development-v1/gold/review.json").read_text())
        self.assertEqual(sum(len(case.annotations) for case in cases), 200)
        self.assertEqual(review["reviewed_annotation_count"], 200)
        self.assertFalse(review["rankings_used_during_annotation"])
        self.assertTrue(review["prior_ranked_result_exposure_possible"])

    def test_checked_final_release_self_verifies_and_has_required_metrics(self) -> None:
        verify_quality_release(ROOT / RESULT_ROOT, repo_root=ROOT)
        scores = json.loads((ROOT / RESULT_ROOT / "scorecard.json").read_text())
        self.assertEqual((scores["query_count"], scores["result_count"]), (8, 24))
        overall = [row for row in scores["quality_rows"] if row["slice_dimension"] == "overall"]
        self.assertEqual({row["baseline_id"] for row in overall}, {"B2", "B3", "B4"})
        for row in overall:
            self.assertEqual(
                set(row["metrics"]),
                {"recall_at_5", "recall_at_10", "ndcg_at_10", "mrr", "relevant_session_recall", "stale_memory_rate"},
            )

    def test_manifest_binds_checkpoint_gold_and_zero_model_usage(self) -> None:
        dataset = json.loads((ROOT / DATA_MANIFEST).read_text())
        result = json.loads((ROOT / RESULT_ROOT / "manifest.json").read_text())
        self.assertEqual(result["dataset_manifest_sha256"], hashlib.sha256((ROOT / DATA_MANIFEST).read_bytes()).hexdigest())
        self.assertEqual(result["inputs"], dataset["inputs"])
        self.assertEqual(result["counts"], {"queries": 8, "results": 24, "annotations": 200, "latency_samples": 240, "failures": 0})
        self.assertEqual(result["model_usage"]["provider_requests"], 0)
        self.assertIsNone(result["composite_score"])

    def test_release_verifier_rejects_bound_input_drift(self) -> None:
        from retrieval import quality_evaluation as module

        original_sha256 = module._sha256

        def drifted_sha256(path: Path) -> str:
            if Path(path).resolve() == (ROOT / module.QUERIES_PATH).resolve():
                return "0" * 64
            return original_sha256(path)

        with patch.object(module, "_sha256", drifted_sha256), self.assertRaisesRegex(
            Exception, "input hash"
        ):
            verify_quality_release(ROOT / RESULT_ROOT, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
