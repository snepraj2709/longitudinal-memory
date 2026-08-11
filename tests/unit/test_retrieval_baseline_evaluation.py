from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from retrieval.baseline_contracts import (
    BaselineRetrievalResult,
    RejectedRetrievalItem,
    RetrievedItem,
    RRFContribution,
    SearchChannelHit,
)
from retrieval.baseline_evaluation import (
    ALLOWED_USERS,
    BASELINES,
    CASE_IDS,
    DATASET_MANIFEST,
    EXPECTED_LABELS,
    PREDECESSOR_DRIFT,
    PROTECTED_HASHES,
    BaselineCaseResult,
    BaselineEvaluationError,
    BaselineExecutionFailure,
    _serialize,
    _predecessor_attestation,
    _validate_dataset_manifest,
    _write_exclusive,
    load_development_queries,
    score_baseline_execution,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_RELEASE_COMMIT = "d849ea1140f97066edb408acd8704268655c7abe"
SHA = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _item(case_number: int, baseline_id: str, kind: str, rank: int) -> RetrievedItem:
    record_id = f"record_{case_number:03d}_{baseline_id.lower()}_{kind}"
    hits = tuple(
        SearchChannelHit(
            record_id,
            kind,
            f"{kind}_{channel}",
            1,
            "1.000000000000" if channel == "lexical" else "0.500000000000",
        )
        for channel in ("lexical", "vector")
    )
    contributions = tuple(
        RRFContribution(hit.channel_name, hit.rank, "0.016393442623")
        for hit in hits
    )
    return RetrievedItem(
        record_id,
        kind,
        f"version_{case_number:03d}_{kind}" if kind == "atomic" else None,
        f"summary_{case_number:03d}" if kind == "session" else None,
        SHA,
        ("candidate",),
        (f"claim_{case_number:03d}_{kind}",),
        (f"version_{case_number:03d}_{kind}",),
        (f"source_{case_number:03d}_{kind}",),
        (f"span_{case_number:03d}_{kind}",),
        hits,
        contributions,
        (),
        "0.032786885246",
        rank,
    )


def _perfect_results() -> tuple[BaselineCaseResult, ...]:
    results = []
    for number, (case_id, label) in enumerate(zip(CASE_IDS, EXPECTED_LABELS), start=1):
        user_id = ALLOWED_USERS[(number - 1) % 2]
        for baseline_id in BASELINES:
            kinds = {"B2": ("atomic",), "B3": ("session",), "B4": ("atomic", "session")}[baseline_id]
            accepted = tuple(
                _item(number, baseline_id, kind, rank)
                for rank, kind in enumerate(kinds, start=1)
            )
            result = BaselineRetrievalResult(
                digest(f"execution:{number}:{baseline_id}"),
                baseline_id,
                f"query_{number:03d}",
                user_id,
                str(number) * 64,
                f"run_{user_id}",
                accepted,
                (
                    RejectedRetrievalItem(
                        f"prefilter_{number:03d}_{baseline_id.lower()}",
                        "pre_filter",
                        ("unknown_valid_time",),
                    ),
                    RejectedRetrievalItem(
                        f"postrank_{number:03d}_{baseline_id.lower()}",
                        "post_rank",
                        ("outside_top_k",),
                    ),
                ),
                (),
            )
            results.append(BaselineCaseResult(case_id, label, label, result))
    return tuple(results)


class BaselineDevelopmentDataTests(unittest.TestCase):
    def test_queries_have_exact_order_labels_users_and_audit_sensitivity(self) -> None:
        cases = load_development_queries(ROOT / "data/retrieval/baseline-execution-development-v1/queries.jsonl")
        self.assertEqual(tuple(item.case_id for item in cases), CASE_IDS)
        self.assertEqual(tuple(item.capability for item in cases), EXPECTED_LABELS)
        self.assertEqual(
            {user: sum(item.request.user_id == user for item in cases) for user in ALLOWED_USERS},
            {"user_001": 4, "user_002": 4},
        )
        self.assertTrue(all(item.request.allow_unclassified_sensitivity for item in cases))

    def test_manifest_binds_runtime_only_and_contains_no_scorer_path(self) -> None:
        manifest = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
        _validate_dataset_manifest(ROOT, manifest)
        bindings = json.dumps(manifest["runtime_bindings"], sort_keys=True)
        for forbidden in ("relevance", "reference", "benchmark_qa", "/gold/", "oracle", "review_queue", "test_user"):
            self.assertNotIn(forbidden, bindings)

    def test_loader_rejects_order_label_user_and_duplicate_drift(self) -> None:
        source = ROOT / "data/retrieval/baseline-execution-development-v1/queries.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
            with self.assertRaisesRegex(BaselineEvaluationError, "order"):
                load_development_queries(path)
            changed = [dict(row) for row in rows]
            changed[0]["capability"] = "unknown"
            path.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
            with self.assertRaisesRegex(BaselineEvaluationError, "label coverage"):
                load_development_queries(path)
            changed = [dict(row) for row in rows]
            changed[0]["user_id"] = "user_003"
            path.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
            with self.assertRaisesRegex(BaselineEvaluationError, "users"):
                load_development_queries(path)
            changed = [dict(row) for row in rows]
            changed[1]["query_id"] = changed[0]["query_id"]
            path.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
            with self.assertRaisesRegex(BaselineEvaluationError, "duplicated"):
                load_development_queries(path)


class BaselineStructuralScoringTests(unittest.TestCase):
    def test_perfect_structural_run_has_exact_24_results_and_64_channels(self) -> None:
        checks = score_baseline_execution(_perfect_results(), (), {})
        self.assertEqual(checks.result_count, 24)
        self.assertEqual((checks.b2_result_count, checks.b3_result_count, checks.b4_result_count), (8, 8, 8))
        self.assertEqual(checks.label_match_count, 24)
        self.assertEqual(checks.expected_channel_count, 64)
        self.assertEqual(checks.observed_channel_count, 64)
        self.assertTrue(checks.type_purity)
        self.assertTrue(checks.complete_lineage)
        self.assertNotIn("composite", checks.__dataclass_fields__)
        for forbidden in ("recall", "ndcg", "mrr", "latency"):
            self.assertFalse(any(forbidden in name.lower() for name in checks.__dataclass_fields__))

    def test_executed_empty_lexical_channel_differs_from_skipped_channel(self) -> None:
        results = []
        for case in _perfect_results():
            accepted = tuple(
                replace(
                    item,
                    component_hits=tuple(
                        hit for hit in item.component_hits if hit.channel_name.endswith("_vector")
                    ),
                    rrf_contributions=tuple(
                        value
                        for value in item.rrf_contributions
                        if value.channel_name.endswith("_vector")
                    ),
                    final_score="0.016393442623",
                )
                for item in case.result.accepted
            )
            results.append(replace(case, result=replace(case.result, accepted=accepted)))
        checks = score_baseline_execution(results, (), {})
        self.assertEqual(checks.observed_channel_count, 64)
        skipped = list(results)
        skipped[0] = replace(
            skipped[0],
            result=replace(skipped[0].result, channel_notices=("empty_fts_query",)),
        )
        with self.assertRaisesRegex(BaselineEvaluationError, "checks failed"):
            score_baseline_execution(skipped, (), {})

    def test_missing_channel_wrong_type_rank_failure_and_leakage_fail_closed(self) -> None:
        results = list(_perfect_results())
        first = results[0]
        results[0] = replace(
            first,
            result=replace(first.result, channel_notices=("empty_fts_query",)),
        )
        with self.assertRaisesRegex(BaselineEvaluationError, "checks failed"):
            score_baseline_execution(results, (), {})
        with self.assertRaisesRegex(BaselineEvaluationError, "checks failed"):
            score_baseline_execution(_perfect_results(), (), {"cross_user_count": 1})
        failure = BaselineExecutionFailure(
            "b" * 64,
            CASE_IDS[0],
            "query_001",
            "user_001",
            "B2",
            "search_failed",
            "database",
        )
        with self.assertRaisesRegex(BaselineEvaluationError, "checks failed"):
            score_baseline_execution(_perfect_results(), (failure,), {})

    def test_results_and_failures_serialize_without_raw_query_text(self) -> None:
        payload = _serialize(_perfect_results())
        self.assertEqual(payload, _serialize(_perfect_results()))
        self.assertNotIn(b"query_text", payload)
        self.assertNotIn("query_text", BaselineExecutionFailure.__dataclass_fields__)


class BaselineReleaseSafetyTests(unittest.TestCase):
    def test_predecessor_drift_and_protected_hash_audit_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            paths = {
                value["path"] for value in PREDECESSOR_DRIFT
            }.union(PROTECTED_HASHES)
            for relative in paths:
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.run(
                    ["git", "show", f"{BASELINE_RELEASE_COMMIT}:{relative}"],
                    cwd=ROOT, check=True, capture_output=True,
                ).stdout)
            attestation = _predecessor_attestation(snapshot)
        self.assertEqual(
            [value["path"] for value in attestation["predecessor_drift"]],
            [value["path"] for value in PREDECESSOR_DRIFT],
        )
        self.assertEqual(
            set(attestation["protected_hash_audit"]["runtime_verified_hashes"]),
            set(PROTECTED_HASHES),
        )
        self.assertEqual(
            attestation["protected_hash_audit"]["authorized_existing_path_count"],
            3,
        )

    def test_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            _write_exclusive(path, b"{}\n")
            with self.assertRaisesRegex(BaselineEvaluationError, "already exists"):
                _write_exclusive(path, b"{}\n")

    def test_runtime_search_imports_have_no_scorer_or_reference_path(self) -> None:
        for relative in (
            "src/retrieval/baselines.py",
            "src/retrieval/search_repository.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            for forbidden in (
                "baseline_evaluation",
                "relevance",
                "reference.jsonl",
                "benchmark_qa",
                "/gold/",
                "oracle",
                "review_queue",
                "test_user",
            ):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
