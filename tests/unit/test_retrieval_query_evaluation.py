from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from retrieval.query_contracts import EligibilityDecision, EligibilityResult
from retrieval.query_evaluation import (
    ALLOWED_USERS,
    CASE_IDS,
    DATASET_MANIFEST,
    LABELS,
    FilterDecisionBundle,
    QueryEvaluationError,
    QueryPrediction,
    QueryReference,
    _plan_expectations,
    _read_object,
    _serialize,
    _validate_dataset_manifest,
    _write_exclusive,
    load_development_reference,
    load_development_requests,
    score_query_planning,
)
from retrieval.query_planner import build_query_plan, load_query_planner_config


ROOT = Path(__file__).resolve().parents[2]


class QueryPlanningDevelopmentDataTests(unittest.TestCase):
    def test_runtime_has_24_ordered_cases_three_per_label_and_two_users(self) -> None:
        cases = load_development_requests(
            ROOT / "data/retrieval/query-planning-development-v1/requests.jsonl"
        )
        config = load_query_planner_config(ROOT / "configs/retrieval/query_planner_v1.json")
        plans = tuple(build_query_plan(item.request, config=config) for item in cases)
        self.assertEqual(tuple(item.case_id for item in cases), CASE_IDS)
        self.assertEqual({item.request.user_id for item in cases}, set(ALLOWED_USERS))
        self.assertEqual(
            {label: sum(plan.primary_label == label for plan in plans) for label in LABELS},
            {label: 3 for label in LABELS},
        )
        self.assertEqual(
            {user: sum(item.request.user_id == user for item in cases) for user in ALLOWED_USERS},
            {"user_001": 12, "user_002": 12},
        )

    def test_dataset_manifest_binds_only_runtime_files_before_reference(self) -> None:
        manifest = _read_object(ROOT / DATASET_MANIFEST)
        _validate_dataset_manifest(ROOT, manifest)
        self.assertEqual(manifest["reference_boundary"], "scorer_only_after_runtime_checkpoint")
        serialized = json.dumps(manifest["runtime_bindings"], sort_keys=True)
        for forbidden in ("gold", "oracle", "review", "test_user", "reference"):
            self.assertNotIn(forbidden, serialized)

    def test_runtime_planner_and_repository_imports_have_no_scorer_dependency(self) -> None:
        for relative in (
            "src/retrieval/query_planner.py",
            "src/retrieval/query_repository.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            for forbidden in (
                "query_evaluation",
                "reference.jsonl",
                "/gold/",
                "oracle",
                "review_queue",
                "test_user",
            ):
                self.assertNotIn(forbidden, source)

    def test_runtime_loader_rejects_order_user_and_duplicate_query_drift(self) -> None:
        source = ROOT / "data/retrieval/query-planning-development-v1/requests.jsonl"
        rows = source.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(QueryEvaluationError, "order or users"):
                load_development_requests(path)
            values = [json.loads(row) for row in rows]
            values[0]["user_id"] = "user_003"
            path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
            with self.assertRaisesRegex(QueryEvaluationError, "order or users"):
                load_development_requests(path)
            values = [json.loads(row) for row in rows]
            values[1]["query_id"] = values[0]["query_id"]
            path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
            with self.assertRaisesRegex(QueryEvaluationError, "duplicated"):
                load_development_requests(path)


class QueryPlanningScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cases = load_development_requests(
            ROOT / "data/retrieval/query-planning-development-v1/requests.jsonl"
        )
        config = load_query_planner_config(ROOT / "configs/retrieval/query_planner_v1.json")
        predictions = []
        bundles = []
        references = []
        for case in cases:
            plan = build_query_plan(case.request, config=config)
            decision = EligibilityDecision(
                f"record_{case.case_id}",
                True,
                (),
                ("candidate",),
                "standard",
                False,
            )
            result = EligibilityResult(
                plan.plan_id,
                case.request.user_id,
                case.request.index_version,
                f"run_{case.request.user_id}",
                (decision,),
            )
            bundle = FilterDecisionBundle(
                case.case_id,
                case.request.query_id,
                case.request.user_id,
                plan.plan_id,
                result.snapshot_run_id,
                result.eligible_record_ids,
                result.decisions,
            )
            predictions.append(
                QueryPrediction(case.case_id, case.request.query_id, case.request.user_id, plan)
            )
            bundles.append(bundle)
            references.append(
                QueryReference(
                    case.case_id,
                    case.request.query_id,
                    case.request.user_id,
                    plan.primary_label,
                    _plan_expectations(plan),
                    bundle.eligible_record_ids,
                    hashlib.sha256(
                        json.dumps(
                            asdict(bundle),
                            sort_keys=True,
                            separators=(",", ":"),
                            default=lambda item: item.isoformat().replace("+00:00", "Z"),
                        ).encode("utf-8")
                        + b"\n"
                    ).hexdigest(),
                    "implementation_reviewed",
                )
            )
        cls.predictions = tuple(predictions)
        cls.bundles = tuple(bundles)
        cls.references = tuple(references)
        cls.checkpoint = {"cross_user_count": 0}

    def test_perfect_score_has_explicit_denominators_and_no_composite(self) -> None:
        checks = score_query_planning(
            self.predictions,
            self.bundles,
            (),
            self.references,
            self.checkpoint,
        )
        self.assertEqual(checks.label_accuracy.numerator, 24)
        self.assertEqual(checks.plan_expectation_accuracy.denominator, 24)
        self.assertEqual(checks.eligibility_expectation_accuracy.value, 1.0)
        self.assertNotIn("composite", checks.__dataclass_fields__)

    def test_label_plan_and_filter_mismatches_remain_visible(self) -> None:
        wrong_label = replace(self.references[0], expected_label="unknown")
        references = (wrong_label, *self.references[1:])
        with self.assertRaisesRegex(QueryEvaluationError, "checks failed"):
            score_query_planning(
                self.predictions, self.bundles, (), references, self.checkpoint
            )
        wrong_plan = replace(self.references[0], expected_plan={"primary_label": "unknown"})
        with self.assertRaisesRegex(QueryEvaluationError, "checks failed"):
            score_query_planning(
                self.predictions,
                self.bundles,
                (),
                (wrong_plan, *self.references[1:]),
                self.checkpoint,
            )
        wrong_filter = replace(self.references[0], expected_filter_sha256="0" * 64)
        with self.assertRaisesRegex(QueryEvaluationError, "checks failed"):
            score_query_planning(
                self.predictions,
                self.bundles,
                (),
                (wrong_filter, *self.references[1:]),
                self.checkpoint,
            )

    def test_failures_and_leakage_cannot_be_scored_as_success(self) -> None:
        with self.assertRaisesRegex(QueryEvaluationError, "checks failed"):
            score_query_planning(
                self.predictions[:-1],
                self.bundles[:-1],
                (),
                self.references,
                self.checkpoint,
            )
        with self.assertRaisesRegex(QueryEvaluationError, "checks failed"):
            score_query_planning(
                self.predictions,
                self.bundles,
                (),
                self.references,
                {"cross_user_count": 1},
            )

    def test_duplicate_case_records_are_rejected(self) -> None:
        with self.assertRaisesRegex(QueryEvaluationError, "duplicate prediction"):
            score_query_planning(
                (*self.predictions, self.predictions[0]),
                self.bundles,
                (),
                self.references,
                self.checkpoint,
            )

    def test_serialization_is_byte_stable(self) -> None:
        self.assertEqual(_serialize(self.predictions), _serialize(self.predictions))
        self.assertEqual(_serialize(self.bundles), _serialize(self.bundles))


class QueryPlanningReferenceAndImmutabilityTests(unittest.TestCase):
    def test_reference_loader_rejects_missing_fields_and_unreviewed_rows(self) -> None:
        valid = {
            "case_id": CASE_IDS[0],
            "query_id": "query_001",
            "user_id": "user_001",
            "expected_label": "current_state",
            "expected_plan": {},
            "expected_eligible_record_ids": [],
            "expected_filter_sha256": "0" * 64,
            "review_status": "implementation_reviewed",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(json.dumps({key: value for key, value in valid.items() if key != "expected_plan"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(QueryEvaluationError, "fields"):
                load_development_reference(path)
            rows = []
            for number, case_id in enumerate(CASE_IDS, start=1):
                value = dict(valid)
                value["case_id"] = case_id
                value["query_id"] = f"query_{number:03d}"
                rows.append(value)
            rows[0]["review_status"] = "pending"
            path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")
            with self.assertRaisesRegex(QueryEvaluationError, "identity"):
                load_development_reference(path)

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            _write_exclusive(path, b"{}\n")
            with self.assertRaisesRegex(QueryEvaluationError, "already exists"):
                _write_exclusive(path, b"{}\n")


if __name__ == "__main__":
    unittest.main()
