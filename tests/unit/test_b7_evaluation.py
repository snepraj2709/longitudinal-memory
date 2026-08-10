from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import unittest

from abstention.b7_evaluation import (
    _metric,
    _scorecard,
    load_b7_evaluation_config,
)
from abstention.b7_evaluation_contracts import (
    CONFIG_VERSION,
    DATASET_VERSION,
    EVALUATION_VERSION,
    FINAL_RELEASE_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    B7EvaluationError,
    B7EvaluationFailure,
    B7EvaluationReference,
    B7PerCase,
    MetricValue,
    canonical_json_bytes,
    per_case_from_mapping,
    reference_from_mapping,
    stable_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


class B7EvaluationUnitTests(unittest.TestCase):
    def _reference(self, decision: str = "answerable") -> B7EvaluationReference:
        fields = {
            "evaluation_version": EVALUATION_VERSION,
            "case_id": "case_001",
            "user_id": "user_001",
            "expected_decision": decision,
            "review_status": "implementation_reviewed",
        }
        return B7EvaluationReference(reference_id=stable_sha256(fields), **fields)

    def _per_case(self, baseline: str, decision: str = "answerable", status: str = "abstained") -> B7PerCase:
        answered = status == "answered"
        abstained = status == "abstained"
        fields = {
            "evaluation_version": EVALUATION_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config_version": CONFIG_VERSION,
            "scorer_version": SCORER_VERSION,
            "dataset_version": DATASET_VERSION,
            "final_release_version": FINAL_RELEASE_VERSION,
            "case_id": "case_001",
            "user_id": "user_001",
            "baseline_id": baseline,
            "pair_id": "1" * 64,
            "pair_sha256": "2" * 64,
            "shared_identity_sha256": "7" * 64,
            "prediction_id": ("3" if baseline == "B6" else "4") * 64,
            "prediction_sha256": ("5" if baseline == "B6" else "6") * 64,
            "expected_decision": decision,
            "predicted_status": status,
            "predicted_action": "answered" if answered else "abstained",
            "answered": answered,
            "abstained": abstained,
            "answer_correct": (decision == "answerable") if answered else None,
            "false_answer": answered and decision != "answerable",
            "unnecessary_abstention": abstained and decision == "answerable",
        }
        return B7PerCase(per_case_id=stable_sha256(fields), **fields)

    def test_config_freezes_prerequisite_model_and_metric_policy(self) -> None:
        config, digest = load_b7_evaluation_config(repo_root=ROOT)
        self.assertEqual(config["baseline_order"], ["B6", "B7"])
        self.assertEqual(config["prerequisite_checkpoint_sha256"], "9902507e5518b2b16fee35ad76be62ef57201bb9271b3de251e2b2063c239186")
        self.assertEqual(config["configured_future_answer_model"], "gpt-4.1-2025-04-14")
        self.assertFalse(config["provider_execution_enabled"])
        self.assertEqual(len(digest), 64)

    def test_reference_and_per_case_ids_are_canonical_and_strict(self) -> None:
        reference = self._reference()
        self.assertEqual(reference_from_mapping(json.loads(canonical_json_bytes(reference))), reference)
        row = self._per_case("B6")
        self.assertEqual(per_case_from_mapping(json.loads(canonical_json_bytes(row))), row)
        changed = json.loads(canonical_json_bytes(row))
        changed["unknown"] = True
        with self.assertRaisesRegex(B7EvaluationError, "fields changed"):
            per_case_from_mapping(changed)
        changed.pop("unknown")
        changed["prediction_sha256"] = "0" * 64
        with self.assertRaisesRegex(B7EvaluationError, "per case id changed"):
            per_case_from_mapping(changed)

    def test_cross_user_duplicate_and_invalid_outcome_are_rejected(self) -> None:
        fields = asdict(self._reference())
        fields.pop("reference_id")
        fields["user_id"] = "user_003"
        with self.assertRaisesRegex(B7EvaluationError, "outside development"):
            B7EvaluationReference(reference_id=stable_sha256(fields), **fields)
        row = asdict(self._per_case("B6"))
        row.pop("per_case_id")
        row["false_answer"] = True
        with self.assertRaisesRegex(B7EvaluationError, "false-answer"):
            B7PerCase(per_case_id=stable_sha256(row), **row)
        row = asdict(self._per_case("B6"))
        row.pop("per_case_id")
        row["predicted_action"] = "answered"
        with self.assertRaisesRegex(B7EvaluationError, "status and action"):
            B7PerCase(per_case_id=stable_sha256(row), **row)

    def test_metric_nulls_are_not_coerced_to_zero(self) -> None:
        metric = _metric("baseline", "B6", "answer_accuracy", 0, 0, "no_answered_cases")
        self.assertIsNone(metric.value)
        self.assertEqual(metric.null_reason, "no_answered_cases")
        with self.assertRaisesRegex(B7EvaluationError, "null metric"):
            MetricValue("baseline", "B6", "answer_accuracy", 0, 0, "0.000000", None)
        with self.assertRaisesRegex(B7EvaluationError, "non-null metric"):
            MetricValue("baseline", "B6", "coverage", 0, 4, "0", None)
        with self.assertRaisesRegex(B7EvaluationError, "does not match"):
            MetricValue("baseline", "B6", "coverage", 0, 4, "1.000000", None)
        with self.assertRaisesRegex(B7EvaluationError, "counts"):
            MetricValue("baseline", "B6", "coverage", True, 4, "0.250000", None)

    def test_exact_scorecard_math_and_delta_order(self) -> None:
        cases = []
        for index, decision in enumerate(("answerable", "answerable", "answerable", "abstain")):
            for baseline in ("B6", "B7"):
                row = self._per_case(baseline, decision)
                values = asdict(row)
                values.pop("per_case_id")
                values["case_id"] = f"case_{index:03d}"
                values["pair_id"] = str(index + 1) * 64
                values["prediction_id"] = (str(index + 2) if baseline == "B6" else str(index + 3)) * 64
                cases.append(B7PerCase(per_case_id=stable_sha256(values), **values))
        scorecard = _scorecard(tuple(cases))
        by_key = {(item.baseline_id, item.metric_name): item for item in scorecard.metrics}
        self.assertEqual(by_key[("B6", "abstention_precision")].value, "0.250000")
        self.assertEqual(by_key[("B6", "abstention_recall")].value, "1.000000")
        self.assertEqual(by_key[("B6", "coverage")].value, "0.000000")
        self.assertEqual(by_key[("B6", "unnecessary_abstention_rate")].value, "1.000000")
        self.assertIsNone(by_key[(None, "answer_accuracy_delta")].value)
        self.assertEqual(by_key[(None, "gate_changed_output_count")].value, "0.000000")

    def test_json_safety_and_sanitized_failures(self) -> None:
        with self.assertRaisesRegex(B7EvaluationError, "non-finite"):
            canonical_json_bytes({"unsafe": math.nan})
        fields = {
            "case_id": "case_001", "user_id": "user_001", "baseline_id": "B6",
            "code": "pair_changed", "location": "b7_evaluation",
        }
        failure = B7EvaluationFailure(failure_id=stable_sha256(fields), **fields)
        self.assertEqual(len(failure.failure_id), 64)
        with self.assertRaisesRegex(B7EvaluationError, "sanitized"):
            bad = {**fields, "code": "contains query text"}
            B7EvaluationFailure(failure_id=stable_sha256(bad), **bad)


if __name__ == "__main__":
    unittest.main()
