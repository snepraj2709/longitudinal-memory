from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import conflicts.relation_evaluation as evaluation_module
from conflicts.classifier import CONFLICT_LABELS, LABEL_RELATION
from conflicts.relation_evaluation import (
    CASE_IDS,
    DATASET_MANIFEST_SHA256,
    GOLD_SHA256,
    PRESENT_GOLD_LABELS,
    RUNTIME_SHA256,
    RelationEvaluationError,
    RelationFailure,
    RelationGoldCase,
    RelationGoldRelation,
    RelationPrediction,
    RelationPredictionRelation,
    STEP52_PREDECESSOR_DRIFT_REASONS,
    _validate_predecessor_drift,
    file_sha256,
    load_relation_dataset_runtime,
    load_relation_gold,
    persist_outputs_before_gold,
    score_relation_classification,
    serialize_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/conflicts/relation-development-v1"
GOLD_PATH = DATASET_ROOT / "gold/cases.jsonl"


def _relation(
    case: object, relation_type: str, *, reverse: bool = False
) -> RelationPredictionRelation:
    source, target = case.pair.left_claim_id, case.pair.right_claim_id
    if reverse:
        source, target = target, source
    return RelationPredictionRelation(
        f"relation_{case.case_id}_{relation_type}",
        source,
        target,
        relation_type,
        1.0,
    )


def _prediction(
    case: object,
    label: str,
    relations: tuple[RelationPredictionRelation, ...] | None = None,
) -> RelationPrediction:
    expected_type = LABEL_RELATION[label]
    if relations is None:
        relations = (
            ()
            if expected_type is None
            else (_relation(case, expected_type),)
        )
    return RelationPrediction(
        case.case_id,
        case.user_id,
        case.pair.pair_id,
        "relation_classifier_v1",
        "relation_rules_v1",
        f"decision_{case.case_id}_{label}",
        "a" * 64,
        label,
        relations,
    )


class ConflictRelationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.runtime = load_relation_dataset_runtime(
            DATASET_ROOT, repo_root=ROOT
        )

    def test_dataset_is_exact_fixed_eight_pair_release(self) -> None:
        self.assertEqual(tuple(item.case_id for item in self.runtime), CASE_IDS)
        self.assertEqual(self.manifest["present_gold_labels"], PRESENT_GOLD_LABELS)
        self.assertEqual(file_sha256(DATASET_ROOT / "manifest.json"), DATASET_MANIFEST_SHA256)
        self.assertEqual(file_sha256(DATASET_ROOT / "runtime/cases.jsonl"), RUNTIME_SHA256)
        self.assertEqual(file_sha256(GOLD_PATH), GOLD_SHA256)
        self.assertTrue(
            all(item.pair.user_id == item.user_id for item in self.runtime)
        )
        runtime_text = (DATASET_ROOT / "runtime/cases.jsonl").read_text()
        self.assertNotIn("expected_label", runtime_text)
        self.assertNotIn("expected_relations", runtime_text)
        self.assertNotIn("user_003", runtime_text)
        self.assertNotIn("supplemental", runtime_text)

    def test_runtime_loader_never_opens_or_hashes_gold(self) -> None:
        opened: list[Path] = []
        hashed: list[Path] = []
        original_read = evaluation_module._read_jsonl
        original_hash = evaluation_module.file_sha256

        def tracked_read(path: str | Path) -> object:
            opened.append(Path(path).resolve())
            return original_read(path)

        def tracked_hash(path: str | Path) -> str:
            hashed.append(Path(path).resolve())
            return original_hash(path)

        with mock.patch.object(
            evaluation_module, "_read_jsonl", side_effect=tracked_read
        ), mock.patch.object(
            evaluation_module, "file_sha256", side_effect=tracked_hash
        ):
            _, cases = load_relation_dataset_runtime(DATASET_ROOT, repo_root=ROOT)
        self.assertEqual(len(cases), 8)
        self.assertNotIn(GOLD_PATH.resolve(), opened)
        self.assertNotIn(GOLD_PATH.resolve(), hashed)

    def test_predictions_and_failures_persist_and_runtime_closes_before_gold(self) -> None:
        labels = {
            "c51_u1_separate_periods": "temporal_change",
            "c51_u2_mixed_unknown": "unresolved_ambiguity",
        }
        predictions = tuple(
            _prediction(case, labels.get(case.case_id, "unrelated"))
            for case in self.runtime
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            opened: list[bool] = []

            def loader(path: str | Path) -> tuple[RelationGoldCase, ...]:
                self.assertTrue((output / "predictions.jsonl").is_file())
                self.assertTrue((output / "failures.jsonl").is_file())
                persisted = {
                    json.loads(line)["case_id"]
                    for artifact in ("predictions.jsonl", "failures.jsonl")
                    for line in (output / artifact).read_text().splitlines()
                }
                self.assertEqual(persisted, set(CASE_IDS))
                opened.append(True)
                return load_relation_gold(path)

            score = persist_outputs_before_gold(
                output,
                self.runtime,
                predictions,
                (),
                GOLD_PATH,
                expected_gold_sha256=GOLD_SHA256,
                runtime_resources_closed=True,
                gold_loader=loader,
            )
            self.assertEqual(opened, [True])
            self.assertTrue(score.exact_match_gate_passed)

        with tempfile.TemporaryDirectory() as directory:
            loader = mock.Mock()
            with self.assertRaisesRegex(RelationEvaluationError, "runtime resources"):
                persist_outputs_before_gold(
                    Path(directory) / "release",
                    self.runtime,
                    predictions,
                    (),
                    GOLD_PATH,
                    expected_gold_sha256=GOLD_SHA256,
                    runtime_resources_closed=False,
                    gold_loader=loader,
                )
            loader.assert_not_called()

    def test_all_eight_labels_and_relation_orientations_score_exactly(self) -> None:
        labels = tuple(CONFLICT_LABELS)
        predictions: list[RelationPrediction] = []
        gold: list[RelationGoldCase] = []
        for case, label in zip(self.runtime, labels, strict=True):
            relation_type = LABEL_RELATION[label]
            reverse = relation_type in {"corrects", "refines"}
            relations = (
                ()
                if relation_type is None
                else (_relation(case, relation_type, reverse=reverse),)
            )
            predictions.append(_prediction(case, label, relations))
            gold.append(
                RelationGoldCase(
                    case.case_id,
                    "relation_development_v1",
                    "development",
                    case.user_id,
                    "approved",
                    label,
                    tuple(
                        RelationGoldRelation(
                            item.source_claim_id,
                            item.target_claim_id,
                            item.relation_type,
                        )
                        for item in relations
                    ),
                )
            )
        score = score_relation_classification(
            self.runtime, tuple(predictions), (), tuple(gold)
        )
        self.assertEqual(score.overall_label_accuracy["value"], 1)
        self.assertEqual(score.exact_relation_set_accuracy["value"], 1)
        self.assertEqual(score.direction_accuracy["value"], 1)
        self.assertEqual(score.macro_f1_supported_labels["value"], 1)
        self.assertTrue(all(item["status"] == "evaluated" for item in score.per_label.values()))

    def test_failures_remain_in_denominators_and_absent_labels_are_not_zero(self) -> None:
        gold = load_relation_gold(GOLD_PATH)
        predictions = tuple(
            _prediction(case, gold[index].expected_label)
            for index, case in enumerate(self.runtime)
        )
        failure = RelationFailure(
            "failure_one", self.runtime[0].case_id, self.runtime[0].user_id,
            "case_execution_failed", "case",
        )
        score = score_relation_classification(
            self.runtime, predictions[1:], (failure,), gold
        )
        self.assertEqual(score.failure_rate["value"], 1 / 8)
        self.assertEqual(score.overall_label_accuracy["denominator"], 8)
        self.assertEqual(score.overall_label_accuracy["value"], 7 / 8)
        self.assertFalse(score.exact_match_gate_passed)
        self.assertEqual(score.per_label["hard_contradiction"]["status"], "not_evaluated")
        self.assertIsNone(score.per_label["hard_contradiction"]["f1"])
        self.assertIsNone(score.direction_accuracy["value"])
        self.assertEqual(score.direction_accuracy["reason"], "no_directed_gold_relations")

    def test_serialization_and_nonempty_output_are_deterministic(self) -> None:
        values = tuple(_prediction(case, "unrelated") for case in reversed(self.runtime))
        first = serialize_jsonl(values)
        second = serialize_jsonl(values)
        self.assertEqual(first, second)
        self.assertEqual(
            [json.loads(line)["case_id"] for line in first.splitlines()],
            sorted(CASE_IDS),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "existing").write_text("occupied")
            with self.assertRaisesRegex(RelationEvaluationError, "absent or empty"):
                persist_outputs_before_gold(
                    path,
                    self.runtime,
                    values,
                    (),
                    GOLD_PATH,
                    expected_gold_sha256=GOLD_SHA256,
                    runtime_resources_closed=True,
                )

    def test_predecessor_drift_records_reject_missing_extra_duplicate_and_old_mismatch(self) -> None:
        predecessor = {
            path: "a" * 64 for path in STEP52_PREDECESSOR_DRIFT_REASONS
        }
        current = {
            path: "b" * 64 for path in STEP52_PREDECESSOR_DRIFT_REASONS
        }
        records = [
            {
                "path": path,
                "predecessor_sha256": predecessor[path],
                "step5_2_sha256": current[path],
                "reason": STEP52_PREDECESSOR_DRIFT_REASONS[path],
            }
            for path in sorted(STEP52_PREDECESSOR_DRIFT_REASONS)
        ]
        _validate_predecessor_drift(predecessor, current, records)
        for invalid in (
            records[:-1],
            records + [records[0]],
            records + [{**records[0], "path": "unexpected"}],
            [{**records[0], "predecessor_sha256": "c" * 64}, *records[1:]],
        ):
            with self.assertRaises(RelationEvaluationError):
                _validate_predecessor_drift(predecessor, current, invalid)


if __name__ == "__main__":
    unittest.main()
