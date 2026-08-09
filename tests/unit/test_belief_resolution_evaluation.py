from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from conflicts.resolution_evaluation import (
    DATASET_MANIFEST_SHA256,
    DATASET_ROOT,
    GOLD_SHA256,
    ResolutionFailure,
    ResolutionPrediction,
    ResolutionPredictionAction,
    file_sha256,
    load_resolution_dataset_runtime,
    load_resolution_gold,
    persist_outputs_before_gold,
    score_belief_resolution,
    serialize_jsonl,
)
from conflicts.resolver import POLICY_VERSION, RESOLVER_VERSION


ROOT = Path(__file__).resolve().parents[2]


class BeliefResolutionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.runtime = load_resolution_dataset_runtime(
            ROOT / DATASET_ROOT, repo_root=ROOT
        )
        cls.gold = load_resolution_gold(
            ROOT / DATASET_ROOT / cls.manifest["gold"]["path"]
        )

    def _predictions(self):
        values = []
        for case, expected in zip(self.runtime, self.gold):
            statuses = {
                "current": () if expected.expected_selected_current_claim_id is None else (expected.expected_selected_current_claim_id,),
                "historical": expected.expected_historical_claim_ids,
                "disputed": expected.expected_disputed_claim_ids,
                "superseded": expected.expected_superseded_claim_ids,
            }
            actions = tuple(
                ResolutionPredictionAction(
                    hashlib.sha256(
                        f"{case.case_id}:{action.action_order}".encode()
                    ).hexdigest(),
                    action.action_order,
                    action.claim_id,
                    "candidate",
                    action.target_status,
                    action.replacement_claim_id,
                    "reviewed_fixture",
                )
                for action in expected.expected_actions
            )
            values.append(
                ResolutionPrediction(
                    case.case_id,
                    case.user_id,
                    case.pair_id,
                    case.decision_id,
                    hashlib.sha256(f"resolution:{case.case_id}".encode()).hexdigest(),
                    RESOLVER_VERSION,
                    POLICY_VERSION,
                    hashlib.sha256(f"snapshot:{case.case_id}".encode()).hexdigest(),
                    expected.expected_outcome,
                    expected.expected_selected_current_claim_id,
                    actions,
                    statuses["current"],
                    statuses["historical"],
                    statuses["disputed"],
                    statuses["superseded"],
                    (hashlib.sha256(f"evidence:{case.case_id}".encode()).hexdigest(),),
                    expected.expected_source_ids,
                    (),
                    None,
                )
            )
        return tuple(values)

    def test_dataset_is_hash_bound_to_step52_runtime_and_predictions_only(self) -> None:
        self.assertEqual(file_sha256(ROOT / DATASET_ROOT / "manifest.json"), DATASET_MANIFEST_SHA256)
        self.assertEqual(len(self.runtime), 8)
        self.assertEqual(
            {case.user_id for case in self.runtime}, {"user_001", "user_002"}
        )
        self.assertEqual(
            self.manifest["source_boundary"],
            "step5_2_runtime_and_predictions_only_no_step5_2_or_older_gold_oracle_review_or_test_users",
        )
        bound_paths = {
            value["path"]
            for value in self.manifest.values()
            if isinstance(value, dict) and "path" in value
        }
        self.assertFalse(any("gold" in path for path in bound_paths if not path.startswith("gold/")))

    def test_runtime_load_does_not_open_or_hash_new_gold(self) -> None:
        from unittest import mock

        original = Path.read_bytes
        gold = (ROOT / DATASET_ROOT / "gold/cases.jsonl").resolve()

        def deny(path):
            if path.resolve() == gold:
                raise AssertionError("gold opened during runtime load")
            return original(path)

        with mock.patch.object(Path, "read_bytes", deny), mock.patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=lambda path, *args, **kwargs: (
                (_ for _ in ()).throw(AssertionError("gold opened during runtime load"))
                if path.resolve() == gold
                else original(path).decode(kwargs.get("encoding") or "utf-8")
            ),
        ):
            _, cases = load_resolution_dataset_runtime(ROOT / DATASET_ROOT, repo_root=ROOT)
        self.assertEqual(len(cases), 8)

    def test_predictions_are_persisted_before_gold_loader_runs(self) -> None:
        predictions = self._predictions()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            observed = []

            def loader(path):
                observed.append(
                    (
                        (output / "predictions.jsonl").is_file(),
                        (output / "failures.jsonl").is_file(),
                        len((output / "predictions.jsonl").read_text().splitlines()),
                    )
                )
                return load_resolution_gold(path)

            score = persist_outputs_before_gold(
                output,
                self.runtime,
                predictions,
                (),
                ROOT / DATASET_ROOT / "gold/cases.jsonl",
                expected_gold_sha256=GOLD_SHA256,
                runtime_resources_closed=True,
                gold_loader=loader,
            )
            self.assertEqual(observed, [(True, True, 8)])
            self.assertTrue(score.exact_match_gate_passed)

    def test_runtime_resources_must_close_before_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "resources must close"):
                persist_outputs_before_gold(
                    Path(directory) / "release",
                    self.runtime,
                    self._predictions(),
                    (),
                    ROOT / DATASET_ROOT / "gold/cases.jsonl",
                    expected_gold_sha256=GOLD_SHA256,
                    runtime_resources_closed=False,
                )

    def test_fixed_scorecard_is_exact_and_zero_supersession_is_not_perfect(self) -> None:
        score = score_belief_resolution(self.runtime, self._predictions(), (), self.gold)
        self.assertTrue(score.exact_match_gate_passed)
        for metric in (
            score.exact_outcome_accuracy,
            score.exact_action_accuracy,
            score.current_selection_accuracy,
            score.historical_preservation_accuracy,
            score.dispute_accuracy,
            score.no_change_accuracy,
            score.evidence_trace_coverage,
        ):
            self.assertEqual(metric["value"], 1.0)
        self.assertEqual(
            score.supersession_accuracy,
            {
                "status": "not_evaluated",
                "numerator": 0,
                "denominator": 0,
                "value": None,
                "reason": "no_expected_supersession",
            },
        )

    def test_failure_remains_in_all_applicable_denominators(self) -> None:
        predictions = self._predictions()
        missing = self.runtime[1]
        failure = ResolutionFailure(
            "failure_" + "a" * 64,
            missing.case_id,
            missing.user_id,
            "case_execution_failed",
            "case",
        )
        score = score_belief_resolution(
            self.runtime,
            predictions[:1] + predictions[2:],
            (failure,),
            self.gold,
        )
        self.assertEqual(score.exact_outcome_accuracy["denominator"], 8)
        self.assertEqual(score.exact_outcome_accuracy["numerator"], 7)
        self.assertEqual(score.current_selection_accuracy["denominator"], 1)
        self.assertEqual(score.current_selection_accuracy["numerator"], 0)
        self.assertFalse(score.exact_match_gate_passed)

    def test_supersession_and_directed_replacement_are_scored_exactly(self) -> None:
        gold = list(self.gold)
        base = gold[0]
        replacement = self.runtime[0].relation_case.pair.right_claim_id
        replaced = self.runtime[0].relation_case.pair.left_claim_id
        action = replace(
            base.expected_actions[0] if base.expected_actions else self.gold[5].expected_actions[0],
            action_order=1,
            claim_id=replaced,
            target_status="superseded",
            replacement_claim_id=replacement,
        )
        gold[0] = replace(
            base,
            expected_outcome="correction_resolved",
            expected_actions=(action,),
            expected_superseded_claim_ids=(replaced,),
        )
        predictions = list(self._predictions())
        predicted_action = ResolutionPredictionAction(
            "b" * 64, 1, replaced, "candidate", "superseded", replacement,
            "correction:replaced_claim",
        )
        predictions[0] = replace(
            predictions[0],
            outcome="correction_resolved",
            actions=(predicted_action,),
            superseded_claim_ids=(replaced,),
        )
        score = score_belief_resolution(self.runtime, predictions, (), gold)
        self.assertEqual(score.supersession_accuracy["value"], 1.0)
        drift = replace(predictions[0], actions=(replace(predicted_action, replacement_claim_id=None),))
        wrong = score_belief_resolution(self.runtime, (drift, *predictions[1:]), (), gold)
        self.assertEqual(wrong.exact_action_accuracy["numerator"], 7)

    def test_serialization_is_byte_stable_and_case_sorted(self) -> None:
        predictions = self._predictions()
        first = serialize_jsonl(tuple(reversed(predictions)))
        second = serialize_jsonl(predictions)
        self.assertEqual(first, second)
        self.assertEqual(first.count(b"\n"), 8)


if __name__ == "__main__":
    unittest.main()
