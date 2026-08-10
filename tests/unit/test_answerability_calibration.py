from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from abstention.calibration import distinct_ids, load_threshold_config, predict_all_fixtures, predict_fixture
from abstention.calibration_contracts import (
    PROFILE_ORDER,
    SELECTED_PROFILE,
    AnswerabilityCalibrationFailure,
    CalibrationError,
    CalibrationFixture,
    canonical_json_bytes,
    fixture_from_mapping,
    load_config_mapping,
    stable_sha256,
)
from abstention.calibration_evaluation import EXPECTED_SWEEP, _metric


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/abstention/answerability-threshold-development-v1/runtime/fixtures.jsonl"


def fixtures() -> tuple[CalibrationFixture, ...]:
    return tuple(fixture_from_mapping(json.loads(line)) for line in FIXTURES.read_text().splitlines())


def invented_fixture(**changes) -> CalibrationFixture:
    h = lambda value: hashlib.sha256(value.encode()).hexdigest()
    fields = {
        "information_kind": "fact",
        "input_decision": "answerable",
        "input_reasons": (),
        "promoted_claim_count": 1,
        "required_part_count": 1,
        "supported_part_count": 1,
        "authoritative_source_ids": (h("source"),),
        "exact_evidence_path_ids": (h("path"),),
        "session_ids": (h("session"),),
        "episode_times": ("2033-01-01T00:00:00Z",),
        "closed_durative_interval": False,
        "causal_relation_verified": False,
        "unresolved_conflict_count": 0,
        "stale_evidence_count": 0,
        "semantic_guards_passed": True,
        "accepted_evidence_ids": (h("evidence"),),
        "rejected_evidence_sha256": h("rejected"),
    }
    fields.update(changes)
    return CalibrationFixture(fixture_id=stable_sha256(fields), **fields)


class AnswerabilityCalibrationContractTests(unittest.TestCase):
    def test_config_versions_profile_order_and_selected_values_are_exact(self) -> None:
        config = load_threshold_config(repo_root=ROOT)
        self.assertEqual(tuple(item.profile_id for item in config.profiles), PROFILE_ORDER)
        self.assertEqual(config.selected_profile, SELECTED_PROFILE)
        selected = config.profiles[0]
        self.assertEqual(
            (
                selected.minimum_promoted_claims,
                selected.minimum_required_part_coverage,
                selected.ordinary_minimum_distinct_authoritative_sources,
                selected.minimum_exact_evidence_paths,
                selected.stable_trait_minimum_distinct_sources,
                selected.stable_trait_minimum_distinct_sessions,
                selected.stable_trait_minimum_distinct_episode_times,
            ),
            (1, "1.000000", 1, 1, 2, 2, 2),
        )

    def test_config_rejects_unknown_fields_and_profile_reorder(self) -> None:
        value = json.loads((ROOT / "configs/abstention/answerability_thresholds_v1.json").read_text())
        with self.assertRaises(CalibrationError):
            load_config_mapping({**value, "extra": True})
        value["profiles"] = list(reversed(value["profiles"]))
        with self.assertRaisesRegex(CalibrationError, "profile order"):
            load_config_mapping(value)

    def test_fixture_ids_unknown_fields_and_json_safety_are_strict(self) -> None:
        value = json.loads(FIXTURES.read_text().splitlines()[0])
        with self.assertRaises(CalibrationError):
            fixture_from_mapping({**value, "extra": "field"})
        with self.assertRaisesRegex(CalibrationError, "fixture ID changed"):
            fixture_from_mapping({**value, "fixture_id": "0" * 64})
        with self.assertRaises(ValueError):
            canonical_json_bytes({"unsafe": float("nan")})

    def test_distinct_source_normalization_deduplicates_before_counting(self) -> None:
        one = hashlib.sha256(b"one").hexdigest()
        two = hashlib.sha256(b"two").hexdigest()
        self.assertEqual(distinct_ids((two, one, one)), tuple(sorted((one, two))))
        with self.assertRaises(CalibrationError):
            distinct_ids(("not-a-hash",))

    def test_failure_contract_is_sanitized(self) -> None:
        item = hashlib.sha256(b"item").hexdigest()
        fields = {"item_id": item, "code": "invalid_fixture", "location": "runtime"}
        failure = AnswerabilityCalibrationFailure(stable_sha256(fields), **fields)
        self.assertEqual(failure.code, "invalid_fixture")
        with self.assertRaises(CalibrationError):
            AnswerabilityCalibrationFailure(stable_sha256(fields), item, "raw query", "runtime")

    def test_metric_denominators_and_six_decimal_values_are_exact(self) -> None:
        defined = _metric("real_development", None, "overall", "all", "coverage", 0, 24)
        missing = _metric("real_development", None, "overall", "all", "selective_risk", 0, 0, "no_answered_cases")
        self.assertEqual(defined.value, "0.000000")
        self.assertEqual((missing.value, missing.null_reason), (None, "no_answered_cases"))


class AnswerabilityCalibrationPolicyTests(unittest.TestCase):
    def test_all_24_by_6_predictions_have_exact_sweep(self) -> None:
        config = load_threshold_config(repo_root=ROOT)
        predictions = predict_all_fixtures(fixtures(), config)
        self.assertEqual(len(predictions), 144)
        for profile_id, expected in EXPECTED_SWEEP.items():
            values = [item for item in predictions if item.profile_id == profile_id]
            self.assertEqual(
                tuple(sum(item.output_decision == decision for item in values) for decision in ("answerable", "abstain", "clarify")),
                expected[:3],
            )

    def test_overlay_never_promotes_abstain_or_clarify(self) -> None:
        config = load_threshold_config(repo_root=ROOT)
        for fixture in fixtures():
            if fixture.input_decision in {"abstain", "clarify"}:
                for profile in config.profiles:
                    prediction = predict_fixture(fixture, profile)
                    self.assertEqual(prediction.output_decision, fixture.input_decision)
                    self.assertEqual(prediction.output_reasons, fixture.input_reasons)
                    self.assertFalse(prediction.withheld_by_threshold)

    def test_ordinary_source_boundaries_are_one_two_and_three(self) -> None:
        config = load_threshold_config(repo_root=ROOT)
        ordinary = next(item for item in fixtures() if item.input_decision == "answerable" and item.information_kind == "fact")
        outputs = [predict_fixture(ordinary, profile) for profile in config.profiles]
        self.assertEqual([item.output_decision for item in outputs], ["answerable", "answerable", "abstain", "abstain", "abstain", "abstain"])

    def test_trait_boundaries_and_closed_interval_override_are_exact(self) -> None:
        config = load_threshold_config(repo_root=ROOT)
        traits = [item for item in fixtures() if item.input_decision == "answerable" and item.information_kind == "stable_trait"]
        two = next(item for item in traits if len(item.session_ids) == 2)
        three = next(item for item in traits if len(item.session_ids) == 3)
        closed = next(item for item in traits if item.closed_durative_interval)
        self.assertEqual([predict_fixture(two, profile).output_decision for profile in config.profiles], ["answerable", "abstain", "answerable", "abstain", "answerable", "abstain"])
        self.assertTrue(all(predict_fixture(three, profile).output_decision == "answerable" for profile in config.profiles))
        self.assertTrue(all(predict_fixture(closed, profile).closed_durative_interval_override_used for profile in config.profiles))

    def test_causal_guard_and_other_semantic_guards_are_not_tunable(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "semantic guard"):
            invented_fixture(semantic_guards_passed=False)
        with self.assertRaisesRegex(CalibrationError, "semantic guard"):
            invented_fixture(information_kind="causal", causal_relation_verified=False)
        causal = invented_fixture(information_kind="causal", causal_relation_verified=True)
        self.assertTrue(all(predict_fixture(causal, profile).output_decision == "answerable" for profile in load_threshold_config(repo_root=ROOT).profiles))

    def test_fixture_reason_grid_covers_all_step91_conditions(self) -> None:
        reasons = {reason for fixture in fixtures() for reason in fixture.input_reasons}
        self.assertEqual(len(reasons), 13)
        self.assertEqual(sum(item.input_decision == "answerable" for item in fixtures()), 8)
        self.assertEqual(sum(item.input_decision == "abstain" for item in fixtures()), 8)
        self.assertEqual(sum(item.input_decision == "clarify" for item in fixtures()), 8)


if __name__ == "__main__":
    unittest.main()
