from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from abstention.interactive_contracts_v2 import (
    InteractiveMetric,
    InteractiveAnsweringError,
    InteractiveRuntimeCase,
    canonical_json_bytes,
    requirement_set_from_mapping,
)
from abstention.interactive_evaluation_v2 import score_interactive
from abstention.interactive_input_v2 import load_runtime_prefix
from abstention.interactive_runtime_v2 import (
    CONFIG_PATH,
    RUNTIME_CASES,
    RUNTIME_REQUIREMENTS,
    _load_cases,
    _load_requirements,
    derive_requirement_set,
    load_interactive_config,
    query_request_for_case,
)


ROOT = Path(__file__).resolve().parents[2]
SET_IDS = (
    "cdd44277dfbdec1ceaf05ff910d8471e7f3729e6a4b238e2080149cff24aa2c1",
    "539f38122ae8d29d7d6b47268e5fa2ac3f5584f93a47d5ec58e4a80e786a53c7",
    "125a9804d61e145e69b21e6fdcb8f63201961e12632fafa0d29444e833f7a6b0",
    "d90a50b7750d97d39b3c24a23b8302a61b75e85aa372e6c4726bbfd9747917f4",
)
PART_IDS = (
    "13ad29084e47e4833dc7d2f08879838d256a5739e4ca421a0203416c0c7e7135",
    "9cf4a239dbc89239f00a70ab74b44bf9e7aff0e0863c59f054fcbb39bd712b0b",
    "65d53e6854ad269a1f2c4e39843aec4abb71a2215815002e89be6ab9f1529136",
    "79ebbd3ae2462e83b7842bbcfaa844bc763d089e663e8887d44cb87556966d06",
    "36d1fc49c602e120809ddaa70104e1659c5bc3e82c1328ae7e8be8eb4e9d0bc1",
)


class InteractiveAnsweringV2UnitTests(unittest.TestCase):
    def test_config_and_corrected_requirement_sets_are_exact(self) -> None:
        config, digest = load_interactive_config(ROOT / CONFIG_PATH)
        self.assertEqual(digest, "69985ddf0a39518571c404bac04cc6e02095c25e32d2d73612098b31f2094ea1")
        self.assertEqual(config["predicate_registry_version"], "predicate_registry_v2")
        cases = _load_cases(ROOT / RUNTIME_CASES)
        sets = _load_requirements(ROOT / RUNTIME_REQUIREMENTS)
        self.assertEqual(tuple(item.requirement_set_id for item in sets), SET_IDS)
        self.assertEqual(
            tuple(part.requirement_id for item in sets for part in item.parts), PART_IDS,
        )
        self.assertEqual(sets, tuple(derive_requirement_set(case, repo_root=ROOT) for case in cases))

    def test_only_registered_exact_predicates_are_frozen(self) -> None:
        sets = _load_requirements(ROOT / RUNTIME_REQUIREMENTS)
        predicates = tuple(part.required_predicate for item in sets for part in item.parts)
        self.assertEqual(
            predicates,
            ("has_mentor", "leads_project", "career_goal", "job_start_date", "office_base"),
        )
        for alias in ("project_assignment", "career_direction", "start_date"):
            self.assertNotIn(alias, predicates)

    def test_requirement_set_rejects_unknown_fields_and_changed_ids(self) -> None:
        row = json.loads((ROOT / RUNTIME_REQUIREMENTS).read_text().splitlines()[0])
        row["expected_behaviour"] = "answerable"
        with self.assertRaises(InteractiveAnsweringError):
            requirement_set_from_mapping(row)
        row.pop("expected_behaviour")
        row["requirement_set_id"] = "0" * 64
        with self.assertRaises(InteractiveAnsweringError):
            requirement_set_from_mapping(row)

    def test_runtime_request_is_user_scoped_b4_and_no_model(self) -> None:
        case = _load_cases(ROOT / RUNTIME_CASES)[0]
        request = query_request_for_case(case)
        self.assertEqual(request.user_id, "user_001")
        self.assertEqual(request.enabled_record_kinds, ("atomic", "session"))
        self.assertEqual(request.entity_ids, ("user_001",))
        self.assertTrue(request.allow_unclassified_sensitivity)

    def test_case_rejects_unknown_fields_naive_time_and_test_user(self) -> None:
        case = _load_cases(ROOT / RUNTIME_CASES)[0]
        row = json.loads(canonical_json_bytes(case))
        row["oracle"] = True
        with self.assertRaises(TypeError):
            InteractiveRuntimeCase(**row)
        with self.assertRaises(InteractiveAnsweringError):
            replace(case, as_of=datetime(2026, 12, 1, 12))
        with self.assertRaises(InteractiveAnsweringError):
            replace(case, user_id="user_003")

    def test_exact_prefix_reader_is_called_four_times_and_never_five(self) -> None:
        count = 0

        def reader(stream):
            nonlocal count
            count += 1
            if count > 4:
                raise AssertionError("record five was requested")
            return stream.readline()

        rows = load_runtime_prefix(
            ROOT / "data/scaled-v1/runtime/interactive.jsonl", record_reader=reader,
        )
        self.assertEqual((len(rows), count), (4, 4))

    def test_metric_null_rules_and_fixed_rounding(self) -> None:
        self.assertIsNone(InteractiveMetric(0, 0, None, "no_factual_outputs").value)
        self.assertEqual(InteractiveMetric(1, 2, "0.500000", None).value, "0.500000")
        with self.assertRaises(InteractiveAnsweringError):
            InteractiveMetric(0, 0, "0.000000", None)

    def test_frozen_predictions_score_without_mutating_runtime(self) -> None:
        from abstention.interactive_contracts_v2 import prediction_from_mapping, reference_from_mapping

        predictions = tuple(prediction_from_mapping(json.loads(line)) for line in (
            ROOT / "results/abstention/interactive-answering-development-runtime-v2/responses.jsonl"
        ).read_text().splitlines())
        references = tuple(reference_from_mapping(json.loads(line)) for line in (
            ROOT / "data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl"
        ).read_text().splitlines())
        per_case, scorecard = score_interactive(predictions, references)
        self.assertEqual(len(per_case), 4)
        overall = {row["metric"]: row for row in scorecard["overall_metrics"]}
        self.assertEqual(overall["expected_decision_accuracy"]["value"], "0.250000")
        self.assertEqual(overall["expected_behaviour_micro_recall"]["value"], "0.100000")
        self.assertIsNone(overall["exact_evidence_tuple_precision"]["value"])


if __name__ == "__main__":
    unittest.main()
