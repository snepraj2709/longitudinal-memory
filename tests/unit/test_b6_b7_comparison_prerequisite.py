from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import math
import unittest

from abstention.comparison_contracts import (
    COMPARISON_VERSION,
    CONFIG_VERSION,
    DIFFERENCE_WHITELIST,
    RUNTIME_RELEASE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    ComparablePair,
    ComparablePrediction,
    ComparableFailure,
    ComparableRuntimeInput,
    ComparisonPrerequisiteError,
    canonical_json_bytes,
    prediction_from_mapping,
    runtime_input_from_mapping,
    stable_sha256,
)
from abstention.comparison_runtime import (
    _assert_matched,
    load_comparison_config,
)


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


class B6B7ComparisonPrerequisiteUnitTests(unittest.TestCase):
    def _input(self) -> ComparableRuntimeInput:
        fields = {
            "case_id": "case_001", "user_id": "user_001", "query_id": "query_001",
            "as_of": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "requirement_set_id": "1" * 64, "requirement_set_sha256": "2" * 64,
            "execution_plan_id": "3" * 64, "plan_id": "4" * 64,
            "snapshot_run_id": "snapshot_001", "retrieval_execution_id": "5" * 64,
            "retrieval_result_sha256": "6" * 64, "package_id": "7" * 64,
            "package_sha256": "8" * 64, "package_baseline_id": "B4",
            "accepted_index_record_ids": ("record_001",),
            "relevant_source_ids": (),
        }
        return ComparableRuntimeInput(input_id=stable_sha256(fields), **fields)

    def _prediction(self, wrapper: str, *, text_hash: str = "9" * 64) -> ComparablePrediction:
        item = self._input()
        gated = wrapper == "B7"
        fields = {
            "comparison_version": COMPARISON_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config_version": CONFIG_VERSION,
            "runtime_version": RUNTIME_VERSION,
            "runtime_release_version": RUNTIME_RELEASE_VERSION,
            "input_id": item.input_id, "case_id": item.case_id, "user_id": item.user_id,
            "query_id": item.query_id, "as_of": item.as_of,
            "requirement_set_id": item.requirement_set_id,
            "execution_plan_id": item.execution_plan_id, "plan_id": item.plan_id,
            "snapshot_run_id": item.snapshot_run_id,
            "retrieval_execution_id": item.retrieval_execution_id,
            "retrieval_result_sha256": item.retrieval_result_sha256,
            "package_id": item.package_id, "package_sha256": item.package_sha256,
            "package_baseline_id": "B4", "wrapper_baseline_id": wrapper,
            "gate_applied": gated,
            "answerability_decision_id": "b" * 64 if gated else None,
            "answerability_decision_sha256": "c" * 64 if gated else None,
            "threshold_profile": "ordinary_1_trait_2" if gated else None,
            "threshold_application_id": "d" * 64 if gated else None,
            "configured_future_answer_model": "gpt-4.1-2025-04-14",
            "prompt_version": "memory_answer_prompt_v1", "prompt_sha256": "e" * 64,
            "memory_answer_config_sha256": "f" * 64,
            "response_action": "abstained", "response_status": "abstained",
            "response_text_sha256": text_hash, "underlying_answer_id": "0" * 64,
            "generation_allowed": False, "provider_eligible": False,
            "provider_returned_model": None, "provider_request_count": 0,
            "retry_count": 0, "input_token_count": 0, "output_token_count": 0,
            "incremental_cost_usd": 0,
        }
        return ComparablePrediction(prediction_id=stable_sha256(fields), **fields)

    def test_config_freezes_real_b6_and_gated_b7_without_provider(self) -> None:
        config, _ = load_comparison_config(repo_root=ROOT)
        self.assertEqual(config["wrapper_baseline_order"], ["B6", "B7"])
        self.assertEqual(config["package_baseline_id"], "B4")
        self.assertNotIn("answerability_v1", config["b6_capabilities"])
        self.assertIn("answerability_v1", config["b7_capabilities"])
        self.assertFalse(config["provider_execution_enabled"])

    def test_runtime_input_and_prediction_ids_are_canonical(self) -> None:
        item = self._input()
        restored = runtime_input_from_mapping(__import__("json").loads(canonical_json_bytes(item)))
        self.assertEqual(restored, item)
        for wrapper in ("B6", "B7"):
            prediction = self._prediction(wrapper)
            restored_prediction = prediction_from_mapping(
                __import__("json").loads(canonical_json_bytes(prediction))
            )
            self.assertEqual(restored_prediction, prediction)

    def test_b6_has_no_gate_and_b7_has_exact_gate_profile(self) -> None:
        b6 = self._prediction("B6")
        b7 = self._prediction("B7")
        self.assertFalse(b6.gate_applied)
        self.assertIsNone(b6.answerability_decision_id)
        self.assertTrue(b7.gate_applied)
        self.assertEqual(b7.threshold_profile, "ordinary_1_trait_2")
        _assert_matched(b6, b7)

    def test_pair_whitelist_and_response_identity_are_exact(self) -> None:
        b6 = self._prediction("B6")
        b7 = self._prediction("B7")
        fields = {
            "input_id": b6.input_id, "case_id": b6.case_id, "user_id": b6.user_id,
            "b6_prediction_id": b6.prediction_id,
            "b6_prediction_sha256": __import__("hashlib").sha256(canonical_json_bytes(b6)).hexdigest(),
            "b7_prediction_id": b7.prediction_id,
            "b7_prediction_sha256": __import__("hashlib").sha256(canonical_json_bytes(b7)).hexdigest(),
            "shared_identity_sha256": b6.input_id,
            "difference_whitelist": DIFFERENCE_WHITELIST,
            "response_identity_match": True,
        }
        pair = ComparablePair(pair_id=stable_sha256(fields), **fields)
        self.assertEqual(pair.difference_whitelist, DIFFERENCE_WHITELIST)
        changed = {**fields, "shared_identity_sha256": "7" * 64}
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "shared identity"):
            ComparablePair(pair_id=stable_sha256(changed), **changed)
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "shared identity differs"):
            _assert_matched(b6, self._prediction("B7", text_hash="8" * 64))

    def test_failure_id_is_canonical_and_fields_are_sanitized(self) -> None:
        fields = {
            "case_id": "case_001", "user_id": "user_001",
            "wrapper_baseline_id": "B6", "code": "identity_changed",
            "location": "comparison_runtime",
        }
        failure = ComparableFailure(failure_id=stable_sha256(fields), **fields)
        self.assertEqual(failure.failure_id, stable_sha256(fields))
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "failure ID"):
            ComparableFailure(failure_id="1" * 64, **fields)

    def test_contracts_reject_unknown_fields_naive_time_and_unsafe_numbers(self) -> None:
        value = __import__("json").loads(canonical_json_bytes(self._input()))
        value["unknown"] = True
        with self.assertRaises(ComparisonPrerequisiteError):
            runtime_input_from_mapping(value)
        value.pop("unknown")
        value["as_of"] = "2026-01-01T00:00:00"
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "timezone-aware"):
            runtime_input_from_mapping(value)
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "non-finite"):
            canonical_json_bytes({"unsafe": math.nan})

    def test_contracts_reject_cross_user_and_baseline_relabelling(self) -> None:
        value = asdict(self._input())
        value.pop("input_id")
        value["user_id"] = "user_003"
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "outside development"):
            ComparableRuntimeInput(input_id=stable_sha256(value), **value)
        value = asdict(self._input())
        value.pop("input_id")
        value["package_baseline_id"] = "B6"
        with self.assertRaisesRegex(ComparisonPrerequisiteError, "relabelled"):
            ComparableRuntimeInput(input_id=stable_sha256(value), **value)


if __name__ == "__main__":
    unittest.main()
