from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import unittest

from evaluation.comparison_freeze import CONFIG_PATH, PROMPT_PATH, _rebuild_definition
from evaluation.comparison_freeze_contracts import (
    BASELINE_ORDER,
    ComparisonFreezeError,
    baseline_from_mapping,
    canonical_json_bytes,
    definition_from_mapping,
    parse_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


class ComparisonFreezeUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definition, _ = _rebuild_definition(ROOT, lambda path: path.read_bytes())

    def test_versions_counts_and_split_are_frozen(self):
        value = self.definition
        self.assertEqual(value.comparison_version, "frozen_comparison_v1")
        self.assertEqual(tuple(item.baseline_id for item in value.baselines), BASELINE_ORDER)
        self.assertEqual(
            [(item.task, item.total_count, item.development_count, item.frozen_test_count) for item in value.tasks],
            [("qa", 500, 100, 400), ("summary", 50, 10, 40), ("interactive", 20, 4, 16)],
        )
        self.assertEqual(value.development_users, ("user_001", "user_002"))
        self.assertEqual(value.frozen_test_users, tuple(f"user_{number:03d}" for number in range(3, 11)))
        self.assertEqual(len(value.manifest_files), 27)

    def test_baseline_capability_increments_are_exact(self):
        actual = {item.baseline_id: (item.memory_semantics, item.capabilities) for item in self.definition.baselines}
        self.assertEqual(actual["B0"], ("current_query_only", ("current_query_only",)))
        self.assertEqual(actual["B1"], ("full_source_history", ("full_source_history",)))
        self.assertEqual(actual["B2"], ("top_k_atomic", ("top_k_atomic_memories",)))
        self.assertEqual(actual["B3"], ("top_k_sessions", ("top_k_session_summaries",)))
        self.assertEqual(actual["B4"][1], ("top_k_atomic_memories", "top_k_session_summaries"))
        self.assertEqual(actual["B5"][1][-1], "temporal_versioning")
        self.assertEqual(actual["B6"][1][-1], "conflict_resolution")
        self.assertEqual(actual["B7"][1][-2:], ("answerability_v1", "threshold_profile:ordinary_1_trait_2"))

    def test_prompts_are_task_specific_and_baseline_identical(self):
        self.assertEqual(tuple(item.task for item in self.definition.prompt_bindings), ("qa", "summary", "interactive"))
        self.assertEqual(len({item.prompt_sha256 for item in self.definition.prompt_bindings}), 3)
        for item in self.definition.prompt_bindings:
            self.assertEqual(item.baseline_ids, BASELINE_ORDER)
        prompts = json.loads((ROOT / "configs/evaluation/frozen_prompts_v1.json").read_text())
        for record in prompts["tasks"]:
            self.assertEqual(record["baseline_ids"], list(BASELINE_ORDER))
            instruction = record["system_instruction"].lower()
            for phrase in ("only the supplied context", "untrusted evidence", "as_of", "abstain", "exact evidence ids", "exact quote", "json only"):
                self.assertIn(phrase, instruction)

    def test_model_ordering_and_no_call_policy_are_exact(self):
        self.assertEqual(self.definition.model_policy, {
            "provider": "OpenAI", "api": "responses",
            "requested_model": "gpt-4.1-2025-04-14",
            "resolved_model": "gpt-4.1-2025-04-14",
            "provider_returned_model": None, "temperature": 0.0,
            "max_output_tokens": 1000, "store": False,
            "text_format": "json_object", "provider_execution_authorized": False,
        })
        self.assertEqual(self.definition.ordering_policy["source_order"], ["observed_at", "source_id", "message_id_null_as_empty"])
        self.assertEqual(self.definition.ordering_policy["ranked_memory_order"], ["rank", "stable_ids"])
        self.assertEqual(self.definition.provider_request_count, 0)
        self.assertFalse(self.definition.referenced_files_opened)

    def test_scorer_denominators_and_nulls_are_frozen_without_scores(self):
        groups = {group["group"]: tuple(group["metrics"]) for group in self.definition.metric_groups}
        self.assertEqual(set(groups), {
            "extraction", "temporal_b4_to_b5", "conflict_b5_to_b6", "retrieval_b2_b3_b4",
            "answer_and_evidence", "coverage_risk_b6_b7", "summary", "interactive", "operations",
        })
        self.assertIn("selective_risk", groups["coverage_risk_b6_b7"])
        self.assertIn("latency_p95_ms", groups["operations"])
        self.assertIn("no_answered_cases", self.definition.null_reasons)
        self.assertIn("no_provider_requests", self.definition.null_reasons)
        self.assertIn("prerequisite_release_missing", self.definition.null_reasons)

    def test_prerequisite_gaps_are_not_zero_results(self):
        self.assertEqual(tuple(item.baseline_id for item in self.definition.prerequisite_gaps), BASELINE_ORDER)
        self.assertTrue(all(item.status == "missing_scaled_runtime_and_prediction_release" for item in self.definition.prerequisite_gaps))
        self.assertTrue(all(item.required_in == ("step_10_2", "step_10_3") for item in self.definition.prerequisite_gaps))

    def test_strict_unknown_duplicate_and_unsafe_json_rejection(self):
        with self.assertRaises(ComparisonFreezeError):
            baseline_from_mapping({
                "baseline_id": "B0", "memory_semantics": "current_query_only",
                "capabilities": ["current_query_only"], "unknown": True,
            })
        with self.assertRaises(ComparisonFreezeError):
            baseline_from_mapping({
                "baseline_id": "B0", "memory_semantics": "current_query_only",
                "capabilities": ["current_query_only", "current_query_only"],
            })
        with self.assertRaises(ComparisonFreezeError):
            parse_json_bytes(b'{"x":NaN}', location="unit")
        with self.assertRaises(ComparisonFreezeError):
            canonical_json_bytes({"x": float("inf")})

    def test_definition_round_trip_and_tamper_rejection(self):
        raw = canonical_json_bytes(asdict(self.definition))
        rebuilt = definition_from_mapping(parse_json_bytes(raw, location="unit definition"))
        self.assertEqual(rebuilt, self.definition)
        changed = parse_json_bytes(raw, location="changed unit definition")
        changed["model_policy"] = dict(changed["model_policy"])
        changed["model_policy"]["resolved_model"] = "changed"
        with self.assertRaises(ComparisonFreezeError):
            definition_from_mapping(changed)

    def test_config_and_prompt_bytes_are_frozen(self):
        for target in (CONFIG_PATH, PROMPT_PATH):
            with self.subTest(path=target.as_posix()):
                def reader(path: Path, *, target: Path = target) -> bytes:
                    raw = path.read_bytes()
                    return raw + b" " if path == ROOT / target else raw

                with self.assertRaises(ComparisonFreezeError):
                    _rebuild_definition(ROOT, reader)


if __name__ == "__main__":
    unittest.main()
