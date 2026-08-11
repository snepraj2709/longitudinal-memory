from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from answering.answer_quality_evaluation import _scorecard
from answering.answer_run_contracts import (
    METRICS,
    NULL_REASONS,
    AnswerQualityFailure,
    AnswerRunError,
    MetricValue,
)
from answering.comparable_answer_run import (
    B1_CONFIG_SHA256,
    build_no_call_preflight,
    load_comparable_answer_run_config,
)


ROOT = Path(__file__).resolve().parents[2]


class AnswerQualityUnitTests(unittest.TestCase):
    def test_configuration_and_b1_comparison_settings_are_frozen(self) -> None:
        config, digest = load_comparable_answer_run_config(repo_root=ROOT)
        self.assertEqual(config.available_baselines, ("B2", "B3", "B4"))
        self.assertEqual(config.deferred_baselines, ("B5", "B6"))
        self.assertEqual(config.expected_prediction_count, 24)
        self.assertEqual(config.prompt_sha256, "69dd688430c55fc35a16369201ef8eea7d470d10f610edec254f5cbd59bf14c1")
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            B1_CONFIG_SHA256,
            "9ac8881719153f09369e195e1c81eb55cc5f2f50e591f87b4fe2664c16bf6d76",
        )
        self.assertEqual(config.generation_settings["temperature"], 0.0)
        self.assertEqual(config.generation_settings["max_output_tokens"], 1000)

    def test_configuration_rejects_unknown_and_changed_fields(self) -> None:
        value = json.loads((ROOT / "configs/answering/comparable_answer_run_v1.json").read_text())
        for key, changed in (("extra", True), ("requested_model", "other-model")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                payload = dict(value)
                payload[key] = changed
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(payload))
                with self.assertRaises(AnswerRunError):
                    load_comparable_answer_run_config(path, repo_root=ROOT)

    def test_no_call_preflight_has_exact_zero_arithmetic(self) -> None:
        value = build_no_call_preflight()
        self.assertEqual(value.package_count, 24)
        self.assertEqual(value.eligible_provider_case_count, 0)
        self.assertEqual(value.planned_requests, 0)
        self.assertEqual(value.maximum_retry_requests, 0)
        self.assertEqual(value.hard_maximum_run_cost_usd, "0")
        self.assertEqual(value.new_cumulative_ceiling_usd, "0.2314404")
        self.assertFalse(value.oracle_and_gold_transmitted)
        self.assertFalse(value.fields_transmitted)

    def test_nonzero_provider_eligibility_is_rejected(self) -> None:
        with self.assertRaisesRegex(AnswerRunError, "eligible provider"):
            build_no_call_preflight(eligible_provider_case_count=1)

    def test_metric_null_denominators_and_reasons_are_exact(self) -> None:
        for metric in METRICS:
            value = MetricValue(metric, 0, 0, None, NULL_REASONS[metric])
            self.assertIsNone(value.value)
        with self.assertRaisesRegex(AnswerRunError, "zero-denominator"):
            MetricValue(METRICS[0], 0, 0, "0.000000", None)
        with self.assertRaisesRegex(AnswerRunError, "defined metric"):
            MetricValue(METRICS[0], 2, 1, "2.000000", None)

    def test_defined_metric_requires_fixed_six_place_value(self) -> None:
        self.assertEqual(MetricValue(METRICS[0], 1, 2, "0.500000", None).value, "0.500000")
        with self.assertRaisesRegex(AnswerRunError, "precision"):
            MetricValue(METRICS[0], 1, 2, "0.5", None)
        with self.assertRaisesRegex(AnswerRunError, "invalid"):
            MetricValue(METRICS[0], 1, 2, "nan00000", None)

    def test_scorecard_has_stable_35_row_order_and_b5_b6_nulls(self) -> None:
        scorecard = _scorecard()
        self.assertEqual(len(scorecard.rows), 35)
        self.assertEqual(
            [(row["baseline_id"], row["metric"]) for row in scorecard.rows],
            [(baseline, metric) for baseline in ("B2", "B3", "B4", "B5", "B6") for metric in METRICS],
        )
        deferred = [row for row in scorecard.rows if row["baseline_id"] in {"B5", "B6"}]
        self.assertTrue(all(row["case_count"] == 0 for row in deferred))
        self.assertTrue(all(row["value"].null_reason == "baseline_not_available" for row in deferred))

    def test_available_scorecard_reasons_match_metric_policy(self) -> None:
        rows = [row for row in _scorecard().rows if row["baseline_id"] == "B2"]
        self.assertEqual(
            {row["metric"]: row["value"].null_reason for row in rows},
            NULL_REASONS,
        )

    def test_scorecard_contract_rejects_changed_reason_and_order(self) -> None:
        scorecard = _scorecard()
        first = dict(scorecard.rows[0])
        first["value"] = replace(first["value"], null_reason="wrong_reason")
        with self.assertRaisesRegex(AnswerRunError, "null reason"):
            replace(scorecard, rows=(first, *scorecard.rows[1:]))
        with self.assertRaisesRegex(AnswerRunError, "ordering"):
            replace(scorecard, rows=tuple(reversed(scorecard.rows)))

    def test_failure_contract_contains_only_sanitized_identity(self) -> None:
        value = AnswerQualityFailure(
            "1" * 64, "2" * 64, "3" * 64, "safe_query", "B2",
            "checkpoint_invalid", "checkpoint",
        )
        self.assertNotIn("quote", json.dumps(value.__dict__))
        with self.assertRaisesRegex(AnswerRunError, "sanitized"):
            replace(value, code="raw source text")


if __name__ == "__main__":
    unittest.main()
