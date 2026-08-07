from __future__ import annotations

from dataclasses import replace
import json
import unittest

from extraction.contracts import AtomicClaimV1, EvidenceSpanV1
from extraction.gold import AtomicGoldCase
from extraction.scoring import score_atomic_case, score_atomic_extraction


class AtomicExtractionScoringTests(unittest.TestCase):
    def evidence(
        self,
        quote: str = "Maya joined Infinity Learn.",
        message_id: str = "msg_001",
    ) -> EvidenceSpanV1:
        return EvidenceSpanV1("source_001", message_id, quote)

    def claim(
        self,
        claim_id: str,
        *,
        subject_id: str = "i_am_maya",
        speaker_id: str = "i_am_maya",
        predicate: str = "employer",
        value: object = "Infinity Learn",
        polarity: str = "positive",
        epistemic_status: str = "asserted",
        valid_from: str | None = "2026-05-04",
        valid_to: str | None = None,
        confidence: float = 1.0,
        evidence: tuple[EvidenceSpanV1, ...] | None = None,
    ) -> AtomicClaimV1:
        return AtomicClaimV1(
            claim_id=claim_id,
            subject_id=subject_id,
            speaker_id=speaker_id,
            predicate=predicate,
            object=value,
            polarity=polarity,
            epistemic_status=epistemic_status,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            evidence=evidence or (self.evidence(),),
        )

    def case(
        self,
        case_id: str = "case_001",
        claims: tuple[AtomicClaimV1, ...] | None = None,
    ) -> AtomicGoldCase:
        return AtomicGoldCase(
            case_id=case_id,
            source_id=f"source_{case_id}",
            expected_claims=claims or (),
        )

    def test_perfect_predictions_ignore_claim_id_and_confidence(self) -> None:
        gold_claim = self.claim("gold_claim", confidence=1.0)
        prediction = replace(gold_claim, claim_id="prediction", confidence=0.2)
        gold_case = self.case(claims=(gold_claim,))

        case_result = score_atomic_case(gold_case, [prediction])
        aggregate = score_atomic_extraction(
            [gold_case], {gold_case.case_id: [prediction]}
        )

        self.assertEqual(
            (
                case_result["true_positives"],
                case_result["false_positives"],
                case_result["false_negatives"],
            ),
            (1, 0, 0),
        )
        for metric in (
            "claim_precision",
            "claim_recall",
            "claim_f1",
            "subject_accuracy",
            "predicate_accuracy",
            "object_accuracy",
            "polarity_accuracy",
            "speaker_accuracy",
            "epistemic_status_accuracy",
            "valid_time_accuracy",
            "provenance_span_precision",
            "provenance_span_recall",
        ):
            self.assertEqual(case_result[metric]["value"], 1.0)
        self.assertEqual(case_result["unsupported_memory_rate"]["value"], 0.0)
        self.assertEqual(aggregate["micro_claim_f1"]["value"], 1.0)
        self.assertEqual(aggregate["speaker_accuracy"]["value"], 1.0)
        json.dumps(aggregate)

    def test_false_positives_false_negatives_and_unsupported_claims(self) -> None:
        first = self.claim("gold_first")
        second = self.claim(
            "gold_second", predicate="job_role", value="marketing associate"
        )
        matched = replace(first, claim_id="predicted_first")
        unsupported = self.claim(
            "unsupported_claim", predicate="home_city", value="Pune"
        )

        result = score_atomic_case(
            self.case(claims=(first, second)), [matched, unsupported]
        )

        self.assertEqual(
            (
                result["true_positives"],
                result["false_positives"],
                result["false_negatives"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(result["claim_precision"]["value"], 0.5)
        self.assertEqual(result["claim_recall"]["value"], 0.5)
        self.assertEqual(result["claim_f1"]["value"], 0.5)
        self.assertEqual(result["unsupported_claim_count"], 1)
        self.assertEqual(result["unsupported_claim_ids"], ["unsupported_claim"])
        self.assertEqual(result["unsupported_memory_rate"]["value"], 0.5)

    def test_secondary_accuracy_scores_only_content_matches(self) -> None:
        speaker_gold = self.claim("speaker_gold", predicate="speaker_test")
        status_gold = self.claim("status_gold", predicate="status_test")
        time_gold = self.claim("time_gold", predicate="time_test")
        predictions = [
            replace(speaker_gold, claim_id="speaker_pred", speaker_id="person_ankita"),
            replace(status_gold, claim_id="status_pred", epistemic_status="uncertain"),
            replace(time_gold, claim_id="time_pred", valid_from="2026-05-05"),
        ]

        result = score_atomic_case(
            self.case(claims=(speaker_gold, status_gold, time_gold)), predictions
        )

        self.assertEqual(result["true_positives"], 3)
        self.assertEqual(result["speaker_accuracy"]["value"], 0.666667)
        self.assertEqual(result["epistemic_status_accuracy"]["value"], 0.666667)
        self.assertEqual(result["valid_time_accuracy"]["value"], 0.666667)
        self.assertEqual(result["provenance_span_precision"]["value"], 1.0)
        self.assertEqual(result["provenance_span_recall"]["value"], 1.0)

    def test_provenance_scores_exact_spans_with_precision_and_recall(self) -> None:
        first_span = self.evidence("First exact quote.", "msg_001")
        second_span = self.evidence("Second exact quote.", "msg_002")
        first_gold = self.claim(
            "first_gold",
            predicate="first_fact",
            evidence=(first_span, second_span),
        )
        second_gold = self.claim(
            "second_gold",
            predicate="second_fact",
            evidence=(first_span,),
        )
        incomplete = replace(
            first_gold,
            claim_id="incomplete",
            evidence=(first_span,),
        )
        incorrect = replace(
            second_gold,
            claim_id="incorrect",
            evidence=(self.evidence("Wrong quote.", "msg_003"),),
        )

        result = score_atomic_case(
            self.case(claims=(first_gold, second_gold)), [incomplete, incorrect]
        )

        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["provenance_span_precision"]["numerator"], 1)
        self.assertEqual(result["provenance_span_precision"]["denominator"], 2)
        self.assertEqual(result["provenance_span_precision"]["value"], 0.5)
        self.assertEqual(result["provenance_span_recall"]["numerator"], 1)
        self.assertEqual(result["provenance_span_recall"]["denominator"], 3)
        self.assertEqual(result["provenance_span_recall"]["value"], 0.333333)

    def test_field_accuracy_uses_evidence_alignment_for_inexact_claims(self) -> None:
        gold = self.claim("gold")
        prediction = replace(
            gold,
            claim_id="prediction",
            subject_id="person_asha",
            predicate="job_role",
            object="marketing associate",
        )

        result = score_atomic_case(self.case(claims=(gold,)), [prediction])

        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["aligned_claim_count"], 1)
        self.assertEqual(result["subject_accuracy"]["value"], 0.0)
        self.assertEqual(result["predicate_accuracy"]["value"], 0.0)
        self.assertEqual(result["object_accuracy"]["value"], 0.0)
        self.assertEqual(result["polarity_accuracy"]["value"], 1.0)
        self.assertEqual(result["unsupported_memory_rate"]["value"], 1.0)

    def test_object_mismatch_is_not_mislabeled_as_unsupported(self) -> None:
        gold = self.claim("gold")
        prediction = replace(
            gold,
            claim_id="prediction",
            object="Infinity Learning",
        )

        result = score_atomic_case(self.case(claims=(gold,)), [prediction])

        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["object_accuracy"]["value"], 0.0)
        self.assertEqual(result["unsupported_claim_count"], 0)
        self.assertEqual(result["unsupported_memory_rate"]["value"], 0.0)

    def test_expected_no_claim_case_accepts_present_empty_predictions(self) -> None:
        no_claim_case = self.case("no_claim")

        result = score_atomic_extraction([no_claim_case], {"no_claim": []})
        case_result = result["case_results"][0]

        self.assertEqual(case_result["gold_claim_count"], 0)
        self.assertEqual(case_result["predicted_claim_count"], 0)
        self.assertIsNone(case_result["claim_precision"]["value"])
        self.assertEqual(
            case_result["claim_precision"]["null_reason"], "no predicted claims"
        )
        self.assertIsNone(case_result["claim_recall"]["value"])
        self.assertEqual(case_result["claim_recall"]["null_reason"], "no gold claims")
        self.assertIsNone(case_result["claim_f1"]["value"])
        self.assertIsNone(result["speaker_accuracy"]["value"])
        self.assertEqual(
            result["speaker_accuracy"]["null_reason"],
            "no aligned claims",
        )
        self.assertIsNone(result["unsupported_memory_rate"]["value"])

    def test_duplicate_predictions_use_one_to_one_matching(self) -> None:
        gold_a = self.claim("gold_a", speaker_id="speaker_a")
        gold_b = self.claim(
            "gold_b",
            speaker_id="speaker_b",
            epistemic_status="uncertain",
        )
        prediction_b = replace(gold_b, claim_id="prediction_b")
        prediction_a = replace(gold_a, claim_id="prediction_a")
        duplicate = replace(gold_a, claim_id="prediction_duplicate")

        result = score_atomic_case(
            self.case(claims=(gold_a, gold_b)),
            [prediction_b, prediction_a, duplicate],
        )

        self.assertEqual(
            (
                result["true_positives"],
                result["false_positives"],
                result["false_negatives"],
            ),
            (2, 1, 0),
        )
        self.assertEqual(result["speaker_accuracy"]["value"], 1.0)
        self.assertEqual(result["epistemic_status_accuracy"]["value"], 1.0)
        self.assertEqual(result["unsupported_claim_ids"], ["prediction_duplicate"])

    def test_matching_tie_uses_stable_gold_claim_id_order(self) -> None:
        gold_a = self.claim(
            "gold_a", speaker_id="speaker_a", epistemic_status="asserted"
        )
        gold_b = self.claim(
            "gold_b", speaker_id="speaker_b", epistemic_status="uncertain"
        )
        prediction = replace(
            gold_a,
            claim_id="prediction",
            speaker_id="speaker_b",
            epistemic_status="asserted",
        )

        result = score_atomic_case(self.case(claims=(gold_b, gold_a)), [prediction])

        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["speaker_accuracy"]["value"], 0.0)
        self.assertEqual(result["epistemic_status_accuracy"]["value"], 1.0)

    def test_rejects_missing_and_unexpected_case_ids(self) -> None:
        first = self.case("case_001")
        second = self.case("case_002")

        with self.assertRaisesRegex(ValueError, "missing case IDs: case_002"):
            score_atomic_extraction([first, second], {"case_001": []})
        with self.assertRaisesRegex(ValueError, "unexpected case IDs: extra"):
            score_atomic_extraction(
                [first],
                {"case_001": [], "extra": []},
            )

    def test_aggregate_preserves_gold_order_and_counts_unsupported_cases(self) -> None:
        first = self.case("case_b")
        second = self.case("case_a")
        unsupported = self.claim("unsupported")

        result = score_atomic_extraction(
            [first, second],
            {"case_a": [unsupported], "case_b": []},
        )

        self.assertEqual(
            [item["case_id"] for item in result["case_results"]],
            ["case_b", "case_a"],
        )
        self.assertEqual(result["total_gold_claims"], 0)
        self.assertEqual(result["total_predicted_claims"], 1)
        self.assertEqual(result["total_unsupported_claims"], 1)
        self.assertEqual(result["case_ids_with_unsupported_claims"], ["case_a"])
        self.assertEqual(result["unsupported_memory_rate"]["value"], 1.0)

    def test_scoring_does_not_mutate_inputs_and_repeats_deterministically(self) -> None:
        gold_claim = self.claim("gold")
        prediction = replace(gold_claim, claim_id="prediction")
        gold_cases = [self.case(claims=(gold_claim,))]
        prediction_list = [prediction]
        predictions = {"case_001": prediction_list}
        gold_before = list(gold_cases)
        predictions_before = list(prediction_list)

        first = score_atomic_extraction(gold_cases, predictions)
        second = score_atomic_extraction(gold_cases, predictions)

        self.assertEqual(first, second)
        self.assertEqual(gold_cases, gold_before)
        self.assertEqual(prediction_list, predictions_before)
        self.assertEqual(list(predictions), ["case_001"])


if __name__ == "__main__":
    unittest.main()
