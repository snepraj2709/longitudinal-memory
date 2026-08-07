"""Deterministic scoring for atomic extraction."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from evaluation.scoring import ScoringDataError, rate

from .contracts import AtomicClaimV1, EvidenceSpanV1
from .gold import AtomicGoldCase


def score_atomic_case(
    gold_case: AtomicGoldCase,
    predicted_claims: Sequence[AtomicClaimV1],
) -> dict[str, object]:
    """Score one case using exact, one-to-one claim-content matching."""

    gold_claims = tuple(gold_case.expected_claims)
    predictions = tuple(predicted_claims)
    matches = _match_claims(gold_claims, predictions)
    matched_prediction_indexes = {prediction_index for _, prediction_index in matches}
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(gold_claims) - true_positives
    unsupported_claim_ids = [
        claim.claim_id
        for index, claim in enumerate(predictions)
        if index not in matched_prediction_indexes
    ]

    speaker_matches = 0
    epistemic_matches = 0
    valid_time_matches = 0
    provenance_matches = 0
    for gold_index, prediction_index in matches:
        evaluated = _evaluated_field_matches(
            gold_claims[gold_index], predictions[prediction_index]
        )
        speaker_matches += evaluated[0]
        epistemic_matches += evaluated[1]
        valid_time_matches += evaluated[2]
        provenance_matches += evaluated[3]

    return {
        "case_id": gold_case.case_id,
        "gold_claim_count": len(gold_claims),
        "predicted_claim_count": len(predictions),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "claim_precision": rate(
            true_positives,
            true_positives + false_positives,
            "no predicted claims",
        ),
        "claim_recall": rate(
            true_positives,
            true_positives + false_negatives,
            "no gold claims",
        ),
        "claim_f1": rate(
            2 * true_positives,
            2 * true_positives + false_positives + false_negatives,
            "no gold or predicted claims",
        ),
        "speaker_accuracy": rate(
            speaker_matches, true_positives, "no content-matched claims"
        ),
        "epistemic_status_accuracy": rate(
            epistemic_matches, true_positives, "no content-matched claims"
        ),
        "valid_time_accuracy": rate(
            valid_time_matches, true_positives, "no content-matched claims"
        ),
        "provenance_accuracy": rate(
            provenance_matches, true_positives, "no content-matched claims"
        ),
        "unsupported_claim_count": len(unsupported_claim_ids),
        "unsupported_claim_ids": unsupported_claim_ids,
    }


def score_atomic_extraction(
    gold_cases: Sequence[AtomicGoldCase],
    predictions_by_case: Mapping[str, Sequence[AtomicClaimV1]],
) -> dict[str, object]:
    """Score every gold case in file order and aggregate exact metrics."""

    cases = tuple(gold_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ScoringDataError("gold cases contain duplicate case IDs")

    expected_ids = set(case_ids)
    prediction_ids = set(predictions_by_case)
    missing = sorted(expected_ids - prediction_ids)
    unexpected = sorted(prediction_ids - expected_ids)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing case IDs: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected case IDs: {', '.join(unexpected)}")
        raise ScoringDataError(
            "prediction case IDs do not match gold cases; " + "; ".join(details)
        )

    case_results = [
        score_atomic_case(case, predictions_by_case[case.case_id]) for case in cases
    ]
    true_positives = sum(result["true_positives"] for result in case_results)
    false_positives = sum(result["false_positives"] for result in case_results)
    false_negatives = sum(result["false_negatives"] for result in case_results)
    unsupported_case_ids = [
        result["case_id"]
        for result in case_results
        if result["unsupported_claim_count"]
    ]

    return {
        "case_results": case_results,
        "total_gold_claims": sum(result["gold_claim_count"] for result in case_results),
        "total_predicted_claims": sum(
            result["predicted_claim_count"] for result in case_results
        ),
        "micro_claim_precision": rate(
            true_positives,
            true_positives + false_positives,
            "no predicted claims across evaluated cases",
        ),
        "micro_claim_recall": rate(
            true_positives,
            true_positives + false_negatives,
            "no gold claims across evaluated cases",
        ),
        "micro_claim_f1": rate(
            2 * true_positives,
            2 * true_positives + false_positives + false_negatives,
            "no gold or predicted claims across evaluated cases",
        ),
        "speaker_accuracy": _aggregate_accuracy(case_results, "speaker_accuracy"),
        "epistemic_status_accuracy": _aggregate_accuracy(
            case_results, "epistemic_status_accuracy"
        ),
        "valid_time_accuracy": _aggregate_accuracy(case_results, "valid_time_accuracy"),
        "provenance_accuracy": _aggregate_accuracy(case_results, "provenance_accuracy"),
        "total_unsupported_claims": sum(
            result["unsupported_claim_count"] for result in case_results
        ),
        "case_ids_with_unsupported_claims": unsupported_case_ids,
    }


def _match_claims(
    gold_claims: tuple[AtomicClaimV1, ...],
    predictions: tuple[AtomicClaimV1, ...],
) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[int, str, str, int, int]] = []
    for gold_index, gold in enumerate(gold_claims):
        for prediction_index, prediction in enumerate(predictions):
            if _claim_content(gold) != _claim_content(prediction):
                continue
            matching_fields = sum(_evaluated_field_matches(gold, prediction))
            candidates.append(
                (
                    -matching_fields,
                    gold.claim_id,
                    prediction.claim_id,
                    gold_index,
                    prediction_index,
                )
            )

    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, _, gold_index, prediction_index in sorted(candidates):
        if gold_index in matched_gold or prediction_index in matched_predictions:
            continue
        matched_gold.add(gold_index)
        matched_predictions.add(prediction_index)
        matches.append((gold_index, prediction_index))
    return tuple(matches)


def _claim_content(claim: AtomicClaimV1) -> tuple[str, str, str, str]:
    return (
        claim.subject_id,
        claim.predicate,
        json.dumps(
            claim.object,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        claim.polarity,
    )


def _evaluated_field_matches(
    gold: AtomicClaimV1, prediction: AtomicClaimV1
) -> tuple[bool, bool, bool, bool]:
    return (
        gold.speaker_id == prediction.speaker_id,
        gold.epistemic_status == prediction.epistemic_status,
        (gold.valid_from, gold.valid_to)
        == (prediction.valid_from, prediction.valid_to),
        _evidence_set(gold.evidence) == _evidence_set(prediction.evidence),
    )


def _evidence_set(
    evidence: tuple[EvidenceSpanV1, ...],
) -> frozenset[tuple[str, str | None, str]]:
    return frozenset(
        (item.source_id, item.message_id, item.quote) for item in evidence
    )


def _aggregate_accuracy(
    case_results: Sequence[dict[str, object]], metric: str
) -> dict[str, object]:
    numerator = sum(result[metric]["numerator"] for result in case_results)
    denominator = sum(result[metric]["denominator"] for result in case_results)
    return rate(
        numerator,
        denominator,
        "no content-matched claims across evaluated cases",
    )
