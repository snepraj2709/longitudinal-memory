"""Deterministic scoring for atomic extraction."""

from __future__ import annotations

from collections import Counter
import json
from typing import Mapping, Sequence

from evaluation.scoring import ScoringDataError, rate

from .contracts import AtomicClaimV1, EvidenceSpanV1
from .gold import AtomicGoldCase


ATOMIC_SCORING_VERSION = "atomic-scoring-v2"


def score_atomic_case(
    gold_case: AtomicGoldCase,
    predicted_claims: Sequence[AtomicClaimV1],
) -> dict[str, object]:
    """Score one case using exact claims and evidence-aligned fields."""

    gold_claims = tuple(gold_case.expected_claims)
    predictions = tuple(predicted_claims)
    exact_matches = _match_claims(gold_claims, predictions, "exact")
    aligned = _match_claims(gold_claims, predictions, "aligned")
    supported = _match_claims(gold_claims, predictions, "supported")
    matched_prediction_indexes = {
        prediction_index for _, prediction_index in supported
    }
    true_positives = len(exact_matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(gold_claims) - true_positives
    unsupported_claim_ids = [
        claim.claim_id
        for index, claim in enumerate(predictions)
        if index not in matched_prediction_indexes
    ]
    field_matches = {
        field: sum(
            _field_matches(gold_claims[gold_index], predictions[prediction_index])[field]
            for gold_index, prediction_index in aligned
        )
        for field in _SCORED_FIELDS
    }
    gold_spans = _evidence_counts(gold_claims)
    predicted_spans = _evidence_counts(predictions)
    matched_spans = sum((gold_spans & predicted_spans).values())

    result: dict[str, object] = {
        "case_id": gold_case.case_id,
        "gold_claim_count": len(gold_claims),
        "predicted_claim_count": len(predictions),
        "aligned_claim_count": len(aligned),
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
    }
    result.update(
        {
            metric: rate(field_matches[field], len(aligned), "no aligned claims")
            for field, metric in _SCORED_FIELDS.items()
        }
    )
    result.update(
        {
            "provenance_span_precision": rate(
                matched_spans, sum(predicted_spans.values()), "no predicted spans"
            ),
            "provenance_span_recall": rate(
                matched_spans, sum(gold_spans.values()), "no gold spans"
            ),
            "unsupported_memory_rate": rate(
                len(unsupported_claim_ids), len(predictions), "no predicted claims"
            ),
            "unsupported_claim_count": len(unsupported_claim_ids),
            "unsupported_claim_ids": unsupported_claim_ids,
        }
    )
    return result


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
    total_predictions = sum(
        result["predicted_claim_count"] for result in case_results
    )
    unsupported_case_ids = [
        result["case_id"]
        for result in case_results
        if result["unsupported_claim_count"]
    ]

    result: dict[str, object] = {
        "case_results": case_results,
        "total_gold_claims": sum(
            result["gold_claim_count"] for result in case_results
        ),
        "total_predicted_claims": total_predictions,
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
    }
    result.update(
        {
            metric: _aggregate_rate(case_results, metric, "no aligned claims")
            for metric in _SCORED_FIELDS.values()
        }
    )
    result.update(
        {
            "provenance_span_precision": _aggregate_rate(
                case_results,
                "provenance_span_precision",
                "no predicted spans across evaluated cases",
            ),
            "provenance_span_recall": _aggregate_rate(
                case_results,
                "provenance_span_recall",
                "no gold spans across evaluated cases",
            ),
            "unsupported_memory_rate": rate(
                sum(result["unsupported_claim_count"] for result in case_results),
                total_predictions,
                "no predicted claims across evaluated cases",
            ),
            "total_unsupported_claims": sum(
                result["unsupported_claim_count"] for result in case_results
            ),
            "case_ids_with_unsupported_claims": unsupported_case_ids,
        }
    )
    return result


_SCORED_FIELDS = {
    "subject": "subject_accuracy",
    "predicate": "predicate_accuracy",
    "object": "object_accuracy",
    "polarity": "polarity_accuracy",
    "speaker": "speaker_accuracy",
    "epistemic_status": "epistemic_status_accuracy",
    "valid_time": "valid_time_accuracy",
}


def _match_claims(
    gold_claims: tuple[AtomicClaimV1, ...],
    predictions: tuple[AtomicClaimV1, ...],
    mode: str,
) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[int, str, str, int, int]] = []
    for gold_index, gold in enumerate(gold_claims):
        for prediction_index, prediction in enumerate(predictions):
            exact = _claim_content(gold) == _claim_content(prediction)
            shared_evidence = bool(
                _evidence_references(gold.evidence)
                & _evidence_references(prediction.evidence)
            )
            if mode == "exact" and not exact:
                continue
            if mode != "exact" and not exact and not shared_evidence:
                continue
            if mode == "supported" and not exact and not (
                gold.subject_id == prediction.subject_id
                and gold.predicate == prediction.predicate
                and gold.polarity == prediction.polarity
            ):
                continue
            matching_fields = sum(_field_matches(gold, prediction).values())
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
        _json_value(claim.object),
        claim.polarity,
    )


def _field_matches(
    gold: AtomicClaimV1, prediction: AtomicClaimV1
) -> dict[str, bool]:
    return {
        "subject": gold.subject_id == prediction.subject_id,
        "predicate": gold.predicate == prediction.predicate,
        "object": _json_value(gold.object) == _json_value(prediction.object),
        "polarity": gold.polarity == prediction.polarity,
        "speaker": gold.speaker_id == prediction.speaker_id,
        "epistemic_status": gold.epistemic_status == prediction.epistemic_status,
        "valid_time": (gold.valid_from, gold.valid_to)
        == (prediction.valid_from, prediction.valid_to),
    }


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _evidence_references(
    evidence: tuple[EvidenceSpanV1, ...],
) -> frozenset[tuple[str, str | None]]:
    return frozenset((item.source_id, item.message_id) for item in evidence)


def _evidence_counts(
    claims: Sequence[AtomicClaimV1],
) -> Counter[tuple[str, str | None, str]]:
    return Counter(
        (item.source_id, item.message_id, item.quote)
        for claim in claims
        for item in claim.evidence
    )


def _aggregate_rate(
    case_results: Sequence[dict[str, object]],
    metric: str,
    null_reason: str,
) -> dict[str, object]:
    numerator = sum(result[metric]["numerator"] for result in case_results)
    denominator = sum(result[metric]["denominator"] for result in case_results)
    return rate(numerator, denominator, null_reason)
