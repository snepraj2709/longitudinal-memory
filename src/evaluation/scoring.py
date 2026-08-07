"""Deterministic scoring for completed B1 prediction generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .prediction import ALLOWED_PREDICTION_STATUSES, validate_prediction


class ScoringDataError(ValueError):
    """Raised when evaluation-side inputs cannot be scored safely."""


def score_b1_results(
    questions_path: str | Path,
    gold_path: str | Path,
    attempts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Score structural, abstention, and evidence-reference behavior.

    This function is the only B1 path that reads gold answers. It does not
    assign a semantic answer score.
    """

    question_records = _load_jsonl_objects(Path(questions_path), "questions")
    gold_records = _load_jsonl_objects(Path(gold_path), "gold answers")
    question_ids = _ordered_case_ids(question_records, "questions")
    gold_ids = _ordered_case_ids(gold_records, "gold answers")
    attempt_ids = [_required_string(item, "case_id", "attempt") for item in attempts]

    if len(question_records) != 25:
        raise ScoringDataError(
            f"questions must contain exactly 25 cases, got {len(question_records)}"
        )
    if question_ids != gold_ids:
        raise ScoringDataError("gold answers must match question case IDs and order")
    if question_ids != attempt_ids:
        raise ScoringDataError("attempts must match question case IDs and order")

    gold_by_case = {record["case_id"]: record for record in gold_records}
    capability_by_case = {
        record["case_id"]: _required_string(record, "capability", "question")
        for record in question_records
    }
    valid_attempts = [item for item in attempts if item.get("passed") is True]
    failed_attempts = [item for item in attempts if item.get("passed") is not True]

    valid_predictions: dict[str, object] = {}
    for item in valid_attempts:
        case_id = _required_string(item, "case_id", "attempt")
        prediction = validate_prediction(item.get("prediction"))
        if prediction.case_id != case_id:
            raise ScoringDataError(
                f"attempt {case_id!r} contains a prediction for {prediction.case_id!r}"
            )
        valid_predictions[case_id] = prediction

    status_counts = {status: 0 for status in sorted(ALLOWED_PREDICTION_STATUSES)}
    for prediction in valid_predictions.values():
        status_counts[prediction.status] += 1

    total_cases = len(question_records)
    valid_count = len(valid_predictions)
    json_valid_count = sum(item.get("valid_json") is True for item in attempts)
    contract_valid_count = sum(item.get("valid_contract") is True for item in attempts)
    citation_valid_count = sum(item.get("exact_evidence") is True for item in attempts)
    answered_count = sum(
        prediction.status != "abstained" for prediction in valid_predictions.values()
    )

    abstention = _abstention_metrics(valid_predictions, gold_by_case)
    evidence = _evidence_metrics(valid_predictions, gold_by_case)
    capabilities = sorted(set(capability_by_case.values()))
    results_by_capability = {
        capability: _score_capability(
            capability,
            question_ids,
            capability_by_case,
            valid_predictions,
            gold_by_case,
        )
        for capability in capabilities
    }

    return {
        "total_cases": total_cases,
        "valid_prediction_count": valid_count,
        "failed_case_count": len(failed_attempts),
        "json_valid_rate": _rate(json_valid_count, total_cases),
        "prediction_contract_valid_rate": _rate(contract_valid_count, total_cases),
        "exact_citation_valid_rate": _rate(citation_valid_count, total_cases),
        "answered_coverage": _rate(answered_count, valid_count),
        "status_counts": status_counts,
        "abstention_precision": abstention["precision"],
        "abstention_recall": abstention["recall"],
        "abstention_accuracy": abstention["accuracy"],
        "abstention_confusion": abstention["confusion"],
        "evidence_reference_precision": evidence["precision"],
        "evidence_reference_recall": evidence["recall"],
        "evidence_reference_counts": evidence["counts"],
        "results_by_capability": results_by_capability,
        "cases_requiring_manual_semantic_review": [
            case_id for case_id in question_ids if case_id in valid_predictions
        ],
        "semantic_answer_score": None,
        "semantic_answer_score_reason": (
            "No formal semantic answer grader is implemented. Review the listed "
            "cases manually."
        ),
        "execution_failure_case_ids": [
            _required_string(item, "case_id", "attempt") for item in failed_attempts
        ],
        "metric_scope": (
            "Abstention and evidence-reference metrics use validated predictions "
            "only. Execution failures are reported separately."
        ),
    }


def _score_capability(
    capability: str,
    ordered_case_ids: Sequence[str],
    capability_by_case: Mapping[str, str],
    valid_predictions: Mapping[str, object],
    gold_by_case: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    case_ids = [
        case_id
        for case_id in ordered_case_ids
        if capability_by_case[case_id] == capability
    ]
    capability_predictions = {
        case_id: valid_predictions[case_id]
        for case_id in case_ids
        if case_id in valid_predictions
    }
    statuses = {status: 0 for status in sorted(ALLOWED_PREDICTION_STATUSES)}
    for prediction in capability_predictions.values():
        statuses[prediction.status] += 1
    abstention = _abstention_metrics(capability_predictions, gold_by_case)
    evidence = _evidence_metrics(capability_predictions, gold_by_case)
    return {
        "case_ids": case_ids,
        "total_cases": len(case_ids),
        "valid_prediction_count": len(capability_predictions),
        "failed_case_count": len(case_ids) - len(capability_predictions),
        "status_counts": statuses,
        "abstention_accuracy": abstention["accuracy"],
        "evidence_reference_precision": evidence["precision"],
        "evidence_reference_recall": evidence["recall"],
    }


def _abstention_metrics(
    predictions: Mapping[str, object],
    gold_by_case: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    true_positive = false_positive = false_negative = true_negative = 0
    for case_id, prediction in predictions.items():
        gold_abstain = gold_by_case[case_id].get("should_abstain")
        if not isinstance(gold_abstain, bool):
            raise ScoringDataError(
                f"gold answer {case_id!r}.should_abstain must be a boolean"
            )
        predicted_abstain = prediction.status == "abstained"
        if predicted_abstain and gold_abstain:
            true_positive += 1
        elif predicted_abstain:
            false_positive += 1
        elif gold_abstain:
            false_negative += 1
        else:
            true_negative += 1

    evaluated = len(predictions)
    return {
        "precision": _rate(true_positive, true_positive + false_positive),
        "recall": _rate(true_positive, true_positive + false_negative),
        "accuracy": _rate(true_positive + true_negative, evaluated),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "evaluated_valid_predictions": evaluated,
        },
    }


def _evidence_metrics(
    predictions: Mapping[str, object],
    gold_by_case: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    true_positive = predicted_total = gold_total = 0
    for case_id, prediction in predictions.items():
        predicted = {
            (item.source_id, item.message_id) for item in prediction.evidence
        }
        gold = _gold_evidence_references(gold_by_case[case_id], case_id)
        true_positive += len(predicted & gold)
        predicted_total += len(predicted)
        gold_total += len(gold)
    return {
        "precision": _rate(true_positive, predicted_total),
        "recall": _rate(true_positive, gold_total),
        "counts": {
            "matching_references": true_positive,
            "predicted_references": predicted_total,
            "gold_references": gold_total,
            "evaluated_valid_predictions": len(predictions),
        },
    }


def _gold_evidence_references(
    gold: Mapping[str, object], case_id: str
) -> set[tuple[str, str | None]]:
    evidence = gold.get("evidence")
    if not isinstance(evidence, list):
        raise ScoringDataError(f"gold answer {case_id!r}.evidence must be a list")
    references: set[tuple[str, str | None]] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ScoringDataError(
                f"gold answer {case_id!r}.evidence[{index}] must be an object"
            )
        source_id = _required_string(
            item, "source_id", f"gold answer {case_id!r}.evidence[{index}]"
        )
        message_ids = item.get("message_ids")
        if not isinstance(message_ids, list):
            raise ScoringDataError(
                f"gold answer {case_id!r}.evidence[{index}].message_ids must be a list"
            )
        if not message_ids:
            references.add((source_id, None))
        for message_index, message_id in enumerate(message_ids):
            if not isinstance(message_id, str) or not message_id.strip():
                raise ScoringDataError(
                    f"gold answer {case_id!r}.evidence[{index}].message_ids"
                    f"[{message_index}] must be a non-empty string"
                )
            references.add((source_id, message_id))
    return references


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else round(numerator / denominator, 6),
    }


def _ordered_case_ids(
    records: Sequence[Mapping[str, object]], description: str
) -> list[str]:
    case_ids = [
        _required_string(record, "case_id", description) for record in records
    ]
    if len(case_ids) != len(set(case_ids)):
        raise ScoringDataError(f"{description} contain duplicate case IDs")
    return case_ids


def _required_string(
    record: Mapping[str, object], field: str, location: str
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScoringDataError(f"{location}.{field} must be a non-empty string")
    return value


def _load_jsonl_objects(path: Path, description: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScoringDataError(f"could not read {description} {path}: {error}") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScoringDataError(
                f"{description} {path}:{line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ScoringDataError(
                f"{description} {path}:{line_number} must be an object"
            )
        records.append(record)
    return records
