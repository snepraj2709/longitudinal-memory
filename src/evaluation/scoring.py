"""Deterministic and manually reviewed scoring for completed B1 runs."""

from __future__ import annotations

from datetime import datetime
import re
import unicodedata
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .prediction import ALLOWED_PREDICTION_STATUSES, BaselinePrediction, validate_prediction


CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "user_modeling",
    "abstention",
)
MANUAL_JUDGMENTS = frozenset({"correct", "partially_correct", "incorrect"})
PARTIAL_CREDIT_VALUE = 0.5


class ScoringDataError(ValueError):
    """Raised when evaluation inputs cannot be scored without guessing."""


def score_b1_results(
    questions_path: str | Path,
    gold_path: str | Path,
    attempts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Keep Step 5 structural scoring separate from the full Step 6 scorer."""

    questions = _load_jsonl_objects(Path(questions_path), "questions")
    gold = _load_jsonl_objects(Path(gold_path), "gold answers")
    question_ids = [required_string(item, "case_id", "question") for item in questions]
    gold_ids = [required_string(item, "case_id", "gold answer") for item in gold]
    attempt_ids = [required_string(item, "case_id", "attempt") for item in attempts]
    if len(question_ids) != 25 or question_ids != gold_ids or question_ids != attempt_ids:
        raise ScoringDataError("questions, gold answers, and attempts must align across 25 cases")

    gold_by_case = {item["case_id"]: item for item in gold}
    capability_by_case = {item["case_id"]: item["capability"] for item in questions}
    valid: dict[str, BaselinePrediction] = {}
    for attempt in attempts:
        if attempt.get("passed") is True:
            prediction = validate_prediction(attempt.get("prediction"))
            valid[prediction.case_id] = prediction

    statuses = {status: 0 for status in sorted(ALLOWED_PREDICTION_STATUSES)}
    for prediction in valid.values():
        statuses[prediction.status] += 1
    expected_abstention = [
        required_bool(gold_by_case[case_id], "should_abstain", case_id)
        for case_id in valid
    ]
    predicted_abstention = [valid[case_id].status == "abstained" for case_id in valid]
    abstention = abstention_metrics(expected_abstention, predicted_abstention)
    evidence_comparisons = [
        compare_references(
            {(item.source_id, item.message_id) for item in valid[case_id].evidence},
            gold_evidence_references(gold_by_case[case_id], case_id),
        )
        for case_id in valid
    ]
    evidence = aggregate_reference_metrics(evidence_comparisons)

    by_capability: dict[str, object] = {}
    for capability in sorted(set(capability_by_case.values())):
        case_ids = [
            case_id for case_id in question_ids if capability_by_case[case_id] == capability
        ]
        capability_predictions = {
            case_id: valid[case_id] for case_id in case_ids if case_id in valid
        }
        capability_statuses = {
            status: sum(item.status == status for item in capability_predictions.values())
            for status in sorted(ALLOWED_PREDICTION_STATUSES)
        }
        by_capability[capability] = {
            "case_ids": case_ids,
            "total_cases": len(case_ids),
            "valid_prediction_count": len(capability_predictions),
            "failed_case_count": len(case_ids) - len(capability_predictions),
            "status_counts": capability_statuses,
        }

    total = len(questions)
    valid_count = len(valid)
    answered = sum(item.status != "abstained" for item in valid.values())
    return {
        "total_cases": total,
        "valid_prediction_count": valid_count,
        "failed_case_count": total - valid_count,
        "json_valid_rate": rate(
            sum(item.get("valid_json") is True for item in attempts), total, "no cases"
        ),
        "prediction_contract_valid_rate": rate(
            sum(item.get("valid_contract") is True for item in attempts), total, "no cases"
        ),
        "exact_citation_valid_rate": rate(
            sum(item.get("exact_evidence") is True for item in attempts), total, "no cases"
        ),
        "answered_coverage": rate(answered, valid_count, "no valid predictions"),
        "status_counts": statuses,
        "abstention_precision": abstention["precision"],
        "abstention_recall": abstention["recall"],
        "abstention_accuracy": abstention["accuracy"],
        "abstention_confusion": abstention["confusion_matrix"],
        "evidence_reference_precision": evidence["micro_precision"],
        "evidence_reference_recall": evidence["micro_recall"],
        "evidence_reference_counts": evidence["counts"],
        "results_by_capability": by_capability,
        "cases_requiring_manual_semantic_review": [
            case_id for case_id in question_ids if case_id in valid
        ],
        "semantic_answer_score": None,
        "semantic_answer_score_reason": (
            "Step 5 records structural checks only. Run the Step 6 scorer for answer scores."
        ),
        "execution_failure_case_ids": [
            required_string(item, "case_id", "attempt")
            for item in attempts
            if item.get("passed") is not True
        ],
    }


def normalize_text(value: str) -> str:
    """Normalize text conservatively for exact factual comparison."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:")


_MONTHS = {
    name.casefold(): number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_MONTH_PATTERN = "|".join(_MONTHS)
_DATE_PATTERNS = (
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    re.compile(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})[,]?\s+(\d{{4}})\b",
        re.IGNORECASE,
    ),
)


def extract_dates(value: str) -> tuple[str, ...]:
    """Return unique complete dates found in text as ISO values."""

    dates: list[str] = []
    for pattern_index, pattern in enumerate(_DATE_PATTERNS):
        for match in pattern.finditer(unicodedata.normalize("NFKC", value)):
            groups = match.groups()
            try:
                if pattern_index == 0:
                    year, month, day = map(int, groups)
                elif pattern_index == 1:
                    month = _MONTHS[groups[0].casefold()]
                    day, year = int(groups[1]), int(groups[2])
                else:
                    day = int(groups[0])
                    month = _MONTHS[groups[1].casefold()]
                    year = int(groups[2])
                parsed = datetime(year, month, day).date().isoformat()
            except (KeyError, ValueError):
                continue
            if parsed not in dates:
                dates.append(parsed)
    return tuple(dates)


def deterministic_answer_match(
    predicted: str,
    reference: str,
    acceptable_answers: Sequence[str],
) -> bool:
    """Compare exact normalized values and equivalent complete dates."""

    expected = (reference, *acceptable_answers)
    predicted_normalized = normalize_text(predicted)
    if any(predicted_normalized == normalize_text(item) for item in expected):
        return True

    predicted_dates = extract_dates(predicted)
    expected_dates = {
        date for item in expected for date in extract_dates(item)
    }
    return (
        len(predicted_dates) == 1
        and len(expected_dates) == 1
        and predicted_dates[0] in expected_dates
    )


def compare_references(
    predicted: Iterable[object], gold: Iterable[object]
) -> dict[str, object]:
    """Return the exact set overlap used for source and message evidence."""

    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positives = predicted_set & gold_set
    false_positives = predicted_set - gold_set
    false_negatives = gold_set - predicted_set
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": rate(
            len(true_positives),
            len(true_positives) + len(false_positives),
            "no predicted evidence",
        ),
        "recall": rate(
            len(true_positives),
            len(true_positives) + len(false_negatives),
            "no gold evidence",
        ),
    }


def rate(
    numerator: int | float,
    denominator: int | float,
    zero_denominator_reason: str,
) -> dict[str, object]:
    """Return a transparent ratio and an explicit zero-denominator reason."""

    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": None,
            "null_reason": zero_denominator_reason,
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6),
        "null_reason": None,
    }


def aggregate_reference_metrics(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Calculate micro and macro precision and recall."""

    true_positives = sum(len(item["true_positives"]) for item in comparisons)
    false_positives = sum(len(item["false_positives"]) for item in comparisons)
    false_negatives = sum(len(item["false_negatives"]) for item in comparisons)

    precision_values = [
        item["precision"]["value"]
        for item in comparisons
        if item["precision"]["value"] is not None
    ]
    recall_values = [
        item["recall"]["value"]
        for item in comparisons
        if item["recall"]["value"] is not None
    ]
    return {
        "counts": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "micro_precision": rate(
            true_positives,
            true_positives + false_positives,
            "no predicted evidence across evaluated cases",
        ),
        "micro_recall": rate(
            true_positives,
            true_positives + false_negatives,
            "no gold evidence across evaluated cases",
        ),
        "macro_precision": rate(
            round(sum(precision_values), 6),
            len(precision_values),
            "no case has defined precision",
        ),
        "macro_recall": rate(
            round(sum(recall_values), 6),
            len(recall_values),
            "no case has defined recall",
        ),
    }


def abstention_metrics(
    expected: Sequence[bool], predicted: Sequence[bool]
) -> dict[str, object]:
    """Calculate abstention confusion counts and metrics."""

    if len(expected) != len(predicted):
        raise ScoringDataError("expected and predicted abstention lists must align")
    true_positive = false_positive = false_negative = true_negative = 0
    for should_abstain, did_abstain in zip(expected, predicted):
        if should_abstain and did_abstain:
            true_positive += 1
        elif did_abstain:
            false_positive += 1
        elif should_abstain:
            false_negative += 1
        else:
            true_negative += 1
    total = len(expected)
    return {
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "precision": rate(
            true_positive,
            true_positive + false_positive,
            "the baseline did not abstain",
        ),
        "recall": rate(
            true_positive,
            true_positive + false_negative,
            "the evaluation set has no required abstentions",
        ),
        "accuracy": rate(
            true_positive + true_negative,
            total,
            "the evaluation set is empty",
        ),
    }


def unsupported_claim_metrics(
    total_claims: int, unsupported_claims: int
) -> dict[str, object]:
    """Calculate unsupported-claim rate after validating the counts."""

    if total_claims < 0 or unsupported_claims < 0:
        raise ScoringDataError("claim counts cannot be negative")
    if unsupported_claims > total_claims:
        raise ScoringDataError("unsupported claims cannot exceed total claims")
    return rate(
        unsupported_claims,
        total_claims,
        "no factual claims were made",
    )


def question_asks_for_current_fact(question: str) -> bool:
    """Identify questions that explicitly ask for a corrected or current fact."""

    normalized = normalize_text(question)
    markers = ("corrected", "current", "keep that date as current", "superseded")
    return any(marker in normalized for marker in markers)


def historical_question(question: str) -> bool:
    """Identify questions that explicitly ask about an earlier period."""

    normalized = normalize_text(question)
    markers = ("when did", "on what date", "at what event", "in may", "in july")
    return any(marker in normalized for marker in markers) and not question_asks_for_current_fact(
        question
    )


def outdated_fact_error(
    *,
    question: str,
    predicted_answer: str,
    reference_answer: str,
    acceptable_answers: Sequence[str],
) -> bool | None:
    """Compare an explicit current/corrected fact; skip historical questions."""

    if historical_question(question) or not question_asks_for_current_fact(question):
        return None
    return not deterministic_answer_match(
        predicted_answer, reference_answer, acceptable_answers
    )


def gold_evidence_references(
    gold: Mapping[str, object], case_id: str
) -> set[tuple[str, str | None]]:
    """Load the gold evidence pairs, including calendar ``(source_id, null)``."""

    evidence = gold.get("evidence")
    if not isinstance(evidence, list):
        raise ScoringDataError(f"gold answer {case_id}.evidence must be a list")
    references: set[tuple[str, str | None]] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ScoringDataError(
                f"gold answer {case_id}.evidence[{index}] must be an object"
            )
        source_id = required_string(
            item, "source_id", f"gold answer {case_id}.evidence[{index}]"
        )
        message_ids = item.get("message_ids")
        if not isinstance(message_ids, list):
            raise ScoringDataError(
                f"gold answer {case_id}.evidence[{index}].message_ids must be a list"
            )
        if not message_ids:
            references.add((source_id, None))
        for message_id in message_ids:
            if not isinstance(message_id, str) or not message_id.strip():
                raise ScoringDataError(
                    f"gold answer {case_id} has an invalid message ID"
                )
            references.add((source_id, message_id))
    return references


def build_manual_review_record(
    question: Mapping[str, object],
    gold: Mapping[str, object],
    prediction: BaselinePrediction,
    existing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one stable manual-review item, preserving reviewer-only fields."""

    record = {
        "case_id": prediction.case_id,
        "question": required_string(question, "question", prediction.case_id),
        "predicted_answer": prediction.answer,
        "reference_answer": required_string(
            gold, "reference_answer", f"gold answer {prediction.case_id}"
        ),
        "acceptable_answers": acceptable_answers(gold, prediction.case_id),
        "cited_evidence": [
            {
                "source_id": item.source_id,
                "message_id": item.message_id,
                "quote": item.quote,
            }
            for item in prediction.evidence
        ],
        "reviewer_judgment": None,
        "short_reason": None,
        "total_claim_count": None,
        "unsupported_claims_found": [],
        "outdated_fact_applicable": None,
        "outdated_facts_found": [],
    }
    if existing is None:
        return record

    for field in (
        "case_id",
        "question",
        "predicted_answer",
        "reference_answer",
        "acceptable_answers",
        "cited_evidence",
    ):
        if existing.get(field) != record[field]:
            raise ScoringDataError(
                f"manual review {prediction.case_id} no longer matches {field}"
            )
    for field in (
        "reviewer_judgment",
        "short_reason",
        "total_claim_count",
        "unsupported_claims_found",
        "outdated_fact_applicable",
        "outdated_facts_found",
    ):
        record[field] = existing.get(field)
    validate_manual_review(record)
    return record


def validate_manual_review(record: Mapping[str, object]) -> None:
    """Validate completed or pending reviewer fields."""

    judgment = record.get("reviewer_judgment")
    if judgment is None:
        return
    if judgment not in MANUAL_JUDGMENTS:
        raise ScoringDataError("manual reviewer judgment is invalid")
    if not isinstance(record.get("short_reason"), str) or not record["short_reason"].strip():
        raise ScoringDataError("completed manual review needs a short reason")
    total_claims = record.get("total_claim_count")
    if not isinstance(total_claims, int) or isinstance(total_claims, bool) or total_claims < 0:
        raise ScoringDataError("completed manual review needs a non-negative claim count")
    unsupported = record.get("unsupported_claims_found")
    outdated = record.get("outdated_facts_found")
    if not isinstance(unsupported, list) or not isinstance(outdated, list):
        raise ScoringDataError("manual claim and outdated-fact findings must be lists")
    if len(unsupported) > total_claims:
        raise ScoringDataError("manual unsupported claims exceed total claims")
    for finding in unsupported:
        _validate_finding(finding, ("claim", "reason"), "unsupported claim")
    if not isinstance(record.get("outdated_fact_applicable"), bool):
        raise ScoringDataError("completed manual review must decide outdated applicability")
    for finding in outdated:
        _validate_finding(
            finding,
            (
                "predicted_outdated_value",
                "expected_current_value",
                "evidence_showing_change",
            ),
            "outdated fact",
        )


def answer_result(
    *,
    capability: str,
    question: Mapping[str, object],
    gold: Mapping[str, object],
    prediction: BaselinePrediction | None,
    manual_review: Mapping[str, object] | None,
    execution_failure: Mapping[str, object] | None,
) -> tuple[str, str, str]:
    """Return correctness, method, and a short scoring note."""

    if execution_failure is not None or prediction is None:
        return "incorrect", "deterministic", "Prediction generation failed."
    expected_abstention = required_bool(gold, "should_abstain", prediction.case_id)
    if expected_abstention or prediction.status == "abstained":
        correct = expected_abstention and prediction.status == "abstained"
        return (
            "correct" if correct else "incorrect",
            "deterministic",
            "Abstention status matched." if correct else "Abstention status did not match.",
        )

    reference = required_string(gold, "reference_answer", prediction.case_id)
    accepted = acceptable_answers(gold, prediction.case_id)
    if deterministic_answer_match(prediction.answer, reference, accepted):
        return "correct", "deterministic", "The normalized factual answer matched."
    if capability not in {"conflict_detection", "user_modeling"}:
        return "incorrect", "deterministic", "The normalized factual answer did not match."
    if manual_review is None or manual_review.get("reviewer_judgment") is None:
        return "unresolved", "manual", "Manual semantic review is pending."
    judgment = manual_review["reviewer_judgment"]
    correctness = "partial" if judgment == "partially_correct" else judgment
    return correctness, "manual", str(manual_review["short_reason"])


def case_claim_counts(
    *,
    correctness: str,
    prediction: BaselinePrediction | None,
    gold_references: set[tuple[str, str | None]],
    manual_review: Mapping[str, object] | None,
) -> tuple[int, int, list[dict[str, str]]]:
    """Count claims using manual findings or a conservative one-claim rule."""

    if prediction is None or prediction.status == "abstained":
        return 0, 0, []
    if manual_review is not None and manual_review.get("reviewer_judgment") is not None:
        findings = list(manual_review["unsupported_claims_found"])
        return int(manual_review["total_claim_count"]), len(findings), findings

    predicted_references = {
        (item.source_id, item.message_id) for item in prediction.evidence
    }
    supported = correctness == "correct" and bool(predicted_references & gold_references)
    findings = [] if supported else [
        {
            "claim": prediction.answer,
            "reason": "The cited evidence does not deterministically support this answer.",
        }
    ]
    return 1, len(findings), findings


def capability_metrics(case_scores: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Group answer accuracy under the five frozen capability labels."""

    present = {item["capability"] for item in case_scores}
    if present != set(CAPABILITIES):
        raise ScoringDataError("case scores must contain exactly the five capabilities")
    output: dict[str, object] = {}
    for capability in CAPABILITIES:
        cases = [item for item in case_scores if item["capability"] == capability]
        counts = {
            status: sum(item["answer_correctness"] == status for item in cases)
            for status in ("correct", "partial", "incorrect", "unresolved")
        }
        output[capability] = {
            "total_cases": len(cases),
            "correct": counts["correct"],
            "partially_correct": counts["partial"],
            "incorrect": counts["incorrect"],
            "unresolved_manual_reviews": counts["unresolved"],
            "execution_failures": sum(item["execution_failure"] is not None for item in cases),
            "strict_accuracy": rate(counts["correct"], len(cases), "capability has no cases"),
        }
    return output


def answer_accuracy(case_scores: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Calculate strict, reviewed, and explicitly weighted lenient accuracy."""

    total = len(case_scores)
    correct = sum(item["answer_correctness"] == "correct" for item in case_scores)
    partial = sum(item["answer_correctness"] == "partial" for item in case_scores)
    unresolved = sum(item["answer_correctness"] == "unresolved" for item in case_scores)
    reviewed = total - unresolved
    return {
        "strict_accuracy": rate(correct, total, "the evaluation set is empty"),
        "reviewed_accuracy": rate(correct, reviewed, "no judgments are complete"),
        "lenient_accuracy": rate(
            correct + partial * PARTIAL_CREDIT_VALUE,
            total,
            "the evaluation set is empty",
        ),
        "partial_credit_value": PARTIAL_CREDIT_VALUE,
        "fully_correct": correct,
        "partially_correct": partial,
        "incorrect": sum(item["answer_correctness"] == "incorrect" for item in case_scores),
        "unresolved": unresolved,
    }


def acceptable_answers(gold: Mapping[str, object], case_id: str) -> list[str]:
    """Return validated acceptable answers without changing their content."""

    value = gold.get("acceptable_answers")
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ScoringDataError(f"gold answer {case_id}.acceptable_answers is invalid")
    return list(value)


def required_string(record: Mapping[str, object], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScoringDataError(f"{location}.{field} must be a non-empty string")
    return value


def required_bool(record: Mapping[str, object], field: str, location: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ScoringDataError(f"{location}.{field} must be a boolean")
    return value


def _validate_finding(
    value: object, fields: Sequence[str], description: str
) -> None:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ScoringDataError(f"{description} finding has invalid fields")
    if any(not isinstance(value[field], str) or not value[field].strip() for field in fields):
        raise ScoringDataError(f"{description} finding fields must be non-empty strings")


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
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScoringDataError(
                f"{description} {path}:{line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ScoringDataError(f"{description} {path}:{line_number} must be an object")
        records.append(value)
    return records
