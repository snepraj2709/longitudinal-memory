"""Structured output contract shared by baseline runners and scorers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


ALLOWED_PREDICTION_STATUSES = frozenset(
    {"answered", "abstained", "disputed", "partially_answered"}
)

_TOP_LEVEL_FIELDS = (
    "case_id",
    "status",
    "answer",
    "confidence",
    "evidence",
    "abstention_reason",
)
_EVIDENCE_FIELDS = ("source_id", "message_id", "quote")
_NON_ABSTAINED_STATUSES = ALLOWED_PREDICTION_STATUSES - {"abstained"}


class PredictionValidationError(ValueError):
    """Reports every validation problem found in one prediction record."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"Invalid baseline prediction:\n{details}")


@dataclass(frozen=True)
class EvidenceReference:
    """A verbatim source reference supporting a generated answer."""

    source_id: str
    message_id: str | None
    quote: str


@dataclass(frozen=True)
class BaselinePrediction:
    """A validated prediction produced by a memory baseline."""

    case_id: str
    status: str
    answer: str
    confidence: Real
    evidence: tuple[EvidenceReference, ...]
    abstention_reason: str | None


def validate_prediction(record: object) -> BaselinePrediction:
    """Validate and return one baseline prediction.

    Values are checked as supplied. The validator does not trim strings, coerce
    types, infer missing fields, or consult gold evaluation data.
    """

    if not isinstance(record, dict):
        raise PredictionValidationError(["prediction must be an object"])

    errors: list[str] = []
    _validate_field_set(record, _TOP_LEVEL_FIELDS, "prediction", errors)

    if "case_id" in record:
        _validate_non_empty_string(record["case_id"], "case_id", errors)

    if "answer" in record:
        _validate_non_empty_string(record["answer"], "answer", errors)

    status = record.get("status")
    if "status" in record and (
        not isinstance(status, str) or status not in ALLOWED_PREDICTION_STATUSES
    ):
        allowed = ", ".join(sorted(ALLOWED_PREDICTION_STATUSES))
        errors.append(f"status must be one of: {allowed}")

    if "confidence" in record:
        _validate_confidence(record["confidence"], errors)

    validated_evidence: list[EvidenceReference] = []
    evidence = record.get("evidence")
    if "evidence" in record:
        if not isinstance(evidence, list):
            errors.append("evidence must be a list")
        else:
            validated_evidence = _validate_evidence(evidence, errors)

    abstention_reason = record.get("abstention_reason")
    if status == "abstained":
        if isinstance(evidence, list) and evidence:
            errors.append("abstained predictions must have an empty evidence list")
        if not _is_non_empty_string(abstention_reason):
            errors.append(
                "abstained predictions must have a non-empty abstention_reason"
            )
    elif isinstance(status, str) and status in _NON_ABSTAINED_STATUSES:
        if isinstance(evidence, list) and not evidence:
            errors.append(f"{status} predictions must include at least one evidence item")
        if abstention_reason is not None:
            errors.append(f"{status} predictions must have a null abstention_reason")

    if errors:
        raise PredictionValidationError(errors)

    return BaselinePrediction(
        case_id=record["case_id"],
        status=record["status"],
        answer=record["answer"],
        confidence=record["confidence"],
        evidence=tuple(validated_evidence),
        abstention_reason=record["abstention_reason"],
    )


def _validate_field_set(
    record: dict[Any, Any],
    expected_fields: tuple[str, ...],
    location: str,
    errors: list[str],
) -> None:
    expected = set(expected_fields)
    for field in expected_fields:
        if field not in record:
            errors.append(f"{location} is missing required field: {field}")
    unknown_fields = (field for field in record if field not in expected)
    for field in sorted(unknown_fields, key=repr):
        errors.append(f"{location} contains unknown field: {field}")


def _validate_non_empty_string(
    value: object, location: str, errors: list[str]
) -> bool:
    if not _is_non_empty_string(value):
        errors.append(f"{location} must be a non-empty string")
        return False
    return True


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_confidence(value: object, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append("confidence must be a real number")
    elif not math.isfinite(value):
        errors.append("confidence must be finite")
    elif not 0 <= value <= 1:
        errors.append("confidence must be between 0 and 1 inclusive")


def _validate_evidence(
    evidence: list[object], errors: list[str]
) -> list[EvidenceReference]:
    validated: list[EvidenceReference] = []
    seen_references: set[tuple[str, str | None]] = set()

    for index, item in enumerate(evidence):
        location = f"evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{location} must be an object")
            continue

        _validate_field_set(item, _EVIDENCE_FIELDS, location, errors)

        source_id_is_valid = False
        if "source_id" in item:
            source_id_is_valid = _validate_non_empty_string(
                item["source_id"], f"{location}.source_id", errors
            )

        message_id_is_valid = False
        if "message_id" in item:
            message_id = item["message_id"]
            message_id_is_valid = message_id is None or _is_non_empty_string(message_id)
            if not message_id_is_valid:
                errors.append(
                    f"{location}.message_id must be a non-empty string or null"
                )

        quote_is_valid = False
        if "quote" in item:
            quote_is_valid = _validate_non_empty_string(
                item["quote"], f"{location}.quote", errors
            )

        if source_id_is_valid and message_id_is_valid:
            reference = (item["source_id"], item["message_id"])
            if reference in seen_references:
                errors.append(
                    f"{location} duplicates evidence reference "
                    f"({item['source_id']!r}, {item['message_id']!r})"
                )
            else:
                seen_references.add(reference)

        if source_id_is_valid and message_id_is_valid and quote_is_valid:
            validated.append(
                EvidenceReference(
                    source_id=item["source_id"],
                    message_id=item["message_id"],
                    quote=item["quote"],
                )
            )

    return validated
