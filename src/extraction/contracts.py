"""Strict standard-library contracts for Phase 3 atomic extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Real
from typing import Any


ALLOWED_POLARITIES = frozenset({"negative", "positive"})
ALLOWED_EPISTEMIC_STATUSES = frozenset(
    {
        "asserted",
        "corrected",
        "denied",
        "hypothetical",
        "inferred",
        "reported_by_other",
        "uncertain",
    }
)

_CLAIM_FIELDS = (
    "claim_id",
    "subject_id",
    "speaker_id",
    "predicate",
    "object",
    "polarity",
    "epistemic_status",
    "valid_from",
    "valid_to",
    "confidence",
    "evidence",
)
_EVIDENCE_FIELDS = ("source_id", "message_id", "quote")


class AtomicClaimValidationError(ValueError):
    """Reports every validation problem found in one atomic claim."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"Invalid atomic claim:\n{details}")


@dataclass(frozen=True)
class EvidenceSpanV1:
    """One exact source excerpt supporting a claim."""

    source_id: str
    message_id: str | None
    quote: str


@dataclass(frozen=True)
class AtomicClaimV1:
    """One immutable source-grounded atomic claim."""

    claim_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object: object
    polarity: str
    epistemic_status: str
    valid_from: str | None
    valid_to: str | None
    confidence: Real
    evidence: tuple[EvidenceSpanV1, ...]


def validate_atomic_claim(record: object) -> AtomicClaimV1:
    """Validate one claim without coercing or inferring supplied values."""

    if not isinstance(record, dict):
        raise AtomicClaimValidationError(["claim must be an object"])

    errors: list[str] = []
    _validate_field_set(record, _CLAIM_FIELDS, "claim", errors)

    for field in ("claim_id", "subject_id", "speaker_id", "predicate"):
        if field in record:
            _validate_non_empty_string(record[field], field, errors)

    if "object" in record and not _is_json_value(record["object"]):
        errors.append("object must be a JSON value")

    if "polarity" in record:
        _validate_choice(record["polarity"], "polarity", ALLOWED_POLARITIES, errors)
    if "epistemic_status" in record:
        _validate_choice(
            record["epistemic_status"],
            "epistemic_status",
            ALLOWED_EPISTEMIC_STATUSES,
            errors,
        )

    valid_from = _validate_time(record.get("valid_from"), "valid_from", errors)
    valid_to = _validate_time(record.get("valid_to"), "valid_to", errors)
    if valid_from is not None and valid_to is not None and _after(valid_from, valid_to):
        errors.append("valid_to must not be before valid_from")

    if "confidence" in record:
        _validate_confidence(record["confidence"], errors)

    evidence = _validate_evidence(record.get("evidence"), errors)

    if errors:
        raise AtomicClaimValidationError(errors)

    return AtomicClaimV1(
        claim_id=record["claim_id"],
        subject_id=record["subject_id"],
        speaker_id=record["speaker_id"],
        predicate=record["predicate"],
        object=record["object"],
        polarity=record["polarity"],
        epistemic_status=record["epistemic_status"],
        valid_from=record["valid_from"],
        valid_to=record["valid_to"],
        confidence=record["confidence"],
        evidence=tuple(evidence),
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
    for field in sorted((field for field in record if field not in expected), key=repr):
        errors.append(f"{location} contains unknown field: {field}")


def _validate_non_empty_string(
    value: object, location: str, errors: list[str]
) -> bool:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location} must be a non-empty string")
        return False
    return True


def _validate_choice(
    value: object,
    location: str,
    allowed: frozenset[str],
    errors: list[str],
) -> None:
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{location} must be one of: {', '.join(sorted(allowed))}")


def _validate_time(
    value: object, location: str, errors: list[str]
) -> date | datetime | None:
    if value is None:
        return None
    message = f"{location} must be an ISO date, timezone-aware ISO datetime, or null"
    if not isinstance(value, str) or not value.strip():
        errors.append(message)
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            errors.append(message)
            return None
        if parsed.utcoffset() is None:
            errors.append(message)
            return None
        return parsed


def _after(start: date | datetime, end: date | datetime) -> bool:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return start > end
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    return start_date > end_date


def _validate_confidence(value: object, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append("confidence must be a real number")
    elif not math.isfinite(value):
        errors.append("confidence must be finite")
    elif not 0 <= value <= 1:
        errors.append("confidence must be between 0 and 1 inclusive")


def _validate_evidence(
    value: object, errors: list[str]
) -> list[EvidenceSpanV1]:
    if not isinstance(value, list):
        errors.append("evidence must be a list")
        return []
    if not value:
        errors.append("evidence must contain at least one item")

    validated: list[EvidenceSpanV1] = []
    seen: set[tuple[str, str | None]] = set()
    for index, item in enumerate(value):
        location = f"evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{location} must be an object")
            continue

        _validate_field_set(item, _EVIDENCE_FIELDS, location, errors)
        source_valid = (
            _validate_non_empty_string(item["source_id"], f"{location}.source_id", errors)
            if "source_id" in item
            else False
        )
        message_valid = False
        if "message_id" in item:
            message_id = item["message_id"]
            message_valid = message_id is None or (
                isinstance(message_id, str) and bool(message_id.strip())
            )
            if not message_valid:
                errors.append(
                    f"{location}.message_id must be a non-empty string or null"
                )
        quote_valid = (
            _validate_non_empty_string(item["quote"], f"{location}.quote", errors)
            if "quote" in item
            else False
        )

        if source_valid and message_valid:
            reference = (item["source_id"], item["message_id"])
            if reference in seen:
                errors.append(f"{location} duplicates evidence reference {reference!r}")
            else:
                seen.add(reference)

        if source_valid and message_valid and quote_valid:
            validated.append(
                EvidenceSpanV1(
                    source_id=item["source_id"],
                    message_id=item["message_id"],
                    quote=item["quote"],
                )
            )
    return validated


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False
