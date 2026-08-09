"""Typed records and boundary validation for Phase 4 storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from numbers import Real
from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

SOURCE_TYPES = frozenset({"conversation", "email", "calendar", "chat"})
PROCESSING_STATES = frozenset({"pending", "running", "succeeded", "failed"})
POLARITIES = frozenset({"positive", "negative"})
EPISTEMIC_STATUSES = frozenset(
    {"asserted", "inferred", "reported_by_other", "hypothetical", "uncertain", "denied", "corrected"}
)
TIME_PRECISIONS = frozenset({"timestamp", "day", "month", "year", "approximate", "unknown"})
MEMORY_KINDS = frozenset({"episodic", "durative"})
LIFECYCLE_STATUSES = frozenset({"candidate", "confirmed", "current", "historical", "disputed", "superseded", "excluded"})
SENSITIVITIES = frozenset({"standard", "sensitive", "restricted"})
SUPPORT_TYPES = frozenset({"supports", "contradicts", "corrects"})


class StorageValidationError(ValueError):
    """Reject an invalid record before it reaches PostgreSQL."""


def _text(value: str, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StorageValidationError(f"{name} must be non-empty text")
    return value


def _enum(value: str, allowed: frozenset[str], name: str) -> str:
    _text(value, name)
    if value not in allowed:
        raise StorageValidationError(f"{name} is invalid")
    return value


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise StorageValidationError(f"{name} must be timezone-aware")
    return value


def _confidence(value: Real | None, name: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StorageValidationError(f"{name} must be a finite number from 0 to 1")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise StorageValidationError(f"{name} must be a finite number from 0 to 1")
    return numeric


def safe_json(value: object, name: str, *, top_type: type | None = None) -> JSONValue:
    """Return a detached JSON value, rejecting Python's non-JSON extensions."""

    def validate(item: object, location: str) -> None:
        if item is None:
            return
        if isinstance(item, bool | str | int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise StorageValidationError(f"{location} must not contain NaN or infinity")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{location}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise StorageValidationError(f"{location} contains a non-string key")
                validate(child, f"{location}.{key}")
            return
        raise StorageValidationError(f"{location} is not JSON-compatible")

    if top_type is not None and not isinstance(value, top_type):
        raise StorageValidationError(f"{name} must be a JSON {top_type.__name__}")
    if value is None:
        raise StorageValidationError(f"{name} must not be JSON null")
    validate(value, name)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True)
class MemoryUser:
    user_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _aware(self.created_at, "created_at")


@dataclass(frozen=True)
class SourceEventRecord:
    source_id: str
    user_id: str
    source_type: str
    session_id: str | None
    idempotency_key: str
    produced_at: datetime
    ingested_at: datetime
    raw_content: str
    participants: list[JSONValue]
    metadata: dict[str, JSONValue]
    content_hash: str

    def __post_init__(self) -> None:
        for name in ("source_id", "user_id", "idempotency_key", "raw_content"):
            _text(getattr(self, name), name)
        _text(self.session_id, "session_id", nullable=True)
        _enum(self.source_type, SOURCE_TYPES, "source_type")
        _aware(self.produced_at, "produced_at")
        _aware(self.ingested_at, "ingested_at")
        if not _is_sha256(self.content_hash):
            raise StorageValidationError("content_hash must be a lowercase SHA-256")
        object.__setattr__(self, "participants", safe_json(self.participants, "participants", top_type=list))
        object.__setattr__(self, "metadata", safe_json(self.metadata, "metadata", top_type=dict))


@dataclass(frozen=True)
class SourceSpanRecord:
    span_id: str
    user_id: str
    source_id: str
    message_id: str | None
    speaker_id: str
    verbatim_quote: str
    start_offset: int | None = None
    end_offset: int | None = None

    def __post_init__(self) -> None:
        for name in ("span_id", "user_id", "source_id", "speaker_id"):
            _text(getattr(self, name), name)
        _text(self.message_id, "message_id", nullable=True)
        if not isinstance(self.verbatim_quote, str) or not self.verbatim_quote:
            raise StorageValidationError("verbatim_quote must be non-empty text")
        if (self.start_offset is None) != (self.end_offset is None):
            raise StorageValidationError("span offsets must both be null or both be set")
        if self.start_offset is not None and (
            isinstance(self.start_offset, bool)
            or isinstance(self.end_offset, bool)
            or not isinstance(self.start_offset, int)
            or not isinstance(self.end_offset, int)
            or self.start_offset < 0
            or self.start_offset >= self.end_offset
        ):
            raise StorageValidationError("span offsets must satisfy 0 <= start < end")


@dataclass(frozen=True)
class ExtractionVersionRecord:
    version_id: str
    model_version: str
    prompt_version: str
    prompt_hash: str
    schema_version: str
    schema_hash: str
    registry_version: str
    registry_hash: str
    input_manifest_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("version_id", "model_version", "prompt_version", "schema_version", "registry_version"):
            _text(getattr(self, name), name)
        for name in ("prompt_hash", "schema_hash", "registry_hash", "input_manifest_hash"):
            if not _is_sha256(getattr(self, name)):
                raise StorageValidationError(f"{name} must be a lowercase SHA-256")
        _aware(self.created_at, "created_at")


@dataclass(frozen=True)
class ProcessingAttemptRecord:
    attempt_id: str
    user_id: str
    source_id: str
    extraction_version_id: str
    attempt_number: int
    state: str
    started_at: datetime
    completed_at: datetime | None = None
    sanitized_error_code: str | None = None
    sanitized_error_metadata: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        for name in ("attempt_id", "user_id", "source_id", "extraction_version_id"):
            _text(getattr(self, name), name)
        if isinstance(self.attempt_number, bool) or not isinstance(self.attempt_number, int) or self.attempt_number <= 0:
            raise StorageValidationError("attempt_number must be positive")
        _enum(self.state, PROCESSING_STATES, "state")
        _aware(self.started_at, "started_at")
        if self.completed_at is not None:
            _aware(self.completed_at, "completed_at")
            if self.completed_at < self.started_at:
                raise StorageValidationError("completed_at cannot precede started_at")
        _text(self.sanitized_error_code, "sanitized_error_code", nullable=True)
        if self.sanitized_error_metadata is not None:
            object.__setattr__(self, "sanitized_error_metadata", safe_json(self.sanitized_error_metadata, "sanitized_error_metadata", top_type=dict))


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    user_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    predicate_registry_version: str
    object_json: JSONValue
    polarity: str
    epistemic_status: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    time_precision: str
    extraction_confidence: Real
    memory_kind: str | None
    sensitivity: str | None
    extraction_version_id: str

    def __post_init__(self) -> None:
        for name in ("claim_id", "user_id", "subject_id", "speaker_id", "predicate", "predicate_registry_version", "extraction_version_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "object_json", safe_json(self.object_json, "object_json"))
        _enum(self.polarity, POLARITIES, "polarity")
        _enum(self.epistemic_status, EPISTEMIC_STATUSES, "epistemic_status")
        _enum(self.time_precision, TIME_PRECISIONS, "time_precision")
        if self.memory_kind is not None:
            _enum(self.memory_kind, MEMORY_KINDS, "memory_kind")
        if self.sensitivity is not None:
            _enum(self.sensitivity, SENSITIVITIES, "sensitivity")
        object.__setattr__(self, "extraction_confidence", _confidence(self.extraction_confidence, "extraction_confidence"))
        self._validate_time()

    def _validate_time(self) -> None:
        dates = (self.valid_from_date, self.valid_to_date)
        timestamps = (self.valid_from_timestamp, self.valid_to_timestamp)
        if any(value is not None and type(value) is not date for value in dates):
            raise StorageValidationError("valid date boundaries must be dates")
        if any(value is not None for value in timestamps):
            for value in timestamps:
                if value is not None:
                    _aware(value, "valid timestamp boundary")
        if any(value is not None for value in dates) and any(value is not None for value in timestamps):
            raise StorageValidationError("valid-time date and timestamp representations cannot mix")
        if self.time_precision == "unknown":
            if any(value is not None for value in (*dates, *timestamps)):
                raise StorageValidationError("unknown precision cannot carry a valid-time boundary")
        elif self.time_precision == "timestamp":
            if not any(value is not None for value in timestamps) or any(value is not None for value in dates):
                raise StorageValidationError("timestamp precision requires timestamp boundaries")
        elif not any(value is not None for value in dates) or any(value is not None for value in timestamps):
            raise StorageValidationError("date precision requires date boundaries")
        if dates[0] is not None and dates[1] is not None and dates[0] > dates[1]:
            raise StorageValidationError("valid date boundaries are inclusive and ordered")
        if timestamps[0] is not None and timestamps[1] is not None and timestamps[0] > timestamps[1]:
            raise StorageValidationError("valid timestamp boundaries are inclusive and ordered")

    def valid_contains(self, value: date | datetime) -> bool:
        """Return membership in the inclusive valid-time interval."""

        if self.time_precision == "unknown":
            return False
        if self.time_precision == "timestamp":
            _aware(value, "valid-time query")
            start, end = self.valid_from_timestamp, self.valid_to_timestamp
        else:
            if type(value) is not date:
                raise StorageValidationError(
                    "date-precision claims require a date query"
                )
            start, end = self.valid_from_date, self.valid_to_date
        return (start is None or start <= value) and (end is None or value <= end)


@dataclass(frozen=True)
class ClaimVersionRecord:
    version_id: str
    user_id: str
    claim_id: str
    lifecycle_status: str
    transaction_from: datetime
    transaction_to: datetime | None = None
    belief_confidence: Real | None = None

    def __post_init__(self) -> None:
        for name in ("version_id", "user_id", "claim_id"):
            _text(getattr(self, name), name)
        _enum(self.lifecycle_status, LIFECYCLE_STATUSES, "lifecycle_status")
        _aware(self.transaction_from, "transaction_from")
        if self.transaction_to is not None:
            _aware(self.transaction_to, "transaction_to")
            if self.transaction_to <= self.transaction_from:
                raise StorageValidationError("transaction interval must be [from, to)")
        object.__setattr__(self, "belief_confidence", _confidence(self.belief_confidence, "belief_confidence", nullable=True))

    def transaction_contains(self, value: datetime) -> bool:
        """Return membership in the start-inclusive, end-exclusive interval."""

        _aware(value, "transaction query time")
        return self.transaction_from <= value and (
            self.transaction_to is None or value < self.transaction_to
        )


@dataclass(frozen=True)
class EvidenceLinkRecord:
    user_id: str
    claim_id: str
    span_id: str
    support_type: str
    extraction_confidence: Real

    def __post_init__(self) -> None:
        for name in ("user_id", "claim_id", "span_id"):
            _text(getattr(self, name), name)
        _enum(self.support_type, SUPPORT_TYPES, "support_type")
        object.__setattr__(self, "extraction_confidence", _confidence(self.extraction_confidence, "extraction_confidence"))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
