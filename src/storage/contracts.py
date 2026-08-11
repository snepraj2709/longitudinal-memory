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
OUTBOX_EVENT_TYPES = frozenset(
    {
        "source_ingested",
        "claims_changed",
        "claim_recompute_required",
        "source_deleted",
        "claim_lifecycle_changed",
        "conflict_recompute_required",
        "belief_resolved",
    }
)
OUTBOX_STATES = frozenset({"pending", "published"})
POLARITIES = frozenset({"positive", "negative"})
EPISTEMIC_STATUSES = frozenset(
    {"asserted", "inferred", "reported_by_other", "hypothetical", "uncertain", "denied", "corrected"}
)
TIME_PRECISIONS = frozenset({"timestamp", "day", "month", "year", "approximate", "unknown"})
MEMORY_KINDS = frozenset({"episodic", "durative"})
LIFECYCLE_STATUSES = frozenset({"candidate", "confirmed", "current", "historical", "disputed", "superseded", "excluded"})
SENSITIVITIES = frozenset({"standard", "sensitive", "restricted"})
SUPPORT_TYPES = frozenset({"supports", "contradicts", "corrects"})
CONFLICT_LABELS = frozenset(
    {
        "hard_contradiction",
        "temporal_change",
        "explicit_correction",
        "refinement",
        "source_disagreement",
        "retraction",
        "unresolved_ambiguity",
        "unrelated",
    }
)
CLAIM_RELATION_TYPES = frozenset(
    {
        "supports",
        "contradicts",
        "corrects",
        "supersedes",
        "refines",
        "same_event_as",
        "caused_by",
        "hindered_by",
        "same_topic_as",
    }
)
SYMMETRIC_CLAIM_RELATION_TYPES = frozenset(
    {"contradicts", "same_event_as", "same_topic_as"}
)


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


def _validate_valid_time(
    time_precision: str,
    valid_from_date: date | None,
    valid_from_timestamp: datetime | None,
    valid_to_date: date | None,
    valid_to_timestamp: datetime | None,
) -> None:
    dates = (valid_from_date, valid_to_date)
    timestamps = (valid_from_timestamp, valid_to_timestamp)
    if any(value is not None and type(value) is not date for value in dates):
        raise StorageValidationError("valid date boundaries must be dates")
    for value in timestamps:
        if value is not None:
            _aware(value, "valid timestamp boundary")
    if any(value is not None for value in dates) and any(
        value is not None for value in timestamps
    ):
        raise StorageValidationError(
            "valid-time date and timestamp representations cannot mix"
        )
    if time_precision == "unknown":
        if any(value is not None for value in (*dates, *timestamps)):
            raise StorageValidationError(
                "unknown precision cannot carry a valid-time boundary"
            )
    elif time_precision == "timestamp":
        if not any(value is not None for value in timestamps) or any(
            value is not None for value in dates
        ):
            raise StorageValidationError(
                "timestamp precision requires timestamp boundaries"
            )
    elif not any(value is not None for value in dates) or any(
        value is not None for value in timestamps
    ):
        raise StorageValidationError("date precision requires date boundaries")
    if dates[0] is not None and dates[1] is not None and dates[0] > dates[1]:
        raise StorageValidationError("valid date boundaries are inclusive and ordered")
    if (
        timestamps[0] is not None
        and timestamps[1] is not None
        and timestamps[0] > timestamps[1]
    ):
        raise StorageValidationError(
            "valid timestamp boundaries are inclusive and ordered"
        )


def _valid_contains(
    value: date | datetime,
    time_precision: str,
    valid_from_date: date | None,
    valid_from_timestamp: datetime | None,
    valid_to_date: date | None,
    valid_to_timestamp: datetime | None,
    owner: str,
) -> bool:
    if time_precision == "unknown":
        return False
    if time_precision == "timestamp":
        _aware(value, "valid-time query")
        start, end = valid_from_timestamp, valid_to_timestamp
    else:
        if type(value) is not date:
            raise StorageValidationError(
                f"date-precision {owner} require a date query"
            )
        start, end = valid_from_date, valid_to_date
    return (start is None or start <= value) and (end is None or value <= end)


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
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    retryable: bool = False

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
        _text(self.lease_owner, "lease_owner", nullable=True)
        if self.lease_expires_at is not None:
            _aware(self.lease_expires_at, "lease_expires_at")
        if not isinstance(self.retryable, bool):
            raise StorageValidationError("retryable must be a boolean")
        self._validate_state()

    def _validate_state(self) -> None:
        has_error = self.sanitized_error_code is not None
        has_lease = self.lease_owner is not None or self.lease_expires_at is not None
        if self.state == "pending" and (
            self.completed_at is not None or has_error or has_lease or not self.retryable
        ):
            raise StorageValidationError("pending attempt fields are inconsistent")
        if self.state == "running" and (
            self.completed_at is not None
            or has_error
            or self.lease_owner is None
            or self.lease_expires_at is None
            or not self.retryable
        ):
            raise StorageValidationError("running attempt fields are inconsistent")
        if self.state == "succeeded" and (
            self.completed_at is None or has_error or has_lease or self.retryable
        ):
            raise StorageValidationError("succeeded attempt fields are inconsistent")
        if self.state == "failed" and (
            self.completed_at is None or not has_error or has_lease
        ):
            raise StorageValidationError("failed attempt fields are inconsistent")


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
        _validate_valid_time(
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
        )

    def valid_contains(self, value: date | datetime) -> bool:
        """Return membership in the inclusive valid-time interval."""

        return _valid_contains(
            value,
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
            "claims",
        )


@dataclass(frozen=True)
class ClaimVersionRecord:
    version_id: str
    user_id: str
    claim_id: str
    lifecycle_status: str
    transaction_from: datetime
    transaction_to: datetime | None = None
    belief_confidence: Real | None = None
    valid_from_date: date | None = None
    valid_from_timestamp: datetime | None = None
    valid_to_date: date | None = None
    valid_to_timestamp: datetime | None = None
    time_precision: str = "unknown"

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
        _enum(self.time_precision, TIME_PRECISIONS, "time_precision")
        self._validate_time()

    def _validate_time(self) -> None:
        _validate_valid_time(
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
        )
        if self.lifecycle_status == "current" and self.time_precision == "unknown":
            raise StorageValidationError("current versions require known valid time")

    def transaction_contains(self, value: datetime) -> bool:
        """Return membership in the start-inclusive, end-exclusive interval."""

        _aware(value, "transaction query time")
        return self.transaction_from <= value and (
            self.transaction_to is None or value < self.transaction_to
        )

    def valid_contains(self, value: date | datetime) -> bool:
        return _valid_contains(
            value,
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
            "versions",
        )


@dataclass(frozen=True)
class LifecycleTransitionRecord:
    transition_id: str
    user_id: str
    idempotency_key: str
    claim_id: str
    from_version_id: str
    to_version_id: str
    target_status: str
    reason: str
    replacement_claim_id: str | None
    transitioned_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "transition_id", "user_id", "idempotency_key", "claim_id",
            "from_version_id", "to_version_id", "reason",
        ):
            _text(getattr(self, name), name)
        _text(self.replacement_claim_id, "replacement_claim_id", nullable=True)
        _enum(
            self.target_status,
            LIFECYCLE_STATUSES - {"candidate"},
            "target_status",
        )
        if len(self.reason) > 500:
            raise StorageValidationError("reason must be at most 500 characters")
        if self.from_version_id == self.to_version_id:
            raise StorageValidationError("transition versions must differ")
        if self.replacement_claim_id == self.claim_id:
            raise StorageValidationError("replacement claim must differ")
        _aware(self.transitioned_at, "transitioned_at")


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


@dataclass(frozen=True)
class ClaimExtractionRecord:
    user_id: str
    claim_id: str
    source_id: str
    extraction_version_id: str
    attempt_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "user_id",
            "claim_id",
            "source_id",
            "extraction_version_id",
            "attempt_id",
        ):
            _text(getattr(self, name), name)
        _aware(self.created_at, "created_at")


@dataclass(frozen=True)
class ProcessingOutboxRecord:
    event_id: str
    user_id: str
    event_type: str
    aggregate_id: str
    dedupe_key: str
    payload: dict[str, JSONValue]
    state: str
    created_at: datetime
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "user_id", "aggregate_id", "dedupe_key"):
            _text(getattr(self, name), name)
        _enum(self.event_type, OUTBOX_EVENT_TYPES, "event_type")
        _enum(self.state, OUTBOX_STATES, "state")
        object.__setattr__(
            self, "payload", safe_json(self.payload, "payload", top_type=dict)
        )
        _aware(self.created_at, "created_at")
        if self.published_at is not None:
            _aware(self.published_at, "published_at")
            if self.published_at < self.created_at:
                raise StorageValidationError("published_at cannot precede created_at")
        if (self.state == "pending") != (self.published_at is None):
            raise StorageValidationError("outbox state and published_at are inconsistent")


@dataclass(frozen=True)
class SourceTombstoneRecord:
    user_id: str
    source_id: str
    idempotency_key: str
    content_hash: str
    deleted_at: datetime

    def __post_init__(self) -> None:
        for name in ("user_id", "source_id", "idempotency_key"):
            _text(getattr(self, name), name)
        if not _is_sha256(self.content_hash):
            raise StorageValidationError("content_hash must be a lowercase SHA-256")
        _aware(self.deleted_at, "deleted_at")


@dataclass(frozen=True)
class ConflictDecisionRecord:
    decision_id: str
    user_id: str
    classifier_version: str
    rule_version: str
    pair_id: str
    left_claim_id: str
    right_claim_id: str
    left_version_id: str
    right_version_id: str
    input_snapshot_sha256: str
    transaction_as_of: datetime
    matched_rule: str
    label: str
    classified_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "decision_id",
            "user_id",
            "classifier_version",
            "rule_version",
            "pair_id",
            "left_claim_id",
            "right_claim_id",
            "left_version_id",
            "right_version_id",
        ):
            _text(getattr(self, name), name)
        if self.left_claim_id >= self.right_claim_id:
            raise StorageValidationError("decision claims must be canonical")
        if not _is_sha256(self.input_snapshot_sha256):
            raise StorageValidationError(
                "input_snapshot_sha256 must be a lowercase SHA-256"
            )
        _aware(self.transaction_as_of, "transaction_as_of")
        _aware(self.classified_at, "classified_at")
        _enum(self.matched_rule, CONFLICT_LABELS, "matched_rule")
        _enum(self.label, CONFLICT_LABELS, "label")
        if self.matched_rule != self.label:
            raise StorageValidationError("matched_rule must equal the frozen label")


@dataclass(frozen=True)
class ClaimRelationRecord:
    relation_id: str
    user_id: str
    decision_id: str
    classifier_version: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    confidence: Real
    input_snapshot_sha256: str
    created_at: datetime
    resolver_version: str | None = None
    resolution_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "user_id",
            "decision_id",
            "classifier_version",
            "source_claim_id",
            "target_claim_id",
        ):
            _text(getattr(self, name), name)
        if self.source_claim_id == self.target_claim_id:
            raise StorageValidationError("relation claims must differ")
        _enum(self.relation_type, CLAIM_RELATION_TYPES, "relation_type")
        if self.relation_type in SYMMETRIC_CLAIM_RELATION_TYPES and (
            self.source_claim_id >= self.target_claim_id
        ):
            raise StorageValidationError("symmetric relation claims must be canonical")
        object.__setattr__(
            self, "confidence", _confidence(self.confidence, "confidence")
        )
        if self.confidence != 1.0:
            raise StorageValidationError("checked relation confidence must be one")
        if not _is_sha256(self.input_snapshot_sha256):
            raise StorageValidationError(
                "input_snapshot_sha256 must be a lowercase SHA-256"
            )
        _aware(self.created_at, "created_at")
        _text(self.resolver_version, "resolver_version", nullable=True)
        _text(self.resolution_id, "resolution_id", nullable=True)
        if (self.resolver_version is None) != (self.resolution_id is None):
            raise StorageValidationError(
                "resolver relation provenance must be wholly null or wholly set"
            )


@dataclass(frozen=True)
class ConflictDecisionEvidenceRecord:
    decision_evidence_id: str
    user_id: str
    decision_id: str
    claim_id: str
    span_id: str
    support_type: str
    input_snapshot_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "decision_evidence_id",
            "user_id",
            "decision_id",
            "claim_id",
            "span_id",
        ):
            _text(getattr(self, name), name)
        _enum(self.support_type, SUPPORT_TYPES, "support_type")
        if not _is_sha256(self.input_snapshot_sha256):
            raise StorageValidationError(
                "input_snapshot_sha256 must be a lowercase SHA-256"
            )


BELIEF_RESOLUTION_OUTCOMES = frozenset(
    {
        "no_change",
        "excluded",
        "temporal_change_resolved",
        "correction_resolved",
        "refinement_resolved",
        "retraction_resolved",
        "authority_resolved",
        "disputed",
    }
)


@dataclass(frozen=True)
class BeliefResolutionRecord:
    resolution_id: str
    user_id: str
    resolver_version: str
    policy_version: str
    decision_id: str
    idempotency_key: str
    input_snapshot_sha256: str
    transaction_as_of: datetime
    valid_at_date: date | None
    valid_at_timestamp: datetime | None
    resolved_at: datetime
    outcome: str
    selected_current_claim_id: str | None
    authority_reason: str
    belief_confidence: None = None

    def __post_init__(self) -> None:
        for name in (
            "resolution_id", "user_id", "resolver_version", "policy_version",
            "decision_id", "idempotency_key", "authority_reason",
        ):
            _text(getattr(self, name), name)
        if not _is_sha256(self.input_snapshot_sha256):
            raise StorageValidationError(
                "input_snapshot_sha256 must be a lowercase SHA-256"
            )
        _aware(self.transaction_as_of, "transaction_as_of")
        _aware(self.resolved_at, "resolved_at")
        if self.resolved_at < self.transaction_as_of:
            raise StorageValidationError("resolved_at precedes transaction_as_of")
        if self.valid_at_date is not None and type(self.valid_at_date) is not date:
            raise StorageValidationError("valid_at_date must be a date")
        if self.valid_at_timestamp is not None:
            _aware(self.valid_at_timestamp, "valid_at_timestamp")
        if self.valid_at_date is not None and self.valid_at_timestamp is not None:
            raise StorageValidationError("valid_at representations cannot mix")
        _enum(self.outcome, BELIEF_RESOLUTION_OUTCOMES, "outcome")
        _text(
            self.selected_current_claim_id,
            "selected_current_claim_id",
            nullable=True,
        )
        if self.selected_current_claim_id is not None and (
            self.valid_at_date is None and self.valid_at_timestamp is None
        ):
            raise StorageValidationError("selected current claim requires valid_at")
        if self.belief_confidence is not None:
            raise StorageValidationError("belief_confidence must remain null")


@dataclass(frozen=True)
class BeliefResolutionActionRecord:
    action_id: str
    user_id: str
    resolution_id: str
    action_order: int
    claim_id: str
    from_status: str
    target_status: str
    replacement_claim_id: str | None
    reason: str
    from_version_id: str
    to_version_id: str
    transition_id: str

    def __post_init__(self) -> None:
        for name in (
            "action_id", "user_id", "resolution_id", "claim_id", "reason",
            "from_version_id", "to_version_id", "transition_id",
        ):
            _text(getattr(self, name), name)
        if isinstance(self.action_order, bool) or not isinstance(
            self.action_order, int
        ) or self.action_order < 1:
            raise StorageValidationError("action_order must be positive")
        _enum(self.from_status, LIFECYCLE_STATUSES, "from_status")
        _enum(
            self.target_status,
            LIFECYCLE_STATUSES - {"candidate"},
            "target_status",
        )
        if self.from_status == self.target_status:
            raise StorageValidationError("resolution action must change status")
        _text(
            self.replacement_claim_id,
            "replacement_claim_id",
            nullable=True,
        )
        if self.replacement_claim_id == self.claim_id:
            raise StorageValidationError("replacement claim must differ")
        if self.from_version_id == self.to_version_id:
            raise StorageValidationError("resolution action versions must differ")


@dataclass(frozen=True)
class BeliefResolutionEvidenceRecord:
    user_id: str
    resolution_id: str
    decision_id: str
    decision_evidence_id: str

    def __post_init__(self) -> None:
        for name in (
            "user_id", "resolution_id", "decision_id", "decision_evidence_id"
        ):
            _text(getattr(self, name), name)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
