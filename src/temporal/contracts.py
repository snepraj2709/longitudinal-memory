"""Typed, explicit inputs and outputs for temporal lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real

from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    LIFECYCLE_STATUSES,
    LifecycleTransitionRecord,
    StorageValidationError,
    _confidence,
)


TRANSITION_MATRIX = {
    "candidate": frozenset({"confirmed", "current", "disputed", "excluded"}),
    "confirmed": frozenset({"disputed", "superseded", "excluded"}),
    "current": frozenset({"historical", "disputed", "superseded", "excluded"}),
    "historical": frozenset({"disputed", "superseded", "excluded"}),
    "disputed": frozenset({"confirmed", "current", "historical", "superseded", "excluded"}),
    "superseded": frozenset(),
    "excluded": frozenset(),
}
ACCEPTED_STATUSES = frozenset({"confirmed", "current", "historical"})


def _text(value: object, name: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorageValidationError(f"{name} must be non-empty text")
    if maximum is not None and len(value) > maximum:
        raise StorageValidationError(f"{name} must be at most {maximum} characters")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise StorageValidationError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class TransitionRequest:
    user_id: str
    claim_id: str
    idempotency_key: str
    target_status: str
    reason: str
    transitioned_at: datetime
    belief_confidence: Real | None = None

    def __post_init__(self) -> None:
        for name in ("user_id", "claim_id", "idempotency_key"):
            _text(getattr(self, name), name)
        if self.target_status not in LIFECYCLE_STATUSES or self.target_status == "candidate":
            raise StorageValidationError("target_status is invalid")
        _text(self.reason, "reason", maximum=500)
        _aware(self.transitioned_at, "transitioned_at")
        object.__setattr__(
            self,
            "belief_confidence",
            _confidence(self.belief_confidence, "belief_confidence", nullable=True),
        )


@dataclass(frozen=True)
class CorrectionRequest:
    user_id: str
    replaced_claim_id: str
    replacement_claim_id: str
    idempotency_key: str
    replacement_target_status: str
    reason: str
    transitioned_at: datetime
    replaced_belief_confidence: Real | None = None
    replacement_belief_confidence: Real | None = None

    def __post_init__(self) -> None:
        for name in (
            "user_id", "replaced_claim_id", "replacement_claim_id", "idempotency_key"
        ):
            _text(getattr(self, name), name)
        if self.replaced_claim_id == self.replacement_claim_id:
            raise StorageValidationError("correction claims must differ")
        if self.replacement_target_status not in {"confirmed", "current"}:
            raise StorageValidationError("replacement target must be confirmed or current")
        _text(self.reason, "reason", maximum=500)
        _aware(self.transitioned_at, "transitioned_at")
        for name in ("replaced_belief_confidence", "replacement_belief_confidence"):
            object.__setattr__(
                self,
                name,
                _confidence(getattr(self, name), name, nullable=True),
            )


@dataclass(frozen=True)
class TransitionResult:
    transition: LifecycleTransitionRecord
    version: ClaimVersionRecord
    replayed: bool


@dataclass(frozen=True)
class CorrectionResult:
    replaced: TransitionResult
    replacement: TransitionResult
    replayed: bool


@dataclass(frozen=True)
class TemporalQuery:
    user_id: str
    transaction_as_of: datetime
    valid_at: date | datetime | None = None
    statuses: frozenset[str] = ACCEPTED_STATUSES

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _aware(self.transaction_as_of, "transaction_as_of")
        if not isinstance(self.statuses, frozenset) or not self.statuses:
            raise StorageValidationError("statuses must be a non-empty frozenset")
        if not self.statuses <= LIFECYCLE_STATUSES:
            raise StorageValidationError("statuses contain an invalid lifecycle status")
        if self.valid_at is not None:
            if isinstance(self.valid_at, datetime):
                _aware(self.valid_at, "valid_at")
            elif type(self.valid_at) is not date:
                raise StorageValidationError("valid_at must be a date or aware datetime")


@dataclass(frozen=True)
class VisibleEvidence:
    source_id: str
    span_id: str
    message_id: str | None
    speaker_id: str
    quote: str


@dataclass(frozen=True)
class TemporalClaim:
    claim: ClaimRecord
    version: ClaimVersionRecord
    evidence: tuple[VisibleEvidence, ...]
