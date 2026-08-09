"""Canonical, user-scoped Phase 4 input claims."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from numbers import Real
from typing import Mapping, Sequence

from .atomic import AtomicExtractionValidationError, _validate_claim_records
from .contracts import AtomicClaimV1, EvidenceSpanV1
from .predicate_registry import PredicateRegistry
from .source import ExtractionSource


PHASE4_INPUT_SCHEMA_VERSION = "phase4_input_v1"
PHASE4_CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "user_id",
        "subject_id",
        "speaker_id",
        "predicate",
        "predicate_registry_version",
        "object",
        "polarity",
        "epistemic_status",
        "valid_from",
        "valid_to",
        "time_precision",
        "confidence",
        "evidence",
    }
)


class Phase4InputValidationError(ValueError):
    """Reject a complete source result when any persisted claim is unsafe."""


@dataclass(frozen=True)
class Phase4InputClaim:
    claim_id: str
    user_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    predicate_registry_version: str
    object: object
    polarity: str
    epistemic_status: str
    valid_from: str | None
    valid_to: str | None
    time_precision: str
    confidence: Real
    evidence: tuple[EvidenceSpanV1, ...]


def build_phase4_source_claims(
    source: ExtractionSource,
    claim_records: Sequence[object],
    registry: PredicateRegistry,
) -> tuple[Phase4InputClaim, ...]:
    """Validate one source result fully, then assign canonical claim IDs."""

    if not source.user_id:
        raise Phase4InputValidationError("source user_id is required")
    try:
        atomic_claims = _validate_claim_records(
            list(claim_records), source, registry=registry
        )
    except AtomicExtractionValidationError as error:
        raise Phase4InputValidationError(str(error)) from None

    claims = tuple(
        _phase4_claim(source.user_id, claim, registry.registry_version)
        for claim in atomic_claims
    )
    _require_unique_canonical_ids(claims)
    source_map = {source.source_id: source}
    for claim in claims:
        validate_phase4_claim_record(
            phase4_claim_record(claim), source_map, registry
        )
    return claims


def validate_phase4_claim_file(
    records: Sequence[object],
    sources: Sequence[ExtractionSource],
    registry: PredicateRegistry,
) -> tuple[Phase4InputClaim, ...]:
    """Validate an immutable claim file without dropping invalid records."""

    source_map = {source.source_id: source for source in sources}
    if len(source_map) != len(tuple(sources)):
        raise Phase4InputValidationError("source IDs must be unique")
    claims = tuple(
        validate_phase4_claim_record(record, source_map, registry)
        for record in records
    )
    _require_unique_canonical_ids(claims)
    return claims


def validate_phase4_claim_record(
    record: object,
    sources_by_id: Mapping[str, ExtractionSource],
    registry: PredicateRegistry,
) -> Phase4InputClaim:
    """Validate one stored claim, including ownership and canonical identity."""

    if not isinstance(record, dict) or set(record) != PHASE4_CLAIM_FIELDS:
        raise Phase4InputValidationError("Phase 4 claim fields changed")
    user_id = record.get("user_id")
    registry_version = record.get("predicate_registry_version")
    if not isinstance(user_id, str) or not user_id:
        raise Phase4InputValidationError("claim user_id is required")
    if registry_version != registry.registry_version:
        raise Phase4InputValidationError("claim predicate registry version is invalid")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise Phase4InputValidationError("claim evidence is required")
    source_ids = {
        item.get("source_id")
        for item in evidence
        if isinstance(item, dict)
    }
    if len(source_ids) != 1:
        raise Phase4InputValidationError("claim evidence must belong to one source")
    source_id = next(iter(source_ids))
    source = sources_by_id.get(source_id) if isinstance(source_id, str) else None
    if source is None:
        raise Phase4InputValidationError("claim evidence references an unknown source")
    if source.user_id != user_id:
        raise Phase4InputValidationError("claim evidence crosses the user boundary")

    atomic_record = {
        "claim_id": record.get("claim_id"),
        "subject_id": record.get("subject_id"),
        "speaker_id": record.get("speaker_id"),
        "predicate": record.get("predicate"),
        "object": record.get("object"),
        "polarity": record.get("polarity"),
        "epistemic_status": record.get("epistemic_status"),
        "valid_from": record.get("valid_from"),
        "valid_to": record.get("valid_to"),
        "confidence": record.get("confidence"),
        "evidence": evidence,
    }
    try:
        atomic = _validate_claim_records(
            [atomic_record], source, registry=registry
        )[0]
    except AtomicExtractionValidationError as error:
        raise Phase4InputValidationError(str(error)) from None
    precision = derive_time_precision(atomic.valid_from, atomic.valid_to)
    if record.get("time_precision") != precision:
        raise Phase4InputValidationError("claim time_precision does not match valid time")
    expected = _phase4_claim(user_id, atomic, registry.registry_version)
    if record.get("claim_id") != expected.claim_id:
        raise Phase4InputValidationError("claim_id is not the canonical SHA-256 ID")
    return expected


def derive_time_precision(valid_from: str | None, valid_to: str | None) -> str:
    """Derive only the precision represented by the accepted boundary values."""

    values = [value for value in (valid_from, valid_to) if value is not None]
    if not values:
        return "unknown"
    kinds = {_time_kind(value) for value in values}
    if len(kinds) != 1:
        raise Phase4InputValidationError("valid-time boundaries use mixed precision")
    return next(iter(kinds))


def phase4_claim_record(claim: Phase4InputClaim) -> dict[str, object]:
    """Return the exact JSONL record handed to Phase 4."""

    return {
        "claim_id": claim.claim_id,
        "user_id": claim.user_id,
        "subject_id": claim.subject_id,
        "speaker_id": claim.speaker_id,
        "predicate": claim.predicate,
        "predicate_registry_version": claim.predicate_registry_version,
        "object": claim.object,
        "polarity": claim.polarity,
        "epistemic_status": claim.epistemic_status,
        "valid_from": claim.valid_from,
        "valid_to": claim.valid_to,
        "time_precision": claim.time_precision,
        "confidence": claim.confidence,
        "evidence": [
            {
                "source_id": item.source_id,
                "message_id": item.message_id,
                "quote": item.quote,
            }
            for item in claim.evidence
        ],
    }


def _phase4_claim(
    user_id: str,
    claim: AtomicClaimV1,
    registry_version: str,
) -> Phase4InputClaim:
    precision = derive_time_precision(claim.valid_from, claim.valid_to)
    semantic = {
        "schema_version": PHASE4_INPUT_SCHEMA_VERSION,
        "user_id": user_id,
        "subject_id": claim.subject_id,
        "speaker_id": claim.speaker_id,
        "predicate": claim.predicate,
        "predicate_registry_version": registry_version,
        "object": claim.object,
        "polarity": claim.polarity,
        "epistemic_status": claim.epistemic_status,
        "valid_from": claim.valid_from,
        "valid_to": claim.valid_to,
        "time_precision": precision,
        "confidence": claim.confidence,
        "evidence": [
            {
                "source_id": item.source_id,
                "message_id": item.message_id,
                "quote": item.quote,
            }
            for item in claim.evidence
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return Phase4InputClaim(
        claim_id=f"claim_{digest}",
        user_id=user_id,
        subject_id=claim.subject_id,
        speaker_id=claim.speaker_id,
        predicate=claim.predicate,
        predicate_registry_version=registry_version,
        object=claim.object,
        polarity=claim.polarity,
        epistemic_status=claim.epistemic_status,
        valid_from=claim.valid_from,
        valid_to=claim.valid_to,
        time_precision=precision,
        confidence=claim.confidence,
        evidence=claim.evidence,
    )


def _time_kind(value: str) -> str:
    try:
        date.fromisoformat(value)
        return "day"
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise Phase4InputValidationError("valid time is not ISO formatted") from None
        if parsed.utcoffset() is None:
            raise Phase4InputValidationError("valid-time timestamp lacks a UTC offset")
        return "timestamp"


def _require_unique_canonical_ids(claims: Sequence[Phase4InputClaim]) -> None:
    ids = [claim.claim_id for claim in claims]
    if len(ids) != len(set(ids)):
        raise Phase4InputValidationError("canonical claim IDs collide or duplicate")
