"""Typed inputs and deterministic identifiers for ingestion workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from storage.contracts import (
    ClaimRecord,
    EvidenceLinkRecord,
    ProcessingAttemptRecord,
    SourceEventRecord,
    SourceSpanRecord,
    StorageValidationError,
)


NONRETRYABLE_FAILURE_CODES = frozenset(
    {
        "validation_failure",
        "provenance_failure",
        "user_isolation_failure",
        "stable_id_conflict",
    }
)


class IngestionError(RuntimeError):
    """A sanitized ingestion failure with no source content or values."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class IngestionConflict(IngestionError):
    """An idempotency, stable-ID, tombstone, or state conflict."""


@dataclass(frozen=True)
class IngestRequest:
    source: SourceEventRecord
    extraction_version_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.extraction_version_id, str) or not self.extraction_version_id.strip():
            raise StorageValidationError("extraction_version_id must be non-empty text")


@dataclass(frozen=True)
class IngestResult:
    source_id: str
    attempt_id: str | None
    created: bool


@dataclass(frozen=True)
class EnqueueResult:
    attempt_id: str
    created: bool


@dataclass(frozen=True)
class ClaimWrite:
    claim: ClaimRecord
    spans: tuple[SourceSpanRecord, ...]
    evidence: tuple[EvidenceLinkRecord, ...]

    def __post_init__(self) -> None:
        if not self.spans or not self.evidence:
            raise StorageValidationError("claim writes require spans and evidence")
        span_ids = {item.span_id for item in self.spans}
        if len(span_ids) != len(self.spans):
            raise StorageValidationError("claim write spans must be unique")
        evidence_keys = {
            (item.claim_id, item.span_id, item.support_type)
            for item in self.evidence
        }
        if len(evidence_keys) != len(self.evidence):
            raise StorageValidationError("claim write evidence must be unique")
        if any(
            item.user_id != self.claim.user_id
            or item.claim_id != self.claim.claim_id
            or item.span_id not in span_ids
            for item in self.evidence
        ):
            raise StorageValidationError("claim write evidence references are invalid")


@dataclass(frozen=True)
class CompletionResult:
    attempt_id: str
    claim_ids: tuple[str, ...]
    created_claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeleteResult:
    source_id: str
    deleted: bool
    affected_claim_ids: tuple[str, ...]
    retired_claim_ids: tuple[str, ...]


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()}"


def attempt_id(
    user_id: str, source_id: str, extraction_version_id: str, attempt_number: int
) -> str:
    return stable_id(
        "attempt", user_id, source_id, extraction_version_id, attempt_number
    )


def claim_version_id(user_id: str, claim_id: str) -> str:
    return stable_id("claim_version", user_id, claim_id, "candidate")


def outbox_id(user_id: str, event_type: str, dedupe_key: str) -> str:
    return stable_id("outbox", user_id, event_type, dedupe_key)


def leased_attempt(record: ProcessingAttemptRecord) -> ProcessingAttemptRecord:
    """Keep a public typed boundary for leased worker records."""

    if record.state != "running":
        raise StorageValidationError("leased attempt must be running")
    return record
