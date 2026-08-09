"""Parameterized, transaction-scoped PostgreSQL record access."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Mapping, TypeVar

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import (
    ClaimExtractionRecord,
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    ProcessingOutboxRecord,
    ProcessingAttemptRecord,
    SourceEventRecord,
    SourceSpanRecord,
    SourceTombstoneRecord,
)


class StorageConflictError(RuntimeError):
    """A stable ID or idempotency key already names different content."""


Record = TypeVar(
    "Record",
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
    ExtractionVersionRecord,
    ProcessingAttemptRecord,
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ClaimExtractionRecord,
    ProcessingOutboxRecord,
    SourceTombstoneRecord,
)

_JSON_COLUMNS = {
    "participants",
    "metadata",
    "sanitized_error_metadata",
    "object_json",
    "payload",
}


class StorageRepository:
    """Insert and read validated records without committing caller transactions."""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    def insert_user(self, record: MemoryUser) -> MemoryUser:
        return self._insert("memory_users", record, {"user_id": record.user_id})

    def get_user(self, user_id: str) -> MemoryUser | None:
        return self._get("memory_users", MemoryUser, {"user_id": user_id})

    def insert_source_event(self, record: SourceEventRecord) -> SourceEventRecord:
        return self._insert(
            "source_events",
            record,
            {"user_id": record.user_id, "source_id": record.source_id},
        )

    def get_source_event(
        self, user_id: str, source_id: str
    ) -> SourceEventRecord | None:
        return self._get(
            "source_events",
            SourceEventRecord,
            {"user_id": user_id, "source_id": source_id},
        )

    def get_source_event_by_idempotency(
        self, user_id: str, idempotency_key: str
    ) -> SourceEventRecord | None:
        return self._get(
            "source_events",
            SourceEventRecord,
            {"user_id": user_id, "idempotency_key": idempotency_key},
        )

    def insert_source_span(self, record: SourceSpanRecord) -> SourceSpanRecord:
        return self._insert(
            "source_spans",
            record,
            {"user_id": record.user_id, "span_id": record.span_id},
        )

    def get_source_span(
        self, user_id: str, span_id: str
    ) -> SourceSpanRecord | None:
        return self._get(
            "source_spans",
            SourceSpanRecord,
            {"user_id": user_id, "span_id": span_id},
        )

    def insert_extraction_version(
        self, record: ExtractionVersionRecord
    ) -> ExtractionVersionRecord:
        return self._insert(
            "extraction_versions", record, {"version_id": record.version_id}
        )

    def get_extraction_version(
        self, version_id: str
    ) -> ExtractionVersionRecord | None:
        return self._get(
            "extraction_versions", ExtractionVersionRecord, {"version_id": version_id}
        )

    def insert_processing_attempt(
        self, record: ProcessingAttemptRecord
    ) -> ProcessingAttemptRecord:
        return self._insert(
            "processing_attempts",
            record,
            {"user_id": record.user_id, "attempt_id": record.attempt_id},
        )

    def get_processing_attempt(
        self, user_id: str, attempt_id: str
    ) -> ProcessingAttemptRecord | None:
        return self._get(
            "processing_attempts",
            ProcessingAttemptRecord,
            {"user_id": user_id, "attempt_id": attempt_id},
        )

    def insert_claim(self, record: ClaimRecord) -> ClaimRecord:
        return self._insert(
            "claims",
            record,
            {"user_id": record.user_id, "claim_id": record.claim_id},
        )

    def get_claim(self, user_id: str, claim_id: str) -> ClaimRecord | None:
        return self._get(
            "claims", ClaimRecord, {"user_id": user_id, "claim_id": claim_id}
        )

    def insert_claim_version(
        self, record: ClaimVersionRecord
    ) -> ClaimVersionRecord:
        return self._insert(
            "claim_versions",
            record,
            {"user_id": record.user_id, "version_id": record.version_id},
        )

    def get_claim_version(
        self, user_id: str, version_id: str
    ) -> ClaimVersionRecord | None:
        return self._get(
            "claim_versions",
            ClaimVersionRecord,
            {"user_id": user_id, "version_id": version_id},
        )

    def insert_evidence_link(
        self, record: EvidenceLinkRecord
    ) -> EvidenceLinkRecord:
        return self._insert(
            "evidence_links",
            record,
            {
                "user_id": record.user_id,
                "claim_id": record.claim_id,
                "span_id": record.span_id,
                "support_type": record.support_type,
            },
        )

    def get_evidence_link(
        self, user_id: str, claim_id: str, span_id: str, support_type: str
    ) -> EvidenceLinkRecord | None:
        return self._get(
            "evidence_links",
            EvidenceLinkRecord,
            {
                "user_id": user_id,
                "claim_id": claim_id,
                "span_id": span_id,
                "support_type": support_type,
            },
        )

    def insert_claim_extraction(
        self, record: ClaimExtractionRecord
    ) -> ClaimExtractionRecord:
        return self._insert(
            "claim_extractions",
            record,
            {
                "user_id": record.user_id,
                "claim_id": record.claim_id,
                "source_id": record.source_id,
                "extraction_version_id": record.extraction_version_id,
            },
        )

    def get_claim_extraction(
        self,
        user_id: str,
        claim_id: str,
        source_id: str,
        extraction_version_id: str,
    ) -> ClaimExtractionRecord | None:
        return self._get(
            "claim_extractions",
            ClaimExtractionRecord,
            {
                "user_id": user_id,
                "claim_id": claim_id,
                "source_id": source_id,
                "extraction_version_id": extraction_version_id,
            },
        )

    def insert_outbox(
        self, record: ProcessingOutboxRecord
    ) -> ProcessingOutboxRecord:
        return self._insert(
            "processing_outbox",
            record,
            {"user_id": record.user_id, "event_id": record.event_id},
        )

    def get_outbox(
        self, user_id: str, event_id: str
    ) -> ProcessingOutboxRecord | None:
        return self._get(
            "processing_outbox",
            ProcessingOutboxRecord,
            {"user_id": user_id, "event_id": event_id},
        )

    def insert_tombstone(
        self, record: SourceTombstoneRecord
    ) -> SourceTombstoneRecord:
        return self._insert(
            "source_tombstones",
            record,
            {"user_id": record.user_id, "source_id": record.source_id},
        )

    def get_tombstone(
        self, user_id: str, source_id: str
    ) -> SourceTombstoneRecord | None:
        return self._get(
            "source_tombstones",
            SourceTombstoneRecord,
            {"user_id": user_id, "source_id": source_id},
        )

    def _insert(
        self, table: str, record: Record, identity: Mapping[str, object]
    ) -> Record:
        values = asdict(record)
        columns = tuple(field.name for field in fields(record))
        parameters = tuple(
            Jsonb(values[name])
            if name in _JSON_COLUMNS and values[name] is not None
            else values[name]
            for name in columns
        )
        placeholders = ", ".join(["%s"] * len(columns))
        conflict_columns = ", ".join(identity)
        self._connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) ON CONFLICT ({conflict_columns}) DO NOTHING",
            parameters,
        )
        existing = self._get(table, type(record), identity)
        if existing is None:
            raise StorageConflictError(
                f"{table} has a conflicting stable ID or idempotency key"
            )
        if existing != record:
            raise StorageConflictError(f"{table} stable ID names different content")
        return existing

    def _get(
        self, table: str, record_type: type[Record], identity: Mapping[str, object]
    ) -> Record | None:
        where = " AND ".join(f"{name} = %s" for name in identity)
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"SELECT {', '.join(field.name for field in fields(record_type))} "
                f"FROM {table} WHERE {where}",
                tuple(identity.values()),
            )
            row = cursor.fetchone()
        return None if row is None else record_type(**row)
