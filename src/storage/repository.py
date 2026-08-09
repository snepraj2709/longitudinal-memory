"""Parameterized, transaction-scoped PostgreSQL record access."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Mapping, TypeVar

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    ProcessingAttemptRecord,
    SourceEventRecord,
    SourceSpanRecord,
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
)

_JSON_COLUMNS = {
    "participants",
    "metadata",
    "sanitized_error_metadata",
    "object_json",
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
        return self._insert("source_events", record, {"source_id": record.source_id})

    def get_source_event(self, source_id: str) -> SourceEventRecord | None:
        return self._get("source_events", SourceEventRecord, {"source_id": source_id})

    def insert_source_span(self, record: SourceSpanRecord) -> SourceSpanRecord:
        return self._insert("source_spans", record, {"span_id": record.span_id})

    def get_source_span(self, span_id: str) -> SourceSpanRecord | None:
        return self._get("source_spans", SourceSpanRecord, {"span_id": span_id})

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
            "processing_attempts", record, {"attempt_id": record.attempt_id}
        )

    def get_processing_attempt(
        self, attempt_id: str
    ) -> ProcessingAttemptRecord | None:
        return self._get(
            "processing_attempts", ProcessingAttemptRecord, {"attempt_id": attempt_id}
        )

    def insert_claim(self, record: ClaimRecord) -> ClaimRecord:
        return self._insert("claims", record, {"claim_id": record.claim_id})

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        return self._get("claims", ClaimRecord, {"claim_id": claim_id})

    def insert_claim_version(
        self, record: ClaimVersionRecord
    ) -> ClaimVersionRecord:
        return self._insert(
            "claim_versions", record, {"version_id": record.version_id}
        )

    def get_claim_version(self, version_id: str) -> ClaimVersionRecord | None:
        return self._get(
            "claim_versions", ClaimVersionRecord, {"version_id": version_id}
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
