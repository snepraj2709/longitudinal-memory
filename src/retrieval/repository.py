"""Transaction-scoped persistence for deterministic retrieval index records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Sequence

from psycopg import IntegrityError
from psycopg.types.json import Jsonb

from .contracts import (
    CONTENT_RENDERER_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_VERSION,
    INDEX_VERSION,
    IndexConfig,
    IndexRecord,
    RetrievalIndexError,
    canonical_json,
    load_index_config,
)


DEFAULT_CONFIG_PATH = Path("configs/retrieval/index_v1.json")
FAILURE_CODE = re.compile(r"^[a-z0-9_]{1,64}$")


class RetrievalPersistenceError(RuntimeError):
    """A sanitized retrieval-index persistence failure."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class RetrievalPersistenceConflict(RetrievalPersistenceError):
    """An idempotency, snapshot, or stable-ID conflict."""


@dataclass(frozen=True)
class IndexBuildRequest:
    user_id: str
    index_version: str
    config_sha256: str
    transaction_as_of: datetime
    idempotency_key: str
    input_snapshot_sha256: str
    started_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        if self.index_version != INDEX_VERSION:
            raise RetrievalIndexError("index_version is unsupported")
        _sha256(self.config_sha256, "config_sha256")
        _aware(self.transaction_as_of, "transaction_as_of")
        _text(self.idempotency_key, "idempotency_key")
        _sha256(self.input_snapshot_sha256, "input_snapshot_sha256")
        _aware(self.started_at, "started_at")
        _aware(self.completed_at, "completed_at")
        if self.completed_at < self.started_at:
            raise RetrievalIndexError("completed_at cannot precede started_at")


@dataclass(frozen=True)
class IndexPersistenceResult:
    run_id: str
    record_ids: tuple[str, ...]
    created: bool
    replayed: bool
    failed: bool


class RetrievalIndexRepository:
    """Persist one user and index version atomically without exposing search."""

    def __init__(
        self,
        connection: object,
        *,
        config: IndexConfig | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.connection = connection
        self.config_path = Path(config_path)
        self.config = config or load_index_config(self.config_path)
        try:
            self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        except OSError as error:
            raise RetrievalPersistenceError("config_unreadable", "config") from error

    def persist(
        self,
        request: IndexBuildRequest,
        records: Sequence[IndexRecord],
    ) -> IndexPersistenceResult:
        ordered = self._validate(request, records)
        records_snapshot = _records_snapshot(ordered)
        run_id = _run_id(request)
        atomic_count = sum(item.record_kind == "atomic" for item in ordered)
        session_count = len(ordered) - atomic_count
        with self.connection.transaction():
            self._lock(request)
            existing = self._existing(request, run_id)
            if existing is not None:
                expected = (
                    run_id,
                    request.config_sha256,
                    request.transaction_as_of,
                    request.input_snapshot_sha256,
                    records_snapshot,
                    "succeeded",
                    atomic_count,
                    session_count,
                    len(ordered),
                    None,
                    request.started_at,
                    request.completed_at,
                )
                if existing != expected:
                    raise RetrievalPersistenceConflict("idempotency_drift", "run")
                stored_ids = self._stored_record_ids(request.user_id, run_id)
                expected_ids = tuple(item.index_record_id for item in ordered)
                if stored_ids != expected_ids:
                    raise RetrievalPersistenceConflict("incomplete_replay", "records")
                return IndexPersistenceResult(
                    run_id, expected_ids, False, True, False
                )
            try:
                self.connection.execute(
                    """
                    INSERT INTO retrieval_index_runs (
                        run_id, user_id, index_version,
                        content_renderer_version, embedding_version,
                        embedding_dimension, config_sha256, idempotency_key,
                        transaction_as_of, input_snapshot_sha256,
                        records_snapshot_sha256, status, atomic_count,
                        session_count, record_count, sanitized_failure_code,
                        started_at, completed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, 'succeeded', %s, %s, %s, NULL, %s, %s
                    )
                    """,
                    (
                        run_id,
                        request.user_id,
                        request.index_version,
                        CONTENT_RENDERER_VERSION,
                        EMBEDDING_VERSION,
                        EMBEDDING_DIMENSION,
                        request.config_sha256,
                        request.idempotency_key,
                        request.transaction_as_of,
                        request.input_snapshot_sha256,
                        records_snapshot,
                        atomic_count,
                        session_count,
                        len(ordered),
                        request.started_at,
                        request.completed_at,
                    ),
                )
                for record in ordered:
                    self._insert_record(run_id, record)
            except IntegrityError as error:
                raise RetrievalPersistenceConflict(
                    "stable_id_drift", "persistence"
                ) from error
        return IndexPersistenceResult(
            run_id,
            tuple(item.index_record_id for item in ordered),
            True,
            False,
            False,
        )

    def record_failure(
        self,
        request: IndexBuildRequest,
        sanitized_failure_code: str,
    ) -> IndexPersistenceResult:
        if not FAILURE_CODE.fullmatch(sanitized_failure_code):
            raise RetrievalIndexError("sanitized failure code is invalid")
        self._validate_request(request)
        run_id = _run_id(request)
        records_snapshot = _records_snapshot(())
        with self.connection.transaction():
            self._lock(request)
            existing = self._existing(request, run_id)
            expected = (
                run_id,
                request.config_sha256,
                request.transaction_as_of,
                request.input_snapshot_sha256,
                records_snapshot,
                "failed",
                0,
                0,
                0,
                sanitized_failure_code,
                request.started_at,
                request.completed_at,
            )
            if existing is not None:
                if existing != expected:
                    raise RetrievalPersistenceConflict("idempotency_drift", "failure")
                return IndexPersistenceResult(run_id, (), False, True, True)
            try:
                self.connection.execute(
                    """
                    INSERT INTO retrieval_index_runs (
                        run_id, user_id, index_version,
                        content_renderer_version, embedding_version,
                        embedding_dimension, config_sha256, idempotency_key,
                        transaction_as_of, input_snapshot_sha256,
                        records_snapshot_sha256, status, atomic_count,
                        session_count, record_count, sanitized_failure_code,
                        started_at, completed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, 'failed', 0, 0, 0, %s, %s, %s
                    )
                    """,
                    (
                        run_id,
                        request.user_id,
                        request.index_version,
                        CONTENT_RENDERER_VERSION,
                        EMBEDDING_VERSION,
                        EMBEDDING_DIMENSION,
                        request.config_sha256,
                        request.idempotency_key,
                        request.transaction_as_of,
                        request.input_snapshot_sha256,
                        records_snapshot,
                        sanitized_failure_code,
                        request.started_at,
                        request.completed_at,
                    ),
                )
            except IntegrityError as error:
                raise RetrievalPersistenceConflict(
                    "stable_id_drift", "failure"
                ) from error
        return IndexPersistenceResult(run_id, (), True, False, True)

    def record_ids(self, user_id: str, index_version: str) -> tuple[str, ...]:
        _text(user_id, "user_id")
        _text(index_version, "index_version")
        return tuple(
            row[0]
            for row in self.connection.execute(
                """
                SELECT index_record_id FROM retrieval_index_records
                WHERE user_id = %s AND index_version = %s
                ORDER BY index_record_id
                """,
                (user_id, index_version),
            ).fetchall()
        )

    def _validate(
        self,
        request: IndexBuildRequest,
        records: Sequence[IndexRecord],
    ) -> tuple[IndexRecord, ...]:
        self._validate_request(request)
        for record in records:
            if record.user_id != request.user_id:
                raise RetrievalPersistenceError("cross_user", "record")
        ordered = tuple(sorted(records, key=lambda item: item.index_record_id))
        if len({item.index_record_id for item in ordered}) != len(ordered):
            raise RetrievalPersistenceError("duplicate_record", "record")
        for record in ordered:
            if record.index_version != request.index_version:
                raise RetrievalPersistenceError("index_version_mismatch", "record")
            if not record.transaction_time.contains(request.transaction_as_of):
                raise RetrievalPersistenceError("stale_record", "record")
        return ordered

    def _validate_request(self, request: IndexBuildRequest) -> None:
        if request.config_sha256 != self.config_sha256:
            raise RetrievalPersistenceError("config_hash_mismatch", "config")
        if request.index_version != self.config.index_version:
            raise RetrievalPersistenceError("index_version_mismatch", "config")

    def _lock(self, request: IndexBuildRequest) -> None:
        self.connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"retrieval:{request.user_id}:{request.index_version}",),
        )

    def _existing(
        self, request: IndexBuildRequest, run_id: str
    ) -> tuple[object, ...] | None:
        rows = self.connection.execute(
            """
            SELECT run_id, config_sha256, transaction_as_of,
                   input_snapshot_sha256, records_snapshot_sha256,
                   status, atomic_count, session_count, record_count,
                   sanitized_failure_code, started_at, completed_at
            FROM retrieval_index_runs
            WHERE user_id = %s AND index_version = %s
              AND (run_id = %s OR idempotency_key = %s)
            FOR UPDATE
            """,
            (
                request.user_id,
                request.index_version,
                run_id,
                request.idempotency_key,
            ),
        ).fetchall()
        if len(rows) > 1:
            raise RetrievalPersistenceConflict("idempotency_drift", "run")
        return tuple(rows[0]) if rows else None

    def _stored_record_ids(self, user_id: str, run_id: str) -> tuple[str, ...]:
        return tuple(
            row[0]
            for row in self.connection.execute(
                """
                SELECT index_record_id FROM retrieval_index_records
                WHERE user_id = %s AND run_id = %s
                ORDER BY index_record_id
                """,
                (user_id, run_id),
            ).fetchall()
        )

    def _insert_record(self, run_id: str, record: IndexRecord) -> None:
        atomic_claim_id = (
            record.claim_lineage[0].claim_id if record.record_kind == "atomic" else None
        )
        valid = record.valid_time
        transaction = record.transaction_time
        self.connection.execute(
            """
            INSERT INTO retrieval_index_records (
                index_record_id, user_id, run_id, index_version, record_kind,
                atomic_claim_id, atomic_claim_version_id, session_summary_id,
                subject_id, speaker_id, predicate, content_text, content_sha256,
                embedding_version, embedding_dimension, embedding,
                lifecycle_statuses, memory_kind, epistemic_status,
                time_precision, valid_from_date, valid_from_timestamp,
                valid_to_date, valid_to_timestamp, transaction_from,
                transaction_to, sensitivity, contains_sensitive,
                input_snapshot_sha256
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::vector, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                record.index_record_id,
                record.user_id,
                run_id,
                record.index_version,
                record.record_kind,
                atomic_claim_id,
                record.claim_version_id,
                record.session_summary_id,
                record.subject_id,
                record.speaker_id,
                record.predicate,
                record.content_text,
                record.content_sha256,
                record.embedding_version,
                len(record.embedding),
                _vector(record.embedding),
                Jsonb(list(record.lifecycle_statuses)),
                record.memory_kind,
                record.epistemic_status,
                valid.time_precision,
                valid.valid_from_date,
                valid.valid_from_timestamp,
                valid.valid_to_date,
                valid.valid_to_timestamp,
                transaction.transaction_from,
                transaction.transaction_to,
                record.sensitivity,
                record.contains_sensitive,
                record.input_snapshot_sha256,
            ),
        )
        for item in record.claim_lineage:
            self.connection.execute(
                """
                INSERT INTO retrieval_index_claim_links (
                    user_id, index_record_id, claim_id, claim_version_id,
                    lifecycle_status, claim_order
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    item.user_id,
                    record.index_record_id,
                    item.claim_id,
                    item.claim_version_id,
                    item.lifecycle_status,
                    item.order,
                ),
            )
        for item in record.source_lineage:
            self.connection.execute(
                """
                INSERT INTO retrieval_index_source_links (
                    user_id, index_record_id, claim_id, claim_version_id,
                    source_id, span_id, support_type, source_order
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    item.user_id,
                    record.index_record_id,
                    item.claim_id,
                    item.claim_version_id,
                    item.source_id,
                    item.span_id,
                    item.support_type,
                    item.order,
                ),
            )
        for item in record.relation_lineage:
            self.connection.execute(
                """
                INSERT INTO retrieval_index_relation_links (
                    user_id, index_record_id, record_kind,
                    relation_id, source_claim_id,
                    target_claim_id, relation_type, direction, relation_order
                ) VALUES (%s, %s, 'atomic', %s, %s, %s, %s, %s, %s)
                """,
                (
                    item.user_id,
                    record.index_record_id,
                    item.relation_id,
                    item.source_claim_id,
                    item.target_claim_id,
                    item.relation_type,
                    item.direction,
                    item.order,
                ),
            )


def _records_snapshot(records: Sequence[IndexRecord]) -> str:
    return hashlib.sha256(
        canonical_json(tuple(_record_value(item) for item in records)).encode("utf-8")
    ).hexdigest()


def _record_value(record: IndexRecord) -> dict[str, object]:
    return {
        "index_record_id": record.index_record_id,
        "user_id": record.user_id,
        "index_version": record.index_version,
        "record_kind": record.record_kind,
        "claim_version_id": record.claim_version_id,
        "session_summary_id": record.session_summary_id,
        "subject_id": record.subject_id,
        "speaker_id": record.speaker_id,
        "predicate": record.predicate,
        "content_text": record.content_text,
        "content_sha256": record.content_sha256,
        "embedding_version": record.embedding_version,
        "embedding": record.embedding,
        "lifecycle_statuses": record.lifecycle_statuses,
        "memory_kind": record.memory_kind,
        "epistemic_status": record.epistemic_status,
        "valid_time": record.valid_time.__dict__,
        "transaction_time": record.transaction_time.__dict__,
        "sensitivity": record.sensitivity,
        "contains_sensitive": record.contains_sensitive,
        "input_snapshot_sha256": record.input_snapshot_sha256,
        "claim_lineage": tuple(item.__dict__ for item in record.claim_lineage),
        "source_lineage": tuple(item.__dict__ for item in record.source_lineage),
        "relation_lineage": tuple(item.__dict__ for item in record.relation_lineage),
    }


def _run_id(request: IndexBuildRequest) -> str:
    return hashlib.sha256(
        canonical_json(
            (
                "retrieval_index_run",
                request.user_id,
                request.index_version,
                request.idempotency_key,
            )
        ).encode("utf-8")
    ).hexdigest()


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(value) for value in values) + "]"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalIndexError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise RetrievalIndexError(f"{name} must be timezone-aware")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RetrievalIndexError(f"{name} must be a lowercase SHA-256")
    return value
