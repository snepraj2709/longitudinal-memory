"""Transactional ingestion, worker leasing, persistence, and source deletion."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta
from typing import Iterable

from psycopg.rows import dict_row

from storage.contracts import (
    ClaimExtractionRecord,
    ClaimRecord,
    ClaimVersionRecord,
    ProcessingAttemptRecord,
    ProcessingOutboxRecord,
    SourceTombstoneRecord,
    StorageValidationError,
)
from storage.repository import StorageConflictError, StorageRepository

from .contracts import (
    ClaimWrite,
    CompletionResult,
    DeleteResult,
    EnqueueResult,
    IngestRequest,
    IngestResult,
    IngestionConflict,
    IngestionError,
    NONRETRYABLE_FAILURE_CODES,
    attempt_id,
    claim_version_id,
    leased_attempt,
    outbox_id,
)


class IngestionService:
    """Keep ingestion and derived writes atomic on a caller-owned connection."""

    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._repository = StorageRepository(connection)

    def ingest(self, request: IngestRequest) -> IngestResult:
        source = request.source
        with self._connection.transaction():
            self._lock("ingest", source.user_id, source.idempotency_key)
            self._require_user_and_extractor(
                source.user_id, request.extraction_version_id
            )
            tombstone = self._connection.execute(
                """
                SELECT 1 FROM source_tombstones
                WHERE user_id = %s AND (source_id = %s OR idempotency_key = %s)
                """,
                (source.user_id, source.source_id, source.idempotency_key),
            ).fetchone()
            if tombstone is not None:
                raise IngestionConflict("source_tombstoned", "source")
            by_key = self._repository.get_source_event_by_idempotency(
                source.user_id, source.idempotency_key
            )
            by_id = self._repository.get_source_event(
                source.user_id, source.source_id
            )
            if by_key is not None or by_id is not None:
                if by_key == source and by_id == source:
                    return IngestResult(source.source_id, None, False)
                raise IngestionConflict("source_conflict", "source")
            try:
                self._repository.insert_source_event(source)
                attempt = self._pending_attempt(
                    source.user_id,
                    source.source_id,
                    request.extraction_version_id,
                    1,
                    source.ingested_at,
                )
                self._repository.insert_processing_attempt(attempt)
                self._insert_outbox(
                    source.user_id,
                    "source_ingested",
                    source.source_id,
                    f"source_ingested:{source.source_id}",
                    {
                        "source_id": source.source_id,
                        "attempt_id": attempt.attempt_id,
                        "extraction_version_id": request.extraction_version_id,
                    },
                    source.ingested_at,
                )
            except StorageConflictError as error:
                raise IngestionConflict("stable_id_conflict", "source") from error
            return IngestResult(source.source_id, attempt.attempt_id, True)

    def enqueue_reprocessing(
        self,
        user_id: str,
        source_id: str,
        extraction_version_id: str,
        requested_at: datetime,
    ) -> EnqueueResult:
        _aware(requested_at, "requested_at")
        with self._connection.transaction():
            self._lock("source", user_id, source_id)
            self._require_user_and_extractor(user_id, extraction_version_id)
            if self._repository.get_source_event(user_id, source_id) is None:
                raise IngestionError("source_not_found", "source")
            rows = self._connection.execute(
                """
                SELECT attempt_id, attempt_number, state, retryable
                FROM processing_attempts
                WHERE user_id = %s AND source_id = %s AND extraction_version_id = %s
                ORDER BY attempt_number DESC
                FOR UPDATE
                """,
                (user_id, source_id, extraction_version_id),
            ).fetchall()
            if rows and rows[0][2] in ("pending", "running", "succeeded"):
                return EnqueueResult(rows[0][0], False)
            if rows and not rows[0][3]:
                raise IngestionConflict("attempt_not_retryable", "attempt")
            number = 1 if not rows else rows[0][1] + 1
            attempt = self._pending_attempt(
                user_id, source_id, extraction_version_id, number, requested_at
            )
            self._repository.insert_processing_attempt(attempt)
            return EnqueueResult(attempt.attempt_id, True)

    def lease_next(
        self,
        lease_owner: str,
        leased_at: datetime,
        lease_duration: timedelta,
    ) -> ProcessingAttemptRecord | None:
        if not isinstance(lease_owner, str) or not lease_owner.strip():
            raise StorageValidationError("lease_owner must be non-empty text")
        _aware(leased_at, "leased_at")
        if not isinstance(lease_duration, timedelta) or lease_duration <= timedelta(0):
            raise StorageValidationError("lease_duration must be positive")
        with self._connection.transaction():
            self._recover_expired_locked(leased_at)
            with self._connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    WITH candidate AS (
                        SELECT user_id, attempt_id
                        FROM processing_attempts
                        WHERE state = 'pending' AND started_at <= %s
                        ORDER BY started_at, user_id, source_id, extraction_version_id,
                                 attempt_number
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE processing_attempts AS attempt
                    SET state = 'running', lease_owner = %s, lease_expires_at = %s
                    FROM candidate
                    WHERE attempt.user_id = candidate.user_id
                      AND attempt.attempt_id = candidate.attempt_id
                    RETURNING attempt.*
                    """,
                    (leased_at, lease_owner, leased_at + lease_duration),
                )
                row = cursor.fetchone()
            return None if row is None else leased_attempt(ProcessingAttemptRecord(**row))

    def recover_expired_leases(self, recovered_at: datetime) -> int:
        _aware(recovered_at, "recovered_at")
        with self._connection.transaction():
            return self._recover_expired_locked(recovered_at)

    def fail_attempt(
        self,
        user_id: str,
        attempt_id_value: str,
        lease_owner: str,
        error_code: str,
        failed_at: datetime,
        *,
        retryable: bool,
    ) -> ProcessingAttemptRecord:
        _aware(failed_at, "failed_at")
        if not isinstance(error_code, str) or not error_code.strip():
            raise StorageValidationError("error_code must be non-empty text")
        if error_code in NONRETRYABLE_FAILURE_CODES:
            retryable = False
        with self._connection.transaction():
            row = self._connection.execute(
                """
                UPDATE processing_attempts
                SET state = 'failed', completed_at = %s,
                    sanitized_error_code = %s,
                    sanitized_error_metadata = NULL,
                    lease_owner = NULL, lease_expires_at = NULL,
                    retryable = %s
                WHERE user_id = %s AND attempt_id = %s
                  AND state = 'running' AND lease_owner = %s
                RETURNING attempt_id
                """,
                (
                    failed_at,
                    error_code,
                    retryable,
                    user_id,
                    attempt_id_value,
                    lease_owner,
                ),
            ).fetchone()
            if row is None:
                raise IngestionConflict("attempt_state_conflict", "attempt")
            record = self._repository.get_processing_attempt(
                user_id, attempt_id_value
            )
            assert record is not None
            return record

    def complete_attempt(
        self,
        user_id: str,
        attempt_id_value: str,
        lease_owner: str,
        completed_at: datetime,
        writes: Iterable[ClaimWrite],
    ) -> CompletionResult:
        _aware(completed_at, "completed_at")
        claim_writes = tuple(writes)
        claim_ids = tuple(item.claim.claim_id for item in claim_writes)
        if len(claim_ids) != len(set(claim_ids)):
            raise StorageValidationError("completion claim IDs must be unique")
        with self._connection.transaction():
            attempt = self._attempt_for_update(user_id, attempt_id_value)
            if attempt.state == "succeeded":
                stored = self._connection.execute(
                    """
                    SELECT claim_id FROM claim_extractions
                    WHERE user_id = %s AND attempt_id = %s
                    ORDER BY claim_id
                    """,
                    (user_id, attempt_id_value),
                ).fetchall()
                return CompletionResult(
                    attempt_id_value, tuple(row[0] for row in stored), ()
                )
            if (
                attempt.state != "running"
                or attempt.lease_owner != lease_owner
                or attempt.lease_expires_at is None
                or attempt.lease_expires_at <= completed_at
            ):
                raise IngestionConflict("attempt_state_conflict", "attempt")
            source = self._repository.get_source_event(user_id, attempt.source_id)
            if source is None:
                raise IngestionError("source_not_found", "source")
            self._lock("source", user_id, source.source_id)
            created: list[str] = []
            try:
                for write in claim_writes:
                    self._validate_write(write, attempt, source.raw_content)
                    for span in write.spans:
                        self._repository.insert_source_span(span)
                    existing = self._repository.get_claim(
                        user_id, write.claim.claim_id
                    )
                    if existing is None:
                        self._repository.insert_claim(write.claim)
                        self._repository.insert_claim_version(
                            ClaimVersionRecord(
                                claim_version_id(user_id, write.claim.claim_id),
                                user_id,
                                write.claim.claim_id,
                                "candidate",
                                completed_at,
                                valid_from_date=write.claim.valid_from_date,
                                valid_from_timestamp=write.claim.valid_from_timestamp,
                                valid_to_date=write.claim.valid_to_date,
                                valid_to_timestamp=write.claim.valid_to_timestamp,
                                time_precision=write.claim.time_precision,
                            )
                        )
                        created.append(write.claim.claim_id)
                    elif not _same_claim(existing, write.claim):
                        raise IngestionConflict("stable_id_conflict", "claim")
                    for evidence in write.evidence:
                        self._repository.insert_evidence_link(evidence)
                    self._repository.insert_claim_extraction(
                        ClaimExtractionRecord(
                            user_id,
                            write.claim.claim_id,
                            attempt.source_id,
                            attempt.extraction_version_id,
                            attempt.attempt_id,
                            completed_at,
                        )
                    )
                self._insert_outbox(
                    user_id,
                    "claims_changed",
                    attempt.source_id,
                    f"claims_changed:{attempt.source_id}:{attempt.extraction_version_id}",
                    {
                        "source_id": attempt.source_id,
                        "extraction_version_id": attempt.extraction_version_id,
                        "claim_ids": list(claim_ids),
                    },
                    completed_at,
                )
            except StorageConflictError as error:
                raise IngestionConflict("stable_id_conflict", "persistence") from error
            updated = self._connection.execute(
                """
                UPDATE processing_attempts
                SET state = 'succeeded', completed_at = %s,
                    lease_owner = NULL, lease_expires_at = NULL, retryable = false
                WHERE user_id = %s AND attempt_id = %s
                  AND state = 'running' AND lease_owner = %s
                RETURNING 1
                """,
                (completed_at, user_id, attempt_id_value, lease_owner),
            ).fetchone()
            if updated is None:
                raise IngestionConflict("attempt_state_conflict", "attempt")
            return CompletionResult(attempt_id_value, claim_ids, tuple(created))

    def delete_source(
        self, user_id: str, source_id: str, deleted_at: datetime
    ) -> DeleteResult:
        _aware(deleted_at, "deleted_at")
        with self._connection.transaction():
            self._lock("source", user_id, source_id)
            source = self._repository.get_source_event(user_id, source_id)
            if source is None:
                if self._repository.get_tombstone(user_id, source_id) is not None:
                    return DeleteResult(source_id, False, (), ())
                raise IngestionError("source_not_found", "source")
            affected = tuple(
                row[0]
                for row in self._connection.execute(
                    """
                    SELECT DISTINCT evidence.claim_id
                    FROM evidence_links AS evidence
                    JOIN source_spans AS span
                      ON span.user_id = evidence.user_id
                     AND span.span_id = evidence.span_id
                    WHERE span.user_id = %s AND span.source_id = %s
                    ORDER BY evidence.claim_id
                    """,
                    (user_id, source_id),
                ).fetchall()
            )
            self._remove_conflicts_for_source(user_id, source_id, deleted_at)
            self._connection.execute(
                """
                DELETE FROM evidence_links
                WHERE user_id = %s AND span_id IN (
                    SELECT span_id FROM source_spans
                    WHERE user_id = %s AND source_id = %s
                )
                """,
                (user_id, user_id, source_id),
            )
            self._connection.execute(
                "DELETE FROM claim_extractions WHERE user_id = %s AND source_id = %s",
                (user_id, source_id),
            )
            self._connection.execute(
                "DELETE FROM source_spans WHERE user_id = %s AND source_id = %s",
                (user_id, source_id),
            )
            retired: list[str] = []
            for claim_id_value in affected:
                remaining = self._connection.execute(
                    "SELECT count(*) FROM evidence_links WHERE user_id = %s AND claim_id = %s",
                    (user_id, claim_id_value),
                ).fetchone()[0]
                if remaining:
                    self._insert_outbox(
                        user_id,
                        "claim_recompute_required",
                        claim_id_value,
                        f"claim_recompute_required:{claim_id_value}:{source_id}",
                        {"claim_id": claim_id_value, "deleted_source_id": source_id},
                        deleted_at,
                    )
                    continue
                self._connection.execute(
                    "DELETE FROM claim_extractions WHERE user_id = %s AND claim_id = %s",
                    (user_id, claim_id_value),
                )
                self._connection.execute(
                    """
                    DELETE FROM lifecycle_transitions
                    WHERE user_id = %s
                      AND (claim_id = %s OR replacement_claim_id = %s)
                    """,
                    (user_id, claim_id_value, claim_id_value),
                )
                self._connection.execute(
                    """
                    DELETE FROM processing_outbox
                    WHERE user_id = %s AND event_type = 'claim_lifecycle_changed'
                      AND (
                          aggregate_id = %s
                          OR payload ->> 'claim_id' = %s
                          OR payload ->> 'replaced_claim_id' = %s
                          OR payload ->> 'replacement_claim_id' = %s
                      )
                    """,
                    (
                        user_id,
                        claim_id_value,
                        claim_id_value,
                        claim_id_value,
                        claim_id_value,
                    ),
                )
                self._connection.execute(
                    "DELETE FROM claim_versions WHERE user_id = %s AND claim_id = %s",
                    (user_id, claim_id_value),
                )
                self._connection.execute(
                    "DELETE FROM claims WHERE user_id = %s AND claim_id = %s",
                    (user_id, claim_id_value),
                )
                retired.append(claim_id_value)
            self._connection.execute(
                """
                DELETE FROM processing_outbox
                WHERE user_id = %s AND aggregate_id = %s
                  AND event_type IN ('source_ingested', 'claims_changed')
                """,
                (user_id, source_id),
            )
            self._connection.execute(
                "DELETE FROM processing_attempts WHERE user_id = %s AND source_id = %s",
                (user_id, source_id),
            )
            self._repository.insert_tombstone(
                SourceTombstoneRecord(
                    user_id,
                    source_id,
                    source.idempotency_key,
                    source.content_hash,
                    deleted_at,
                )
            )
            self._connection.execute(
                "DELETE FROM source_events WHERE user_id = %s AND source_id = %s",
                (user_id, source_id),
            )
            self._insert_outbox(
                user_id,
                "source_deleted",
                source_id,
                f"source_deleted:{source_id}",
                {
                    "source_id": source_id,
                    "affected_claim_ids": list(affected),
                    "retired_claim_ids": retired,
                },
                deleted_at,
            )
            return DeleteResult(
                source_id, True, affected, tuple(retired)
            )

    def _remove_conflicts_for_source(
        self, user_id: str, source_id: str, deleted_at: datetime
    ) -> None:
        rows = self._connection.execute(
            """
            SELECT DISTINCT decision.decision_id, decision.pair_id,
                            decision.left_claim_id, decision.right_claim_id
            FROM conflict_decision_evidence AS cited
            JOIN source_spans AS span
              ON span.user_id = cited.user_id
             AND span.span_id = cited.span_id
            JOIN conflict_decisions AS decision
              ON decision.user_id = cited.user_id
             AND decision.decision_id = cited.decision_id
            WHERE span.user_id = %s AND span.source_id = %s
            ORDER BY decision.pair_id, decision.decision_id
            """,
            (user_id, source_id),
        ).fetchall()
        decision_ids = tuple(row[0] for row in rows)
        if not decision_ids:
            return
        self._connection.execute(
            "DELETE FROM claim_relations WHERE user_id = %s AND decision_id = ANY(%s)",
            (user_id, list(decision_ids)),
        )
        self._connection.execute(
            """
            DELETE FROM conflict_decision_evidence
            WHERE user_id = %s AND decision_id = ANY(%s)
            """,
            (user_id, list(decision_ids)),
        )
        self._connection.execute(
            "DELETE FROM conflict_decisions WHERE user_id = %s AND decision_id = ANY(%s)",
            (user_id, list(decision_ids)),
        )
        pairs = {
            (pair_id, left_claim_id, right_claim_id)
            for _, pair_id, left_claim_id, right_claim_id in rows
        }
        for pair_id, left_claim_id, right_claim_id in sorted(pairs):
            counts = self._connection.execute(
                """
                SELECT claim_id, count(*)
                FROM evidence_links AS evidence
                JOIN source_spans AS span
                  ON span.user_id = evidence.user_id
                 AND span.span_id = evidence.span_id
                WHERE evidence.user_id = %s
                  AND evidence.claim_id = ANY(%s)
                  AND span.source_id <> %s
                GROUP BY claim_id
                """,
                (user_id, [left_claim_id, right_claim_id], source_id),
            ).fetchall()
            surviving = {claim_id for claim_id, count in counts if count > 0}
            if surviving != {left_claim_id, right_claim_id}:
                continue
            dedupe_key = f"conflict_recompute_required:{pair_id}:{source_id}"
            self._insert_outbox(
                user_id,
                "conflict_recompute_required",
                pair_id,
                dedupe_key,
                {
                    "pair_id": pair_id,
                    "left_claim_id": left_claim_id,
                    "right_claim_id": right_claim_id,
                    "deleted_source_id": source_id,
                },
                deleted_at,
            )

    def _recover_expired_locked(self, recovered_at: datetime) -> int:
        rows = self._connection.execute(
            """
            SELECT user_id, source_id, extraction_version_id, attempt_number
            FROM processing_attempts
            WHERE state = 'running' AND lease_expires_at <= %s
            ORDER BY lease_expires_at, user_id, source_id
            FOR UPDATE SKIP LOCKED
            """,
            (recovered_at,),
        ).fetchall()
        for user_id, source_id, extraction_version_id, number in rows:
            self._connection.execute(
                """
                UPDATE processing_attempts
                SET state = 'failed', completed_at = %s,
                    sanitized_error_code = 'lease_expired',
                    sanitized_error_metadata = NULL,
                    lease_owner = NULL, lease_expires_at = NULL,
                    retryable = true
                WHERE user_id = %s AND source_id = %s
                  AND extraction_version_id = %s AND attempt_number = %s
                """,
                (recovered_at, user_id, source_id, extraction_version_id, number),
            )
            next_attempt = self._pending_attempt(
                user_id,
                source_id,
                extraction_version_id,
                number + 1,
                recovered_at,
            )
            self._repository.insert_processing_attempt(next_attempt)
        return len(rows)

    def _validate_write(
        self,
        write: ClaimWrite,
        attempt: ProcessingAttemptRecord,
        raw_content: str,
    ) -> None:
        if write.claim.user_id != attempt.user_id:
            raise IngestionError("user_isolation_failure", "claim")
        if write.claim.extraction_version_id != attempt.extraction_version_id:
            raise IngestionError("validation_failure", "claim.extraction_version_id")
        spans = {item.span_id: item for item in write.spans}
        for span in write.spans:
            if span.user_id != attempt.user_id or span.source_id != attempt.source_id:
                raise IngestionError("provenance_failure", "span")
            if span.verbatim_quote not in raw_content:
                raise IngestionError("provenance_failure", "span.quote")
            if span.start_offset is not None and (
                raw_content[span.start_offset : span.end_offset] != span.verbatim_quote
            ):
                raise IngestionError("provenance_failure", "span.offsets")
        for evidence in write.evidence:
            span = spans[evidence.span_id]
            if evidence.user_id != span.user_id:
                raise IngestionError("user_isolation_failure", "evidence")

    def _attempt_for_update(
        self, user_id: str, attempt_id_value: str
    ) -> ProcessingAttemptRecord:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT * FROM processing_attempts
                WHERE user_id = %s AND attempt_id = %s
                FOR UPDATE
                """,
                (user_id, attempt_id_value),
            )
            row = cursor.fetchone()
        if row is None:
            raise IngestionError("attempt_not_found", "attempt")
        return ProcessingAttemptRecord(**row)

    def _pending_attempt(
        self,
        user_id: str,
        source_id: str,
        extraction_version_id: str,
        number: int,
        created_at: datetime,
    ) -> ProcessingAttemptRecord:
        return ProcessingAttemptRecord(
            attempt_id(user_id, source_id, extraction_version_id, number),
            user_id,
            source_id,
            extraction_version_id,
            number,
            "pending",
            created_at,
            retryable=True,
        )

    def _insert_outbox(
        self,
        user_id: str,
        event_type: str,
        aggregate_id: str,
        dedupe_key: str,
        payload: dict[str, object],
        created_at: datetime,
    ) -> None:
        self._repository.insert_outbox(
            ProcessingOutboxRecord(
                outbox_id(user_id, event_type, dedupe_key),
                user_id,
                event_type,
                aggregate_id,
                dedupe_key,
                payload,
                "pending",
                created_at,
            )
        )

    def _require_user_and_extractor(
        self, user_id: str, extraction_version_id: str
    ) -> None:
        if self._repository.get_user(user_id) is None:
            raise IngestionError("user_not_found", "user")
        if self._repository.get_extraction_version(extraction_version_id) is None:
            raise IngestionError("extraction_version_not_found", "extraction_version")

    def _lock(self, scope: str, user_id: str, value: str) -> None:
        self._connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{scope}:{user_id}:{value}",),
        )


def _same_claim(existing: ClaimRecord, candidate: ClaimRecord) -> bool:
    ignored = {"extraction_version_id"}
    return all(
        getattr(existing, field.name) == getattr(candidate, field.name)
        for field in fields(ClaimRecord)
        if field.name not in ignored
    )


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise StorageValidationError(f"{name} must be timezone-aware")
