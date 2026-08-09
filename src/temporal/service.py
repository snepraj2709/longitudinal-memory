"""Transaction-safe lifecycle changes and user-scoped as-of queries."""

from __future__ import annotations

from datetime import datetime

from psycopg.rows import dict_row

from ingestion.contracts import outbox_id, stable_id
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    LifecycleTransitionRecord,
    ProcessingOutboxRecord,
)
from storage.repository import StorageRepository

from .contracts import (
    CorrectionRequest,
    CorrectionResult,
    TemporalClaim,
    TemporalQuery,
    TRANSITION_MATRIX,
    TransitionRequest,
    TransitionResult,
    VisibleEvidence,
)


class TemporalError(RuntimeError):
    """A sanitized temporal failure containing no claim values."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class TemporalConflict(TemporalError):
    """A lifecycle, idempotency, or concurrency conflict."""


class TemporalService:
    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._repository = StorageRepository(connection)

    def transition(self, request: TransitionRequest) -> TransitionResult:
        with self._connection.transaction():
            self._lock_claims(request.user_id, (request.claim_id,))
            replay = self._repository.get_lifecycle_transition_by_idempotency(
                request.user_id, request.idempotency_key
            )
            if replay is not None:
                return self._replay_transition(request, replay)
            claim = self._claim(request.user_id, request.claim_id)
            version = self._open_version(request.user_id, request.claim_id)
            self._require_transition(claim, version, request.target_status)
            successor, transition = self._advance(
                claim,
                version,
                request.idempotency_key,
                request.target_status,
                request.reason,
                request.transitioned_at,
                request.belief_confidence,
                None,
            )
            self._outbox(
                request.user_id,
                request.claim_id,
                request.idempotency_key,
                request.transitioned_at,
                {"claim_id": request.claim_id, "target_status": request.target_status},
            )
            return TransitionResult(transition, successor, False)

    def correct(self, request: CorrectionRequest) -> CorrectionResult:
        secondary_key = f"{request.idempotency_key}:replacement"
        with self._connection.transaction():
            self._lock_claims(
                request.user_id,
                (request.replaced_claim_id, request.replacement_claim_id),
            )
            replay = self._repository.get_lifecycle_transition_by_idempotency(
                request.user_id, request.idempotency_key
            )
            if replay is not None:
                return self._replay_correction(request, replay, secondary_key)
            replaced_claim = self._claim(request.user_id, request.replaced_claim_id)
            replacement_claim = self._claim(request.user_id, request.replacement_claim_id)
            replaced_version = self._open_version(request.user_id, request.replaced_claim_id)
            replacement_version = self._open_version(request.user_id, request.replacement_claim_id)
            self._require_transition(replaced_claim, replaced_version, "superseded")
            if replacement_version.lifecycle_status != "candidate":
                raise TemporalConflict("replacement_not_candidate", "replacement")
            self._require_transition(
                replacement_claim,
                replacement_version,
                request.replacement_target_status,
            )
            replaced_next, replaced_transition = self._advance(
                replaced_claim,
                replaced_version,
                request.idempotency_key,
                "superseded",
                request.reason,
                request.transitioned_at,
                request.replaced_belief_confidence,
                request.replacement_claim_id,
            )
            replacement_next, replacement_transition = self._advance(
                replacement_claim,
                replacement_version,
                secondary_key,
                request.replacement_target_status,
                request.reason,
                request.transitioned_at,
                request.replacement_belief_confidence,
                None,
            )
            self._outbox(
                request.user_id,
                request.replaced_claim_id,
                request.idempotency_key,
                request.transitioned_at,
                {
                    "replaced_claim_id": request.replaced_claim_id,
                    "replacement_claim_id": request.replacement_claim_id,
                    "replacement_target_status": request.replacement_target_status,
                },
            )
            return CorrectionResult(
                TransitionResult(replaced_transition, replaced_next, False),
                TransitionResult(replacement_transition, replacement_next, False),
                False,
            )

    def query(self, request: TemporalQuery) -> tuple[TemporalClaim, ...]:
        version_ids = tuple(
            row[0]
            for row in self._connection.execute(
                """
                SELECT version.version_id
                FROM claim_versions AS version
                WHERE version.user_id = %s
                  AND version.lifecycle_status = ANY(%s)
                  AND version.transaction_from <= %s
                  AND (version.transaction_to IS NULL OR %s < version.transaction_to)
                  AND EXISTS (
                      SELECT 1
                      FROM evidence_links AS evidence
                      JOIN source_spans AS span
                        ON span.user_id = evidence.user_id
                       AND span.span_id = evidence.span_id
                      JOIN source_events AS source
                        ON source.user_id = span.user_id
                       AND source.source_id = span.source_id
                      WHERE evidence.user_id = version.user_id
                        AND evidence.claim_id = version.claim_id
                        AND source.ingested_at <= %s
                  )
                ORDER BY version.claim_id, version.transaction_from, version.version_id
                """,
                (
                    request.user_id,
                    list(sorted(request.statuses)),
                    request.transaction_as_of,
                    request.transaction_as_of,
                    request.transaction_as_of,
                ),
            ).fetchall()
        )
        results: list[TemporalClaim] = []
        for version_id in version_ids:
            version = self._repository.get_claim_version(request.user_id, version_id)
            if version is None:
                continue
            if request.valid_at is not None and not version.valid_contains(request.valid_at):
                continue
            claim = self._claim(request.user_id, version.claim_id)
            evidence = tuple(
                VisibleEvidence(*row)
                for row in self._connection.execute(
                    """
                    SELECT source.source_id, span.span_id, span.message_id,
                           span.speaker_id, span.verbatim_quote
                    FROM evidence_links AS evidence
                    JOIN source_spans AS span
                      ON span.user_id = evidence.user_id
                     AND span.span_id = evidence.span_id
                    JOIN source_events AS source
                      ON source.user_id = span.user_id
                     AND source.source_id = span.source_id
                    WHERE evidence.user_id = %s AND evidence.claim_id = %s
                      AND source.ingested_at <= %s
                    ORDER BY source.source_id, span.span_id
                    """,
                    (request.user_id, claim.claim_id, request.transaction_as_of),
                ).fetchall()
            )
            if evidence:
                results.append(TemporalClaim(claim, version, evidence))
        return tuple(results)

    def _advance(
        self,
        claim: ClaimRecord,
        version: ClaimVersionRecord,
        idempotency_key: str,
        target_status: str,
        reason: str,
        transitioned_at: datetime,
        belief_confidence: float | None,
        replacement_claim_id: str | None,
    ) -> tuple[ClaimVersionRecord, LifecycleTransitionRecord]:
        if transitioned_at <= version.transaction_from:
            raise TemporalConflict("stale_transition", "transitioned_at")
        closed = self._connection.execute(
            """
            UPDATE claim_versions SET transaction_to = %s
            WHERE user_id = %s AND version_id = %s AND transaction_to IS NULL
            RETURNING version_id
            """,
            (transitioned_at, claim.user_id, version.version_id),
        ).fetchone()
        if closed is None:
            raise TemporalConflict("version_changed", "claim_version")
        successor = ClaimVersionRecord(
            version_id=stable_id(
                "claim_version",
                claim.user_id,
                claim.claim_id,
                idempotency_key,
                target_status,
                transitioned_at.isoformat(),
            ),
            user_id=claim.user_id,
            claim_id=claim.claim_id,
            lifecycle_status=target_status,
            transaction_from=transitioned_at,
            belief_confidence=(
                version.belief_confidence
                if belief_confidence is None
                else belief_confidence
            ),
            valid_from_date=version.valid_from_date,
            valid_from_timestamp=version.valid_from_timestamp,
            valid_to_date=version.valid_to_date,
            valid_to_timestamp=version.valid_to_timestamp,
            time_precision=version.time_precision,
        )
        self._repository.insert_claim_version(successor)
        transition = LifecycleTransitionRecord(
            transition_id=stable_id(
                "lifecycle_transition",
                claim.user_id,
                claim.claim_id,
                idempotency_key,
                target_status,
                transitioned_at.isoformat(),
            ),
            user_id=claim.user_id,
            idempotency_key=idempotency_key,
            claim_id=claim.claim_id,
            from_version_id=version.version_id,
            to_version_id=successor.version_id,
            target_status=target_status,
            reason=reason,
            replacement_claim_id=replacement_claim_id,
            transitioned_at=transitioned_at,
        )
        self._repository.insert_lifecycle_transition(transition)
        return successor, transition

    def _require_transition(
        self, claim: ClaimRecord, version: ClaimVersionRecord, target: str
    ) -> None:
        if claim.memory_kind is None:
            raise TemporalConflict("memory_kind_required", "claim")
        if target not in TRANSITION_MATRIX[version.lifecycle_status]:
            raise TemporalConflict("transition_not_allowed", "target_status")
        if target == "current" and version.time_precision == "unknown":
            raise TemporalConflict("current_requires_known_valid_time", "claim_version")
        if target == "historical" and (
            version.valid_to_date is None and version.valid_to_timestamp is None
        ):
            raise TemporalConflict("historical_requires_finite_end", "claim_version")

    def _claim(self, user_id: str, claim_id: str) -> ClaimRecord:
        claim = self._repository.get_claim(user_id, claim_id)
        if claim is None:
            raise TemporalError("claim_not_found", "claim")
        return claim

    def _open_version(self, user_id: str, claim_id: str) -> ClaimVersionRecord:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT * FROM claim_versions
                WHERE user_id = %s AND claim_id = %s AND transaction_to IS NULL
                FOR UPDATE
                """,
                (user_id, claim_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise TemporalConflict("open_version_not_found", "claim_version")
        return ClaimVersionRecord(**row)

    def _replay_transition(
        self, request: TransitionRequest, transition: LifecycleTransitionRecord
    ) -> TransitionResult:
        version = self._repository.get_claim_version(
            request.user_id, transition.to_version_id
        )
        if (
            version is None
            or transition.claim_id != request.claim_id
            or transition.target_status != request.target_status
            or transition.reason != request.reason
            or transition.replacement_claim_id is not None
            or transition.transitioned_at != request.transitioned_at
        ):
            raise TemporalConflict("idempotency_drift", "idempotency_key")
        previous = self._repository.get_claim_version(
            request.user_id, transition.from_version_id
        )
        expected_confidence = (
            None if previous is None else previous.belief_confidence
        ) if request.belief_confidence is None else request.belief_confidence
        if previous is None or version.belief_confidence != expected_confidence:
            raise TemporalConflict("idempotency_drift", "idempotency_key")
        return TransitionResult(transition, version, True)

    def _replay_correction(
        self,
        request: CorrectionRequest,
        primary: LifecycleTransitionRecord,
        secondary_key: str,
    ) -> CorrectionResult:
        secondary = self._repository.get_lifecycle_transition_by_idempotency(
            request.user_id, secondary_key
        )
        first_version = self._repository.get_claim_version(
            request.user_id, primary.to_version_id
        )
        second_version = (
            None
            if secondary is None
            else self._repository.get_claim_version(request.user_id, secondary.to_version_id)
        )
        if (
            secondary is None
            or first_version is None
            or second_version is None
            or primary.claim_id != request.replaced_claim_id
            or primary.target_status != "superseded"
            or primary.replacement_claim_id != request.replacement_claim_id
            or secondary.claim_id != request.replacement_claim_id
            or secondary.target_status != request.replacement_target_status
            or primary.reason != request.reason
            or secondary.reason != request.reason
            or primary.transitioned_at != request.transitioned_at
            or secondary.transitioned_at != request.transitioned_at
        ):
            raise TemporalConflict("idempotency_drift", "idempotency_key")
        first_previous = self._repository.get_claim_version(
            request.user_id, primary.from_version_id
        )
        second_previous = self._repository.get_claim_version(
            request.user_id, secondary.from_version_id
        )
        expected_first = (
            None if first_previous is None else first_previous.belief_confidence
        ) if request.replaced_belief_confidence is None else request.replaced_belief_confidence
        expected_second = (
            None if second_previous is None else second_previous.belief_confidence
        ) if request.replacement_belief_confidence is None else request.replacement_belief_confidence
        if (
            first_previous is None
            or second_previous is None
            or first_version.belief_confidence != expected_first
            or second_version.belief_confidence != expected_second
        ):
            raise TemporalConflict("idempotency_drift", "idempotency_key")
        return CorrectionResult(
            TransitionResult(primary, first_version, True),
            TransitionResult(secondary, second_version, True),
            True,
        )

    def _outbox(
        self,
        user_id: str,
        aggregate_id: str,
        idempotency_key: str,
        created_at: datetime,
        payload: dict[str, object],
    ) -> None:
        dedupe = f"claim_lifecycle_changed:{idempotency_key}"
        self._repository.insert_outbox(
            ProcessingOutboxRecord(
                event_id=outbox_id(user_id, "claim_lifecycle_changed", dedupe),
                user_id=user_id,
                event_type="claim_lifecycle_changed",
                aggregate_id=aggregate_id,
                dedupe_key=dedupe,
                payload=payload,
                state="pending",
                created_at=created_at,
            )
        )

    def _lock_claims(self, user_id: str, claim_ids: tuple[str, ...]) -> None:
        for claim_id in sorted(set(claim_ids)):
            self._connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"lifecycle:{user_id}:{claim_id}",),
            )
