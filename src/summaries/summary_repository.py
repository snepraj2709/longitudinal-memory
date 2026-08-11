"""Transaction-scoped persistence for deterministic session summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import SessionDefinition
from .summary_contracts import (
    GroundedSummaryError,
    GroundedSummaryResult,
    SummaryEvidence,
    canonical_json,
)


class SummaryPersistenceError(RuntimeError):
    """A sanitized persistence failure."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


@dataclass(frozen=True)
class SummaryPersistenceResult:
    summary_id: str | None
    created: bool
    replayed: bool
    retired: bool


class SessionSummaryRepository:
    """Store source-grounded summary versions without committing caller work."""

    def __init__(self, connection: object) -> None:
        self.connection = connection

    def load_visible_evidence(
        self,
        definition: SessionDefinition,
        transaction_as_of: datetime,
    ) -> tuple[SummaryEvidence, ...]:
        if definition.transaction_as_of != transaction_as_of:
            raise SummaryPersistenceError("stale_session_definition", "session")
        source_order = {source_id: position for position, source_id in enumerate(definition.source_ids)}
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT claim.claim_id, version.version_id AS claim_version_id,
                       claim.subject_id, claim.speaker_id, claim.predicate,
                       claim.object_json, claim.epistemic_status,
                       version.lifecycle_status,
                       COALESCE(claim.sensitivity, 'standard') AS sensitivity,
                       evidence.support_type, version.time_precision,
                       version.valid_from_date, version.valid_from_timestamp,
                       version.valid_to_date, version.valid_to_timestamp,
                       version.transaction_from, version.transaction_to,
                       source.source_id, source.source_type,
                       source.produced_at AS source_produced_at,
                       source.ingested_at AS source_ingested_at,
                       span.span_id, span.start_offset AS span_start_offset,
                       span.end_offset AS span_end_offset,
                       span.verbatim_quote AS exact_quote
                FROM evidence_links AS evidence
                JOIN source_spans AS span
                  ON span.user_id = evidence.user_id
                 AND span.span_id = evidence.span_id
                JOIN source_events AS source
                  ON source.user_id = span.user_id
                 AND source.source_id = span.source_id
                JOIN claims AS claim
                  ON claim.user_id = evidence.user_id
                 AND claim.claim_id = evidence.claim_id
                JOIN claim_versions AS version
                  ON version.user_id = claim.user_id
                 AND version.claim_id = claim.claim_id
                WHERE evidence.user_id = %s
                  AND source.source_id = ANY(%s)
                  AND source.ingested_at <= %s
                  AND version.transaction_from <= %s
                  AND (version.transaction_to IS NULL OR %s < version.transaction_to)
                ORDER BY source.produced_at,
                         array_position(%s::text[], source.source_id),
                         span.start_offset NULLS FIRST,
                         claim.claim_id, span.span_id, evidence.support_type
                """,
                (
                    definition.user_id,
                    list(definition.source_ids),
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                    list(definition.source_ids),
                ),
            )
            rows = tuple(cursor.fetchall())
        linked = self._dispute_links(
            definition.user_id,
            tuple(sorted({row["claim_id"] for row in rows})),
            transaction_as_of,
        )
        evidence: list[SummaryEvidence] = []
        for row in rows:
            claim_id = row["claim_id"]
            if row["lifecycle_status"] == "disputed":
                links = linked.get(claim_id, ())
                if not links:
                    continue
            else:
                links = ()
            evidence.append(
                SummaryEvidence(
                    evidence_id=_stable_id(
                        "summary_evidence",
                        definition.user_id,
                        claim_id,
                        row["claim_version_id"],
                        row["span_id"],
                        row["support_type"],
                    ),
                    user_id=definition.user_id,
                    session_definition_id=definition.definition_id,
                    claim_id=claim_id,
                    claim_version_id=row["claim_version_id"],
                    subject_id=row["subject_id"],
                    speaker_id=row["speaker_id"],
                    predicate=row["predicate"],
                    object_json=row["object_json"],
                    epistemic_status=row["epistemic_status"],
                    lifecycle_status=row["lifecycle_status"],
                    sensitivity=row["sensitivity"],
                    support_type=row["support_type"],
                    time_precision=row["time_precision"],
                    valid_from_date=row["valid_from_date"],
                    valid_from_timestamp=row["valid_from_timestamp"],
                    valid_to_date=row["valid_to_date"],
                    valid_to_timestamp=row["valid_to_timestamp"],
                    transaction_from=row["transaction_from"],
                    transaction_to=row["transaction_to"],
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    source_produced_at=row["source_produced_at"],
                    source_ingested_at=row["source_ingested_at"],
                    source_order=source_order[row["source_id"]],
                    span_id=row["span_id"],
                    span_start_offset=row["span_start_offset"],
                    span_end_offset=row["span_end_offset"],
                    exact_quote=row["exact_quote"],
                    linked_claim_ids=links,
                )
            )
        return tuple(evidence)

    def persist(
        self,
        planned: GroundedSummaryResult,
        session_source_ids: Sequence[str],
    ) -> SummaryPersistenceResult:
        source_ids = _source_ids(session_source_ids)
        with self.connection.transaction():
            return self.persist_in_transaction(planned, source_ids)

    def persist_in_transaction(
        self,
        planned: GroundedSummaryResult,
        session_source_ids: Sequence[str],
    ) -> SummaryPersistenceResult:
        source_ids = _source_ids(session_source_ids)
        request = planned.request
        summary = planned.summary
        self._lock(request.user_id, request.session_definition_id, request.renderer_version)
        self._validate_sources(request.user_id, source_ids, request.transaction_as_of)
        self._validate_evidence(planned)
        keyed = self.connection.execute(
            """
            SELECT summary_id FROM session_summaries
            WHERE user_id = %s AND idempotency_key = %s
            FOR UPDATE
            """,
            (request.user_id, request.idempotency_key),
        ).fetchone()
        if keyed is not None:
            if keyed[0] != summary.summary_id or not self._matches(
                keyed[0], planned, source_ids, exact_request=True
            ):
                raise SummaryPersistenceError("idempotency_drift", "idempotency_key")
            return SummaryPersistenceResult(keyed[0], False, True, False)
        current = self.connection.execute(
            """
            SELECT summary_id, input_snapshot_sha256, transaction_from
            FROM session_summaries
            WHERE user_id = %s AND session_definition_id = %s
              AND renderer_version = %s AND transaction_to IS NULL
            FOR UPDATE
            """,
            (
                request.user_id,
                request.session_definition_id,
                request.renderer_version,
            ),
        ).fetchone()
        if not summary.observed_facts:
            retired = self._close_current(current, planned)
            return SummaryPersistenceResult(None, False, False, retired)
        if current is not None and current[1] == planned.input_snapshot_sha256:
            if not self._matches(current[0], planned, source_ids, exact_request=False):
                raise SummaryPersistenceError("snapshot_drift", "summary")
            return SummaryPersistenceResult(current[0], False, False, False)
        self._close_current(current, planned)
        self._insert(planned, source_ids)
        return SummaryPersistenceResult(summary.summary_id, True, False, False)

    def retire_missing_current(
        self,
        user_id: str,
        renderer_version: str,
        current_definition_ids: Sequence[str],
        retired_at: datetime,
    ) -> int:
        definitions = tuple(sorted(set(current_definition_ids)))
        rows = self.connection.execute(
            """
            SELECT summary_id, transaction_from
            FROM session_summaries
            WHERE user_id = %s AND renderer_version = %s
              AND transaction_to IS NULL
              AND NOT (session_definition_id = ANY(%s))
            FOR UPDATE
            """,
            (user_id, renderer_version, list(definitions)),
        ).fetchall()
        if any(retired_at <= row[1] for row in rows):
            raise SummaryPersistenceError("stale_summary_read", "transaction_as_of")
        if rows:
            self.connection.execute(
                """
                UPDATE session_summaries SET transaction_to = %s
                WHERE user_id = %s AND summary_id = ANY(%s)
                  AND transaction_to IS NULL
                """,
                (retired_at, user_id, [row[0] for row in rows]),
            )
        return len(rows)

    def retire_definition(
        self,
        user_id: str,
        session_definition_id: str,
        renderer_version: str,
        retired_at: datetime,
    ) -> bool:
        self._lock(user_id, session_definition_id, renderer_version)
        row = self.connection.execute(
            """
            SELECT summary_id, transaction_from
            FROM session_summaries
            WHERE user_id = %s AND session_definition_id = %s
              AND renderer_version = %s AND transaction_to IS NULL
            FOR UPDATE
            """,
            (user_id, session_definition_id, renderer_version),
        ).fetchone()
        if row is None:
            return False
        if retired_at <= row[1]:
            raise SummaryPersistenceError("stale_summary_read", "transaction_as_of")
        updated = self.connection.execute(
            """
            UPDATE session_summaries SET transaction_to = %s
            WHERE user_id = %s AND summary_id = %s AND transaction_to IS NULL
            RETURNING summary_id
            """,
            (retired_at, user_id, row[0]),
        ).fetchone()
        if updated is None:
            raise SummaryPersistenceError("summary_concurrency_conflict", "summary")
        return True

    def _insert(
        self,
        planned: GroundedSummaryResult,
        source_ids: tuple[str, ...],
    ) -> None:
        request = planned.request
        summary = planned.summary
        try:
            self.connection.execute(
                """
                INSERT INTO session_summaries (
                    summary_id, user_id, session_definition_id,
                    session_membership_sha256, renderer_version,
                    idempotency_key, input_snapshot_sha256, summary_text,
                    valid_time_kind, valid_time_start_date, valid_time_end_date,
                    valid_time_start_timestamp, valid_time_end_timestamp,
                    contains_sensitive, transaction_from, transaction_to
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, NULL
                )
                """,
                (
                    summary.summary_id,
                    request.user_id,
                    request.session_definition_id,
                    request.session_membership_sha256,
                    request.renderer_version,
                    request.idempotency_key,
                    planned.input_snapshot_sha256,
                    summary.summary_text,
                    summary.valid_time_kind,
                    summary.valid_time_start_date,
                    summary.valid_time_end_date,
                    summary.valid_time_start_timestamp,
                    summary.valid_time_end_timestamp,
                    summary.contains_sensitive,
                    request.transaction_as_of,
                ),
            )
            for source_order, source_id in enumerate(source_ids):
                self.connection.execute(
                    """
                    INSERT INTO session_summary_sources (
                        user_id, summary_id, source_id, source_order
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (request.user_id, summary.summary_id, source_id, source_order),
                )
            statements = (*summary.observed_facts, *summary.unresolved_questions)
            for statement_order, statement in enumerate(statements):
                self.connection.execute(
                    """
                    INSERT INTO session_summary_statements (
                        statement_id, user_id, summary_id, statement_kind,
                        lifecycle_view, statement_text, claim_ids,
                        statement_order, contains_sensitive
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        statement.statement_id,
                        request.user_id,
                        summary.summary_id,
                        statement.statement_kind,
                        statement.lifecycle_view,
                        statement.text,
                        Jsonb(list(statement.claim_ids)),
                        statement_order,
                        statement.sensitive,
                    ),
                )
                for evidence_order, item in enumerate(statement.evidence):
                    self.connection.execute(
                        """
                        INSERT INTO session_summary_statement_evidence (
                            user_id, summary_id, statement_id, evidence_id,
                            claim_id, claim_version_id, source_id, span_id,
                            support_type, evidence_order
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            request.user_id,
                            summary.summary_id,
                            statement.statement_id,
                            item.evidence_id,
                            item.claim_id,
                            item.claim_version_id,
                            item.source_id,
                            item.span_id,
                            item.support_type,
                            evidence_order,
                        ),
                    )
        except Exception as error:
            if isinstance(error, SummaryPersistenceError):
                raise
            raise SummaryPersistenceError("summary_write_failed", "persistence") from error

    def _close_current(
        self,
        current: tuple[object, ...] | None,
        planned: GroundedSummaryResult,
    ) -> bool:
        if current is None:
            return False
        retired_at = planned.request.transaction_as_of
        if retired_at <= current[2]:
            raise SummaryPersistenceError("stale_summary_read", "transaction_as_of")
        row = self.connection.execute(
            """
            UPDATE session_summaries SET transaction_to = %s
            WHERE user_id = %s AND summary_id = %s AND transaction_to IS NULL
            RETURNING summary_id
            """,
            (retired_at, planned.request.user_id, current[0]),
        ).fetchone()
        if row is None:
            raise SummaryPersistenceError("summary_concurrency_conflict", "summary")
        return True

    def _matches(
        self,
        summary_id: str,
        planned: GroundedSummaryResult,
        source_ids: tuple[str, ...],
        *,
        exact_request: bool,
    ) -> bool:
        request = planned.request
        summary = planned.summary
        row = self.connection.execute(
            """
            SELECT session_definition_id, session_membership_sha256,
                   renderer_version, idempotency_key, input_snapshot_sha256,
                   summary_text, valid_time_kind, valid_time_start_date,
                   valid_time_end_date, valid_time_start_timestamp,
                   valid_time_end_timestamp, contains_sensitive,
                   transaction_from
            FROM session_summaries
            WHERE user_id = %s AND summary_id = %s
            """,
            (request.user_id, summary_id),
        ).fetchone()
        if row is None:
            return False
        expected = (
            request.session_definition_id,
            request.session_membership_sha256,
            request.renderer_version,
            request.idempotency_key if exact_request else row[3],
            planned.input_snapshot_sha256,
            summary.summary_text,
            summary.valid_time_kind,
            summary.valid_time_start_date,
            summary.valid_time_end_date,
            summary.valid_time_start_timestamp,
            summary.valid_time_end_timestamp,
            summary.contains_sensitive,
            request.transaction_as_of if exact_request else row[12],
        )
        if row != expected:
            return False
        stored_sources = tuple(
            value[0]
            for value in self.connection.execute(
                """
                SELECT source_id FROM session_summary_sources
                WHERE user_id = %s AND summary_id = %s
                ORDER BY source_order
                """,
                (request.user_id, summary_id),
            ).fetchall()
        )
        if stored_sources != source_ids:
            return False
        stored_statements = self.connection.execute(
            """
            SELECT statement_id, statement_kind, lifecycle_view,
                   statement_text, claim_ids, contains_sensitive
            FROM session_summary_statements
            WHERE user_id = %s AND summary_id = %s
            ORDER BY statement_order
            """,
            (request.user_id, summary_id),
        ).fetchall()
        statements = (*summary.observed_facts, *summary.unresolved_questions)
        expected_statements = tuple(
            (
                item.statement_id,
                item.statement_kind,
                item.lifecycle_view,
                item.text,
                list(item.claim_ids),
                item.sensitive,
            )
            for item in statements
        )
        if tuple(stored_statements) != expected_statements:
            return False
        for statement in statements:
            stored_evidence = tuple(
                value
                for value in self.connection.execute(
                    """
                    SELECT evidence_id, claim_id, claim_version_id,
                           source_id, span_id, support_type
                    FROM session_summary_statement_evidence
                    WHERE user_id = %s AND summary_id = %s AND statement_id = %s
                    ORDER BY evidence_order
                    """,
                    (request.user_id, summary_id, statement.statement_id),
                ).fetchall()
            )
            expected_evidence = tuple(
                (
                    item.evidence_id,
                    item.claim_id,
                    item.claim_version_id,
                    item.source_id,
                    item.span_id,
                    item.support_type,
                )
                for item in statement.evidence
            )
            if stored_evidence != expected_evidence:
                return False
        return True

    def _validate_sources(
        self,
        user_id: str,
        source_ids: tuple[str, ...],
        transaction_as_of: datetime,
    ) -> None:
        rows = self.connection.execute(
            """
            SELECT source_id, ingested_at FROM source_events
            WHERE user_id = %s AND source_id = ANY(%s)
            """,
            (user_id, list(source_ids)),
        ).fetchall()
        visible = {source_id for source_id, ingested_at in rows if ingested_at <= transaction_as_of}
        if visible != set(source_ids):
            raise SummaryPersistenceError("stale_summary_read", "source")

    def _validate_evidence(self, planned: GroundedSummaryResult) -> None:
        seen: set[str] = set()
        for statement in (
            *planned.summary.observed_facts,
            *planned.summary.unresolved_questions,
        ):
            for item in statement.evidence:
                if item.evidence_id in seen:
                    continue
                seen.add(item.evidence_id)
                row = self.connection.execute(
                    """
                    SELECT claim.subject_id, claim.speaker_id, claim.predicate,
                           claim.object_json, claim.epistemic_status,
                           COALESCE(claim.sensitivity, 'standard'),
                           version.lifecycle_status, version.time_precision,
                           version.valid_from_date, version.valid_from_timestamp,
                           version.valid_to_date, version.valid_to_timestamp,
                           version.transaction_from, version.transaction_to,
                           source.source_type, source.produced_at, source.ingested_at,
                           span.start_offset, span.end_offset, span.verbatim_quote
                    FROM claims AS claim
                    JOIN claim_versions AS version
                      ON version.user_id = claim.user_id
                     AND version.claim_id = claim.claim_id
                    JOIN evidence_links AS evidence
                      ON evidence.user_id = claim.user_id
                     AND evidence.claim_id = claim.claim_id
                    JOIN source_spans AS span
                      ON span.user_id = evidence.user_id
                     AND span.span_id = evidence.span_id
                    JOIN source_events AS source
                      ON source.user_id = span.user_id
                     AND source.source_id = span.source_id
                    WHERE claim.user_id = %s AND claim.claim_id = %s
                      AND version.version_id = %s AND span.span_id = %s
                      AND evidence.support_type = %s AND source.source_id = %s
                    """,
                    (
                        item.user_id,
                        item.claim_id,
                        item.claim_version_id,
                        item.span_id,
                        item.support_type,
                        item.source_id,
                    ),
                ).fetchone()
                expected = (
                    item.subject_id,
                    item.speaker_id,
                    item.predicate,
                    item.object_json,
                    item.epistemic_status,
                    item.sensitivity,
                    item.lifecycle_status,
                    item.time_precision,
                    item.valid_from_date,
                    item.valid_from_timestamp,
                    item.valid_to_date,
                    item.valid_to_timestamp,
                    item.transaction_from,
                    item.transaction_to,
                    item.source_type,
                    item.source_produced_at,
                    item.source_ingested_at,
                    item.span_start_offset,
                    item.span_end_offset,
                    item.exact_quote,
                )
                if row != expected:
                    raise SummaryPersistenceError("stale_summary_read", "evidence")

    def _dispute_links(
        self,
        user_id: str,
        claim_ids: tuple[str, ...],
        transaction_as_of: datetime,
    ) -> dict[str, tuple[str, ...]]:
        if not claim_ids:
            return {}
        rows = self.connection.execute(
            """
            SELECT relation.source_claim_id, relation.target_claim_id
            FROM claim_relations AS relation
            JOIN conflict_decisions AS decision
              ON decision.user_id = relation.user_id
             AND decision.decision_id = relation.decision_id
            WHERE relation.user_id = %s
              AND relation.relation_type = 'contradicts'
              AND relation.source_claim_id = ANY(%s)
              AND relation.target_claim_id = ANY(%s)
              AND decision.transaction_as_of <= %s
            ORDER BY relation.source_claim_id, relation.target_claim_id
            """,
            (user_id, list(claim_ids), list(claim_ids), transaction_as_of),
        ).fetchall()
        graph: dict[str, set[str]] = {claim_id: {claim_id} for claim_id in claim_ids}
        for left, right in rows:
            graph[left].add(right)
            graph[right].add(left)
        result: dict[str, tuple[str, ...]] = {}
        for claim_id in claim_ids:
            pending = [claim_id]
            component: set[str] = set()
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(graph[current] - component)
            if len(component) > 1:
                linked = tuple(sorted(component))
                for member in component:
                    result[member] = linked
        return result

    def _lock(self, user_id: str, definition_id: str, renderer_version: str) -> None:
        self.connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"summary:{user_id}:{definition_id}:{renderer_version}",),
        )


def _source_ids(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise GroundedSummaryError("session source IDs are invalid")
    result = tuple(values)
    if not result or len(result) != len(set(result)) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise GroundedSummaryError("session source IDs are invalid")
    return result


def _stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(canonical_json([namespace, *values]).encode("utf-8")).hexdigest()
