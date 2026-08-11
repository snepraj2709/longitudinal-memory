"""User-first pre-search eligibility filtering for retrieval index records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Sequence

from .query_contracts import (
    EligibilityDecision,
    EligibilityResult,
    QueryPlan,
    RequestedValidTime,
)


SAFE_FAILURE = re.compile(r"^[a-z0-9_]{1,64}$")


class RetrievalQueryRepositoryError(RuntimeError):
    """A sanitized, fail-closed query-filter error."""

    def __init__(self, code: str, location: str) -> None:
        if SAFE_FAILURE.fullmatch(code) is None or SAFE_FAILURE.fullmatch(location) is None:
            raise ValueError("repository error fields must be sanitized")
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


@dataclass(frozen=True)
class _Candidate:
    index_record_id: str
    record_kind: str
    lifecycle_statuses: tuple[str, ...]
    sensitivity: str | None
    contains_sensitive: bool
    time_precision: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    transaction_visible: bool
    source_after_as_of: bool
    lineage_complete: bool
    relation_after_as_of: bool
    relation_complete: bool
    speaker_match: bool
    entity_match: bool


class RetrievalQueryRepository:
    """Select one safe index snapshot and apply deterministic eligibility rules."""

    SNAPSHOT_SQL = """
        SELECT run_id
        FROM retrieval_index_runs
        WHERE user_id = %s
          AND index_version = %s
          AND status = 'succeeded'
          AND transaction_as_of <= %s
        ORDER BY transaction_as_of DESC, run_id DESC
    """

    RECORDS_SQL = """
        WITH user_snapshot_records AS MATERIALIZED (
            SELECT record.*
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
        )
        SELECT
            record.index_record_id,
            record.record_kind,
            record.lifecycle_statuses,
            record.sensitivity,
            record.contains_sensitive,
            record.time_precision,
            record.valid_from_date,
            record.valid_from_timestamp,
            record.valid_to_date,
            record.valid_to_timestamp,
            (
                record.transaction_from <= %s
                AND (record.transaction_to IS NULL OR %s < record.transaction_to)
            ) AS transaction_visible,
            EXISTS (
                SELECT 1
                FROM retrieval_index_source_links AS source_link
                JOIN source_events AS source
                  ON source.user_id = source_link.user_id
                 AND source.source_id = source_link.source_id
                WHERE source_link.user_id = record.user_id
                  AND source_link.index_record_id = record.index_record_id
                  AND source.ingested_at > %s
            ) AS source_after_as_of,
            (
                EXISTS (
                    SELECT 1 FROM retrieval_index_claim_links AS claim_link
                    WHERE claim_link.user_id = record.user_id
                      AND claim_link.index_record_id = record.index_record_id
                )
                AND EXISTS (
                    SELECT 1 FROM retrieval_index_source_links AS source_link
                    WHERE source_link.user_id = record.user_id
                      AND source_link.index_record_id = record.index_record_id
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM retrieval_index_claim_links AS claim_link
                    WHERE claim_link.user_id = record.user_id
                      AND claim_link.index_record_id = record.index_record_id
                      AND NOT EXISTS (
                          SELECT 1
                          FROM retrieval_index_source_links AS source_link
                          JOIN source_events AS source
                            ON source.user_id = source_link.user_id
                           AND source.source_id = source_link.source_id
                          JOIN source_spans AS span
                            ON span.user_id = source_link.user_id
                           AND span.source_id = source_link.source_id
                           AND span.span_id = source_link.span_id
                          WHERE source_link.user_id = claim_link.user_id
                            AND source_link.index_record_id = claim_link.index_record_id
                            AND source_link.claim_id = claim_link.claim_id
                            AND source_link.claim_version_id = claim_link.claim_version_id
                      )
                )
            ) AS lineage_complete,
            EXISTS (
                SELECT 1
                FROM retrieval_index_relation_links AS relation_link
                JOIN claim_relations AS relation
                  ON relation.user_id = relation_link.user_id
                 AND relation.relation_id = relation_link.relation_id
                JOIN conflict_decisions AS decision
                  ON decision.user_id = relation.user_id
                 AND decision.decision_id = relation.decision_id
                WHERE relation_link.user_id = record.user_id
                  AND relation_link.index_record_id = record.index_record_id
                  AND (
                      relation.created_at > %s
                      OR decision.transaction_as_of > %s
                      OR decision.classified_at > %s
                  )
            ) AS relation_after_as_of,
            NOT EXISTS (
                SELECT 1
                FROM retrieval_index_relation_links AS relation_link
                LEFT JOIN claim_relations AS relation
                  ON relation.user_id = relation_link.user_id
                 AND relation.relation_id = relation_link.relation_id
                LEFT JOIN conflict_decisions AS decision
                  ON decision.user_id = relation.user_id
                 AND decision.decision_id = relation.decision_id
                WHERE relation_link.user_id = record.user_id
                  AND relation_link.index_record_id = record.index_record_id
                  AND (relation.relation_id IS NULL OR decision.decision_id IS NULL)
            ) AS relation_complete,
            CASE
                WHEN cardinality(%s::text[]) = 0 THEN true
                WHEN record.record_kind = 'atomic' THEN record.speaker_id = ANY(%s::text[])
                ELSE EXISTS (
                    SELECT 1
                    FROM retrieval_index_claim_links AS claim_link
                    JOIN claims AS claim
                      ON claim.user_id = claim_link.user_id
                     AND claim.claim_id = claim_link.claim_id
                    WHERE claim_link.user_id = record.user_id
                      AND claim_link.index_record_id = record.index_record_id
                      AND claim.speaker_id = ANY(%s::text[])
                )
            END AS speaker_match,
            CASE
                WHEN cardinality(%s::text[]) = 0 THEN true
                WHEN record.record_kind = 'atomic' THEN record.subject_id = ANY(%s::text[])
                ELSE EXISTS (
                    SELECT 1
                    FROM retrieval_index_claim_links AS claim_link
                    JOIN claims AS claim
                      ON claim.user_id = claim_link.user_id
                     AND claim.claim_id = claim_link.claim_id
                    WHERE claim_link.user_id = record.user_id
                      AND claim_link.index_record_id = record.index_record_id
                      AND claim.subject_id = ANY(%s::text[])
                )
            END AS entity_match
        FROM user_snapshot_records AS record
        WHERE record.record_kind = ANY(%s::text[])
        ORDER BY record.index_record_id
    """

    def __init__(self, connection: object) -> None:
        self.connection = connection

    def filter(self, plan: QueryPlan) -> EligibilityResult:
        snapshot = self.connection.execute(
            self.SNAPSHOT_SQL,
            (plan.user_id, plan.index_version, plan.as_of),
        ).fetchone()
        if snapshot is None:
            raise RetrievalQueryRepositoryError("no_safe_index_snapshot", "snapshot")
        run_id = snapshot[0]
        speakers = list(plan.speaker_ids)
        entities = list(plan.entity_ids)
        rows = self.connection.execute(
            self.RECORDS_SQL,
            (
                plan.user_id,
                plan.index_version,
                run_id,
                plan.as_of,
                plan.as_of,
                plan.as_of,
                plan.as_of,
                plan.as_of,
                plan.as_of,
                speakers,
                speakers,
                speakers,
                entities,
                entities,
                entities,
                list(plan.enabled_record_kinds),
            ),
        ).fetchall()
        decisions = tuple(self._decision(plan, _candidate(row)) for row in rows)
        return EligibilityResult(
            plan.plan_id,
            plan.user_id,
            plan.index_version,
            run_id,
            decisions,
        )

    def _decision(self, plan: QueryPlan, item: _Candidate) -> EligibilityDecision:
        reasons: set[str] = set()
        if not item.transaction_visible:
            reasons.add("transaction_hidden")
        if item.source_after_as_of:
            reasons.add("source_after_as_of")
        if not item.lineage_complete:
            reasons.add("partial_lineage")
        if item.relation_after_as_of:
            reasons.add("relation_after_as_of")
        if not item.relation_complete:
            reasons.add("stale_lineage")
        valid_reason = _valid_time_rejection(plan, item)
        if valid_reason is not None:
            reasons.add(valid_reason)
        if not item.speaker_match:
            reasons.add("speaker_mismatch")
        if not item.entity_match:
            reasons.add("entity_mismatch")
        if not set(item.lifecycle_statuses).intersection(plan.allowed_lifecycle_statuses):
            reasons.add("lifecycle_blocked")
        if item.sensitivity == "restricted":
            reasons.add("restricted")
        elif item.sensitivity == "sensitive" and not plan.allow_sensitive:
            reasons.add("sensitive_not_authorized")
        elif item.sensitivity is None and not plan.allow_unclassified_sensitivity:
            reasons.add("unclassified_sensitivity")
        if item.contains_sensitive != (item.sensitivity == "sensitive"):
            reasons.add("partial_lineage")
        ordered = tuple(sorted(reasons))
        return EligibilityDecision(
            item.index_record_id,
            not ordered,
            ordered,
            item.lifecycle_statuses,
            item.sensitivity,
            item.sensitivity is None,
        )


def _candidate(row: Sequence[object]) -> _Candidate:
    if len(row) != 17:
        raise RetrievalQueryRepositoryError("invalid_record_shape", "records")
    lifecycle = row[2]
    if not isinstance(lifecycle, list) or any(not isinstance(item, str) for item in lifecycle):
        raise RetrievalQueryRepositoryError("invalid_record_shape", "lifecycle")
    try:
        return _Candidate(
            str(row[0]),
            str(row[1]),
            tuple(lifecycle),
            row[3],
            bool(row[4]),
            str(row[5]),
            row[6],
            row[7],
            row[8],
            row[9],
            bool(row[10]),
            bool(row[11]),
            bool(row[12]),
            bool(row[13]),
            bool(row[14]),
            bool(row[15]),
            bool(row[16]),
        )
    except (TypeError, ValueError) as error:
        raise RetrievalQueryRepositoryError("invalid_record_shape", "records") from error


def _valid_time_rejection(plan: QueryPlan, item: _Candidate) -> str | None:
    requested = plan.requested_valid_time
    if requested is None and plan.primary_label == "current_state":
        requested = RequestedValidTime(kind="point", point_timestamp=plan.as_of)
    if requested is None:
        return None
    if item.time_precision in {"unknown", "mixed"}:
        return "unknown_valid_time"
    if requested.kind == "point":
        if requested.point_date is not None:
            if item.time_precision == "timestamp":
                return "valid_time_mismatch"
            return None if _point_in_interval(
                requested.point_date, item.valid_from_date, item.valid_to_date
            ) else "valid_time_mismatch"
        point = requested.point_timestamp
        assert point is not None
        if item.time_precision == "timestamp":
            return None if _point_in_interval(
                point, item.valid_from_timestamp, item.valid_to_timestamp
            ) else "valid_time_mismatch"
        return None if _point_in_interval(
            point.date(), item.valid_from_date, item.valid_to_date
        ) else "valid_time_mismatch"
    if requested.range_start_date is not None or requested.range_end_date is not None:
        if item.time_precision == "timestamp":
            return "valid_time_mismatch"
        return None if _intervals_overlap(
            item.valid_from_date,
            item.valid_to_date,
            requested.range_start_date,
            requested.range_end_date,
        ) else "valid_time_mismatch"
    return None if item.time_precision == "timestamp" and _intervals_overlap(
        item.valid_from_timestamp,
        item.valid_to_timestamp,
        requested.range_start_timestamp,
        requested.range_end_timestamp,
    ) else "valid_time_mismatch"


def _point_in_interval(value, start, end) -> bool:
    return (start is None or start <= value) and (end is None or value <= end)


def _intervals_overlap(left_start, left_end, right_start, right_end) -> bool:
    return (left_start is None or right_end is None or left_start <= right_end) and (
        right_start is None or left_end is None or right_start <= left_end
    )
