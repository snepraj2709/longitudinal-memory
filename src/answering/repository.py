"""Read-only, user-first hydration for frozen retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from retrieval.query_contracts import RequestedValidTime

from .contracts import (
    CheckedRelationReference,
    EvidenceAnchor,
    EvidencePackageBuildRequest,
    EvidencePackageError,
    EvidenceSpan,
    evidence_id,
)


@dataclass(frozen=True)
class HydratedRelation:
    reference: CheckedRelationReference
    user_id: str
    left_claim_id: str
    left_version_id: str
    left_transaction_from: datetime
    left_transaction_to: datetime | None
    left_valid_from_date: date | None
    left_valid_from_timestamp: datetime | None
    left_valid_to_date: date | None
    left_valid_to_timestamp: datetime | None
    left_time_precision: str
    right_claim_id: str
    right_version_id: str
    right_transaction_from: datetime
    right_transaction_to: datetime | None
    right_valid_from_date: date | None
    right_valid_from_timestamp: datetime | None
    right_valid_to_date: date | None
    right_valid_to_timestamp: datetime | None
    right_time_precision: str
    created_at: datetime
    decision_transaction_as_of: datetime
    classified_at: datetime


@dataclass(frozen=True)
class HydratedClaim:
    user_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: object
    polarity: str
    epistemic_status: str
    memory_kind: str | None
    lifecycle_status: str
    extraction_confidence: float
    belief_confidence: float | None
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    time_precision: str
    transaction_from: datetime
    transaction_to: datetime | None
    sensitivity: str | None
    anchors: tuple[EvidenceAnchor, ...]
    evidence: tuple[EvidenceSpan, ...]
    relations: tuple[HydratedRelation, ...]


@dataclass(frozen=True)
class HydratedSource:
    user_id: str
    source_id: str
    source_type: str
    session_id: str | None
    produced_at: datetime
    ingested_at: datetime
    content_hash: str
    raw_content: str


@dataclass(frozen=True)
class HydratedPackageInput:
    claims: tuple[HydratedClaim, ...]
    sources: tuple[HydratedSource, ...]


class EvidencePackageRepositoryError(RuntimeError):
    """A sanitized fail-closed hydration error."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class EvidencePackageRepository:
    """Hydrate accepted lineage without rerunning retrieval or reading rejected content."""

    CLAIMS_SQL = """
        WITH user_records AS MATERIALIZED (
            SELECT record.index_record_id, record.record_kind,
                   record.atomic_claim_version_id, record.session_summary_id,
                   record.lifecycle_statuses, record.content_sha256
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT record.index_record_id, record.record_kind,
               record.atomic_claim_version_id, record.session_summary_id,
               record.lifecycle_statuses, record.content_sha256,
               link.claim_order, link.claim_id, link.claim_version_id,
               link.lifecycle_status, version.lifecycle_status,
               claim.subject_id, claim.speaker_id, claim.predicate,
               claim.object_json, claim.polarity, claim.epistemic_status,
               claim.memory_kind, claim.extraction_confidence,
               claim.valid_from_date, claim.valid_from_timestamp,
               claim.valid_to_date, claim.valid_to_timestamp,
               claim.time_precision, claim.sensitivity,
               version.transaction_from, version.transaction_to,
               version.belief_confidence
        FROM user_records AS record
        JOIN retrieval_index_claim_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN claims AS claim
          ON claim.user_id = link.user_id
         AND claim.claim_id = link.claim_id
        JOIN claim_versions AS version
          ON version.user_id = link.user_id
         AND version.claim_id = link.claim_id
         AND version.version_id = link.claim_version_id
        ORDER BY record.index_record_id, link.claim_order
    """

    EVIDENCE_SQL = """
        WITH user_records AS MATERIALIZED (
            SELECT record.index_record_id, record.record_kind,
                   record.session_summary_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT record.index_record_id, record.record_kind,
               link.claim_id, link.claim_version_id,
               link.source_id, link.span_id, link.support_type,
               evidence.extraction_confidence,
               source.source_type, source.session_id, source.produced_at,
               source.ingested_at, source.content_hash, source.raw_content,
               span.message_id, span.speaker_id, span.verbatim_quote,
               span.start_offset, span.end_offset,
               CASE WHEN record.record_kind = 'atomic' THEN true ELSE EXISTS (
                   SELECT 1
                   FROM session_summary_statement_evidence AS statement_evidence
                   WHERE statement_evidence.user_id = %s
                     AND statement_evidence.summary_id = record.session_summary_id
                     AND statement_evidence.claim_id = link.claim_id
                     AND statement_evidence.claim_version_id = link.claim_version_id
                     AND statement_evidence.source_id = link.source_id
                     AND statement_evidence.span_id = link.span_id
                     AND statement_evidence.support_type = link.support_type
               ) END AS statement_lineage
        FROM user_records AS record
        JOIN retrieval_index_source_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN evidence_links AS evidence
          ON evidence.user_id = link.user_id
         AND evidence.claim_id = link.claim_id
         AND evidence.span_id = link.span_id
         AND evidence.support_type = link.support_type
        JOIN source_spans AS span
          ON span.user_id = link.user_id
         AND span.source_id = link.source_id
         AND span.span_id = link.span_id
        JOIN source_events AS source
          ON source.user_id = span.user_id
         AND source.source_id = span.source_id
        ORDER BY record.index_record_id, link.source_order
    """

    RELATIONS_SQL = """
        WITH user_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT link.index_record_id, link.relation_id, link.relation_type,
               link.direction, link.source_claim_id, link.target_claim_id,
               relation.created_at, decision.transaction_as_of,
               decision.classified_at,
               decision.left_claim_id, decision.left_version_id,
               left_version.transaction_from, left_version.transaction_to,
               left_claim.valid_from_date, left_claim.valid_from_timestamp,
               left_claim.valid_to_date, left_claim.valid_to_timestamp,
               left_claim.time_precision,
               decision.right_claim_id, decision.right_version_id,
               right_version.transaction_from, right_version.transaction_to,
               right_claim.valid_from_date, right_claim.valid_from_timestamp,
               right_claim.valid_to_date, right_claim.valid_to_timestamp,
               right_claim.time_precision
        FROM user_records AS record
        JOIN retrieval_index_relation_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN claim_relations AS relation
          ON relation.user_id = link.user_id
         AND relation.relation_id = link.relation_id
         AND relation.source_claim_id = link.source_claim_id
         AND relation.target_claim_id = link.target_claim_id
         AND relation.relation_type = link.relation_type
        JOIN conflict_decisions AS decision
          ON decision.user_id = relation.user_id
         AND decision.decision_id = relation.decision_id
        JOIN claim_versions AS left_version
          ON left_version.user_id = decision.user_id
         AND left_version.claim_id = decision.left_claim_id
         AND left_version.version_id = decision.left_version_id
        JOIN claims AS left_claim
          ON left_claim.user_id = left_version.user_id
         AND left_claim.claim_id = left_version.claim_id
        JOIN claim_versions AS right_version
          ON right_version.user_id = decision.user_id
         AND right_version.claim_id = decision.right_claim_id
         AND right_version.version_id = decision.right_version_id
        JOIN claims AS right_claim
          ON right_claim.user_id = right_version.user_id
         AND right_claim.claim_id = right_version.claim_id
        ORDER BY link.index_record_id, link.relation_order
    """

    RELATION_LINK_COUNT_SQL = """
        WITH user_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT count(*)
        FROM user_records AS record
        JOIN retrieval_index_relation_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
    """

    def __init__(self, connection: object) -> None:
        self.connection = connection

    def hydrate(self, request: EvidencePackageBuildRequest) -> HydratedPackageInput:
        try:
            with self.connection.transaction():
                self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                return self._hydrate_bound(request)
        except EvidencePackageRepositoryError:
            raise
        except Exception as error:
            raise EvidencePackageRepositoryError("lineage_incomplete", "database") from error

    def _hydrate_bound(self, request: EvidencePackageBuildRequest) -> HydratedPackageInput:
        result = request.retrieval_result
        accepted = {item.index_record_id: item for item in result.accepted}
        if not accepted:
            return HydratedPackageInput((), ())
        params = (
            request.query.user_id, request.query.index_version,
            result.snapshot_run_id, list(accepted), request.query.user_id,
        )
        claim_rows = self.connection.execute(self.CLAIMS_SQL, params).fetchall()
        evidence_rows = self.connection.execute(
            self.EVIDENCE_SQL,
            (
                request.query.user_id, request.query.index_version,
                result.snapshot_run_id, list(accepted), request.query.user_id,
                request.query.user_id,
            ),
        ).fetchall()
        relation_rows = self.connection.execute(self.RELATIONS_SQL, params).fetchall()
        relation_link_count = self.connection.execute(
            self.RELATION_LINK_COUNT_SQL, params
        ).fetchone()[0]
        if relation_link_count != len(relation_rows):
            raise EvidencePackageRepositoryError("lineage_incomplete", "relation")
        return self._assemble_hydration(request, accepted, claim_rows, evidence_rows, relation_rows)

    def _assemble_hydration(self, request, accepted, claim_rows, evidence_rows, relation_rows):
        claim_data: dict[tuple[str, str], Mapping[str, object]] = {}
        anchors: dict[tuple[str, str], set[EvidenceAnchor]] = {}
        expected_by_record: dict[str, set[tuple[str, str]]] = {key: set() for key in accepted}
        for row in claim_rows:
            record_id = str(row[0])
            item = accepted.get(record_id)
            if item is None:
                raise EvidencePackageRepositoryError("stale_result", "claim")
            lifecycle = tuple(row[4])
            if row[5] != item.content_sha256 or lifecycle != item.lifecycle_statuses:
                raise EvidencePackageRepositoryError("stale_result", "record")
            key = (str(row[7]), str(row[8]))
            if key not in set(zip(item.claim_ids, item.claim_version_ids, strict=True)):
                raise EvidencePackageRepositoryError("stale_result", "claim")
            expected_by_record[record_id].add(key)
            if row[9] != row[10] or row[9] not in lifecycle or row[9] == "excluded":
                raise EvidencePackageRepositoryError("invariant_violation", "lifecycle")
            if row[24] == "restricted":
                raise EvidencePackageRepositoryError("invariant_violation", "sensitivity")
            if not (row[25] <= request.query.as_of and (row[26] is None or request.query.as_of < row[26])):
                raise EvidencePackageRepositoryError("stale_result", "transaction")
            if not _valid_time_matches(request.plan.requested_valid_time, row[19:24]):
                raise EvidencePackageRepositoryError("stale_result", "valid_time")
            previous = claim_data.setdefault(key, row)
            if tuple(previous[7:]) != tuple(row[7:]):
                raise EvidencePackageRepositoryError("stale_result", "claim")
            anchors.setdefault(key, set()).add(EvidenceAnchor(
                record_id, str(row[1]), item.rank,
                item.atomic_claim_version_id, item.session_summary_id,
            ))
        for record_id, item in accepted.items():
            if expected_by_record[record_id] != set(zip(item.claim_ids, item.claim_version_ids, strict=True)):
                raise EvidencePackageRepositoryError("lineage_incomplete", "claim")

        evidence_by_claim: dict[tuple[str, str], dict[str, EvidenceSpan]] = {key: {} for key in claim_data}
        sources: dict[str, HydratedSource] = {}
        evidence_records: dict[str, set[tuple[str, str]]] = {key: set() for key in accepted}
        evidence_sources: dict[str, set[str]] = {key: set() for key in accepted}
        evidence_spans: dict[str, set[str]] = {key: set() for key in accepted}
        for row in evidence_rows:
            record_id = str(row[0])
            key = (str(row[2]), str(row[3]))
            if record_id not in accepted or key not in claim_data or not row[19]:
                raise EvidencePackageRepositoryError("lineage_incomplete", "evidence")
            if row[11] > request.query.as_of:
                raise EvidencePackageRepositoryError("stale_result", "source")
            quote, raw = str(row[16]), str(row[13])
            if row[17] is not None:
                if (
                    type(row[17]) is not int or type(row[18]) is not int
                    or row[17] < 0 or row[18] > len(raw) or row[17] >= row[18]
                ):
                    raise EvidencePackageRepositoryError("lineage_incomplete", "offset")
                if raw[row[17]:row[18]] != quote:
                    raise EvidencePackageRepositoryError("lineage_incomplete", "quote")
            elif quote not in raw:
                raise EvidencePackageRepositoryError("lineage_incomplete", "quote")
            if row[8] == "calendar" and row[14] is not None:
                raise EvidencePackageRepositoryError("lineage_incomplete", "calendar_message")
            identifier = evidence_id(request.query.user_id, *key, str(row[4]), str(row[5]), str(row[6]))
            span = EvidenceSpan(
                identifier, *key, str(row[4]), str(row[5]), row[14], str(row[15]),
                quote, row[17], row[18], str(row[6]), float(row[7]),
            )
            evidence_by_claim[key][identifier] = span
            evidence_records[record_id].add(key)
            evidence_sources[record_id].add(str(row[4]))
            evidence_spans[record_id].add(str(row[5]))
            source = HydratedSource(
                request.query.user_id, str(row[4]), str(row[8]), row[9], row[10], row[11],
                str(row[12]), raw,
            )
            if source.source_id in sources and sources[source.source_id] != source:
                raise EvidencePackageRepositoryError("stale_result", "source")
            sources[source.source_id] = source
        if any(evidence_records[key] != expected_by_record[key] for key in accepted):
            raise EvidencePackageRepositoryError("lineage_incomplete", "evidence")
        for record_id, item in accepted.items():
            if (
                evidence_sources[record_id] != set(item.source_ids)
                or evidence_spans[record_id] != set(item.span_ids)
            ):
                raise EvidencePackageRepositoryError("lineage_incomplete", "retrieval_source")

        relation_by_claim: dict[tuple[str, str], set[HydratedRelation]] = {
            key: set() for key in claim_data
        }
        for row in relation_rows:
            if max(row[6], row[7], row[8]) > request.query.as_of:
                raise EvidencePackageRepositoryError("stale_result", "relation")
            reference = CheckedRelationReference(
                str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5])
            )
            left_key = (str(row[9]), str(row[10]))
            right_key = (str(row[18]), str(row[19]))
            record_endpoint_keys = {left_key, right_key} & expected_by_record[str(row[0])]
            if (
                left_key[0] >= right_key[0]
                or {reference.source_claim_id, reference.target_claim_id}
                != {left_key[0], right_key[0]}
                or not record_endpoint_keys
            ):
                raise EvidencePackageRepositoryError("lineage_incomplete", "relation_endpoint")
            if not _transaction_visible(request.query.as_of, row[11], row[12]) or not _transaction_visible(
                request.query.as_of, row[20], row[21]
            ):
                raise EvidencePackageRepositoryError("stale_result", "relation_transaction")
            if not _valid_time_matches(request.plan.requested_valid_time, row[13:18]) or not _valid_time_matches(
                request.plan.requested_valid_time, row[22:27]
            ):
                raise EvidencePackageRepositoryError("stale_result", "relation_valid_time")
            relation = HydratedRelation(
                reference, request.query.user_id,
                left_key[0], left_key[1], row[11], row[12],
                row[13], row[14], row[15], row[16], str(row[17]),
                right_key[0], right_key[1], row[20], row[21],
                row[22], row[23], row[24], row[25], str(row[26]),
                row[6], row[7], row[8],
            )
            for key in record_endpoint_keys:
                relation_by_claim[key].add(relation)

        claims = []
        for key, row in claim_data.items():
            claims.append(HydratedClaim(
                request.query.user_id, *key, str(row[11]), str(row[12]), str(row[13]),
                row[14], str(row[15]), str(row[16]), row[17], str(row[9]), float(row[18]),
                None if row[27] is None else float(row[27]), row[19], row[20], row[21], row[22],
                str(row[23]), row[25], row[26], row[24],
                tuple(sorted(anchors[key], key=lambda item: (item.retrieval_rank, item.index_record_id))),
                tuple(sorted(evidence_by_claim[key].values(), key=lambda item: item.evidence_id)),
                tuple(sorted(
                    relation_by_claim[key],
                    key=lambda item: (
                        item.reference.relation_type,
                        item.reference.relation_id,
                        item.reference.direction,
                    ),
                )),
            ))
        return HydratedPackageInput(
            tuple(sorted(claims, key=lambda item: (min(value.retrieval_rank for value in item.anchors), item.claim_id, item.claim_version_id))),
            tuple(sorted(sources.values(), key=lambda item: item.source_id)),
        )


def _valid_time_matches(requested: RequestedValidTime | None, values) -> bool:
    if requested is None:
        return True
    start_date, start_timestamp, end_date, end_timestamp, precision = values
    if precision == "unknown":
        return False
    if requested.kind == "point":
        point = requested.point_date or requested.point_timestamp
        if requested.point_date is not None and precision == "timestamp":
            return False
        if requested.point_timestamp is not None and precision != "timestamp":
            point = requested.point_timestamp.date()
        start, end = (start_timestamp, end_timestamp) if precision == "timestamp" else (start_date, end_date)
        return (start is None or start <= point) and (end is None or point <= end)
    if requested.range_start_date is not None or requested.range_end_date is not None:
        if precision == "timestamp":
            return False
        left, right = requested.range_start_date, requested.range_end_date
        start, end = start_date, end_date
    else:
        if precision != "timestamp":
            return False
        left, right = requested.range_start_timestamp, requested.range_end_timestamp
        start, end = start_timestamp, end_timestamp
    return (start is None or right is None or start <= right) and (left is None or end is None or left <= end)


def _transaction_visible(as_of: datetime, start: datetime, end: datetime | None) -> bool:
    return start <= as_of and (end is None or as_of < end)
