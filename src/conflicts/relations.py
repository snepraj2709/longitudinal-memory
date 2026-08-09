"""Atomic persistence for deterministic conflict decisions and relations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

from psycopg.rows import dict_row

from storage.contracts import (
    ClaimRelationRecord,
    ConflictDecisionEvidenceRecord,
    ConflictDecisionRecord,
    StorageValidationError,
)
from storage.repository import StorageConflictError, StorageRepository

from .classifier import (
    ClassificationRequest,
    ConflictClassifier,
    ConflictDecision,
)


class ConflictRelationError(RuntimeError):
    """A sanitized persistence failure containing no claim or evidence text."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class ConflictRelationConflict(ConflictRelationError):
    """An idempotency, visibility, or stable-ID conflict."""


@dataclass(frozen=True)
class PersistedConflict:
    decision: ConflictDecisionRecord
    relations: tuple[ClaimRelationRecord, ...]
    evidence: tuple[ConflictDecisionEvidenceRecord, ...]
    replayed: bool


class ConflictRelationService:
    """Persist one checked pair without changing claim lifecycle state."""

    def __init__(self, connection: object, *, repo_root: str | Path = ".") -> None:
        self._connection = connection
        self._repository = StorageRepository(connection)
        self._classifier = ConflictClassifier(repo_root=repo_root)

    def persist(
        self,
        request: ClassificationRequest,
        decision: ConflictDecision,
        classified_at: datetime,
    ) -> PersistedConflict:
        _aware(classified_at, "classified_at")
        expected = self._classifier.classify(request)
        if decision != expected:
            raise ConflictRelationConflict("decision_mismatch", "decision")
        with self._connection.transaction():
            self._connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"conflict:{request.user_id}:{request.pair.pair_id}",),
            )
            evidence = self._validated_evidence(request, decision)
            decision_record, relations = _decision_records(
                self._classifier.config.rule_version,
                request,
                decision,
                classified_at,
            )
            existing = self._repository.get_conflict_decision(
                request.user_id, decision.decision_id
            )
            if existing is not None:
                if existing != decision_record:
                    raise ConflictRelationConflict("stable_id_drift", "decision")
                stored_relations, stored_evidence = self._stored_children(
                    request.user_id, decision.decision_id
                )
                if stored_relations != relations or stored_evidence != evidence:
                    raise ConflictRelationConflict("incomplete_replay", "decision")
                return PersistedConflict(
                    decision_record, relations, evidence, True
                )
            try:
                self._repository.insert_conflict_decision(decision_record)
                for record in evidence:
                    self._repository.insert_conflict_decision_evidence(record)
                for record in relations:
                    self._repository.insert_claim_relation(record)
            except StorageConflictError as error:
                raise ConflictRelationConflict(
                    "stable_id_drift", "persistence"
                ) from error
            return PersistedConflict(decision_record, relations, evidence, False)

    def _validated_evidence(
        self,
        request: ClassificationRequest,
        decision: ConflictDecision,
    ) -> tuple[ConflictDecisionEvidenceRecord, ...]:
        ingestion_rows = self._connection.execute(
            """
            SELECT source_id, ingested_at
            FROM source_events
            WHERE user_id = %s AND source_id = ANY(%s)
            ORDER BY source_id
            """,
            (request.user_id, list(request.pair.source_ids)),
        ).fetchall()
        if tuple(ingestion_rows) != request.source_ingested_at:
            raise ConflictRelationConflict("source_snapshot_drift", "evidence")
        records: list[ConflictDecisionEvidenceRecord] = []
        for item in (request.left, request.right):
            stored_claim = self._repository.get_claim(
                request.user_id, item.claim.claim_id
            )
            stored_version = self._repository.get_claim_version(
                request.user_id, item.version.version_id
            )
            if stored_claim != item.claim or stored_version != item.version:
                raise ConflictRelationConflict("claim_snapshot_drift", "claim")
            rows = self._connection.execute(
                """
                SELECT source.source_id, span.span_id, span.message_id,
                       span.speaker_id, span.verbatim_quote, evidence.support_type
                FROM evidence_links AS evidence
                JOIN source_spans AS span
                  ON span.user_id = evidence.user_id
                 AND span.span_id = evidence.span_id
                JOIN source_events AS source
                  ON source.user_id = span.user_id
                 AND source.source_id = span.source_id
                WHERE evidence.user_id = %s AND evidence.claim_id = %s
                  AND source.ingested_at <= %s
                ORDER BY source.source_id, span.span_id, evidence.support_type
                """,
                (
                    request.user_id,
                    item.claim.claim_id,
                    request.transaction_as_of,
                ),
            ).fetchall()
            visible = tuple(
                (
                    value.source_id,
                    value.span_id,
                    value.message_id,
                    value.speaker_id,
                    value.quote,
                )
                for value in item.evidence
            )
            if tuple(row[:5] for row in rows) != visible:
                raise ConflictRelationConflict("evidence_snapshot_drift", "evidence")
            for row in rows:
                span_id, support_type = row[1], row[5]
                records.append(
                    ConflictDecisionEvidenceRecord(
                        decision_evidence_id=conflict_decision_evidence_id(
                            request.user_id,
                            decision.decision_id,
                            item.claim.claim_id,
                            span_id,
                            support_type,
                            decision.input_snapshot_sha256,
                        ),
                        user_id=request.user_id,
                        decision_id=decision.decision_id,
                        claim_id=item.claim.claim_id,
                        span_id=span_id,
                        support_type=support_type,
                        input_snapshot_sha256=decision.input_snapshot_sha256,
                    )
                )
        return tuple(
            sorted(
                records,
                key=lambda value: (
                    value.claim_id, value.span_id, value.support_type
                ),
            )
        )

    def _stored_children(
        self, user_id: str, decision_id: str
    ) -> tuple[
        tuple[ClaimRelationRecord, ...],
        tuple[ConflictDecisionEvidenceRecord, ...],
    ]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT relation_id, user_id, decision_id, classifier_version,
                       source_claim_id, target_claim_id, relation_type, confidence,
                       input_snapshot_sha256, created_at
                FROM claim_relations
                WHERE user_id = %s AND decision_id = %s
                ORDER BY relation_id
                """,
                (user_id, decision_id),
            )
            relations = tuple(ClaimRelationRecord(**row) for row in cursor.fetchall())
            cursor.execute(
                """
                SELECT decision_evidence_id, user_id, decision_id, claim_id,
                       span_id, support_type, input_snapshot_sha256
                FROM conflict_decision_evidence
                WHERE user_id = %s AND decision_id = %s
                ORDER BY claim_id, span_id, support_type
                """,
                (user_id, decision_id),
            )
            evidence = tuple(
                ConflictDecisionEvidenceRecord(**row) for row in cursor.fetchall()
            )
        return relations, evidence


def _decision_records(
    rule_version: str,
    request: ClassificationRequest,
    decision: ConflictDecision,
    classified_at: datetime,
) -> tuple[ConflictDecisionRecord, tuple[ClaimRelationRecord, ...]]:
    record = ConflictDecisionRecord(
        decision_id=decision.decision_id,
        user_id=decision.user_id,
        classifier_version=decision.classifier_version,
        rule_version=rule_version,
        pair_id=decision.pair_id,
        left_claim_id=request.left.claim.claim_id,
        right_claim_id=request.right.claim.claim_id,
        left_version_id=request.left.version.version_id,
        right_version_id=request.right.version.version_id,
        input_snapshot_sha256=decision.input_snapshot_sha256,
        transaction_as_of=request.transaction_as_of,
        matched_rule=decision.label,
        label=decision.label,
        classified_at=classified_at,
    )
    relations = tuple(
        ClaimRelationRecord(
            relation_id=value.relation_id,
            user_id=value.user_id,
            decision_id=decision.decision_id,
            classifier_version=value.classifier_version,
            source_claim_id=value.source_claim_id,
            target_claim_id=value.target_claim_id,
            relation_type=value.relation_type,
            confidence=value.confidence,
            input_snapshot_sha256=value.input_snapshot_sha256,
            created_at=classified_at,
        )
        for value in decision.relations
    )
    return record, tuple(sorted(relations, key=lambda value: value.relation_id))


def conflict_decision_evidence_id(
    user_id: str,
    decision_id: str,
    claim_id: str,
    span_id: str,
    support_type: str,
    input_snapshot_sha256: str,
) -> str:
    payload = json.dumps(
        [
            "conflict_decision_evidence",
            user_id,
            decision_id,
            claim_id,
            span_id,
            support_type,
            input_snapshot_sha256,
        ],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise StorageValidationError(f"{name} must be timezone-aware")
    return value
