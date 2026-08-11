"""Atomic persistence and lifecycle execution for belief-resolution plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

from psycopg.rows import dict_row

from ingestion.contracts import outbox_id
from storage.contracts import (
    BeliefResolutionActionRecord,
    BeliefResolutionEvidenceRecord,
    BeliefResolutionRecord,
    ClaimRelationRecord,
    ConflictDecisionEvidenceRecord,
    ProcessingOutboxRecord,
)
from storage.repository import StorageConflictError, StorageRepository
from temporal.contracts import TemporalClaim, TransitionRequest, VisibleEvidence
from temporal.service import TemporalError, TemporalService

from .resolver import (
    AuthorityEvidence,
    BeliefResolutionPlan,
    BeliefResolver,
    PersistedResolutionInput,
    ResolutionRequest,
)


class BeliefResolutionError(RuntimeError):
    """A sanitized resolver persistence or execution failure."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class BeliefResolutionConflict(BeliefResolutionError):
    """An idempotency, visibility, ownership, or concurrency conflict."""


@dataclass(frozen=True)
class PersistedBeliefResolution:
    resolution: BeliefResolutionRecord
    actions: tuple[BeliefResolutionActionRecord, ...]
    evidence: tuple[BeliefResolutionEvidenceRecord, ...]
    relations: tuple[ClaimRelationRecord, ...]
    replayed: bool


class BeliefResolutionService:
    """Load one persisted decision and execute its deterministic plan atomically."""

    def __init__(
        self,
        connection: object,
        *,
        repo_root: str | Path = ".",
    ) -> None:
        self._connection = connection
        self._repository = StorageRepository(connection)
        self._planner = BeliefResolver(repo_root=repo_root)
        self._temporal = TemporalService(connection)

    def resolve(self, request: ResolutionRequest) -> PersistedBeliefResolution:
        with self._connection.transaction():
            return self.resolve_in_transaction(request)

    def resolve_in_transaction(
        self, request: ResolutionRequest
    ) -> PersistedBeliefResolution:
        decision = self._repository.get_conflict_decision(
            request.user_id, request.decision_id
        )
        if decision is None:
            raise BeliefResolutionError("decision_not_found", "decision")
        self._connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"belief:{request.user_id}:{decision.pair_id}",),
        )
        persisted = self._load_input(request)
        replay = self._repository.get_belief_resolution_by_idempotency(
            request.user_id, request.idempotency_key
        )
        if replay is not None:
            persisted = self._replay_input(persisted, replay)
        try:
            plan = self._planner.plan(request, persisted)
        except ValueError as error:
            raise BeliefResolutionConflict(
                "snapshot_validation_failed", "resolver_input"
            ) from error
        expected = _resolution_record(plan)
        if replay is not None:
            return self._replay(plan, expected, replay)
        self._require_open_action_versions(request, persisted, plan)
        try:
            self._repository.insert_belief_resolution(expected)
            evidence = tuple(
                BeliefResolutionEvidenceRecord(
                    user_id=request.user_id,
                    resolution_id=plan.resolution_id,
                    decision_id=request.decision_id,
                    decision_evidence_id=value,
                )
                for value in plan.decision_evidence_ids
            )
            for record in evidence:
                self._repository.insert_belief_resolution_evidence(record)
            actions = self._execute_actions(request, plan)
            relations = self._persist_relations(
                persisted.decision.classifier_version, plan
            )
            self._repository.insert_outbox(_resolution_outbox(plan))
        except StorageConflictError as error:
            raise BeliefResolutionConflict(
                "stable_id_drift", "persistence"
            ) from error
        except TemporalError as error:
            raise BeliefResolutionConflict(
                "lifecycle_execution_failed", error.location
            ) from error
        return PersistedBeliefResolution(expected, actions, evidence, relations, False)

    def _load_input(
        self, request: ResolutionRequest
    ) -> PersistedResolutionInput:
        decision = self._repository.get_conflict_decision(
            request.user_id, request.decision_id
        )
        if decision is None:
            raise BeliefResolutionError("decision_not_found", "decision")
        if (
            decision.transaction_as_of > request.transaction_as_of
            or decision.classified_at > request.transaction_as_of
        ):
            raise BeliefResolutionConflict("future_decision", "transaction_as_of")
        left_claim = self._repository.get_claim(
            request.user_id, decision.left_claim_id
        )
        right_claim = self._repository.get_claim(
            request.user_id, decision.right_claim_id
        )
        left_version = self._repository.get_claim_version(
            request.user_id, decision.left_version_id
        )
        right_version = self._repository.get_claim_version(
            request.user_id, decision.right_version_id
        )
        if None in (left_claim, right_claim, left_version, right_version):
            raise BeliefResolutionConflict("snapshot_missing", "claim")
        rows = self._connection.execute(
            """
            SELECT cited.decision_evidence_id, cited.user_id,
                   cited.decision_id, cited.claim_id, cited.span_id,
                   cited.support_type, cited.input_snapshot_sha256,
                   source.source_id, source.source_type, source.ingested_at,
                   span.message_id, span.speaker_id, span.verbatim_quote
            FROM conflict_decision_evidence AS cited
            JOIN source_spans AS span
              ON span.user_id = cited.user_id AND span.span_id = cited.span_id
            JOIN source_events AS source
              ON source.user_id = span.user_id AND source.source_id = span.source_id
            WHERE cited.user_id = %s AND cited.decision_id = %s
            ORDER BY cited.claim_id, source.source_id, cited.span_id,
                     cited.support_type, cited.decision_evidence_id
            """,
            (request.user_id, request.decision_id),
        ).fetchall()
        if not rows:
            raise BeliefResolutionConflict("evidence_missing", "decision")
        evidence_records = tuple(
            sorted(
                (ConflictDecisionEvidenceRecord(*row[:7]) for row in rows),
                key=lambda value: (
                    value.claim_id,
                    value.span_id,
                    value.support_type,
                    value.decision_evidence_id,
                ),
            )
        )
        evidence_by_claim: dict[str, list[VisibleEvidence]] = {
            decision.left_claim_id: [],
            decision.right_claim_id: [],
        }
        seen_spans: set[tuple[str, str]] = set()
        source_details: dict[str, tuple[str, datetime]] = {}
        source_spans: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            claim_id = row[3]
            if claim_id not in evidence_by_claim:
                raise BeliefResolutionConflict("evidence_claim_drift", "evidence")
            source_id, source_type, ingested_at = row[7:10]
            if ingested_at > request.transaction_as_of:
                raise BeliefResolutionConflict("future_evidence", "evidence")
            prior = source_details.setdefault(source_id, (source_type, ingested_at))
            if prior != (source_type, ingested_at):
                raise BeliefResolutionConflict("source_snapshot_drift", "evidence")
            key = (claim_id, row[4])
            if key not in seen_spans:
                evidence_by_claim[claim_id].append(
                    VisibleEvidence(
                        source_id, row[4], row[10], row[11], row[12]
                    )
                )
                seen_spans.add(key)
            source_spans.setdefault((claim_id, source_id), []).append(row[4])
        left = TemporalClaim(
            left_claim,
            left_version,
            tuple(evidence_by_claim[decision.left_claim_id]),
        )
        right = TemporalClaim(
            right_claim,
            right_version,
            tuple(evidence_by_claim[decision.right_claim_id]),
        )
        relations = tuple(
            ClaimRelationRecord(**row)
            for row in self._relation_rows(
                request.user_id, request.decision_id, classifier_only=True
            )
        )
        claims = {
            left.claim.claim_id: left,
            right.claim.claim_id: right,
        }
        authority: list[AuthorityEvidence] = []
        for (claim_id, source_id), span_ids in sorted(source_spans.items()):
            source_type, ingested_at = source_details[source_id]
            if source_type not in self._planner.config.official_source_types:
                continue
            claim = claims[claim_id].claim
            authority.append(
                AuthorityEvidence(
                    user_id=request.user_id,
                    claim_id=claim_id,
                    authority_kind="official_record",
                    speaker_id=claim.speaker_id,
                    subject_id=claim.subject_id,
                    predicate=claim.predicate,
                    source_type=source_type,
                    source_id=source_id,
                    span_ids=tuple(sorted(set(span_ids))),
                    source_ingested_at=ingested_at,
                )
            )
        return PersistedResolutionInput(
            decision=decision,
            left=left,
            right=right,
            relations=relations,
            decision_evidence=evidence_records,
            source_ingested_at=tuple(
                (source_id, details[1])
                for source_id, details in sorted(source_details.items())
            ),
            authority=tuple(authority),
            deleted_source_ids=(),
        )

    def _replay_input(
        self,
        persisted: PersistedResolutionInput,
        resolution: BeliefResolutionRecord,
    ) -> PersistedResolutionInput:
        rows = self._connection.execute(
            """
            SELECT claim_id, from_version_id
            FROM belief_resolution_actions
            WHERE user_id = %s AND resolution_id = %s
            ORDER BY action_order
            """,
            (resolution.user_id, resolution.resolution_id),
        ).fetchall()
        acted = {claim_id: version_id for claim_id, version_id in rows}

        def original(item: TemporalClaim) -> TemporalClaim:
            expected = acted.get(item.claim.claim_id)
            if expected is None:
                return item
            if (
                item.version.version_id != expected
                or item.version.transaction_to != resolution.resolved_at
            ):
                raise BeliefResolutionConflict(
                    "replay_version_drift", "claim_version"
                )
            return replace(
                item,
                version=replace(item.version, transaction_to=None),
            )

        return replace(
            persisted,
            left=original(persisted.left),
            right=original(persisted.right),
        )

    def _require_open_action_versions(
        self,
        request: ResolutionRequest,
        persisted: PersistedResolutionInput,
        plan: BeliefResolutionPlan,
    ) -> None:
        by_claim = {
            persisted.left.claim.claim_id: persisted.left.version,
            persisted.right.claim.claim_id: persisted.right.version,
        }
        for action in plan.actions:
            row = self._connection.execute(
                """
                SELECT version_id FROM claim_versions
                WHERE user_id = %s AND claim_id = %s AND transaction_to IS NULL
                FOR UPDATE
                """,
                (request.user_id, action.claim_id),
            ).fetchone()
            if row is None or row[0] != by_claim[action.claim_id].version_id:
                raise BeliefResolutionConflict(
                    "open_version_drift", "claim_version"
                )

    def _execute_actions(
        self,
        request: ResolutionRequest,
        plan: BeliefResolutionPlan,
    ) -> tuple[BeliefResolutionActionRecord, ...]:
        records: list[BeliefResolutionActionRecord] = []
        for action in plan.actions:
            result = self._temporal.transition_in_transaction(
                TransitionRequest(
                    user_id=request.user_id,
                    claim_id=action.claim_id,
                    idempotency_key=f"belief:{plan.resolution_id}:{action.action_id}",
                    target_status=action.target_status,
                    reason=action.reason,
                    transitioned_at=request.resolved_at,
                    belief_confidence=None,
                )
            )
            record = BeliefResolutionActionRecord(
                action_id=action.action_id,
                user_id=request.user_id,
                resolution_id=plan.resolution_id,
                action_order=action.action_order,
                claim_id=action.claim_id,
                from_status=action.from_status,
                target_status=action.target_status,
                replacement_claim_id=action.replacement_claim_id,
                reason=action.reason,
                from_version_id=result.transition.from_version_id,
                to_version_id=result.transition.to_version_id,
                transition_id=result.transition.transition_id,
            )
            self._repository.insert_belief_resolution_action(record)
            records.append(record)
        return tuple(records)

    def _persist_relations(
        self,
        classifier_version: str,
        plan: BeliefResolutionPlan,
    ) -> tuple[ClaimRelationRecord, ...]:
        records = tuple(
            ClaimRelationRecord(
                relation_id=value.relation_id,
                user_id=value.user_id,
                decision_id=value.decision_id,
                classifier_version=classifier_version,
                source_claim_id=value.source_claim_id,
                target_claim_id=value.target_claim_id,
                relation_type=value.relation_type,
                confidence=1,
                input_snapshot_sha256=value.input_snapshot_sha256,
                created_at=plan.resolved_at,
                resolver_version=plan.resolver_version,
                resolution_id=plan.resolution_id,
            )
            for value in plan.relations
        )
        for record in records:
            self._repository.insert_claim_relation(record)
        return records

    def _replay(
        self,
        plan: BeliefResolutionPlan,
        expected: BeliefResolutionRecord,
        stored: BeliefResolutionRecord,
    ) -> PersistedBeliefResolution:
        if stored != expected:
            raise BeliefResolutionConflict("idempotency_drift", "resolution")
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT * FROM belief_resolution_actions
                WHERE user_id = %s AND resolution_id = %s
                ORDER BY action_order
                """,
                (plan.user_id, plan.resolution_id),
            )
            actions = tuple(
                BeliefResolutionActionRecord(**row) for row in cursor.fetchall()
            )
            cursor.execute(
                """
                SELECT user_id, resolution_id, decision_id, decision_evidence_id
                FROM belief_resolution_evidence
                WHERE user_id = %s AND resolution_id = %s
                ORDER BY decision_evidence_id
                """,
                (plan.user_id, plan.resolution_id),
            )
            evidence = tuple(
                BeliefResolutionEvidenceRecord(**row) for row in cursor.fetchall()
            )
        relations = tuple(
            ClaimRelationRecord(**row)
            for row in self._relation_rows(
                plan.user_id, plan.decision_id, resolution_id=plan.resolution_id
            )
        )
        if not _matches_plan(plan, actions, evidence, relations):
            raise BeliefResolutionConflict("incomplete_replay", "resolution")
        return PersistedBeliefResolution(stored, actions, evidence, relations, True)

    def _relation_rows(
        self,
        user_id: str,
        decision_id: str,
        *,
        classifier_only: bool = False,
        resolution_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        where = ["user_id = %s", "decision_id = %s"]
        parameters: list[object] = [user_id, decision_id]
        if classifier_only:
            where.append("resolution_id IS NULL")
        if resolution_id is not None:
            where.append("resolution_id = %s")
            parameters.append(resolution_id)
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""
                SELECT relation_id, user_id, decision_id, classifier_version,
                       source_claim_id, target_claim_id, relation_type, confidence,
                       input_snapshot_sha256, created_at, resolver_version,
                       resolution_id
                FROM claim_relations
                WHERE {' AND '.join(where)}
                ORDER BY relation_id
                """,
                tuple(parameters),
            )
            return tuple(cursor.fetchall())


def _resolution_record(plan: BeliefResolutionPlan) -> BeliefResolutionRecord:
    valid_at_date = plan.valid_at if type(plan.valid_at) is date else None
    valid_at_timestamp = plan.valid_at if isinstance(plan.valid_at, datetime) else None
    return BeliefResolutionRecord(
        resolution_id=plan.resolution_id,
        user_id=plan.user_id,
        resolver_version=plan.resolver_version,
        policy_version=plan.policy_version,
        decision_id=plan.decision_id,
        idempotency_key=plan.idempotency_key,
        input_snapshot_sha256=plan.input_snapshot_sha256,
        transaction_as_of=plan.transaction_as_of,
        valid_at_date=valid_at_date,
        valid_at_timestamp=valid_at_timestamp,
        resolved_at=plan.resolved_at,
        outcome=plan.outcome,
        selected_current_claim_id=plan.selected_current_claim_id,
        authority_reason=plan.authority_reason,
        belief_confidence=None,
    )


def _resolution_outbox(plan: BeliefResolutionPlan) -> ProcessingOutboxRecord:
    payload: dict[str, object] = {
        "resolution_id": plan.resolution_id,
        "decision_id": plan.decision_id,
        "action_ids": [value.action_id for value in plan.actions],
        "relation_ids": [value.relation_id for value in plan.relations],
        "decision_evidence_ids": list(plan.decision_evidence_ids),
    }
    if plan.selected_current_claim_id is not None:
        payload["selected_current_claim_id"] = plan.selected_current_claim_id
    dedupe = f"belief_resolved:{plan.resolution_id}"
    return ProcessingOutboxRecord(
        event_id=outbox_id(plan.user_id, "belief_resolved", dedupe),
        user_id=plan.user_id,
        event_type="belief_resolved",
        aggregate_id=plan.resolution_id,
        dedupe_key=dedupe,
        payload=payload,
        state="pending",
        created_at=plan.resolved_at,
    )


def _matches_plan(
    plan: BeliefResolutionPlan,
    actions: tuple[BeliefResolutionActionRecord, ...],
    evidence: tuple[BeliefResolutionEvidenceRecord, ...],
    relations: tuple[ClaimRelationRecord, ...],
) -> bool:
    if len(actions) != len(plan.actions) or len(relations) != len(plan.relations):
        return False
    for stored, expected in zip(actions, plan.actions):
        if (
            stored.action_id != expected.action_id
            or stored.action_order != expected.action_order
            or stored.claim_id != expected.claim_id
            or stored.from_status != expected.from_status
            or stored.target_status != expected.target_status
            or stored.replacement_claim_id != expected.replacement_claim_id
            or stored.reason != expected.reason
        ):
            return False
    if tuple(value.decision_evidence_id for value in evidence) != plan.decision_evidence_ids:
        return False
    return all(
        stored.relation_id == expected.relation_id
        and stored.source_claim_id == expected.source_claim_id
        and stored.target_claim_id == expected.target_claim_id
        and stored.relation_type == expected.relation_type
        and stored.resolver_version == plan.resolver_version
        and stored.resolution_id == plan.resolution_id
        for stored, expected in zip(relations, plan.relations)
    )
