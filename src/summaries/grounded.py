"""Pure planning and rendering for grounded session summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
from typing import Sequence

from storage.contracts import ProcessingOutboxRecord

from .contracts import BOUNDARY_VERSION, SessionizationRequest
from .repository import SessionSourceRepository
from .sessions import SessionizationService
from .summary_contracts import (
    RENDERER_VERSION,
    GroundedStatement,
    GroundedSummaryError,
    GroundedSummaryRequest,
    GroundedSummaryResult,
    SessionSummary,
    SummaryEvidence,
    SummaryRendererConfig,
    canonical_json,
    load_summary_renderer_config,
)


DEFAULT_CONFIG_PATH = Path("configs/summaries/session_summary_renderer_v1.json")
SUMMARY_REBUILD_EVENT_TYPES = frozenset(
    {
        "source_ingested",
        "claims_changed",
        "claim_recompute_required",
        "source_deleted",
        "claim_lifecycle_changed",
        "conflict_recompute_required",
        "belief_resolved",
    }
)


@dataclass(frozen=True)
class SummaryCoordinatorResult:
    event_id: str
    session_count: int
    created_count: int
    replayed_count: int
    retired_count: int


class GroundedSummaryCoordinator:
    """Rebuild one user's current summaries from an existing outbox event."""

    def __init__(
        self,
        connection: object,
        *,
        config: SummaryRendererConfig | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        from .summary_repository import SessionSummaryRepository

        self.connection = connection
        self.config = config or load_summary_renderer_config(config_path)
        self.session_service = SessionizationService(SessionSourceRepository(connection))
        self.summary_repository = SessionSummaryRepository(connection)

    def process(self, event: ProcessingOutboxRecord) -> SummaryCoordinatorResult:
        if event.event_type not in SUMMARY_REBUILD_EVENT_TYPES:
            raise GroundedSummaryError("outbox event type does not rebuild summaries")
        created = 0
        replayed = 0
        retired = 0
        with self.connection.transaction():
            sessions = self.session_service.define_sessions(
                SessionizationRequest(
                    event.user_id,
                    event.created_at,
                    BOUNDARY_VERSION,
                )
            )
            for definition in sessions.definitions:
                evidence = self.summary_repository.load_visible_evidence(
                    definition,
                    event.created_at,
                )
                if not evidence:
                    retired += self.summary_repository.retire_definition(
                        event.user_id,
                        definition.definition_id,
                        self.config.renderer_version,
                        event.created_at,
                    )
                    continue
                request = GroundedSummaryRequest(
                    user_id=event.user_id,
                    session_definition_id=definition.definition_id,
                    session_membership_sha256=definition.membership_sha256,
                    transaction_as_of=event.created_at,
                    idempotency_key=f"summary:{event.event_id}:{definition.definition_id}",
                    renderer_version=self.config.renderer_version,
                )
                planned = plan_grounded_summary(request, evidence, config=self.config)
                outcome = self.summary_repository.persist_in_transaction(
                    planned,
                    definition.source_ids,
                )
                created += outcome.created
                replayed += outcome.replayed
                retired += outcome.retired
            retired += self.summary_repository.retire_missing_current(
                event.user_id,
                self.config.renderer_version,
                tuple(item.definition_id for item in sessions.definitions),
                event.created_at,
            )
        return SummaryCoordinatorResult(
            event.event_id,
            len(sessions.definitions),
            created,
            replayed,
            retired,
        )


def plan_grounded_summary(
    request: GroundedSummaryRequest,
    evidence: Sequence[SummaryEvidence],
    *,
    config: SummaryRendererConfig | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> GroundedSummaryResult:
    renderer = config or load_summary_renderer_config(config_path)
    if request.renderer_version != renderer.renderer_version:
        raise GroundedSummaryError("renderer version changed")
    ordered = _validate_and_order(request, evidence)
    included = tuple(item for item in ordered if _included(item, renderer))
    bundles = _claim_bundles(included)
    _validate_disputes(bundles)
    snapshot = _snapshot(request, renderer, included)
    facts = tuple(_fact_statement(bundle, snapshot) for bundle in bundles)
    questions = _questions(bundles, snapshot)
    facts = tuple(sorted(facts, key=_statement_order))
    questions = tuple(sorted(questions, key=_statement_order))
    summary_text = _render_summary(renderer, facts, questions)
    kind, start_date, end_date, start_timestamp, end_timestamp = _aggregate_time(
        tuple(bundle[0] for bundle in bundles)
    )
    all_statement_evidence = tuple(
        item for statement in (*facts, *questions) for item in statement.evidence
    )
    source_ids = tuple(sorted({item.source_id for item in all_statement_evidence}))
    claim_ids = tuple(sorted({item.claim_id for item in all_statement_evidence}))
    summary_id = _stable_id(
        "session_summary", request.user_id, request.session_definition_id,
        request.renderer_version, snapshot,
    )
    summary = SessionSummary(
        summary_id,
        request.user_id,
        request.session_definition_id,
        request.renderer_version,
        snapshot,
        summary_text,
        facts,
        questions,
        kind,
        start_date,
        end_date,
        start_timestamp,
        end_timestamp,
        source_ids,
        claim_ids,
        any(item.sensitive for item in facts),
    )
    plan_id = _stable_id(
        "summary_plan", request.user_id, request.session_definition_id,
        request.renderer_version, request.idempotency_key,
    )
    return GroundedSummaryResult(plan_id, snapshot, request, summary)


def replay_grounded_summary(
    existing: GroundedSummaryResult,
    planned: GroundedSummaryResult,
) -> tuple[GroundedSummaryResult, bool]:
    """Return an exact replay or reject idempotency drift."""

    if existing.plan_id != planned.plan_id:
        raise GroundedSummaryError("summary replay uses a different idempotency key")
    if existing.input_snapshot_sha256 != planned.input_snapshot_sha256 or existing != planned:
        raise GroundedSummaryError("summary idempotency key has input drift")
    return existing, False


def _validate_and_order(
    request: GroundedSummaryRequest,
    evidence: Sequence[SummaryEvidence],
) -> tuple[SummaryEvidence, ...]:
    if not evidence:
        raise GroundedSummaryError("summary evidence is empty")
    seen: set[str] = set()
    for item in evidence:
        if item.evidence_id in seen:
            raise GroundedSummaryError("summary evidence ID is duplicated")
        seen.add(item.evidence_id)
        if item.user_id != request.user_id or item.session_definition_id != request.session_definition_id:
            raise GroundedSummaryError("summary evidence leaves the user or session boundary")
        if item.source_ingested_at > request.transaction_as_of:
            raise GroundedSummaryError("summary evidence source is not visible")
        if not (
            item.transaction_from <= request.transaction_as_of
            and (item.transaction_to is None or request.transaction_as_of < item.transaction_to)
        ):
            raise GroundedSummaryError("summary claim version is not visible")
    return tuple(sorted(evidence, key=_evidence_order))


def _included(item: SummaryEvidence, config: SummaryRendererConfig) -> bool:
    return (
        item.lifecycle_status in config.eligible_lifecycle_statuses
        and item.epistemic_status not in config.excluded_epistemic_statuses
        and item.sensitivity not in config.excluded_sensitivities
    )


def _claim_bundles(
    evidence: Sequence[SummaryEvidence],
) -> tuple[tuple[SummaryEvidence, ...], ...]:
    grouped: dict[tuple[str, str], list[SummaryEvidence]] = defaultdict(list)
    for item in evidence:
        grouped[(item.claim_id, item.claim_version_id)].append(item)
    bundles: list[tuple[SummaryEvidence, ...]] = []
    semantic_fields = (
        "user_id", "session_definition_id", "claim_id", "claim_version_id",
        "subject_id", "speaker_id", "predicate", "object_json", "epistemic_status",
        "lifecycle_status", "sensitivity", "time_precision", "valid_from_date",
        "valid_from_timestamp", "valid_to_date", "valid_to_timestamp",
        "transaction_from", "transaction_to", "linked_claim_ids",
    )
    for members in grouped.values():
        first = members[0]
        if any(
            any(getattr(item, field) != getattr(first, field) for field in semantic_fields)
            for item in members[1:]
        ):
            raise GroundedSummaryError("claim evidence has inconsistent semantics")
        bundles.append(tuple(sorted(members, key=_evidence_order)))
    return tuple(sorted(bundles, key=lambda item: _evidence_order(item[0])))


def _validate_disputes(bundles: Sequence[tuple[SummaryEvidence, ...]]) -> None:
    by_claim = {bundle[0].claim_id: bundle[0] for bundle in bundles}
    present = set(by_claim)
    for bundle in bundles:
        first = bundle[0]
        if first.lifecycle_status == "disputed":
            if not set(first.linked_claim_ids).issubset(present):
                raise GroundedSummaryError("disputed summary is missing a linked side")
            if any(
                by_claim[claim_id].lifecycle_status != "disputed"
                or by_claim[claim_id].linked_claim_ids != first.linked_claim_ids
                for claim_id in first.linked_claim_ids
            ):
                raise GroundedSummaryError("disputed summary links are inconsistent")


def _fact_statement(
    bundle: tuple[SummaryEvidence, ...], snapshot: str
) -> GroundedStatement:
    first = bundle[0]
    fact = _fact_text(first)
    epistemic = {
        "reported_by_other": f"Reported by {first.speaker_id}: {fact}",
        "uncertain": f"{first.speaker_id} was uncertain: {fact}",
        "denied": f"{first.speaker_id} denied: {fact}",
        "corrected": f"{first.speaker_id} corrected the record: {fact}",
    }.get(first.epistemic_status, fact)
    if first.lifecycle_status == "candidate":
        text = f"Candidate, attributed to {first.speaker_id}: {epistemic}"
        view = "candidate"
    elif first.lifecycle_status == "historical":
        text = f"Historical: {epistemic}"
        view = "historical"
    elif first.lifecycle_status == "superseded":
        text = f"Superseded: {epistemic}"
        view = "historical"
    elif first.lifecycle_status == "disputed":
        text = f"Disputed: {epistemic}"
        view = "disputed"
    else:
        text = epistemic
        view = "accepted"
    claim_ids = (first.claim_id,)
    statement_id = _stable_id(
        "summary_statement", RENDERER_VERSION, snapshot, "observed_fact",
        claim_ids, tuple(item.evidence_id for item in bundle), text,
    )
    return GroundedStatement(
        statement_id,
        "observed_fact",
        view,
        text,
        claim_ids,
        bundle,
        any(item.sensitivity == "sensitive" for item in bundle),
    )


def _questions(
    bundles: Sequence[tuple[SummaryEvidence, ...]], snapshot: str
) -> tuple[GroundedStatement, ...]:
    questions: list[GroundedStatement] = []
    handled_disputes: set[tuple[str, ...]] = set()
    by_claim = {bundle[0].claim_id: bundle for bundle in bundles}
    for bundle in bundles:
        first = bundle[0]
        if first.lifecycle_status == "disputed":
            claim_ids = first.linked_claim_ids
            if claim_ids in handled_disputes:
                continue
            linked = tuple(by_claim[claim_id] for claim_id in claim_ids)
            evidence = tuple(
                item for members in linked for item in members
            )
            facts = " | ".join(_fact_text(members[0]) for members in linked)
            text = f"Which disputed statement is accurate? {facts}"
            handled_disputes.add(claim_ids)
        elif first.epistemic_status == "uncertain":
            claim_ids = (first.claim_id,)
            evidence = bundle
            text = f"Is this uncertain statement accurate? {_fact_text(first)}"
        else:
            continue
        statement_id = _stable_id(
            "summary_statement", RENDERER_VERSION, snapshot,
            "unresolved_question", claim_ids,
            tuple(item.evidence_id for item in evidence), text,
        )
        questions.append(
            GroundedStatement(
                statement_id,
                "unresolved_question",
                "disputed" if len(claim_ids) > 1 else "candidate",
                text,
                claim_ids,
                evidence,
                any(item.sensitivity == "sensitive" for item in evidence),
            )
        )
    return tuple(questions)


def _fact_text(item: SummaryEvidence) -> str:
    predicate = " ".join(item.predicate.split("_"))
    return f"{item.subject_id} {predicate} {canonical_json(item.object_json)}."


def _render_summary(
    config: SummaryRendererConfig,
    facts: Sequence[GroundedStatement],
    questions: Sequence[GroundedStatement],
) -> str:
    fact_lines = "\n".join(f"- {item.text}" for item in facts) or f"- {config.empty_marker}"
    question_lines = "\n".join(f"- {item.text}" for item in questions) or f"- {config.empty_marker}"
    return (
        f"{config.observed_facts_heading}\n{fact_lines}\n\n"
        f"{config.unresolved_questions_heading}\n{question_lines}"
    )


def _aggregate_time(
    evidence: Sequence[SummaryEvidence],
) -> tuple[str, date | None, date | None, datetime | None, datetime | None]:
    if not evidence or all(item.time_precision == "unknown" for item in evidence):
        return "unknown", None, None, None, None
    representations = {
        "timestamp" if item.time_precision == "timestamp"
        else "unknown" if item.time_precision == "unknown"
        else "date"
        for item in evidence
    }
    if len(representations) != 1 or "unknown" in representations:
        return "mixed", None, None, None, None
    if representations == {"date"}:
        starts = [item.valid_from_date for item in evidence]
        ends = [item.valid_to_date for item in evidence]
        return (
            "date",
            min(starts) if all(item is not None for item in starts) else None,
            max(ends) if all(item is not None for item in ends) else None,
            None,
            None,
        )
    starts = [item.valid_from_timestamp for item in evidence]
    ends = [item.valid_to_timestamp for item in evidence]
    return (
        "timestamp",
        None,
        None,
        min(starts) if all(item is not None for item in starts) else None,
        max(ends) if all(item is not None for item in ends) else None,
    )


def _snapshot(
    request: GroundedSummaryRequest,
    config: SummaryRendererConfig,
    evidence: Sequence[SummaryEvidence],
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "config": asdict(config),
                "session": {
                    "renderer_version": request.renderer_version,
                    "session_definition_id": request.session_definition_id,
                    "session_membership_sha256": request.session_membership_sha256,
                    "user_id": request.user_id,
                },
                "evidence": [asdict(item) for item in evidence],
            }
        ).encode("utf-8")
    ).hexdigest()


def _statement_order(statement: GroundedStatement) -> tuple[object, ...]:
    first = statement.evidence[0]
    return (
        first.source_produced_at,
        first.source_order,
        -1 if first.span_start_offset is None else first.span_start_offset,
        first.claim_id,
        statement.statement_kind,
        statement.statement_id,
    )


def _evidence_order(item: SummaryEvidence) -> tuple[object, ...]:
    return (
        item.source_produced_at,
        item.source_order,
        -1 if item.span_start_offset is None else item.span_start_offset,
        item.claim_id,
        item.span_id,
    )


def _stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(canonical_json([namespace, *values]).encode("utf-8")).hexdigest()
