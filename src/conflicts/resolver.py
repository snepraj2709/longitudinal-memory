"""Deterministic belief-resolution planning over persisted conflict snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from extraction.predicate_registry import load_predicate_registry
from storage.contracts import (
    CLAIM_RELATION_TYPES,
    LIFECYCLE_STATUSES,
    SOURCE_TYPES,
    ClaimRelationRecord,
    ConflictDecisionEvidenceRecord,
    ConflictDecisionRecord,
    StorageValidationError,
)
from temporal.contracts import TemporalClaim

from .classifier import CONFLICT_LABELS


DEFAULT_RESOLVER_CONFIG_PATH = Path(
    "configs/conflicts/belief_resolver_v1.json"
)
RESOLVER_VERSION = "belief_resolver_v1"
POLICY_VERSION = "belief_resolution_rules_v1"
AUTHORITY_ORDER = (
    "official_record",
    "subject_correction",
    "subject_assertion",
    "participant_firsthand",
    "third_party_report",
    "inference",
)
EXCLUSION_PRECEDENCE = (
    "restricted",
    "hypothetical",
    "wrong_subject",
    "deleted",
    "unsupported",
)
TERMINAL_STATUSES = ("excluded", "superseded")
OFFICIAL_SOURCE_TYPES = ("calendar", "email")
OUTCOMES = frozenset(
    {
        "no_change",
        "excluded",
        "temporal_change_resolved",
        "correction_resolved",
        "refinement_resolved",
        "retraction_resolved",
        "authority_resolved",
        "disputed",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "resolver_version",
        "policy_version",
        "classifier_config_path",
        "classifier_config_sha256",
        "classifier_version",
        "predicate_registry_path",
        "predicate_registry_version",
        "predicate_registry_sha256",
        "labels",
        "authority_order",
        "decisive_authority_kind",
        "official_source_types",
        "exclusion_precedence",
        "terminal_lifecycle_statuses",
        "belief_confidence",
        "review_status",
    }
)
LABEL_RELATION_TYPE = {
    "hard_contradiction": "contradicts",
    "temporal_change": "same_topic_as",
    "explicit_correction": "corrects",
    "refinement": "refines",
    "source_disagreement": "contradicts",
    "retraction": "corrects",
    "unresolved_ambiguity": None,
    "unrelated": None,
}


class BeliefResolverError(ValueError):
    """Reject an unsafe, inconsistent, or non-persisted resolver snapshot."""


@dataclass(frozen=True)
class ResolverConfig:
    resolver_version: str
    policy_version: str
    classifier_config_path: str
    classifier_config_sha256: str
    classifier_version: str
    predicate_registry_path: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    labels: tuple[str, ...]
    authority_order: tuple[str, ...]
    decisive_authority_kind: str
    official_source_types: tuple[str, ...]
    exclusion_precedence: tuple[str, ...]
    terminal_lifecycle_statuses: tuple[str, ...]
    belief_confidence: str
    review_status: str

    def __post_init__(self) -> None:
        identity = (
            self.resolver_version,
            self.policy_version,
            self.classifier_config_path,
            self.classifier_version,
            self.predicate_registry_path,
            self.predicate_registry_version,
        )
        if identity != (
            RESOLVER_VERSION,
            POLICY_VERSION,
            "configs/conflicts/relation_classifier_v1.json",
            "relation_classifier_v1",
            "configs/extraction/predicate_registry_v2.json",
            "predicate_registry_v2",
        ):
            raise BeliefResolverError("resolver identity changed")
        if not _sha256_text(self.classifier_config_sha256) or not _sha256_text(
            self.predicate_registry_sha256
        ):
            raise BeliefResolverError("resolver dependency hash is invalid")
        if self.labels != CONFLICT_LABELS:
            raise BeliefResolverError("resolver label vocabulary changed")
        if self.authority_order != AUTHORITY_ORDER:
            raise BeliefResolverError("authority order changed")
        if self.decisive_authority_kind != "official_record":
            raise BeliefResolverError("decisive authority changed")
        if self.official_source_types != OFFICIAL_SOURCE_TYPES:
            raise BeliefResolverError("official source types changed")
        if self.exclusion_precedence != EXCLUSION_PRECEDENCE:
            raise BeliefResolverError("exclusion precedence changed")
        if self.terminal_lifecycle_statuses != TERMINAL_STATUSES:
            raise BeliefResolverError("terminal lifecycle statuses changed")
        if (
            self.belief_confidence != "null_only"
            or self.review_status != "implementation_reviewed"
        ):
            raise BeliefResolverError("resolver config is not reviewed")


@dataclass(frozen=True)
class ResolutionRequest:
    user_id: str
    decision_id: str
    transaction_as_of: datetime
    valid_at: date | datetime | None
    resolved_at: datetime
    idempotency_key: str
    resolver_version: str

    def __post_init__(self) -> None:
        for name in ("user_id", "decision_id", "idempotency_key"):
            _text(getattr(self, name), name)
        _aware(self.transaction_as_of, "transaction_as_of")
        _aware(self.resolved_at, "resolved_at")
        if self.resolved_at < self.transaction_as_of:
            raise BeliefResolverError("resolved_at precedes the transaction cutoff")
        if self.valid_at is not None:
            if isinstance(self.valid_at, datetime):
                _aware(self.valid_at, "valid_at")
            elif type(self.valid_at) is not date:
                raise BeliefResolverError("valid_at must be a date or aware datetime")
        if self.resolver_version != RESOLVER_VERSION:
            raise BeliefResolverError("request resolver version changed")


@dataclass(frozen=True)
class AuthorityEvidence:
    user_id: str
    claim_id: str
    authority_kind: str
    speaker_id: str
    subject_id: str
    predicate: str
    source_type: str
    source_id: str
    span_ids: tuple[str, ...]
    source_ingested_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "user_id",
            "claim_id",
            "speaker_id",
            "subject_id",
            "predicate",
            "source_id",
        ):
            _text(getattr(self, name), name)
        if self.authority_kind not in AUTHORITY_ORDER:
            raise BeliefResolverError("authority kind is invalid")
        if self.source_type not in SOURCE_TYPES:
            raise BeliefResolverError("authority source type is invalid")
        if (
            not isinstance(self.span_ids, tuple)
            or not self.span_ids
            or tuple(sorted(set(self.span_ids))) != self.span_ids
            or any(not isinstance(value, str) or not value.strip() for value in self.span_ids)
        ):
            raise BeliefResolverError("authority span IDs must be sorted and unique")
        _aware(self.source_ingested_at, "authority source_ingested_at")


@dataclass(frozen=True)
class PersistedResolutionInput:
    decision: ConflictDecisionRecord
    left: TemporalClaim
    right: TemporalClaim
    relations: tuple[ClaimRelationRecord, ...]
    decision_evidence: tuple[ConflictDecisionEvidenceRecord, ...]
    source_ingested_at: tuple[tuple[str, datetime], ...]
    authority: tuple[AuthorityEvidence, ...] = ()
    deleted_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ConflictDecisionRecord):
            raise BeliefResolverError("decision must be a persisted decision")
        if not isinstance(self.left, TemporalClaim) or not isinstance(
            self.right, TemporalClaim
        ):
            raise BeliefResolverError("claim inputs must be exact temporal snapshots")
        if not isinstance(self.relations, tuple) or any(
            not isinstance(value, ClaimRelationRecord) for value in self.relations
        ):
            raise BeliefResolverError("relations must be persisted relation records")
        if tuple(sorted(self.relations, key=lambda value: value.relation_id)) != self.relations:
            raise BeliefResolverError("relations must be sorted by stable ID")
        if not isinstance(self.decision_evidence, tuple) or any(
            not isinstance(value, ConflictDecisionEvidenceRecord)
            for value in self.decision_evidence
        ):
            raise BeliefResolverError("decision evidence must be persisted records")
        if (
            tuple(
                sorted(
                    self.decision_evidence,
                    key=lambda value: (
                        value.claim_id,
                        value.span_id,
                        value.support_type,
                        value.decision_evidence_id,
                    ),
                )
            )
            != self.decision_evidence
        ):
            raise BeliefResolverError("decision evidence must be canonical")
        sources: list[str] = []
        for item in self.source_ingested_at:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BeliefResolverError("source ingestion snapshot is invalid")
            source_id, ingested_at = item
            sources.append(_text(source_id, "source_id"))
            _aware(ingested_at, "source_ingested_at")
        if sources != sorted(set(sources)):
            raise BeliefResolverError("source ingestion snapshot must be canonical")
        if not isinstance(self.authority, tuple) or any(
            not isinstance(value, AuthorityEvidence) for value in self.authority
        ):
            raise BeliefResolverError("authority must be a tuple of structured records")
        authority_key = lambda value: (
            value.claim_id,
            value.source_id,
            value.span_ids,
            value.authority_kind,
        )
        if tuple(sorted(self.authority, key=authority_key)) != self.authority:
            raise BeliefResolverError("authority records must be canonical")
        if (
            not isinstance(self.deleted_source_ids, tuple)
            or tuple(sorted(set(self.deleted_source_ids))) != self.deleted_source_ids
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.deleted_source_ids
            )
        ):
            raise BeliefResolverError("deleted source IDs must be canonical")


@dataclass(frozen=True)
class LifecycleAction:
    action_id: str
    action_order: int
    claim_id: str
    from_status: str
    target_status: str
    replacement_claim_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if not _sha256_text(self.action_id):
            raise BeliefResolverError("action ID is invalid")
        if isinstance(self.action_order, bool) or not isinstance(
            self.action_order, int
        ) or self.action_order < 1:
            raise BeliefResolverError("action order must be positive")
        _text(self.claim_id, "action claim_id")
        if self.from_status not in LIFECYCLE_STATUSES:
            raise BeliefResolverError("action source status is invalid")
        if self.target_status not in LIFECYCLE_STATUSES - {"candidate"}:
            raise BeliefResolverError("action target status is invalid")
        if self.from_status == self.target_status:
            raise BeliefResolverError("action must change lifecycle status")
        if self.replacement_claim_id is not None:
            _text(self.replacement_claim_id, "replacement_claim_id")
            if self.replacement_claim_id == self.claim_id:
                raise BeliefResolverError("replacement claim must differ")
        _text(self.reason, "action reason")


@dataclass(frozen=True)
class PlannedRelation:
    relation_id: str
    user_id: str
    decision_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    input_snapshot_sha256: str

    def __post_init__(self) -> None:
        if not _sha256_text(self.relation_id):
            raise BeliefResolverError("planned relation ID is invalid")
        for name in (
            "user_id", "decision_id", "source_claim_id", "target_claim_id"
        ):
            _text(getattr(self, name), name)
        if self.source_claim_id == self.target_claim_id:
            raise BeliefResolverError("planned relation claims must differ")
        if self.relation_type != "supersedes" or self.relation_type not in CLAIM_RELATION_TYPES:
            raise BeliefResolverError("resolver may only plan supersedes relations")
        if not _sha256_text(self.input_snapshot_sha256):
            raise BeliefResolverError("planned relation snapshot is invalid")


@dataclass(frozen=True)
class BeliefResolutionPlan:
    resolution_id: str
    resolver_version: str
    policy_version: str
    user_id: str
    decision_id: str
    idempotency_key: str
    transaction_as_of: datetime
    valid_at: date | datetime | None
    resolved_at: datetime
    outcome: str
    selected_current_claim_id: str | None
    actions: tuple[LifecycleAction, ...]
    relations: tuple[PlannedRelation, ...]
    decision_relation_ids: tuple[str, ...]
    decision_evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    authority_reason: str
    input_snapshot_sha256: str
    belief_confidence: None = None

    def __post_init__(self) -> None:
        if not _sha256_text(self.resolution_id) or not _sha256_text(
            self.input_snapshot_sha256
        ):
            raise BeliefResolverError("resolution identity is invalid")
        if self.resolver_version != RESOLVER_VERSION or self.policy_version != POLICY_VERSION:
            raise BeliefResolverError("resolution version changed")
        for name in ("user_id", "decision_id", "idempotency_key", "authority_reason"):
            _text(getattr(self, name), name)
        _aware(self.transaction_as_of, "transaction_as_of")
        _aware(self.resolved_at, "resolved_at")
        if self.valid_at is not None:
            if isinstance(self.valid_at, datetime):
                _aware(self.valid_at, "valid_at")
            elif type(self.valid_at) is not date:
                raise BeliefResolverError("valid_at is invalid")
        if self.outcome not in OUTCOMES:
            raise BeliefResolverError("resolution outcome is invalid")
        if self.selected_current_claim_id is not None:
            _text(self.selected_current_claim_id, "selected_current_claim_id")
        if not isinstance(self.actions, tuple) or tuple(
            value.action_order for value in self.actions
        ) != tuple(range(1, len(self.actions) + 1)):
            raise BeliefResolverError("lifecycle actions must be ordered")
        if len({value.claim_id for value in self.actions}) != len(self.actions):
            raise BeliefResolverError("a plan may act on each claim only once")
        for name in ("decision_relation_ids", "decision_evidence_ids", "source_ids"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or tuple(sorted(set(value))) != value:
                raise BeliefResolverError(f"{name} must be sorted and unique")
        if self.belief_confidence is not None:
            raise BeliefResolverError("belief confidence must remain null")


def load_resolver_config(
    path: str | Path = DEFAULT_RESOLVER_CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> ResolverConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeliefResolverError("resolver config is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise BeliefResolverError("resolver config fields changed")
    tuple_fields = {
        "labels",
        "authority_order",
        "official_source_types",
        "exclusion_precedence",
        "terminal_lifecycle_statuses",
    }
    if any(not isinstance(value[name], list) for name in tuple_fields):
        raise BeliefResolverError("resolver config vocabularies must be lists")
    return ResolverConfig(
        **{
            **value,
            **{name: tuple(value[name]) for name in tuple_fields},
        }
    )


class BeliefResolver:
    """Validate frozen dependencies and produce a persistence-ready plan."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_RESOLVER_CONFIG_PATH,
        repo_root: str | Path = ".",
    ) -> None:
        root = Path(repo_root).resolve()
        self.config = load_resolver_config(config_path, repo_root=root)
        classifier_path = _inside(root, self.config.classifier_config_path)
        if _file_sha256(classifier_path) != self.config.classifier_config_sha256:
            raise BeliefResolverError("classifier config binding changed")
        registry_path = _inside(root, self.config.predicate_registry_path)
        registry = load_predicate_registry(registry_path)
        if (
            registry.registry_version != self.config.predicate_registry_version
            or registry.content_sha256 != self.config.predicate_registry_sha256
        ):
            raise BeliefResolverError("predicate registry binding changed")

    def plan(
        self,
        request: ResolutionRequest,
        persisted: PersistedResolutionInput,
    ) -> BeliefResolutionPlan:
        return resolve_belief(self.config, request, persisted)


def resolve_belief(
    config: ResolverConfig,
    request: ResolutionRequest,
    persisted: PersistedResolutionInput,
) -> BeliefResolutionPlan:
    _validate_input(config, request, persisted)
    snapshot = _snapshot_hash(config, request, persisted)
    resolution_id = _stable_id(
        "belief_resolution",
        config.resolver_version,
        config.policy_version,
        request.user_id,
        request.decision_id,
        request.idempotency_key,
        request.resolved_at,
        snapshot,
    )
    actions: list[LifecycleAction] = []
    planned_relations: list[PlannedRelation] = []
    selected: str | None = None
    authority_reason = "not_applicable"
    claims = (persisted.left, persisted.right)

    exclusions = [
        (item, _exclusion_reason(request, persisted, item)) for item in claims
    ]
    exclusions = [(item, reason) for item, reason in exclusions if reason is not None]
    if exclusions:
        exclusions.sort(
            key=lambda value: (
                config.exclusion_precedence.index(value[1]),
                value[0].claim.claim_id,
            )
        )
        for item, reason in exclusions:
            _append_action(
                actions,
                resolution_id,
                item,
                "excluded",
                f"exclusion:{reason}",
            )
        outcome = "excluded"
        authority_reason = "excluded_before_resolution"
    else:
        label = persisted.decision.label
        if label == "unrelated":
            outcome = "no_change"
        elif label == "temporal_change":
            earlier, later = _temporal_order(claims)
            _append_action(
                actions,
                resolution_id,
                earlier,
                "historical",
                "temporal_change:earlier_finite_period",
            )
            later_status = _time_appropriate_status(later, request.valid_at)
            _append_action(
                actions,
                resolution_id,
                later,
                later_status,
                "temporal_change:later_period",
            )
            selected = later.claim.claim_id if later_status == "current" else None
            outcome = "temporal_change_resolved"
        elif label in {"explicit_correction", "refinement", "retraction"}:
            source, target = _directed_claims(persisted)
            if label == "retraction":
                _append_action(
                    actions,
                    resolution_id,
                    target,
                    "superseded",
                    "retraction:proposition_withdrawn",
                )
                _append_action(
                    actions,
                    resolution_id,
                    source,
                    "confirmed",
                    "retraction:event_preserved",
                )
                outcome = "retraction_resolved"
            else:
                reason = "correction" if label == "explicit_correction" else "refinement"
                _append_action(
                    actions,
                    resolution_id,
                    target,
                    "superseded",
                    f"{reason}:replaced_claim",
                    (
                        source.claim.claim_id
                        if source.version.lifecycle_status not in TERMINAL_STATUSES
                        else None
                    ),
                )
                source_status = _time_appropriate_status(source, request.valid_at)
                _append_action(
                    actions,
                    resolution_id,
                    source,
                    source_status,
                    f"{reason}:replacement_claim",
                )
                if source_status == "current":
                    selected = source.claim.claim_id
                if source_status not in TERMINAL_STATUSES:
                    planned_relations.append(
                        _planned_supersedes(
                            resolution_id,
                            request,
                            snapshot,
                            source.claim.claim_id,
                            target.claim.claim_id,
                        )
                    )
                outcome = (
                    "correction_resolved"
                    if label == "explicit_correction"
                    else "refinement_resolved"
                )
        elif label in {"hard_contradiction", "source_disagreement"}:
            decisive = _decisive_claim(config, request, persisted)
            if decisive is None:
                for item in claims:
                    _append_action(
                        actions,
                        resolution_id,
                        item,
                        "disputed",
                        f"{label}:authority_not_decisive",
                    )
                outcome = "disputed"
                authority_reason = "no_exact_decisive_official_record"
            else:
                loser = persisted.right if decisive is persisted.left else persisted.left
                _append_action(
                    actions,
                    resolution_id,
                    loser,
                    "disputed",
                    f"{label}:nonselected_claim",
                )
                winner_status = _time_appropriate_status(decisive, request.valid_at)
                _append_action(
                    actions,
                    resolution_id,
                    decisive,
                    winner_status,
                    f"{label}:decisive_official_record",
                )
                selected = (
                    decisive.claim.claim_id if winner_status == "current" else None
                )
                outcome = "authority_resolved"
                authority_reason = "exact_current_official_record"
        else:
            for item in claims:
                _append_action(
                    actions,
                    resolution_id,
                    item,
                    "disputed",
                    "unresolved_ambiguity:preserve_both",
                )
            outcome = "disputed"
            authority_reason = "authority_cannot_override_ambiguity"

    source_ids = tuple(
        sorted(
            {
                evidence.source_id
                for item in claims
                for evidence in item.evidence
            }
        )
    )
    return BeliefResolutionPlan(
        resolution_id=resolution_id,
        resolver_version=config.resolver_version,
        policy_version=config.policy_version,
        user_id=request.user_id,
        decision_id=request.decision_id,
        idempotency_key=request.idempotency_key,
        transaction_as_of=request.transaction_as_of,
        valid_at=request.valid_at,
        resolved_at=request.resolved_at,
        outcome=outcome,
        selected_current_claim_id=selected,
        actions=tuple(actions),
        relations=tuple(planned_relations),
        decision_relation_ids=tuple(value.relation_id for value in persisted.relations),
        decision_evidence_ids=tuple(
            sorted(value.decision_evidence_id for value in persisted.decision_evidence)
        ),
        source_ids=source_ids,
        authority_reason=authority_reason,
        input_snapshot_sha256=snapshot,
        belief_confidence=None,
    )


def _validate_input(
    config: ResolverConfig,
    request: ResolutionRequest,
    persisted: PersistedResolutionInput,
) -> None:
    if request.resolver_version != config.resolver_version:
        raise BeliefResolverError("request and config versions differ")
    decision = persisted.decision
    if decision.decision_id != request.decision_id:
        raise BeliefResolverError("request decision differs from persisted decision")
    if decision.user_id != request.user_id:
        raise BeliefResolverError("decision user differs from request user")
    if decision.classifier_version != config.classifier_version:
        raise BeliefResolverError("decision classifier version changed")
    if decision.label not in config.labels or decision.matched_rule != decision.label:
        raise BeliefResolverError("decision label is invalid")
    if decision.transaction_as_of > request.transaction_as_of:
        raise BeliefResolverError("decision cutoff is after the resolution cutoff")
    if decision.classified_at > request.transaction_as_of:
        raise BeliefResolverError("decision was classified after the cutoff")
    left, right = persisted.left, persisted.right
    if (
        left.claim.claim_id != decision.left_claim_id
        or right.claim.claim_id != decision.right_claim_id
        or left.version.version_id != decision.left_version_id
        or right.version.version_id != decision.right_version_id
    ):
        raise BeliefResolverError("claim snapshots differ from the persisted decision")
    for item in (left, right):
        claim, version = item.claim, item.version
        if (
            claim.user_id != request.user_id
            or version.user_id != request.user_id
            or version.claim_id != claim.claim_id
        ):
            raise BeliefResolverError("claim ownership differs from the request")
        if not version.transaction_contains(request.transaction_as_of):
            raise BeliefResolverError("claim version is not visible at the cutoff")
        if version.belief_confidence is not None:
            raise BeliefResolverError("input belief confidence must remain null")
        evidence_keys = [(value.source_id, value.span_id) for value in item.evidence]
        if evidence_keys != sorted(set(evidence_keys)):
            raise BeliefResolverError("claim evidence must be canonical")
    relation_type = LABEL_RELATION_TYPE[decision.label]
    if relation_type is None:
        if persisted.relations:
            raise BeliefResolverError("decision label must not have checked relations")
    elif len(persisted.relations) != 1 or persisted.relations[0].relation_type != relation_type:
        raise BeliefResolverError("checked relation differs from the decision label")
    pair_ids = {decision.left_claim_id, decision.right_claim_id}
    for relation in persisted.relations:
        if (
            relation.user_id != request.user_id
            or relation.decision_id != decision.decision_id
            or relation.classifier_version != decision.classifier_version
            or relation.input_snapshot_sha256 != decision.input_snapshot_sha256
            or {relation.source_claim_id, relation.target_claim_id} != pair_ids
        ):
            raise BeliefResolverError("checked relation provenance changed")
    expected_evidence = {
        (item.claim.claim_id, evidence.span_id)
        for item in (left, right)
        for evidence in item.evidence
    }
    actual_evidence = {
        (value.claim_id, value.span_id) for value in persisted.decision_evidence
    }
    if actual_evidence != expected_evidence or len(actual_evidence) != len(
        persisted.decision_evidence
    ):
        raise BeliefResolverError("decision evidence differs from visible evidence")
    for value in persisted.decision_evidence:
        if (
            value.user_id != request.user_id
            or value.decision_id != decision.decision_id
            or value.input_snapshot_sha256 != decision.input_snapshot_sha256
            or value.claim_id not in pair_ids
        ):
            raise BeliefResolverError("decision evidence provenance changed")
    source_ids = tuple(
        sorted(
            {
                evidence.source_id
                for item in (left, right)
                for evidence in item.evidence
            }
        )
    )
    ingested = dict(persisted.source_ingested_at)
    if tuple(ingested) != source_ids:
        raise BeliefResolverError("source ingestion snapshot differs from evidence")
    if any(value > request.transaction_as_of for value in ingested.values()):
        raise BeliefResolverError("future evidence is not resolver-visible")
    for authority in persisted.authority:
        if authority.user_id != request.user_id or authority.claim_id not in pair_ids:
            raise BeliefResolverError("authority ownership changed")
        if authority.source_ingested_at != ingested.get(authority.source_id):
            raise BeliefResolverError("authority source cutoff changed")


def _exclusion_reason(
    request: ResolutionRequest,
    persisted: PersistedResolutionInput,
    item: TemporalClaim,
) -> str | None:
    claim_id = item.claim.claim_id
    evidence_ids = {
        value.span_id
        for value in persisted.decision_evidence
        if value.claim_id == claim_id
    }
    source_ids = {value.source_id for value in item.evidence}
    reasons = {
        "restricted": item.claim.sensitivity == "restricted",
        "hypothetical": item.claim.epistemic_status == "hypothetical",
        "wrong_subject": item.claim.subject_id != request.user_id,
        "deleted": bool(source_ids)
        and source_ids <= set(persisted.deleted_source_ids),
        "unsupported": not item.evidence or not evidence_ids,
    }
    return next((name for name in EXCLUSION_PRECEDENCE if reasons[name]), None)


def _append_action(
    actions: list[LifecycleAction],
    resolution_id: str,
    item: TemporalClaim,
    target_status: str,
    reason: str,
    replacement_claim_id: str | None = None,
) -> None:
    from_status = item.version.lifecycle_status
    if from_status in TERMINAL_STATUSES or from_status == target_status:
        return
    order = len(actions) + 1
    actions.append(
        LifecycleAction(
            action_id=_stable_id(
                "belief_lifecycle_action",
                resolution_id,
                order,
                item.claim.claim_id,
                from_status,
                target_status,
                replacement_claim_id,
                reason,
            ),
            action_order=order,
            claim_id=item.claim.claim_id,
            from_status=from_status,
            target_status=target_status,
            replacement_claim_id=replacement_claim_id,
            reason=reason,
        )
    )


def _temporal_order(
    claims: tuple[TemporalClaim, TemporalClaim],
) -> tuple[TemporalClaim, TemporalClaim]:
    intervals = [(_finite_interval(value), value) for value in claims]
    if any(interval is None for interval, _ in intervals):
        raise BeliefResolverError("temporal change requires finite claim intervals")
    assert intervals[0][0] is not None and intervals[1][0] is not None
    if intervals[0][0][0] != intervals[1][0][0]:
        raise BeliefResolverError("temporal change cannot mix time representations")
    if intervals[0][0][2] >= intervals[1][0][1] and intervals[1][0][2] >= intervals[0][0][1]:
        raise BeliefResolverError("temporal change intervals must not overlap")
    intervals.sort(key=lambda value: (value[0][1], value[1].claim.claim_id))
    return intervals[0][1], intervals[1][1]


def _directed_claims(
    persisted: PersistedResolutionInput,
) -> tuple[TemporalClaim, TemporalClaim]:
    relation = persisted.relations[0]
    by_id = {
        persisted.left.claim.claim_id: persisted.left,
        persisted.right.claim.claim_id: persisted.right,
    }
    return by_id[relation.source_claim_id], by_id[relation.target_claim_id]


def _time_appropriate_status(
    item: TemporalClaim,
    valid_at: date | datetime | None,
) -> str:
    version = item.version
    if version.lifecycle_status in TERMINAL_STATUSES:
        return version.lifecycle_status
    if valid_at is not None and version.time_precision not in {"unknown", "approximate"}:
        try:
            if version.valid_contains(valid_at):
                return "current"
        except StorageValidationError:
            pass
    interval = _finite_interval(item)
    if valid_at is not None and interval is not None:
        representation, _, end = interval
        if (
            (representation == "date" and type(valid_at) is date)
            or (representation == "timestamp" and isinstance(valid_at, datetime))
        ) and end < valid_at:
            return "historical"
    return "confirmed"


def _finite_interval(
    item: TemporalClaim,
) -> tuple[str, date | datetime, date | datetime] | None:
    version = item.version
    if version.time_precision in {"unknown", "approximate"}:
        return None
    if version.time_precision == "timestamp":
        if version.valid_from_timestamp is None or version.valid_to_timestamp is None:
            return None
        return "timestamp", version.valid_from_timestamp, version.valid_to_timestamp
    if version.valid_from_date is None or version.valid_to_date is None:
        return None
    return "date", version.valid_from_date, version.valid_to_date


def _decisive_claim(
    config: ResolverConfig,
    request: ResolutionRequest,
    persisted: PersistedResolutionInput,
) -> TemporalClaim | None:
    matches: set[str] = set()
    by_id = {
        persisted.left.claim.claim_id: persisted.left,
        persisted.right.claim.claim_id: persisted.right,
    }
    for authority in persisted.authority:
        item = by_id[authority.claim_id]
        claim = item.claim
        if (
            authority.authority_kind != config.decisive_authority_kind
            or authority.source_type not in config.official_source_types
            or authority.source_ingested_at > request.transaction_as_of
            or authority.speaker_id != claim.speaker_id
            or authority.subject_id != claim.subject_id
            or authority.predicate != claim.predicate
            or request.valid_at is None
            or claim.time_precision in {"unknown", "approximate"}
        ):
            continue
        evidence_span_ids = tuple(
            sorted(
                evidence.span_id
                for evidence in item.evidence
                if evidence.source_id == authority.source_id
            )
        )
        if authority.span_ids != evidence_span_ids:
            continue
        try:
            current = item.version.valid_contains(request.valid_at)
        except StorageValidationError:
            current = False
        if current:
            matches.add(claim.claim_id)
    return by_id[next(iter(matches))] if len(matches) == 1 else None


def _planned_supersedes(
    resolution_id: str,
    request: ResolutionRequest,
    snapshot: str,
    source_claim_id: str,
    target_claim_id: str,
) -> PlannedRelation:
    return PlannedRelation(
        relation_id=_stable_id(
            "belief_relation",
            resolution_id,
            source_claim_id,
            target_claim_id,
            "supersedes",
            snapshot,
        ),
        user_id=request.user_id,
        decision_id=request.decision_id,
        source_claim_id=source_claim_id,
        target_claim_id=target_claim_id,
        relation_type="supersedes",
        input_snapshot_sha256=snapshot,
    )


def _snapshot_hash(
    config: ResolverConfig,
    request: ResolutionRequest,
    persisted: PersistedResolutionInput,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "config": asdict(config),
                "request": asdict(request),
                "persisted": asdict(persisted),
            }
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: object) -> object:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BeliefResolverError("snapshot JSON keys must be strings")
            result[key] = _json_value(item)
        return result
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BeliefResolverError("snapshot JSON must be finite")
        return value
    raise BeliefResolverError("snapshot input is not JSON-safe")


def _stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(
        _canonical_json([namespace, *values]).encode("utf-8")
    ).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise BeliefResolverError("resolver dependency path leaves repository") from None
    return path


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BeliefResolverError("resolver dependency is unreadable") from error


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BeliefResolverError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise BeliefResolverError(f"{name} must be timezone-aware")
    return value


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
