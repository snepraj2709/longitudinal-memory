"""Pure, deterministic planning for evidence-backed durative claims."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import hashlib
from pathlib import Path
from typing import Iterable

from extraction.predicate_registry import (
    PredicateDefinition,
    PredicateRegistry,
    PredicateRegistryError,
    load_predicate_registry,
    validate_object_shape,
)

from .durative_contracts import (
    ALLOWED_FAMILIES,
    ELIGIBLE_EPISTEMIC,
    ELIGIBLE_LIFECYCLE,
    REGISTRY_PATH,
    REGISTRY_SHA256,
    REGISTRY_VERSION,
    RULES_VERSION,
    DurativeClaimError,
    DurativeClaimPlan,
    DurativeDecision,
    DurativeEpisode,
    DurativeEvidenceRef,
    DurativeExtractionMetadata,
    DurativePropositionPlan,
    DurativePropositionRequest,
    DurativeRulesConfig,
    canonical_json,
    load_durative_rules_config,
    stable_id,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DURATIVE_RULES_PATH = REPO_ROOT / "configs/summaries/durative_claim_rules_v1.json"
SENSITIVITY_RANK = {None: 0, "standard": 0, "sensitive": 1}


def infer_durative_claim(
    request: DurativePropositionRequest,
    episodes: Iterable[DurativeEpisode],
    *,
    config_path: str | Path = DEFAULT_DURATIVE_RULES_PATH,
) -> DurativePropositionPlan:
    """Plan one candidate durative claim without writing storage state."""

    config, rules_sha256, registry = _load_dependencies(config_path)
    ordered = tuple(sorted(episodes, key=lambda item: item.episode_id))
    if len({item.episode_id for item in ordered}) != len(ordered):
        raise DurativeClaimError("episode IDs must be unique")
    if any(item.user_id != request.user_id for item in ordered):
        raise DurativeClaimError("cross-user episodes are forbidden")
    _validate_registered_inputs(request, ordered, registry)

    snapshot_sha256 = _snapshot_sha256(
        request, ordered, config, rules_sha256, registry.content_sha256
    )
    plan_id = stable_id(
        "durative_plan", config.rules_version, request.user_id,
        request.idempotency_key,
    )
    decision_id = stable_id("durative_decision", plan_id)
    definition = registry.by_predicate[request.predicate]
    if not _predicate_is_durative(definition, config):
        return _rejected(
            request, ordered, snapshot_sha256, plan_id, decision_id,
            "predicate_not_durative", (), (),
        )

    raw_supports = tuple(
        item for item in ordered
        if item.support_type == "supports" and _same_proposition(request, item)
    )
    if not raw_supports:
        return _rejected(
            request, ordered, snapshot_sha256, plan_id, decision_id,
            "no_exact_support", (), (),
        )

    supports, rejection = _eligible_supports(request, raw_supports)
    counters = tuple(
        item for item in ordered
        if _is_counter_evidence(request, item, definition, raw_supports)
    )
    counter_ids = tuple(item.episode_id for item in counters)
    counter_id_set = set(counter_ids)
    support_ids = tuple(
        item.episode_id for item in supports
        if item.episode_id not in counter_id_set
    )
    if counters:
        return _rejected(
            request, ordered, snapshot_sha256, plan_id, decision_id,
            "counterevidence", support_ids, counter_ids,
        )
    if rejection is not None:
        return _rejected(
            request, ordered, snapshot_sha256, plan_id, decision_id,
            rejection, (), (),
        )

    explicit = tuple(item for item in supports if _has_closed_interval(item))
    distinct_sessions = {item.session_definition_id for item in supports}
    distinct_sources = {item.source_id for item in supports}
    distinct_times = {
        canonical_json(item.episode_at)
        for item in supports if item.episode_at is not None
    }
    repeated = (
        len(distinct_sessions) >= config.minimum_distinct_sessions
        and len(distinct_sources) >= config.minimum_distinct_sources
        and len(distinct_times) >= config.minimum_distinct_episode_times
    )
    if len(supports) == 1 and explicit:
        selected = explicit[0]
        time_values = (
            selected.valid_from_date, selected.valid_from_timestamp,
            selected.valid_to_date, selected.valid_to_timestamp,
            selected.time_precision,
        )
        reason = "accepted_explicit_closed_interval"
    elif repeated:
        time_values = _aggregate_episode_time(supports)
        reason = "accepted_repeated_episodes"
    elif explicit:
        time_values = _aggregate_episode_time(supports)
        reason = "accepted_explicit_closed_interval"
    else:
        return _rejected(
            request, ordered, snapshot_sha256, plan_id, decision_id,
            "insufficient_repetition", support_ids, (),
        )

    evidence = tuple(_evidence(item) for item in supports)
    sensitivity = max(
        (item.sensitivity or "standard" for item in supports),
        key=lambda item: SENSITIVITY_RANK[item],
    )
    extraction_confidence = min(item.extraction_confidence for item in supports)
    semantic = {
        "schema_version": "durative_claim_semantic_v1",
        "user_id": request.user_id,
        "subject_id": request.subject_id,
        "speaker_id": "memory_system",
        "predicate": request.predicate,
        "predicate_registry_version": request.predicate_registry_version,
        "object": request.object_json,
        "polarity": request.polarity,
        "epistemic_status": "inferred",
        "memory_kind": "durative",
    }
    semantic_sha256 = hashlib.sha256(
        canonical_json(semantic).encode("utf-8")
    ).hexdigest()
    extraction_version_id = stable_id(
        "durative_extraction_version", RULES_VERSION, rules_sha256,
        REGISTRY_VERSION, REGISTRY_SHA256,
    )
    extraction = DurativeExtractionMetadata(
        extraction_version_id=extraction_version_id,
        extractor_kind="deterministic_rules",
        rules_version=RULES_VERSION,
        rules_sha256=rules_sha256,
        predicate_registry_version=REGISTRY_VERSION,
        predicate_registry_sha256=REGISTRY_SHA256,
        input_snapshot_sha256=snapshot_sha256,
    )
    claim = DurativeClaimPlan(
        claim_id=semantic_sha256,
        semantic_sha256=semantic_sha256,
        user_id=request.user_id,
        subject_id=request.subject_id,
        speaker_id="memory_system",
        predicate=request.predicate,
        predicate_registry_version=request.predicate_registry_version,
        object_json=request.object_json,
        polarity=request.polarity,
        epistemic_status="inferred",
        lifecycle_status="candidate",
        memory_kind="durative",
        sensitivity=sensitivity,
        extraction_confidence=extraction_confidence,
        belief_confidence=None,
        valid_from_date=time_values[0],
        valid_from_timestamp=time_values[1],
        valid_to_date=time_values[2],
        valid_to_timestamp=time_values[3],
        time_precision=time_values[4],
        evidence=evidence,
        extraction=extraction,
    )
    decision = _decision(
        request, ordered, snapshot_sha256, decision_id, "accepted", reason,
        support_ids, (), claim.claim_id,
    )
    return DurativePropositionPlan(
        plan_id=plan_id,
        input_snapshot_sha256=snapshot_sha256,
        request=request,
        decision=decision,
        claim=claim,
    )


plan_durative_claim = infer_durative_claim


def _load_dependencies(
    config_path: str | Path,
) -> tuple[DurativeRulesConfig, str, PredicateRegistry]:
    path = Path(config_path)
    config = load_durative_rules_config(path)
    try:
        rules_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        registry = load_predicate_registry(REPO_ROOT / config.predicate_registry_path)
    except OSError as error:
        raise DurativeClaimError("durative dependency is unreadable") from error
    except PredicateRegistryError as error:
        raise DurativeClaimError("durative predicate registry is invalid") from error
    if (
        registry.registry_version != config.predicate_registry_version
        or registry.content_sha256 != config.predicate_registry_sha256
    ):
        raise DurativeClaimError("durative predicate registry binding changed")
    return config, rules_sha256, registry


def _validate_registered_inputs(
    request: DurativePropositionRequest,
    episodes: tuple[DurativeEpisode, ...],
    registry: PredicateRegistry,
) -> None:
    definition = registry.by_predicate.get(request.predicate)
    if definition is None:
        raise DurativeClaimError("request predicate is unknown")
    if validate_object_shape(request.object_json, definition.object_shape):
        raise DurativeClaimError("request object does not match its predicate")
    for item in episodes:
        item_definition = registry.by_predicate.get(item.predicate)
        if item_definition is None:
            raise DurativeClaimError("episode predicate is unknown")
        if validate_object_shape(item.object_json, item_definition.object_shape):
            raise DurativeClaimError("episode object does not match its predicate")


def _predicate_is_durative(
    definition: PredicateDefinition,
    config: DurativeRulesConfig,
) -> bool:
    return (
        definition.temporal_behavior == config.required_temporal_behavior
        and definition.family in ALLOWED_FAMILIES
    )


def _same_proposition(
    request: DurativePropositionRequest,
    episode: DurativeEpisode,
) -> bool:
    return (
        episode.subject_id == request.subject_id
        and episode.predicate == request.predicate
        and episode.predicate_registry_version == request.predicate_registry_version
        and episode.polarity == request.polarity
        and canonical_json(episode.object_json) == canonical_json(request.object_json)
    )


def _visible(request: DurativePropositionRequest, episode: DurativeEpisode) -> bool:
    return (
        episode.transaction_from <= request.transaction_as_of
        and (
            episode.transaction_to is None
            or request.transaction_as_of < episode.transaction_to
        )
    )


def _eligible_supports(
    request: DurativePropositionRequest,
    supports: tuple[DurativeEpisode, ...],
) -> tuple[tuple[DurativeEpisode, ...], str | None]:
    remaining = tuple(item for item in supports if item.sensitivity != "restricted")
    if not remaining:
        return (), "restricted_evidence"
    non_recursive = tuple(item for item in remaining if item.memory_kind != "durative")
    if not non_recursive:
        return (), "recursive_durative"
    remaining = tuple(item for item in non_recursive if item.memory_kind == "episodic")
    if not remaining:
        return (), "memory_kind_ineligible"
    remaining = tuple(item for item in remaining if _visible(request, item))
    if not remaining:
        return (), "not_visible"
    remaining = tuple(
        item for item in remaining if item.lifecycle_status in ELIGIBLE_LIFECYCLE
    )
    if not remaining:
        return (), "lifecycle_ineligible"
    remaining = tuple(
        item for item in remaining if item.epistemic_status in ELIGIBLE_EPISTEMIC
    )
    if not remaining:
        return (), "epistemic_ineligible"
    return remaining, None


def _is_counter_evidence(
    request: DurativePropositionRequest,
    episode: DurativeEpisode,
    definition: PredicateDefinition,
    supports: tuple[DurativeEpisode, ...],
) -> bool:
    same_identity = (
        episode.subject_id == request.subject_id
        and episode.predicate == request.predicate
        and episode.predicate_registry_version == request.predicate_registry_version
    )
    if not same_identity:
        return False
    if (
        not _visible(request, episode)
        or episode.sensitivity == "restricted"
        or episode.memory_kind == "durative"
    ):
        return False
    if set(episode.conflict_labels) & {
        "hard_contradiction", "explicit_correction", "source_disagreement",
        "retraction", "unresolved_ambiguity",
    }:
        return True
    if set(episode.relation_types) & {"contradicts", "corrects"}:
        return True
    same_object = canonical_json(episode.object_json) == canonical_json(request.object_json)
    same_polarity = episode.polarity == request.polarity
    if same_object and same_polarity:
        return (
            episode.support_type in {"contradicts", "corrects"}
            or episode.epistemic_status in {"denied", "uncertain", "hypothetical"}
            or episode.lifecycle_status in {"candidate", "disputed"}
        )
    incompatible = (
        episode.support_type == "supports"
        and (
            (same_object and not same_polarity)
            or (
                not same_object
                and same_polarity
                and definition.conflict_compatibility == "single_value"
            )
        )
    )
    return incompatible and _overlaps_or_unknown(episode, supports)


def _overlaps_or_unknown(
    counter: DurativeEpisode,
    supports: tuple[DurativeEpisode, ...],
) -> bool:
    counter_interval = _closed_boundary_pair(counter)
    if counter_interval is None:
        return True
    for support in supports:
        support_interval = _closed_boundary_pair(support)
        if support_interval is None or type(support_interval[0]) is not type(counter_interval[0]):
            return True
        if counter_interval[0] <= support_interval[1] and support_interval[0] <= counter_interval[1]:
            return True
    return False


def _closed_boundary_pair(
    episode: DurativeEpisode,
) -> tuple[date, date] | tuple[datetime, datetime] | None:
    if episode.valid_from_date is not None and episode.valid_to_date is not None:
        return episode.valid_from_date, episode.valid_to_date
    if episode.valid_from_timestamp is not None and episode.valid_to_timestamp is not None:
        return episode.valid_from_timestamp, episode.valid_to_timestamp
    return None


def _has_closed_interval(episode: DurativeEpisode) -> bool:
    return (
        episode.valid_from_date is not None and episode.valid_to_date is not None
    ) or (
        episode.valid_from_timestamp is not None
        and episode.valid_to_timestamp is not None
    )


def _aggregate_episode_time(
    supports: tuple[DurativeEpisode, ...],
) -> tuple[date | None, datetime | None, date | None, datetime | None, str]:
    if supports and all(
        item.valid_from_date is not None
        and item.valid_to_date is not None
        and item.valid_from_timestamp is None
        and item.valid_to_timestamp is None
        for item in supports
    ):
        return (
            min(item.valid_from_date for item in supports),
            None,
            max(item.valid_to_date for item in supports),
            None,
            "approximate",
        )
    if supports and all(
        item.valid_from_timestamp is not None
        and item.valid_to_timestamp is not None
        and item.valid_from_date is None
        and item.valid_to_date is None
        for item in supports
    ):
        return (
            None,
            min(item.valid_from_timestamp for item in supports),
            None,
            max(item.valid_to_timestamp for item in supports),
            "approximate",
        )
    return None, None, None, None, "unknown"


def _evidence(episode: DurativeEpisode) -> DurativeEvidenceRef:
    return DurativeEvidenceRef(
        episode_id=episode.episode_id,
        claim_id=episode.claim_id,
        claim_version_id=episode.claim_version_id,
        session_definition_id=episode.session_definition_id,
        source_id=episode.source_id,
        span_id=episode.span_id,
        support_type=episode.support_type,
        episode_at=episode.episode_at,
    )


def _snapshot_sha256(
    request: DurativePropositionRequest,
    episodes: tuple[DurativeEpisode, ...],
    config: DurativeRulesConfig,
    rules_sha256: str,
    registry_sha256: str,
) -> str:
    payload = {
        "schema_version": "durative_input_snapshot_v1",
        "request": asdict(request),
        "episodes": [asdict(item) for item in episodes],
        "rules": asdict(config),
        "rules_file_sha256": rules_sha256,
        "predicate_registry_sha256": registry_sha256,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _decision(
    request: DurativePropositionRequest,
    episodes: tuple[DurativeEpisode, ...],
    snapshot_sha256: str,
    decision_id: str,
    status: str,
    reason: str,
    support_ids: tuple[str, ...],
    counter_ids: tuple[str, ...],
    claim_id: str | None,
) -> DurativeDecision:
    inputs = tuple(item.episode_id for item in episodes)
    categorized = set(support_ids) | set(counter_ids)
    ignored_ids = tuple(item for item in inputs if item not in categorized)
    return DurativeDecision(
        decision_id=decision_id,
        user_id=request.user_id,
        rules_version=request.rules_version,
        status=status,
        reason=reason,
        input_snapshot_sha256=snapshot_sha256,
        input_episode_ids=inputs,
        support_episode_ids=tuple(sorted(support_ids)),
        counter_episode_ids=tuple(sorted(counter_ids)),
        ignored_episode_ids=ignored_ids,
        claim_id=claim_id,
    )


def _rejected(
    request: DurativePropositionRequest,
    episodes: tuple[DurativeEpisode, ...],
    snapshot_sha256: str,
    plan_id: str,
    decision_id: str,
    reason: str,
    support_ids: tuple[str, ...],
    counter_ids: tuple[str, ...],
) -> DurativePropositionPlan:
    decision = _decision(
        request, episodes, snapshot_sha256, decision_id, "rejected", reason,
        support_ids, counter_ids, None,
    )
    return DurativePropositionPlan(
        plan_id=plan_id,
        input_snapshot_sha256=snapshot_sha256,
        request=request,
        decision=decision,
        claim=None,
    )
