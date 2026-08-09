"""Pure deterministic fusion, expansion, and tie-breaking for B2, B3, and B4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .baseline_contracts import (
    BASELINE_RECORD_KINDS,
    CHANNEL_NAMES,
    CHANNEL_NOTICES,
    FAILURE_CODES,
    POST_RANK_REASONS,
    RANKING_CONFIG_VERSION,
    RELATION_DIRECTIONS,
    RELATION_TYPES,
    BaselineRetrievalError,
    BaselineRetrievalRequest,
    BaselineRetrievalResult,
    ExpansionPath,
    RejectedRetrievalItem,
    RetrievedItem,
    RRFContribution,
    SearchChannelHit,
)
from .query_contracts import (
    LIFECYCLE_STATUSES,
    QUERY_LABELS,
    EligibilityResult,
    QueryPlan,
    RetrievalQueryRequest,
    canonical_json_bytes,
    request_canonical_value,
    stable_sha256,
)


CONFIG_PATH = Path("configs/retrieval/baseline_v1.json")
RRF_CONSTANT = 60
SCORE_DECIMAL_PLACES = 12
SCORE_QUANTUM = Decimal("0.000000000001")
KIND_PREFERENCES = {
    "current_state": ("atomic", "session"),
    "historical_state": ("session", "atomic"),
    "change_over_time": ("session", "atomic"),
    "specific_event": ("atomic", "session"),
    "relationship": ("session", "atomic"),
    "commitment": ("atomic", "session"),
    "evidence_request": ("atomic", "session"),
    "unknown": ("atomic", "session"),
}
LIFECYCLE_PREFERENCES = {
    "current_state": ("current", "confirmed", "candidate", "disputed", "historical", "superseded"),
    "historical_state": ("historical", "superseded", "current", "confirmed", "disputed", "candidate"),
    "change_over_time": ("historical", "superseded", "current", "confirmed", "disputed", "candidate"),
    "specific_event": ("confirmed", "current", "historical", "candidate", "disputed", "superseded"),
    "relationship": ("confirmed", "current", "historical", "candidate", "disputed", "superseded"),
    "commitment": ("confirmed", "current", "historical", "candidate", "disputed", "superseded"),
    "evidence_request": ("confirmed", "current", "historical", "candidate", "disputed", "superseded"),
    "unknown": ("confirmed", "current", "historical", "candidate", "disputed", "superseded"),
}


@dataclass(frozen=True)
class BaselineConfig:
    ranking_config_version: str
    index_version: str
    embedding_version: str
    embedding_dimension: int
    distance: str
    fts_configuration: str
    baseline_record_kinds: Mapping[str, tuple[str, ...]]
    k: int
    candidate_pool_size: int
    rrf_constant: int
    score_decimal_places: int
    score_rounding: str
    channel_order: tuple[str, ...]
    relation_expansion_types: tuple[str, ...]
    kind_preferences: Mapping[str, tuple[str, ...]]
    lifecycle_preferences: Mapping[str, tuple[str, ...]]
    reranker_version: str
    post_rank_reasons: tuple[str, ...]
    channel_notices: tuple[str, ...]
    failure_codes: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        if (
            self.ranking_config_version != RANKING_CONFIG_VERSION
            or self.index_version != "retrieval_index_v1"
            or self.embedding_version != "deterministic_token_hash_v1"
            or self.embedding_dimension != 256
            or self.distance != "cosine"
            or self.fts_configuration != "simple"
            or dict(self.baseline_record_kinds) != BASELINE_RECORD_KINDS
            or self.k != 10
            or self.candidate_pool_size != 40
            or self.rrf_constant != RRF_CONSTANT
            or self.score_decimal_places != SCORE_DECIMAL_PLACES
            or self.score_rounding != "ROUND_HALF_EVEN"
            or self.channel_order != CHANNEL_NAMES
            or self.relation_expansion_types != RELATION_TYPES
            or dict(self.kind_preferences) != KIND_PREFERENCES
            or dict(self.lifecycle_preferences) != LIFECYCLE_PREFERENCES
            or self.reranker_version != "ontology_tie_break_v1"
            or self.post_rank_reasons != POST_RANK_REASONS
            or self.channel_notices != CHANNEL_NOTICES
            or self.failure_codes != FAILURE_CODES
            or len(self.sha256) != 64
        ):
            raise BaselineRetrievalError("baseline configuration changed")


@dataclass(frozen=True)
class FusedCandidate:
    index_record_id: str
    record_kind: str
    lifecycle_statuses: tuple[str, ...]
    component_hits: tuple[SearchChannelHit, ...]
    expansion_paths: tuple[ExpansionPath, ...]
    contributions: tuple[RRFContribution, ...]
    final_score: str


@dataclass(frozen=True)
class AtomicExpansionSeed:
    index_record_id: str
    seed_rank: int
    claim_id: str


@dataclass(frozen=True)
class OldVersionNeighbor:
    index_record_id: str
    claim_id: str
    transaction_from: datetime


@dataclass(frozen=True)
class RelationNeighbor:
    seed_record_id: str
    seed_rank: int
    index_record_id: str
    relation_id: str
    relation_type: str
    direction: str


@dataclass(frozen=True)
class RetrievalRecordMetadata:
    index_record_id: str
    record_kind: str
    atomic_claim_version_id: str | None
    session_summary_id: str | None
    content_sha256: str
    lifecycle_statuses: tuple[str, ...]
    claim_ids: tuple[str, ...]
    claim_version_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    span_ids: tuple[str, ...]


def load_baseline_config(path: str | Path = CONFIG_PATH) -> BaselineConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    try:
        value = json.loads(
            raw,
            parse_constant=lambda item: (_ for _ in ()).throw(
                BaselineRetrievalError(f"invalid JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineRetrievalError("baseline configuration is invalid") from error
    expected_fields = {
        "ranking_config_version", "index_version", "embedding_version",
        "embedding_dimension", "distance", "fts_configuration",
        "baseline_record_kinds", "k", "candidate_pool_size", "rrf_constant",
        "score_decimal_places", "score_rounding", "channel_order",
        "relation_expansion_types", "kind_preferences", "lifecycle_preferences",
        "reranker_version", "post_rank_reasons", "channel_notices", "failure_codes",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise BaselineRetrievalError("baseline configuration fields changed")
    return BaselineConfig(
        value["ranking_config_version"], value["index_version"],
        value["embedding_version"], value["embedding_dimension"],
        value["distance"], value["fts_configuration"],
        _mapping_of_tuples(value["baseline_record_kinds"]), value["k"],
        value["candidate_pool_size"], value["rrf_constant"],
        value["score_decimal_places"], value["score_rounding"],
        _tuple(value["channel_order"]), _tuple(value["relation_expansion_types"]),
        _mapping_of_tuples(value["kind_preferences"]),
        _mapping_of_tuples(value["lifecycle_preferences"]),
        value["reranker_version"], _tuple(value["post_rank_reasons"]),
        _tuple(value["channel_notices"]), _tuple(value["failure_codes"]),
        hashlib.sha256(raw).hexdigest(),
    )


def build_baseline_request(
    request: RetrievalQueryRequest,
    plan: QueryPlan,
    eligibility: EligibilityResult,
    baseline_id: str,
    *,
    config: BaselineConfig,
) -> BaselineRetrievalRequest:
    execution_id = stable_sha256(
        {
            "request": request_canonical_value(request),
            "plan_id": plan.plan_id,
            "eligibility_snapshot_run_id": eligibility.snapshot_run_id,
            "eligibility_record_ids": tuple(
                (item.index_record_id, item.eligible, item.rejection_reasons)
                for item in eligibility.decisions
            ),
            "baseline_id": baseline_id,
            "ranking_config_version": config.ranking_config_version,
            "ranking_config_sha256": config.sha256,
        }
    )
    return BaselineRetrievalRequest(
        request, plan, eligibility, execution_id, baseline_id,
        config.k, config.candidate_pool_size,
        config.ranking_config_version, config.sha256,
    )


def fixed_decimal(value: Decimal | float | int | str) -> str:
    if isinstance(value, bool):
        raise BaselineRetrievalError("score is invalid")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise BaselineRetrievalError("score is invalid") from error
    if not parsed.is_finite():
        raise BaselineRetrievalError("score is not finite")
    return format(parsed.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_EVEN), ".12f")


def rrf_contribution(rank: int) -> str:
    if type(rank) is not int or rank < 1:
        raise BaselineRetrievalError("RRF rank is invalid")
    return fixed_decimal(Decimal(1) / Decimal(RRF_CONSTANT + rank))


def fuse_channels(
    hits: Sequence[SearchChannelHit],
    expansions: Sequence[ExpansionPath],
    record_attributes: Mapping[str, tuple[str, tuple[str, ...]]],
) -> tuple[FusedCandidate, ...]:
    _validate_channel_ranks(hits, expansions)
    components: dict[str, list[SearchChannelHit]] = {}
    paths: dict[str, list[ExpansionPath]] = {}
    for hit in hits:
        if hit.channel_name.endswith("_lexical") and Decimal(hit.raw_score) < 0:
            raise BaselineRetrievalError("lexical score is negative")
        components.setdefault(hit.index_record_id, []).append(hit)
    for path in expansions:
        paths.setdefault(path.index_record_id, []).append(path)
    candidates: list[FusedCandidate] = []
    for record_id in sorted(set(components) | set(paths)):
        attributes = record_attributes.get(record_id)
        if attributes is None:
            raise BaselineRetrievalError("fused record metadata is missing")
        kind, lifecycle = attributes
        record_hits = tuple(sorted(components.get(record_id, ()), key=_component_key))
        record_paths = tuple(sorted(paths.get(record_id, ()), key=_path_key))
        contributions = tuple(
            RRFContribution(item.channel_name, item.rank, rrf_contribution(item.rank))
            for item in (*record_hits, *record_paths)
        )
        total = sum((Decimal(item.value) for item in contributions), Decimal(0))
        candidates.append(
            FusedCandidate(
                record_id, kind, lifecycle, record_hits, record_paths,
                contributions, fixed_decimal(total),
            )
        )
    return tuple(candidates)


def rerank_candidates(
    candidates: Sequence[FusedCandidate],
    query_label: str,
) -> tuple[FusedCandidate, ...]:
    if query_label not in QUERY_LABELS:
        raise BaselineRetrievalError("query label is invalid")
    kind_order = KIND_PREFERENCES[query_label]
    lifecycle_order = LIFECYCLE_PREFERENCES[query_label]

    def key(item: FusedCandidate):
        if item.record_kind not in kind_order:
            raise BaselineRetrievalError("candidate record kind is invalid")
        if not item.lifecycle_statuses or not set(item.lifecycle_statuses).issubset(LIFECYCLE_STATUSES):
            raise BaselineRetrievalError("candidate lifecycle is invalid")
        lifecycle_rank = min(lifecycle_order.index(status) for status in item.lifecycle_statuses)
        return (-Decimal(item.final_score), kind_order.index(item.record_kind), lifecycle_rank, item.index_record_id)

    return tuple(sorted(candidates, key=key))


def build_old_version_paths(
    seeds: Sequence[AtomicExpansionSeed],
    neighbors: Sequence[OldVersionNeighbor],
) -> tuple[ExpansionPath, ...]:
    selected: list[tuple[int, float, str, str]] = []
    for seed in sorted(seeds, key=lambda item: (item.seed_rank, item.index_record_id))[:10]:
        if seed.seed_rank < 1:
            raise BaselineRetrievalError("seed rank is invalid")
        for neighbor in neighbors:
            if neighbor.index_record_id == seed.index_record_id or neighbor.claim_id != seed.claim_id:
                continue
            if neighbor.transaction_from.tzinfo is None or neighbor.transaction_from.utcoffset() is None:
                raise BaselineRetrievalError("old-version transaction time is naive")
            selected.append(
                (seed.seed_rank, -neighbor.transaction_from.timestamp(), neighbor.index_record_id, seed.index_record_id)
            )
    paths: list[ExpansionPath] = []
    seen: set[str] = set()
    for _, _, neighbor_id, seed_id in sorted(selected):
        if neighbor_id in seen:
            continue
        seen.add(neighbor_id)
        paths.append(ExpansionPath(neighbor_id, seed_id, "old_version", len(paths) + 1))
    return tuple(paths)


def build_relation_paths(
    neighbors: Sequence[RelationNeighbor],
) -> tuple[ExpansionPath, ...]:
    ordered = sorted(
        neighbors,
        key=lambda item: (
            item.seed_rank,
            RELATION_TYPES.index(item.relation_type) if item.relation_type in RELATION_TYPES else len(RELATION_TYPES),
            item.relation_id,
            item.index_record_id,
        ),
    )
    paths: list[ExpansionPath] = []
    seen: set[str] = set()
    for item in ordered:
        if item.seed_rank < 1 or item.relation_type not in RELATION_TYPES or item.direction not in RELATION_DIRECTIONS:
            raise BaselineRetrievalError("relation expansion input is invalid")
        if item.index_record_id == item.seed_record_id:
            raise BaselineRetrievalError("relation expansion returns its seed")
        if item.index_record_id in seen:
            continue
        seen.add(item.index_record_id)
        paths.append(
            ExpansionPath(
                item.index_record_id, item.seed_record_id, "checked_relation",
                len(paths) + 1, item.relation_id, item.relation_type, item.direction,
            )
        )
    return tuple(paths)


def build_retrieved_items(
    candidates: Sequence[FusedCandidate],
    metadata: Mapping[str, RetrievalRecordMetadata],
    *,
    k: int = 10,
) -> tuple[RetrievedItem, ...]:
    if k != 10:
        raise BaselineRetrievalError("retrieval depth changed")
    items: list[RetrievedItem] = []
    for rank, candidate in enumerate(candidates[:k], start=1):
        value = metadata.get(candidate.index_record_id)
        if value is None or value.record_kind != candidate.record_kind or value.lifecycle_statuses != candidate.lifecycle_statuses:
            raise BaselineRetrievalError("retrieval metadata changed")
        items.append(
            RetrievedItem(
                value.index_record_id, value.record_kind, value.atomic_claim_version_id,
                value.session_summary_id, value.content_sha256, value.lifecycle_statuses,
                value.claim_ids, value.claim_version_ids, value.source_ids, value.span_ids,
                candidate.component_hits, candidate.contributions, candidate.expansion_paths,
                candidate.final_score, rank,
            )
        )
    return tuple(items)


def build_rejections(
    eligibility: EligibilityResult,
    surfaced_record_ids: Sequence[str],
    accepted_record_ids: Sequence[str],
) -> tuple[RejectedRetrievalItem, ...]:
    surfaced = set(surfaced_record_ids)
    accepted = set(accepted_record_ids)
    if len(accepted) != len(tuple(accepted_record_ids)):
        raise BaselineRetrievalError("accepted IDs are duplicated")
    rejected: list[RejectedRetrievalItem] = []
    for decision in eligibility.decisions:
        if not decision.eligible:
            rejected.append(
                RejectedRetrievalItem(decision.index_record_id, "pre_filter", decision.rejection_reasons)
            )
        elif decision.index_record_id not in accepted:
            reason = "outside_top_k" if decision.index_record_id in surfaced else "no_channel_match"
            rejected.append(RejectedRetrievalItem(decision.index_record_id, "post_rank", (reason,)))
    return tuple(rejected)


def build_result(
    execution: BaselineRetrievalRequest,
    ranked_candidates: Sequence[FusedCandidate],
    metadata: Mapping[str, RetrievalRecordMetadata],
    channel_notices: Sequence[str] = (),
) -> BaselineRetrievalResult:
    accepted = build_retrieved_items(ranked_candidates, metadata, k=execution.k)
    surfaced = tuple(item.index_record_id for item in ranked_candidates)
    rejected = build_rejections(
        execution.eligibility,
        surfaced,
        tuple(item.index_record_id for item in accepted),
    )
    return BaselineRetrievalResult(
        execution.execution_id, execution.baseline_id, execution.request.query_id,
        execution.request.user_id, execution.plan.plan_id,
        execution.eligibility.snapshot_run_id, accepted, rejected,
        tuple(sorted(set(channel_notices))),
    )


def serialize_result(result: BaselineRetrievalResult) -> bytes:
    from dataclasses import asdict

    return canonical_json_bytes(asdict(result))


def _validate_channel_ranks(
    hits: Sequence[SearchChannelHit], expansions: Sequence[ExpansionPath]
) -> None:
    seen: set[tuple[str, int]] = set()
    seen_record_channel: set[tuple[str, str]] = set()
    for item in (*hits, *expansions):
        rank_key = (item.channel_name, item.rank)
        record_key = (item.index_record_id, item.channel_name)
        if rank_key in seen or record_key in seen_record_channel:
            raise BaselineRetrievalError("channel rank or record is duplicated")
        seen.add(rank_key)
        seen_record_channel.add(record_key)


def _component_key(item: SearchChannelHit) -> tuple[int, int]:
    return CHANNEL_NAMES.index(item.channel_name), item.rank


def _path_key(item: ExpansionPath) -> tuple[int, int]:
    return CHANNEL_NAMES.index(item.channel_name), item.rank


def _tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise BaselineRetrievalError("configuration list is invalid")
    return tuple(value)


def _mapping_of_tuples(value: object) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BaselineRetrievalError("configuration mapping is invalid")
    return {key: _tuple(items) for key, items in value.items()}
