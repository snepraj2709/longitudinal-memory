"""Typed contracts for deterministic B2, B3, and B4 retrieval execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from .query_contracts import (
    LIFECYCLE_STATUSES,
    EligibilityResult,
    QueryPlan,
    RetrievalQueryRequest,
)


RANKING_CONFIG_VERSION = "retrieval_baseline_v1"
RERANKER_VERSION = "ontology_tie_break_v1"
BASELINE_RECORD_KINDS = {
    "B2": ("atomic",),
    "B3": ("session",),
    "B4": ("atomic", "session"),
}
CHANNEL_NAMES = (
    "atomic_lexical",
    "atomic_vector",
    "session_lexical",
    "session_vector",
    "old_version",
    "checked_relation",
)
SEARCH_CHANNELS = CHANNEL_NAMES[:4]
EXPANSION_CHANNELS = CHANNEL_NAMES[4:]
RELATION_TYPES = (
    "corrects",
    "supersedes",
    "contradicts",
    "refines",
    "same_event_as",
)
RELATION_DIRECTIONS = ("incoming", "outgoing", "symmetric")
POST_RANK_REASONS = ("no_channel_match", "outside_top_k")
CHANNEL_NOTICES = (
    "empty_fts_query",
    "empty_query_vector",
    "relation_expansion_not_applicable",
)
FAILURE_CODES = ("invalid_request", "search_failed", "stale_snapshot", "lineage_incomplete")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_FAILURE = re.compile(r"^[a-z0-9_]{1,64}$")
FIXED_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{12}$")


class BaselineRetrievalError(ValueError):
    """Reject an invalid retrieval request, score, trace, or result."""


@dataclass(frozen=True)
class BaselineRetrievalRequest:
    request: RetrievalQueryRequest
    plan: QueryPlan
    eligibility: EligibilityResult
    execution_id: str
    baseline_id: str
    k: int
    candidate_pool_size: int
    ranking_config_version: str
    ranking_config_sha256: str

    def __post_init__(self) -> None:
        _sha(self.execution_id, "execution ID")
        kinds = BASELINE_RECORD_KINDS.get(self.baseline_id)
        if kinds is None:
            raise BaselineRetrievalError("baseline ID is invalid")
        if self.request.enabled_record_kinds != kinds or self.plan.enabled_record_kinds != kinds:
            raise BaselineRetrievalError("baseline record kinds are inconsistent")
        if self.request.query_id != self.plan.query_id:
            raise BaselineRetrievalError("request and plan query IDs differ")
        if not (
            self.request.user_id
            == self.plan.user_id
            == self.eligibility.user_id
        ):
            raise BaselineRetrievalError("retrieval users differ")
        if not (
            self.request.index_version
            == self.plan.index_version
            == self.eligibility.index_version
        ):
            raise BaselineRetrievalError("retrieval index versions differ")
        if self.plan.plan_id != self.eligibility.plan_id:
            raise BaselineRetrievalError("plan and eligibility IDs differ")
        if self.k != 10 or self.candidate_pool_size != 40:
            raise BaselineRetrievalError("retrieval depth changed")
        if self.ranking_config_version != RANKING_CONFIG_VERSION:
            raise BaselineRetrievalError("ranking version changed")
        _sha(self.ranking_config_sha256, "ranking config hash")


@dataclass(frozen=True)
class SearchChannelHit:
    index_record_id: str
    record_kind: str
    channel_name: str
    rank: int
    raw_score: str

    def __post_init__(self) -> None:
        _safe_id(self.index_record_id, "index record ID")
        if self.record_kind not in {"atomic", "session"}:
            raise BaselineRetrievalError("record kind is invalid")
        if self.channel_name not in SEARCH_CHANNELS:
            raise BaselineRetrievalError("search channel is invalid")
        if not self.channel_name.startswith(f"{self.record_kind}_"):
            raise BaselineRetrievalError("search channel and record kind differ")
        _rank(self.rank, maximum=40)
        _decimal(self.raw_score, "raw score")


@dataclass(frozen=True)
class ExpansionPath:
    index_record_id: str
    seed_record_id: str
    channel_name: str
    rank: int
    relation_id: str | None = None
    relation_type: str | None = None
    direction: str | None = None

    def __post_init__(self) -> None:
        _safe_id(self.index_record_id, "expanded record ID")
        _safe_id(self.seed_record_id, "seed record ID")
        if self.index_record_id == self.seed_record_id:
            raise BaselineRetrievalError("expansion cannot return its seed")
        if self.channel_name not in EXPANSION_CHANNELS:
            raise BaselineRetrievalError("expansion channel is invalid")
        _rank(self.rank)
        relation_values = (self.relation_id, self.relation_type, self.direction)
        if self.channel_name == "old_version":
            if any(value is not None for value in relation_values):
                raise BaselineRetrievalError("old-version path cannot carry a relation")
        else:
            if any(value is None for value in relation_values):
                raise BaselineRetrievalError("relation path is incomplete")
            _safe_id(self.relation_id, "relation ID")
            if self.relation_type not in RELATION_TYPES:
                raise BaselineRetrievalError("relation type is invalid")
            if self.direction not in RELATION_DIRECTIONS:
                raise BaselineRetrievalError("relation direction is invalid")


@dataclass(frozen=True)
class RRFContribution:
    channel_name: str
    rank: int
    value: str

    def __post_init__(self) -> None:
        if self.channel_name not in CHANNEL_NAMES:
            raise BaselineRetrievalError("RRF channel is invalid")
        _rank(self.rank)
        if _decimal(self.value, "RRF contribution") < 0:
            raise BaselineRetrievalError("RRF contribution is negative")


@dataclass(frozen=True)
class RetrievedItem:
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
    component_hits: tuple[SearchChannelHit, ...]
    rrf_contributions: tuple[RRFContribution, ...]
    expansion_paths: tuple[ExpansionPath, ...]
    final_score: str
    rank: int

    def __post_init__(self) -> None:
        _safe_id(self.index_record_id, "index record ID")
        if self.record_kind not in {"atomic", "session"}:
            raise BaselineRetrievalError("record kind is invalid")
        if self.record_kind == "atomic":
            _safe_id(self.atomic_claim_version_id, "atomic claim version ID")
            if self.session_summary_id is not None:
                raise BaselineRetrievalError("atomic item has a session anchor")
        else:
            _safe_id(self.session_summary_id, "session summary ID")
            if self.atomic_claim_version_id is not None:
                raise BaselineRetrievalError("session item has an atomic anchor")
        _sha(self.content_sha256, "content hash")
        _unique_ids(self.lifecycle_statuses, "lifecycle statuses", sorted_values=False)
        if not self.lifecycle_statuses or not set(self.lifecycle_statuses).issubset(LIFECYCLE_STATUSES):
            raise BaselineRetrievalError("lifecycle statuses are invalid")
        _unique_ids(self.claim_ids, "claim IDs", sorted_values=False)
        _unique_ids(self.claim_version_ids, "claim version IDs", sorted_values=False)
        _unique_ids(self.source_ids, "source IDs", sorted_values=False)
        _unique_ids(self.span_ids, "span IDs", sorted_values=False)
        if len(self.claim_ids) != len(self.claim_version_ids):
            raise BaselineRetrievalError("claim lineage is incomplete")
        if not self.claim_ids or not self.source_ids or not self.span_ids:
            raise BaselineRetrievalError("retrieved lineage is empty")
        if any(hit.index_record_id != self.index_record_id for hit in self.component_hits):
            raise BaselineRetrievalError("component hit belongs to another record")
        if any(path.index_record_id != self.index_record_id for path in self.expansion_paths):
            raise BaselineRetrievalError("expansion path belongs to another record")
        keys = tuple((item.channel_name, item.rank) for item in self.rrf_contributions)
        expected_keys = tuple(
            (item.channel_name, item.rank)
            for item in (*self.component_hits, *self.expansion_paths)
        )
        if len(keys) != len(set(keys)) or set(keys) != set(expected_keys) or len(keys) != len(expected_keys):
            raise BaselineRetrievalError("RRF contributions are duplicated")
        final_score = _decimal(self.final_score, "final score")
        if final_score < 0:
            raise BaselineRetrievalError("final score is negative")
        if final_score != sum((_decimal(item.value, "RRF contribution") for item in self.rrf_contributions), Decimal(0)):
            raise BaselineRetrievalError("final score does not equal RRF contributions")
        _rank(self.rank, maximum=10)


@dataclass(frozen=True)
class RejectedRetrievalItem:
    index_record_id: str
    stage: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _safe_id(self.index_record_id, "rejected record ID")
        if self.stage not in {"pre_filter", "post_rank"}:
            raise BaselineRetrievalError("rejection stage is invalid")
        if not self.reasons or self.reasons != tuple(sorted(set(self.reasons))):
            raise BaselineRetrievalError("rejection reasons must be sorted and unique")
        if self.stage == "post_rank" and (
            len(self.reasons) != 1 or self.reasons[0] not in POST_RANK_REASONS
        ):
            raise BaselineRetrievalError("post-rank reason is invalid")


@dataclass(frozen=True)
class BaselineRetrievalResult:
    execution_id: str
    baseline_id: str
    query_id: str
    user_id: str
    plan_id: str
    snapshot_run_id: str
    accepted: tuple[RetrievedItem, ...]
    rejected: tuple[RejectedRetrievalItem, ...]
    channel_notices: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.execution_id, "execution ID")
        if self.baseline_id not in BASELINE_RECORD_KINDS:
            raise BaselineRetrievalError("baseline ID is invalid")
        _safe_id(self.query_id, "query ID")
        _safe_id(self.user_id, "user ID")
        _sha(self.plan_id, "plan ID")
        _safe_id(self.snapshot_run_id, "snapshot run ID")
        accepted_ids = tuple(item.index_record_id for item in self.accepted)
        if len(accepted_ids) != len(set(accepted_ids)) or len(accepted_ids) > 10:
            raise BaselineRetrievalError("accepted records are duplicated or over depth")
        if tuple(item.rank for item in self.accepted) != tuple(range(1, len(self.accepted) + 1)):
            raise BaselineRetrievalError("accepted ranks are not contiguous")
        expected_kinds = set(BASELINE_RECORD_KINDS[self.baseline_id])
        if any(item.record_kind not in expected_kinds for item in self.accepted):
            raise BaselineRetrievalError("accepted record kind is outside the baseline")
        rejected_ids = tuple(item.index_record_id for item in self.rejected)
        if len(rejected_ids) != len(set(rejected_ids)):
            raise BaselineRetrievalError("rejected records are duplicated")
        if set(accepted_ids).intersection(rejected_ids):
            raise BaselineRetrievalError("a record is both accepted and rejected")
        if self.channel_notices != tuple(sorted(set(self.channel_notices))):
            raise BaselineRetrievalError("channel notices must be sorted and unique")
        if not set(self.channel_notices).issubset(CHANNEL_NOTICES):
            raise BaselineRetrievalError("channel notice is invalid")


@dataclass(frozen=True)
class BaselineRetrievalFailure:
    failure_id: str
    execution_id: str
    baseline_id: str
    query_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        _sha(self.execution_id, "execution ID")
        if self.baseline_id not in BASELINE_RECORD_KINDS:
            raise BaselineRetrievalError("baseline ID is invalid")
        _safe_id(self.query_id, "query ID")
        _safe_id(self.user_id, "user ID")
        if self.code not in FAILURE_CODES:
            raise BaselineRetrievalError("failure code is invalid")
        if SAFE_FAILURE.fullmatch(self.location) is None:
            raise BaselineRetrievalError("failure location is not sanitized")


def _decimal(value: str, name: str) -> Decimal:
    if not isinstance(value, str) or FIXED_DECIMAL.fullmatch(value) is None:
        raise BaselineRetrievalError(f"{name} is not a fixed decimal")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise BaselineRetrievalError(f"{name} is invalid") from error
    if not parsed.is_finite():
        raise BaselineRetrievalError(f"{name} is not finite")
    return parsed


def _rank(value: int, *, maximum: int | None = None) -> None:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise BaselineRetrievalError("rank is invalid")


def _safe_id(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise BaselineRetrievalError(f"{name} is invalid")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise BaselineRetrievalError(f"{name} is invalid")


def _unique_ids(values: tuple[str, ...], name: str, *, sorted_values: bool) -> None:
    if not isinstance(values, tuple) or len(values) != len(set(values)):
        raise BaselineRetrievalError(f"{name} are duplicated")
    if sorted_values and values != tuple(sorted(values)):
        raise BaselineRetrievalError(f"{name} are not sorted")
    for value in values:
        _safe_id(value, name)
