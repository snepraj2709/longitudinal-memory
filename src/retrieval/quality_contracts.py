"""Typed scorer-only contracts for the Step 7.4 development evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


BASELINES = ("B2", "B3", "B4")
QUERY_TYPES = (
    "current_state",
    "historical_state",
    "change_over_time",
    "specific_event",
    "relationship",
    "commitment",
    "evidence_request",
    "unknown",
)
CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "abstention",
    "user_modeling",
)
DIFFICULTIES = ("focused", "multi_source")
SOURCE_TYPES = ("calendar", "chat", "conversation", "email")
RECORD_KINDS = ("atomic", "session")
STALE_REASONS = ("corrected", "expired", "misleading", "outdated")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RetrievalQualityError(RuntimeError):
    """Fail closed on malformed scorer data or incomplete accounting."""


@dataclass(frozen=True)
class RelevanceAnnotation:
    index_record_id: str
    record_kind: str
    relevance_grade: int
    stale_for_query: bool
    stale_reason: str | None
    source_types: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.index_record_id) is None:
            raise RetrievalQualityError("annotation record ID is invalid")
        if self.record_kind not in RECORD_KINDS:
            raise RetrievalQualityError("annotation record kind is invalid")
        if type(self.relevance_grade) is not int or self.relevance_grade not in (0, 1, 2):
            raise RetrievalQualityError("annotation grade is invalid")
        if type(self.stale_for_query) is not bool:
            raise RetrievalQualityError("annotation stale flag is invalid")
        if self.stale_for_query != (self.stale_reason is not None):
            raise RetrievalQualityError("annotation stale reason is inconsistent")
        if self.stale_reason is not None and self.stale_reason not in STALE_REASONS:
            raise RetrievalQualityError("annotation stale reason is invalid")
        if not self.source_types or self.source_types != tuple(sorted(set(self.source_types))):
            raise RetrievalQualityError("annotation source types are invalid")
        if any(value not in SOURCE_TYPES for value in self.source_types):
            raise RetrievalQualityError("annotation source type is invalid")
        if self.review_status != "implementation_reviewed":
            raise RetrievalQualityError("annotation review status is invalid")


@dataclass(frozen=True)
class RelevanceCase:
    case_id: str
    query_id: str
    user_id: str
    split: str
    query_type: str
    capability: str
    difficulty: str
    annotations: tuple[RelevanceAnnotation, ...]

    def __post_init__(self) -> None:
        if not self.case_id.startswith("baseline_case_"):
            raise RetrievalQualityError("relevance case ID is invalid")
        if not self.query_id.startswith("baseline_query_"):
            raise RetrievalQualityError("relevance query ID is invalid")
        if self.user_id not in ("user_001", "user_002") or self.split != "development":
            raise RetrievalQualityError("relevance ownership or split is invalid")
        if self.query_type not in QUERY_TYPES:
            raise RetrievalQualityError("relevance query type is invalid")
        if self.capability not in CAPABILITIES or self.difficulty not in DIFFICULTIES:
            raise RetrievalQualityError("relevance slice is invalid")
        ids = tuple(item.index_record_id for item in self.annotations)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise RetrievalQualityError("annotations are not unique stable-ID order")


@dataclass(frozen=True)
class MetricValue:
    numerator: str | int
    denominator: str | int
    value: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if self.value is None:
            if not self.null_reason:
                raise RetrievalQualityError("null metric needs a reason")
        elif self.null_reason is not None:
            raise RetrievalQualityError("defined metric cannot have a null reason")


@dataclass(frozen=True)
class PerQueryQuality:
    case_id: str
    query_id: str
    user_id: str
    baseline_id: str
    query_type: str
    capability: str
    difficulty: str
    source_types: tuple[str, ...]
    recall_at_5: MetricValue
    recall_at_10: MetricValue
    ndcg_at_10: MetricValue
    mrr: MetricValue
    relevant_session_recall: MetricValue
    stale_memory_rate: MetricValue
    accepted_count: int
    relevant_enabled_count: int
    stale_accepted_count: int

    def __post_init__(self) -> None:
        if self.baseline_id not in BASELINES:
            raise RetrievalQualityError("per-query baseline is invalid")
        if self.accepted_count < 0 or self.relevant_enabled_count < 0 or self.stale_accepted_count < 0:
            raise RetrievalQualityError("per-query count is invalid")


@dataclass(frozen=True)
class QualityScorecard:
    evaluation_version: str
    query_count: int
    result_count: int
    annotation_count: int
    latency_sample_count: int
    quality_rows: tuple[Mapping[str, object], ...]
    latency_rows: tuple[Mapping[str, object], ...]
    cross_user_count: int
    runtime_failure_count: int
    model_usage: Mapping[str, int | str]
    blind_evaluation: bool
    prior_ranked_result_exposure_possible: bool

    def __post_init__(self) -> None:
        if self.evaluation_version != "retrieval_quality_development_v1":
            raise RetrievalQualityError("scorecard version is invalid")
        if (self.query_count, self.result_count, self.annotation_count, self.latency_sample_count) != (
            8,
            24,
            200,
            240,
        ):
            raise RetrievalQualityError("scorecard accounting is incomplete")
        if self.cross_user_count or self.runtime_failure_count:
            raise RetrievalQualityError("scorecard contains leakage or failure")
        if self.blind_evaluation or not self.prior_ranked_result_exposure_possible:
            raise RetrievalQualityError("evaluation disclosure is inaccurate")
        if any(value not in (0, "$0") for value in self.model_usage.values()):
            raise RetrievalQualityError("model use is forbidden")
