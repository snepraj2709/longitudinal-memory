"""Contracts for the frozen, zero-call comparable answer run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Mapping

from retrieval.query_contracts import RequestedValidTime

from .contracts import canonical_json_bytes, stable_sha256


RUN_VERSION = "comparable_memory_answer_run_v1"
RUNTIME_RELEASE = "memory_answer_quality_development_runtime_v1"
SCORER_VERSION = "memory_answer_quality_scorer_v1"
FINAL_RELEASE = "memory_answer_quality_development_v1"
PROMPT_VERSION = "memory_answer_prompt_v1"
PROMPT_SHA256 = "69dd688430c55fc35a16369201ef8eea7d470d10f610edec254f5cbd59bf14c1"
MODEL = "gpt-4.1-2025-04-14"
BASELINES = ("B2", "B3", "B4", "B5", "B6")
AVAILABLE_BASELINES = ("B2", "B3", "B4")
DEFERRED_BASELINES = ("B5", "B6")
BASELINE_ORDER = {value: index for index, value in enumerate(BASELINES)}
METRICS = (
    "strict_answer_correctness",
    "lenient_answer_correctness",
    "evidence_precision",
    "evidence_recall",
    "citation_correctness",
    "unsupported_claim_rate",
    "current_vs_outdated_fact_error_rate",
)
NULL_REASONS = {
    "strict_answer_correctness": "no_non_abstained_predictions",
    "lenient_answer_correctness": "no_non_abstained_predictions",
    "evidence_precision": "no_predicted_citations",
    "evidence_recall": "no_authorized_answer_evidence_gold",
    "citation_correctness": "no_predicted_citations",
    "unsupported_claim_rate": "no_factual_statements",
    "current_vs_outdated_fact_error_rate": "no_factual_statements",
}


class AnswerRunError(ValueError):
    """A deterministic, sanitized run-contract failure."""


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AnswerRunError(f"{name} is invalid")
    _json_safe(value)


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AnswerRunError(f"{name} is invalid")


def _commit(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AnswerRunError(f"{name} is invalid")


def _count(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise AnswerRunError(f"{name} is invalid")


def _json_safe(value: object) -> None:
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise AnswerRunError("value is not JSON-safe") from error


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AnswerRunError(f"{name} must be timezone-aware")


def _sorted_unique(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise AnswerRunError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class ComparableAnswerRunConfig:
    run_version: str
    runtime_release: str
    scorer_version: str
    final_release: str
    input_answer_version: str
    input_package_version: str
    prompt_version: str
    prompt_sha256: str
    requested_model: str
    configured_resolved_model: str
    provider_returned_model: None
    generation_settings: Mapping[str, object]
    available_baselines: tuple[str, ...]
    deferred_baselines: tuple[str, ...]
    expected_prediction_count: int
    expected_baseline_case_counts: Mapping[str, int]
    metric_names: tuple[str, ...]
    null_reasons: Mapping[str, str]
    hard_maximum_run_cost_usd: str
    historical_openai_spend_usd: str

    def __post_init__(self) -> None:
        if (
            self.run_version != RUN_VERSION
            or self.runtime_release != RUNTIME_RELEASE
            or self.scorer_version != SCORER_VERSION
            or self.final_release != FINAL_RELEASE
            or self.input_answer_version != "memory_answer_v1"
            or self.input_package_version != "evidence_package_v1"
            or self.prompt_version != PROMPT_VERSION
            or self.prompt_sha256 != PROMPT_SHA256
            or self.requested_model != MODEL
            or self.configured_resolved_model != MODEL
            or self.provider_returned_model is not None
            or self.available_baselines != AVAILABLE_BASELINES
            or self.deferred_baselines != DEFERRED_BASELINES
            or self.expected_prediction_count != 24
            or dict(self.expected_baseline_case_counts) != {
                "B2": 8, "B3": 8, "B4": 8, "B5": 0, "B6": 0
            }
            or self.metric_names != METRICS
            or dict(self.null_reasons) != NULL_REASONS
            or self.hard_maximum_run_cost_usd != "0"
            or self.historical_openai_spend_usd != "0.2314404"
        ):
            raise AnswerRunError("comparable answer configuration changed")
        if dict(self.generation_settings) != {
            "provider": "OpenAI",
            "api": "responses",
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "store": False,
            "text_format": "json_object",
        }:
            raise AnswerRunError("generation settings changed")
        _json_safe(asdict(self))


@dataclass(frozen=True)
class NoCallPreflight:
    run_version: str
    runtime_cases_transmitted: tuple[str, ...]
    users_transmitted: tuple[str, ...]
    source_ids_transmitted: tuple[str, ...]
    fields_transmitted: tuple[str, ...]
    planned_requests: int
    maximum_retry_requests: int
    estimated_input_tokens: int
    expected_output_tokens: int
    hard_output_token_ceiling: int
    expected_run_cost_usd: str
    hard_maximum_run_cost_usd: str
    spend_already_used_usd: str
    new_cumulative_ceiling_usd: str
    provider_returned_model: None
    oracle_and_gold_transmitted: bool
    eligible_provider_case_count: int
    package_count: int

    def __post_init__(self) -> None:
        if self.run_version != RUN_VERSION:
            raise AnswerRunError("preflight run version changed")
        for values, name in (
            (self.runtime_cases_transmitted, "runtime cases"),
            (self.users_transmitted, "users"),
            (self.source_ids_transmitted, "source IDs"),
            (self.fields_transmitted, "fields"),
        ):
            if values:
                raise AnswerRunError(f"{name} were transmitted")
        for value, name in (
            (self.planned_requests, "planned requests"),
            (self.maximum_retry_requests, "maximum retry requests"),
            (self.estimated_input_tokens, "estimated input tokens"),
            (self.expected_output_tokens, "expected output tokens"),
            (self.hard_output_token_ceiling, "hard output-token ceiling"),
            (self.eligible_provider_case_count, "eligible provider cases"),
        ):
            if value != 0:
                raise AnswerRunError(f"{name} must be zero")
        if (
            self.expected_run_cost_usd != "0"
            or self.hard_maximum_run_cost_usd != "0"
            or self.spend_already_used_usd != "0.2314404"
            or self.new_cumulative_ceiling_usd != "0.2314404"
            or self.provider_returned_model is not None
            or self.oracle_and_gold_transmitted is not False
            or self.package_count != 24
        ):
            raise AnswerRunError("no-call preflight changed")


@dataclass(frozen=True)
class AnswerRuntimeCheckpoint:
    run_version: str
    runtime_release: str
    starting_commit: str
    guidance_version: str
    guidance_sha256: str
    config_sha256: str
    input_authorities: Mapping[str, str]
    implementation_hashes: Mapping[str, str]
    preflight_sha256: str
    predictions_sha256: str
    failures_sha256: str
    package_count: int
    prediction_count: int
    eligible_provider_case_count: int
    baseline_case_counts: Mapping[str, int]
    provider_request_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int
    failure_count: int
    scorer_opened: bool
    answer_gold_opened: bool

    def __post_init__(self) -> None:
        if self.run_version != RUN_VERSION or self.runtime_release != RUNTIME_RELEASE:
            raise AnswerRunError("runtime checkpoint version changed")
        _commit(self.starting_commit, "starting commit")
        for value, name in (
            (self.guidance_sha256, "guidance hash"),
            (self.config_sha256, "configuration hash"),
            (self.preflight_sha256, "preflight hash"),
            (self.predictions_sha256, "predictions hash"),
            (self.failures_sha256, "failures hash"),
        ):
            _sha(value, name)
        if not isinstance(self.guidance_version, str) or not self.guidance_version:
            raise AnswerRunError("guidance version is invalid")
        for mapping, name in (
            (self.input_authorities, "input authorities"),
            (self.implementation_hashes, "implementation hashes"),
        ):
            if not mapping:
                raise AnswerRunError(f"{name} are empty")
            for path, digest in mapping.items():
                _text(path, name)
                _sha(digest, name)
        if (
            self.package_count != 24
            or self.prediction_count != 24
            or dict(self.baseline_case_counts) != {"B2": 8, "B3": 8, "B4": 8}
            or any((
                self.eligible_provider_case_count,
                self.provider_request_count,
                self.retry_count,
                self.input_token_count,
                self.output_token_count,
                self.incremental_cost_usd,
                self.failure_count,
            ))
            or self.scorer_opened
            or self.answer_gold_opened
        ):
            raise AnswerRunError("runtime checkpoint accounting changed")


@dataclass(frozen=True)
class MetricValue:
    metric: str
    numerator: int
    denominator: int
    value: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise AnswerRunError("metric name is invalid")
        _count(self.numerator, "metric numerator")
        _count(self.denominator, "metric denominator")
        if self.denominator == 0:
            if self.numerator != 0 or self.value is not None or not self.null_reason:
                raise AnswerRunError("zero-denominator metric is invalid")
        else:
            if self.numerator > self.denominator or self.null_reason is not None:
                raise AnswerRunError("defined metric is invalid")
            if not isinstance(self.value, str) or len(self.value.rsplit(".", 1)[-1]) != 6:
                raise AnswerRunError("metric value precision is invalid")
            try:
                number = float(self.value)
            except ValueError as error:
                raise AnswerRunError("metric value is invalid") from error
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise AnswerRunError("metric value is invalid")


@dataclass(frozen=True)
class PerAnswerQuality:
    query_id: str
    user_id: str
    baseline_id: str
    answer_id: str
    package_id: str
    package_sha256: str
    execution_id: str
    plan_id: str
    snapshot_run_id: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    status: str
    structural_blockers: tuple[str, ...]
    statement_count: int
    citation_count: int
    metrics: tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.query_id, "query ID"), (self.user_id, "user ID"),
        ):
            _text(value, name)
        if self.baseline_id not in AVAILABLE_BASELINES:
            raise AnswerRunError("per-case baseline is unavailable")
        for value, name in (
            (self.answer_id, "answer ID"), (self.package_id, "package ID"),
            (self.package_sha256, "package hash"), (self.execution_id, "execution ID"),
            (self.plan_id, "plan ID"), (self.snapshot_run_id, "snapshot ID"),
        ):
            _sha(value, name)
        _aware(self.as_of, "per-case as_of")
        if (
            self.status != "abstained"
            or self.structural_blockers != ("no_promoted_claims",)
            or self.statement_count != 0
            or self.citation_count != 0
            or tuple(item.metric for item in self.metrics) != METRICS
        ):
            raise AnswerRunError("per-case structural result changed")


@dataclass(frozen=True)
class AnswerQualityScorecard:
    scorer_version: str
    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if self.scorer_version != SCORER_VERSION or len(self.rows) != 35:
            raise AnswerRunError("scorecard shape changed")
        expected = [(baseline, metric) for baseline in BASELINES for metric in METRICS]
        observed = []
        for row in self.rows:
            if set(row) != {"baseline_id", "metric", "case_count", "abstained_count", "value"}:
                raise AnswerRunError("scorecard row fields changed")
            baseline = row["baseline_id"]
            metric = row["metric"]
            observed.append((baseline, metric))
            count = 8 if baseline in AVAILABLE_BASELINES else 0
            if row["case_count"] != count or row["abstained_count"] != count:
                raise AnswerRunError("scorecard case accounting changed")
            value = row["value"]
            if not isinstance(value, MetricValue) or value.metric != metric:
                raise AnswerRunError("scorecard metric value is invalid")
            reason = NULL_REASONS[metric] if baseline in AVAILABLE_BASELINES else "baseline_not_available"
            if value.null_reason != reason or value.denominator != 0 or value.numerator != 0:
                raise AnswerRunError("scorecard null reason changed")
        if observed != expected:
            raise AnswerRunError("scorecard ordering changed")


@dataclass(frozen=True)
class AnswerQualityChecks:
    final_release: str
    per_case_count: int
    scorecard_row_count: int
    B2_case_count: int
    B3_case_count: int
    B4_case_count: int
    B5_case_count: int
    B6_case_count: int
    abstained_count: int
    factual_prediction_count: int
    statement_count: int
    citation_count: int
    non_null_metric_count: int
    failure_count: int
    provider_request_count: int

    def __post_init__(self) -> None:
        if self.final_release != FINAL_RELEASE:
            raise AnswerRunError("quality release version changed")
        for name, value in asdict(self).items():
            if name != "final_release":
                _count(value, name)
        if (
            (self.per_case_count, self.scorecard_row_count) != (24, 35)
            or (self.B2_case_count, self.B3_case_count, self.B4_case_count) != (8, 8, 8)
            or (self.B5_case_count, self.B6_case_count) != (0, 0)
            or self.abstained_count != 24
            or any((
                self.factual_prediction_count, self.statement_count, self.citation_count,
                self.non_null_metric_count, self.failure_count, self.provider_request_count,
            ))
        ):
            raise AnswerRunError("quality checks changed")


@dataclass(frozen=True)
class AnswerQualityFailure:
    failure_id: str
    answer_id: str | None
    package_id: str | None
    query_id: str | None
    baseline_id: str | None
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        for value, name in (
            (self.answer_id, "answer ID"), (self.package_id, "package ID"),
        ):
            if value is not None:
                _sha(value, name)
        if self.query_id is not None:
            _text(self.query_id, "query ID")
        if self.baseline_id is not None and self.baseline_id not in BASELINES:
            raise AnswerRunError("failure baseline is invalid")
        if self.code not in {
            "input_invalid", "checkpoint_invalid", "prediction_invalid",
            "score_invalid", "output_invalid", "runtime_failure",
        } or self.location not in {
            "input", "checkpoint", "prediction", "score", "output", "runtime",
        }:
            raise AnswerRunError("failure is not sanitized")


def failure_id(value: Mapping[str, object]) -> str:
    return stable_sha256(value)
