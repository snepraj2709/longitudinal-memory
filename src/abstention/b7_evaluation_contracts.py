"""Strict contracts for the four-case B6/B7 development evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Mapping


EVALUATION_VERSION = "b7_evaluation_v1"
SCHEMA_VERSION = "b7_evaluation_schema_v1"
CONFIG_VERSION = "b7_evaluation_config_v1"
SCORER_VERSION = "b7_evaluation_scorer_v1"
DATASET_VERSION = "b7-evaluation-development-v1"
FINAL_RELEASE_VERSION = "b7-evaluation-development-v1"
BASELINES = ("B6", "B7")
EXPECTED_DECISIONS = ("answerable", "abstain", "clarify")
PREDICTED_STATUSES = ("answered", "abstained", "clarify", "disputed", "partially_answered")
PREDICTED_ACTIONS = ("answered", "abstained", "clarification_requested", "disputed", "partially_answered")
STATUS_ACTION = {
    "answered": "answered",
    "abstained": "abstained",
    "clarify": "clarification_requested",
    "disputed": "disputed",
    "partially_answered": "partially_answered",
}
ALLOWED_USERS = ("user_001", "user_002")
BASELINE_METRICS = (
    "abstention_precision",
    "abstention_recall",
    "coverage",
    "answer_accuracy",
    "selective_risk",
    "false_answer_rate",
    "unnecessary_abstention_rate",
)
DELTA_METRICS = (
    "coverage_delta",
    "abstention_precision_delta",
    "abstention_recall_delta",
    "answer_accuracy_delta",
    "selective_risk_delta",
    "false_answer_rate_delta",
    "unnecessary_abstention_rate_delta",
    "gate_changed_output_count",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,80}$")
SIX_DECIMAL = re.compile(r"^-?[0-9]+\.[0-9]{6}$")


class B7EvaluationError(ValueError):
    """Reject unsafe, changed, or non-canonical evaluation data."""


@dataclass(frozen=True)
class B7EvaluationReference:
    reference_id: str
    evaluation_version: str
    case_id: str
    user_id: str
    expected_decision: str
    review_status: str

    def __post_init__(self) -> None:
        _sha(self.reference_id, "reference ID")
        if self.evaluation_version != EVALUATION_VERSION:
            raise B7EvaluationError("reference version changed")
        _id(self.case_id, "reference case ID")
        if self.user_id not in ALLOWED_USERS:
            raise B7EvaluationError("reference user is outside development")
        if self.expected_decision not in EXPECTED_DECISIONS:
            raise B7EvaluationError("reference decision is invalid")
        if self.review_status != "implementation_reviewed":
            raise B7EvaluationError("reference is not reviewed")
        _canonical_id(self, "reference_id")


@dataclass(frozen=True)
class B7PerCase:
    per_case_id: str
    evaluation_version: str
    schema_version: str
    config_version: str
    scorer_version: str
    dataset_version: str
    final_release_version: str
    case_id: str
    user_id: str
    baseline_id: str
    pair_id: str
    pair_sha256: str
    shared_identity_sha256: str
    prediction_id: str
    prediction_sha256: str
    expected_decision: str
    predicted_status: str
    predicted_action: str
    answered: bool
    abstained: bool
    answer_correct: bool | None
    false_answer: bool
    unnecessary_abstention: bool

    def __post_init__(self) -> None:
        _sha(self.per_case_id, "per-case ID")
        if (
            self.evaluation_version,
            self.schema_version,
            self.config_version,
            self.scorer_version,
            self.dataset_version,
            self.final_release_version,
        ) != (
            EVALUATION_VERSION,
            SCHEMA_VERSION,
            CONFIG_VERSION,
            SCORER_VERSION,
            DATASET_VERSION,
            FINAL_RELEASE_VERSION,
        ):
            raise B7EvaluationError("per-case version changed")
        _id(self.case_id, "per-case case ID")
        if self.user_id not in ALLOWED_USERS or self.baseline_id not in BASELINES:
            raise B7EvaluationError("per-case identity is invalid")
        for value in (
            self.pair_id, self.pair_sha256, self.shared_identity_sha256,
            self.prediction_id, self.prediction_sha256,
        ):
            _sha(value, "per-case identity hash")
        if self.expected_decision not in EXPECTED_DECISIONS:
            raise B7EvaluationError("per-case expected decision is invalid")
        if self.predicted_status not in PREDICTED_STATUSES or self.predicted_action not in PREDICTED_ACTIONS:
            raise B7EvaluationError("per-case prediction is invalid")
        if self.predicted_action != STATUS_ACTION[self.predicted_status]:
            raise B7EvaluationError("per-case status and action disagree")
        if self.answered != (self.predicted_status == "answered"):
            raise B7EvaluationError("answered indicator changed")
        if self.abstained != (self.predicted_status == "abstained"):
            raise B7EvaluationError("abstained indicator changed")
        if self.answer_correct is not None and not self.answered:
            raise B7EvaluationError("answer correctness exists without an answer")
        if self.answered and self.answer_correct is None:
            raise B7EvaluationError("answered case lacks correctness")
        if self.false_answer != (self.answered and self.expected_decision != "answerable"):
            raise B7EvaluationError("false-answer indicator changed")
        if self.unnecessary_abstention != (self.abstained and self.expected_decision == "answerable"):
            raise B7EvaluationError("unnecessary-abstention indicator changed")
        _canonical_id(self, "per_case_id")


@dataclass(frozen=True)
class B7PairDelta:
    pair_delta_id: str
    evaluation_version: str
    schema_version: str
    case_id: str
    user_id: str
    pair_id: str
    pair_sha256: str
    shared_identity_sha256: str
    b6_per_case_id: str
    b6_per_case_sha256: str
    b7_per_case_id: str
    b7_per_case_sha256: str
    expected_decision: str
    b6_answered: bool
    b7_answered: bool
    b6_abstained: bool
    b7_abstained: bool
    b6_false_answer: bool
    b7_false_answer: bool
    b6_unnecessary_abstention: bool
    b7_unnecessary_abstention: bool
    gate_changed_output: bool

    def __post_init__(self) -> None:
        _sha(self.pair_delta_id, "pair-delta ID")
        if (self.evaluation_version, self.schema_version) != (EVALUATION_VERSION, SCHEMA_VERSION):
            raise B7EvaluationError("pair-delta version changed")
        _id(self.case_id, "pair-delta case ID")
        if self.user_id not in ALLOWED_USERS or self.expected_decision not in EXPECTED_DECISIONS:
            raise B7EvaluationError("pair-delta identity is invalid")
        for value in (
            self.pair_id,
            self.pair_sha256,
            self.shared_identity_sha256,
            self.b6_per_case_id,
            self.b6_per_case_sha256,
            self.b7_per_case_id,
            self.b7_per_case_sha256,
        ):
            _sha(value, "pair-delta identity hash")
        if self.gate_changed_output != (
            self.b6_answered != self.b7_answered or self.b6_abstained != self.b7_abstained
        ):
            raise B7EvaluationError("gate-change indicator changed")
        _canonical_id(self, "pair_delta_id")


@dataclass(frozen=True)
class MetricValue:
    scope: str
    baseline_id: str | None
    metric_name: str
    numerator: int
    denominator: int
    value: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if self.scope not in ("baseline", "pair_delta"):
            raise B7EvaluationError("metric scope is invalid")
        if self.scope == "baseline":
            if self.baseline_id not in BASELINES or self.metric_name not in BASELINE_METRICS:
                raise B7EvaluationError("baseline metric identity is invalid")
        elif self.baseline_id is not None or self.metric_name not in DELTA_METRICS:
            raise B7EvaluationError("delta metric identity is invalid")
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise B7EvaluationError("metric counts are invalid")
        if self.denominator < 0 or abs(self.numerator) > self.denominator:
            raise B7EvaluationError("metric denominator is invalid")
        if self.denominator == 0:
            if self.numerator != 0 or self.value is not None or self.null_reason not in (
                "no_answered_cases", "no_answered_cases_both_baselines",
            ):
                raise B7EvaluationError("null metric policy changed")
        else:
            if self.value is None or SIX_DECIMAL.fullmatch(self.value) is None or self.null_reason is not None:
                raise B7EvaluationError("non-null metric is invalid")
            if self.value != f"{float(Fraction(self.numerator, self.denominator)):.6f}":
                raise B7EvaluationError("metric value does not match its counts")


@dataclass(frozen=True)
class B7Scorecard:
    evaluation_version: str
    schema_version: str
    config_version: str
    scorer_version: str
    dataset_version: str
    final_release_version: str
    composite_score: None
    metrics: tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        if (
            self.evaluation_version,
            self.schema_version,
            self.config_version,
            self.scorer_version,
            self.dataset_version,
            self.final_release_version,
        ) != (
            EVALUATION_VERSION,
            SCHEMA_VERSION,
            CONFIG_VERSION,
            SCORER_VERSION,
            DATASET_VERSION,
            FINAL_RELEASE_VERSION,
        ):
            raise B7EvaluationError("scorecard version changed")
        expected = tuple(("baseline", baseline, metric) for baseline in BASELINES for metric in BASELINE_METRICS) + tuple(
            ("pair_delta", None, metric) for metric in DELTA_METRICS
        )
        actual = tuple((item.scope, item.baseline_id, item.metric_name) for item in self.metrics)
        if actual != expected:
            raise B7EvaluationError("metric order changed")


@dataclass(frozen=True)
class B7EvaluationChecks:
    evaluation_version: str
    final_release_version: str
    case_count: int
    pair_count: int
    per_case_count: int
    pair_delta_count: int
    b6_prediction_count: int
    b7_prediction_count: int
    b6_abstained_count: int
    b7_abstained_count: int
    b6_answered_count: int
    b7_answered_count: int
    expected_answerable_count: int
    expected_unanswerable_count: int
    expected_clarify_count: int
    b6_false_answer_count: int
    b7_false_answer_count: int
    b6_unnecessary_abstention_count: int
    b7_unnecessary_abstention_count: int
    gate_changed_output_count: int
    provider_request_count: int
    failure_count: int
    roadmap_target_case_count: int
    development_executed_count: int
    frozen_test_deferred_count: int
    full_step9_3_roadmap_complete: bool
    step9_4_development_evaluation_complete: bool
    phase9_exit: bool

    def __post_init__(self) -> None:
        if (self.evaluation_version, self.final_release_version) != (
            EVALUATION_VERSION, FINAL_RELEASE_VERSION,
        ):
            raise B7EvaluationError("checks version changed")
        counts = (
            self.case_count, self.pair_count, self.per_case_count, self.pair_delta_count,
            self.b6_prediction_count, self.b7_prediction_count,
            self.b6_abstained_count, self.b7_abstained_count,
            self.b6_answered_count, self.b7_answered_count,
            self.expected_answerable_count, self.expected_unanswerable_count,
            self.expected_clarify_count, self.b6_false_answer_count,
            self.b7_false_answer_count, self.b6_unnecessary_abstention_count,
            self.b7_unnecessary_abstention_count, self.gate_changed_output_count,
            self.provider_request_count, self.failure_count, self.roadmap_target_case_count,
            self.development_executed_count, self.frozen_test_deferred_count,
        )
        if counts != (4, 4, 8, 4, 4, 4, 4, 4, 0, 0, 3, 1, 0, 0, 0, 3, 3, 0, 0, 0, 20, 4, 16):
            raise B7EvaluationError("checks accounting changed")
        if self.full_step9_3_roadmap_complete or not self.step9_4_development_evaluation_complete or self.phase9_exit:
            raise B7EvaluationError("completion boundary changed")


@dataclass(frozen=True)
class B7EvaluationFailure:
    failure_id: str
    case_id: str
    user_id: str
    baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        _id(self.case_id, "failure case ID")
        if self.user_id not in ALLOWED_USERS or self.baseline_id not in BASELINES:
            raise B7EvaluationError("failure identity is invalid")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise B7EvaluationError("failure is not sanitized")
        _canonical_id(self, "failure_id")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def reference_from_mapping(value: object) -> B7EvaluationReference:
    return B7EvaluationReference(**_strict(value, B7EvaluationReference))


def per_case_from_mapping(value: object) -> B7PerCase:
    return B7PerCase(**_strict(value, B7PerCase))


def pair_delta_from_mapping(value: object) -> B7PairDelta:
    return B7PairDelta(**_strict(value, B7PairDelta))


def metric_from_mapping(value: object) -> MetricValue:
    return MetricValue(**_strict(value, MetricValue))


def scorecard_from_mapping(value: object) -> B7Scorecard:
    row = _strict(value, B7Scorecard)
    metrics = row["metrics"]
    if not isinstance(metrics, list):
        raise B7EvaluationError("scorecard metrics are invalid")
    row["metrics"] = tuple(metric_from_mapping(item) for item in metrics)
    return B7Scorecard(**row)


def checks_from_mapping(value: object) -> B7EvaluationChecks:
    return B7EvaluationChecks(**_strict(value, B7EvaluationChecks))


def failure_from_mapping(value: object) -> B7EvaluationFailure:
    return B7EvaluationFailure(**_strict(value, B7EvaluationFailure))


def _strict(value: object, contract: type) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {item.name for item in fields(contract)}:
        raise B7EvaluationError("contract fields changed")
    return dict(value)


def _canonical(value: object) -> object:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise B7EvaluationError("datetime is not timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise B7EvaluationError("non-finite number is unsafe")
        return value
    raise B7EvaluationError("value is not JSON-safe")


def _canonical_id(value: object, name: str) -> None:
    row = asdict(value)
    actual = row.pop(name)
    if actual != stable_sha256(row):
        raise B7EvaluationError(f"{name.replace('_', ' ')} changed")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise B7EvaluationError(f"{name} is invalid")


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise B7EvaluationError(f"{name} is invalid")
