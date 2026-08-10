"""Strict contracts for deterministic answerability threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping


THRESHOLD_VERSION = "answerability_thresholds_v1"
SCHEMA_VERSION = "answerability_calibration_schema_v1"
RUNTIME_VERSION = "answerability_calibration_runtime_v1"
FIXTURE_DATASET_VERSION = "answerability-threshold-fixtures-development-v1"
RUNTIME_RELEASE_VERSION = "answerability-threshold-development-runtime-v1"
FINAL_RELEASE_VERSION = "answerability-threshold-development-v1"
INPUT_ANSWERABILITY_VERSION = "answerability_v1"
INPUT_RELEASE_VERSION = "answerability-development-v1"
SELECTED_PROFILE = "ordinary_1_trait_2"
PROFILE_ORDER = (
    "ordinary_1_trait_2", "ordinary_1_trait_3", "ordinary_2_trait_2",
    "ordinary_2_trait_3", "ordinary_3_trait_2", "ordinary_3_trait_3",
)
DECISIONS = ("answerable", "abstain", "clarify")
BASELINE_ORDER = ("B2", "B3", "B4")
INFORMATION_KINDS = (
    "fact", "current_state", "historical_state", "change_over_time",
    "specific_event", "relationship", "commitment", "evidence_request",
    "stable_trait", "causal", "unknown",
)
STEP91_REASONS = (
    "no_retrieved_claims", "incomplete_evidence", "no_promoted_claims",
    "clarification_required", "wrong_person_risk", "requested_information_absent",
    "requested_time_not_covered", "stale_evidence",
    "insufficient_speaker_authority", "insufficient_source_authority",
    "unresolved_conflict", "stable_trait_support_insufficient",
    "causal_support_insufficient",
)
SAFE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SIX_DECIMAL = re.compile(r"^-?[0-9]+\.[0-9]{6}$")


class CalibrationError(ValueError):
    """Reject malformed or internally inconsistent calibration data."""


@dataclass(frozen=True)
class AnswerabilityThresholdProfile:
    profile_id: str
    minimum_promoted_claims: int
    minimum_required_part_coverage: str
    ordinary_minimum_distinct_authoritative_sources: int
    minimum_exact_evidence_paths: int
    stable_trait_minimum_distinct_sources: int
    stable_trait_minimum_distinct_sessions: int
    stable_trait_minimum_distinct_episode_times: int
    explicit_closed_durative_interval_override: bool
    maximum_unresolved_conflicts: int
    maximum_stale_evidence: int

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_ORDER:
            raise CalibrationError("threshold profile is invalid")
        values = (
            self.minimum_promoted_claims,
            self.ordinary_minimum_distinct_authoritative_sources,
            self.minimum_exact_evidence_paths,
            self.stable_trait_minimum_distinct_sources,
            self.stable_trait_minimum_distinct_sessions,
            self.stable_trait_minimum_distinct_episode_times,
            self.maximum_unresolved_conflicts,
            self.maximum_stale_evidence,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise CalibrationError("threshold count is invalid")
        _six(self.minimum_required_part_coverage, "required-part coverage")
        if self.minimum_required_part_coverage != "1.000000":
            raise CalibrationError("semantic coverage floor changed")
        ordinary = int(self.profile_id.split("_")[1])
        trait = int(self.profile_id.rsplit("_", 1)[1])
        if (
            self.minimum_promoted_claims != 1
            or self.ordinary_minimum_distinct_authoritative_sources != ordinary
            or self.minimum_exact_evidence_paths != 1
            or self.stable_trait_minimum_distinct_sources != trait
            or self.stable_trait_minimum_distinct_sessions != trait
            or self.stable_trait_minimum_distinct_episode_times != trait
            or self.explicit_closed_durative_interval_override is not True
            or self.maximum_unresolved_conflicts != 0
            or self.maximum_stale_evidence != 0
        ):
            raise CalibrationError("threshold profile drifted")


@dataclass(frozen=True)
class ThresholdConfig:
    threshold_version: str
    schema_version: str
    runtime_version: str
    fixture_dataset_version: str
    runtime_release_version: str
    input_answerability_version: str
    input_release_version: str
    selected_profile: str
    model: str
    confidence_policy: Mapping[str, object]
    hard_semantic_guards: tuple[str, ...]
    profiles: tuple[AnswerabilityThresholdProfile, ...]

    def __post_init__(self) -> None:
        if (
            self.threshold_version != THRESHOLD_VERSION
            or self.schema_version != SCHEMA_VERSION
            or self.runtime_version != RUNTIME_VERSION
            or self.fixture_dataset_version != FIXTURE_DATASET_VERSION
            or self.runtime_release_version != RUNTIME_RELEASE_VERSION
            or self.input_answerability_version != INPUT_ANSWERABILITY_VERSION
            or self.input_release_version != INPUT_RELEASE_VERSION
            or self.selected_profile != SELECTED_PROFILE
            or self.model != "deterministic_no_model"
        ):
            raise CalibrationError("threshold configuration version changed")
        if self.confidence_policy != {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        }:
            raise CalibrationError("confidence policy changed")
        if self.hard_semantic_guards != (
            "structural_coverage", "same_user_identity", "exact_requested_proposition",
            "transaction_visibility", "valid_time_coverage", "source_ingestion_cutoff",
            "speaker_authority", "source_authority", "sensitivity",
            "unresolved_conflict", "stable_trait_proposition", "checked_causality",
            "exact_provenance",
        ):
            raise CalibrationError("hard semantic guards changed")
        if tuple(item.profile_id for item in self.profiles) != PROFILE_ORDER:
            raise CalibrationError("threshold profile order changed")


@dataclass(frozen=True)
class CalibrationFixture:
    fixture_id: str
    information_kind: str
    input_decision: str
    input_reasons: tuple[str, ...]
    promoted_claim_count: int
    required_part_count: int
    supported_part_count: int
    authoritative_source_ids: tuple[str, ...]
    exact_evidence_path_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    episode_times: tuple[str, ...]
    closed_durative_interval: bool
    causal_relation_verified: bool
    unresolved_conflict_count: int
    stale_evidence_count: int
    semantic_guards_passed: bool
    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_sha256: str

    def __post_init__(self) -> None:
        _sha(self.fixture_id, "fixture ID")
        if self.information_kind not in INFORMATION_KINDS:
            raise CalibrationError("fixture information kind is invalid")
        if self.input_decision not in DECISIONS:
            raise CalibrationError("fixture decision is invalid")
        _ordered_reasons(self.input_reasons)
        for name in (
            "promoted_claim_count", "required_part_count", "supported_part_count",
            "unresolved_conflict_count", "stale_evidence_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CalibrationError("fixture count is invalid")
        if self.supported_part_count > self.required_part_count:
            raise CalibrationError("fixture supported parts exceed requirements")
        for values, name in (
            (self.authoritative_source_ids, "sources"),
            (self.exact_evidence_path_ids, "evidence paths"),
            (self.session_ids, "sessions"),
            (self.accepted_evidence_ids, "accepted evidence"),
        ):
            _sorted_hashes(values, name)
        if self.episode_times != tuple(sorted(set(self.episode_times))):
            raise CalibrationError("episode times are not canonical")
        _sha(self.rejected_evidence_sha256, "rejected evidence hash")
        if self.input_decision == "answerable":
            if (
                self.input_reasons
                or not self.semantic_guards_passed
                or self.promoted_claim_count < 1
                or self.required_part_count < 1
                or self.supported_part_count != self.required_part_count
                or not self.accepted_evidence_ids
                or self.unresolved_conflict_count
                or self.stale_evidence_count
                or (self.information_kind == "causal" and not self.causal_relation_verified)
            ):
                raise CalibrationError("answerable fixture bypasses a semantic guard")
        elif self.input_decision == "clarify":
            if self.input_reasons != ("clarification_required",) or self.accepted_evidence_ids:
                raise CalibrationError("clarification fixture is inconsistent")
        elif not self.input_reasons:
            raise CalibrationError("abstention fixture lacks a reason")
        if self.fixture_id != fixture_id_from_fixture(self):
            raise CalibrationError("fixture ID changed")


@dataclass(frozen=True)
class CalibrationReference:
    reference_id: str
    fixture_id: str
    expected_decision: str
    expected_reasons: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        _sha(self.reference_id, "reference ID")
        _sha(self.fixture_id, "fixture ID")
        if self.expected_decision not in DECISIONS:
            raise CalibrationError("reference decision is invalid")
        _ordered_reasons(self.expected_reasons)
        if self.review_status != "implementation_reviewed":
            raise CalibrationError("reference is not reviewed")
        if self.reference_id != stable_sha256({
            "fixture_id": self.fixture_id,
            "expected_decision": self.expected_decision,
            "expected_reasons": self.expected_reasons,
            "review_status": self.review_status,
        }):
            raise CalibrationError("reference ID changed")


@dataclass(frozen=True)
class ThresholdPrediction:
    prediction_id: str
    threshold_version: str
    fixture_id: str
    profile_id: str
    input_decision: str
    output_decision: str
    input_reasons: tuple[str, ...]
    output_reasons: tuple[str, ...]
    withheld_by_threshold: bool
    distinct_authoritative_source_count: int
    exact_evidence_path_count: int
    distinct_session_count: int
    distinct_episode_time_count: int
    closed_durative_interval_override_used: bool

    def __post_init__(self) -> None:
        _sha(self.prediction_id, "prediction ID")
        _sha(self.fixture_id, "fixture ID")
        if self.threshold_version != THRESHOLD_VERSION or self.profile_id not in PROFILE_ORDER:
            raise CalibrationError("prediction version changed")
        if self.input_decision not in DECISIONS or self.output_decision not in DECISIONS:
            raise CalibrationError("prediction decision is invalid")
        _ordered_reasons(self.input_reasons)
        if self.withheld_by_threshold:
            if self.input_decision != "answerable" or self.output_decision != "abstain" or self.output_reasons != ("withheld_by_threshold",):
                raise CalibrationError("threshold withholding is invalid")
        elif self.input_decision != self.output_decision or self.input_reasons != self.output_reasons:
            raise CalibrationError("threshold prediction mutated its input")
        for value in (
            self.distinct_authoritative_source_count, self.exact_evidence_path_count,
            self.distinct_session_count, self.distinct_episode_time_count,
        ):
            if type(value) is not int or value < 0:
                raise CalibrationError("prediction count is invalid")
        if self.prediction_id != prediction_id_from_prediction(self):
            raise CalibrationError("prediction ID changed")


@dataclass(frozen=True)
class ThresholdApplication:
    application_id: str
    threshold_version: str
    profile_id: str
    input_decision_id: str
    input_decision_sha256: str
    package_id: str
    user_id: str
    query_id: str
    baseline_id: str
    input_decision: str
    output_decision: str
    primary_reason: str | None
    reasons: tuple[str, ...]
    generation_allowed: bool
    accepted_evidence_count: int
    accepted_evidence_sha256: str
    rejected_evidence_count: int
    rejected_evidence_sha256: str
    withheld_by_threshold: bool
    confidence: Mapping[str, object]

    def __post_init__(self) -> None:
        for value in (self.application_id, self.input_decision_id, self.input_decision_sha256, self.package_id, self.accepted_evidence_sha256, self.rejected_evidence_sha256):
            _sha(value, "application hash")
        for value in (self.user_id, self.query_id, self.baseline_id):
            _safe(value, "application identity")
        if self.threshold_version != THRESHOLD_VERSION or self.profile_id != SELECTED_PROFILE:
            raise CalibrationError("application threshold changed")
        if self.input_decision not in DECISIONS or self.output_decision not in DECISIONS:
            raise CalibrationError("application decision is invalid")
        _ordered_reasons(self.reasons)
        if self.primary_reason != (self.reasons[0] if self.reasons else None):
            raise CalibrationError("application primary reason changed")
        if self.input_decision in {"abstain", "clarify"} and (
            self.output_decision != self.input_decision or self.withheld_by_threshold
        ):
            raise CalibrationError("application promoted a blocked decision")
        if self.output_decision != "answerable" and self.generation_allowed:
            raise CalibrationError("blocked application permits generation")
        if any(type(value) is not int or value < 0 for value in (self.accepted_evidence_count, self.rejected_evidence_count)):
            raise CalibrationError("application evidence count is invalid")
        if self.confidence != {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        }:
            raise CalibrationError("application confidence changed")
        if self.application_id != application_id_from_application(self):
            raise CalibrationError("application ID changed")


@dataclass(frozen=True)
class ThresholdSweepRow:
    profile_id: str
    fixture_count: int
    answerable_count: int
    abstain_count: int
    clarify_count: int
    total_coverage: str
    answerable_case_coverage: str
    selective_risk: str

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_ORDER:
            raise CalibrationError("sweep profile is invalid")
        if self.fixture_count != 24 or self.answerable_count + self.abstain_count + self.clarify_count != 24:
            raise CalibrationError("sweep accounting changed")
        for value in (self.total_coverage, self.answerable_case_coverage, self.selective_risk):
            _six(value, "sweep metric")


@dataclass(frozen=True)
class FrozenThresholdSelection:
    threshold_version: str
    selected_profile: str
    selection_basis: str
    profile: AnswerabilityThresholdProfile
    confidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.threshold_version != THRESHOLD_VERSION or self.selected_profile != SELECTED_PROFILE or self.profile.profile_id != SELECTED_PROFILE:
            raise CalibrationError("selected threshold changed")
        if self.selection_basis != "preselected_by_clean_room_contract_before_reference":
            raise CalibrationError("threshold selection basis changed")
        if self.confidence != {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        }:
            raise CalibrationError("selection confidence changed")


@dataclass(frozen=True)
class CalibrationMetric:
    metric_id: str
    scope: str
    profile_id: str | None
    slice_name: str
    slice_value: str
    metric: str
    numerator: int
    denominator: int
    value: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        _sha(self.metric_id, "metric ID")
        if self.scope not in {"real_development", "controlled_fixture_only"}:
            raise CalibrationError("metric scope is invalid")
        if self.profile_id is not None and self.profile_id not in PROFILE_ORDER:
            raise CalibrationError("metric profile is invalid")
        for value in (self.slice_name, self.slice_value, self.metric):
            _safe(value, "metric identity")
        if any(type(value) is not int or value < 0 for value in (self.numerator, self.denominator)):
            raise CalibrationError("metric count is invalid")
        if self.denominator == 0:
            if self.value is not None or not self.null_reason:
                raise CalibrationError("null metric lacks a reason")
        else:
            if self.value is None or self.null_reason is not None:
                raise CalibrationError("defined metric is malformed")
            _six(self.value, "metric value")
        if self.metric_id != metric_id_from_metric(self):
            raise CalibrationError("metric ID changed")


@dataclass(frozen=True)
class AnswerabilityCalibrationScorecard:
    development_metrics: tuple[CalibrationMetric, ...]
    controlled_fixture_metrics: tuple[CalibrationMetric, ...]
    composite_score: None

    def __post_init__(self) -> None:
        if self.composite_score is not None:
            raise CalibrationError("composite score is forbidden")
        for values in (self.development_metrics, self.controlled_fixture_metrics):
            ids = tuple(item.metric_id for item in values)
            if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
                raise CalibrationError("metrics are not canonical")


@dataclass(frozen=True)
class AnswerabilityCalibrationChecks:
    development_decision_count: int
    development_abstain_count: int
    development_answerable_count: int
    development_clarify_count: int
    development_generation_allowed_count: int
    development_null_confidence_count: int
    input_rejection_count: int
    accepted_evidence_count: int
    fixture_result_count: int
    threshold_sweep_count: int
    selected_profile: str
    failure_count: int
    provider_request_count: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name != "selected_profile" and (type(value) is not int or value < 0):
                raise CalibrationError("check count is invalid")
        if self.selected_profile != SELECTED_PROFILE:
            raise CalibrationError("check selected profile changed")


@dataclass(frozen=True)
class AnswerabilityCalibrationFailure:
    failure_id: str
    item_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        _sha(self.item_id, "failure item ID")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise CalibrationError("failure is not sanitized")


def fixture_id_from_fixture(value: CalibrationFixture) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "fixture_id"})


def prediction_id_from_prediction(value: ThresholdPrediction) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "prediction_id"})


def application_id_from_application(value: ThresholdApplication) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "application_id"})


def metric_id_from_metric(value: CalibrationMetric) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "metric_id"})


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_config_mapping(value: object) -> ThresholdConfig:
    item = _strict(value, set(ThresholdConfig.__dataclass_fields__), "threshold config")
    profiles = tuple(AnswerabilityThresholdProfile(**_strict(profile, set(AnswerabilityThresholdProfile.__dataclass_fields__), "profile")) for profile in _sequence(item["profiles"], "profiles"))
    return ThresholdConfig(
        **{name: item[name] for name in ThresholdConfig.__dataclass_fields__ if name not in {"profiles", "hard_semantic_guards"}},
        hard_semantic_guards=tuple(item["hard_semantic_guards"]),
        profiles=profiles,
    )


def fixture_from_mapping(value: object) -> CalibrationFixture:
    item = _strict(value, set(CalibrationFixture.__dataclass_fields__), "fixture")
    return CalibrationFixture(**{
        **item,
        "input_reasons": tuple(item["input_reasons"]),
        "authoritative_source_ids": tuple(item["authoritative_source_ids"]),
        "exact_evidence_path_ids": tuple(item["exact_evidence_path_ids"]),
        "session_ids": tuple(item["session_ids"]),
        "episode_times": tuple(item["episode_times"]),
        "accepted_evidence_ids": tuple(item["accepted_evidence_ids"]),
    })


def reference_from_mapping(value: object) -> CalibrationReference:
    item = _strict(value, set(CalibrationReference.__dataclass_fields__), "reference")
    return CalibrationReference(**{**item, "expected_reasons": tuple(item["expected_reasons"])})


def prediction_from_mapping(value: object) -> ThresholdPrediction:
    item = _strict(value, set(ThresholdPrediction.__dataclass_fields__), "prediction")
    return ThresholdPrediction(**{
        **item,
        "input_reasons": tuple(item["input_reasons"]),
        "output_reasons": tuple(item["output_reasons"]),
    })


def application_from_mapping(value: object) -> ThresholdApplication:
    item = _strict(value, set(ThresholdApplication.__dataclass_fields__), "application")
    return ThresholdApplication(**{**item, "reasons": tuple(item["reasons"])})


def metric_from_mapping(value: object) -> CalibrationMetric:
    return CalibrationMetric(**_strict(value, set(CalibrationMetric.__dataclass_fields__), "metric"))


def _ordered_reasons(values: tuple[str, ...]) -> None:
    if any(value not in STEP91_REASONS for value in values):
        raise CalibrationError("fixture reason is invalid")
    present = set(values)
    expected = tuple(value for value in STEP91_REASONS if value in present)
    if values != expected or len(values) != len(present):
        raise CalibrationError("fixture reasons are not canonical")


def _json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _strict(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CalibrationError(f"{name} fields changed")
    _json_safe(value)
    return value


def _sequence(value: object, name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CalibrationError(f"{name} is invalid")
    return tuple(value)


def _json_safe(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationError("non-finite JSON value")
        return
    if isinstance(value, list):
        for item in value:
            _json_safe(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CalibrationError("JSON object key is invalid")
        for item in value.values():
            _json_safe(item)
        return
    raise CalibrationError("value is not JSON safe")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise CalibrationError(f"{name} is invalid")


def _safe(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE.fullmatch(value) is None:
        raise CalibrationError(f"{name} is invalid")


def _sorted_hashes(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise CalibrationError(f"{name} are not canonical")
    for value in values:
        _sha(value, name)


def _six(value: object, name: str) -> None:
    if not isinstance(value, str) or SIX_DECIMAL.fullmatch(value) is None:
        raise CalibrationError(f"{name} is not fixed six-decimal")
