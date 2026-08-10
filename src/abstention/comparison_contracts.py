"""Strict contracts for the matched B6/B7 development prerequisite runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import re
from typing import Mapping


COMPARISON_VERSION = "b6_b7_comparison_prerequisite_v1"
SCHEMA_VERSION = "b6_b7_comparison_schema_v1"
CONFIG_VERSION = "b6_b7_comparison_config_v1"
RUNTIME_VERSION = "b6_b7_comparison_runtime_v1"
DATASET_VERSION = "b6-b7-comparable-development-v1"
RUNTIME_RELEASE_VERSION = "b6-b7-comparable-development-runtime-v1"
WRAPPER_BASELINES = ("B6", "B7")
PACKAGE_BASELINE = "B4"
ALLOWED_USERS = ("user_001", "user_002")
DIFFERENCE_WHITELIST = (
    "answerability_decision_id",
    "answerability_decision_sha256",
    "gate_applied",
    "prediction_id",
    "threshold_application_id",
    "threshold_profile",
    "wrapper_baseline_id",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,80}$")


class ComparisonPrerequisiteError(ValueError):
    """Reject changed identity, unsafe data, or an incomplete matched pair."""


@dataclass(frozen=True)
class ComparableRuntimeInput:
    input_id: str
    case_id: str
    user_id: str
    query_id: str
    as_of: datetime
    requirement_set_id: str
    requirement_set_sha256: str
    execution_plan_id: str
    plan_id: str
    snapshot_run_id: str
    retrieval_execution_id: str
    retrieval_result_sha256: str
    package_id: str
    package_sha256: str
    package_baseline_id: str
    accepted_index_record_ids: tuple[str, ...]
    relevant_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.input_id, "input ID")
        for value, name in (
            (self.case_id, "case ID"), (self.user_id, "user ID"),
            (self.query_id, "query ID"), (self.snapshot_run_id, "snapshot ID"),
        ):
            _id(value, name)
        if self.user_id not in ALLOWED_USERS:
            raise ComparisonPrerequisiteError("runtime input user is outside development")
        _aware(self.as_of, "runtime input as_of")
        for value in (
            self.requirement_set_id, self.requirement_set_sha256,
            self.execution_plan_id, self.plan_id, self.retrieval_execution_id,
            self.retrieval_result_sha256, self.package_id, self.package_sha256,
        ):
            _sha(value, "runtime identity hash")
        if self.package_baseline_id != PACKAGE_BASELINE:
            raise ComparisonPrerequisiteError("package baseline was relabelled")
        _sorted_unique(self.accepted_index_record_ids, "accepted index records")
        _sorted_unique(self.relevant_source_ids, "relevant sources")
        if self.input_id != stable_sha256(_without_id(self, "input_id")):
            raise ComparisonPrerequisiteError("runtime input ID changed")


@dataclass(frozen=True)
class ComparablePrediction:
    prediction_id: str
    comparison_version: str
    schema_version: str
    config_version: str
    runtime_version: str
    runtime_release_version: str
    input_id: str
    case_id: str
    user_id: str
    query_id: str
    as_of: datetime
    requirement_set_id: str
    execution_plan_id: str
    plan_id: str
    snapshot_run_id: str
    retrieval_execution_id: str
    retrieval_result_sha256: str
    package_id: str
    package_sha256: str
    package_baseline_id: str
    wrapper_baseline_id: str
    gate_applied: bool
    answerability_decision_id: str | None
    answerability_decision_sha256: str | None
    threshold_profile: str | None
    threshold_application_id: str | None
    configured_future_answer_model: str
    prompt_version: str
    prompt_sha256: str
    memory_answer_config_sha256: str
    response_action: str
    response_status: str
    response_text_sha256: str
    underlying_answer_id: str
    generation_allowed: bool
    provider_eligible: bool
    provider_returned_model: str | None
    provider_request_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int

    def __post_init__(self) -> None:
        if (
            self.comparison_version, self.schema_version, self.config_version,
            self.runtime_version, self.runtime_release_version,
        ) != (
            COMPARISON_VERSION, SCHEMA_VERSION, CONFIG_VERSION,
            RUNTIME_VERSION, RUNTIME_RELEASE_VERSION,
        ):
            raise ComparisonPrerequisiteError("prediction version changed")
        for value in (
            self.prediction_id, self.input_id, self.requirement_set_id,
            self.execution_plan_id, self.plan_id, self.retrieval_execution_id,
            self.retrieval_result_sha256, self.package_id, self.package_sha256,
            self.prompt_sha256, self.memory_answer_config_sha256,
            self.response_text_sha256, self.underlying_answer_id,
        ):
            _sha(value, "prediction identity hash")
        for value in (self.case_id, self.user_id, self.query_id, self.snapshot_run_id):
            _id(value, "prediction identity")
        if self.user_id not in ALLOWED_USERS:
            raise ComparisonPrerequisiteError("prediction user is outside development")
        _aware(self.as_of, "prediction as_of")
        if self.package_baseline_id != PACKAGE_BASELINE or self.wrapper_baseline_id not in WRAPPER_BASELINES:
            raise ComparisonPrerequisiteError("prediction baseline is invalid")
        if self.wrapper_baseline_id == "B6":
            if self.gate_applied or any((
                self.answerability_decision_id, self.answerability_decision_sha256,
                self.threshold_profile, self.threshold_application_id,
            )):
                raise ComparisonPrerequisiteError("B6 passed through the Step 9 gate")
        else:
            if not self.gate_applied or self.threshold_profile != "ordinary_1_trait_2":
                raise ComparisonPrerequisiteError("B7 gate/profile identity changed")
            for value in (
                self.answerability_decision_id, self.answerability_decision_sha256,
                self.threshold_application_id,
            ):
                if value is None:
                    raise ComparisonPrerequisiteError("B7 gate identity is missing")
                _sha(value, "B7 gate hash")
        if (
            self.configured_future_answer_model != "gpt-4.1-2025-04-14"
            or self.prompt_version != "memory_answer_prompt_v1"
            or self.response_action != "abstained"
            or self.response_status != "abstained"
            or self.generation_allowed
            or self.provider_eligible
            or self.provider_returned_model is not None
            or any((self.provider_request_count, self.retry_count, self.input_token_count,
                    self.output_token_count, self.incremental_cost_usd))
        ):
            raise ComparisonPrerequisiteError("prediction crossed the no-call boundary")
        if self.prediction_id != stable_sha256(_without_id(self, "prediction_id")):
            raise ComparisonPrerequisiteError("prediction ID changed")


@dataclass(frozen=True)
class ComparablePair:
    pair_id: str
    input_id: str
    case_id: str
    user_id: str
    b6_prediction_id: str
    b6_prediction_sha256: str
    b7_prediction_id: str
    b7_prediction_sha256: str
    shared_identity_sha256: str
    difference_whitelist: tuple[str, ...]
    response_identity_match: bool

    def __post_init__(self) -> None:
        for value in (
            self.pair_id, self.input_id, self.b6_prediction_id,
            self.b6_prediction_sha256, self.b7_prediction_id,
            self.b7_prediction_sha256, self.shared_identity_sha256,
        ):
            _sha(value, "pair hash")
        _id(self.case_id, "pair case ID")
        if self.user_id not in ALLOWED_USERS:
            raise ComparisonPrerequisiteError("pair user is outside development")
        if self.difference_whitelist != DIFFERENCE_WHITELIST:
            raise ComparisonPrerequisiteError("pair difference whitelist changed")
        if self.shared_identity_sha256 != self.input_id:
            raise ComparisonPrerequisiteError("pair shared identity changed")
        if self.response_identity_match is not True:
            raise ComparisonPrerequisiteError("matched responses differ")
        if self.pair_id != stable_sha256(_without_id(self, "pair_id")):
            raise ComparisonPrerequisiteError("pair ID changed")


@dataclass(frozen=True)
class ComparablePreflight:
    comparison_version: str
    runtime_release_version: str
    guidance_sha256: str
    starting_commit: str
    development_case_count: int
    roadmap_target_case_count: int
    frozen_test_deferred_count: int
    step9_4_evaluated: bool
    step9_4_reference_absent: bool
    step9_4_scorer_absent: bool
    step9_4_final_absent: bool
    step9_3_gold_opened: bool
    step9_3_reference_opened: bool
    runtime_rows_requested: int
    record_five_requested: bool
    development_user_count: int
    development_source_count: int
    next_user_or_source_requested: bool
    implementer_pilot_answer_reference_exposure: bool
    implementer_pilot_answer_reference_used: bool
    reviewer_frozen_snippet_exposure: bool
    reviewer_frozen_snippet_used: bool
    root_contract_deriver_prohibited_exposure: bool
    prompt_rendered: bool
    provider_initialized: bool
    credential_or_environment_opened: bool
    network_opened: bool
    provider_request_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int

    def __post_init__(self) -> None:
        if (self.comparison_version, self.runtime_release_version) != (
            COMPARISON_VERSION, RUNTIME_RELEASE_VERSION,
        ):
            raise ComparisonPrerequisiteError("preflight version changed")
        _sha(self.guidance_sha256, "guidance hash")
        if not isinstance(self.starting_commit, str) or GIT_SHA.fullmatch(self.starting_commit) is None:
            raise ComparisonPrerequisiteError("starting commit is invalid")
        if (
            self.development_case_count, self.roadmap_target_case_count,
            self.frozen_test_deferred_count, self.runtime_rows_requested,
            self.development_user_count, self.development_source_count,
        ) != (4, 20, 16, 4, 2, 20):
            raise ComparisonPrerequisiteError("preflight accounting changed")
        if any((
            self.step9_4_evaluated, not self.step9_4_reference_absent,
            not self.step9_4_scorer_absent, not self.step9_4_final_absent,
            self.step9_3_gold_opened, self.step9_3_reference_opened,
            self.record_five_requested, self.next_user_or_source_requested,
            not self.implementer_pilot_answer_reference_exposure,
            self.implementer_pilot_answer_reference_used,
            not self.reviewer_frozen_snippet_exposure, self.reviewer_frozen_snippet_used,
            self.root_contract_deriver_prohibited_exposure, self.prompt_rendered,
            self.provider_initialized, self.credential_or_environment_opened,
            self.network_opened, self.provider_request_count, self.retry_count,
            self.input_token_count, self.output_token_count, self.incremental_cost_usd,
        )):
            raise ComparisonPrerequisiteError("preflight boundary changed")


@dataclass(frozen=True)
class ComparableCheckpoint:
    comparison_version: str
    runtime_release_version: str
    guidance_sha256: str
    config_sha256: str
    case_count: int
    pair_count: int
    b6_prediction_count: int
    b7_prediction_count: int
    b6_abstained_count: int
    b7_abstained_count: int
    b6_gate_application_count: int
    b7_answerability_decision_count: int
    b7_threshold_application_count: int
    response_identity_match_count: int
    provider_eligible_case_count: int
    provider_request_count: int
    failure_count: int
    roadmap_target_case_count: int
    development_executed_count: int
    frozen_test_deferred_count: int
    full_step9_3_roadmap_complete: bool
    step9_4_evaluated: bool
    artifacts: tuple[tuple[str, str], ...]
    implementation: tuple[tuple[str, str], ...]
    predecessor_drift: tuple[tuple[str, str, str, str], ...]

    def __post_init__(self) -> None:
        if (self.comparison_version, self.runtime_release_version) != (
            COMPARISON_VERSION, RUNTIME_RELEASE_VERSION,
        ):
            raise ComparisonPrerequisiteError("checkpoint version changed")
        _sha(self.guidance_sha256, "guidance hash")
        _sha(self.config_sha256, "config hash")
        counts = (
            self.case_count, self.pair_count, self.b6_prediction_count,
            self.b7_prediction_count, self.b6_abstained_count,
            self.b7_abstained_count, self.b6_gate_application_count,
            self.b7_answerability_decision_count,
            self.b7_threshold_application_count, self.response_identity_match_count,
            self.provider_eligible_case_count, self.provider_request_count,
            self.failure_count, self.roadmap_target_case_count,
            self.development_executed_count, self.frozen_test_deferred_count,
        )
        if counts != (4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 0, 0, 20, 4, 16):
            raise ComparisonPrerequisiteError("checkpoint accounting changed")
        if self.full_step9_3_roadmap_complete or self.step9_4_evaluated:
            raise ComparisonPrerequisiteError("checkpoint overstates completion")
        _bindings(self.artifacts, "artifacts")
        _bindings(self.implementation, "implementation")
        if self.predecessor_drift != tuple(sorted(self.predecessor_drift)):
            raise ComparisonPrerequisiteError("predecessor drift is not canonical")
        for path, old, new, reason in self.predecessor_drift:
            if not path.startswith("tests/integration/test_") or not reason:
                raise ComparisonPrerequisiteError("predecessor drift is unsafe")
            _sha(old, "predecessor old hash")
            _sha(new, "predecessor new hash")


@dataclass(frozen=True)
class ComparableFailure:
    failure_id: str
    case_id: str
    user_id: str
    wrapper_baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        _id(self.case_id, "failure case ID")
        if self.user_id not in ALLOWED_USERS or self.wrapper_baseline_id not in WRAPPER_BASELINES:
            raise ComparisonPrerequisiteError("failure identity is unsafe")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise ComparisonPrerequisiteError("failure fields are not sanitized")
        if self.failure_id != stable_sha256(_without_id(self, "failure_id")):
            raise ComparisonPrerequisiteError("failure ID changed")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def runtime_input_from_mapping(value: object) -> ComparableRuntimeInput:
    row = _strict(value, ComparableRuntimeInput)
    row["as_of"] = _datetime(row["as_of"])
    row["accepted_index_record_ids"] = _tuple(row["accepted_index_record_ids"], "accepted index records")
    row["relevant_source_ids"] = _tuple(row["relevant_source_ids"], "relevant sources")
    return ComparableRuntimeInput(**row)


def prediction_from_mapping(value: object) -> ComparablePrediction:
    row = _strict(value, ComparablePrediction)
    row["as_of"] = _datetime(row["as_of"])
    return ComparablePrediction(**row)


def pair_from_mapping(value: object) -> ComparablePair:
    row = _strict(value, ComparablePair)
    row["difference_whitelist"] = _tuple(row["difference_whitelist"], "difference whitelist")
    return ComparablePair(**row)


def preflight_from_mapping(value: object) -> ComparablePreflight:
    return ComparablePreflight(**_strict(value, ComparablePreflight))


def checkpoint_from_mapping(value: object) -> ComparableCheckpoint:
    row = _strict(value, ComparableCheckpoint)
    row["artifacts"] = _binding_tuple(row["artifacts"], "artifacts")
    row["implementation"] = _binding_tuple(row["implementation"], "implementation")
    value = row["predecessor_drift"]
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 4
        or not all(isinstance(part, str) for part in item)
        for item in value
    ):
        raise ComparisonPrerequisiteError("predecessor drift is invalid")
    row["predecessor_drift"] = tuple(tuple(item) for item in value)
    return ComparableCheckpoint(**row)


def _strict(value: object, contract: type) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {item.name for item in fields(contract)}:
        raise ComparisonPrerequisiteError("contract fields changed")
    return dict(value)


def _canonical(value: object) -> object:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        _aware(value, "datetime")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComparisonPrerequisiteError("non-finite number is unsafe")
        return value
    raise ComparisonPrerequisiteError("value is not JSON-safe")


def _without_id(value: object, name: str) -> dict[str, object]:
    result = asdict(value)
    del result[name]
    return result


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ComparisonPrerequisiteError(f"{name} is invalid")


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ComparisonPrerequisiteError(f"{name} is invalid")


def _aware(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ComparisonPrerequisiteError(f"{name} is not timezone-aware")


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ComparisonPrerequisiteError("datetime is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _aware(parsed, "datetime")
    return parsed


def _tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ComparisonPrerequisiteError(f"{name} are invalid")
    return tuple(value)


def _sorted_unique(value: tuple[str, ...], name: str) -> None:
    if value != tuple(sorted(set(value))):
        raise ComparisonPrerequisiteError(f"{name} are not canonical")
    for item in value:
        _id(item, name)


def _bindings(value: tuple[tuple[str, str], ...], name: str) -> None:
    if value != tuple(sorted(value)) or len({item[0] for item in value}) != len(value):
        raise ComparisonPrerequisiteError(f"{name} bindings are not canonical")
    for path, digest in value:
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ComparisonPrerequisiteError(f"{name} path is unsafe")
        _sha(digest, f"{name} hash")


def _binding_tuple(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 or not all(isinstance(part, str) for part in item)
        for item in value
    ):
        raise ComparisonPrerequisiteError(f"{name} bindings are invalid")
    return tuple((item[0], item[1]) for item in value)
