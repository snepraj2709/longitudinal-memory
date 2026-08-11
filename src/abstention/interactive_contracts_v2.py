"""Strict contracts for the four-case development interactive activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import re

from retrieval.query_contracts import RequestedValidTime


INTERACTIVE_VERSION = "interactive_answering_v2"
SCHEMA_VERSION = "interactive_answering_schema_v2"
CONFIG_VERSION = "interactive_answering_config_v2"
RUNTIME_VERSION = "interactive_answering_development_runtime_v2"
SCORER_VERSION = "interactive_answering_scorer_v2"
INPUT_DATASET_VERSION = "interactive-answering-development-v2"
RUNTIME_RELEASE_VERSION = "interactive-answering-development-runtime-v2"
FINAL_RELEASE_VERSION = "interactive-answering-development-v2"
SYSTEM_VARIANT = "development_dual_retrieval_plus_answerability_v2"
PREDICATE_REGISTRY_VERSION = "predicate_registry_v2"
PREDICATE_REGISTRY_SHA256 = "15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1"
DERIVATION_INPUTS = (
    "defect_summary", "predicate_registry_v2", "runtime_case_initial_user_message",
)
REQUIREMENT_REVIEW_STATUS = "runtime_only_reviewed_nonblind_recovery"
ALLOWED_USERS = ("user_001", "user_002")
ACTIONS = (
    "answered", "abstained", "clarification_requested", "disputed", "partially_answered",
)
DECISIONS = ("answerable", "abstain", "clarify")
BEHAVIOURS = (
    "memory_use", "current_vs_historical", "correction_handling",
    "clarification", "evidence", "abstention",
)
INFORMATION_KINDS = (
    "fact", "current_state", "historical_state", "change_over_time",
    "specific_event", "relationship", "commitment", "evidence_request",
    "stable_trait", "causal", "unknown",
)
AUTHORITY_CLASSES = (
    "exact_supported", "direct_subject", "firsthand_participant",
    "official_record", "accepted_durative", "checked_causal",
)
SAFE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SIX_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")


class InteractiveAnsweringError(ValueError):
    """Reject an unsafe, ambiguous, or internally inconsistent trace."""


@dataclass(frozen=True)
class InteractiveRuntimeCase:
    case_id: str
    benchmark_version: str
    split: str
    user_id: str
    task: str
    capability: str
    as_of: datetime
    difficulty: str
    scenario: str
    initial_user_message: str
    allowed_turns: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case ID"), (self.benchmark_version, "benchmark version"),
            (self.user_id, "user ID"), (self.capability, "capability"),
            (self.difficulty, "difficulty"),
        ):
            _id(value, name)
        if self.split != "development" or self.task != "interactive":
            raise InteractiveAnsweringError("runtime case is outside the development interactive slice")
        if self.user_id not in ALLOWED_USERS:
            raise InteractiveAnsweringError("runtime case user is not authorized")
        _aware(self.as_of, "case as_of")
        _text(self.scenario, "scenario")
        _text(self.initial_user_message, "initial user message")
        if type(self.allowed_turns) is not int or self.allowed_turns < 1:
            raise InteractiveAnsweringError("allowed turns are invalid")


@dataclass(frozen=True)
class InteractiveRequirementPart:
    requirement_id: str
    information_kind: str
    target_subject_ids: tuple[str, ...]
    permitted_speaker_ids: tuple[str, ...]
    requested_valid_time: RequestedValidTime | None
    required_authority_class: str
    required_predicate: str | None
    required_object_json: object | None
    required_polarity: str | None
    complete_subpart_coverage: bool
    conflict_reporting_requested: bool
    partial_response_requested: bool
    subrequirement_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.requirement_id, "requirement ID")
        if self.information_kind not in INFORMATION_KINDS:
            raise InteractiveAnsweringError("requirement information kind is invalid")
        _sorted_ids(self.target_subject_ids, "target subjects")
        _sorted_ids(self.permitted_speaker_ids, "permitted speakers")
        if self.required_authority_class not in AUTHORITY_CLASSES:
            raise InteractiveAnsweringError("requirement authority is invalid")
        if self.required_predicate is not None:
            _id(self.required_predicate, "required predicate")
        if self.required_object_json is not None:
            _json_safe(self.required_object_json)
        if self.required_polarity not in {None, "positive", "negative"}:
            raise InteractiveAnsweringError("required polarity is invalid")
        for name in ("complete_subpart_coverage", "conflict_reporting_requested", "partial_response_requested"):
            if type(getattr(self, name)) is not bool:
                raise InteractiveAnsweringError("requirement boolean is invalid")
        _sorted_ids(self.subrequirement_ids, "subrequirements")
        if self.partial_response_requested and not self.subrequirement_ids:
            raise InteractiveAnsweringError("partial response needs structured subrequirements")
        if self.requirement_id != stable_sha256(_without_id(self, "requirement_id")):
            raise InteractiveAnsweringError("requirement ID changed")


@dataclass(frozen=True)
class InteractiveRequirementSet:
    requirement_set_id: str
    case_id: str
    query_id: str
    query_type: str
    parts: tuple[InteractiveRequirementPart, ...]
    predicate_registry_version: str
    predicate_registry_sha256: str
    derivation_inputs: tuple[str, ...]
    prior_development_gold_exposure: bool
    development_gold_used: bool
    review_status: str

    def __post_init__(self) -> None:
        _sha(self.requirement_set_id, "requirement set ID")
        _id(self.case_id, "case ID")
        _id(self.query_id, "query ID")
        if self.query_type not in INFORMATION_KINDS:
            raise InteractiveAnsweringError("requirement set query type is invalid")
        if not self.parts or self.parts != tuple(sorted(self.parts, key=lambda item: item.requirement_id)):
            raise InteractiveAnsweringError("requirement parts are not canonical")
        if len({item.requirement_id for item in self.parts}) != len(self.parts):
            raise InteractiveAnsweringError("requirement part IDs are duplicated")
        if (self.predicate_registry_version, self.predicate_registry_sha256) != (
            PREDICATE_REGISTRY_VERSION, PREDICATE_REGISTRY_SHA256,
        ):
            raise InteractiveAnsweringError("predicate registry authority changed")
        if self.derivation_inputs != DERIVATION_INPUTS:
            raise InteractiveAnsweringError("requirement derivation inputs changed")
        if self.prior_development_gold_exposure is not True or self.development_gold_used is not False:
            raise InteractiveAnsweringError("requirement gold-exposure disclosure changed")
        if self.review_status != REQUIREMENT_REVIEW_STATUS:
            raise InteractiveAnsweringError("runtime requirement set is not reviewed")
        if self.requirement_set_id != stable_sha256(_without_id(self, "requirement_set_id")):
            raise InteractiveAnsweringError("requirement set ID changed")


@dataclass(frozen=True)
class InteractiveExecutionPlan:
    execution_plan_id: str
    case_id: str
    query_id: str
    requirement_set_id: str
    requirement_set_sha256: str
    required_predicates: tuple[str, ...]
    user_id: str
    request_sha256: str
    plan_id: str
    primary_label: str
    snapshot_run_id: str
    retrieval_execution_id: str

    def __post_init__(self) -> None:
        for value in (
            self.execution_plan_id, self.requirement_set_id, self.requirement_set_sha256,
            self.request_sha256, self.plan_id, self.retrieval_execution_id,
        ):
            _sha(value, "execution plan hash")
        for value in (self.case_id, self.query_id, self.user_id, self.snapshot_run_id):
            _id(value, "execution plan identity")
        if self.primary_label not in INFORMATION_KINDS:
            raise InteractiveAnsweringError("execution plan label is invalid")
        _sorted_ids(self.required_predicates, "required predicates")
        if self.execution_plan_id != stable_sha256(_without_id(self, "execution_plan_id")):
            raise InteractiveAnsweringError("execution plan ID changed")


@dataclass(frozen=True)
class InteractiveTurn:
    turn_index: int
    role: str
    action: str
    text: str
    answer_id: str

    def __post_init__(self) -> None:
        if self.turn_index != 1 or self.role != "assistant" or self.action not in ACTIONS:
            raise InteractiveAnsweringError("interactive turn is invalid")
        _text(self.text, "turn text")
        _sha(self.answer_id, "answer ID")


@dataclass(frozen=True)
class InteractivePrediction:
    prediction_id: str
    interactive_version: str
    schema_version: str
    runtime_version: str
    case_id: str
    benchmark_version: str
    split: str
    user_id: str
    capability: str
    difficulty: str
    as_of: datetime
    allowed_turns: int
    system_variant: str
    query_id: str
    requirement_set_id: str
    requirement_set_sha256: str
    required_predicates: tuple[str, ...]
    execution_plan_id: str
    plan_id: str
    snapshot_run_id: str
    retrieval_execution_id: str
    retrieval_result_sha256: str
    package_id: str
    package_sha256: str
    answerability_decision_id: str
    answerability_decision_sha256: str
    threshold_profile: str
    threshold_application_id: str
    response_action: str
    decision: str
    generation_allowed: bool
    turn_count: int
    turns: tuple[InteractiveTurn, ...]
    predicted_behaviours: tuple[str, ...]
    predicted_claim_ids: tuple[str, ...]
    predicted_evidence_tuples: tuple[tuple[str, str, str, str], ...]
    factual_statement_count: int
    citation_count: int
    provider_eligible: bool

    def __post_init__(self) -> None:
        if (self.interactive_version, self.schema_version, self.runtime_version) != (
            INTERACTIVE_VERSION, SCHEMA_VERSION, RUNTIME_VERSION,
        ):
            raise InteractiveAnsweringError("prediction version changed")
        for value in (
            self.prediction_id, self.requirement_set_id, self.requirement_set_sha256,
            self.execution_plan_id, self.plan_id,
            self.retrieval_execution_id, self.retrieval_result_sha256,
            self.package_id, self.package_sha256, self.answerability_decision_id,
            self.answerability_decision_sha256, self.threshold_application_id,
        ):
            _sha(value, "prediction hash")
        for value in (self.case_id, self.benchmark_version, self.user_id, self.capability, self.difficulty, self.query_id, self.snapshot_run_id):
            _id(value, "prediction identity")
        if self.split != "development" or self.user_id not in ALLOWED_USERS:
            raise InteractiveAnsweringError("prediction is outside the development slice")
        _aware(self.as_of, "prediction as_of")
        if self.system_variant != SYSTEM_VARIANT or self.threshold_profile != "ordinary_1_trait_2":
            raise InteractiveAnsweringError("prediction policy changed")
        _sorted_ids(self.required_predicates, "required predicates")
        if self.response_action not in ACTIONS or self.decision not in DECISIONS:
            raise InteractiveAnsweringError("prediction action or decision is invalid")
        if self.turn_count != 1 or len(self.turns) != 1 or self.turn_count > self.allowed_turns:
            raise InteractiveAnsweringError("interactive turn accounting is invalid")
        if self.turns[0].action != self.response_action:
            raise InteractiveAnsweringError("turn action differs from prediction")
        expected_action = {
            "abstain": "abstained", "clarify": "clarification_requested",
        }.get(self.decision)
        if expected_action is not None and self.response_action != expected_action:
            raise InteractiveAnsweringError("blocked decision action is invalid")
        if self.generation_allowed or self.provider_eligible:
            raise InteractiveAnsweringError("v2 prediction is provider eligible")
        if self.response_action in {"abstained", "clarification_requested"} and (
            self.factual_statement_count or self.citation_count or self.predicted_claim_ids
            or self.predicted_evidence_tuples
        ):
            raise InteractiveAnsweringError("blocked prediction contains factual output")
        _ordered_subset(self.predicted_behaviours, BEHAVIOURS, "predicted behaviours")
        _sorted_ids(self.predicted_claim_ids, "predicted claim IDs")
        if self.predicted_evidence_tuples != tuple(sorted(set(self.predicted_evidence_tuples))):
            raise InteractiveAnsweringError("predicted evidence tuples are not canonical")
        if self.prediction_id != stable_sha256(_without_id(self, "prediction_id")):
            raise InteractiveAnsweringError("prediction ID changed")


@dataclass(frozen=True)
class InteractiveFailure:
    failure_id: str
    case_id: str
    query_id: str
    user_id: str
    system_variant: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        for value in (self.case_id, self.query_id, self.user_id):
            _id(value, "failure identity")
        if self.system_variant != SYSTEM_VARIANT:
            raise InteractiveAnsweringError("failure system changed")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise InteractiveAnsweringError("failure is not sanitized")


@dataclass(frozen=True)
class InteractiveBehaviourReference:
    case_id: str
    user_id: str
    expected_decision: str
    expected_behaviours: tuple[str, ...]
    required_claim_ids: tuple[str, ...]
    exact_evidence_tuples: tuple[tuple[str, str, str, str], ...]
    source_types: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        _id(self.case_id, "reference case ID")
        if self.user_id not in ALLOWED_USERS or self.expected_decision not in DECISIONS:
            raise InteractiveAnsweringError("reference identity is invalid")
        _ordered_subset(self.expected_behaviours, BEHAVIOURS, "expected behaviours")
        _sorted_ids(self.required_claim_ids, "required claim IDs")
        if self.exact_evidence_tuples != tuple(sorted(set(self.exact_evidence_tuples))):
            raise InteractiveAnsweringError("reference evidence tuples are not canonical")
        _sorted_ids(self.source_types, "source types")
        if self.review_status != "implementation_reviewed":
            raise InteractiveAnsweringError("reference is not reviewed")


@dataclass(frozen=True)
class InteractiveMetric:
    numerator: int
    denominator: int
    value: str | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (self.numerator, self.denominator)):
            raise InteractiveAnsweringError("metric counts are invalid")
        if self.denominator == 0:
            if self.value is not None or not isinstance(self.null_reason, str) or not self.null_reason:
                raise InteractiveAnsweringError("null metric is invalid")
        elif self.value is None or SIX_DECIMAL.fullmatch(self.value) is None or self.null_reason is not None:
            raise InteractiveAnsweringError("scored metric is invalid")


@dataclass(frozen=True)
class InteractiveChecks:
    roadmap_target_case_count: int
    development_case_count: int
    frozen_test_deferred_count: int
    full_step_9_3_roadmap_complete: bool
    response_count: int
    turn_count: int
    abstained_count: int
    clarification_count: int
    answered_count: int
    disputed_count: int
    partially_answered_count: int
    factual_statement_count: int
    citation_count: int
    provider_eligible_case_count: int
    provider_request_count: int
    failure_count: int

    def __post_init__(self) -> None:
        if self.full_step_9_3_roadmap_complete is not False:
            raise InteractiveAnsweringError("full roadmap completion was claimed")
        for name, value in self.__dict__.items():
            if name != "full_step_9_3_roadmap_complete" and (type(value) is not int or value < 0):
                raise InteractiveAnsweringError("check count is invalid")


def runtime_case_from_mapping(value: object) -> InteractiveRuntimeCase:
    item = _strict(value, set(InteractiveRuntimeCase.__dataclass_fields__), "runtime case")
    return InteractiveRuntimeCase(**{**item, "as_of": _datetime(item["as_of"])})


def requirement_part_from_mapping(value: object) -> InteractiveRequirementPart:
    item = _strict(value, set(InteractiveRequirementPart.__dataclass_fields__), "requirement part")
    return InteractiveRequirementPart(**{
        **item,
        "target_subject_ids": tuple(item["target_subject_ids"]),
        "permitted_speaker_ids": tuple(item["permitted_speaker_ids"]),
        "requested_valid_time": _valid_time(item["requested_valid_time"]),
        "subrequirement_ids": tuple(item["subrequirement_ids"]),
    })


def requirement_set_from_mapping(value: object) -> InteractiveRequirementSet:
    item = _strict(value, set(InteractiveRequirementSet.__dataclass_fields__), "requirement set")
    return InteractiveRequirementSet(**{
        **item,
        "parts": tuple(requirement_part_from_mapping(part) for part in item["parts"]),
        "derivation_inputs": tuple(item["derivation_inputs"]),
    })


def prediction_from_mapping(value: object) -> InteractivePrediction:
    item = _strict(value, set(InteractivePrediction.__dataclass_fields__), "prediction")
    turns = tuple(InteractiveTurn(**_strict(raw, set(InteractiveTurn.__dataclass_fields__), "turn")) for raw in item["turns"])
    tuples = tuple(tuple(entry) for entry in item["predicted_evidence_tuples"])
    return InteractivePrediction(**{
        **item, "as_of": _datetime(item["as_of"]), "turns": turns,
        "required_predicates": tuple(item["required_predicates"]),
        "predicted_behaviours": tuple(item["predicted_behaviours"]),
        "predicted_claim_ids": tuple(item["predicted_claim_ids"]),
        "predicted_evidence_tuples": tuples,
    })


def reference_from_mapping(value: object) -> InteractiveBehaviourReference:
    item = _strict(value, set(InteractiveBehaviourReference.__dataclass_fields__), "reference")
    return InteractiveBehaviourReference(**{
        **item,
        "expected_behaviours": tuple(item["expected_behaviours"]),
        "required_claim_ids": tuple(item["required_claim_ids"]),
        "exact_evidence_tuples": tuple(tuple(entry) for entry in item["exact_evidence_tuples"]),
        "source_types": tuple(item["source_types"]),
    })


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_json_default,
    ) + "\n").encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _without_id(value: object, name: str) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__ if field != name}


def _json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, (date, datetime)):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _strict(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InteractiveAnsweringError(f"{name} fields are invalid")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise InteractiveAnsweringError("datetime is invalid")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    item = _strict(value, set(RequestedValidTime.__dataclass_fields__), "requested valid time")
    return RequestedValidTime(
        item["kind"],
        date.fromisoformat(item["point_date"]) if item["point_date"] else None,
        _datetime(item["point_timestamp"]) if item["point_timestamp"] else None,
        date.fromisoformat(item["range_start_date"]) if item["range_start_date"] else None,
        date.fromisoformat(item["range_end_date"]) if item["range_end_date"] else None,
        _datetime(item["range_start_timestamp"]) if item["range_start_timestamp"] else None,
        _datetime(item["range_end_timestamp"]) if item["range_end_timestamp"] else None,
    )


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE.fullmatch(value) is None:
        raise InteractiveAnsweringError(f"{name} is invalid")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise InteractiveAnsweringError(f"{name} is invalid")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InteractiveAnsweringError(f"{name} is invalid")
    _json_safe(value)


def _aware(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InteractiveAnsweringError(f"{name} is not timezone-aware")


def _sorted_ids(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))) or any(not isinstance(item, str) or not item for item in values):
        raise InteractiveAnsweringError(f"{name} are not canonical")


def _ordered_subset(values: tuple[str, ...], order: tuple[str, ...], name: str) -> None:
    if values != tuple(item for item in order if item in values) or len(values) != len(set(values)):
        raise InteractiveAnsweringError(f"{name} are not canonical")


def _json_safe(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise InteractiveAnsweringError("value is not JSON-safe") from error
    if isinstance(value, float) and not math.isfinite(value):
        raise InteractiveAnsweringError("value is not finite")
