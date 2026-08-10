"""Strict immutable contracts for deterministic answerability decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import re

from retrieval.query_contracts import RequestedValidTime


ANSWERABILITY_VERSION = "answerability_v1"
SCHEMA_VERSION = "answerability_schema_v1"
CONFIG_VERSION = "answerability_config_v1"
POLICY_VERSION = "answerability_policy_v1"
RUNTIME_VERSION = "answerability_runtime_v1"
INPUT_RELEASE_VERSION = "evidence_package_development_v1"
OUTPUT_RELEASE_VERSION = "answerability-development-v1"
DECISIONS = ("answerable", "abstain", "clarify")
ANSWER_STATUSES = ("answered", "disputed", "partially_answered")
INFORMATION_KINDS = (
    "fact", "current_state", "historical_state", "change_over_time",
    "specific_event", "relationship", "commitment", "evidence_request",
    "stable_trait", "causal", "unknown",
)
AUTHORITY_CLASSES = (
    "exact_supported", "direct_subject", "firsthand_participant",
    "official_record", "accepted_durative", "checked_causal",
)
CLAIM_CATEGORIES = ("current_claims", "historical_claims", "conflicting_claims")
REJECTION_STAGES = ("pre_filter", "post_rank", "package_validation", "semantic_policy")
REASON_PRECEDENCE = (
    "no_retrieved_claims",
    "incomplete_evidence",
    "no_promoted_claims",
    "clarification_required",
    "wrong_person_risk",
    "requested_information_absent",
    "requested_time_not_covered",
    "stale_evidence",
    "insufficient_speaker_authority",
    "insufficient_source_authority",
    "unresolved_conflict",
    "stable_trait_support_insufficient",
    "causal_support_insufficient",
)
SEMANTIC_REASONS = REASON_PRECEDENCE[4:]
SAFE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AnswerabilityError(ValueError):
    """Reject unsafe or internally inconsistent answerability data."""


@dataclass(frozen=True)
class AnswerabilityRequirement:
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
            raise AnswerabilityError("information kind is invalid")
        _sorted_ids(self.target_subject_ids, "target subjects")
        _sorted_ids(self.permitted_speaker_ids, "permitted speakers")
        if self.required_authority_class not in AUTHORITY_CLASSES:
            raise AnswerabilityError("authority class is invalid")
        if self.required_predicate is not None:
            _id(self.required_predicate, "required predicate")
        if self.required_object_json is not None:
            _json_safe(self.required_object_json)
        if self.required_polarity not in {None, "positive", "negative"}:
            raise AnswerabilityError("required polarity is invalid")
        _sorted_ids(self.subrequirement_ids, "subrequirements")
        if self.partial_response_requested and not self.subrequirement_ids:
            raise AnswerabilityError("partial response needs structured subrequirements")
        if self.requirement_id != requirement_id_from_requirement(self):
            raise AnswerabilityError("requirement ID changed")


@dataclass(frozen=True)
class AnswerabilityRequest:
    request_id: str
    answerability_version: str
    schema_version: str
    config_version: str
    config_sha256: str
    policy_version: str
    policy_sha256: str
    runtime_version: str
    input_release_version: str
    input_release_manifest_sha256: str
    package_id: str
    package_sha256: str
    user_id: str
    query_id: str
    baseline_id: str
    execution_id: str
    plan_id: str
    snapshot_run_id: str
    index_version: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    query_type: str
    subject_scope: tuple[str, ...]
    speaker_scope: tuple[str, ...]
    sensitivity_scope: str
    requirements: tuple[AnswerabilityRequirement, ...]

    def __post_init__(self) -> None:
        for actual, expected in (
            (self.answerability_version, ANSWERABILITY_VERSION),
            (self.schema_version, SCHEMA_VERSION),
            (self.config_version, CONFIG_VERSION),
            (self.policy_version, POLICY_VERSION),
            (self.runtime_version, RUNTIME_VERSION),
            (self.input_release_version, INPUT_RELEASE_VERSION),
        ):
            if actual != expected:
                raise AnswerabilityError("answerability request version changed")
        for value in (
            self.request_id, self.config_sha256, self.policy_sha256,
            self.input_release_manifest_sha256, self.package_id, self.package_sha256,
            self.execution_id, self.plan_id,
        ):
            _sha(value, "request hash")
        for value in (self.user_id, self.query_id, self.baseline_id, self.snapshot_run_id, self.index_version):
            _id(value, "request identity")
        _aware(self.as_of, "as of")
        if self.query_type not in INFORMATION_KINDS:
            raise AnswerabilityError("query type is invalid")
        _sorted_ids(self.subject_scope, "subject scope")
        _sorted_ids(self.speaker_scope, "speaker scope")
        if self.sensitivity_scope not in {"standard", "sensitive", "unclassified"}:
            raise AnswerabilityError("sensitivity scope is invalid")
        if not self.requirements:
            raise AnswerabilityError("request needs a structured requirement")
        expected = tuple(sorted(self.requirements, key=lambda item: item.requirement_id))
        if (
            self.requirements != expected
            or len({item.requirement_id for item in expected}) != len(expected)
        ):
            raise AnswerabilityError("requirements are not canonical")
        if any(
            (self.subject_scope and not set(item.target_subject_ids).issubset(self.subject_scope))
            or (self.speaker_scope and not set(item.permitted_speaker_ids).issubset(self.speaker_scope))
            for item in self.requirements
        ):
            raise AnswerabilityError("requirement scope exceeds request scope")
        if self.request_id != request_id_from_request(self):
            raise AnswerabilityError("request ID changed")


@dataclass(frozen=True)
class AnswerabilityEvidenceReference:
    reference_id: str
    claim_id: str
    claim_version_id: str
    category: str
    source_ids: tuple[str, ...]
    span_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.reference_id, "evidence reference ID")
        _id(self.claim_id, "claim ID")
        _id(self.claim_version_id, "claim version ID")
        if self.category not in CLAIM_CATEGORIES:
            raise AnswerabilityError("claim category is invalid")
        for values, name in (
            (self.source_ids, "source IDs"),
            (self.span_ids, "span IDs"),
            (self.evidence_ids, "evidence IDs"),
            (self.relation_ids, "relation IDs"),
        ):
            _sorted_ids(values, name)
        if not self.source_ids or not self.span_ids or not self.evidence_ids:
            raise AnswerabilityError("accepted evidence lacks provenance")
        if self.reference_id != evidence_reference_id(self):
            raise AnswerabilityError("evidence reference ID changed")


@dataclass(frozen=True)
class RejectedAnswerabilityEvidence:
    rejection_id: str
    index_record_id: str
    retrieval_rank: int | None
    claim_id: str | None
    claim_version_id: str | None
    stage: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.rejection_id, "rejection ID")
        _id(self.index_record_id, "rejected record ID")
        if (self.claim_id is None) != (self.claim_version_id is None):
            raise AnswerabilityError("rejected claim identity is incomplete")
        if self.claim_id is not None:
            _id(self.claim_id, "rejected claim ID")
            _id(self.claim_version_id, "rejected version ID")
        if self.stage not in REJECTION_STAGES:
            raise AnswerabilityError("rejection stage is invalid")
        if self.stage == "package_validation":
            if type(self.retrieval_rank) is not int or not 1 <= self.retrieval_rank <= 10:
                raise AnswerabilityError("package rejection rank is invalid")
        elif self.stage in {"pre_filter", "post_rank"} and self.retrieval_rank is not None:
            raise AnswerabilityError("retrieval rejection cannot invent rank")
        elif self.stage == "semantic_policy" and self.retrieval_rank is not None and not 1 <= self.retrieval_rank <= 10:
            raise AnswerabilityError("semantic rejection rank is invalid")
        if not self.reasons or self.reasons != tuple(sorted(set(self.reasons))):
            raise AnswerabilityError("rejection reasons are not canonical")
        if self.rejection_id != stable_sha256({
            "index_record_id": self.index_record_id,
            "retrieval_rank": self.retrieval_rank,
            "claim_id": self.claim_id,
            "claim_version_id": self.claim_version_id,
            "stage": self.stage,
            "reasons": self.reasons,
        }):
            raise AnswerabilityError("rejection ID changed")


@dataclass(frozen=True)
class DecisionConfidence:
    value: None
    calibration_status: str
    null_reason: str

    def __post_init__(self) -> None:
        if self.value is not None or self.calibration_status != "not_calibrated" or self.null_reason != "step_9_2_not_run":
            raise AnswerabilityError("decision confidence must remain uncalibrated")


@dataclass(frozen=True)
class CoverageRiskAssessment:
    structural_coverage_complete: bool
    promoted_claim_count: int
    required_part_count: int
    supported_part_count: int
    identity_match: bool
    transaction_visible: bool
    valid_time_covered: bool
    stale_evidence_count: int
    speaker_authority_sufficient: bool
    source_authority_sufficient: bool
    unresolved_conflict_count: int
    stable_trait_supported: bool | None
    causal_supported: bool | None
    requested_information_present: bool
    unsupported_subrequirements: tuple[str, ...]
    applicable_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        counts = (
            self.promoted_claim_count, self.required_part_count, self.supported_part_count,
            self.stale_evidence_count, self.unresolved_conflict_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise AnswerabilityError("assessment count is invalid")
        if self.supported_part_count > self.required_part_count:
            raise AnswerabilityError("supported parts exceed required parts")
        _sorted_ids(self.unsupported_subrequirements, "unsupported subrequirements")
        if self.applicable_reasons != ordered_reasons(self.applicable_reasons):
            raise AnswerabilityError("assessment reasons are not canonical")


@dataclass(frozen=True)
class AnswerabilityDecision:
    decision_id: str
    answerability_version: str
    schema_version: str
    config_version: str
    config_sha256: str
    policy_version: str
    policy_sha256: str
    runtime_version: str
    input_release_version: str
    input_release_manifest_sha256: str
    package_id: str
    package_sha256: str
    request_id: str
    user_id: str
    query_id: str
    baseline_id: str
    execution_id: str
    plan_id: str
    snapshot_run_id: str
    index_version: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    query_type: str
    decision: str
    permitted_answer_statuses: tuple[str, ...]
    generation_allowed: bool
    confidence: DecisionConfidence
    assessment: CoverageRiskAssessment
    primary_reason: str | None
    reasons: tuple[str, ...]
    accepted_evidence: tuple[AnswerabilityEvidenceReference, ...]
    rejected_evidence: tuple[RejectedAnswerabilityEvidence, ...]

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise AnswerabilityError("decision is invalid")
        if self.reasons != ordered_reasons(self.reasons):
            raise AnswerabilityError("decision reasons are not canonical")
        if self.primary_reason != (self.reasons[0] if self.reasons else None):
            raise AnswerabilityError("primary reason is invalid")
        if self.reasons != self.assessment.applicable_reasons:
            raise AnswerabilityError("decision and assessment reasons differ")
        if (
            self.assessment.supported_part_count
            + len(self.assessment.unsupported_subrequirements)
            != self.assessment.required_part_count
        ):
            raise AnswerabilityError("requirement accounting is incomplete")
        expected_statuses = tuple(sorted(set(self.permitted_answer_statuses), key=ANSWER_STATUSES.index))
        if self.permitted_answer_statuses != expected_statuses:
            raise AnswerabilityError("permitted statuses are not canonical")
        if self.decision == "answerable":
            if not self.generation_allowed or not self.permitted_answer_statuses or not self.accepted_evidence:
                raise AnswerabilityError("answerable decision blocks generation")
        elif self.generation_allowed or self.permitted_answer_statuses:
            raise AnswerabilityError("blocked decision permits generation")
        expected_accepted = tuple(sorted(self.accepted_evidence, key=lambda item: item.reference_id))
        expected_rejected = tuple(sorted(self.rejected_evidence, key=_rejection_key))
        if self.accepted_evidence != expected_accepted or len(expected_accepted) != len(set(expected_accepted)):
            raise AnswerabilityError("accepted evidence is not canonical")
        if self.rejected_evidence != expected_rejected or len(expected_rejected) != len(set(expected_rejected)):
            raise AnswerabilityError("rejected evidence is not canonical")
        accepted_claims = {(item.claim_id, item.claim_version_id) for item in self.accepted_evidence}
        rejected_claims = {
            (item.claim_id, item.claim_version_id) for item in self.rejected_evidence
            if item.claim_id is not None and item.stage == "semantic_policy"
        }
        if accepted_claims & rejected_claims:
            raise AnswerabilityError("accepted and semantically rejected evidence overlap")
        if self.decision == "answerable":
            if self.permitted_answer_statuses == ("answered",):
                if self.reasons or any(item.category == "conflicting_claims" for item in self.accepted_evidence):
                    raise AnswerabilityError("answered decision contains unresolved evidence")
            elif self.permitted_answer_statuses == ("disputed",):
                if self.reasons or len(accepted_claims) < 2 or any(
                    item.category != "conflicting_claims" for item in self.accepted_evidence
                ):
                    raise AnswerabilityError("disputed decision is not conflict-grounded")
            elif self.permitted_answer_statuses == ("partially_answered",):
                if not self.reasons or not self.assessment.unsupported_subrequirements:
                    raise AnswerabilityError("partial decision lacks an unresolved subpart")
            else:
                raise AnswerabilityError("answerable status set is invalid")
        elif self.decision == "clarify" and self.primary_reason != "clarification_required":
            raise AnswerabilityError("clarify decision lacks a clarification reason")
        elif self.decision == "abstain" and self.primary_reason == "clarification_required":
            raise AnswerabilityError("abstention hides a clarification decision")
        request_identity = (
            self.answerability_version, self.schema_version, self.config_version,
            self.policy_version, self.runtime_version, self.input_release_version,
        )
        if request_identity != (
            ANSWERABILITY_VERSION, SCHEMA_VERSION, CONFIG_VERSION, POLICY_VERSION,
            RUNTIME_VERSION, INPUT_RELEASE_VERSION,
        ):
            raise AnswerabilityError("decision version changed")
        for value in (
            self.decision_id, self.config_sha256, self.policy_sha256,
            self.input_release_manifest_sha256, self.package_id, self.package_sha256,
            self.request_id, self.execution_id, self.plan_id,
        ):
            _sha(value, "decision hash")
        for value in (self.user_id, self.query_id, self.baseline_id, self.snapshot_run_id, self.index_version):
            _id(value, "decision identity")
        _aware(self.as_of, "as of")
        if self.decision_id != decision_id_from_decision(self):
            raise AnswerabilityError("decision ID changed")


@dataclass(frozen=True)
class AnswerabilityFailure:
    failure_id: str
    package_id: str
    query_id: str
    baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for value in (self.failure_id, self.package_id):
            _sha(value, "failure hash")
        for value in (self.query_id, self.baseline_id):
            _id(value, "failure identity")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise AnswerabilityError("failure is not sanitized")


@dataclass(frozen=True)
class AnswerabilityChecks:
    package_count: int
    decision_count: int
    b2_count: int
    b3_count: int
    b4_count: int
    answerable_count: int
    abstain_count: int
    clarify_count: int
    generation_allowed_count: int
    accepted_evidence_count: int
    rejected_evidence_count: int
    candidate_rejection_count: int
    retrieval_rejection_count: int
    no_promoted_claims_count: int
    uncalibrated_confidence_count: int
    failure_count: int
    provider_request_count: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.__dict__.values()):
            raise AnswerabilityError("check count is invalid")


def requirement_id_from_requirement(value: AnswerabilityRequirement) -> str:
    return stable_sha256({
        name: getattr(value, name) for name in value.__dataclass_fields__ if name != "requirement_id"
    })


def request_id_from_request(value: AnswerabilityRequest) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "request_id"})


def evidence_reference_id(value: AnswerabilityEvidenceReference) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "reference_id"})


def decision_id_from_decision(value: AnswerabilityDecision) -> str:
    return stable_sha256({name: getattr(value, name) for name in value.__dataclass_fields__ if name != "decision_id"})


def ordered_reasons(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(value not in REASON_PRECEDENCE for value in values):
        raise AnswerabilityError("reason is invalid")
    present = set(values)
    if len(present) != len(values):
        raise AnswerabilityError("reasons are duplicated")
    return tuple(value for value in REASON_PRECEDENCE if value in present)


def answerability_decision_from_mapping(value: object) -> AnswerabilityDecision:
    """Strictly deserialize a canonical answerability decision."""

    item = _strict_mapping(value, set(AnswerabilityDecision.__dataclass_fields__), "decision")
    confidence = DecisionConfidence(**_strict_mapping(
        item["confidence"], set(DecisionConfidence.__dataclass_fields__), "confidence",
    ))
    assessment_value = _strict_mapping(
        item["assessment"], set(CoverageRiskAssessment.__dataclass_fields__), "assessment",
    )
    assessment = CoverageRiskAssessment(
        **{
            **assessment_value,
            "unsupported_subrequirements": tuple(assessment_value["unsupported_subrequirements"]),
            "applicable_reasons": tuple(assessment_value["applicable_reasons"]),
        }
    )
    accepted = tuple(
        AnswerabilityEvidenceReference(
            entry["reference_id"], entry["claim_id"], entry["claim_version_id"],
            entry["category"], tuple(entry["source_ids"]), tuple(entry["span_ids"]),
            tuple(entry["evidence_ids"]), tuple(entry["relation_ids"]),
        )
        for raw in _mapping_sequence(item["accepted_evidence"], "accepted evidence")
        for entry in (_strict_mapping(
            raw, set(AnswerabilityEvidenceReference.__dataclass_fields__), "accepted evidence",
        ),)
    )
    rejected = tuple(
        RejectedAnswerabilityEvidence(
            entry["rejection_id"], entry["index_record_id"], entry["retrieval_rank"],
            entry["claim_id"], entry["claim_version_id"], entry["stage"],
            tuple(entry["reasons"]),
        )
        for raw in _mapping_sequence(item["rejected_evidence"], "rejected evidence")
        for entry in (_strict_mapping(
            raw, set(RejectedAnswerabilityEvidence.__dataclass_fields__), "rejected evidence",
        ),)
    )
    return AnswerabilityDecision(
        item["decision_id"], item["answerability_version"], item["schema_version"],
        item["config_version"], item["config_sha256"], item["policy_version"],
        item["policy_sha256"], item["runtime_version"], item["input_release_version"],
        item["input_release_manifest_sha256"], item["package_id"], item["package_sha256"],
        item["request_id"], item["user_id"], item["query_id"], item["baseline_id"],
        item["execution_id"], item["plan_id"], item["snapshot_run_id"], item["index_version"],
        _parse_datetime(item["as_of"]), _parse_valid_time(item["requested_valid_time"]),
        item["query_type"], item["decision"], tuple(item["permitted_answer_statuses"]),
        item["generation_allowed"], confidence, assessment, item["primary_reason"],
        tuple(item["reasons"]), accepted, rejected,
    )


def answerability_failure_from_mapping(value: object) -> AnswerabilityFailure:
    item = _strict_mapping(value, set(AnswerabilityFailure.__dataclass_fields__), "failure")
    return AnswerabilityFailure(**item)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_json_default,
    ) + "\n").encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _rejection_key(value: RejectedAnswerabilityEvidence) -> tuple[int, int, str, str, str]:
    return (
        REJECTION_STAGES.index(value.stage),
        value.retrieval_rank or 0,
        value.index_record_id,
        value.claim_id or "",
        value.claim_version_id or "",
    )


def _json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, (date, datetime)):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or SAFE.fullmatch(value) is None:
        raise AnswerabilityError(f"{name} is invalid")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise AnswerabilityError(f"{name} is invalid")


def _aware(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AnswerabilityError(f"{name} is not timezone-aware")


def _sorted_ids(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))) or any(not isinstance(value, str) or not value for value in values):
        raise AnswerabilityError(f"{name} are not sorted and unique")


def _json_safe(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise AnswerabilityError("value is not safe JSON") from error
    if isinstance(value, float) and not math.isfinite(value):
        raise AnswerabilityError("value is not safe JSON")


def _strict_mapping(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AnswerabilityError(f"{name} fields are invalid")
    return value


def _mapping_sequence(value: object, name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise AnswerabilityError(f"{name} list is invalid")
    return tuple(value)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise AnswerabilityError("datetime is invalid")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    item = _strict_mapping(value, set(RequestedValidTime.__dataclass_fields__), "valid time")
    return RequestedValidTime(
        item["kind"],
        date.fromisoformat(item["point_date"]) if item["point_date"] is not None else None,
        _parse_datetime(item["point_timestamp"]) if item["point_timestamp"] is not None else None,
        date.fromisoformat(item["range_start_date"]) if item["range_start_date"] is not None else None,
        date.fromisoformat(item["range_end_date"]) if item["range_end_date"] is not None else None,
        _parse_datetime(item["range_start_timestamp"]) if item["range_start_timestamp"] is not None else None,
        _parse_datetime(item["range_end_timestamp"]) if item["range_end_timestamp"] is not None else None,
    )
