"""Validated immutable contracts for source-grounded evidence packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import re

from retrieval.baseline_contracts import (
    BASELINE_RECORD_KINDS,
    BaselineRetrievalResult,
    RejectedRetrievalItem,
)
from retrieval.query_contracts import QueryPlan, RequestedValidTime, RetrievalQueryRequest


PACKAGE_VERSION = "evidence_package_v1"
CONFIG_VERSION = "evidence_package_config_v1"
SCHEMA_VERSION = "evidence_package_schema_v1"
RUNTIME_VERSION = "evidence_package_runtime_v1"
INPUT_RELEASE_VERSION = "baseline_execution_development_v1"
LIFECYCLE_STATUSES = (
    "candidate", "confirmed", "current", "historical", "disputed", "superseded",
)
CATEGORY_BY_STATUS = {
    "current": "current_claims",
    "confirmed": "current_claims",
    "historical": "historical_claims",
    "superseded": "historical_claims",
    "disputed": "conflicting_claims",
}
EPISTEMIC_STATUSES = (
    "asserted", "inferred", "reported_by_other", "hypothetical",
    "uncertain", "denied", "corrected",
)
TIME_PRECISIONS = ("timestamp", "day", "month", "year", "approximate", "unknown")
SUPPORT_TYPES = ("supports", "contradicts", "corrects")
REJECTION_STAGES = ("pre_filter", "post_rank", "package_validation")
BLOCKERS = (
    "no_retrieved_claims",
    "incomplete_evidence",
    "no_promoted_claims",
    "clarification_required",
)
SAFE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_CODE = re.compile(r"^[a-z0-9_]{1,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RATIO = re.compile(r"^(?:0\.[0-9]{6}|1\.000000)$")


class EvidencePackageError(ValueError):
    """Reject an unsafe or internally inconsistent evidence package."""


@dataclass(frozen=True)
class FrozenEligibilitySnapshot:
    """Eligibility authority carried verbatim from the frozen retrieval result."""

    plan_id: str
    user_id: str
    index_version: str
    snapshot_run_id: str
    eligible_record_ids: tuple[str, ...]
    pre_filter_rejections: tuple[RejectedRetrievalItem, ...]
    snapshot_sha256: str

    def __post_init__(self) -> None:
        _sha(self.plan_id, "eligibility plan ID")
        _id(self.user_id, "eligibility user")
        _id(self.snapshot_run_id, "eligibility snapshot")
        _id(self.index_version, "eligibility index version")
        _sorted_unique(self.eligible_record_ids, "eligible record IDs")
        pre_ids = tuple(item.index_record_id for item in self.pre_filter_rejections)
        if (
            any(item.stage != "pre_filter" for item in self.pre_filter_rejections)
            or pre_ids != tuple(sorted(set(pre_ids)))
            or set(pre_ids) & set(self.eligible_record_ids)
        ):
            raise EvidencePackageError("frozen eligibility is not canonical")
        _sha(self.snapshot_sha256, "eligibility hash")
        if self.snapshot_sha256 != stable_sha256(self.canonical_value()):
            raise EvidencePackageError("frozen eligibility hash changed")

    def canonical_value(self) -> object:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "index_version": self.index_version,
            "snapshot_run_id": self.snapshot_run_id,
            "eligible_record_ids": self.eligible_record_ids,
            "pre_filter_rejections": self.pre_filter_rejections,
        }


@dataclass(frozen=True)
class EvidencePackageBuildRequest:
    package_version: str
    schema_version: str
    config_version: str
    config_sha256: str
    runtime_version: str
    input_release_version: str
    input_release_manifest_sha256: str
    input_release_checkpoint_sha256: str
    input_results_sha256: str
    query: RetrievalQueryRequest
    plan: QueryPlan
    eligibility: FrozenEligibilitySnapshot
    retrieval_result: BaselineRetrievalResult
    retrieval_result_sha256: str

    def __post_init__(self) -> None:
        if (
            self.package_version != PACKAGE_VERSION
            or self.schema_version != SCHEMA_VERSION
            or self.config_version != CONFIG_VERSION
            or self.runtime_version != RUNTIME_VERSION
            or self.input_release_version != INPUT_RELEASE_VERSION
        ):
            raise EvidencePackageError("package schema, config, runtime, or input release changed")
        for value, name in (
            (self.config_sha256, "config hash"),
            (self.input_release_manifest_sha256, "input manifest hash"),
            (self.input_release_checkpoint_sha256, "input checkpoint hash"),
            (self.input_results_sha256, "input results hash"),
            (self.retrieval_result_sha256, "retrieval result hash"),
        ):
            _sha(value, name)
        result = self.retrieval_result
        if (
            self.query.user_id != self.plan.user_id
            or self.query.user_id != self.eligibility.user_id
            or self.query.user_id != result.user_id
            or self.query.query_id != self.plan.query_id
            or self.query.query_id != result.query_id
            or self.plan.plan_id != self.eligibility.plan_id
            or self.plan.plan_id != result.plan_id
            or self.query.index_version != self.plan.index_version
            or self.query.index_version != self.eligibility.index_version
            or self.eligibility.snapshot_run_id != result.snapshot_run_id
            or self.query.enabled_record_kinds != BASELINE_RECORD_KINDS.get(result.baseline_id)
            or self.plan.enabled_record_kinds != self.query.enabled_record_kinds
            or self.plan.as_of != self.query.as_of
        ):
            raise EvidencePackageError("query, plan, eligibility, and retrieval result differ")
        eligible = {item.index_record_id for item in result.accepted}
        eligible.update(
            item.index_record_id for item in result.rejected if item.stage == "post_rank"
        )
        pre = tuple(sorted(
            (item for item in result.rejected if item.stage == "pre_filter"),
            key=lambda item: item.index_record_id,
        ))
        if tuple(sorted(eligible)) != self.eligibility.eligible_record_ids or pre != self.eligibility.pre_filter_rejections:
            raise EvidencePackageError("frozen eligibility differs from retrieval result")
        if self.retrieval_result_sha256 != stable_sha256(result):
            raise EvidencePackageError("retrieval result bytes changed")


@dataclass(frozen=True)
class EvidenceAnchor:
    index_record_id: str
    record_kind: str
    retrieval_rank: int
    atomic_claim_version_id: str | None
    session_summary_id: str | None

    def __post_init__(self) -> None:
        _id(self.index_record_id, "record ID")
        if self.record_kind not in {"atomic", "session"} or not 1 <= self.retrieval_rank <= 10:
            raise EvidencePackageError("evidence anchor is invalid")
        if self.record_kind == "atomic":
            _id(self.atomic_claim_version_id, "claim version ID")
            if self.session_summary_id is not None:
                raise EvidencePackageError("atomic anchor has a session")
        else:
            _id(self.session_summary_id, "summary ID")
            if self.atomic_claim_version_id is not None:
                raise EvidencePackageError("session anchor has an atomic version")


@dataclass(frozen=True)
class CheckedRelationReference:
    relation_id: str
    relation_type: str
    direction: str
    source_claim_id: str
    target_claim_id: str

    def __post_init__(self) -> None:
        for value in (self.relation_id, self.source_claim_id, self.target_claim_id):
            _id(value, "relation field")
        if self.relation_type not in {
            "supports", "contradicts", "corrects", "supersedes", "refines",
            "same_event_as", "caused_by", "hindered_by", "same_topic_as",
        } or self.direction not in {"incoming", "outgoing", "symmetric"}:
            raise EvidencePackageError("relation reference is invalid")


@dataclass(frozen=True)
class EvidenceClaim:
    user_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: object
    polarity: str
    epistemic_status: str
    memory_kind: str | None
    lifecycle_status: str
    extraction_confidence: float
    belief_confidence: float | None
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    time_precision: str
    transaction_from: datetime
    transaction_to: datetime | None
    sensitivity: str | None
    retrieval_anchors: tuple[EvidenceAnchor, ...]
    evidence_ids: tuple[str, ...]
    checked_relations: tuple[CheckedRelationReference, ...]

    def __post_init__(self) -> None:
        for value in (self.user_id, self.claim_id, self.claim_version_id, self.subject_id, self.speaker_id, self.predicate):
            _id(value, "claim field")
        _json_safe(self.object_json)
        if self.polarity not in {"positive", "negative"}:
            raise EvidencePackageError("claim polarity is invalid")
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise EvidencePackageError("claim epistemic status is invalid")
        if self.lifecycle_status not in LIFECYCLE_STATUSES:
            raise EvidencePackageError("claim lifecycle is invalid")
        if self.memory_kind not in {None, "episodic", "durative"}:
            raise EvidencePackageError("memory kind is invalid")
        _confidence(self.extraction_confidence, "extraction confidence")
        if self.belief_confidence is not None:
            _confidence(self.belief_confidence, "belief confidence")
        _valid_time(
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
        )
        _aware(self.transaction_from, "transaction from")
        if self.transaction_to is not None:
            _aware(self.transaction_to, "transaction to")
            if self.transaction_to <= self.transaction_from:
                raise EvidencePackageError("transaction interval is invalid")
        if self.sensitivity not in {None, "standard", "sensitive"}:
            raise EvidencePackageError("claim sensitivity is invalid")
        _sorted_unique(self.evidence_ids, "evidence IDs")
        expected_anchors = tuple(sorted(
            set(self.retrieval_anchors), key=lambda item: (item.retrieval_rank, item.index_record_id)
        ))
        if not self.retrieval_anchors or self.retrieval_anchors != expected_anchors:
            raise EvidencePackageError("claim anchors are not canonical")
        expected_relations = tuple(sorted(
            set(self.checked_relations), key=lambda item: (item.relation_type, item.relation_id, item.direction)
        ))
        if self.checked_relations != expected_relations:
            raise EvidencePackageError("claim relations are not canonical")


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    claim_id: str
    claim_version_id: str
    source_id: str
    span_id: str
    message_id: str | None
    speaker_id: str
    verbatim_quote: str
    start_offset: int | None
    end_offset: int | None
    support_type: str
    extraction_confidence: float

    def __post_init__(self) -> None:
        _sha(self.evidence_id, "evidence ID")
        for value in (self.claim_id, self.claim_version_id, self.source_id, self.span_id, self.speaker_id):
            _id(value, "evidence field")
        if self.message_id is not None:
            _id(self.message_id, "message ID")
        if not isinstance(self.verbatim_quote, str) or not self.verbatim_quote:
            raise EvidencePackageError("evidence quote is empty")
        if (self.start_offset is None) != (self.end_offset is None):
            raise EvidencePackageError("evidence offsets are incomplete")
        if self.start_offset is not None and not (
            type(self.start_offset) is int and type(self.end_offset) is int
            and 0 <= self.start_offset < self.end_offset
        ):
            raise EvidencePackageError("evidence offsets are invalid")
        if self.support_type not in SUPPORT_TYPES:
            raise EvidencePackageError("support type is invalid")
        _confidence(self.extraction_confidence, "extraction confidence")


@dataclass(frozen=True)
class EvidenceSource:
    user_id: str
    source_id: str
    source_type: str
    session_id: str | None
    produced_at: datetime
    ingested_at: datetime
    content_hash: str
    evidence_spans: tuple[EvidenceSpan, ...]

    def __post_init__(self) -> None:
        _id(self.user_id, "source user")
        _id(self.source_id, "source ID")
        if self.session_id is not None:
            _id(self.session_id, "session ID")
        if self.source_type not in {"conversation", "email", "calendar", "chat"}:
            raise EvidencePackageError("source type is invalid")
        _aware(self.produced_at, "produced at")
        _aware(self.ingested_at, "ingested at")
        _sha(self.content_hash, "content hash")
        expected = tuple(sorted(
            set(self.evidence_spans),
            key=lambda item: (item.span_id, item.claim_id, item.claim_version_id, item.support_type),
        ))
        if not self.evidence_spans or self.evidence_spans != expected:
            raise EvidencePackageError("source spans are not canonical")
        if any(item.source_id != self.source_id for item in self.evidence_spans):
            raise EvidencePackageError("source and evidence IDs differ")
        if self.source_type == "calendar" and any(item.message_id is not None for item in self.evidence_spans):
            raise EvidencePackageError("calendar evidence cannot carry a message ID")


@dataclass(frozen=True)
class RejectedEvidence:
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
            raise EvidencePackageError("rejected claim anchor is incomplete")
        if self.claim_id is not None:
            _id(self.claim_id, "rejected claim ID")
            _id(self.claim_version_id, "rejected claim version ID")
        if self.stage not in REJECTION_STAGES:
            raise EvidencePackageError("rejection stage is invalid")
        if self.stage == "package_validation":
            if type(self.retrieval_rank) is not int or not 1 <= self.retrieval_rank <= 10:
                raise EvidencePackageError("package rejection rank is invalid")
        elif self.retrieval_rank is not None:
            raise EvidencePackageError("carried rejection cannot invent a retrieval rank")
        if not self.reasons or self.reasons != tuple(sorted(set(self.reasons))):
            raise EvidencePackageError("rejection reasons are not canonical")
        expected_id = rejection_id(
            self.index_record_id,
            self.retrieval_rank,
            self.claim_id,
            self.claim_version_id,
            self.stage,
            self.reasons,
        )
        if self.rejection_id != expected_id:
            raise EvidencePackageError("rejection ID changed")


@dataclass(frozen=True)
class EvidenceCoverage:
    retrieved_claim_version_count: int
    supported_claim_version_count: int
    relevant_source_count: int
    relevant_span_count: int
    ratio: str | None
    null_reason: str | None
    complete: bool

    def __post_init__(self) -> None:
        values = (
            self.retrieved_claim_version_count,
            self.supported_claim_version_count,
            self.relevant_source_count,
            self.relevant_span_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise EvidencePackageError("coverage count is invalid")
        if self.supported_claim_version_count > self.retrieved_claim_version_count:
            raise EvidencePackageError("coverage numerator exceeds denominator")
        if self.retrieved_claim_version_count == 0:
            if self.ratio is not None or self.null_reason != "no_retrieved_claims" or self.complete:
                raise EvidencePackageError("zero coverage denominator is invalid")
        else:
            expected_ratio = f"{self.supported_claim_version_count / self.retrieved_claim_version_count:.6f}"
            if (
                self.ratio != expected_ratio
                or RATIO.fullmatch(self.ratio or "") is None
                or self.null_reason is not None
                or self.complete != (
                    self.supported_claim_version_count == self.retrieved_claim_version_count
                )
            ):
                raise EvidencePackageError("coverage result is invalid")


@dataclass(frozen=True)
class EvidencePackage:
    package_id: str
    package_version: str
    schema_version: str
    config_version: str
    config_sha256: str
    runtime_version: str
    input_release_version: str
    input_release_manifest_sha256: str
    input_release_checkpoint_sha256: str
    input_results_sha256: str
    query: RetrievalQueryRequest
    user_id: str
    baseline_id: str
    execution_id: str
    plan: QueryPlan
    eligibility: FrozenEligibilitySnapshot
    snapshot_run_id: str
    index_version: str
    retrieval_result_sha256: str
    query_type: str
    requested_valid_time: RequestedValidTime | None
    as_of: datetime
    current_claims: tuple[EvidenceClaim, ...]
    historical_claims: tuple[EvidenceClaim, ...]
    conflicting_claims: tuple[EvidenceClaim, ...]
    relevant_sources: tuple[EvidenceSource, ...]
    rejected_evidence: tuple[RejectedEvidence, ...]
    evidence_coverage: EvidenceCoverage
    structural_blockers: tuple[str, ...]
    answer_allowed: bool

    def __post_init__(self) -> None:
        _sha(self.package_id, "package ID")
        _sha(self.execution_id, "execution ID")
        for value, name in (
            (self.config_sha256, "config hash"),
            (self.input_release_manifest_sha256, "input manifest hash"),
            (self.input_release_checkpoint_sha256, "input checkpoint hash"),
            (self.input_results_sha256, "input results hash"),
            (self.retrieval_result_sha256, "retrieval result hash"),
        ):
            _sha(value, name)
        _id(self.snapshot_run_id, "snapshot run ID")
        if (
            self.package_version != PACKAGE_VERSION
            or self.schema_version != SCHEMA_VERSION
            or self.config_version != CONFIG_VERSION
            or self.runtime_version != RUNTIME_VERSION
            or self.input_release_version != INPUT_RELEASE_VERSION
        ):
            raise EvidencePackageError("package version metadata changed")
        if (
            self.user_id != self.query.user_id
            or self.user_id != self.plan.user_id
            or self.user_id != self.eligibility.user_id
            or self.query.query_id != self.plan.query_id
            or self.plan.plan_id != self.eligibility.plan_id
            or self.snapshot_run_id != self.eligibility.snapshot_run_id
            or self.index_version != self.query.index_version
            or self.index_version != self.plan.index_version
            or self.query_type != self.plan.primary_label
            or self.requested_valid_time != self.plan.requested_valid_time
            or self.as_of != self.query.as_of
            or self.baseline_id not in BASELINE_RECORD_KINDS
            or self.query.enabled_record_kinds != BASELINE_RECORD_KINDS[self.baseline_id]
            or self.package_id != package_id_from_package(self)
        ):
            raise EvidencePackageError("package metadata differs from frozen input")
        _aware(self.as_of, "as of")
        groups = (
            ("current_claims", self.current_claims),
            ("historical_claims", self.historical_claims),
            ("conflicting_claims", self.conflicting_claims),
        )
        claim_keys: list[tuple[str, str]] = []
        categorized_evidence: set[str] = set()
        categorized_records: set[str] = set()
        for category, claims in groups:
            expected = tuple(sorted(claims, key=_claim_order_key))
            group_keys = tuple((item.claim_id, item.claim_version_id) for item in claims)
            if claims != expected or len(group_keys) != len(set(group_keys)):
                raise EvidencePackageError("claim category order is not canonical")
            for claim in claims:
                if claim.user_id != self.user_id or CATEGORY_BY_STATUS.get(claim.lifecycle_status) != category:
                    raise EvidencePackageError("claim lifecycle category is invalid")
                claim_keys.append((claim.claim_id, claim.claim_version_id))
                categorized_evidence.update(claim.evidence_ids)
                categorized_records.update(anchor.index_record_id for anchor in claim.retrieval_anchors)
        if len(claim_keys) != len(set(claim_keys)):
            raise EvidencePackageError("claim categories overlap")
        if self.relevant_sources != tuple(sorted(self.relevant_sources, key=lambda item: item.source_id)):
            raise EvidencePackageError("sources are not canonical")
        if len(self.relevant_sources) != len({item.source_id for item in self.relevant_sources}):
            raise EvidencePackageError("sources are duplicated")
        if any(item.user_id != self.user_id for item in self.relevant_sources):
            raise EvidencePackageError("package contains another user")
        source_evidence = {
            span.evidence_id for source in self.relevant_sources for span in source.evidence_spans
        }
        if source_evidence != categorized_evidence:
            raise EvidencePackageError("claim and source evidence differ")
        rejection_expected = tuple(sorted(self.rejected_evidence, key=_rejection_order_key))
        if self.rejected_evidence != rejection_expected or len(self.rejected_evidence) != len(set(self.rejected_evidence)):
            raise EvidencePackageError("rejections are not canonical")
        rejected_claims = {
            (item.claim_id, item.claim_version_id)
            for item in self.rejected_evidence if item.claim_id is not None
        }
        if set(claim_keys) & rejected_claims:
            raise EvidencePackageError("categorized and rejected claims overlap")
        carried_rejected_records = {
            item.index_record_id for item in self.rejected_evidence if item.stage != "package_validation"
        }
        if categorized_records & carried_rejected_records:
            raise EvidencePackageError("accepted and carried-rejected records overlap")
        retrieved_claims = set(claim_keys) | rejected_claims
        supported_claims = {
            (claim.claim_id, claim.claim_version_id)
            for _, claims in groups
            for claim in claims
            if claim.evidence_ids
        } | {
            (item.claim_id, item.claim_version_id)
            for item in self.rejected_evidence
            if item.stage == "package_validation"
            and item.reasons == ("candidate_not_promoted",)
            and item.claim_id is not None
        }
        expected_coverage = EvidenceCoverage(
            len(retrieved_claims),
            len(supported_claims),
            len(self.relevant_sources),
            sum(len(item.evidence_spans) for item in self.relevant_sources),
            None if not retrieved_claims else f"{len(supported_claims) / len(retrieved_claims):.6f}",
            "no_retrieved_claims" if not retrieved_claims else None,
            bool(retrieved_claims) and len(supported_claims) == len(retrieved_claims),
        )
        if self.evidence_coverage != expected_coverage:
            raise EvidencePackageError("coverage does not recompute")
        expected_blockers = blocker_tuple(
            expected_coverage,
            bool(claim_keys),
            self.plan.clarification_required,
        )
        if self.structural_blockers != expected_blockers:
            raise EvidencePackageError("blockers do not recompute")
        if self.answer_allowed != (not expected_blockers):
            raise EvidencePackageError("answer flag and blockers differ")


@dataclass(frozen=True)
class EvidencePackageFailure:
    failure_id: str
    user_id: str
    query_id: str
    baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        for value in (self.user_id, self.query_id, self.baseline_id):
            _id(value, "failure identity")
        if SAFE_CODE.fullmatch(self.code) is None or SAFE_CODE.fullmatch(self.location) is None:
            raise EvidencePackageError("failure fields are not sanitized")


def package_id(request: EvidencePackageBuildRequest) -> str:
    return stable_sha256(_package_id_value(
        request.package_version,
        request.schema_version,
        request.config_version,
        request.config_sha256,
        request.runtime_version,
        request.input_release_version,
        request.input_release_manifest_sha256,
        request.input_release_checkpoint_sha256,
        request.input_results_sha256,
        request.query.user_id,
        request.query.query_id,
        request.retrieval_result.baseline_id,
        request.retrieval_result.execution_id,
        request.retrieval_result_sha256,
        request.retrieval_result.snapshot_run_id,
        request.query.index_version,
        request.query.as_of,
        request.plan.requested_valid_time,
    ))


def package_id_from_package(value: EvidencePackage) -> str:
    return stable_sha256(_package_id_value(
        value.package_version,
        value.schema_version,
        value.config_version,
        value.config_sha256,
        value.runtime_version,
        value.input_release_version,
        value.input_release_manifest_sha256,
        value.input_release_checkpoint_sha256,
        value.input_results_sha256,
        value.user_id,
        value.query.query_id,
        value.baseline_id,
        value.execution_id,
        value.retrieval_result_sha256,
        value.snapshot_run_id,
        value.index_version,
        value.as_of,
        value.requested_valid_time,
    ))


def _package_id_value(*values: object) -> object:
    names = (
        "package_version", "schema_version", "config_version", "config_sha256",
        "runtime_version", "input_release_version", "input_release_manifest_sha256",
        "input_release_checkpoint_sha256", "input_results_sha256", "user_id", "query_id",
        "baseline_id", "execution_id", "retrieval_result_sha256", "snapshot_run_id",
        "index_version", "as_of", "requested_valid_time",
    )
    return dict(zip(names, values, strict=True))


def rejection_id(
    index_record_id: str,
    retrieval_rank: int | None,
    claim_id: str | None,
    claim_version_id: str | None,
    stage: str,
    reasons: tuple[str, ...],
) -> str:
    return stable_sha256({
        "index_record_id": index_record_id,
        "retrieval_rank": retrieval_rank,
        "claim_id": claim_id,
        "claim_version_id": claim_version_id,
        "stage": stage,
        "reasons": reasons,
    })


def evidence_id(user_id: str, claim_id: str, version_id: str, source_id: str, span_id: str, support_type: str) -> str:
    return stable_sha256({
        "user_id": user_id,
        "claim_id": claim_id,
        "claim_version_id": version_id,
        "source_id": source_id,
        "span_id": span_id,
        "support_type": support_type,
    })


def blocker_tuple(coverage: EvidenceCoverage, has_promoted: bool, clarification: bool) -> tuple[str, ...]:
    applicable = {
        "no_retrieved_claims": coverage.retrieved_claim_version_count == 0,
        "incomplete_evidence": coverage.retrieved_claim_version_count > 0 and not coverage.complete,
        "no_promoted_claims": not has_promoted,
        "clarification_required": clarification,
    }
    return tuple(name for name in BLOCKERS if applicable[name])


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _claim_order_key(item: EvidenceClaim) -> tuple[int, str, str]:
    return (
        min(anchor.retrieval_rank for anchor in item.retrieval_anchors),
        item.claim_id,
        item.claim_version_id,
    )


def _rejection_order_key(item: RejectedEvidence) -> tuple[int, int, str, str, str]:
    return (
        REJECTION_STAGES.index(item.stage),
        item.retrieval_rank or 0,
        item.index_record_id,
        item.claim_id or "",
        item.claim_version_id or "",
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
        raise EvidencePackageError(f"{name} is invalid")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise EvidencePackageError(f"{name} is invalid")


def _aware(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EvidencePackageError(f"{name} is not timezone-aware")


def _confidence(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise EvidencePackageError(f"{name} is invalid")


def _sorted_unique(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))) or any(not isinstance(value, str) or not value for value in values):
        raise EvidencePackageError(f"{name} are not sorted and unique")


def _valid_time(
    precision: str,
    start_date: date | None,
    start_timestamp: datetime | None,
    end_date: date | None,
    end_timestamp: datetime | None,
) -> None:
    if precision not in TIME_PRECISIONS:
        raise EvidencePackageError("time precision is invalid")
    dates = (start_date, end_date)
    timestamps = (start_timestamp, end_timestamp)
    if any(value is not None and type(value) is not date for value in dates):
        raise EvidencePackageError("valid dates are invalid")
    for value in timestamps:
        if value is not None:
            _aware(value, "valid timestamp")
    if precision == "unknown":
        valid = not any(value is not None for value in (*dates, *timestamps))
    elif precision == "timestamp":
        valid = any(value is not None for value in timestamps) and not any(value is not None for value in dates)
    else:
        valid = any(value is not None for value in dates) and not any(value is not None for value in timestamps)
    if not valid:
        raise EvidencePackageError("valid-time representation is invalid")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise EvidencePackageError("valid date interval is reversed")
    if start_timestamp is not None and end_timestamp is not None and start_timestamp > end_timestamp:
        raise EvidencePackageError("valid timestamp interval is reversed")


def _json_safe(value: object) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if value is None or isinstance(value, (dict, list)) and not encoded:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise EvidencePackageError("object JSON is unsafe") from error
