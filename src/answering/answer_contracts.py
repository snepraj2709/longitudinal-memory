"""Immutable contracts for grounded memory answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import math
from typing import Mapping

from retrieval.query_contracts import RequestedValidTime

from .contracts import canonical_json_bytes, stable_sha256


ANSWER_VERSION = "memory_answer_v1"
SCHEMA_VERSION = "memory_answer_schema_v1"
CONFIG_VERSION = "memory_answer_config_v1"
PROMPT_VERSION = "memory_answer_prompt_v1"
RUNTIME_VERSION = "memory_answer_runtime_v1"
INPUT_RELEASE_VERSION = "evidence_package_development_v1"
OUTPUT_RELEASE_VERSION = "memory_answer_contract_development_v1"
DORMANT_MODEL = "gpt-4.1-2025-04-14"
STATUSES = ("answered", "abstained", "disputed", "partially_answered")
CATEGORIES = ("current_claims", "historical_claims", "conflicting_claims")
TIME_PRECISIONS = ("timestamp", "day", "month", "year", "approximate", "unknown")
BLOCKERS = (
    "no_retrieved_claims",
    "incomplete_evidence",
    "no_promoted_claims",
    "clarification_required",
)
BASELINE_ORDER = {"B2": 0, "B3": 1, "B4": 2}
FIXED_ABSTENTION_TEXT = "I cannot answer this from the available memory."


class MemoryAnswerError(ValueError):
    """A deterministic contract failure."""


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryAnswerError(f"{name} is invalid")
    _json_safe(value)


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise MemoryAnswerError(f"{name} is invalid")


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MemoryAnswerError(f"{name} must be timezone-aware")


def _json_safe(value: object) -> None:
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise MemoryAnswerError("value is not JSON-safe") from error


def _valid_time(
    precision: str,
    start_date: date | None,
    start_timestamp: datetime | None,
    end_date: date | None,
    end_timestamp: datetime | None,
) -> None:
    if precision not in TIME_PRECISIONS:
        raise MemoryAnswerError("time precision is invalid")
    dates = (start_date, end_date)
    timestamps = (start_timestamp, end_timestamp)
    if any(value is not None and type(value) is not date for value in dates):
        raise MemoryAnswerError("valid date is invalid")
    for value in timestamps:
        if value is not None:
            _aware(value, "valid timestamp")
    if precision == "unknown":
        valid = not any(value is not None for value in (*dates, *timestamps))
    elif precision == "timestamp":
        valid = any(value is not None for value in timestamps) and not any(
            value is not None for value in dates
        )
    else:
        valid = any(value is not None for value in dates) and not any(
            value is not None for value in timestamps
        )
    if not valid:
        raise MemoryAnswerError("valid-time representation is invalid")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise MemoryAnswerError("valid date interval is reversed")
    if (
        start_timestamp is not None
        and end_timestamp is not None
        and start_timestamp > end_timestamp
    ):
        raise MemoryAnswerError("valid timestamp interval is reversed")


def _sorted_unique(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise MemoryAnswerError(f"{name} contains an invalid value")
    if values != tuple(sorted(set(values))):
        raise MemoryAnswerError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class AnswerPackageClaim:
    category: str
    user_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: object
    polarity: str
    epistemic_status: str
    lifecycle_status: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    time_precision: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise MemoryAnswerError("package claim category is invalid")
        for value, name in (
            (self.user_id, "claim user ID"),
            (self.claim_id, "claim ID"),
            (self.claim_version_id, "claim version ID"),
            (self.subject_id, "subject ID"),
            (self.speaker_id, "speaker ID"),
            (self.predicate, "predicate"),
        ):
            _text(value, name)
        _json_safe(self.object_json)
        expected = {
            "current_claims": {"current", "confirmed"},
            "historical_claims": {"historical", "superseded"},
            "conflicting_claims": {"disputed"},
        }[self.category]
        if self.lifecycle_status not in expected:
            raise MemoryAnswerError("claim lifecycle does not match category")
        if self.polarity not in {"positive", "negative"}:
            raise MemoryAnswerError("claim polarity is invalid")
        if self.epistemic_status not in {
            "asserted", "inferred", "reported_by_other", "hypothetical",
            "uncertain", "denied", "corrected",
        }:
            raise MemoryAnswerError("claim epistemic status is invalid")
        _valid_time(
            self.time_precision,
            self.valid_from_date,
            self.valid_from_timestamp,
            self.valid_to_date,
            self.valid_to_timestamp,
        )
        _sorted_unique(self.evidence_ids, "claim evidence IDs")
        if not self.evidence_ids:
            raise MemoryAnswerError("package claim has no evidence")


@dataclass(frozen=True)
class AnswerPackageSpan:
    evidence_id: str
    user_id: str
    claim_id: str
    claim_version_id: str
    source_id: str
    span_id: str
    message_id: str | None
    quote: str

    def __post_init__(self) -> None:
        _sha(self.evidence_id, "evidence ID")
        for value, name in (
            (self.user_id, "evidence user ID"),
            (self.claim_id, "claim ID"),
            (self.claim_version_id, "claim version ID"),
            (self.source_id, "source ID"),
            (self.span_id, "span ID"),
            (self.quote, "quote"),
        ):
            _text(value, name)
        if self.message_id is not None:
            _text(self.message_id, "message ID")


@dataclass(frozen=True)
class AnswerPackageView:
    input_release_version: str
    input_release_manifest_sha256: str
    input_packages_sha256: str
    package_id: str
    package_sha256: str
    user_id: str
    query_id: str
    query_text: str
    query_type: str
    baseline_id: str
    execution_id: str
    plan_id: str
    snapshot_run_id: str
    index_version: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    answer_allowed: bool
    structural_blockers: tuple[str, ...]
    claims: tuple[AnswerPackageClaim, ...]
    evidence_spans: tuple[AnswerPackageSpan, ...]

    def __post_init__(self) -> None:
        if self.input_release_version != INPUT_RELEASE_VERSION:
            raise MemoryAnswerError("input release version changed")
        for value, name in (
            (self.input_release_manifest_sha256, "input release manifest hash"),
            (self.input_packages_sha256, "input package file hash"),
            (self.package_id, "package ID"),
            (self.package_sha256, "package hash"),
            (self.execution_id, "execution ID"),
            (self.plan_id, "plan ID"),
            (self.snapshot_run_id, "snapshot run ID"),
        ):
            _sha(value, name)
        for value, name in (
            (self.user_id, "user ID"),
            (self.query_id, "query ID"),
            (self.query_text, "query text"),
            (self.query_type, "query type"),
            (self.index_version, "index version"),
        ):
            _text(value, name)
        if self.baseline_id not in BASELINE_ORDER:
            raise MemoryAnswerError("baseline ID is invalid")
        _aware(self.as_of, "package as_of")
        if type(self.answer_allowed) is not bool:
            raise MemoryAnswerError("answer_allowed is invalid")
        expected_blockers = tuple(item for item in BLOCKERS if item in self.structural_blockers)
        if self.structural_blockers != expected_blockers or len(set(self.structural_blockers)) != len(
            self.structural_blockers
        ):
            raise MemoryAnswerError("structural blockers are not canonical")
        if self.answer_allowed == bool(self.structural_blockers):
            raise MemoryAnswerError("answer_allowed and blockers disagree")
        category_order = {value: index for index, value in enumerate(CATEGORIES)}
        expected_claims = tuple(sorted(
            self.claims,
            key=lambda item: (
                category_order[item.category], item.claim_id, item.claim_version_id
            ),
        ))
        if self.claims != expected_claims or len({
            (item.claim_id, item.claim_version_id) for item in self.claims
        }) != len(self.claims):
            raise MemoryAnswerError("package claims are not canonical")
        expected_spans = tuple(sorted(
            self.evidence_spans,
            key=lambda item: (
                item.claim_id, item.claim_version_id, item.evidence_id,
                item.source_id, item.span_id,
            ),
        ))
        if self.evidence_spans != expected_spans or len({
            item.evidence_id for item in self.evidence_spans
        }) != len(self.evidence_spans):
            raise MemoryAnswerError("package spans are not canonical")
        spans = {item.evidence_id: item for item in self.evidence_spans}
        required = set()
        for claim in self.claims:
            if claim.user_id != self.user_id:
                raise MemoryAnswerError("package claim belongs to another user")
            required.update(claim.evidence_ids)
            for evidence_id in claim.evidence_ids:
                span = spans.get(evidence_id)
                if span is None or (span.claim_id, span.claim_version_id) != (
                    claim.claim_id, claim.claim_version_id
                ) or span.user_id != self.user_id:
                    raise MemoryAnswerError("package evidence lineage is incomplete")
        if required != set(spans):
            raise MemoryAnswerError("package span index is not exact")


@dataclass(frozen=True)
class AnswerClaimReference:
    claim_id: str
    claim_version_id: str
    category: str

    def __post_init__(self) -> None:
        _text(self.claim_id, "claim ID")
        _text(self.claim_version_id, "claim version ID")
        if self.category not in CATEGORIES:
            raise MemoryAnswerError("claim reference category is invalid")


@dataclass(frozen=True)
class AnswerCitation:
    claim_id: str
    claim_version_id: str
    evidence_id: str
    source_id: str
    span_id: str
    message_id: str | None
    quote: str

    def __post_init__(self) -> None:
        _text(self.claim_id, "citation claim ID")
        _text(self.claim_version_id, "citation version ID")
        _sha(self.evidence_id, "citation evidence ID")
        _text(self.source_id, "citation source ID")
        _text(self.span_id, "citation span ID")
        if self.message_id is not None:
            _text(self.message_id, "citation message ID")
        _text(self.quote, "citation quote")


def _reference_key(item: AnswerClaimReference) -> tuple[str, str, str]:
    return (item.category, item.claim_id, item.claim_version_id)


def _citation_key(item: AnswerCitation) -> tuple[str, str, str, str, str]:
    return (
        item.claim_id, item.claim_version_id, item.evidence_id,
        item.source_id, item.span_id,
    )


@dataclass(frozen=True)
class CandidateStatement:
    text: str
    claim_references: tuple[AnswerClaimReference, ...]
    citations: tuple[AnswerCitation, ...]

    def __post_init__(self) -> None:
        _text(self.text, "statement text")
        if not self.claim_references or self.claim_references != tuple(sorted(
            set(self.claim_references), key=_reference_key
        )):
            raise MemoryAnswerError("statement claim references are not canonical")
        if not self.citations or self.citations != tuple(sorted(
            set(self.citations), key=_citation_key
        )):
            raise MemoryAnswerError("statement citations are not canonical")


def statement_id(statement: CandidateStatement) -> str:
    return stable_sha256(statement)


@dataclass(frozen=True)
class AnswerStatement:
    statement_id: str
    text: str
    claim_references: tuple[AnswerClaimReference, ...]
    citations: tuple[AnswerCitation, ...]

    def __post_init__(self) -> None:
        candidate = CandidateStatement(self.text, self.claim_references, self.citations)
        if self.statement_id != statement_id(candidate):
            raise MemoryAnswerError("statement ID does not recompute")


@dataclass(frozen=True)
class MemoryAnswerCandidate:
    status: str
    answer: str
    confidence: float
    statements: tuple[CandidateStatement, ...]
    unresolved_parts: tuple[str, ...]
    abstention_reason: str | None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise MemoryAnswerError("candidate status is invalid")
        if not isinstance(self.answer, str):
            raise MemoryAnswerError("candidate answer is invalid")
        if type(self.confidence) not in {int, float} or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise MemoryAnswerError("candidate confidence is invalid")
        if self.statements != tuple(sorted(
            set(self.statements), key=statement_id
        )):
            raise MemoryAnswerError("candidate statements are not canonical")
        _sorted_unique(self.unresolved_parts, "candidate unresolved parts")
        if self.abstention_reason is not None:
            _text(self.abstention_reason, "candidate abstention reason")
        if self.status == "abstained":
            if (
                self.statements or self.unresolved_parts or self.confidence != 0
                or not self.abstention_reason or self.answer != FIXED_ABSTENTION_TEXT
            ):
                raise MemoryAnswerError("candidate abstention is invalid")
        else:
            if not self.statements or self.abstention_reason is not None:
                raise MemoryAnswerError("candidate lacks grounded statements")
            if self.answer != "\n".join(item.text for item in self.statements):
                raise MemoryAnswerError("candidate answer contains untracked prose")
        categories = [
            reference.category
            for statement in self.statements
            for reference in statement.claim_references
        ]
        if self.status == "answered" and (
            self.unresolved_parts or "conflicting_claims" in categories
        ):
            raise MemoryAnswerError("answered candidate is inconsistent")
        if self.status == "disputed" and (
            any(item != "conflicting_claims" for item in categories)
            or len({
                (reference.claim_id, reference.claim_version_id)
                for statement in self.statements
                for reference in statement.claim_references
            }) < 2
        ):
            raise MemoryAnswerError("disputed candidate is incomplete")
        if self.status == "partially_answered" and not self.unresolved_parts:
            raise MemoryAnswerError("partial candidate has no unresolved part")


@dataclass(frozen=True)
class MemoryAnswer:
    answer_id: str
    answer_version: str
    schema_version: str
    config_version: str
    config_sha256: str
    prompt_version: str
    prompt_sha256: str
    runtime_version: str
    input_release_version: str
    input_release_manifest_sha256: str
    input_packages_sha256: str
    package_id: str
    package_sha256: str
    user_id: str
    query_id: str
    query_text: str
    baseline_id: str
    execution_id: str
    plan_id: str
    snapshot_run_id: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    generation_mode: str
    requested_model: str | None
    resolved_model: str | None
    status: str
    answer: str
    confidence: float
    statements: tuple[AnswerStatement, ...]
    unresolved_parts: tuple[str, ...]
    abstention_reason: str | None
    structural_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.answer_version != ANSWER_VERSION
            or self.schema_version != SCHEMA_VERSION
            or self.config_version != CONFIG_VERSION
            or self.prompt_version != PROMPT_VERSION
            or self.runtime_version != RUNTIME_VERSION
            or self.input_release_version != INPUT_RELEASE_VERSION
        ):
            raise MemoryAnswerError("answer version changed")
        for value, name in (
            (self.answer_id, "answer ID"), (self.config_sha256, "config hash"),
            (self.prompt_sha256, "prompt hash"),
            (self.input_release_manifest_sha256, "release manifest hash"),
            (self.input_packages_sha256, "packages hash"),
            (self.package_id, "package ID"), (self.package_sha256, "package hash"),
            (self.execution_id, "execution ID"), (self.plan_id, "plan ID"),
            (self.snapshot_run_id, "snapshot ID"),
        ):
            _sha(value, name)
        for value, name in (
            (self.user_id, "user ID"), (self.query_id, "query ID"),
            (self.query_text, "query text"),
        ):
            _text(value, name)
        if self.baseline_id not in BASELINE_ORDER:
            raise MemoryAnswerError("answer baseline is invalid")
        _aware(self.as_of, "answer as_of")
        if self.status not in STATUSES:
            raise MemoryAnswerError("answer status is invalid")
        if self.generation_mode not in {
            "deterministic_structural_abstention", "validated_supplied_candidate"
        }:
            raise MemoryAnswerError("generation mode is invalid")
        if type(self.confidence) not in {int, float} or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise MemoryAnswerError("answer confidence is invalid")
        if self.statements != tuple(sorted(set(self.statements), key=lambda item: item.statement_id)):
            raise MemoryAnswerError("answer statements are not canonical")
        _sorted_unique(self.unresolved_parts, "answer unresolved parts")
        expected_blockers = tuple(item for item in BLOCKERS if item in self.structural_blockers)
        if self.structural_blockers != expected_blockers:
            raise MemoryAnswerError("answer blockers are not canonical")
        if self.status == "abstained":
            if (
                self.statements or self.unresolved_parts or self.confidence != 0
                or not self.abstention_reason or self.answer != FIXED_ABSTENTION_TEXT
            ):
                raise MemoryAnswerError("abstention fields are invalid")
        else:
            if not self.statements or self.abstention_reason is not None:
                raise MemoryAnswerError("grounded answer fields are invalid")
            expected_answer = "\n".join(item.text for item in self.statements)
            if self.answer != expected_answer:
                raise MemoryAnswerError("answer contains untracked prose")
        if self.status == "answered" and (
            self.unresolved_parts
            or any(ref.category == "conflicting_claims" for item in self.statements for ref in item.claim_references)
        ):
            raise MemoryAnswerError("answered status is inconsistent")
        if self.status == "disputed":
            disputed = {
                (ref.claim_id, ref.claim_version_id)
                for item in self.statements for ref in item.claim_references
                if ref.category == "conflicting_claims"
            }
            if len(disputed) < 2 or any(
                ref.category != "conflicting_claims"
                for item in self.statements for ref in item.claim_references
            ):
                raise MemoryAnswerError("disputed status is inconsistent")
        if self.status == "partially_answered" and not self.unresolved_parts:
            raise MemoryAnswerError("partial answer has no unresolved part")
        if self.structural_blockers and (
            self.status != "abstained"
            or self.generation_mode != "deterministic_structural_abstention"
            or self.requested_model is not None
            or self.resolved_model is not None
        ):
            raise MemoryAnswerError("blocked package did not short-circuit generation")
        if not self.structural_blockers and (
            self.generation_mode != "validated_supplied_candidate"
            or self.requested_model != DORMANT_MODEL
            or self.resolved_model != DORMANT_MODEL
        ):
            raise MemoryAnswerError("candidate model binding is invalid")
        if self.answer_id != answer_id(self):
            raise MemoryAnswerError("answer ID does not recompute")


def answer_id(value: MemoryAnswer) -> str:
    payload = asdict(value)
    payload.pop("answer_id")
    return stable_sha256(payload)


def answer_id_payload(payload: Mapping[str, object]) -> str:
    value = dict(payload)
    value.pop("answer_id", None)
    return stable_sha256(value)


@dataclass(frozen=True)
class MemoryAnswerFailure:
    failure_id: str
    answer_id: str | None
    package_id: str
    query_id: str
    baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        if self.answer_id is not None:
            _sha(self.answer_id, "failure answer ID")
        _sha(self.package_id, "failure package ID")
        _text(self.query_id, "failure query ID")
        if self.baseline_id not in BASELINE_ORDER:
            raise MemoryAnswerError("failure baseline is invalid")
        if self.code not in {
            "input_invalid", "candidate_invalid", "provenance_invalid",
            "output_invalid", "runtime_failure",
        } or self.location not in {
            "input", "candidate", "claim", "citation", "output", "runtime",
        }:
            raise MemoryAnswerError("failure is not sanitized")


@dataclass(frozen=True)
class MemoryAnswerChecks:
    output_release_version: str
    package_count: int
    answer_count: int
    abstained_count: int
    answered_count: int
    disputed_count: int
    partially_answered_count: int
    no_promoted_claims_count: int
    statement_count: int
    citation_count: int
    failure_count: int
    duplicate_count: int
    cross_user_count: int
    invalid_provenance_count: int
    provider_request_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int

    def __post_init__(self) -> None:
        if self.output_release_version != OUTPUT_RELEASE_VERSION:
            raise MemoryAnswerError("check release version changed")
        for name, value in asdict(self).items():
            if name != "output_release_version" and (type(value) is not int or value < 0):
                raise MemoryAnswerError("check count is invalid")
