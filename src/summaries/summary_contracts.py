"""Typed contracts for deterministic, source-grounded session summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping

from storage.contracts import StorageValidationError, safe_json


RENDERER_VERSION = "session_summary_renderer_v1"
BOUNDARY_VERSION = "session_boundaries_v1"
ELIGIBLE_LIFECYCLE = (
    "candidate", "confirmed", "current", "historical", "disputed", "superseded"
)
EPISTEMIC_STATUSES = frozenset(
    {"asserted", "inferred", "reported_by_other", "hypothetical", "uncertain", "denied", "corrected"}
)
LIFECYCLE_STATUSES = frozenset((*ELIGIBLE_LIFECYCLE, "excluded"))
SENSITIVITIES = frozenset({"standard", "sensitive", "restricted"})
TIME_PRECISIONS = frozenset({"timestamp", "day", "month", "year", "approximate", "unknown"})
SUPPORT_TYPES = frozenset({"supports", "contradicts", "corrects"})
SOURCE_TYPES = frozenset({"conversation", "email", "chat", "calendar"})
STATEMENT_KINDS = frozenset({"observed_fact", "unresolved_question"})
LIFECYCLE_VIEWS = frozenset({"accepted", "candidate", "historical", "disputed"})
CONFIG_FIELDS = frozenset(
    {
        "renderer_version", "rule_version", "session_boundary_version",
        "observed_facts_heading", "unresolved_questions_heading", "empty_marker",
        "predicate_format", "object_format", "eligible_lifecycle_statuses",
        "excluded_epistemic_statuses", "excluded_sensitivities", "question_triggers",
        "statement_order", "review_status",
    }
)


class GroundedSummaryError(ValueError):
    """Reject unsafe or inconsistent summary input without exposing source text."""


@dataclass(frozen=True)
class SummaryRendererConfig:
    renderer_version: str
    rule_version: str
    session_boundary_version: str
    observed_facts_heading: str
    unresolved_questions_heading: str
    empty_marker: str
    predicate_format: str
    object_format: str
    eligible_lifecycle_statuses: tuple[str, ...]
    excluded_epistemic_statuses: tuple[str, ...]
    excluded_sensitivities: tuple[str, ...]
    question_triggers: tuple[str, ...]
    statement_order: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        expected = (
            RENDERER_VERSION,
            "grounded_summary_rules_v1",
            BOUNDARY_VERSION,
            "Observed facts",
            "Unresolved questions",
            "None.",
            "underscores_to_spaces",
            "canonical_json",
            ELIGIBLE_LIFECYCLE,
            ("hypothetical",),
            ("restricted",),
            ("uncertain", "disputed"),
            ("source_produced_at", "source_order", "span_start_offset", "claim_id", "statement_kind"),
            "implementation_reviewed",
        )
        actual = (
            self.renderer_version,
            self.rule_version,
            self.session_boundary_version,
            self.observed_facts_heading,
            self.unresolved_questions_heading,
            self.empty_marker,
            self.predicate_format,
            self.object_format,
            self.eligible_lifecycle_statuses,
            self.excluded_epistemic_statuses,
            self.excluded_sensitivities,
            self.question_triggers,
            self.statement_order,
            self.review_status,
        )
        if actual != expected:
            raise GroundedSummaryError("summary renderer configuration changed")


@dataclass(frozen=True)
class GroundedSummaryRequest:
    user_id: str
    session_definition_id: str
    session_membership_sha256: str
    transaction_as_of: datetime
    idempotency_key: str
    renderer_version: str = RENDERER_VERSION

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _sha256(self.session_definition_id, "session_definition_id")
        _sha256(self.session_membership_sha256, "session_membership_sha256")
        _aware(self.transaction_as_of, "transaction_as_of")
        _text(self.idempotency_key, "idempotency_key")
        if self.renderer_version != RENDERER_VERSION:
            raise GroundedSummaryError("renderer_version is unsupported")


@dataclass(frozen=True)
class SummaryEvidence:
    evidence_id: str
    user_id: str
    session_definition_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: object
    epistemic_status: str
    lifecycle_status: str
    sensitivity: str
    support_type: str
    time_precision: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    transaction_from: datetime
    transaction_to: datetime | None
    source_id: str
    source_type: str
    source_produced_at: datetime
    source_ingested_at: datetime
    source_order: int
    span_id: str
    span_start_offset: int | None
    span_end_offset: int | None
    exact_quote: str
    linked_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "user_id", "claim_id", "claim_version_id", "subject_id", "speaker_id",
            "predicate", "source_id", "span_id", "exact_quote",
        ):
            _text(getattr(self, name), name)
        _sha256(self.evidence_id, "evidence_id")
        _sha256(self.session_definition_id, "session_definition_id")
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise GroundedSummaryError("epistemic_status is invalid")
        if self.lifecycle_status not in LIFECYCLE_STATUSES:
            raise GroundedSummaryError("lifecycle_status is invalid")
        if self.sensitivity not in SENSITIVITIES:
            raise GroundedSummaryError("sensitivity is invalid")
        if self.support_type not in SUPPORT_TYPES:
            raise GroundedSummaryError("support_type is invalid")
        if self.time_precision not in TIME_PRECISIONS:
            raise GroundedSummaryError("time_precision is invalid")
        if self.source_type not in SOURCE_TYPES:
            raise GroundedSummaryError("source_type is invalid")
        if not isinstance(self.source_order, int) or isinstance(self.source_order, bool) or self.source_order < 0:
            raise GroundedSummaryError("source_order is invalid")
        for name in ("transaction_from", "source_produced_at", "source_ingested_at"):
            _aware(getattr(self, name), name)
        if self.transaction_to is not None:
            _aware(self.transaction_to, "transaction_to")
            if self.transaction_to <= self.transaction_from:
                raise GroundedSummaryError("transaction interval is invalid")
        _valid_time(self)
        if (self.span_start_offset is None) != (self.span_end_offset is None):
            raise GroundedSummaryError("span offsets must both be null or set")
        if self.span_start_offset is not None and (
            isinstance(self.span_start_offset, bool)
            or isinstance(self.span_end_offset, bool)
            or not isinstance(self.span_start_offset, int)
            or not isinstance(self.span_end_offset, int)
            or self.span_start_offset < 0
            or self.span_end_offset <= self.span_start_offset
        ):
            raise GroundedSummaryError("span offsets are invalid")
        if tuple(sorted(set(self.linked_claim_ids))) != self.linked_claim_ids:
            raise GroundedSummaryError("linked_claim_ids must be sorted and unique")
        if self.lifecycle_status == "disputed":
            if self.claim_id not in self.linked_claim_ids or len(self.linked_claim_ids) < 2:
                raise GroundedSummaryError("disputed evidence must link every side")
        elif self.linked_claim_ids:
            raise GroundedSummaryError("only disputed evidence may link claims")
        try:
            value = safe_json(self.object_json, "object_json")
        except StorageValidationError as error:
            raise GroundedSummaryError("object_json is not JSON-safe") from error
        object.__setattr__(self, "object_json", value)


@dataclass(frozen=True)
class GroundedStatement:
    statement_id: str
    statement_kind: str
    lifecycle_view: str
    text: str
    claim_ids: tuple[str, ...]
    evidence: tuple[SummaryEvidence, ...]
    sensitive: bool

    def __post_init__(self) -> None:
        _sha256(self.statement_id, "statement_id")
        if self.statement_kind not in STATEMENT_KINDS:
            raise GroundedSummaryError("statement_kind is invalid")
        if self.lifecycle_view not in LIFECYCLE_VIEWS:
            raise GroundedSummaryError("lifecycle_view is invalid")
        _text(self.text, "text")
        if tuple(sorted(set(self.claim_ids))) != self.claim_ids or not self.claim_ids:
            raise GroundedSummaryError("statement claim IDs must be sorted and unique")
        if not self.evidence:
            raise GroundedSummaryError("statement evidence is required")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise GroundedSummaryError("statement evidence IDs must be unique")
        evidence_claims = {item.claim_id for item in self.evidence}
        if set(self.claim_ids) != evidence_claims:
            raise GroundedSummaryError("statement claims and evidence must match exactly")
        if self.sensitive != any(item.sensitivity == "sensitive" for item in self.evidence):
            raise GroundedSummaryError("statement sensitivity flag changed")


@dataclass(frozen=True)
class SessionSummary:
    summary_id: str
    user_id: str
    session_definition_id: str
    renderer_version: str
    input_snapshot_sha256: str
    summary_text: str
    observed_facts: tuple[GroundedStatement, ...]
    unresolved_questions: tuple[GroundedStatement, ...]
    valid_time_kind: str
    valid_time_start_date: date | None
    valid_time_end_date: date | None
    valid_time_start_timestamp: datetime | None
    valid_time_end_timestamp: datetime | None
    source_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    contains_sensitive: bool

    def __post_init__(self) -> None:
        _sha256(self.summary_id, "summary_id")
        _text(self.user_id, "user_id")
        _sha256(self.session_definition_id, "session_definition_id")
        _sha256(self.input_snapshot_sha256, "input_snapshot_sha256")
        if self.renderer_version != RENDERER_VERSION:
            raise GroundedSummaryError("summary renderer version changed")
        _text(self.summary_text, "summary_text")
        if any(item.statement_kind != "observed_fact" for item in self.observed_facts):
            raise GroundedSummaryError("observed fact view changed")
        if any(item.statement_kind != "unresolved_question" for item in self.unresolved_questions):
            raise GroundedSummaryError("question view changed")
        if self.valid_time_kind not in {"date", "timestamp", "unknown", "mixed"}:
            raise GroundedSummaryError("valid_time_kind is invalid")
        _summary_time(self)
        for name in ("source_ids", "claim_ids"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise GroundedSummaryError(f"{name} must be sorted and unique")
        evidence = tuple(
            item for statement in (*self.observed_facts, *self.unresolved_questions)
            for item in statement.evidence
        )
        if set(self.source_ids) != {item.source_id for item in evidence}:
            raise GroundedSummaryError("summary source view changed")
        if set(self.claim_ids) != {item.claim_id for item in evidence}:
            raise GroundedSummaryError("summary claim view changed")
        if self.contains_sensitive != any(item.sensitive for item in self.observed_facts):
            raise GroundedSummaryError("summary sensitivity flag changed")

    @property
    def accepted_view(self) -> tuple[GroundedStatement, ...]:
        return tuple(item for item in self.observed_facts if item.lifecycle_view == "accepted")

    @property
    def candidate_view(self) -> tuple[GroundedStatement, ...]:
        return tuple(item for item in self.observed_facts if item.lifecycle_view == "candidate")

    @property
    def historical_view(self) -> tuple[GroundedStatement, ...]:
        return tuple(item for item in self.observed_facts if item.lifecycle_view == "historical")

    @property
    def disputed_view(self) -> tuple[GroundedStatement, ...]:
        return tuple(item for item in self.observed_facts if item.lifecycle_view == "disputed")


@dataclass(frozen=True)
class GroundedSummaryResult:
    plan_id: str
    input_snapshot_sha256: str
    request: GroundedSummaryRequest
    summary: SessionSummary

    def __post_init__(self) -> None:
        _sha256(self.plan_id, "plan_id")
        _sha256(self.input_snapshot_sha256, "input_snapshot_sha256")
        if (
            self.summary.user_id != self.request.user_id
            or self.summary.session_definition_id != self.request.session_definition_id
            or self.summary.input_snapshot_sha256 != self.input_snapshot_sha256
            or self.summary.renderer_version != self.request.renderer_version
        ):
            raise GroundedSummaryError("summary result leaves its request boundary")


def load_summary_renderer_config(path: str | Path) -> SummaryRendererConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedSummaryError("summary renderer configuration is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise GroundedSummaryError("summary renderer configuration fields changed")
    for name in (
        "eligible_lifecycle_statuses", "excluded_epistemic_statuses",
        "excluded_sensitivities", "question_triggers", "statement_order",
    ):
        value[name] = tuple(value[name])
    return SummaryRendererConfig(**value)


def canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )


def _json_value(value: object) -> object:
    if isinstance(value, datetime | date):
        if isinstance(value, datetime):
            _aware(value, "canonical timestamp")
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GroundedSummaryError("canonical JSON keys must be strings")
            result[key] = _json_value(item)
        return result
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise GroundedSummaryError("value is not canonical JSON")


def _valid_time(value: SummaryEvidence) -> None:
    dates = (value.valid_from_date, value.valid_to_date)
    timestamps = (value.valid_from_timestamp, value.valid_to_timestamp)
    if any(item is not None and type(item) is not date for item in dates):
        raise GroundedSummaryError("valid date boundary is invalid")
    for item in timestamps:
        if item is not None:
            _aware(item, "valid timestamp boundary")
    if any(item is not None for item in dates) and any(item is not None for item in timestamps):
        raise GroundedSummaryError("valid-time representations cannot mix")
    if value.time_precision == "unknown":
        if any(item is not None for item in (*dates, *timestamps)):
            raise GroundedSummaryError("unknown valid time must be empty")
    elif value.time_precision == "timestamp":
        if not any(item is not None for item in timestamps) or any(item is not None for item in dates):
            raise GroundedSummaryError("timestamp precision requires timestamp boundaries")
    elif not any(item is not None for item in dates) or any(item is not None for item in timestamps):
        raise GroundedSummaryError("date precision requires date boundaries")
    if dates[0] is not None and dates[1] is not None and dates[1] < dates[0]:
        raise GroundedSummaryError("valid date interval is reversed")
    if timestamps[0] is not None and timestamps[1] is not None and timestamps[1] < timestamps[0]:
        raise GroundedSummaryError("valid timestamp interval is reversed")
    if value.lifecycle_status == "current" and not any(
        item is not None for item in (*dates, *timestamps)
    ):
        raise GroundedSummaryError("current evidence requires valid time")


def _summary_time(value: SessionSummary) -> None:
    dates = (value.valid_time_start_date, value.valid_time_end_date)
    timestamps = (value.valid_time_start_timestamp, value.valid_time_end_timestamp)
    if value.valid_time_kind == "date":
        if any(item is not None for item in timestamps):
            raise GroundedSummaryError("date summary cannot carry timestamps")
        if dates[0] is not None and dates[1] is not None and dates[1] < dates[0]:
            raise GroundedSummaryError("summary date interval is reversed")
    elif value.valid_time_kind == "timestamp":
        if any(item is not None for item in dates):
            raise GroundedSummaryError("timestamp summary cannot carry dates")
        for item in timestamps:
            if item is not None:
                _aware(item, "summary timestamp")
        if timestamps[0] is not None and timestamps[1] is not None and timestamps[1] < timestamps[0]:
            raise GroundedSummaryError("summary timestamp interval is reversed")
    elif any(item is not None for item in (*dates, *timestamps)):
        raise GroundedSummaryError("unknown or mixed summary time must be empty")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundedSummaryError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise GroundedSummaryError(f"{name} must be timezone-aware")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise GroundedSummaryError(f"{name} must be a lowercase SHA-256")
    return value
