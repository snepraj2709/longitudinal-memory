"""Typed contracts for deterministic retrieval-query planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import re
from typing import Mapping, Sequence


PLANNER_VERSION = "retrieval_query_planner_v1"
INDEX_VERSION = "retrieval_index_v1"
QUERY_LABELS = (
    "current_state",
    "historical_state",
    "change_over_time",
    "specific_event",
    "relationship",
    "commitment",
    "evidence_request",
    "unknown",
)
RECORD_KINDS = ("atomic", "session")
SENSITIVITY_SCOPES = ("standard", "sensitive")
LIFECYCLE_STATUSES = (
    "candidate",
    "confirmed",
    "current",
    "historical",
    "disputed",
    "superseded",
)
RELATION_EXPANSION_TYPES = (
    "corrects",
    "supersedes",
    "contradicts",
    "refines",
    "same_event_as",
)
REJECTION_REASONS = frozenset(
    {
        "transaction_hidden",
        "source_after_as_of",
        "relation_after_as_of",
        "valid_time_mismatch",
        "unknown_valid_time",
        "speaker_mismatch",
        "entity_mismatch",
        "lifecycle_blocked",
        "sensitive_not_authorized",
        "unclassified_sensitivity",
        "restricted",
        "stale_lineage",
        "partial_lineage",
    }
)
FAILURE_CODES = frozenset({"invalid_request", "invalid_config", "planning_failed"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SAFE_FAILURE_FIELD = re.compile(r"^[a-z0-9_]{1,64}$")


class RetrievalQueryError(ValueError):
    """Reject an unsafe query, plan, or eligibility value."""


@dataclass(frozen=True)
class RequestedValidTime:
    kind: str
    point_date: date | None = None
    point_timestamp: datetime | None = None
    range_start_date: date | None = None
    range_end_date: date | None = None
    range_start_timestamp: datetime | None = None
    range_end_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"point", "range"}:
            raise RetrievalQueryError("valid-time kind is invalid")
        dates = (
            self.point_date,
            self.range_start_date,
            self.range_end_date,
        )
        timestamps = (
            self.point_timestamp,
            self.range_start_timestamp,
            self.range_end_timestamp,
        )
        if any(value is not None and type(value) is not date for value in dates):
            raise RetrievalQueryError("valid date is invalid")
        for value in timestamps:
            if value is not None:
                _aware(value, "valid timestamp")
        if any(value is not None for value in dates) and any(
            value is not None for value in timestamps
        ):
            raise RetrievalQueryError("valid-time representations cannot mix")
        if self.kind == "point":
            if sum(value is not None for value in (self.point_date, self.point_timestamp)) != 1:
                raise RetrievalQueryError("valid-time point requires one value")
            if any(
                value is not None
                for value in (
                    self.range_start_date,
                    self.range_end_date,
                    self.range_start_timestamp,
                    self.range_end_timestamp,
                )
            ):
                raise RetrievalQueryError("valid-time point cannot have range values")
        else:
            if self.point_date is not None or self.point_timestamp is not None:
                raise RetrievalQueryError("valid-time range cannot have a point")
            date_range = (self.range_start_date, self.range_end_date)
            timestamp_range = (self.range_start_timestamp, self.range_end_timestamp)
            if not any(value is not None for value in (*date_range, *timestamp_range)):
                raise RetrievalQueryError("valid-time range requires a boundary")
            if date_range[0] is not None and date_range[1] is not None:
                if date_range[1] < date_range[0]:
                    raise RetrievalQueryError("valid-time range is reversed")
            if timestamp_range[0] is not None and timestamp_range[1] is not None:
                if timestamp_range[1] < timestamp_range[0]:
                    raise RetrievalQueryError("valid-time range is reversed")


@dataclass(frozen=True)
class RetrievalQueryRequest:
    query_id: str
    user_id: str
    query_text: str
    as_of: datetime
    index_version: str
    enabled_record_kinds: tuple[str, ...]
    requested_valid_time: RequestedValidTime | None
    speaker_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    sensitivity_scope: str
    allow_unclassified_sensitivity: bool

    def __post_init__(self) -> None:
        _stable_id(self.query_id, "query_id")
        _stable_id(self.user_id, "user_id")
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise RetrievalQueryError("query text is empty")
        if len(self.query_text) > 4096:
            raise RetrievalQueryError("query text is too long")
        _aware(self.as_of, "as_of")
        if self.index_version != INDEX_VERSION:
            raise RetrievalQueryError("index version is invalid")
        _sorted_unique(self.enabled_record_kinds, RECORD_KINDS, "record kinds")
        _sorted_unique_ids(self.speaker_ids, "speaker IDs")
        _sorted_unique_ids(self.entity_ids, "entity IDs")
        if self.sensitivity_scope not in SENSITIVITY_SCOPES:
            raise RetrievalQueryError("sensitivity scope is invalid")
        if type(self.allow_unclassified_sensitivity) is not bool:
            raise RetrievalQueryError("unclassified sensitivity flag is invalid")


@dataclass(frozen=True)
class QueryPlan:
    plan_id: str
    query_id: str
    user_id: str
    planner_version: str
    planner_config_sha256: str
    index_version: str
    primary_label: str
    as_of: datetime
    enabled_record_kinds: tuple[str, ...]
    requested_valid_time: RequestedValidTime | None
    speaker_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    allowed_lifecycle_statuses: tuple[str, ...]
    allow_sensitive: bool
    allow_unclassified_sensitivity: bool
    include_previous_versions: bool
    checked_relation_expansion_intent: bool
    relation_expansion_types: tuple[str, ...]
    source_evidence_intent: bool
    unresolved_time: bool
    clarification_required: bool

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.plan_id) is None:
            raise RetrievalQueryError("plan ID is invalid")
        _stable_id(self.query_id, "query_id")
        _stable_id(self.user_id, "user_id")
        if self.planner_version != PLANNER_VERSION:
            raise RetrievalQueryError("planner version is invalid")
        if SHA256.fullmatch(self.planner_config_sha256) is None:
            raise RetrievalQueryError("planner config hash is invalid")
        if self.index_version != INDEX_VERSION:
            raise RetrievalQueryError("index version is invalid")
        if self.primary_label not in QUERY_LABELS:
            raise RetrievalQueryError("query label is invalid")
        _aware(self.as_of, "as_of")
        _sorted_unique(self.enabled_record_kinds, RECORD_KINDS, "record kinds")
        _sorted_unique_ids(self.speaker_ids, "speaker IDs")
        _sorted_unique_ids(self.entity_ids, "entity IDs")
        _ordered_subset(
            self.allowed_lifecycle_statuses,
            LIFECYCLE_STATUSES,
            "lifecycle statuses",
        )
        if self.relation_expansion_types not in ((), RELATION_EXPANSION_TYPES):
            raise RetrievalQueryError("relation expansion types are invalid")
        for name in (
            "allow_sensitive",
            "allow_unclassified_sensitivity",
            "include_previous_versions",
            "checked_relation_expansion_intent",
            "source_evidence_intent",
            "unresolved_time",
            "clarification_required",
        ):
            if type(getattr(self, name)) is not bool:
                raise RetrievalQueryError(f"{name} is invalid")
        if self.checked_relation_expansion_intent != bool(self.relation_expansion_types):
            raise RetrievalQueryError("relation expansion intent is inconsistent")
        if self.clarification_required != self.unresolved_time:
            raise RetrievalQueryError("clarification flag is inconsistent")


@dataclass(frozen=True)
class EligibilityDecision:
    index_record_id: str
    eligible: bool
    rejection_reasons: tuple[str, ...]
    lifecycle_statuses: tuple[str, ...]
    sensitivity: str | None
    unclassified_sensitivity: bool

    def __post_init__(self) -> None:
        _stable_id(self.index_record_id, "index_record_id")
        if type(self.eligible) is not bool:
            raise RetrievalQueryError("eligibility flag is invalid")
        if tuple(sorted(set(self.rejection_reasons))) != self.rejection_reasons:
            raise RetrievalQueryError("rejection reasons must be sorted and unique")
        if not set(self.rejection_reasons).issubset(REJECTION_REASONS):
            raise RetrievalQueryError("rejection reason is invalid")
        if self.eligible == bool(self.rejection_reasons):
            raise RetrievalQueryError("eligibility reasons are inconsistent")
        _ordered_subset(self.lifecycle_statuses, LIFECYCLE_STATUSES, "lifecycle statuses")
        if self.sensitivity not in {None, "standard", "sensitive", "restricted"}:
            raise RetrievalQueryError("sensitivity is invalid")
        if type(self.unclassified_sensitivity) is not bool:
            raise RetrievalQueryError("unclassified sensitivity flag is invalid")
        if self.unclassified_sensitivity != (self.sensitivity is None):
            raise RetrievalQueryError("unclassified sensitivity is inconsistent")


@dataclass(frozen=True)
class EligibilityResult:
    plan_id: str
    user_id: str
    index_version: str
    snapshot_run_id: str
    decisions: tuple[EligibilityDecision, ...]

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.plan_id) is None:
            raise RetrievalQueryError("plan ID is invalid")
        _stable_id(self.user_id, "user_id")
        _stable_id(self.snapshot_run_id, "snapshot_run_id")
        if self.index_version != INDEX_VERSION:
            raise RetrievalQueryError("index version is invalid")
        record_ids = tuple(item.index_record_id for item in self.decisions)
        if record_ids != tuple(sorted(set(record_ids))):
            raise RetrievalQueryError("eligibility decisions must be sorted and unique")

    @property
    def eligible_record_ids(self) -> tuple[str, ...]:
        return tuple(item.index_record_id for item in self.decisions if item.eligible)


@dataclass(frozen=True)
class RetrievalQueryFailure:
    failure_id: str
    query_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.failure_id) is None:
            raise RetrievalQueryError("failure ID is invalid")
        _stable_id(self.query_id, "query_id")
        _stable_id(self.user_id, "user_id")
        if self.code not in FAILURE_CODES:
            raise RetrievalQueryError("failure code is invalid")
        if SAFE_FAILURE_FIELD.fullmatch(self.location) is None:
            raise RetrievalQueryError("failure location is not sanitized")


def parse_retrieval_query_request(value: Mapping[str, object]) -> RetrievalQueryRequest:
    fields = {
        "query_id",
        "user_id",
        "query_text",
        "as_of",
        "index_version",
        "enabled_record_kinds",
        "requested_valid_time",
        "speaker_ids",
        "entity_ids",
        "sensitivity_scope",
        "allow_unclassified_sensitivity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RetrievalQueryError("request fields are invalid")
    return RetrievalQueryRequest(
        query_id=_required_string(value, "query_id"),
        user_id=_required_string(value, "user_id"),
        query_text=_required_string(value, "query_text"),
        as_of=_parse_datetime(value["as_of"], "as_of"),
        index_version=_required_string(value, "index_version"),
        enabled_record_kinds=_string_tuple(value["enabled_record_kinds"], "record kinds"),
        requested_valid_time=_parse_requested_valid_time(value["requested_valid_time"]),
        speaker_ids=_string_tuple(value["speaker_ids"], "speaker IDs"),
        entity_ids=_string_tuple(value["entity_ids"], "entity IDs"),
        sensitivity_scope=_required_string(value, "sensitivity_scope"),
        allow_unclassified_sensitivity=value["allow_unclassified_sensitivity"],
    )


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


def request_canonical_value(request: RetrievalQueryRequest) -> Mapping[str, object]:
    valid_time = None
    if request.requested_valid_time is not None:
        valid_time = {
            name: getattr(request.requested_valid_time, name)
            for name in RequestedValidTime.__dataclass_fields__
        }
    return {
        "query_id": request.query_id,
        "user_id": request.user_id,
        "query_text": request.query_text,
        "as_of": request.as_of,
        "index_version": request.index_version,
        "enabled_record_kinds": request.enabled_record_kinds,
        "requested_valid_time": valid_time,
        "speaker_ids": request.speaker_ids,
        "entity_ids": request.entity_ids,
        "sensitivity_scope": request.sensitivity_scope,
        "allow_unclassified_sensitivity": request.allow_unclassified_sensitivity,
    }


def _parse_requested_valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    fields = set(RequestedValidTime.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RetrievalQueryError("requested valid-time fields are invalid")
    return RequestedValidTime(
        kind=_required_string(value, "kind"),
        point_date=_parse_date(value["point_date"], "point_date"),
        point_timestamp=_optional_datetime(value["point_timestamp"], "point_timestamp"),
        range_start_date=_parse_date(value["range_start_date"], "range_start_date"),
        range_end_date=_parse_date(value["range_end_date"], "range_end_date"),
        range_start_timestamp=_optional_datetime(
            value["range_start_timestamp"], "range_start_timestamp"
        ),
        range_end_timestamp=_optional_datetime(
            value["range_end_timestamp"], "range_end_timestamp"
        ),
    )


def _required_string(value: Mapping[str, object], name: str) -> str:
    item = value[name]
    if not isinstance(item, str):
        raise RetrievalQueryError(f"{name} is invalid")
    return item


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RetrievalQueryError(f"{name} are invalid")
    return tuple(value)


def _parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise RetrievalQueryError(f"{name} is invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RetrievalQueryError(f"{name} is invalid") from error
    _aware(result, name)
    return result


def _optional_datetime(value: object, name: str) -> datetime | None:
    return None if value is None else _parse_datetime(value, name)


def _parse_date(value: object, name: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RetrievalQueryError(f"{name} is invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise RetrievalQueryError(f"{name} is invalid") from error


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RetrievalQueryError(f"{name} must be timezone-aware")


def _stable_id(value: str, name: str) -> None:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise RetrievalQueryError(f"{name} is invalid")


def _sorted_unique(values: tuple[str, ...], allowed: Sequence[str], name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise RetrievalQueryError(f"{name} must be nonempty")
    if values != tuple(sorted(set(values))) or not set(values).issubset(allowed):
        raise RetrievalQueryError(f"{name} must be sorted, unique, and known")


def _sorted_unique_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or values != tuple(sorted(set(values))):
        raise RetrievalQueryError(f"{name} must be sorted and unique")
    for value in values:
        _stable_id(value, name)


def _ordered_subset(values: tuple[str, ...], allowed: Sequence[str], name: str) -> None:
    expected = tuple(item for item in allowed if item in values)
    if not values or values != expected or len(set(values)) != len(values):
        raise RetrievalQueryError(f"{name} are invalid")


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")
