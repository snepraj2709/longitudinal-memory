"""Deterministic retrieval-query classification and immutable plan construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Mapping

from .query_contracts import (
    INDEX_VERSION,
    LIFECYCLE_STATUSES,
    PLANNER_VERSION,
    QUERY_LABELS,
    RECORD_KINDS,
    RELATION_EXPANSION_TYPES,
    SENSITIVITY_SCOPES,
    QueryPlan,
    RequestedValidTime,
    RetrievalQueryError,
    RetrievalQueryRequest,
    request_canonical_value,
    stable_sha256,
)


CONFIG_PATH = Path("configs/retrieval/query_planner_v1.json")
NORMALIZATION = ("NFKC", "casefold", "unicode_alphanumeric_tokens")
CLASSIFICATION_PRECEDENCE = (
    "evidence_request",
    "change_over_time",
    "historical_state",
    "current_state",
    "specific_event",
    "relationship",
    "commitment",
    "unknown",
)
CLASSIFICATION_PHRASES = {
    "evidence_request": (
        "what evidence",
        "show evidence",
        "source for",
        "proof of",
        "how do you know",
        "where did this come from",
    ),
    "change_over_time": (
        "change over time",
        "changed over time",
        "how has",
        "before and after",
        "used to but now",
        "correction",
        "corrected",
    ),
    "historical_state": (
        "historical",
        "historically",
        "in the past",
        "previously",
        "back then",
        "used to",
    ),
    "current_state": ("current", "currently", "right now", "now", "today"),
    "specific_event": ("what happened", "when did", "specific event", "which event"),
    "relationship": ("relationship", "related to", "connected to", "who is"),
    "commitment": ("commitment", "committed to", "promised", "plan to", "deadline"),
    "unknown": (),
}
CURRENT_TIME_PHRASES = ("right now", "now", "current", "currently", "today")
UNRESOLVED_TIME_PHRASES = (
    "yesterday",
    "tomorrow",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "recently",
    "on monday",
    "on tuesday",
    "on wednesday",
    "on thursday",
    "on friday",
    "on saturday",
    "on sunday",
)
UNRESOLVED_TIME_PATTERNS = ("iso_like_date", "slash_date", "clock_time")
CURRENT_LIFECYCLE_STATUSES = ("candidate", "confirmed", "current", "disputed")
FAILURE_CODES = ("invalid_request", "invalid_config", "planning_failed")
CONFIG_FIELDS = {
    "planner_version",
    "index_version",
    "normalization",
    "classification_precedence",
    "classification_phrases",
    "current_time_phrases",
    "unresolved_time_phrases",
    "unresolved_time_patterns",
    "record_kinds",
    "nonexcluded_lifecycle_statuses",
    "current_lifecycle_statuses",
    "sensitivity_scopes",
    "relation_expansion_types",
    "failure_codes",
}
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class QueryPlannerConfig:
    planner_version: str
    index_version: str
    normalization: tuple[str, ...]
    classification_precedence: tuple[str, ...]
    classification_phrases: Mapping[str, tuple[str, ...]]
    current_time_phrases: tuple[str, ...]
    unresolved_time_phrases: tuple[str, ...]
    unresolved_time_patterns: tuple[str, ...]
    record_kinds: tuple[str, ...]
    nonexcluded_lifecycle_statuses: tuple[str, ...]
    current_lifecycle_statuses: tuple[str, ...]
    sensitivity_scopes: tuple[str, ...]
    relation_expansion_types: tuple[str, ...]
    failure_codes: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        actual = (
            self.planner_version,
            self.index_version,
            self.normalization,
            self.classification_precedence,
            dict(self.classification_phrases),
            self.current_time_phrases,
            self.unresolved_time_phrases,
            self.unresolved_time_patterns,
            self.record_kinds,
            self.nonexcluded_lifecycle_statuses,
            self.current_lifecycle_statuses,
            self.sensitivity_scopes,
            self.relation_expansion_types,
            self.failure_codes,
        )
        expected = (
            PLANNER_VERSION,
            INDEX_VERSION,
            NORMALIZATION,
            CLASSIFICATION_PRECEDENCE,
            CLASSIFICATION_PHRASES,
            CURRENT_TIME_PHRASES,
            UNRESOLVED_TIME_PHRASES,
            UNRESOLVED_TIME_PATTERNS,
            RECORD_KINDS,
            LIFECYCLE_STATUSES,
            CURRENT_LIFECYCLE_STATUSES,
            SENSITIVITY_SCOPES,
            RELATION_EXPANSION_TYPES,
            FAILURE_CODES,
        )
        if actual != expected or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise RetrievalQueryError("query planner configuration changed")


def load_query_planner_config(path: str | Path = CONFIG_PATH) -> QueryPlannerConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    try:
        value = json.loads(
            raw,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RetrievalQueryError(f"invalid JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalQueryError("query planner configuration is invalid") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise RetrievalQueryError("query planner configuration fields changed")
    phrases = value["classification_phrases"]
    if not isinstance(phrases, dict) or set(phrases) != set(QUERY_LABELS):
        raise RetrievalQueryError("classification phrase fields changed")
    return QueryPlannerConfig(
        planner_version=_string(value, "planner_version"),
        index_version=_string(value, "index_version"),
        normalization=_strings(value, "normalization"),
        classification_precedence=_strings(value, "classification_precedence"),
        classification_phrases={
            label: _string_list(phrases[label], f"phrases for {label}")
            for label in QUERY_LABELS
        },
        current_time_phrases=_strings(value, "current_time_phrases"),
        unresolved_time_phrases=_strings(value, "unresolved_time_phrases"),
        unresolved_time_patterns=_strings(value, "unresolved_time_patterns"),
        record_kinds=_strings(value, "record_kinds"),
        nonexcluded_lifecycle_statuses=_strings(value, "nonexcluded_lifecycle_statuses"),
        current_lifecycle_statuses=_strings(value, "current_lifecycle_statuses"),
        sensitivity_scopes=_strings(value, "sensitivity_scopes"),
        relation_expansion_types=_strings(value, "relation_expansion_types"),
        failure_codes=_strings(value, "failure_codes"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def normalized_query_tokens(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise RetrievalQueryError("query text is invalid")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(TOKEN.findall(normalized))


def classify_query(
    query_text: str,
    *,
    config: QueryPlannerConfig,
) -> str:
    tokens = normalized_query_tokens(query_text)
    for label in config.classification_precedence:
        if label == "unknown":
            return label
        if _matches_any(tokens, config.classification_phrases[label]):
            return label
    return "unknown"


def build_query_plan(
    request: RetrievalQueryRequest,
    *,
    config: QueryPlannerConfig,
) -> QueryPlan:
    if request.index_version != config.index_version:
        raise RetrievalQueryError("request and planner index versions differ")
    tokens = normalized_query_tokens(request.query_text)
    primary_label = classify_query(request.query_text, config=config)
    current_signal = _matches_any(tokens, config.current_time_phrases)
    change_signal = _matches_any(tokens, config.classification_phrases["change_over_time"])
    unresolved_signal = _matches_any(tokens, config.unresolved_time_phrases) or _matches_time_pattern(
        request.query_text, config.unresolved_time_patterns
    )
    requested_valid_time = request.requested_valid_time
    if requested_valid_time is None and current_signal:
        requested_valid_time = RequestedValidTime(
            kind="point",
            point_timestamp=request.as_of,
        )
    unresolved_time = request.requested_valid_time is None and unresolved_signal
    include_previous = primary_label in {"historical_state", "change_over_time"}
    relation_intent = change_signal
    allowed_lifecycle = (
        config.current_lifecycle_statuses
        if primary_label == "current_state"
        else config.nonexcluded_lifecycle_statuses
    )
    plan_value = {
        "request": request_canonical_value(request),
        "planner_version": config.planner_version,
        "planner_config_sha256": config.sha256,
        "primary_label": primary_label,
        "resolved_requested_valid_time": _valid_time_value(requested_valid_time),
        "allowed_lifecycle_statuses": allowed_lifecycle,
        "include_previous_versions": include_previous,
        "checked_relation_expansion_intent": relation_intent,
        "source_evidence_intent": primary_label == "evidence_request",
        "unresolved_time": unresolved_time,
    }
    return QueryPlan(
        plan_id=stable_sha256(plan_value),
        query_id=request.query_id,
        user_id=request.user_id,
        planner_version=config.planner_version,
        planner_config_sha256=config.sha256,
        index_version=request.index_version,
        primary_label=primary_label,
        as_of=request.as_of,
        enabled_record_kinds=request.enabled_record_kinds,
        requested_valid_time=requested_valid_time,
        speaker_ids=request.speaker_ids,
        entity_ids=request.entity_ids,
        allowed_lifecycle_statuses=allowed_lifecycle,
        allow_sensitive=request.sensitivity_scope == "sensitive",
        allow_unclassified_sensitivity=request.allow_unclassified_sensitivity,
        include_previous_versions=include_previous,
        checked_relation_expansion_intent=relation_intent,
        relation_expansion_types=(
            config.relation_expansion_types if relation_intent else ()
        ),
        source_evidence_intent=primary_label == "evidence_request",
        unresolved_time=unresolved_time,
        clarification_required=unresolved_time,
    )


class RetrievalQueryPlanner:
    """Build deterministic plans from already validated requests."""

    def __init__(self, config: QueryPlannerConfig) -> None:
        self._config = config

    def plan(self, request: RetrievalQueryRequest) -> QueryPlan:
        return build_query_plan(request, config=self._config)


def _matches_any(tokens: tuple[str, ...], phrases: tuple[str, ...]) -> bool:
    return any(_contains(tokens, normalized_query_tokens(phrase)) for phrase in phrases)


def _contains(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    return any(tokens[index : index + len(phrase)] == phrase for index in range(len(tokens) - len(phrase) + 1))


def _matches_time_pattern(value: str, patterns: tuple[str, ...]) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    expressions = {
        "iso_like_date": r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)",
        "slash_date": r"(?<!\d)\d{1,4}/\d{1,2}/\d{1,4}(?!\d)",
        "clock_time": r"(?<!\d)\d{1,2}:\d{2}(?!\d)",
    }
    return any(re.search(expressions[name], normalized) for name in patterns)


def _valid_time_value(value: RequestedValidTime | None) -> object:
    if value is None:
        return None
    return {
        name: getattr(value, name)
        for name in RequestedValidTime.__dataclass_fields__
    }


def _string(value: Mapping[str, object], name: str) -> str:
    item = value[name]
    if not isinstance(item, str):
        raise RetrievalQueryError(f"{name} is invalid")
    return item


def _strings(value: Mapping[str, object], name: str) -> tuple[str, ...]:
    return _string_list(value[name], name)


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RetrievalQueryError(f"{name} is invalid")
    return tuple(value)
