"""Typed contracts for deterministic atomic and session index records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, TypeAlias


INDEX_VERSION = "retrieval_index_v1"
CONTENT_RENDERER_VERSION = "retrieval_content_v1"
EMBEDDING_VERSION = "deterministic_token_hash_v1"
EMBEDDING_DIMENSION = 256
ELIGIBLE_LIFECYCLE_STATUSES = (
    "candidate",
    "confirmed",
    "current",
    "historical",
    "disputed",
    "superseded",
)
LIFECYCLE_STATUSES = frozenset((*ELIGIBLE_LIFECYCLE_STATUSES, "excluded"))
RECORD_KINDS = frozenset({"atomic", "session"})
SENSITIVITIES = frozenset({"standard", "sensitive", "restricted"})
TIME_PRECISIONS = frozenset(
    {"timestamp", "day", "month", "year", "approximate", "unknown", "mixed"}
)
EPISTEMIC_STATUSES = frozenset(
    {
        "asserted",
        "inferred",
        "reported_by_other",
        "hypothetical",
        "uncertain",
        "denied",
        "corrected",
    }
)
POLARITIES = frozenset({"positive", "negative"})
MEMORY_KINDS = frozenset({"episodic", "durative"})
SUPPORT_TYPES = frozenset({"supports", "contradicts", "corrects"})
RELATION_TYPES = frozenset(
    {
        "supports",
        "contradicts",
        "corrects",
        "supersedes",
        "refines",
        "same_event_as",
        "caused_by",
        "hindered_by",
        "same_topic_as",
    }
)
RELATION_DIRECTIONS = frozenset({"incoming", "outgoing", "symmetric"})
STATEMENT_KINDS = frozenset({"observed_fact", "unresolved_question"})
LIFECYCLE_VIEWS = frozenset({"accepted", "candidate", "historical", "disputed"})

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

CONFIG_FIELDS = frozenset(
    {
        "index_version",
        "content_renderer_version",
        "embedding_version",
        "embedding_dimension",
        "distance",
        "fts_configuration",
        "token_normalization",
        "bucket_rule",
        "sign_rule",
        "empty_embedding",
        "eligible_record_kinds",
        "eligible_lifecycle_statuses",
        "excluded_lifecycle_statuses",
        "excluded_sensitivities",
        "atomic_content_fields",
        "session_content_fields",
    }
)


class RetrievalIndexError(ValueError):
    """Reject an unsafe or inconsistent retrieval-index value."""


@dataclass(frozen=True)
class IndexConfig:
    index_version: str
    content_renderer_version: str
    embedding_version: str
    embedding_dimension: int
    distance: str
    fts_configuration: str
    token_normalization: tuple[str, ...]
    bucket_rule: str
    sign_rule: str
    empty_embedding: str
    eligible_record_kinds: tuple[str, ...]
    eligible_lifecycle_statuses: tuple[str, ...]
    excluded_lifecycle_statuses: tuple[str, ...]
    excluded_sensitivities: tuple[str, ...]
    atomic_content_fields: tuple[str, ...]
    session_content_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (
            INDEX_VERSION,
            CONTENT_RENDERER_VERSION,
            EMBEDDING_VERSION,
            EMBEDDING_DIMENSION,
            "cosine",
            "simple",
            ("NFKC", "casefold", "unicode_alphanumeric_runs"),
            "sha256_bytes_0_7_big_endian_mod_dimension",
            "sha256_byte_8_low_bit_zero_positive",
            "zero_vector",
            ("atomic", "session"),
            ELIGIBLE_LIFECYCLE_STATUSES,
            ("excluded",),
            ("restricted",),
            (
                "subject_id",
                "speaker_id",
                "predicate",
                "object_json",
                "polarity",
                "epistemic_status",
                "memory_kind",
                "lifecycle_status",
                "valid_time",
                "transaction_time",
                "checked_relations",
            ),
            (
                "summary_text",
                "ordered_grounded_statements",
                "unresolved_questions",
            ),
        )
        actual = (
            self.index_version,
            self.content_renderer_version,
            self.embedding_version,
            self.embedding_dimension,
            self.distance,
            self.fts_configuration,
            self.token_normalization,
            self.bucket_rule,
            self.sign_rule,
            self.empty_embedding,
            self.eligible_record_kinds,
            self.eligible_lifecycle_statuses,
            self.excluded_lifecycle_statuses,
            self.excluded_sensitivities,
            self.atomic_content_fields,
            self.session_content_fields,
        )
        if actual != expected:
            raise RetrievalIndexError("retrieval index configuration changed")


@dataclass(frozen=True)
class ValidTime:
    time_precision: str
    valid_from_date: date | None = None
    valid_from_timestamp: datetime | None = None
    valid_to_date: date | None = None
    valid_to_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.time_precision not in TIME_PRECISIONS:
            raise RetrievalIndexError("time_precision is invalid")
        dates = (self.valid_from_date, self.valid_to_date)
        timestamps = (self.valid_from_timestamp, self.valid_to_timestamp)
        if any(value is not None and type(value) is not date for value in dates):
            raise RetrievalIndexError("valid date boundary is invalid")
        for value in timestamps:
            if value is not None:
                _aware(value, "valid timestamp boundary")
        if any(value is not None for value in dates) and any(
            value is not None for value in timestamps
        ):
            raise RetrievalIndexError("valid-time representations cannot mix")
        if self.time_precision in {"unknown", "mixed"}:
            if any(value is not None for value in (*dates, *timestamps)):
                raise RetrievalIndexError("unknown or mixed valid time must be empty")
        elif self.time_precision == "timestamp":
            if not any(value is not None for value in timestamps):
                raise RetrievalIndexError(
                    "timestamp precision requires a timestamp boundary"
                )
        elif not any(value is not None for value in dates):
            raise RetrievalIndexError("date precision requires a date boundary")
        if dates[0] is not None and dates[1] is not None and dates[1] < dates[0]:
            raise RetrievalIndexError("valid date interval is reversed")
        if (
            timestamps[0] is not None
            and timestamps[1] is not None
            and timestamps[1] < timestamps[0]
        ):
            raise RetrievalIndexError("valid timestamp interval is reversed")


@dataclass(frozen=True)
class TransactionTime:
    transaction_from: datetime
    transaction_to: datetime | None = None

    def __post_init__(self) -> None:
        _aware(self.transaction_from, "transaction_from")
        if self.transaction_to is not None:
            _aware(self.transaction_to, "transaction_to")
            if self.transaction_to <= self.transaction_from:
                raise RetrievalIndexError("transaction interval must be [from, to)")

    def contains(self, value: datetime) -> bool:
        _aware(value, "transaction cutoff")
        return self.transaction_from <= value and (
            self.transaction_to is None or value < self.transaction_to
        )


@dataclass(frozen=True)
class ClaimVersionLineage:
    user_id: str
    claim_id: str
    claim_version_id: str
    lifecycle_status: str
    order: int

    def __post_init__(self) -> None:
        for name in ("user_id", "claim_id", "claim_version_id"):
            _text(getattr(self, name), name)
        _enum(self.lifecycle_status, LIFECYCLE_STATUSES, "lifecycle_status")
        _order(self.order, "claim lineage order")


@dataclass(frozen=True)
class SourceSpanLineage:
    user_id: str
    claim_id: str
    claim_version_id: str
    source_id: str
    span_id: str
    support_type: str
    order: int

    def __post_init__(self) -> None:
        for name in (
            "user_id",
            "claim_id",
            "claim_version_id",
            "source_id",
            "span_id",
        ):
            _text(getattr(self, name), name)
        _enum(self.support_type, SUPPORT_TYPES, "support_type")
        _order(self.order, "source lineage order")


@dataclass(frozen=True)
class RelationLineage:
    relation_id: str
    user_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    direction: str
    order: int

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "user_id",
            "source_claim_id",
            "target_claim_id",
        ):
            _text(getattr(self, name), name)
        if self.source_claim_id == self.target_claim_id:
            raise RetrievalIndexError("relation claims must differ")
        _enum(self.relation_type, RELATION_TYPES, "relation_type")
        _enum(self.direction, RELATION_DIRECTIONS, "relation direction")
        _order(self.order, "relation lineage order")


@dataclass(frozen=True)
class AtomicIndexInput:
    user_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: JSONValue
    polarity: str
    epistemic_status: str
    memory_kind: str | None
    lifecycle_status: str
    valid_time: ValidTime
    transaction_time: TransactionTime
    sensitivity: str | None
    claim_lineage: tuple[ClaimVersionLineage, ...]
    source_lineage: tuple[SourceSpanLineage, ...]
    relation_lineage: tuple[RelationLineage, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "user_id",
            "claim_id",
            "claim_version_id",
            "subject_id",
            "speaker_id",
            "predicate",
        ):
            _text(getattr(self, name), name)
        object.__setattr__(self, "object_json", safe_json(self.object_json))
        _enum(self.polarity, POLARITIES, "polarity")
        _enum(self.epistemic_status, EPISTEMIC_STATUSES, "epistemic_status")
        if self.memory_kind is not None:
            _enum(self.memory_kind, MEMORY_KINDS, "memory_kind")
        _enum(self.lifecycle_status, LIFECYCLE_STATUSES, "lifecycle_status")
        if self.sensitivity is not None:
            _enum(self.sensitivity, SENSITIVITIES, "sensitivity")
        if self.lifecycle_status == "current" and self.valid_time.time_precision == "unknown":
            raise RetrievalIndexError("current atomic input requires known valid time")
        if self.valid_time.time_precision == "mixed":
            raise RetrievalIndexError("atomic valid time cannot be mixed")
        expected_claim = ClaimVersionLineage(
            self.user_id,
            self.claim_id,
            self.claim_version_id,
            self.lifecycle_status,
            0,
        )
        if self.claim_lineage != (expected_claim,):
            raise RetrievalIndexError("atomic claim lineage changed")
        if not self.source_lineage:
            raise RetrievalIndexError("atomic input requires exact source lineage")
        _ordered_unique(self.source_lineage, "source lineage")
        _ordered_unique(self.relation_lineage, "relation lineage")
        for item in self.source_lineage:
            if (
                item.user_id != self.user_id
                or item.claim_id != self.claim_id
                or item.claim_version_id != self.claim_version_id
            ):
                raise RetrievalIndexError("atomic source lineage crosses its anchor")
        for item in self.relation_lineage:
            if item.user_id != self.user_id or self.claim_id not in {
                item.source_claim_id,
                item.target_claim_id,
            }:
                raise RetrievalIndexError("atomic relation lineage crosses its anchor")


@dataclass(frozen=True)
class SessionStatementInput:
    statement_id: str
    statement_kind: str
    lifecycle_view: str
    text: str
    claim_lineage: tuple[ClaimVersionLineage, ...]
    source_lineage: tuple[SourceSpanLineage, ...]
    order: int

    def __post_init__(self) -> None:
        for name in ("statement_id", "text"):
            _text(getattr(self, name), name)
        _enum(self.statement_kind, STATEMENT_KINDS, "statement_kind")
        _enum(self.lifecycle_view, LIFECYCLE_VIEWS, "lifecycle_view")
        _order(self.order, "statement order")
        if not self.claim_lineage or not self.source_lineage:
            raise RetrievalIndexError("session statement requires claim and source lineage")
        _ordered_unique(self.claim_lineage, "statement claim lineage")
        _ordered_unique(self.source_lineage, "statement source lineage")
        claims = {(item.user_id, item.claim_id, item.claim_version_id) for item in self.claim_lineage}
        sources = {(item.user_id, item.claim_id, item.claim_version_id) for item in self.source_lineage}
        if claims != sources:
            raise RetrievalIndexError("statement claim and source lineage differ")


@dataclass(frozen=True)
class SessionIndexInput:
    user_id: str
    session_summary_id: str
    session_definition_id: str
    renderer_version: str
    summary_text: str
    statements: tuple[SessionStatementInput, ...]
    session_source_ids: tuple[str, ...]
    valid_time: ValidTime
    transaction_time: TransactionTime
    sensitivity: str | None
    contains_sensitive: bool

    def __post_init__(self) -> None:
        for name in (
            "user_id",
            "session_summary_id",
            "session_definition_id",
            "renderer_version",
            "summary_text",
        ):
            _text(getattr(self, name), name)
        if not self.statements:
            raise RetrievalIndexError("session input requires grounded statements")
        _ordered_unique(self.statements, "session statements")
        if tuple(sorted(set(self.session_source_ids))) != self.session_source_ids:
            raise RetrievalIndexError("session source IDs must be sorted and unique")
        if not self.session_source_ids:
            raise RetrievalIndexError("session source IDs are required")
        if self.sensitivity is not None:
            _enum(self.sensitivity, SENSITIVITIES, "sensitivity")
        if not isinstance(self.contains_sensitive, bool):
            raise RetrievalIndexError("contains_sensitive must be a boolean")
        if self.contains_sensitive != (self.sensitivity == "sensitive"):
            raise RetrievalIndexError("session sensitivity flag changed")
        lineage_sources: set[str] = set()
        for statement in self.statements:
            for claim in statement.claim_lineage:
                if claim.user_id != self.user_id:
                    raise RetrievalIndexError("session claim lineage crosses users")
            for source in statement.source_lineage:
                if source.user_id != self.user_id:
                    raise RetrievalIndexError("session source lineage crosses users")
                lineage_sources.add(source.source_id)
        if not lineage_sources.issubset(set(self.session_source_ids)):
            raise RetrievalIndexError("statement source is outside the session")

    @property
    def claim_lineage(self) -> tuple[ClaimVersionLineage, ...]:
        by_key: dict[tuple[str, str], ClaimVersionLineage] = {}
        for statement in self.statements:
            for item in statement.claim_lineage:
                key = (item.claim_id, item.claim_version_id)
                existing = by_key.get(key)
                if existing is not None and existing.lifecycle_status != item.lifecycle_status:
                    raise RetrievalIndexError("session lifecycle lineage changed")
                by_key[key] = item
        return tuple(
            ClaimVersionLineage(
                item.user_id,
                item.claim_id,
                item.claim_version_id,
                item.lifecycle_status,
                order,
            )
            for order, item in enumerate(
                sorted(by_key.values(), key=lambda value: (value.claim_id, value.claim_version_id))
            )
        )

    @property
    def source_lineage(self) -> tuple[SourceSpanLineage, ...]:
        by_key: dict[tuple[str, str, str, str], SourceSpanLineage] = {}
        for statement in self.statements:
            for item in statement.source_lineage:
                key = (item.claim_id, item.claim_version_id, item.source_id, item.span_id)
                existing = by_key.get(key)
                if existing is not None and existing.support_type != item.support_type:
                    raise RetrievalIndexError("session source support changed")
                by_key[key] = item
        return tuple(
            SourceSpanLineage(
                item.user_id,
                item.claim_id,
                item.claim_version_id,
                item.source_id,
                item.span_id,
                item.support_type,
                order,
            )
            for order, item in enumerate(
                sorted(
                    by_key.values(),
                    key=lambda value: (
                        value.claim_id,
                        value.claim_version_id,
                        value.source_id,
                        value.span_id,
                        value.support_type,
                    ),
                )
            )
        )


@dataclass(frozen=True)
class IndexRecord:
    index_record_id: str
    user_id: str
    index_version: str
    record_kind: str
    claim_version_id: str | None
    session_summary_id: str | None
    subject_id: str | None
    speaker_id: str | None
    predicate: str | None
    content_text: str
    content_sha256: str
    embedding_version: str
    embedding: tuple[float, ...]
    lifecycle_statuses: tuple[str, ...]
    memory_kind: str | None
    epistemic_status: str | None
    valid_time: ValidTime
    transaction_time: TransactionTime
    sensitivity: str | None
    contains_sensitive: bool
    input_snapshot_sha256: str
    claim_lineage: tuple[ClaimVersionLineage, ...]
    source_lineage: tuple[SourceSpanLineage, ...]
    relation_lineage: tuple[RelationLineage, ...]

    def __post_init__(self) -> None:
        _sha256(self.index_record_id, "index_record_id")
        _text(self.user_id, "user_id")
        if self.index_version != INDEX_VERSION:
            raise RetrievalIndexError("index_version is unsupported")
        _enum(self.record_kind, RECORD_KINDS, "record_kind")
        anchors = (self.claim_version_id is not None, self.session_summary_id is not None)
        if anchors != ((self.record_kind == "atomic"), (self.record_kind == "session")):
            raise RetrievalIndexError("index record anchor shape is invalid")
        _text(self.claim_version_id, "claim_version_id", nullable=True)
        _text(self.session_summary_id, "session_summary_id", nullable=True)
        _text(self.subject_id, "subject_id", nullable=True)
        _text(self.speaker_id, "speaker_id", nullable=True)
        _text(self.predicate, "predicate", nullable=True)
        _text(self.content_text, "content_text")
        _sha256(self.content_sha256, "content_sha256")
        if self.content_sha256 != hashlib.sha256(self.content_text.encode("utf-8")).hexdigest():
            raise RetrievalIndexError("content hash does not match content")
        if self.embedding_version != EMBEDDING_VERSION:
            raise RetrievalIndexError("embedding_version is unsupported")
        if len(self.embedding) != EMBEDDING_DIMENSION or any(
            not isinstance(value, float) or not math.isfinite(value)
            for value in self.embedding
        ):
            raise RetrievalIndexError("embedding must contain 256 finite floats")
        norm = math.sqrt(sum(value * value for value in self.embedding))
        if norm != 0.0 and not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RetrievalIndexError("embedding must be L2 normalized")
        if not self.lifecycle_statuses or tuple(sorted(set(self.lifecycle_statuses))) != self.lifecycle_statuses:
            raise RetrievalIndexError("lifecycle statuses must be sorted and unique")
        if any(value not in LIFECYCLE_STATUSES for value in self.lifecycle_statuses):
            raise RetrievalIndexError("lifecycle status is invalid")
        if "excluded" in self.lifecycle_statuses:
            raise RetrievalIndexError("excluded lifecycle cannot be indexed")
        if self.memory_kind is not None:
            _enum(self.memory_kind, MEMORY_KINDS, "memory_kind")
        if self.epistemic_status is not None:
            _enum(self.epistemic_status, EPISTEMIC_STATUSES, "epistemic_status")
        if self.sensitivity is not None:
            _enum(self.sensitivity, SENSITIVITIES, "sensitivity")
        if self.sensitivity == "restricted":
            raise RetrievalIndexError("restricted content cannot be indexed")
        if not isinstance(self.contains_sensitive, bool):
            raise RetrievalIndexError("contains_sensitive must be a boolean")
        if self.contains_sensitive != (self.sensitivity == "sensitive"):
            raise RetrievalIndexError("record sensitivity flag changed")
        _sha256(self.input_snapshot_sha256, "input_snapshot_sha256")
        _ordered_unique(self.claim_lineage, "record claim lineage")
        _ordered_unique(self.source_lineage, "record source lineage")
        _ordered_unique(self.relation_lineage, "record relation lineage")
        if any(item.user_id != self.user_id for item in self.claim_lineage):
            raise RetrievalIndexError("record claim lineage crosses users")
        if any(item.user_id != self.user_id for item in self.source_lineage):
            raise RetrievalIndexError("record source lineage crosses users")
        if any(item.user_id != self.user_id for item in self.relation_lineage):
            raise RetrievalIndexError("record relation lineage crosses users")
        if self.record_kind == "atomic":
            if (
                len(self.claim_lineage) != 1
                or self.claim_lineage[0].claim_version_id != self.claim_version_id
                or self.subject_id is None
                or self.speaker_id is None
                or self.predicate is None
                or self.epistemic_status is None
                or not self.source_lineage
            ):
                raise RetrievalIndexError("atomic record lineage changed")
            if (
                self.lifecycle_statuses == ("current",)
                and self.valid_time.time_precision == "unknown"
            ):
                raise RetrievalIndexError("current atomic record requires known valid time")
            if self.valid_time.time_precision == "mixed":
                raise RetrievalIndexError("atomic valid time cannot be mixed")
        elif any(
            value is not None
            for value in (
                self.subject_id,
                self.speaker_id,
                self.predicate,
                self.memory_kind,
                self.epistemic_status,
            )
        ):
            raise RetrievalIndexError("session record cannot copy atomic-only fields")


def load_index_config(path: str | Path) -> IndexConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalIndexError("retrieval index configuration is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise RetrievalIndexError("retrieval index configuration fields changed")
    for name in (
        "token_normalization",
        "eligible_record_kinds",
        "eligible_lifecycle_statuses",
        "excluded_lifecycle_statuses",
        "excluded_sensitivities",
        "atomic_content_fields",
        "session_content_fields",
    ):
        if not isinstance(value[name], list):
            raise RetrievalIndexError("retrieval index configuration list changed")
        value[name] = tuple(value[name])
    return IndexConfig(**value)


def safe_json(value: object) -> JSONValue:
    """Return a detached canonical JSON value without Python extensions."""

    def validate(item: object) -> None:
        if item is None or isinstance(item, bool | str | int):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise RetrievalIndexError("JSON keys must be strings")
                validate(child)
            return
        raise RetrievalIndexError("value is not JSON-safe")

    validate(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_value(value: object) -> object:
    if isinstance(value, datetime):
        _aware(value, "canonical timestamp")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RetrievalIndexError("canonical JSON keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise RetrievalIndexError("value is not canonical JSON")


def _ordered_unique(values: tuple[object, ...], name: str) -> None:
    orders = tuple(getattr(value, "order", None) for value in values)
    if orders != tuple(range(len(values))):
        raise RetrievalIndexError(f"{name} must use consecutive order")
    if len(values) != len(set(values)):
        raise RetrievalIndexError(f"{name} must be unique")


def _text(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RetrievalIndexError(f"{name} must be nonempty text")
    return value


def _enum(value: str, allowed: frozenset[str], name: str) -> str:
    _text(value, name)
    if value not in allowed:
        raise RetrievalIndexError(f"{name} is invalid")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise RetrievalIndexError(f"{name} must be timezone-aware")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RetrievalIndexError(f"{name} must be a lowercase SHA-256")
    return value


def _order(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalIndexError(f"{name} must be a nonnegative integer")
    return value
