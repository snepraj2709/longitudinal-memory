"""Typed contracts for deterministic source session boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from storage.contracts import SOURCE_TYPES, StorageValidationError, safe_json


BOUNDARY_VERSION = "session_boundaries_v1"
CONFIG_FIELDS = frozenset(
    {
        "boundary_version",
        "calendar_boundary_kind",
        "chat_declared_boundary_kind",
        "chat_declared_key_precedence",
        "chat_unthreaded_boundary_kind",
        "chat_unthreaded_gap_seconds",
        "conversation_boundary_kind",
        "email_boundary_kind",
        "email_declared_key_precedence",
        "email_thread_conflict",
        "review_status",
        "session_order",
        "source_order",
        "visibility_rule",
    }
)
BOUNDARY_KINDS = frozenset(
    {"source", "declared_thread", "declared_thread_or_source", "inactivity"}
)


class SessionBoundaryError(ValueError):
    """Reject an invalid sessionization contract without source content."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionBoundaryError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise SessionBoundaryError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class SessionBoundaryConfig:
    boundary_version: str
    calendar_boundary_kind: str
    chat_declared_boundary_kind: str
    chat_declared_key_precedence: tuple[str, ...]
    chat_unthreaded_boundary_kind: str
    chat_unthreaded_gap_seconds: int
    conversation_boundary_kind: str
    email_boundary_kind: str
    email_declared_key_precedence: tuple[str, ...]
    email_thread_conflict: str
    review_status: str
    session_order: tuple[str, ...]
    source_order: tuple[str, ...]
    visibility_rule: str

    def __post_init__(self) -> None:
        expected = (
            BOUNDARY_VERSION,
            "source",
            "declared_thread",
            ("session_id", "metadata.thread_id"),
            "inactivity",
            1800,
            "source",
            "declared_thread_or_source",
            ("session_id", "metadata.thread_id"),
            "reject",
            "implementation_reviewed",
            ("start_at", "source_type", "first_source_id", "definition_id"),
            ("produced_at", "source_id"),
            "ingested_at_lte_transaction_as_of",
        )
        actual = (
            self.boundary_version,
            self.calendar_boundary_kind,
            self.chat_declared_boundary_kind,
            self.chat_declared_key_precedence,
            self.chat_unthreaded_boundary_kind,
            self.chat_unthreaded_gap_seconds,
            self.conversation_boundary_kind,
            self.email_boundary_kind,
            self.email_declared_key_precedence,
            self.email_thread_conflict,
            self.review_status,
            self.session_order,
            self.source_order,
            self.visibility_rule,
        )
        if actual != expected:
            raise SessionBoundaryError("session boundary configuration changed")


@dataclass(frozen=True)
class SessionizationRequest:
    user_id: str
    transaction_as_of: datetime
    boundary_version: str = BOUNDARY_VERSION

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _aware(self.transaction_as_of, "transaction_as_of")
        if self.boundary_version != BOUNDARY_VERSION:
            raise SessionBoundaryError("boundary_version is unsupported")


@dataclass(frozen=True)
class SessionSource:
    source_id: str
    user_id: str
    source_type: str
    session_id: str | None
    produced_at: datetime
    ingested_at: datetime
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _text(self.source_id, "source_id")
        _text(self.user_id, "user_id")
        if self.source_type not in SOURCE_TYPES:
            raise SessionBoundaryError("source_type is invalid")
        if self.session_id is not None:
            _text(self.session_id, "session_id")
        _aware(self.produced_at, "produced_at")
        _aware(self.ingested_at, "ingested_at")
        try:
            metadata = safe_json(self.metadata, "metadata", top_type=dict)
        except StorageValidationError as error:
            raise SessionBoundaryError("metadata is not JSON-safe") from error
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class SessionDefinition:
    definition_id: str
    user_id: str
    source_type: str
    boundary_kind: str
    boundary_version: str
    source_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    transaction_as_of: datetime
    membership_sha256: str

    def __post_init__(self) -> None:
        if not _sha256(self.definition_id):
            raise SessionBoundaryError("definition_id must be a lowercase SHA-256")
        _text(self.user_id, "user_id")
        if self.source_type not in SOURCE_TYPES:
            raise SessionBoundaryError("source_type is invalid")
        if self.boundary_kind not in BOUNDARY_KINDS:
            raise SessionBoundaryError("boundary_kind is invalid")
        if self.boundary_version != BOUNDARY_VERSION:
            raise SessionBoundaryError("boundary_version is unsupported")
        if (
            not isinstance(self.source_ids, tuple)
            or not self.source_ids
            or any(not isinstance(item, str) or not item.strip() for item in self.source_ids)
            or len(self.source_ids) != len(set(self.source_ids))
        ):
            raise SessionBoundaryError("source_ids must be unique nonempty IDs")
        _aware(self.start_at, "start_at")
        _aware(self.end_at, "end_at")
        _aware(self.transaction_as_of, "transaction_as_of")
        if self.end_at < self.start_at:
            raise SessionBoundaryError("session boundaries must be ordered")
        if not _sha256(self.membership_sha256):
            raise SessionBoundaryError("membership_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class SessionizationResult:
    user_id: str
    transaction_as_of: datetime
    boundary_version: str
    definitions: tuple[SessionDefinition, ...]

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _aware(self.transaction_as_of, "transaction_as_of")
        if self.boundary_version != BOUNDARY_VERSION:
            raise SessionBoundaryError("boundary_version is unsupported")
        if not isinstance(self.definitions, tuple):
            raise SessionBoundaryError("definitions must be a tuple")
        for definition in self.definitions:
            if (
                definition.user_id != self.user_id
                or definition.transaction_as_of != self.transaction_as_of
                or definition.boundary_version != self.boundary_version
            ):
                raise SessionBoundaryError("definition leaves the request boundary")
        expected = tuple(
            sorted(
                self.definitions,
                key=lambda item: (
                    item.start_at,
                    item.source_type,
                    item.source_ids[0],
                    item.definition_id,
                ),
            )
        )
        if self.definitions != expected:
            raise SessionBoundaryError("definitions must use canonical order")
        ids = tuple(item.definition_id for item in self.definitions)
        if len(ids) != len(set(ids)):
            raise SessionBoundaryError("definition IDs must be unique")


def load_session_boundary_config(path: Path) -> SessionBoundaryConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionBoundaryError("session boundary configuration is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != CONFIG_FIELDS:
        raise SessionBoundaryError("session boundary configuration fields changed")
    return SessionBoundaryConfig(
        **{
            **payload,
            "chat_declared_key_precedence": tuple(payload["chat_declared_key_precedence"]),
            "email_declared_key_precedence": tuple(payload["email_declared_key_precedence"]),
            "session_order": tuple(payload["session_order"]),
            "source_order": tuple(payload["source_order"]),
        }
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
