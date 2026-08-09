"""Load only the frozen scaled-development sources used by Step 3.5."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Mapping

from evaluation.history import HistoryDataError, HistoryObservation

from .source import ExtractionSource, KnownEntity


DEVELOPMENT_USER_IDS = ("user_001", "user_002")
DEVELOPMENT_SOURCE_SUFFIXES = (
    "conversation_001",
    "email_001",
    "chat_001",
    "calendar_001",
    "conversation_002",
    "email_002",
    "chat_002",
    "calendar_002",
    "conversation_003",
    "conversation_004",
)
QUALIFICATION_SOURCE_SUFFIXES = (
    "conversation_001",
    "email_002",
    "chat_002",
    "calendar_001",
)
DEVELOPMENT_SOURCE_REFS = tuple(
    (user_id, f"scaled_{user_id}_{suffix}")
    for user_id in DEVELOPMENT_USER_IDS
    for suffix in DEVELOPMENT_SOURCE_SUFFIXES
)
QUALIFICATION_SOURCE_REFS = tuple(
    (DEVELOPMENT_USER_IDS[0], f"scaled_{DEVELOPMENT_USER_IDS[0]}_{suffix}")
    for suffix in QUALIFICATION_SOURCE_SUFFIXES
)

_USER_PATH = Path("data/scaled-v1/runtime/users.jsonl")
_SOURCE_PATH = Path("data/scaled-v1/runtime/sources.jsonl")
_USER_FIELDS = {"user_id", "display_name", "timezone", "split", "profile_note"}
_SOURCE_FIELDS = {
    "source_id",
    "source_type",
    "user_id",
    "created_at",
    "ingested_at",
    "participants",
    "content",
    "messages",
    "metadata",
}
_MESSAGE_FIELDS = {"message_id", "speaker_id", "text"}


def load_scaled_development_sources(
    repo_root: str | Path = ".",
) -> tuple[ExtractionSource, ...]:
    """Adapt the two development users without parsing frozen test records."""

    root = Path(repo_root).resolve()
    users = _load_development_users(root / _USER_PATH)
    records = _read_prefix(root / _SOURCE_PATH, len(DEVELOPMENT_SOURCE_REFS))
    sources = tuple(
        _adapt_source(record, expected_ref, users)
        for record, expected_ref in zip(records, DEVELOPMENT_SOURCE_REFS)
    )
    return sources


def select_scaled_sources(
    sources: tuple[ExtractionSource, ...],
    refs: tuple[tuple[str, str], ...],
) -> tuple[ExtractionSource, ...]:
    """Select an exact ordered subset from the development source set."""

    by_ref = {(source.user_id, source.source_id): source for source in sources}
    if len(by_ref) != len(sources):
        raise HistoryDataError("scaled development sources contain duplicate IDs")
    missing = [ref for ref in refs if ref not in by_ref]
    if missing:
        raise HistoryDataError(f"unknown scaled development source reference: {missing[0]!r}")
    return tuple(by_ref[ref] for ref in refs)


def _load_development_users(path: Path) -> dict[str, str]:
    records = _read_prefix(path, len(DEVELOPMENT_USER_IDS))
    users: dict[str, str] = {}
    for expected_id, record in zip(DEVELOPMENT_USER_IDS, records):
        if set(record) != _USER_FIELDS:
            raise HistoryDataError("scaled development user fields changed")
        if record.get("user_id") != expected_id or record.get("split") != "development":
            raise HistoryDataError("scaled development user order or split changed")
        display_name = record.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise HistoryDataError("scaled development display_name must be non-empty")
        users[expected_id] = display_name
    return users


def _adapt_source(
    record: Mapping[str, object],
    expected_ref: tuple[str, str],
    user_names: Mapping[str, str],
) -> ExtractionSource:
    user_id, source_id = expected_ref
    if set(record) != _SOURCE_FIELDS:
        raise HistoryDataError(f"scaled source {source_id!r} fields changed")
    if record.get("user_id") != user_id or record.get("source_id") != source_id:
        raise HistoryDataError("scaled development source order or ownership changed")
    source_type = _string(record.get("source_type"), f"source {source_id} source_type")
    expected_type = source_id.rsplit("_", 2)[-2]
    if source_type != expected_type or source_type not in {
        "conversation",
        "email",
        "chat",
        "calendar",
    }:
        raise HistoryDataError(f"source {source_id!r} has an invalid source_type")
    created_at = _timestamp(record.get("created_at"), f"source {source_id} created_at")
    participants_raw = record.get("participants")
    if not isinstance(participants_raw, list) or not participants_raw:
        raise HistoryDataError(f"source {source_id!r} participants must be non-empty")
    participants = tuple(
        _string(value, f"source {source_id} participant") for value in participants_raw
    )
    if len(participants) != len(set(participants)) or user_id not in participants:
        raise HistoryDataError(f"source {source_id!r} participants are invalid")
    known_entities = tuple(
        KnownEntity(
            entity_id=participant,
            display_name=user_names.get(participant, participant),
        )
        for participant in participants
    )

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise HistoryDataError(f"source {source_id!r} messages must be a list")
    if source_type == "calendar":
        if messages:
            raise HistoryDataError(f"calendar source {source_id!r} must not have messages")
        content = _string(record.get("content"), f"source {source_id} content")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise HistoryDataError(f"source {source_id!r} metadata must be an object")
        title = _string(metadata.get("title"), f"source {source_id} title")
        observations = (
            HistoryObservation(
                observed_at=created_at,
                source_type=source_type,
                source_id=source_id,
                message_id=None,
                author_id=user_id,
                author_name=user_names[user_id],
                text=content,
                title=title,
            ),
        )
    else:
        if not messages:
            raise HistoryDataError(f"source {source_id!r} must contain a message")
        observations_list: list[HistoryObservation] = []
        seen_messages: set[str] = set()
        for item in messages:
            if not isinstance(item, dict) or set(item) != _MESSAGE_FIELDS:
                raise HistoryDataError(f"source {source_id!r} message fields changed")
            message_id = _string(item.get("message_id"), "message_id")
            speaker_id = _string(item.get("speaker_id"), "speaker_id")
            if message_id in seen_messages or speaker_id not in participants:
                raise HistoryDataError(f"source {source_id!r} message identity is invalid")
            seen_messages.add(message_id)
            observations_list.append(
                HistoryObservation(
                    observed_at=created_at,
                    source_type=source_type,
                    source_id=source_id,
                    message_id=message_id,
                    author_id=speaker_id,
                    author_name=user_names.get(speaker_id, speaker_id),
                    text=_string(item.get("text"), f"message {message_id} text"),
                )
            )
        observations = tuple(observations_list)

    return ExtractionSource(
        source_id=source_id,
        source_type=source_type,
        observations=observations,
        known_entities=known_entities,
        user_id=user_id,
    )


def _read_prefix(path: Path, count: int) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number in range(1, count + 1):
                line = handle.readline()
                if not line:
                    raise HistoryDataError(f"{path} ended before line {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise HistoryDataError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from None
                if not isinstance(record, dict):
                    raise HistoryDataError(f"{path}:{line_number}: record must be an object")
                records.append(record)
    except OSError as error:
        raise HistoryDataError(f"could not load {path.name}: {error}") from error
    return tuple(records)


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoryDataError(f"{location} must be a non-empty string")
    return value


def _timestamp(value: object, location: str) -> datetime:
    text = _string(value, location)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise HistoryDataError(f"{location} must be an ISO timestamp") from None
    if parsed.utcoffset() is None:
        raise HistoryDataError(f"{location} must include a UTC offset")
    return parsed
