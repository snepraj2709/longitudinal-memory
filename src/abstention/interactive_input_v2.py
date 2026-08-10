"""Exact-prefix input boundary for development interactive cases and references."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import BinaryIO, Callable

from .interactive_contracts_v2 import (
    InteractiveAnsweringError,
    InteractiveRuntimeCase,
    canonical_json_bytes,
)


RUNTIME_SOURCE = Path("data/scaled-v1/runtime/interactive.jsonl")
GOLD_SOURCE = Path("data/scaled-v1/gold/interactive.jsonl")
PREFIX_COUNT = 4
SCORER_ONLY_FIELDS = {
    "abstention_reason", "acceptable_answers", "evidence", "expected_behaviours",
    "failure_tags", "gold_event_ids", "reference_answer", "reference_summary",
    "required_claim_ids", "review_status", "should_abstain",
}
RUNTIME_FIELDS = set(InteractiveRuntimeCase.__dataclass_fields__)


def load_runtime_prefix(
    path: str | Path,
    *,
    record_reader: Callable[[BinaryIO], bytes] | None = None,
) -> tuple[InteractiveRuntimeCase, ...]:
    """Read exactly four records, without requesting, seeking, or hashing record five."""

    reader = record_reader or _read_one
    rows: list[InteractiveRuntimeCase] = []
    with Path(path).open("rb") as stream:
        for _ in range(PREFIX_COUNT):
            raw = reader(stream)
            if not raw:
                raise InteractiveAnsweringError("runtime prefix ended before four records")
            value = _json_mapping(raw, "runtime case")
            if set(value).intersection(SCORER_ONLY_FIELDS):
                raise InteractiveAnsweringError("runtime case contains scorer-only fields")
            missing = RUNTIME_FIELDS - set(value)
            if missing:
                raise InteractiveAnsweringError("runtime case is missing required fields")
            retained = {name: value[name] for name in RUNTIME_FIELDS}
            rows.append(InteractiveRuntimeCase(**{
                **retained,
                "as_of": _datetime(retained["as_of"]),
            }))
    _validate_runtime_rows(rows)
    return tuple(rows)


def load_gold_prefix_raw(
    path: str | Path,
    *,
    record_reader: Callable[[BinaryIO], bytes] | None = None,
) -> tuple[dict[str, object], ...]:
    """Read exactly the authorized four development gold records after checkpoint freeze."""

    reader = record_reader or _read_one
    rows: list[dict[str, object]] = []
    with Path(path).open("rb") as stream:
        for _ in range(PREFIX_COUNT):
            raw = reader(stream)
            if not raw:
                raise InteractiveAnsweringError("gold prefix ended before four records")
            rows.append(_json_mapping(raw, "gold case"))
    identities = tuple((row.get("case_id"), row.get("user_id"), row.get("split"), row.get("task")) for row in rows)
    if len(set(identities)) != PREFIX_COUNT or any(
        split != "development" or task != "interactive" or user not in {"user_001", "user_002"}
        for _, user, split, task in identities
    ):
        raise InteractiveAnsweringError("gold prefix is outside the development interactive slice")
    return tuple(rows)


def runtime_prefix_bytes(rows: tuple[InteractiveRuntimeCase, ...]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def runtime_prefix_sha256(rows: tuple[InteractiveRuntimeCase, ...]) -> str:
    return hashlib.sha256(runtime_prefix_bytes(rows)).hexdigest()


def _validate_runtime_rows(rows: list[InteractiveRuntimeCase]) -> None:
    if len(rows) != PREFIX_COUNT or len({item.case_id for item in rows}) != PREFIX_COUNT:
        raise InteractiveAnsweringError("runtime prefix accounting changed")


def _read_one(stream: BinaryIO) -> bytes:
    return stream.readline()


def _json_mapping(raw: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InteractiveAnsweringError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise InteractiveAnsweringError(f"{name} is not an object")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise InteractiveAnsweringError("case as_of is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InteractiveAnsweringError("case as_of is not timezone-aware")
    return parsed
