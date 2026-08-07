"""Load the frozen, source-grounded atomic-extraction gold cases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from .atomic import AtomicExtractionValidationError, _validate_claim_records
from .contracts import AtomicClaimV1
from .source import ExtractionSource, load_pilot_sources


ATOMIC_GOLD_PATH = Path("data/phase3/atomic_extraction_gold.jsonl")
_CASE_FIELDS = frozenset({"case_id", "source_id", "expected_claims"})


class AtomicGoldDataError(ValueError):
    """Reports invalid atomic-extraction gold data."""


@dataclass(frozen=True)
class AtomicGoldCase:
    """One source and its ordered, manually annotated expected claims."""

    case_id: str
    source_id: str
    expected_claims: tuple[AtomicClaimV1, ...]


def load_atomic_gold(
    path: str | Path = ATOMIC_GOLD_PATH,
    source_groups: Sequence[ExtractionSource] | None = None,
) -> tuple[AtomicGoldCase, ...]:
    """Load gold cases in file order and validate them against pilot sources."""

    path = Path(path)
    sources = tuple(source_groups) if source_groups is not None else load_pilot_sources()
    sources_by_id = {source.source_id: source for source in sources}
    cases: list[AtomicGoldCase] = []
    seen_case_ids: set[str] = set()
    seen_source_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AtomicGoldDataError(
                    f"{location}: invalid JSON: {error.msg}"
                ) from None
            cases.append(
                _validate_case(
                    record,
                    location,
                    sources_by_id,
                    seen_case_ids,
                    seen_source_ids,
                )
            )

    return tuple(cases)


def _validate_case(
    record: object,
    location: str,
    sources_by_id: dict[str, ExtractionSource],
    seen_case_ids: set[str],
    seen_source_ids: set[str],
) -> AtomicGoldCase:
    if not isinstance(record, dict):
        raise AtomicGoldDataError(f"{location}: case must be an object")

    errors: list[str] = []
    missing = sorted(_CASE_FIELDS - record.keys())
    unknown = sorted(record.keys() - _CASE_FIELDS)
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")

    case_id = record.get("case_id")
    source_id = record.get("source_id")
    expected_claims = record.get("expected_claims")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id must be a non-empty string")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id must be a non-empty string")
    if not isinstance(expected_claims, list):
        errors.append("expected_claims must be a list")
    if errors:
        raise AtomicGoldDataError(f"{location}: {'; '.join(errors)}")

    if case_id in seen_case_ids:
        raise AtomicGoldDataError(f"{location}: duplicate case_id {case_id!r}")
    if source_id in seen_source_ids:
        raise AtomicGoldDataError(f"{location}: duplicate source_id {source_id!r}")
    source_group = sources_by_id.get(source_id)
    if source_group is None:
        raise AtomicGoldDataError(f"{location}: unknown source_id {source_id!r}")

    try:
        claims = _validate_claim_records(expected_claims, source_group)
    except AtomicExtractionValidationError as error:
        raise AtomicGoldDataError(f"{location}: {error}") from None

    seen_case_ids.add(case_id)
    seen_source_ids.add(source_id)
    return AtomicGoldCase(
        case_id=case_id,
        source_id=source_id,
        expected_claims=claims,
    )
