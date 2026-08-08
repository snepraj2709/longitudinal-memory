"""Versioned predicate definitions shared by extraction runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


DEFAULT_PREDICATE_REGISTRY_PATH = Path(
    "configs/extraction/predicate_registry_v1.json"
)

PREDICATE_FAMILIES = frozenset(
    {
        "assessment",
        "belief",
        "commitment",
        "event",
        "goal",
        "identity",
        "preference",
        "relationship",
        "role",
        "schedule",
        "state",
        "task",
    }
)
SUBJECT_SCOPES = frozenset({"entity", "person"})
OBJECT_SHAPES = frozenset(
    {
        "boolean",
        "boolean_or_text",
        "date",
        "date_list",
        "date_list_or_text",
        "percentage_allocation",
        "scheduled_event",
        "text",
    }
)
TEMPORAL_BEHAVIORS = frozenset(
    {"atemporal", "event", "interval", "scheduled"}
)
CONFLICT_COMPATIBILITIES = frozenset(
    {"multi_value", "repeatable", "single_value"}
)

_ENTRY_FIELDS = frozenset(
    {
        "predicate",
        "family",
        "subject_scope",
        "object_shape",
        "temporal_behavior",
        "conflict_compatibility",
        "introduced_in",
    }
)
_REGISTRY_FIELDS = frozenset(
    {"registry_version", "review_status", "predicates"}
)
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class PredicateRegistryError(ValueError):
    """Raised when a predicate registry is malformed or incompatible."""


@dataclass(frozen=True)
class PredicateDefinition:
    """One immutable predicate definition from the active registry."""

    predicate: str
    family: str
    subject_scope: str
    object_shape: str
    temporal_behavior: str
    conflict_compatibility: str
    introduced_in: str


@dataclass(frozen=True)
class PredicateRegistry:
    """Validated registry metadata and definitions indexed by predicate."""

    registry_version: str
    review_status: str
    content_sha256: str
    definitions: tuple[PredicateDefinition, ...]
    by_predicate: Mapping[str, PredicateDefinition]

    @property
    def predicates(self) -> frozenset[str]:
        return frozenset(self.by_predicate)


def load_predicate_registry(path: str | Path) -> PredicateRegistry:
    """Load a registry and reject malformed or unreviewed definitions."""

    registry_path = Path(path)
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredicateRegistryError(
            f"could not load predicate registry: {error}"
        ) from error

    if not isinstance(raw, dict) or set(raw) != _REGISTRY_FIELDS:
        raise PredicateRegistryError("predicate registry fields changed")
    version = _require_snake_case(raw["registry_version"], "registry_version")
    if raw["review_status"] != "implementation_reviewed":
        raise PredicateRegistryError(
            "predicate registry review_status must be implementation_reviewed"
        )
    records = raw["predicates"]
    if not isinstance(records, list) or not records:
        raise PredicateRegistryError("predicates must be a non-empty list")

    definitions: list[PredicateDefinition] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        location = f"predicates[{index}]"
        if not isinstance(record, dict) or set(record) != _ENTRY_FIELDS:
            raise PredicateRegistryError(f"{location} fields changed")
        predicate = _require_snake_case(record["predicate"], f"{location}.predicate")
        if predicate in seen:
            raise PredicateRegistryError(f"duplicate predicate: {predicate}")
        seen.add(predicate)
        family = _require_choice(
            record["family"], PREDICATE_FAMILIES, f"{location}.family"
        )
        subject_scope = _require_choice(
            record["subject_scope"], SUBJECT_SCOPES, f"{location}.subject_scope"
        )
        object_shape = _require_choice(
            record["object_shape"], OBJECT_SHAPES, f"{location}.object_shape"
        )
        temporal_behavior = _require_choice(
            record["temporal_behavior"],
            TEMPORAL_BEHAVIORS,
            f"{location}.temporal_behavior",
        )
        conflict_compatibility = _require_choice(
            record["conflict_compatibility"],
            CONFLICT_COMPATIBILITIES,
            f"{location}.conflict_compatibility",
        )
        introduced_in = _require_snake_case(
            record["introduced_in"], f"{location}.introduced_in"
        )
        definitions.append(
            PredicateDefinition(
                predicate=predicate,
                family=family,
                subject_scope=subject_scope,
                object_shape=object_shape,
                temporal_behavior=temporal_behavior,
                conflict_compatibility=conflict_compatibility,
                introduced_in=introduced_in,
            )
        )

    if [item.predicate for item in definitions] != sorted(seen):
        raise PredicateRegistryError("predicates must be sorted by predicate")

    digest = hashlib.sha256(
        json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    mapping = MappingProxyType({item.predicate: item for item in definitions})
    return PredicateRegistry(
        registry_version=version,
        review_status=raw["review_status"],
        content_sha256=digest,
        definitions=tuple(definitions),
        by_predicate=mapping,
    )


@lru_cache(maxsize=1)
def load_default_predicate_registry() -> PredicateRegistry:
    """Load the repository's active runtime registry once."""

    repo_root = Path(__file__).resolve().parents[2]
    return load_predicate_registry(repo_root / DEFAULT_PREDICATE_REGISTRY_PATH)


def validate_object_shape(value: object, shape: str) -> tuple[str, ...]:
    """Return deterministic validation errors for one registered object shape."""

    if shape == "boolean":
        return () if isinstance(value, bool) else ("object must be a boolean",)
    if shape == "boolean_or_text":
        return () if isinstance(value, bool) or _is_text(value) else (
            "object must be a boolean or non-empty string",
        )
    if shape == "date":
        return () if _is_date(value) else ("object must be an ISO date string",)
    if shape == "date_list":
        return _validate_date_list(value)
    if shape == "date_list_or_text":
        return () if _is_text(value) else _validate_date_list(value)
    if shape == "scheduled_event":
        return _validate_scheduled_event(value)
    if shape == "percentage_allocation":
        return _validate_percentage_allocation(value)
    if shape == "text":
        return () if _is_text(value) else ("object must be a non-empty string",)
    return (f"object shape is unsupported: {shape}",)


def render_registry_for_prompt(registry: PredicateRegistry) -> str:
    """Render active definitions in a stable, value-free prompt format."""

    return "\n".join(
        (
            f"{item.predicate}: family={item.family}; "
            f"subject_scope={item.subject_scope}; object_shape={item.object_shape}; "
            f"temporal_behavior={item.temporal_behavior}; "
            f"conflict_compatibility={item.conflict_compatibility}; "
            f"introduced_in={item.introduced_in}"
        )
        for item in registry.definitions
    )


def _require_snake_case(value: object, location: str) -> str:
    if not isinstance(value, str) or not _SNAKE_CASE.fullmatch(value):
        raise PredicateRegistryError(f"{location} must be lowercase snake_case")
    return value


def _require_choice(value: object, allowed: frozenset[str], location: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise PredicateRegistryError(
            f"{location} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _validate_date_list(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(_is_date(item) for item in value)
    ):
        return ("object must be a non-empty list of ISO date strings",)
    return ()


def _validate_scheduled_event(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict) or set(value) != {"location", "title"}:
        return ("object must contain exactly title and location",)
    if not _is_text(value["title"]):
        return ("object.title must be a non-empty string",)
    if value["location"] is not None and not _is_text(value["location"]):
        return ("object.location must be a non-empty string or null",)
    return ()


def _validate_percentage_allocation(value: object) -> tuple[str, ...]:
    expected = {"marketing_percent", "product_percent"}
    if not isinstance(value, dict) or set(value) != expected:
        return (
            "object must contain exactly marketing_percent and product_percent",
        )
    percentages = tuple(value[field] for field in sorted(expected))
    if any(not isinstance(item, int) or isinstance(item, bool) for item in percentages):
        return ("object percentages must be integers",)
    if any(item < 0 or item > 100 for item in percentages):
        return ("object percentages must be between 0 and 100",)
    if sum(percentages) != 100:
        return ("object percentages must total 100",)
    return ()
