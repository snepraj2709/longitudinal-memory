"""Strict structured-output schema for Phase 3 atomic extraction."""

from __future__ import annotations

from copy import deepcopy

from .contracts import (
    ALLOWED_EPISTEMIC_STATUSES,
    ALLOWED_POLARITIES,
    ALLOWED_PREDICATES,
)
from .predicate_registry import PredicateRegistry


ATOMIC_EXTRACTION_SCHEMA_VERSION = "atomic_extraction_v1"

_NON_EMPTY_STRING = {"type": "string", "minLength": 1}
_NULLABLE_STRING = {"type": ["string", "null"]}
_DATE_OR_DATETIME_OR_NULL = {
    "anyOf": [
        {"type": "string", "format": "date"},
        {"type": "string", "format": "date-time"},
        {"type": "null"},
    ]
}
_SCHEDULED_EVENT = {
    "type": "object",
    "properties": {
        "title": _NON_EMPTY_STRING,
        "location": _NULLABLE_STRING,
    },
    "required": ["title", "location"],
    "additionalProperties": False,
}
_PERCENTAGE_ALLOCATION = {
    "type": "object",
    "properties": {
        "marketing_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "product_percent": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["marketing_percent", "product_percent"],
    "additionalProperties": False,
}
_OBJECT_VALUE = {
    "anyOf": [
        _NON_EMPTY_STRING,
        {"type": "boolean"},
        {
            "type": "array",
            "items": {"type": "string", "format": "date"},
            "minItems": 1,
        },
        _SCHEDULED_EVENT,
        _PERCENTAGE_ALLOCATION,
    ]
}
_EVIDENCE = {
    "type": "object",
    "properties": {
        "source_id": _NON_EMPTY_STRING,
        "message_id": _NULLABLE_STRING,
        "quote": _NON_EMPTY_STRING,
    },
    "required": ["source_id", "message_id", "quote"],
    "additionalProperties": False,
}
_CLAIM = {
    "type": "object",
    "properties": {
        "claim_id": _NON_EMPTY_STRING,
        "subject_id": _NON_EMPTY_STRING,
        "speaker_id": _NON_EMPTY_STRING,
        "predicate": {"type": "string", "enum": sorted(ALLOWED_PREDICATES)},
        "object": _OBJECT_VALUE,
        "polarity": {"type": "string", "enum": sorted(ALLOWED_POLARITIES)},
        "epistemic_status": {
            "type": "string",
            "enum": sorted(ALLOWED_EPISTEMIC_STATUSES),
        },
        "valid_from": _DATE_OR_DATETIME_OR_NULL,
        "valid_to": _DATE_OR_DATETIME_OR_NULL,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": _EVIDENCE, "minItems": 1},
    },
    "required": [
        "claim_id",
        "subject_id",
        "speaker_id",
        "predicate",
        "object",
        "polarity",
        "epistemic_status",
        "valid_from",
        "valid_to",
        "confidence",
        "evidence",
    ],
    "additionalProperties": False,
}
_TEXT_FORMAT = {
    "type": "json_schema",
    "name": ATOMIC_EXTRACTION_SCHEMA_VERSION,
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": _CLAIM},
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


def atomic_extraction_text_format(
    registry: PredicateRegistry | None = None,
) -> dict[str, object]:
    """Return an isolated copy of the frozen Responses text format."""

    result = deepcopy(_TEXT_FORMAT)
    if registry is not None:
        result["schema"]["properties"]["claims"]["items"]["properties"][
            "predicate"
        ]["enum"] = sorted(registry.predicates)
    return result
