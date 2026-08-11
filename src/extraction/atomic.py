"""Extract validated atomic claims from one prepared source."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Sequence

from evaluation.history import HistoryObservation
from evaluation.openai_client import OpenAIResponseMetadata

from .contracts import (
    AtomicClaimV1,
    AtomicClaimValidationError,
    validate_atomic_claim,
)
from .prompt import (
    ATOMIC_EXTRACTION_PROMPT_VERSION,
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
    get_atomic_extraction_system_prompt,
)
from .predicate_registry import PredicateRegistry, load_default_predicate_registry
from .source import ExtractionSource


class AtomicExtractionValidationError(ValueError):
    """Reports invalid model output without retaining the raw response."""

    def __init__(self, source_id: str, errors: Sequence[str]) -> None:
        self.source_id = source_id
        self.errors = tuple(errors)
        super().__init__(f"source {source_id!r}: {'; '.join(self.errors)}")


@dataclass(frozen=True)
class ValidationDiagnostic:
    """A stable validation category and safe structural location."""

    code: str
    location: str


@dataclass(frozen=True)
class AtomicExtractionResult:
    """Validated claims and non-secret metadata from one provider response."""

    source_id: str
    claims: tuple[AtomicClaimV1, ...]
    response_metadata: OpenAIResponseMetadata
    normalization_diagnostics: tuple[ValidationDiagnostic, ...]


def sanitized_validation_diagnostics(
    error: AtomicExtractionValidationError,
) -> tuple[ValidationDiagnostic, ...]:
    """Describe validation failures without retaining generated values or output."""

    return tuple(
        ValidationDiagnostic(
            code=_validation_error_code(message),
            location=_validation_error_location(message),
        )
        for message in error.errors
    )


def _validation_error_code(message: str) -> str:
    patterns = (
        ("response is not valid JSON", "response_invalid_json"),
        ("response must be an object", "response_invalid_root"),
        ("response is missing required field", "response_missing_field"),
        ("response contains unknown field", "response_unknown_field"),
        ("claims must be a list", "claims_invalid_type"),
        ("claim must be an object", "claim_invalid_type"),
        ("is missing required field", "claim_missing_field"),
        ("contains unknown field", "claim_unknown_field"),
        ("duplicates claim_id", "claim_duplicate_id"),
        ("predicate must be one of", "claim_unknown_predicate"),
        ("object must", "claim_invalid_object"),
        ("object.", "claim_invalid_object"),
        ("polarity must be one of", "claim_invalid_polarity"),
        ("epistemic_status must be one of", "claim_invalid_epistemic_status"),
        ("valid_to must not be before valid_from", "claim_reversed_valid_time"),
        ("must be an ISO date", "claim_invalid_valid_time"),
        ("confidence must", "claim_invalid_confidence"),
        ("evidence must", "claim_invalid_evidence"),
        ("duplicates evidence reference", "evidence_duplicate_reference"),
        (".source_id must match source", "evidence_wrong_source"),
        ("calendar evidence must use message_id", "evidence_invalid_message_id"),
        ("non-calendar evidence must use a message_id", "evidence_invalid_message_id"),
        (".message_id does not exist in source", "evidence_unknown_message_id"),
        (".message_id must be", "evidence_invalid_message_id"),
        (".quote is not an exact substring", "evidence_inexact_quote"),
        ("speaker_id", "claim_invalid_speaker"),
        ("subject_id", "claim_invalid_subject"),
        ("must be a non-empty string", "claim_invalid_string"),
    )
    return next((code for fragment, code in patterns if fragment in message), "validation_unknown")


def _validation_error_location(message: str) -> str:
    nested = re.match(r"^(claims\[\d+\]): (.+)$", message)
    if nested:
        claim_location, detail = nested.groups()
        evidence = re.match(r"^(evidence\[\d+\](?:\.[a-z_]+)?)", detail)
        if evidence:
            return f"{claim_location}.{evidence.group(1)}"
        field = re.match(
            r"^(claim_id|subject_id|speaker_id|predicate|object(?:\.[a-z_]+)?|"
            r"polarity|epistemic_status|valid_from|valid_to|confidence|evidence)",
            detail,
        )
        return f"{claim_location}.{field.group(1)}" if field else claim_location
    direct = re.match(
        r"^(claims\[\d+\](?:\.evidence\[\d+\])?(?:\.[a-z_]+)?)",
        message,
    )
    if direct:
        return direct.group(1)
    if message.startswith("claims ") or message.startswith("claims must"):
        return "claims"
    return "response"


def extract_atomic_claims(
    source_group: ExtractionSource,
    client: object,
    *,
    evidence_normalization_version: str | None = None,
    registry: PredicateRegistry | None = None,
) -> AtomicExtractionResult:
    """Make one model call and validate its claims against the supplied source."""

    raw_response, metadata = getattr(client, "complete_with_metadata")(
        system_prompt=(
            ATOMIC_EXTRACTION_SYSTEM_PROMPT
            if registry is None
            else get_atomic_extraction_system_prompt(
                ATOMIC_EXTRACTION_PROMPT_VERSION,
                registry=registry,
            )
        ),
        user_prompt=build_atomic_extraction_prompt(source_group),
    )
    return validate_atomic_response(
        source_group,
        raw_response,
        metadata,
        evidence_normalization_version=evidence_normalization_version,
        registry=registry,
    )


def validate_atomic_response(
    source_group: ExtractionSource,
    raw_response: object,
    metadata: OpenAIResponseMetadata,
    *,
    evidence_normalization_version: str | None = None,
    registry: PredicateRegistry | None = None,
) -> AtomicExtractionResult:
    """Validate one returned response without making another provider call."""

    records = _parse_claim_records(raw_response, source_group.source_id)
    normalization_diagnostics: tuple[ValidationDiagnostic, ...] = ()
    if evidence_normalization_version is not None:
        records, normalization_diagnostics = _normalize_claim_records(
            records, source_group, evidence_normalization_version
        )
    claims = _validate_claim_records(records, source_group, registry=registry)
    return AtomicExtractionResult(
        source_id=source_group.source_id,
        claims=claims,
        response_metadata=metadata,
        normalization_diagnostics=normalization_diagnostics,
    )


_UNICODE_PUNCTUATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }
)

_BOOLEAN_PREDICATES = {
    definition.predicate
    for definition in load_default_predicate_registry().definitions
    if definition.object_shape in {"boolean", "boolean_or_text"}
}


def _normalize_claim_records(
    records: list[object],
    source_group: ExtractionSource,
    normalization_version: str,
) -> tuple[list[object], tuple[ValidationDiagnostic, ...]]:
    allowed = {
        "unicode_punctuation_v1",
        "source_span_v1",
        "source_span_boolean_polarity_v2",
    }
    if normalization_version not in allowed:
        raise ValueError("unknown evidence normalization version")
    normalized = deepcopy(records)
    diagnostics: list[ValidationDiagnostic] = []
    if normalization_version == "source_span_boolean_polarity_v2":
        for claim_index, record in enumerate(normalized):
            if not isinstance(record, dict):
                continue
            if (
                record.get("predicate") in _BOOLEAN_PREDICATES
                and record.get("object") is False
                and record.get("polarity") in {"positive", "negative"}
            ):
                record["object"] = True
                record["polarity"] = (
                    "negative" if record["polarity"] == "positive" else "positive"
                )
                diagnostics.append(
                    ValidationDiagnostic(
                        code="claim_boolean_polarity_normalized",
                        location=f"claims[{claim_index}].object",
                    )
                )
    normalized, evidence_diagnostics = _normalize_evidence_quotes(
        normalized, source_group, normalization_version
    )
    return normalized, tuple([*diagnostics, *evidence_diagnostics])


def _normalize_evidence_quotes(
    records: list[object],
    source_group: ExtractionSource,
    normalization_version: str,
) -> tuple[list[object], tuple[ValidationDiagnostic, ...]]:
    normalized = deepcopy(records)
    observations = {
        (observation.source_id, observation.message_id): observation
        for observation in source_group.observations
    }
    diagnostics: list[ValidationDiagnostic] = []
    for claim_index, record in enumerate(normalized):
        if not isinstance(record, dict) or not isinstance(record.get("evidence"), list):
            continue
        for evidence_index, evidence in enumerate(record["evidence"]):
            if not isinstance(evidence, dict):
                continue
            quote = evidence.get("quote")
            if not isinstance(quote, str):
                continue
            observation = observations.get(
                (evidence.get("source_id"), evidence.get("message_id"))
            )
            if observation is None or quote in observation.text:
                continue
            normalized_quote = quote.translate(_UNICODE_PUNCTUATION)
            normalized_text = observation.text.translate(_UNICODE_PUNCTUATION)
            starts = _substring_starts(normalized_text, normalized_quote)
            if len(starts) > 1:
                continue
            if starts:
                start = starts[0]
                evidence["quote"] = observation.text[start : start + len(quote)]
                code = "evidence_quote_unicode_punctuation_normalized"
            elif normalization_version in (
                "source_span_v1",
                "source_span_boolean_polarity_v2",
            ):
                evidence["quote"] = observation.text
                code = "evidence_quote_replaced_with_cited_observation"
            else:
                continue
            diagnostics.append(
                ValidationDiagnostic(
                    code=code,
                    location=(
                        f"claims[{claim_index}].evidence[{evidence_index}].quote"
                    ),
                )
            )
    return normalized, tuple(diagnostics)


def _substring_starts(text: str, substring: str) -> tuple[int, ...]:
    if not substring:
        return ()
    starts: list[int] = []
    cursor = 0
    while True:
        start = text.find(substring, cursor)
        if start < 0:
            return tuple(starts)
        starts.append(start)
        cursor = start + 1


def _parse_claim_records(raw_response: object, source_id: str) -> list[object]:
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise AtomicExtractionValidationError(
            source_id, [f"response is not valid JSON: {detail}"]
        ) from None

    if not isinstance(parsed, dict):
        raise AtomicExtractionValidationError(
            source_id, ["response must be an object"]
        )

    errors: list[str] = []
    if "claims" not in parsed:
        errors.append("response is missing required field: claims")
    for field in sorted(field for field in parsed if field != "claims"):
        errors.append(f"response contains unknown field: {field}")
    if errors:
        raise AtomicExtractionValidationError(source_id, errors)

    claims = parsed["claims"]
    if not isinstance(claims, list):
        raise AtomicExtractionValidationError(source_id, ["claims must be a list"])
    return claims


def _validate_claim_records(
    records: list[object],
    source_group: ExtractionSource,
    *,
    registry: PredicateRegistry | None = None,
) -> tuple[AtomicClaimV1, ...]:
    errors: list[str] = []
    claims: list[AtomicClaimV1] = []
    seen_claim_ids: set[str] = set()
    observations = {
        (observation.source_id, observation.message_id): observation
        for observation in source_group.observations
    }
    source_speakers = {
        observation.author_id for observation in source_group.observations
    }
    known_entity_ids = {entity.entity_id for entity in source_group.known_entities}

    for index, record in enumerate(records):
        try:
            claim = validate_atomic_claim(record, registry=registry)
        except AtomicClaimValidationError as error:
            errors.extend(f"claims[{index}]: {message}" for message in error.errors)
            continue

        if claim.claim_id in seen_claim_ids:
            errors.append(f"claims[{index}] duplicates claim_id {claim.claim_id!r}")
        else:
            seen_claim_ids.add(claim.claim_id)

        errors.extend(
            _evidence_errors(
                claim,
                index,
                source_group,
                observations,
                source_speakers,
                known_entity_ids,
            )
        )
        claims.append(claim)

    if errors:
        raise AtomicExtractionValidationError(source_group.source_id, errors)

    claims.sort(
        key=lambda claim: (
            min(
                observations[(evidence.source_id, evidence.message_id)].observed_at
                for evidence in claim.evidence
            ),
            claim.claim_id,
        )
    )
    return tuple(claims)


def _evidence_errors(
    claim: AtomicClaimV1,
    claim_index: int,
    source_group: ExtractionSource,
    observations: dict[tuple[str, str | None], HistoryObservation],
    source_speakers: set[str],
    known_entity_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    cited: list[HistoryObservation] = []

    for evidence_index, evidence in enumerate(claim.evidence):
        location = f"claims[{claim_index}].evidence[{evidence_index}]"
        if evidence.source_id != source_group.source_id:
            errors.append(
                f"{location}.source_id must match source {source_group.source_id!r}"
            )
            continue
        if source_group.source_type == "calendar" and evidence.message_id is not None:
            errors.append(f"{location} calendar evidence must use message_id: null")
            continue
        if source_group.source_type != "calendar" and evidence.message_id is None:
            errors.append(f"{location} non-calendar evidence must use a message_id")
            continue

        observation = observations.get((evidence.source_id, evidence.message_id))
        if observation is None:
            errors.append(
                f"{location}.message_id does not exist in source "
                f"{source_group.source_id!r}"
            )
            continue
        cited.append(observation)
        if evidence.quote not in observation.text:
            errors.append(
                f"{location}.quote is not an exact substring of the cited text"
            )

    if claim.speaker_id not in source_speakers:
        errors.append(
            f"claims[{claim_index}].speaker_id {claim.speaker_id!r} "
            "does not exist in the source"
        )
    elif cited and not any(
        observation.author_id == claim.speaker_id for observation in cited
    ):
        errors.append(
            f"claims[{claim_index}].speaker_id {claim.speaker_id!r} "
            "does not match any cited observation"
        )
    if claim.subject_id not in known_entity_ids:
        errors.append(
            f"claims[{claim_index}].subject_id {claim.subject_id!r} "
            "does not exist in known_entities"
        )

    return errors
