"""Extract validated atomic claims from one prepared source."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

from evaluation.history import HistoryObservation
from evaluation.openai_client import OpenAIResponseMetadata

from .contracts import (
    AtomicClaimV1,
    AtomicClaimValidationError,
    validate_atomic_claim,
)
from .prompt import (
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
)
from .source import ExtractionSource


class AtomicExtractionValidationError(ValueError):
    """Reports invalid model output without retaining the raw response."""

    def __init__(self, source_id: str, errors: Sequence[str]) -> None:
        self.source_id = source_id
        self.errors = tuple(errors)
        super().__init__(f"source {source_id!r}: {'; '.join(self.errors)}")


@dataclass(frozen=True)
class AtomicExtractionResult:
    """Validated claims and non-secret metadata from one provider response."""

    source_id: str
    claims: tuple[AtomicClaimV1, ...]
    response_metadata: OpenAIResponseMetadata


def extract_atomic_claims(
    source_group: ExtractionSource,
    client: object,
) -> AtomicExtractionResult:
    """Make one model call and validate its claims against the supplied source."""

    raw_response, metadata = getattr(client, "complete_with_metadata")(
        system_prompt=ATOMIC_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_atomic_extraction_prompt(source_group),
    )
    return validate_atomic_response(source_group, raw_response, metadata)


def validate_atomic_response(
    source_group: ExtractionSource,
    raw_response: object,
    metadata: OpenAIResponseMetadata,
) -> AtomicExtractionResult:
    """Validate one returned response without making another provider call."""

    records = _parse_claim_records(raw_response, source_group.source_id)
    claims = _validate_claim_records(records, source_group)
    return AtomicExtractionResult(
        source_id=source_group.source_id,
        claims=claims,
        response_metadata=metadata,
    )


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
    records: list[object], source_group: ExtractionSource
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
            claim = validate_atomic_claim(record)
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
