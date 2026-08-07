"""Prepare pilot source observations for atomic extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evaluation.history import (
    HistoryDataError,
    HistoryObservation,
    load_history_observations,
)


PILOT_SOURCE_DIR = Path("data/pilot/sources")


@dataclass(frozen=True)
class KnownEntity:
    """One observation-side identity available to the extractor."""

    entity_id: str
    display_name: str


@dataclass(frozen=True)
class ExtractionSource:
    """One source and its observations in their supplied order."""

    source_id: str
    source_type: str
    observations: tuple[HistoryObservation, ...]
    known_entities: tuple[KnownEntity, ...]


def group_source_observations(
    observations: Sequence[HistoryObservation],
) -> tuple[ExtractionSource, ...]:
    """Group observations by source without changing their within-source order."""

    grouped: dict[str, list[HistoryObservation]] = {}
    source_types: dict[str, str] = {}
    seen_references: set[tuple[str, str | None]] = set()
    entity_names: dict[str, str] = {}

    for observation in observations:
        reference = (observation.source_id, observation.message_id)
        if reference in seen_references:
            raise HistoryDataError(f"duplicate evidence reference {reference!r}")
        seen_references.add(reference)

        previous_name = entity_names.setdefault(
            observation.author_id, observation.author_name
        )
        if previous_name != observation.author_name:
            raise HistoryDataError(
                f"entity {observation.author_id!r} has conflicting names: "
                f"{previous_name!r} and {observation.author_name!r}"
            )

        previous_type = source_types.setdefault(
            observation.source_id, observation.source_type
        )
        if previous_type != observation.source_type:
            raise HistoryDataError(
                f"source {observation.source_id!r} contains mixed source types: "
                f"{previous_type!r} and {observation.source_type!r}"
            )
        grouped.setdefault(observation.source_id, []).append(observation)

    known_entities = tuple(
        KnownEntity(entity_id=entity_id, display_name=display_name)
        for entity_id, display_name in sorted(entity_names.items())
    )
    sources = [
        ExtractionSource(
            source_id=source_id,
            source_type=source_types[source_id],
            observations=tuple(source_observations),
            known_entities=known_entities,
        )
        for source_id, source_observations in grouped.items()
    ]
    sources.sort(
        key=lambda source: (
            min(observation.observed_at for observation in source.observations),
            source.source_id,
        )
    )
    return tuple(sources)


def load_pilot_sources() -> tuple[ExtractionSource, ...]:
    """Load and group only the frozen pilot source directory."""

    return group_source_observations(load_history_observations(PILOT_SOURCE_DIR))
