"""User-scoped source reads for deterministic sessionization."""

from __future__ import annotations

from typing import Protocol

from .contracts import SessionBoundaryError, SessionSource, SessionizationRequest


VISIBLE_SOURCES_SQL = """
SELECT source_id, user_id, source_type, session_id, produced_at, ingested_at, metadata
FROM source_events
WHERE user_id = %s AND ingested_at <= %s
ORDER BY produced_at, source_id
""".strip()


class SupportsExecute(Protocol):
    def execute(self, query: str, params: tuple[object, ...]): ...


class SessionSourceRepository:
    """Read the current, visible sources for one user and cutoff."""

    def __init__(self, connection: SupportsExecute) -> None:
        self.connection = connection

    def list_visible_sources(
        self, request: SessionizationRequest
    ) -> tuple[SessionSource, ...]:
        rows = self.connection.execute(
            VISIBLE_SOURCES_SQL,
            (request.user_id, request.transaction_as_of),
        ).fetchall()
        sources: list[SessionSource] = []
        seen: set[str] = set()
        for row in rows:
            source = SessionSource(
                source_id=row[0],
                user_id=row[1],
                source_type=row[2],
                session_id=row[3],
                produced_at=row[4],
                ingested_at=row[5],
                metadata=row[6],
            )
            if source.user_id != request.user_id:
                raise SessionBoundaryError("source read crossed the user boundary")
            if source.ingested_at > request.transaction_as_of:
                raise SessionBoundaryError("source read crossed the visibility cutoff")
            if source.source_id in seen:
                raise SessionBoundaryError("source read returned a duplicate ID")
            seen.add(source.source_id)
            sources.append(source)
        ordered = tuple(sorted(sources, key=lambda item: (item.produced_at, item.source_id)))
        if tuple(sources) != ordered:
            raise SessionBoundaryError("source read order changed")
        return ordered
