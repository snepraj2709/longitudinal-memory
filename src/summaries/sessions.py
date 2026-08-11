"""Deterministic read-only session boundaries over visible source events."""

from __future__ import annotations

from collections import defaultdict
from datetime import timezone
import hashlib
import json
from pathlib import Path
from .contracts import (
    BOUNDARY_VERSION,
    SessionBoundaryConfig,
    SessionBoundaryError,
    SessionDefinition,
    SessionSource,
    SessionizationRequest,
    SessionizationResult,
    load_session_boundary_config,
)
from .repository import SessionSourceRepository


DEFAULT_CONFIG_PATH = Path("configs/summaries/session_boundaries_v1.json")


class SessionizationServiceError(RuntimeError):
    """A sanitized service failure with no source or thread value."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class SessionizationService:
    def __init__(
        self,
        repository: SessionSourceRepository,
        *,
        config: SessionBoundaryConfig | None = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.repository = repository
        self.config = config or load_session_boundary_config(config_path)

    def define_sessions(self, request: SessionizationRequest) -> SessionizationResult:
        if request.boundary_version != self.config.boundary_version:
            raise SessionizationServiceError("boundary_version_mismatch", "request")
        try:
            sources = self.repository.list_visible_sources(request)
        except SessionBoundaryError as error:
            raise SessionizationServiceError("source_contract_failed", "repository") from error
        except Exception as error:
            raise SessionizationServiceError("source_read_failed", "repository") from error
        self._validate_sources(request, sources)
        groups: list[tuple[str, str, str, tuple[SessionSource, ...]]] = []
        groups.extend(self._single_source_groups(sources, "conversation"))
        groups.extend(self._single_source_groups(sources, "calendar"))
        groups.extend(self._email_groups(sources))
        groups.extend(self._chat_groups(sources))
        definitions = tuple(
            self._definition(request, source_type, boundary_kind, boundary_key, members)
            for source_type, boundary_kind, boundary_key, members in groups
        )
        ordered = tuple(
            sorted(
                definitions,
                key=lambda item: (
                    item.start_at,
                    item.source_type,
                    item.source_ids[0],
                    item.definition_id,
                ),
            )
        )
        ids = [item.definition_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise SessionizationServiceError("definition_id_collision", "output")
        return SessionizationResult(
            user_id=request.user_id,
            transaction_as_of=request.transaction_as_of,
            boundary_version=request.boundary_version,
            definitions=ordered,
        )

    @staticmethod
    def _validate_sources(
        request: SessionizationRequest, sources: tuple[SessionSource, ...]
    ) -> None:
        expected = tuple(sorted(sources, key=lambda item: (item.produced_at, item.source_id)))
        if sources != expected:
            raise SessionizationServiceError("source_order_invalid", "repository")
        for source in sources:
            if source.user_id != request.user_id:
                raise SessionizationServiceError("source_user_mismatch", "repository")
            if source.ingested_at > request.transaction_as_of:
                raise SessionizationServiceError("source_not_visible", "repository")

    @staticmethod
    def _single_source_groups(
        sources: tuple[SessionSource, ...], source_type: str
    ) -> list[tuple[str, str, str, tuple[SessionSource, ...]]]:
        return [
            (source_type, "source", f"source:{source.source_id}", (source,))
            for source in sources
            if source.source_type == source_type
        ]

    def _email_groups(
        self, sources: tuple[SessionSource, ...]
    ) -> list[tuple[str, str, str, tuple[SessionSource, ...]]]:
        declared: dict[str, list[SessionSource]] = defaultdict(list)
        groups: list[tuple[str, str, str, tuple[SessionSource, ...]]] = []
        for source in sources:
            if source.source_type != "email":
                continue
            session_key = source.session_id
            metadata_key = _metadata_thread_id(source)
            if session_key is not None and metadata_key is not None and session_key != metadata_key:
                raise SessionizationServiceError("email_thread_conflict", "email_boundary")
            key = session_key or metadata_key
            if key is None:
                groups.append(
                    ("email", "declared_thread_or_source", f"source:{source.source_id}", (source,))
                )
            else:
                declared[key].append(source)
        groups.extend(
            (
                "email",
                "declared_thread_or_source",
                f"declared:{key}",
                tuple(members),
            )
            for key, members in declared.items()
        )
        return groups

    def _chat_groups(
        self, sources: tuple[SessionSource, ...]
    ) -> list[tuple[str, str, str, tuple[SessionSource, ...]]]:
        declared: dict[str, list[SessionSource]] = defaultdict(list)
        unthreaded: list[list[SessionSource]] = []
        current: list[SessionSource] = []
        previous_was_declared = False
        for source in sources:
            if source.source_type != "chat":
                continue
            key = source.session_id or _metadata_thread_id(source)
            if key is not None:
                if current:
                    unthreaded.append(current)
                    current = []
                declared[key].append(source)
                previous_was_declared = True
                continue
            if (
                current
                and not previous_was_declared
                and (source.produced_at - current[-1].produced_at).total_seconds()
                <= self.config.chat_unthreaded_gap_seconds
            ):
                current.append(source)
            else:
                if current:
                    unthreaded.append(current)
                current = [source]
            previous_was_declared = False
        if current:
            unthreaded.append(current)
        groups = [
            ("chat", "declared_thread", f"declared:{key}", tuple(members))
            for key, members in declared.items()
        ]
        groups.extend(
            (
                "chat",
                "inactivity",
                f"inactivity:{members[0].source_id}",
                tuple(members),
            )
            for members in unthreaded
        )
        return groups

    @staticmethod
    def _definition(
        request: SessionizationRequest,
        source_type: str,
        boundary_kind: str,
        boundary_key: str,
        members: tuple[SessionSource, ...],
    ) -> SessionDefinition:
        ordered = tuple(sorted(members, key=lambda item: (item.produced_at, item.source_id)))
        definition_id = _sha256(
            {
                "boundary_key": boundary_key,
                "boundary_kind": boundary_kind,
                "boundary_version": BOUNDARY_VERSION,
                "source_type": source_type,
                "user_id": request.user_id,
            }
        )
        source_ids = tuple(item.source_id for item in ordered)
        start_at = ordered[0].produced_at
        end_at = ordered[-1].produced_at
        membership = _sha256(
            {
                "boundary_kind": boundary_kind,
                "boundary_version": BOUNDARY_VERSION,
                "definition_id": definition_id,
                "end_at": _time(end_at),
                "source_ids": source_ids,
                "source_type": source_type,
                "start_at": _time(start_at),
                "transaction_as_of": _time(request.transaction_as_of),
                "user_id": request.user_id,
            }
        )
        return SessionDefinition(
            definition_id=definition_id,
            user_id=request.user_id,
            source_type=source_type,
            boundary_kind=boundary_kind,
            boundary_version=BOUNDARY_VERSION,
            source_ids=source_ids,
            start_at=start_at,
            end_at=end_at,
            transaction_as_of=request.transaction_as_of,
            membership_sha256=membership,
        )


def _metadata_thread_id(source: SessionSource) -> str | None:
    value = source.metadata.get("thread_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SessionizationServiceError("thread_id_invalid", f"{source.source_type}_boundary")
    return value


def _time(value) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
