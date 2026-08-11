"""Pure rendering and planning for atomic and session index records."""

from __future__ import annotations

from datetime import datetime
import hashlib

from .contracts import (
    CONTENT_RENDERER_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_VERSION,
    AtomicIndexInput,
    IndexConfig,
    IndexRecord,
    RetrievalIndexError,
    SessionIndexInput,
    SessionStatementInput,
    canonical_json,
)
from .embeddings import Embedder


def build_atomic_index_record(
    requested_user_id: str,
    transaction_as_of: datetime,
    value: AtomicIndexInput,
    *,
    config: IndexConfig,
    embedder: Embedder,
) -> IndexRecord | None:
    """Build one visible atomic record, or exclude it before feature work."""

    _user_boundary(requested_user_id, value.user_id)
    if not value.transaction_time.contains(transaction_as_of):
        return None
    if (
        value.lifecycle_status not in config.eligible_lifecycle_statuses
        or value.lifecycle_status in config.excluded_lifecycle_statuses
        or value.sensitivity in config.excluded_sensitivities
    ):
        return None
    _embedder_boundary(config, embedder)
    content = render_atomic_content(value, config=config)
    snapshot = _atomic_snapshot(value, config)
    snapshot_sha256 = _sha256(canonical_json(snapshot))
    return IndexRecord(
        index_record_id=_stable_id(
            config.index_version,
            value.user_id,
            "atomic",
            value.claim_version_id,
            snapshot_sha256,
        ),
        user_id=value.user_id,
        index_version=config.index_version,
        record_kind="atomic",
        claim_version_id=value.claim_version_id,
        session_summary_id=None,
        subject_id=value.subject_id,
        speaker_id=value.speaker_id,
        predicate=value.predicate,
        content_text=content,
        content_sha256=_sha256(content),
        embedding_version=embedder.version,
        embedding=embedder.embed(content),
        lifecycle_statuses=(value.lifecycle_status,),
        memory_kind=value.memory_kind,
        epistemic_status=value.epistemic_status,
        valid_time=value.valid_time,
        transaction_time=value.transaction_time,
        sensitivity=value.sensitivity,
        contains_sensitive=value.sensitivity == "sensitive",
        input_snapshot_sha256=snapshot_sha256,
        claim_lineage=value.claim_lineage,
        source_lineage=value.source_lineage,
        relation_lineage=value.relation_lineage,
    )


def build_session_index_record(
    requested_user_id: str,
    transaction_as_of: datetime,
    value: SessionIndexInput,
    *,
    config: IndexConfig,
    embedder: Embedder,
) -> IndexRecord | None:
    """Build one visible session record, or exclude it before feature work."""

    _user_boundary(requested_user_id, value.user_id)
    if not value.transaction_time.contains(transaction_as_of):
        return None
    lifecycle_statuses = tuple(
        sorted({item.lifecycle_status for item in value.claim_lineage})
    )
    if (
        not lifecycle_statuses
        or any(status not in config.eligible_lifecycle_statuses for status in lifecycle_statuses)
        or value.sensitivity in config.excluded_sensitivities
    ):
        return None
    _embedder_boundary(config, embedder)
    content = render_session_content(value, config=config)
    snapshot = _session_snapshot(value, config)
    snapshot_sha256 = _sha256(canonical_json(snapshot))
    return IndexRecord(
        index_record_id=_stable_id(
            config.index_version,
            value.user_id,
            "session",
            value.session_summary_id,
            snapshot_sha256,
        ),
        user_id=value.user_id,
        index_version=config.index_version,
        record_kind="session",
        claim_version_id=None,
        session_summary_id=value.session_summary_id,
        subject_id=None,
        speaker_id=None,
        predicate=None,
        content_text=content,
        content_sha256=_sha256(content),
        embedding_version=embedder.version,
        embedding=embedder.embed(content),
        lifecycle_statuses=lifecycle_statuses,
        memory_kind=None,
        epistemic_status=None,
        valid_time=value.valid_time,
        transaction_time=value.transaction_time,
        sensitivity=value.sensitivity,
        contains_sensitive=value.contains_sensitive,
        input_snapshot_sha256=snapshot_sha256,
        claim_lineage=value.claim_lineage,
        source_lineage=value.source_lineage,
        relation_lineage=(),
    )


def render_atomic_content(value: AtomicIndexInput, *, config: IndexConfig) -> str:
    """Render only the frozen Claim and ClaimVersion fields."""

    fields = {
        "subject_id": value.subject_id,
        "speaker_id": value.speaker_id,
        "predicate": value.predicate,
        "object_json": value.object_json,
        "polarity": value.polarity,
        "epistemic_status": value.epistemic_status,
        "memory_kind": value.memory_kind,
        "lifecycle_status": value.lifecycle_status,
        "valid_time": _valid_time_value(value.valid_time),
        "transaction_time": _transaction_time_value(value.transaction_time),
        "checked_relations": [_relation_value(item) for item in value.relation_lineage],
    }
    return "\n".join(
        f"{name}:{canonical_json(fields[name])}"
        for name in config.atomic_content_fields
    )


def render_session_content(value: SessionIndexInput, *, config: IndexConfig) -> str:
    """Retain the persisted summary, statements, and questions without rewriting."""

    facts = [
        _statement_value(item)
        for item in value.statements
        if item.statement_kind == "observed_fact"
    ]
    questions = [
        _statement_value(item)
        for item in value.statements
        if item.statement_kind == "unresolved_question"
    ]
    fields = {
        "summary_text": value.summary_text,
        "ordered_grounded_statements": facts,
        "unresolved_questions": questions,
    }
    return "\n".join(
        f"{name}:{canonical_json(fields[name])}"
        for name in config.session_content_fields
    )


def _atomic_snapshot(value: AtomicIndexInput, config: IndexConfig) -> dict[str, object]:
    return {
        "content_renderer_version": CONTENT_RENDERER_VERSION,
        "index_version": config.index_version,
        "user_id": value.user_id,
        "claim_id": value.claim_id,
        "claim_version_id": value.claim_version_id,
        "content": render_atomic_content(value, config=config),
        "sensitivity": value.sensitivity,
        "claim_lineage": [_claim_value(item) for item in value.claim_lineage],
        "source_lineage": [_source_value(item) for item in value.source_lineage],
        "relation_lineage": [_relation_value(item) for item in value.relation_lineage],
    }


def _session_snapshot(value: SessionIndexInput, config: IndexConfig) -> dict[str, object]:
    return {
        "content_renderer_version": CONTENT_RENDERER_VERSION,
        "index_version": config.index_version,
        "user_id": value.user_id,
        "session_summary_id": value.session_summary_id,
        "session_definition_id": value.session_definition_id,
        "renderer_version": value.renderer_version,
        "content": render_session_content(value, config=config),
        "session_source_ids": value.session_source_ids,
        "valid_time": _valid_time_value(value.valid_time),
        "transaction_time": _transaction_time_value(value.transaction_time),
        "sensitivity": value.sensitivity,
        "contains_sensitive": value.contains_sensitive,
        "claim_lineage": [_claim_value(item) for item in value.claim_lineage],
        "source_lineage": [_source_value(item) for item in value.source_lineage],
    }


def _statement_value(value: SessionStatementInput) -> dict[str, object]:
    return {
        "statement_id": value.statement_id,
        "statement_kind": value.statement_kind,
        "lifecycle_view": value.lifecycle_view,
        "text": value.text,
        "claim_ids": tuple(item.claim_id for item in value.claim_lineage),
        "claim_version_ids": tuple(item.claim_version_id for item in value.claim_lineage),
    }


def _claim_value(value: object) -> dict[str, object]:
    return {
        "user_id": getattr(value, "user_id"),
        "claim_id": getattr(value, "claim_id"),
        "claim_version_id": getattr(value, "claim_version_id"),
        "lifecycle_status": getattr(value, "lifecycle_status"),
        "order": getattr(value, "order"),
    }


def _source_value(value: object) -> dict[str, object]:
    return {
        "user_id": getattr(value, "user_id"),
        "claim_id": getattr(value, "claim_id"),
        "claim_version_id": getattr(value, "claim_version_id"),
        "source_id": getattr(value, "source_id"),
        "span_id": getattr(value, "span_id"),
        "support_type": getattr(value, "support_type"),
        "order": getattr(value, "order"),
    }


def _relation_value(value: object) -> dict[str, object]:
    return {
        "relation_id": getattr(value, "relation_id"),
        "source_claim_id": getattr(value, "source_claim_id"),
        "target_claim_id": getattr(value, "target_claim_id"),
        "relation_type": getattr(value, "relation_type"),
        "direction": getattr(value, "direction"),
        "order": getattr(value, "order"),
    }


def _valid_time_value(value: object) -> dict[str, object]:
    return {
        "time_precision": getattr(value, "time_precision"),
        "valid_from_date": getattr(value, "valid_from_date"),
        "valid_from_timestamp": getattr(value, "valid_from_timestamp"),
        "valid_to_date": getattr(value, "valid_to_date"),
        "valid_to_timestamp": getattr(value, "valid_to_timestamp"),
    }


def _transaction_time_value(value: object) -> dict[str, object]:
    return {
        "transaction_from": getattr(value, "transaction_from"),
        "transaction_to": getattr(value, "transaction_to"),
    }


def _embedder_boundary(config: IndexConfig, embedder: Embedder) -> None:
    if (
        embedder.version != EMBEDDING_VERSION
        or embedder.version != config.embedding_version
        or embedder.dimension != EMBEDDING_DIMENSION
        or embedder.dimension != config.embedding_dimension
    ):
        raise RetrievalIndexError("embedder contract changed")


def _user_boundary(requested_user_id: str, record_user_id: str) -> None:
    if not isinstance(requested_user_id, str) or not requested_user_id.strip():
        raise RetrievalIndexError("requested user_id must be nonempty text")
    if requested_user_id != record_user_id:
        raise RetrievalIndexError("index input crosses users")


def _stable_id(*values: object) -> str:
    return _sha256(canonical_json(values))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
