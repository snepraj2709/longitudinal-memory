"""PostgreSQL-backed Qwen extraction and B0-B7 context materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from conflicts.candidates import CandidateRequest, ConflictCandidateService
from conflicts.classifier import CLASSIFIER_VERSION, ClassificationRequest, ConflictClassifier
from conflicts.relations import ConflictRelationService
from conflicts.resolution import BeliefResolutionService
from conflicts.resolver import RESOLVER_VERSION, ResolutionRequest
from extraction.phase4_input import Phase4InputClaim, build_phase4_source_claims
from extraction.predicate_registry import load_predicate_registry
from extraction.scaled_source import _adapt_source
from ingestion.contracts import ClaimWrite, IngestRequest, stable_id
from ingestion.service import IngestionService
from retrieval.baselines import load_baseline_config
from retrieval.contracts import INDEX_VERSION, canonical_json, load_index_config
from retrieval.embeddings import DeterministicTokenHashEmbedder
from retrieval.index_evaluation import _atomic_inputs, _session_inputs
from retrieval.indexing import build_atomic_index_record, build_session_index_record
from retrieval.query_contracts import RetrievalQueryRequest
from retrieval.repository import IndexBuildRequest, RetrievalIndexRepository
from retrieval.search_repository import RetrievalSearchRepository
from storage.contracts import (
    ClaimRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations
from storage.repository import StorageRepository
from summaries.grounded import GroundedSummaryCoordinator
from temporal.contracts import TemporalQuery, TransitionRequest
from temporal.service import TemporalService

from .qwen_v2_contract import SERIES_ID, load_qwen_v2_config


REGISTRY_PATH = Path("configs/extraction/predicate_registry_v2.json")
INDEX_CONFIG_PATH = Path("configs/retrieval/index_v1.json")
BASELINE_CONFIG_PATH = Path("configs/retrieval/baseline_v1.json")
PROMPT_MODULE_PATH = Path("src/extraction/prompt.py")
SCHEMA_MODULE_PATH = Path("src/extraction/schema.py")
TASK_KEYS = ("qa", "summary", "interactive")
ALL_STATUSES = frozenset(
    {"candidate", "confirmed", "current", "historical", "disputed", "superseded"}
)


class QwenMaterializationError(RuntimeError):
    """Raised when runtime data cannot produce an isolated persisted context release."""


@dataclass(frozen=True)
class ContextPackage:
    series_id: str
    split: str
    task: str
    case_id: str
    user_id: str
    as_of: str
    baseline_id: str
    provider_call: bool
    snapshot: str
    context_records: tuple[dict[str, object], ...]
    context_sha256: str


@dataclass(frozen=True)
class MaterializationFailure:
    stage: str
    user_id: str
    record_id: str
    code: str


@dataclass(frozen=True)
class MaterializationResult:
    split: str
    contexts: tuple[ContextPackage, ...]
    failures: tuple[MaterializationFailure, ...]
    database_counts: Mapping[str, int]


def materialize_qwen_contexts(
    connection: object,
    *,
    repo_root: Path,
    split: str,
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    extraction_rows: Sequence[Mapping[str, object]],
    series_id: str = SERIES_ID,
    config_path: Path | None = None,
) -> MaterializationResult:
    """Build every baseline context through the persisted Phase 4-7 services."""

    root = repo_root.resolve()
    if config_path is None:
        load_qwen_v2_config(root)
    context_record_limit = _context_record_limit(root, config_path)
    citation_evidence_limit = _citation_evidence_limit(root, config_path)
    _validate_runtime(split, runtime, extraction_rows)
    _require_clean_database(connection)
    apply_migrations(connection, root / "migrations")

    users = tuple(runtime["users"])
    sources = tuple(sorted(runtime["sources"], key=_source_order))
    source_citations = _source_citation_index(sources)
    as_of = _shared_as_of(runtime)
    candidate_cutoff = as_of - timedelta(seconds=3)
    temporal_at = as_of - timedelta(seconds=2)
    conflict_at = as_of - timedelta(seconds=1)
    conflict_index_at = as_of - timedelta(milliseconds=100)
    failures: list[MaterializationFailure] = []
    claim_ids_by_user, extraction_failures = _materialize_extraction(
        connection,
        root,
        series_id,
        config_path,
        users,
        sources,
        extraction_rows,
    )
    failures.extend(extraction_failures)

    _rebuild_summaries(connection, tuple(claim_ids_by_user), candidate_cutoff)
    contexts: list[ContextPackage] = []
    contexts.extend(_base_contexts(split, runtime, series_id))
    _build_index(connection, root, series_id, tuple(claim_ids_by_user), candidate_cutoff, "candidate")
    contexts.extend(
        _retrieved_contexts(
            connection,
            root,
            split,
            runtime,
            {"B2": "B2", "B3": "B3", "B4": "B4"},
            "candidate_extraction",
            series_id,
            context_record_limit=context_record_limit,
            citation_evidence_limit=citation_evidence_limit,
            source_citations=source_citations,
        )
    )
    _drop_index_snapshots(connection)

    _apply_temporal_lifecycle(connection, claim_ids_by_user, as_of, temporal_at, series_id)
    _rebuild_summaries(connection, tuple(claim_ids_by_user), temporal_at)
    _build_index(connection, root, series_id, tuple(claim_ids_by_user), temporal_at, "temporal")
    contexts.extend(
        _retrieved_contexts(
            connection,
            root,
            split,
            runtime,
            {"B5": "B4"},
            "persisted_temporal_lifecycle",
            series_id,
            context_record_limit=context_record_limit,
            citation_evidence_limit=citation_evidence_limit,
            source_citations=source_citations,
        )
    )
    _drop_index_snapshots(connection)

    # B6 is a separate ablation snapshot. Rebuild the same extraction state, then
    # let the Phase 5 resolver own pair lifecycle changes before applying the
    # Phase 4 policy to claims that remain candidates.
    _reset_materialization_database(connection, root)
    claim_ids_by_user, _ = _materialize_extraction(
        connection,
        root,
        series_id,
        config_path,
        users,
        sources,
        extraction_rows,
    )
    failures.extend(_apply_conflicts(connection, root, claim_ids_by_user, temporal_at, as_of, series_id))
    _apply_temporal_lifecycle(connection, claim_ids_by_user, as_of, conflict_at, series_id)
    _rebuild_summaries(connection, tuple(claim_ids_by_user), conflict_index_at)
    _build_index(connection, root, series_id, tuple(claim_ids_by_user), conflict_index_at, "conflict")
    b6_contexts = _retrieved_contexts(
        connection,
        root,
        split,
        runtime,
        {"B6": "B4"},
        "persisted_conflict_resolution",
        series_id,
        context_record_limit=context_record_limit,
        citation_evidence_limit=citation_evidence_limit,
        source_citations=source_citations,
    )
    contexts.extend(b6_contexts)
    contexts.extend(_b7_contexts(b6_contexts))

    ordered = tuple(sorted(contexts, key=_context_order))
    _validate_context_accounting(runtime, ordered)
    return MaterializationResult(
        split,
        ordered,
        tuple(sorted(failures, key=lambda item: (item.stage, item.user_id, item.record_id))),
        _database_counts(connection),
    )


def write_materialization(
    result: MaterializationResult,
    output_dir: Path,
    *,
    series_id: str = SERIES_ID,
) -> None:
    """Write one immutable, canonically ordered context release."""

    output_dir.mkdir(parents=True, exist_ok=False)
    _exclusive_jsonl(output_dir / "contexts.jsonl", (asdict(item) for item in result.contexts))
    _exclusive_jsonl(output_dir / "failures.jsonl", (asdict(item) for item in result.failures))
    manifest = {
        "schema_version": "qwen_context_materialization_v2",
        "series_id": series_id,
        "split": result.split,
        "status": "completed" if not result.failures else "completed_with_failures",
        "context_count": len(result.contexts),
        "failure_count": len(result.failures),
        "contexts_sha256": _file_sha(output_dir / "contexts.jsonl"),
        "failures_sha256": _file_sha(output_dir / "failures.jsonl"),
        "database_counts": dict(sorted(result.database_counts.items())),
        "b7_provider_request_count": 0,
        "b6_b7_context_identity": all(
            _paired_context_identity(result.contexts, item)
            for item in result.contexts
            if item.baseline_id == "B7"
        ),
    }
    _exclusive_json(output_dir / "manifest.json", manifest)


def _validate_runtime(
    split: str,
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    extraction_rows: Sequence[Mapping[str, object]],
) -> None:
    if split not in {"development", "test"}:
        raise QwenMaterializationError("split must be development or test")
    if set(runtime) != {"users", "sources", *TASK_KEYS}:
        raise QwenMaterializationError("runtime fields changed")
    user_ids = {str(row["user_id"]) for row in runtime["users"]}
    if not user_ids or any(row.get("split") != split for row in runtime["users"]):
        raise QwenMaterializationError("runtime user split changed")
    for key in ("sources", *TASK_KEYS):
        if any(str(row.get("user_id")) not in user_ids for row in runtime[key]):
            raise QwenMaterializationError(f"{key} crosses the user boundary")
    source_ids = {str(row["source_id"]) for row in runtime["sources"]}
    checkpoint_ids = [str(row.get("record_id")) for row in extraction_rows]
    if len(checkpoint_ids) != len(set(checkpoint_ids)) or set(checkpoint_ids) != source_ids:
        raise QwenMaterializationError("extraction checkpoints do not cover the split sources")
    if any(str(row.get("user_id")) not in user_ids for row in extraction_rows):
        raise QwenMaterializationError("extraction checkpoint crosses the user boundary")


def _shared_as_of(runtime: Mapping[str, Sequence[Mapping[str, object]]]) -> datetime:
    values = {
        datetime.fromisoformat(str(row["as_of"]))
        for key in TASK_KEYS
        for row in runtime[key]
    }
    if len(values) != 1:
        raise QwenMaterializationError("v2 requires one frozen as_of boundary")
    value = next(iter(values))
    if value.utcoffset() is None:
        raise QwenMaterializationError("as_of must be timezone-aware")
    return value


def _require_clean_database(connection: object) -> None:
    table = connection.execute("SELECT to_regclass('public.memory_users')").fetchone()[0]
    if table is not None and connection.execute("SELECT count(*) FROM memory_users").fetchone()[0]:
        raise QwenMaterializationError("materialization database must be clean")


def _reset_materialization_database(connection: object, root: Path) -> None:
    connection.execute("DROP SCHEMA public CASCADE")
    connection.execute("CREATE SCHEMA public")
    apply_migrations(connection, root / "migrations")


def _materialize_extraction(
    connection: object,
    root: Path,
    series_id: str,
    config_path: Path | None,
    users: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
    extraction_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, list[str]], tuple[MaterializationFailure, ...]]:
    repository = StorageRepository(connection)
    created_at = min(datetime.fromisoformat(str(item["created_at"])) for item in sources)
    for user in users:
        repository.insert_user(MemoryUser(str(user["user_id"]), created_at))
    extraction_version = _extraction_version(root, created_at, series_id, config_path)
    repository.insert_extraction_version(extraction_version)

    rows_by_source = {str(row["record_id"]): row for row in extraction_rows}
    names = {str(row["user_id"]): str(row["display_name"]) for row in users}
    registry = load_predicate_registry(root / REGISTRY_PATH)
    ingestion = IngestionService(connection)
    claim_ids_by_user: dict[str, list[str]] = {str(row["user_id"]): [] for row in users}
    failures = []
    for source in sources:
        source_record = _source_record(source, series_id)
        result = ingestion.ingest(IngestRequest(source_record, extraction_version.version_id))
        if not result.created or result.attempt_id is None:
            raise QwenMaterializationError("source ingestion unexpectedly replayed")
        lease_time = source_record.ingested_at + timedelta(microseconds=1)
        leased = ingestion.lease_next("qwen-v2-materializer", lease_time, timedelta(minutes=5))
        if leased is None or leased.attempt_id != result.attempt_id:
            raise QwenMaterializationError("ingestion lease order changed")
        checkpoint = rows_by_source[source_record.source_id]
        completed_at = lease_time + timedelta(microseconds=1)
        if checkpoint.get("valid") is not True:
            ingestion.fail_attempt(
                source_record.user_id,
                leased.attempt_id,
                "qwen-v2-materializer",
                "validation_failure",
                completed_at,
                retryable=False,
            )
            failures.append(
                MaterializationFailure(
                    "extraction",
                    source_record.user_id,
                    source_record.source_id,
                    "invalid_extraction_checkpoint",
                )
            )
            continue
        raw_claims = checkpoint.get("claims")
        if not isinstance(raw_claims, list):
            raise QwenMaterializationError("valid extraction checkpoint has no claims list")
        adapted = _adapt_source(
            source,
            (source_record.user_id, source_record.source_id),
            names,
        )
        claims = build_phase4_source_claims(adapted, raw_claims, registry)
        writes = tuple(
            _claim_write(source_record, claim, extraction_version.version_id)
            for claim in claims
        )
        ingestion.complete_attempt(
            source_record.user_id,
            leased.attempt_id,
            "qwen-v2-materializer",
            completed_at,
            writes,
        )
        claim_ids_by_user[source_record.user_id].extend(claim.claim_id for claim in claims)
    return claim_ids_by_user, tuple(failures)


def _extraction_version(
    root: Path,
    created_at: datetime,
    series_id: str,
    config_path: Path | None,
) -> ExtractionVersionRecord:
    config = (
        load_qwen_v2_config(root)
        if config_path is None
        else json.loads((root / config_path).read_text(encoding="utf-8"))
    )
    model = config["model"]
    return ExtractionVersionRecord(
        f"{series_id.replace('-', '_')}_extraction",
        f"{model['hugging_face_id']}@{model['revision']}",
        "atomic-extraction-v3",
        _file_sha(root / PROMPT_MODULE_PATH),
        "atomic_extraction_v1",
        _file_sha(root / SCHEMA_MODULE_PATH),
        "predicate_registry_v2",
        _file_sha(root / REGISTRY_PATH),
        _file_sha(root / "data/scaled-v1/manifest.json"),
        created_at,
    )


def _source_record(source: Mapping[str, object], series_id: str) -> SourceEventRecord:
    metadata = source["metadata"]
    if not isinstance(metadata, dict):
        raise QwenMaterializationError("source metadata must be an object")
    session_id = metadata.get("thread_id") or metadata.get("event_id") or source["source_id"]
    content = str(source["content"])
    return SourceEventRecord(
        str(source["source_id"]),
        str(source["user_id"]),
        str(source["source_type"]),
        str(session_id),
        f"{series_id}:{source['source_id']}",
        datetime.fromisoformat(str(source["created_at"])),
        datetime.fromisoformat(str(source["ingested_at"])),
        content,
        list(source["participants"]),
        metadata,
        sha256(content.encode("utf-8")).hexdigest(),
    )


def _claim_write(
    source: SourceEventRecord,
    claim: Phase4InputClaim,
    extraction_version_id: str,
) -> ClaimWrite:
    spans: list[SourceSpanRecord] = []
    evidence_links: list[EvidenceLinkRecord] = []
    for evidence in claim.evidence:
        start = source.raw_content.find(evidence.quote)
        if start < 0:
            raise QwenMaterializationError("validated evidence quote is absent from source")
        span_id = stable_id(
            "span",
            claim.user_id,
            evidence.source_id,
            evidence.message_id,
            evidence.quote,
        )
        spans.append(
            SourceSpanRecord(
                span_id,
                claim.user_id,
                evidence.source_id,
                evidence.message_id,
                claim.speaker_id,
                evidence.quote,
                start,
                start + len(evidence.quote),
            )
        )
        evidence_links.append(
            EvidenceLinkRecord(
                claim.user_id,
                claim.claim_id,
                span_id,
                "supports",
                claim.confidence,
            )
        )
    valid_from_date, valid_from_timestamp = _time_boundary(claim.valid_from, claim.time_precision)
    valid_to_date, valid_to_timestamp = _time_boundary(claim.valid_to, claim.time_precision)
    record = ClaimRecord(
        claim.claim_id,
        claim.user_id,
        claim.subject_id,
        claim.speaker_id,
        claim.predicate,
        claim.predicate_registry_version,
        claim.object,
        claim.polarity,
        claim.epistemic_status,
        valid_from_date,
        valid_from_timestamp,
        valid_to_date,
        valid_to_timestamp,
        claim.time_precision,
        claim.confidence,
        "episodic",
        "standard",
        extraction_version_id,
    )
    return ClaimWrite(record, tuple(spans), tuple(evidence_links))


def _time_boundary(value: str | None, precision: str) -> tuple[date | None, datetime | None]:
    if value is None:
        return None, None
    if precision == "timestamp":
        return None, datetime.fromisoformat(value)
    return date.fromisoformat(value), None


def _rebuild_summaries(connection: object, user_ids: tuple[str, ...], cutoff: datetime) -> None:
    repository = StorageRepository(connection)
    coordinator = GroundedSummaryCoordinator(connection)
    for user_id in sorted(user_ids):
        row = connection.execute(
            """
            SELECT event_id FROM processing_outbox
            WHERE user_id = %s AND created_at <= %s
              AND event_type = ANY(%s)
            ORDER BY created_at DESC, event_id DESC LIMIT 1
            """,
            (
                user_id,
                cutoff,
                [
                    "claims_changed",
                    "claim_lifecycle_changed",
                    "belief_resolved",
                    "conflict_recompute_required",
                ],
            ),
        ).fetchone()
        if row is None:
            continue
        event = repository.get_outbox(user_id, row[0])
        if event is None:
            raise QwenMaterializationError("summary event disappeared")
        coordinator.process(event)


def _build_index(
    connection: object,
    root: Path,
    series_id: str,
    user_ids: tuple[str, ...],
    cutoff: datetime,
    snapshot: str,
) -> None:
    config = load_index_config(root / INDEX_CONFIG_PATH)
    embedder = DeterministicTokenHashEmbedder()
    repository = RetrievalIndexRepository(connection, config=config, config_path=root / INDEX_CONFIG_PATH)
    for user_id in sorted(user_ids):
        records = []
        for value in _atomic_inputs(connection, user_id, cutoff):
            record = build_atomic_index_record(user_id, cutoff, value, config=config, embedder=embedder)
            if record is not None:
                records.append(record)
        for value in _session_inputs(connection, user_id, cutoff):
            record = build_session_index_record(user_id, cutoff, value, config=config, embedder=embedder)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.index_record_id)
        input_snapshot = sha256(
            canonical_json(tuple((item.index_record_id, item.input_snapshot_sha256) for item in records)).encode("utf-8")
        ).hexdigest()
        repository.persist(
            IndexBuildRequest(
                user_id,
                INDEX_VERSION,
                repository.config_sha256,
                cutoff,
                f"{series_id}:{snapshot}:{user_id}",
                input_snapshot,
                cutoff,
                cutoff,
            ),
            records,
        )


def _base_contexts(
    split: str,
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    series_id: str = SERIES_ID,
) -> tuple[ContextPackage, ...]:
    sources_by_user: dict[str, list[Mapping[str, object]]] = {}
    for source in runtime["sources"]:
        sources_by_user.setdefault(str(source["user_id"]), []).append(source)
    contexts = []
    for task in TASK_KEYS:
        for case in runtime[task]:
            as_of = datetime.fromisoformat(str(case["as_of"]))
            visible_sources = tuple(
                _source_context_record(source)
                for source in sorted(sources_by_user[str(case["user_id"])], key=_source_order)
                if datetime.fromisoformat(str(source["ingested_at"])) <= as_of
            )
            contexts.append(_context(split, task, case, "B0", True, "query_only", (), series_id))
            contexts.append(
                _context(
                    split,
                    task,
                    case,
                    "B1",
                    True,
                    "same_user_full_history",
                    visible_sources,
                    series_id,
                )
            )
    return tuple(contexts)


def _retrieved_contexts(
    connection: object,
    root: Path,
    split: str,
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    baseline_map: Mapping[str, str],
    snapshot: str,
    series_id: str = SERIES_ID,
    context_record_limit: int | None = None,
    citation_evidence_limit: int | None = None,
    source_citations: Mapping[tuple[str, str | None], Mapping[str, object]] | None = None,
) -> tuple[ContextPackage, ...]:
    repository = RetrievalSearchRepository(connection)
    baseline_config = load_baseline_config(root / BASELINE_CONFIG_PATH)
    contexts = []
    for output_baseline, retrieval_baseline in baseline_map.items():
        kinds = {"B2": ("atomic",), "B3": ("session",), "B4": ("atomic", "session")}[retrieval_baseline]
        for task in TASK_KEYS:
            for case in runtime[task]:
                query = _case_query(task, case)
                request = RetrievalQueryRequest(
                    _query_id(output_baseline, str(case["case_id"])),
                    str(case["user_id"]),
                    query,
                    datetime.fromisoformat(str(case["as_of"])),
                    INDEX_VERSION,
                    kinds,
                    None,
                    (),
                    (),
                    "sensitive",
                    True,
                )
                result = repository.retrieve(
                    request,
                    retrieval_baseline,
                    baseline_config=baseline_config,
                )
                records = _hydrate_context_records(
                    connection,
                    result,
                    source_citations=source_citations or {},
                    citation_evidence_limit=citation_evidence_limit,
                )
                if context_record_limit is not None:
                    records = records[:context_record_limit]
                contexts.append(
                    _context(
                        split,
                        task,
                        case,
                        output_baseline,
                        True,
                        snapshot,
                        records,
                        series_id,
                    )
                )
    return tuple(contexts)


def _context_record_limit(root: Path, config_path: Path | None) -> int | None:
    return _positive_optional_config_value(root, config_path, "answer_context_record_limit")


def _citation_evidence_limit(root: Path, config_path: Path | None) -> int | None:
    return _positive_optional_config_value(root, config_path, "answer_citation_evidence_limit")


def _positive_optional_config_value(root: Path, config_path: Path | None, name: str) -> int | None:
    if config_path is None:
        return None
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or name not in runtime:
        return None
    value = runtime[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QwenMaterializationError(f"{name} must be a positive integer")
    return value


def _hydrate_context_records(
    connection: object,
    result: object,
    *,
    source_citations: Mapping[tuple[str, str | None], Mapping[str, object]],
    citation_evidence_limit: int | None = None,
) -> tuple[dict[str, object], ...]:
    records = []
    for item in result.accepted:
        content = connection.execute(
            """
            SELECT content_text FROM retrieval_index_records
            WHERE user_id = %s AND run_id = %s AND index_record_id = %s
            """,
            (result.user_id, result.snapshot_run_id, item.index_record_id),
        ).fetchone()
        evidence_rows = connection.execute(
            """
            SELECT link.claim_id, link.claim_version_id, source.source_id,
                   span.message_id, span.speaker_id, span.verbatim_quote,
                   link.support_type
            FROM retrieval_index_source_links AS link
            JOIN source_spans AS span
              ON span.user_id = link.user_id AND span.source_id = link.source_id
             AND span.span_id = link.span_id
            JOIN source_events AS source
              ON source.user_id = span.user_id AND source.source_id = span.source_id
            WHERE link.user_id = %s AND link.index_record_id = %s
            ORDER BY link.source_order
            """,
            (result.user_id, item.index_record_id),
        ).fetchall()
        relation_rows = connection.execute(
            """
            SELECT relation_id, source_claim_id, target_claim_id, relation_type, direction
            FROM retrieval_index_relation_links
            WHERE user_id = %s AND index_record_id = %s
            ORDER BY relation_order
            """,
            (result.user_id, item.index_record_id),
        ).fetchall()
        citation_evidence = _citation_evidence_for_record(
            result.user_id,
            item.rank,
            evidence_rows,
            source_citations,
            citation_evidence_limit,
        )
        records.append(
            {
                "rank": item.rank,
                "user_id": result.user_id,
                "record_id": item.index_record_id,
                "record_kind": item.record_kind,
                "content": content[0],
                "lifecycle_statuses": list(item.lifecycle_statuses),
                "claim_ids": list(item.claim_ids),
                "source_ids": list(item.source_ids),
                "evidence": [
                    {
                        "claim_id": row[0],
                        "claim_version_id": row[1],
                        "source_id": row[2],
                        "message_id": row[3],
                        "speaker_id": row[4],
                        "quote": row[5],
                        "support_type": row[6],
                    }
                    for row in evidence_rows
                ],
                "citation_evidence": citation_evidence,
                "relations": [
                    {
                        "relation_id": row[0],
                        "source_claim_id": row[1],
                        "target_claim_id": row[2],
                        "relation_type": row[3],
                        "direction": row[4],
                    }
                    for row in relation_rows
                ],
                "score": item.final_score,
            }
        )
    return tuple(records)


def _source_citation_index(
    sources: Sequence[Mapping[str, object]],
) -> Mapping[tuple[str, str | None], Mapping[str, object]]:
    citations: dict[tuple[str, str | None], Mapping[str, object]] = {}
    for source in sources:
        source_id = str(source["source_id"])
        source_type = str(source["source_type"])
        created_at = str(source["created_at"])
        messages = source.get("messages")
        if isinstance(messages, list) and messages:
            for message in messages:
                message_id = str(message["message_id"])
                citations[(source_id, message_id)] = {
                    "user_id": str(source["user_id"]),
                    "source_id": source_id,
                    "message_id": message_id,
                    "speaker_id": str(message["speaker_id"]),
                    "source_type": source_type,
                    "created_at": created_at,
                    "quote": str(message["text"]),
                }
        else:
            citations[(source_id, None)] = {
                "user_id": str(source["user_id"]),
                "source_id": source_id,
                "message_id": None,
                "speaker_id": str(source["user_id"]),
                "source_type": source_type,
                "created_at": created_at,
                "quote": str(source["content"]),
            }
    return citations


def _citation_evidence_for_record(
    user_id: str,
    rank: int,
    evidence_rows: Sequence[Sequence[object]],
    source_citations: Mapping[tuple[str, str | None], Mapping[str, object]],
    limit: int | None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str | None], dict[str, object]] = {}
    for row in evidence_rows:
        source_id = str(row[2])
        message_id = row[3] if row[3] is None else str(row[3])
        base = source_citations.get((source_id, message_id))
        if base is None:
            base = {
                "user_id": user_id,
                "source_id": source_id,
                "message_id": message_id,
                "speaker_id": str(row[4]),
                "source_type": "unknown",
                "created_at": "",
                "quote": str(row[5]),
            }
        if base.get("source_id") != source_id or base.get("message_id") != message_id:
            raise QwenMaterializationError("source citation identity changed")
        if base.get("user_id") != user_id:
            raise QwenMaterializationError("source citation crossed the user boundary")
        key = (source_id, message_id)
        item = grouped.setdefault(
            key,
            {
                **dict(base),
                "linked_claim_ids": [],
                "support_types": [],
                "originating_record_rank": rank,
            },
        )
        claim_id = str(row[0])
        support_type = str(row[6])
        if claim_id not in item["linked_claim_ids"]:
            item["linked_claim_ids"].append(claim_id)
        if support_type not in item["support_types"]:
            item["support_types"].append(support_type)
    ordered = sorted(
        grouped.values(),
        key=lambda value: (
            int(value["originating_record_rank"]),
            str(value.get("created_at", "")),
            str(value["source_id"]),
            "" if value.get("message_id") is None else str(value["message_id"]),
        ),
    )
    if any(not str(item.get("quote", "")).strip() for item in ordered):
        raise QwenMaterializationError("citation evidence has an empty quote")
    if limit is not None:
        ordered = ordered[:limit]
    return [dict(item) for item in ordered]


def _apply_temporal_lifecycle(
    connection: object,
    claim_ids_by_user: Mapping[str, Sequence[str]],
    as_of: datetime,
    transitioned_at: datetime,
    series_id: str,
) -> None:
    repository = StorageRepository(connection)
    temporal = TemporalService(connection)
    for user_id in sorted(claim_ids_by_user):
        for claim_id in sorted(set(claim_ids_by_user[user_id])):
            claim = repository.get_claim(user_id, claim_id)
            if claim is None:
                raise QwenMaterializationError("persisted claim disappeared")
            open_status = connection.execute(
                """
                SELECT lifecycle_status FROM claim_versions
                WHERE user_id = %s AND claim_id = %s AND transaction_to IS NULL
                """,
                (user_id, claim_id),
            ).fetchone()
            if open_status is None:
                raise QwenMaterializationError("claim has no open lifecycle version")
            if open_status[0] != "candidate":
                continue
            if claim.time_precision == "unknown":
                target = "confirmed"
            elif claim.valid_contains(as_of if claim.time_precision == "timestamp" else as_of.date()):
                target = "current"
            else:
                target = "historical"
            temporal.transition(
                TransitionRequest(
                    user_id,
                    claim_id,
                    f"{series_id}:temporal:{claim_id}",
                    target,
                    "qwen_v2_as_of_lifecycle",
                    transitioned_at,
                )
            )


def _apply_conflicts(
    connection: object,
    root: Path,
    claim_ids_by_user: Mapping[str, Sequence[str]],
    conflict_at: datetime,
    as_of: datetime,
    series_id: str,
) -> tuple[MaterializationFailure, ...]:
    failures = []
    candidate_service = ConflictCandidateService(connection, repo_root=root)
    classifier = ConflictClassifier(repo_root=root)
    relation_service = ConflictRelationService(connection, repo_root=root)
    resolver = BeliefResolutionService(connection, repo_root=root)
    decisions: list[tuple[str, str]] = []
    temporal = TemporalService(connection)
    for user_id in sorted(claim_ids_by_user):
        incoming = tuple(sorted(set(claim_ids_by_user[user_id])))
        if not incoming:
            continue
        pairs = candidate_service.generate(CandidateRequest(user_id, conflict_at, incoming))
        visible = {
            item.claim.claim_id: item
            for item in temporal.query(TemporalQuery(user_id, conflict_at, statuses=ALL_STATUSES))
        }
        for pair in pairs:
            left, right = visible[pair.left_claim_id], visible[pair.right_claim_id]
            source_rows = tuple(
                connection.execute(
                    """
                    SELECT source_id, ingested_at FROM source_events
                    WHERE user_id = %s AND source_id = ANY(%s)
                    ORDER BY source_id
                    """,
                    (user_id, list(pair.source_ids)),
                ).fetchall()
            )
            target = _explicit_target(left, right)
            request = ClassificationRequest(
                CLASSIFIER_VERSION,
                user_id,
                conflict_at,
                pair,
                left,
                right,
                source_rows,
                target,
            )
            decision = classifier.classify(request)
            relation_service.persist(request, decision, conflict_at)
            decisions.append((user_id, decision.decision_id))
    for index, (user_id, decision_id) in enumerate(decisions, 1):
        resolved_at = conflict_at + timedelta(microseconds=index)
        if resolved_at >= as_of:
            raise QwenMaterializationError("conflict resolution crossed case as_of")
        try:
            resolver.resolve(
                ResolutionRequest(
                    user_id,
                    decision_id,
                    conflict_at,
                    None,
                    resolved_at,
                    f"{series_id}:resolution:{decision_id}",
                    RESOLVER_VERSION,
                )
            )
        except Exception as error:
            failures.append(
                MaterializationFailure(
                    "belief_resolution",
                    user_id,
                    decision_id,
                    str(getattr(error, "code", type(error).__name__)),
                )
            )
    return tuple(failures)


def _explicit_target(left: object, right: object) -> str | None:
    left_special = left.claim.epistemic_status in {"corrected", "denied"}
    right_special = right.claim.epistemic_status in {"corrected", "denied"}
    if left_special == right_special:
        return None
    return right.claim.claim_id if left_special else left.claim.claim_id


def _b7_contexts(b6_contexts: Sequence[ContextPackage]) -> tuple[ContextPackage, ...]:
    return tuple(
        ContextPackage(
            item.series_id,
            item.split,
            item.task,
            item.case_id,
            item.user_id,
            item.as_of,
            "B7",
            False,
            "exact_b6_context_before_answerability",
            item.context_records,
            item.context_sha256,
        )
        for item in b6_contexts
    )


def _context(
    split: str,
    task: str,
    case: Mapping[str, object],
    baseline: str,
    provider_call: bool,
    snapshot: str,
    records: Sequence[dict[str, object]],
    series_id: str = SERIES_ID,
) -> ContextPackage:
    canonical_records = tuple(records)
    return ContextPackage(
        series_id,
        split,
        task,
        str(case["case_id"]),
        str(case["user_id"]),
        str(case["as_of"]),
        baseline,
        provider_call,
        snapshot,
        canonical_records,
        sha256(_canonical_bytes(canonical_records)).hexdigest(),
    )


def _source_context_record(source: Mapping[str, object]) -> dict[str, object]:
    evidence = []
    citation_evidence = []
    for message in source["messages"]:
        citation = {
            "user_id": source["user_id"],
            "source_id": source["source_id"],
            "message_id": message["message_id"],
            "speaker_id": message["speaker_id"],
            "source_type": source["source_type"],
            "created_at": source["created_at"],
            "quote": message["text"],
            "support_type": "source_history",
        }
        evidence.append({
            "source_id": citation["source_id"],
            "message_id": citation["message_id"],
            "speaker_id": citation["speaker_id"],
            "quote": citation["quote"],
            "support_type": citation["support_type"],
        })
        citation_evidence.append({
            **citation,
            "linked_claim_ids": [],
            "support_types": ["source_history"],
            "originating_record_rank": 0,
        })
    if not source["messages"]:
        citation = {
            "user_id": source["user_id"],
            "source_id": source["source_id"],
            "message_id": None,
            "speaker_id": source["user_id"],
            "source_type": source["source_type"],
            "created_at": source["created_at"],
            "quote": source["content"],
            "support_type": "source_history",
        }
        evidence.append({
            "source_id": citation["source_id"],
            "message_id": citation["message_id"],
            "speaker_id": citation["speaker_id"],
            "quote": citation["quote"],
            "support_type": citation["support_type"],
        })
        citation_evidence.append({
            **citation,
            "linked_claim_ids": [],
            "support_types": ["source_history"],
            "originating_record_rank": 0,
        })
    return {
        "record_kind": "source",
        "user_id": source["user_id"],
        "source_id": source["source_id"],
        "source_type": source["source_type"],
        "created_at": source["created_at"],
        "content": source["content"],
        "messages": source["messages"],
        "evidence": evidence,
        "citation_evidence": citation_evidence,
    }


def _drop_index_snapshots(connection: object) -> None:
    connection.execute("DELETE FROM retrieval_index_runs")


def _validate_context_accounting(
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    contexts: Sequence[ContextPackage],
) -> None:
    case_count = sum(len(runtime[task]) for task in TASK_KEYS)
    if len(contexts) != case_count * 8:
        raise QwenMaterializationError("B0-B7 context accounting changed")
    by_key = {(item.task, item.case_id, item.baseline_id): item for item in contexts}
    if len(by_key) != len(contexts):
        raise QwenMaterializationError("context identity is duplicated")
    for task in TASK_KEYS:
        for case in runtime[task]:
            key = (task, str(case["case_id"]))
            b6 = by_key[(*key, "B6")]
            b7 = by_key[(*key, "B7")]
            if b6.context_sha256 != b7.context_sha256 or b6.context_records != b7.context_records:
                raise QwenMaterializationError("B6 and B7 contexts differ")
            if b7.provider_call:
                raise QwenMaterializationError("B7 is marked for a provider call")


def _paired_context_identity(contexts: Sequence[ContextPackage], b7: ContextPackage) -> bool:
    return any(
        item.task == b7.task
        and item.case_id == b7.case_id
        and item.baseline_id == "B6"
        and item.context_sha256 == b7.context_sha256
        and item.context_records == b7.context_records
        for item in contexts
    )


def _database_counts(connection: object) -> Mapping[str, int]:
    tables = (
        "memory_users",
        "source_events",
        "claims",
        "claim_versions",
        "conflict_decisions",
        "claim_relations",
        "belief_resolutions",
        "session_summaries",
        "retrieval_index_runs",
        "retrieval_index_records",
    )
    return {table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in tables}


def _case_query(task: str, case: Mapping[str, object]) -> str:
    return str(
        case[
            {"qa": "question", "summary": "instruction", "interactive": "initial_user_message"}[task]
        ]
    )


def _query_id(baseline: str, case_id: str) -> str:
    return f"qv2_{sha256(f'{baseline}:{case_id}'.encode()).hexdigest()}"


def _source_order(source: Mapping[str, object]) -> tuple[str, str]:
    return str(source["created_at"]), str(source["source_id"])


def _context_order(item: ContextPackage) -> tuple[int, int, str]:
    return int(item.baseline_id[1:]), TASK_KEYS.index(item.task), item.case_id


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def _exclusive_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, object]] | object) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
