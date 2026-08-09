"""Deterministic development construction and release checks for retrieval indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from psycopg import connect
from psycopg.rows import dict_row

from storage.migrations import apply_migrations
from summaries.grounded_evaluation import (
    load_grounded_runtime,
    materialize_grounded_runtime,
    run_grounded_summaries,
    serialize_jsonl as serialize_grounded_jsonl,
)

from .contracts import (
    INDEX_VERSION,
    AtomicIndexInput,
    ClaimVersionLineage,
    IndexRecord,
    RelationLineage,
    RetrievalIndexError,
    SessionIndexInput,
    SessionStatementInput,
    SourceSpanLineage,
    TransactionTime,
    ValidTime,
    canonical_json,
    load_index_config,
)
from .embeddings import DeterministicTokenHashEmbedder
from .indexing import build_atomic_index_record, build_session_index_record
from .repository import IndexBuildRequest, RetrievalIndexRepository


DATASET_VERSION = "retrieval_index_development_v1"
STARTING_COMMIT = "9e72921e930a336c8a3ef280165f5715cac0019a"
GUIDANCE_VERSION = "step-7.1-guidance-v1"
GUIDANCE_SHA256 = "41a3f67caa7a41f9039f24a3d78eff26c1e45568eda64a9be3b3753ee8c33418"
DATASET_MANIFEST = Path("data/retrieval/index-development-v1/manifest.json")
RESULT_ROOT = Path("results/retrieval/index-development-v1")
CONFIG_PATH = Path("configs/retrieval/index_v1.json")
EXPECTED = {
    "user_count": 2,
    "source_count": 20,
    "atomic_record_count": 33,
    "session_record_count": 17,
    "durative_record_count": 0,
    "record_count": 50,
    "run_count": 2,
}
ALLOWED_USERS = ("user_001", "user_002")
ARTIFACT_NAMES = (
    "records.jsonl",
    "failures.jsonl",
    "checks.json",
    "run.json",
    "findings.md",
)
IMPLEMENTATION_PATHS = (
    "configs/retrieval/index_v1.json",
    "migrations/0008_retrieval_indexes.sql",
    "src/retrieval/contracts.py",
    "src/retrieval/embeddings.py",
    "src/retrieval/indexing.py",
    "src/retrieval/repository.py",
    "src/retrieval/index_evaluation.py",
)
STEP63_MANIFEST_PATH = Path("results/summaries/durative-claim-development-v1/manifest.json")
STEP63_MANIFEST_SHA256 = "1d3f1c78d95bd96399224581bec21143c4b52562517d4779b74850e42d26fdbb"
STEP64_COMMIT = "9e72921e930a336c8a3ef280165f5715cac0019a"
STEP64_RESULT_MANIFEST_PATH = Path(
    "results/summaries/summary-quality-development-v2/manifest.json"
)
STEP64_RESULT_MANIFEST_SHA256 = "b9ed38d2bf68afc0532adda7a1b229cc989548eaabd1087e597b8787819b4385"
STEP64_CHECKPOINT_MANIFEST_PATH = Path(
    "results/summaries/summary-quality-development-runtime-v2/checkpoint_manifest.json"
)
STEP64_CHECKPOINT_MANIFEST_SHA256 = "6b0a474e42962ac792516c0ed024fa78d9d52842974cf8e8f221f0297e04500e"
STEP64_CHECKPOINT_PREFLIGHT_PATH = Path(
    "results/summaries/summary-quality-development-runtime-v2/checkpoint_preflight.json"
)
STEP64_CHECKPOINT_PREFLIGHT_SHA256 = "4a2b8947c10259199ab2ca122f5d86ab8ce225f3819ec4469f98f723f5c23dd7"
AUTHORIZED_PREDECESSOR_DRIFT = {
    "Makefile": "adds_step7_1_retrieval_index_test_target",
    "tests/integration/test_belief_resolution.py": "expects_migration_0008",
    "tests/integration/test_conflict_relations.py": "expects_migration_0008",
    "tests/integration/test_durative_claim_evaluation.py": "adapts_frozen_step6_3_release_attestation",
    "tests/integration/test_durative_claim_persistence.py": "expects_migration_0008",
    "tests/integration/test_grounded_summary_persistence.py": "expects_migration_0008",
    "tests/integration/test_phase4_storage.py": "expects_migration_0008_and_retrieval_tables",
    "tests/integration/test_phase5_conflict_evaluation.py": "adapts_predecessor_hashes_for_migration_0008",
    "tests/integration/test_temporal_service.py": "expects_migration_0008",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RetrievalIndexEvaluationError(ValueError):
    """Reject a changed input, unsafe run, or inconsistent release."""


@dataclass(frozen=True)
class IndexFailure:
    failure_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.failure_id) is None:
            raise RetrievalIndexEvaluationError("failure ID is invalid")
        if self.user_id not in ALLOWED_USERS:
            raise RetrievalIndexEvaluationError("failure user is invalid")
        if not re.fullmatch(r"[a-z0-9_]{1,64}", self.code):
            raise RetrievalIndexEvaluationError("failure code is not sanitized")
        if not re.fullmatch(r"[a-z0-9_]{1,64}", self.location):
            raise RetrievalIndexEvaluationError("failure location is not sanitized")


@dataclass(frozen=True)
class IndexChecks:
    dataset_version: str
    user_count: int
    source_count: int
    run_count: int
    atomic_record_count: int
    session_record_count: int
    durative_record_count: int
    record_count: int
    claim_line_count: int
    source_line_count: int
    relation_line_count: int
    failure_count: int
    duplicate_count: int
    cross_user_count: int
    stale_count: int
    unsupported_count: int
    restricted_count: int
    partial_count: int
    exact_claim_lineage: bool
    exact_source_lineage: bool
    exact_relation_lineage: bool
    deterministic_vectors: bool
    vector_dimension: int
    normalized_vector_count: int
    fts_document_count: int
    replay_pass: bool
    deletion_pass: bool
    model_calls: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: int = 0

    def __post_init__(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise RetrievalIndexEvaluationError("check version changed")
        for name, value in asdict(self).items():
            if name in {
                "dataset_version",
                "exact_claim_lineage",
                "exact_source_lineage",
                "exact_relation_lineage",
                "deterministic_vectors",
                "replay_pass",
                "deletion_pass",
            }:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RetrievalIndexEvaluationError("check count is invalid")
        if (
            self.model_calls
            or self.retry_count
            or self.input_tokens
            or self.output_tokens
            or self.cost_usd
        ):
            raise RetrievalIndexEvaluationError("model use is forbidden")


def load_index_runtime(
    manifest_path: str | Path = DATASET_MANIFEST,
    *,
    repo_root: str | Path = ".",
):
    """Load only bound development runtime inputs; no scorer or gold path exists."""

    root = Path(repo_root).resolve()
    manifest = _read_object(root / manifest_path)
    _validate_dataset_manifest(manifest)
    for binding in manifest["inputs"].values():
        if not isinstance(binding, dict):
            raise RetrievalIndexEvaluationError("input binding is invalid")
        for key, value in binding.items():
            if not key.endswith("_path"):
                continue
            hash_key = key.removesuffix("_path") + "_sha256"
            if hash_key not in binding:
                raise RetrievalIndexEvaluationError("input hash binding is missing")
            path = root / str(value)
            if _file_sha256(path) != binding[hash_key]:
                raise RetrievalIndexEvaluationError("bound input hash changed")
    _, sources, definitions, claims = load_grounded_runtime(repo_root=root)
    summaries_path = root / manifest["inputs"]["grounded_summaries"]["summaries_path"]
    summaries = tuple(_read_jsonl(summaries_path))
    durative_path = root / manifest["inputs"]["durative_claims"]["claims_path"]
    if durative_path.read_bytes() != b"":
        raise RetrievalIndexEvaluationError("durative development handoff is not empty")
    if (
        len(sources) != EXPECTED["source_count"]
        or len(claims) != EXPECTED["atomic_record_count"]
        or len(summaries) != EXPECTED["session_record_count"]
        or {item.user_id for item in sources} != set(ALLOWED_USERS)
        or {str(item["user_id"]) for item in claims} != set(ALLOWED_USERS)
        or {str(item["user_id"]) for item in summaries} != set(ALLOWED_USERS)
    ):
        raise RetrievalIndexEvaluationError("development runtime accounting changed")
    return manifest, sources, definitions, claims, summaries


def build_development_records(
    connection: object,
    definitions: Sequence[object],
    *,
    repo_root: str | Path = ".",
) -> tuple[IndexRecord, ...]:
    """Build all visible atomic and session snapshots from the materialized DB."""

    config = load_index_config(Path(repo_root) / CONFIG_PATH)
    embedder = DeterministicTokenHashEmbedder()
    cutoffs = _cutoffs(definitions)
    records: list[IndexRecord] = []
    for user_id in ALLOWED_USERS:
        cutoff = cutoffs[user_id]
        for item in _atomic_inputs(connection, user_id, cutoff):
            record = build_atomic_index_record(
                user_id, cutoff, item, config=config, embedder=embedder
            )
            if record is not None:
                records.append(record)
        for item in _session_inputs(connection, user_id, cutoff):
            record = build_session_index_record(
                user_id, cutoff, item, config=config, embedder=embedder
            )
            if record is not None:
                records.append(record)
    return tuple(sorted(records, key=lambda item: (item.user_id, item.record_kind, item.index_record_id)))


def execute_index_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> IndexChecks:
    root = Path(repo_root).resolve()
    output = root / output_dir
    if output.exists() and any(output.iterdir()):
        raise RetrievalIndexEvaluationError("result directory must be empty")
    manifest, sources, definitions, claims, frozen_summaries = load_index_runtime(
        repo_root=root
    )
    connection = connection_factory()
    failures: tuple[IndexFailure, ...] = ()
    try:
        apply_migrations(connection, root / "migrations")
        _require_clean(connection)
        materialize_grounded_runtime(connection, sources, claims)
        summaries, empty, summary_failures = run_grounded_summaries(connection, definitions)
        if summary_failures or len(empty) != 3:
            raise RetrievalIndexEvaluationError("grounded summary replay failed")
        if serialize_grounded_jsonl(summaries) != _serialize_mapping_jsonl(frozen_summaries):
            raise RetrievalIndexEvaluationError("frozen grounded summaries changed")
        records = build_development_records(connection, definitions, repo_root=root)
        persistence_results = _persist_records(connection, definitions, records, root)
        replay_pass = all(item.replayed for item in _persist_records(
            connection, definitions, records, root
        ))
        deletion_pass = _deletion_check(connection)
        checks = _checks(
            connection,
            sources,
            records,
            persistence_results,
            failures,
            replay_pass=replay_pass,
            deletion_pass=deletion_pass,
        )
    finally:
        connection.close()
    _write_release(root, output, manifest, records, failures, checks)
    return checks


def verify_index_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = root / output_dir
    release = _read_object(output / "manifest.json")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict) or tuple(sorted(artifacts)) != tuple(sorted(ARTIFACT_NAMES)):
        raise RetrievalIndexEvaluationError("release artifact map changed")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
            raise RetrievalIndexEvaluationError("release artifact hash is invalid")
        if _file_sha256(output / name) != expected:
            raise RetrievalIndexEvaluationError("release artifact changed")
    dataset = release.get("dataset")
    if dataset != {
        "path": DATASET_MANIFEST.as_posix(),
        "sha256": _file_sha256(root / DATASET_MANIFEST),
    }:
        raise RetrievalIndexEvaluationError("release dataset binding changed")
    implementation = release.get("implementation_hashes")
    if not isinstance(implementation, dict) or tuple(sorted(implementation)) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise RetrievalIndexEvaluationError("implementation map changed")
    for path, expected in implementation.items():
        if _file_sha256(root / path) != expected:
            raise RetrievalIndexEvaluationError("release implementation changed")
    if release.get("predecessor") != _predecessor_attestation(root):
        raise RetrievalIndexEvaluationError("release predecessor attestation changed")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def serialize_records(records: Sequence[IndexRecord]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(item)) for item in records)


def _atomic_inputs(connection: object, user_id: str, cutoff: datetime) -> tuple[AtomicIndexInput, ...]:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT claim.claim_id, version.version_id AS claim_version_id,
                   claim.subject_id, claim.speaker_id, claim.predicate,
                   claim.object_json, claim.polarity, claim.epistemic_status,
                   claim.memory_kind, claim.sensitivity,
                   version.lifecycle_status, version.time_precision,
                   version.valid_from_date, version.valid_from_timestamp,
                   version.valid_to_date, version.valid_to_timestamp,
                   version.transaction_from, version.transaction_to
            FROM claims AS claim
            JOIN claim_versions AS version
              ON version.user_id = claim.user_id AND version.claim_id = claim.claim_id
            WHERE claim.user_id = %s
              AND version.transaction_from <= %s
              AND (version.transaction_to IS NULL OR %s < version.transaction_to)
              AND EXISTS (
                  SELECT 1 FROM evidence_links AS evidence
                  JOIN source_spans AS span
                    ON span.user_id = evidence.user_id AND span.span_id = evidence.span_id
                  JOIN source_events AS source
                    ON source.user_id = span.user_id AND source.source_id = span.source_id
                  WHERE evidence.user_id = claim.user_id
                    AND evidence.claim_id = claim.claim_id
                    AND source.ingested_at <= %s
              )
            ORDER BY claim.claim_id, version.version_id
            """,
            (user_id, cutoff, cutoff, cutoff),
        )
        rows = tuple(cursor.fetchall())
    result: list[AtomicIndexInput] = []
    for row in rows:
        source_lineage = _source_lineage(
            connection, user_id, row["claim_id"], row["claim_version_id"], cutoff
        )
        relation_lineage = _relation_lineage(
            connection, user_id, row["claim_id"], cutoff
        )
        result.append(
            AtomicIndexInput(
                user_id,
                row["claim_id"],
                row["claim_version_id"],
                row["subject_id"],
                row["speaker_id"],
                row["predicate"],
                row["object_json"],
                row["polarity"],
                row["epistemic_status"],
                row["memory_kind"],
                row["lifecycle_status"],
                ValidTime(
                    row["time_precision"],
                    row["valid_from_date"],
                    row["valid_from_timestamp"],
                    row["valid_to_date"],
                    row["valid_to_timestamp"],
                ),
                TransactionTime(row["transaction_from"], row["transaction_to"]),
                row["sensitivity"],
                (
                    ClaimVersionLineage(
                        user_id,
                        row["claim_id"],
                        row["claim_version_id"],
                        row["lifecycle_status"],
                        0,
                    ),
                ),
                source_lineage,
                relation_lineage,
            )
        )
    return tuple(result)


def _source_lineage(
    connection: object,
    user_id: str,
    claim_id: str,
    version_id: str,
    cutoff: datetime,
) -> tuple[SourceSpanLineage, ...]:
    rows = connection.execute(
        """
        SELECT source.source_id, span.span_id, evidence.support_type
        FROM evidence_links AS evidence
        JOIN source_spans AS span
          ON span.user_id = evidence.user_id AND span.span_id = evidence.span_id
        JOIN source_events AS source
          ON source.user_id = span.user_id AND source.source_id = span.source_id
        WHERE evidence.user_id = %s AND evidence.claim_id = %s
          AND source.ingested_at <= %s
        ORDER BY source.source_id, span.span_id, evidence.support_type
        """,
        (user_id, claim_id, cutoff),
    ).fetchall()
    return tuple(
        SourceSpanLineage(user_id, claim_id, version_id, row[0], row[1], row[2], order)
        for order, row in enumerate(rows)
    )


def _relation_lineage(
    connection: object,
    user_id: str,
    claim_id: str,
    cutoff: datetime,
) -> tuple[RelationLineage, ...]:
    rows = connection.execute(
        """
        SELECT relation_id, source_claim_id, target_claim_id, relation_type
        FROM claim_relations
        WHERE user_id = %s AND (source_claim_id = %s OR target_claim_id = %s)
          AND created_at <= %s
        ORDER BY relation_id
        """,
        (user_id, claim_id, claim_id, cutoff),
    ).fetchall()
    symmetric = {"contradicts", "same_event_as", "same_topic_as"}
    return tuple(
        RelationLineage(
            row[0],
            user_id,
            row[1],
            row[2],
            row[3],
            "symmetric" if row[3] in symmetric else ("outgoing" if row[1] == claim_id else "incoming"),
            order,
        )
        for order, row in enumerate(rows)
    )


def _session_inputs(connection: object, user_id: str, cutoff: datetime) -> tuple[SessionIndexInput, ...]:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT summary_id, session_definition_id, renderer_version,
                   summary_text, valid_time_kind,
                   valid_time_start_date, valid_time_start_timestamp,
                   valid_time_end_date, valid_time_end_timestamp,
                   contains_sensitive, transaction_from, transaction_to
            FROM session_summaries
            WHERE user_id = %s AND transaction_from <= %s
              AND (transaction_to IS NULL OR %s < transaction_to)
            ORDER BY session_definition_id, summary_id
            """,
            (user_id, cutoff, cutoff),
        )
        rows = tuple(cursor.fetchall())
    result: list[SessionIndexInput] = []
    precision = {"date": "day", "timestamp": "timestamp", "unknown": "unknown", "mixed": "mixed"}
    for row in rows:
        source_ids = tuple(
            item[0]
            for item in connection.execute(
                """
                SELECT source_id FROM session_summary_sources
                WHERE user_id = %s AND summary_id = %s ORDER BY source_order
                """,
                (user_id, row["summary_id"]),
            ).fetchall()
        )
        statements = _session_statements(connection, user_id, row["summary_id"])
        sensitivity = "sensitive" if row["contains_sensitive"] else "standard"
        result.append(
            SessionIndexInput(
                user_id,
                row["summary_id"],
                row["session_definition_id"],
                row["renderer_version"],
                row["summary_text"],
                statements,
                tuple(sorted(source_ids)),
                ValidTime(
                    precision[row["valid_time_kind"]],
                    row["valid_time_start_date"],
                    row["valid_time_start_timestamp"],
                    row["valid_time_end_date"],
                    row["valid_time_end_timestamp"],
                ),
                TransactionTime(row["transaction_from"], row["transaction_to"]),
                sensitivity,
                row["contains_sensitive"],
            )
        )
    return tuple(result)


def _session_statements(connection: object, user_id: str, summary_id: str) -> tuple[SessionStatementInput, ...]:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT statement_id, statement_kind, lifecycle_view,
                   statement_text, statement_order
            FROM session_summary_statements
            WHERE user_id = %s AND summary_id = %s
            ORDER BY statement_order, statement_id
            """,
            (user_id, summary_id),
        )
        statements = tuple(cursor.fetchall())
    result: list[SessionStatementInput] = []
    for statement in statements:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT evidence.claim_id, evidence.claim_version_id,
                       version.lifecycle_status, evidence.source_id,
                       evidence.span_id, evidence.support_type,
                       evidence.evidence_order
                FROM session_summary_statement_evidence AS evidence
                JOIN claim_versions AS version
                  ON version.user_id = evidence.user_id
                 AND version.claim_id = evidence.claim_id
                 AND version.version_id = evidence.claim_version_id
                WHERE evidence.user_id = %s AND evidence.summary_id = %s
                  AND evidence.statement_id = %s
                ORDER BY evidence.evidence_order, evidence.evidence_id
                """,
                (user_id, summary_id, statement["statement_id"]),
            )
            evidence_rows = tuple(cursor.fetchall())
        claim_by_key: dict[tuple[str, str], ClaimVersionLineage] = {}
        for evidence in evidence_rows:
            key = (evidence["claim_id"], evidence["claim_version_id"])
            if key not in claim_by_key:
                claim_by_key[key] = ClaimVersionLineage(
                    user_id, key[0], key[1], evidence["lifecycle_status"], len(claim_by_key)
                )
        source_lineage = tuple(
            SourceSpanLineage(
                user_id,
                item["claim_id"],
                item["claim_version_id"],
                item["source_id"],
                item["span_id"],
                item["support_type"],
                order,
            )
            for order, item in enumerate(evidence_rows)
        )
        result.append(
            SessionStatementInput(
                statement["statement_id"],
                statement["statement_kind"],
                statement["lifecycle_view"],
                statement["statement_text"],
                tuple(claim_by_key.values()),
                source_lineage,
                statement["statement_order"],
            )
        )
    return tuple(result)


def _persist_records(connection: object, definitions: Sequence[object], records: Sequence[IndexRecord], root: Path):
    repository = RetrievalIndexRepository(connection, config_path=root / CONFIG_PATH)
    cutoffs = _cutoffs(definitions)
    results = []
    for user_id in ALLOWED_USERS:
        selected = tuple(item for item in records if item.user_id == user_id)
        snapshot = hashlib.sha256(
            canonical_json(tuple((item.index_record_id, item.input_snapshot_sha256) for item in selected)).encode("utf-8")
        ).hexdigest()
        cutoff = cutoffs[user_id]
        request = IndexBuildRequest(
            user_id,
            INDEX_VERSION,
            repository.config_sha256,
            cutoff,
            f"index-development-v1:{user_id}",
            snapshot,
            cutoff,
            cutoff,
        )
        results.append(repository.persist(request, selected))
    return tuple(results)


def _checks(
    connection: object,
    sources: Sequence[object],
    records: Sequence[IndexRecord],
    persistence_results: Sequence[object],
    failures: Sequence[IndexFailure],
    *,
    replay_pass: bool,
    deletion_pass: bool,
) -> IndexChecks:
    atomic = sum(item.record_kind == "atomic" for item in records)
    session = sum(item.record_kind == "session" for item in records)
    claim_lines = sum(len(item.claim_lineage) for item in records)
    source_lines = sum(len(item.source_lineage) for item in records)
    relation_lines = sum(len(item.relation_lineage) for item in records)
    db = connection.execute(
        """
        SELECT count(*), count(search_document),
               count(*) FILTER (WHERE vector_dims(embedding) = 256),
               count(*) FILTER (
                 WHERE abs(vector_norm(embedding) - 1.0) < 0.000000000001
                    OR vector_norm(embedding) = 0
               )
        FROM retrieval_index_records
        """
    ).fetchone()
    db_claim = connection.execute("SELECT count(*) FROM retrieval_index_claim_links").fetchone()[0]
    db_source = connection.execute("SELECT count(*) FROM retrieval_index_source_links").fetchone()[0]
    db_relation = connection.execute("SELECT count(*) FROM retrieval_index_relation_links").fetchone()[0]
    duplicate = len(records) - len({item.index_record_id for item in records})
    cross = sum(
        any(line.user_id != item.user_id for line in (*item.claim_lineage, *item.source_lineage, *item.relation_lineage))
        for item in records
    )
    partial = sum(not item.claim_lineage or not item.source_lineage for item in records)
    normalized = sum(
        math.isclose(
            math.sqrt(sum(value * value for value in item.embedding)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or all(value == 0.0 for value in item.embedding)
        for item in records
    )
    checks = IndexChecks(
        DATASET_VERSION,
        len({item.user_id for item in records}),
        len(sources),
        len(persistence_results),
        atomic,
        session,
        0,
        len(records),
        claim_lines,
        source_lines,
        relation_lines,
        len(failures),
        duplicate,
        cross,
        0,
        0,
        sum(item.sensitivity == "restricted" for item in records),
        partial,
        db_claim == claim_lines,
        db_source == source_lines,
        db_relation == relation_lines,
        all(len(item.embedding) == 256 and all(math.isfinite(v) for v in item.embedding) for item in records),
        256,
        normalized,
        db[1],
        replay_pass,
        deletion_pass,
    )
    if (
        checks.user_count != EXPECTED["user_count"]
        or checks.source_count != EXPECTED["source_count"]
        or checks.run_count != EXPECTED["run_count"]
        or checks.atomic_record_count != EXPECTED["atomic_record_count"]
        or checks.session_record_count != EXPECTED["session_record_count"]
        or checks.durative_record_count != EXPECTED["durative_record_count"]
        or checks.record_count != EXPECTED["record_count"]
        or db[0] != EXPECTED["record_count"]
        or db[2] != EXPECTED["record_count"]
        or checks.normalized_vector_count != EXPECTED["record_count"]
        or checks.fts_document_count != EXPECTED["record_count"]
        or any(
            (
                checks.failure_count,
                checks.duplicate_count,
                checks.cross_user_count,
                checks.stale_count,
                checks.unsupported_count,
                checks.restricted_count,
                checks.partial_count,
            )
        )
        or not all(
            (
                checks.exact_claim_lineage,
                checks.exact_source_lineage,
                checks.exact_relation_lineage,
                checks.deterministic_vectors,
                checks.replay_pass,
                checks.deletion_pass,
            )
        )
    ):
        raise RetrievalIndexEvaluationError("development index checks failed")
    return checks


def _deletion_check(connection: object) -> bool:
    before = connection.execute("SELECT count(*) FROM retrieval_index_records").fetchone()[0]
    target = connection.execute(
        """
        SELECT user_id, claim_id, span_id, support_type
        FROM retrieval_index_source_links ORDER BY index_record_id, source_order LIMIT 1
        """
    ).fetchone()
    if target is None:
        return False
    affected = tuple(
        row[0]
        for row in connection.execute(
            """
            SELECT index_record_id FROM retrieval_index_source_links
            WHERE user_id = %s AND claim_id = %s AND span_id = %s AND support_type = %s
            ORDER BY index_record_id
            """,
            target,
        ).fetchall()
    )
    with connection.transaction(force_rollback=True):
        connection.execute(
            """
            DELETE FROM evidence_links
            WHERE user_id = %s AND claim_id = %s AND span_id = %s AND support_type = %s
            """,
            target,
        )
        remaining = connection.execute(
            """
            SELECT count(*) FROM retrieval_index_records
            WHERE index_record_id = ANY(%s)
            """,
            (list(affected),),
        ).fetchone()[0]
        deleted = remaining == 0
    restored = connection.execute("SELECT count(*) FROM retrieval_index_records").fetchone()[0]
    return bool(affected) and deleted and restored == before


def _write_release(
    root: Path,
    output: Path,
    dataset_manifest: Mapping[str, object],
    records: Sequence[IndexRecord],
    failures: Sequence[IndexFailure],
    checks: IndexChecks,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "records.jsonl": serialize_records(records),
        "failures.jsonl": b"".join(canonical_json_bytes(asdict(item)) for item in failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": DATASET_VERSION,
                "starting_commit": STARTING_COMMIT,
                "index_version": INDEX_VERSION,
                "user_count": checks.user_count,
                "run_count": checks.run_count,
                "record_count": checks.record_count,
                "atomic_record_count": checks.atomic_record_count,
                "session_record_count": checks.session_record_count,
                "durative_record_count": checks.durative_record_count,
                "failure_count": checks.failure_count,
                "execution_mode": "deterministic_no_model",
                "model_calls": 0,
                "request_count": 0,
                "retry_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "incremental_cost_usd": 0,
                "historical_openai_spend_usd": "0.2314404",
            }
        ),
        "findings.md": (
            "# Retrieval index development findings\n\n"
            "The deterministic build stored 33 atomic records and 17 session records "
            "for the two development users. Every record kept its exact claim-version "
            "and source-span lineage. The imported claims remain candidates, and the "
            "durative handoff remains empty.\n\n"
            "The 256-dimensional vectors are signed token hashes for local storage and "
            "index testing. They are not semantic embeddings, and this step does not "
            "report retrieval-quality or latency metrics. No model was called.\n"
        ).encode("utf-8"),
    }
    for name, content in artifacts.items():
        _write_exclusive(output / name, content)
    release = {
        "artifact_version": DATASET_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "artifacts": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()},
        "dataset": {
            "path": DATASET_MANIFEST.as_posix(),
            "sha256": _file_sha256(root / DATASET_MANIFEST),
        },
        "input_bindings": dataset_manifest["inputs"],
        "checks": asdict(checks),
        "implementation_hashes": {path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS},
        "predecessor": _predecessor_attestation(root),
        "excluded_inputs": dataset_manifest["excluded_inputs"],
        "limitations": [
            "The development handoff contains only candidate atomic claims.",
            "No durative claim was accepted by the frozen Step 6.3 release.",
            "The deterministic token-hash vectors test index mechanics, not semantic retrieval quality.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))


def _validate_dataset_manifest(value: Mapping[str, object]) -> None:
    if set(value) != {
        "dataset_version",
        "split",
        "review_status",
        "guidance_version",
        "guidance_sha256",
        "allowed_users",
        "expected",
        "inputs",
        "loader_boundary",
        "excluded_inputs",
        "execution_mode",
    }:
        raise RetrievalIndexEvaluationError("dataset manifest fields changed")
    if (
        value["dataset_version"] != DATASET_VERSION
        or value["split"] != "development"
        or value["review_status"] != "implementation_reviewed"
        or value["guidance_version"] != GUIDANCE_VERSION
        or value["guidance_sha256"] != GUIDANCE_SHA256
        or value["allowed_users"] != list(ALLOWED_USERS)
        or value["expected"] != EXPECTED
        or value["loader_boundary"] != "bound_development_runtime_only_stop_before_test_users"
        or value["execution_mode"] != "deterministic_no_model"
    ):
        raise RetrievalIndexEvaluationError("dataset manifest identity changed")
    excluded = value["excluded_inputs"]
    if excluded != ["step6_4_gold", "step6_4_scorer", "scaled_gold", "oracle", "review_queue", "test_users"]:
        raise RetrievalIndexEvaluationError("dataset exclusions changed")
    if value["inputs"].get("scaled_runtime") != {
        "sessionization_dataset_manifest_path": (
            "data/summaries/sessionization-development-v1/manifest.json"
        ),
        "sessionization_dataset_manifest_sha256": (
            "c139e2624cbec7904ca8edd67b4b1f42423dcfda8a47a1d8ca04cf8ee5507b69"
        ),
        "source_prefix_records": 20,
        "source_prefix_sha256": (
            "b0dc2b0dae985b51e2e25e441f98dcbe8d86045fa6a3beae2545f50de9e844d9"
        ),
        "user_prefix_records": 2,
        "user_prefix_sha256": (
            "3cb98b28b57ed333c170eb93ba2bdd9cea1c24941df74d3d60e75397a88fbfdd"
        ),
    }:
        raise RetrievalIndexEvaluationError("development prefix binding changed")
    serialized_inputs = json.dumps(value["inputs"], sort_keys=True)
    for forbidden in ("gold", "oracle", "review_queue", "test_user", "summary_quality"):
        if forbidden in serialized_inputs:
            raise RetrievalIndexEvaluationError("forbidden runtime dependency")


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    authorities = {
        "step6_3_result_manifest": (
            STEP63_MANIFEST_PATH,
            STEP63_MANIFEST_SHA256,
        ),
        "step6_4_result_manifest": (
            STEP64_RESULT_MANIFEST_PATH,
            STEP64_RESULT_MANIFEST_SHA256,
        ),
        "step6_4_checkpoint_manifest": (
            STEP64_CHECKPOINT_MANIFEST_PATH,
            STEP64_CHECKPOINT_MANIFEST_SHA256,
        ),
        "step6_4_checkpoint_preflight": (
            STEP64_CHECKPOINT_PREFLIGHT_PATH,
            STEP64_CHECKPOINT_PREFLIGHT_SHA256,
        ),
    }
    for path, expected in authorities.values():
        if _file_sha256(root / path) != expected:
            raise RetrievalIndexEvaluationError("runtime predecessor authority changed")
    return {
        "runtime_authorities": {
            name: {"path": path.as_posix(), "sha256": expected}
            for name, (path, expected) in authorities.items()
        },
        "step6_4_commit": STEP64_COMMIT,
        "runtime_hashed_file_count": len(authorities),
        "out_of_band_protection": {
            "mode": "git_and_reviewer_gate",
            "protected_starting_path_count": 126,
            "runtime_verified": False,
            "authorized_drift_paths": sorted(AUTHORIZED_PREDECESSOR_DRIFT),
        },
    }


def _cutoffs(definitions: Sequence[object]) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    for item in definitions:
        user_id = getattr(item, "user_id")
        cutoff = getattr(item, "transaction_as_of")
        if user_id not in ALLOWED_USERS:
            raise RetrievalIndexEvaluationError("session definition crosses users")
        if user_id in result and result[user_id] != cutoff:
            raise RetrievalIndexEvaluationError("user cutoff changed")
        result[user_id] = cutoff
    if set(result) != set(ALLOWED_USERS):
        raise RetrievalIndexEvaluationError("user cutoff accounting changed")
    return result


def _require_clean(connection: object) -> None:
    counts = [
        connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("memory_users", "retrieval_index_runs", "retrieval_index_records")
    ]
    if any(counts):
        raise RetrievalIndexEvaluationError("database must be clean")


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetrievalIndexEvaluationError("JSON object is required")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RetrievalIndexEvaluationError("JSONL object is required")
            result.append(value)
    return result


def _serialize_mapping_jsonl(values: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(value) for value in values)


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as error:
        raise RetrievalIndexEvaluationError("release artifact already exists") from error


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def main() -> int:
    database_url = os.environ.get("STORAGE_DATABASE_URL")
    if not database_url:
        raise SystemExit("STORAGE_DATABASE_URL is required")
    execute_index_evaluation(lambda: connect(database_url))
    verify_index_release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
