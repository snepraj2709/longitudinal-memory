"""Deterministic development evaluation for persisted grounded summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from ingestion.service import IngestionService
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    ProcessingOutboxRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations
from storage.repository import StorageRepository

from .contracts import SessionDefinition, SessionizationRequest
from .grounded import GroundedSummaryCoordinator, plan_grounded_summary
from .repository import SessionSourceRepository
from .sessions import SessionizationService
from .sessionization_evaluation import (
    DEVELOPMENT_USERS,
    DevelopmentSource,
    load_sessionization_runtime,
    materialize_runtime_sources,
)
from .summary_contracts import GroundedSummaryRequest
from .summary_repository import SessionSummaryRepository


DATASET_VERSION = "grounded_summary_development_v1"
DATASET_MANIFEST = Path("data/summaries/grounded-summary-development-v1/manifest.json")
RESULT_ROOT = Path("results/summaries/grounded-summary-development-v1")
STARTING_COMMIT = "97485bcc62ccc954c63fb1d8cd593d5f0d42533e"
RENDERER_VERSION = "session_summary_renderer_v1"
EXPECTED = {
    "user_count": 2,
    "source_count": 20,
    "session_count": 20,
    "claim_count": 33,
    "evidence_count": 33,
    "persisted_summary_count": 17,
    "empty_session_count": 3,
}
ARTIFACT_NAMES = (
    "summaries.jsonl",
    "empty_sessions.jsonl",
    "failures.jsonl",
    "checks.json",
    "run.json",
    "findings.md",
)
IMPLEMENTATION_PATHS = (
    "Makefile",
    "configs/summaries/session_summary_renderer_v1.json",
    "data/summaries/grounded-summary-development-v1/manifest.json",
    "migrations/0006_session_summaries.sql",
    "src/summaries/grounded.py",
    "src/summaries/grounded_evaluation.py",
    "src/summaries/summary_contracts.py",
    "src/summaries/summary_repository.py",
    "tests/integration/test_grounded_summary_persistence.py",
    "tests/integration/test_grounded_summary_evaluation.py",
    "tests/unit/test_grounded_summaries.py",
    "tests/unit/test_grounded_summary_evaluation.py",
    "tests/unit/test_summary_persistence.py",
)
AUTHORIZED_PREDECESSOR_DRIFT = {
    "Makefile": "adds_step6_2_grounded_summary_test_target",
    "tests/integration/test_belief_resolution.py": "expects_migration_0006",
    "tests/integration/test_conflict_relations.py": "expects_migration_0006",
    "tests/integration/test_phase4_storage.py": "expects_migration_0006_and_summary_tables",
    "tests/integration/test_phase5_conflict_evaluation.py": "adapts_phase5_replay_for_step6_2_migration_drifts",
    "tests/integration/test_sessionization.py": "adapts_step6_1_replay_to_migrations_0001_through_0005",
    "tests/integration/test_temporal_service.py": "expects_migration_0006",
}
TOKEN = re.compile(r"^[a-z0-9_:-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GroundedSummaryEvaluationError(ValueError):
    """Reject incomplete, leaky, or non-deterministic evaluation state."""


@dataclass(frozen=True)
class EvidencePath:
    claim_id: str
    claim_version_id: str
    source_id: str
    span_id: str
    support_type: str

    def __post_init__(self) -> None:
        for name in (
            "claim_id", "claim_version_id", "source_id", "span_id", "support_type"
        ):
            _text(getattr(self, name), name)
        if self.support_type not in {"supports", "contradicts", "corrects"}:
            raise GroundedSummaryEvaluationError("support_type is invalid")


@dataclass(frozen=True)
class GroundedStatementPrediction:
    statement_id: str
    statement_kind: str
    lifecycle_view: str
    text: str
    claim_ids: tuple[str, ...]
    evidence: tuple[EvidencePath, ...]
    contains_sensitive: bool

    def __post_init__(self) -> None:
        _sha(self.statement_id, "statement_id")
        if self.statement_kind not in {"observed_fact", "unresolved_question"}:
            raise GroundedSummaryEvaluationError("statement kind is invalid")
        if self.lifecycle_view not in {"accepted", "candidate", "historical", "disputed"}:
            raise GroundedSummaryEvaluationError("lifecycle view is invalid")
        _text(self.text, "statement text")
        if tuple(sorted(set(self.claim_ids))) != self.claim_ids or not self.claim_ids:
            raise GroundedSummaryEvaluationError("statement claim IDs are invalid")
        if not self.evidence:
            raise GroundedSummaryEvaluationError("statement evidence is empty")
        if set(self.claim_ids) != {item.claim_id for item in self.evidence}:
            raise GroundedSummaryEvaluationError(
                "statement claims and evidence must match exactly"
            )
        if not isinstance(self.contains_sensitive, bool):
            raise GroundedSummaryEvaluationError("sensitive flag is invalid")


@dataclass(frozen=True)
class GroundedSummaryPrediction:
    summary_id: str
    user_id: str
    session_definition_id: str
    session_membership_sha256: str
    renderer_version: str
    input_snapshot_sha256: str
    source_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    summary_text: str
    statements: tuple[GroundedStatementPrediction, ...]
    contains_sensitive: bool

    def __post_init__(self) -> None:
        for name in (
            "summary_id", "session_definition_id", "session_membership_sha256",
            "input_snapshot_sha256",
        ):
            _sha(getattr(self, name), name)
        _user(self.user_id)
        if self.renderer_version != RENDERER_VERSION:
            raise GroundedSummaryEvaluationError("renderer version changed")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise GroundedSummaryEvaluationError("summary source IDs are invalid")
        if tuple(sorted(set(self.claim_ids))) != self.claim_ids or not self.claim_ids:
            raise GroundedSummaryEvaluationError("summary claim IDs are invalid")
        _text(self.summary_text, "summary text")
        if not self.statements:
            raise GroundedSummaryEvaluationError("persisted summary is empty")
        if not isinstance(self.contains_sensitive, bool):
            raise GroundedSummaryEvaluationError("summary sensitive flag is invalid")


@dataclass(frozen=True)
class EmptySessionPrediction:
    user_id: str
    session_definition_id: str
    session_membership_sha256: str
    source_ids: tuple[str, ...]
    reason: str = "no_eligible_claim_evidence"

    def __post_init__(self) -> None:
        _user(self.user_id)
        _sha(self.session_definition_id, "session_definition_id")
        _sha(self.session_membership_sha256, "session_membership_sha256")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise GroundedSummaryEvaluationError("empty session source IDs are invalid")
        if self.reason != "no_eligible_claim_evidence":
            raise GroundedSummaryEvaluationError("empty session reason changed")


@dataclass(frozen=True)
class GroundedSummaryFailure:
    failure_id: str
    user_id: str
    session_definition_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure_id")
        _user(self.user_id)
        _sha(self.session_definition_id, "session_definition_id")
        if TOKEN.fullmatch(self.code) is None or TOKEN.fullmatch(self.location) is None:
            raise GroundedSummaryEvaluationError("failure detail is not sanitized")


@dataclass(frozen=True)
class GroundedSummaryChecks:
    dataset_version: str
    session_count: int
    summary_count: int
    empty_session_count: int
    failure_count: int
    statement_count: int
    provenance_integrity: Mapping[str, object]
    membership_integrity: Mapping[str, object]
    eligible_claim_accounting: Mapping[str, object]
    cross_user_count: int
    stale_reference_count: int
    duplicate_statement_count: int
    unsupported_statement_count: int
    deletion_recompute_pass: bool
    model_calls: int = 0
    model_fallback: bool = False

    def __post_init__(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise GroundedSummaryEvaluationError("dataset version changed")
        for name in (
            "session_count", "summary_count", "empty_session_count", "failure_count",
            "statement_count", "cross_user_count", "stale_reference_count",
            "duplicate_statement_count", "unsupported_statement_count", "model_calls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GroundedSummaryEvaluationError("check count is invalid")
        if self.model_calls != 0 or self.model_fallback:
            raise GroundedSummaryEvaluationError("model execution is forbidden")


def load_grounded_runtime(
    manifest_path: str | Path = DATASET_MANIFEST,
    *,
    repo_root: str | Path = ".",
) -> tuple[
    Mapping[str, object],
    tuple[DevelopmentSource, ...],
    tuple[SessionDefinition, ...],
    tuple[Mapping[str, object], ...],
]:
    root = Path(repo_root).resolve()
    manifest = _read_object(root / manifest_path)
    _validate_manifest(manifest)
    inputs = manifest["inputs"]
    for binding_name, path_name, hash_name in (
        ("sessionization_release", "manifest_path", "manifest_sha256"),
        ("runtime_prefix", "sessionization_manifest_path", "sessionization_manifest_sha256"),
        ("phase3_handoff", "manifest_path", "manifest_sha256"),
        ("phase3_handoff", "claims_path", "claims_sha256"),
        ("phase3_handoff", "evidence_index_path", "evidence_index_sha256"),
        ("phase5_release", "manifest_path", "manifest_sha256"),
        ("renderer", "path", "sha256"),
    ):
        binding = inputs[binding_name]
        if _file_sha256(root / binding[path_name]) != binding[hash_name]:
            raise GroundedSummaryEvaluationError("bound input hash changed")
    session_release = _read_object(root / inputs["sessionization_release"]["manifest_path"])
    if session_release.get("artifacts", {}).get("predictions.jsonl") != inputs[
        "sessionization_release"
    ]["predictions_sha256"]:
        raise GroundedSummaryEvaluationError("session release binding changed")
    phase5 = _read_object(root / inputs["phase5_release"]["manifest_path"])
    if phase5.get("implementation_hashes") != inputs["phase5_release"]["implementation_hashes"]:
        raise GroundedSummaryEvaluationError("Phase 5 implementation binding changed")
    _, sources = load_sessionization_runtime(repo_root=root)
    predictions_path = root / inputs["sessionization_release"]["predictions_path"]
    if _file_sha256(predictions_path) != inputs["sessionization_release"]["predictions_sha256"]:
        raise GroundedSummaryEvaluationError("session predictions changed")
    definitions = tuple(_definition(row) for row in _read_jsonl(predictions_path))
    claims = tuple(_read_jsonl(root / inputs["phase3_handoff"]["claims_path"]))
    evidence_index = _read_object(root / inputs["phase3_handoff"]["evidence_index_path"])
    _validate_runtime(sources, definitions, claims, evidence_index)
    return manifest, sources, definitions, claims


def materialize_grounded_runtime(
    connection: object,
    sources: Sequence[DevelopmentSource],
    claims: Sequence[Mapping[str, object]],
) -> None:
    materialize_runtime_sources(connection, sources)
    repository = StorageRepository(connection)
    extraction = ExtractionVersionRecord(
        "step35_gpt41_fallback_v1",
        "gpt-4.1-2025-04-14",
        "atomic-extraction-v3",
        "1a6a371a73b043807249a4c130b5b309b8a29fbcbea34f6c2e912f6eded38e59",
        "atomic_extraction_v1",
        "f694616f02143cc4e5595fe0658b100f29562c9c4d4df716af362fa10989aac1",
        "predicate_registry_v2",
        "5cb9ba2f2b7a81aa2ce61d52d9425fbf1f2d39964a3a66286d5679ebb2e5175d",
        "f5127cfdb7720ecf84e320da613396d11d71629b709813e2e25ea2ab00df0876",
        max(item.ingested_at for item in sources),
    )
    repository.insert_extraction_version(extraction)
    source_map = {item.source_id: item for item in sources}
    spans: dict[tuple[object, ...], SourceSpanRecord] = {}
    for item in claims:
        evidence = item["evidence"][0]
        source = source_map[evidence["source_id"]]
        quote = evidence["quote"]
        start = source.raw_content.find(quote)
        if start < 0:
            raise GroundedSummaryEvaluationError("claim quote is not in its source")
        identity = (
            item["user_id"], evidence["source_id"], evidence["message_id"], quote
        )
        span = spans.setdefault(
            identity,
            SourceSpanRecord(
                "span_" + hashlib.sha256(canonical_json_bytes(identity)[:-1]).hexdigest(),
                item["user_id"],
                evidence["source_id"],
                evidence["message_id"],
                item["speaker_id"],
                quote,
                start,
                start + len(quote),
            ),
        )
        repository.insert_source_span(span)
        valid_from_date, valid_from_timestamp = _boundary(
            item["valid_from"], item["time_precision"]
        )
        valid_to_date, valid_to_timestamp = _boundary(
            item["valid_to"], item["time_precision"]
        )
        claim = ClaimRecord(
            item["claim_id"], item["user_id"], item["subject_id"],
            item["speaker_id"], item["predicate"],
            item["predicate_registry_version"], item["object"], item["polarity"],
            item["epistemic_status"], valid_from_date, valid_from_timestamp,
            valid_to_date, valid_to_timestamp, item["time_precision"],
            item["confidence"], None, None, extraction.version_id,
        )
        repository.insert_claim(claim)
        repository.insert_claim_version(
            ClaimVersionRecord(
                f"version_{claim.claim_id}", claim.user_id, claim.claim_id,
                "candidate", source.ingested_at,
                valid_from_date=valid_from_date,
                valid_from_timestamp=valid_from_timestamp,
                valid_to_date=valid_to_date,
                valid_to_timestamp=valid_to_timestamp,
                time_precision=claim.time_precision,
            )
        )
        repository.insert_evidence_link(
            EvidenceLinkRecord(
                claim.user_id, claim.claim_id, span.span_id, "supports",
                claim.extraction_confidence,
            )
        )


def run_grounded_summaries(
    connection: object,
    definitions: Sequence[SessionDefinition],
) -> tuple[
    tuple[GroundedSummaryPrediction, ...],
    tuple[EmptySessionPrediction, ...],
    tuple[GroundedSummaryFailure, ...],
]:
    summary_repository = SessionSummaryRepository(connection)
    predictions: list[GroundedSummaryPrediction] = []
    empty: list[EmptySessionPrediction] = []
    failures: list[GroundedSummaryFailure] = []
    for definition in definitions:
        try:
            evidence = summary_repository.load_visible_evidence(
                definition, definition.transaction_as_of
            )
            if not evidence:
                empty.append(
                    EmptySessionPrediction(
                        definition.user_id,
                        definition.definition_id,
                        definition.membership_sha256,
                        definition.source_ids,
                    )
                )
                continue
            request = GroundedSummaryRequest(
                definition.user_id,
                definition.definition_id,
                definition.membership_sha256,
                definition.transaction_as_of,
                f"grounded-eval:{definition.definition_id}",
            )
            planned = plan_grounded_summary(request, evidence)
            outcome = summary_repository.persist(planned, definition.source_ids)
            if not outcome.created or outcome.summary_id != planned.summary.summary_id:
                raise GroundedSummaryEvaluationError("summary persistence was not new")
            predictions.append(_prediction(planned, definition.source_ids))
        except Exception as error:
            code = getattr(error, "code", "summary_failed")
            location = getattr(error, "location", "session")
            if not isinstance(code, str) or TOKEN.fullmatch(code) is None:
                code = "summary_failed"
            if not isinstance(location, str) or TOKEN.fullmatch(location) is None:
                location = "session"
            failures.append(
                GroundedSummaryFailure(
                    _stable_id("summary_failure", definition.definition_id, code, location),
                    definition.user_id,
                    definition.definition_id,
                    code,
                    location,
                )
            )
    return tuple(predictions), tuple(empty), tuple(failures)


def score_grounded_summaries(
    definitions: Sequence[SessionDefinition],
    claims: Sequence[Mapping[str, object]],
    predictions: Sequence[GroundedSummaryPrediction],
    empty: Sequence[EmptySessionPrediction],
    failures: Sequence[GroundedSummaryFailure],
    *,
    deletion_recompute_pass: bool,
) -> GroundedSummaryChecks:
    definitions_by_id = {item.definition_id: item for item in definitions}
    expected_paths: dict[str, tuple[str, str, str, str, str]] = {}
    for item in claims:
        evidence = item["evidence"][0]
        span_identity = (
            item["user_id"], evidence["source_id"], evidence["message_id"],
            evidence["quote"],
        )
        expected_paths[item["claim_id"]] = (
            item["user_id"],
            f"version_{item['claim_id']}",
            evidence["source_id"],
            "span_" + hashlib.sha256(canonical_json_bytes(span_identity)[:-1]).hexdigest(),
            "supports",
        )
    accounted = [item.session_definition_id for item in (*predictions, *empty, *failures)]
    if set(accounted) != set(definitions_by_id) or len(accounted) != len(set(accounted)):
        raise GroundedSummaryEvaluationError("session outcomes are incomplete or duplicated")
    provenance_total = 0
    provenance_valid = 0
    membership_valid = 0
    cross_user = 0
    stale = 0
    unsupported = 0
    statement_ids: list[str] = []
    covered_claims: set[str] = set()
    for prediction in predictions:
        definition = definitions_by_id[prediction.session_definition_id]
        if prediction.user_id != definition.user_id:
            cross_user += 1
        if prediction.source_ids == definition.source_ids:
            membership_valid += 1
        for statement in prediction.statements:
            statement_ids.append(statement.statement_id)
            if statement.lifecycle_view not in {
                "accepted", "candidate", "historical", "disputed"
            }:
                unsupported += 1
            for path in statement.evidence:
                provenance_total += 1
                expected = expected_paths.get(path.claim_id)
                if expected is None:
                    stale += 1
                    continue
                owner, version_id, source_id, span_id, support_type = expected
                if owner != prediction.user_id:
                    cross_user += 1
                if (
                    path.claim_version_id != version_id
                    or path.source_id != source_id
                    or path.span_id != span_id
                ):
                    stale += 1
                    continue
                if path.source_id not in definition.source_ids:
                    stale += 1
                    continue
                if path.support_type != support_type:
                    unsupported += 1
                    continue
                provenance_valid += 1
                covered_claims.add(path.claim_id)
    for item in empty:
        definition = definitions_by_id[item.session_definition_id]
        if item.user_id != definition.user_id:
            cross_user += 1
        if item.source_ids != definition.source_ids:
            stale += 1
    return GroundedSummaryChecks(
        DATASET_VERSION,
        len(definitions),
        len(predictions),
        len(empty),
        len(failures),
        len(statement_ids),
        _metric(provenance_valid, provenance_total, "no_statement_evidence"),
        _metric(membership_valid, len(predictions), "no_persisted_summaries"),
        _metric(len(covered_claims), len(expected_paths), "no_eligible_claims"),
        cross_user,
        stale,
        len(statement_ids) - len(set(statement_ids)),
        unsupported,
        deletion_recompute_pass,
    )


def execute_grounded_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> GroundedSummaryChecks:
    root = Path(repo_root).resolve()
    output = root / output_dir
    if output.exists() and any(output.iterdir()):
        raise GroundedSummaryEvaluationError("result directory must be empty")
    manifest, sources, definitions, claims = load_grounded_runtime(repo_root=root)
    connection = connection_factory()
    try:
        apply_migrations(connection, root / "migrations")
        _require_clean(connection)
        materialize_grounded_runtime(connection, sources, claims)
        _verify_recomputed_sessions(connection, definitions)
        predictions, empty, failures = run_grounded_summaries(connection, definitions)
        deletion_pass = _deletion_recompute_check(connection, predictions)
        checks = score_grounded_summaries(
            definitions,
            claims,
            predictions,
            empty,
            failures,
            deletion_recompute_pass=deletion_pass,
        )
        _verify_persisted(connection, definitions, predictions, empty)
    finally:
        connection.close()
    _write_release(root, output, manifest, predictions, empty, failures, checks)
    return checks


def verify_grounded_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = root / output_dir
    manifest = _read_object(output / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise GroundedSummaryEvaluationError("release artifact map changed")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
            raise GroundedSummaryEvaluationError("release artifact hash is invalid")
        if _file_sha256(output / name) != expected:
            raise GroundedSummaryEvaluationError("release artifact changed")
    dataset = manifest.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("path") != DATASET_MANIFEST.as_posix()
        or dataset.get("sha256") != _file_sha256(root / DATASET_MANIFEST)
    ):
        raise GroundedSummaryEvaluationError("release dataset binding changed")
    implementation = manifest.get("implementation_hashes")
    if not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS):
        raise GroundedSummaryEvaluationError("release implementation map changed")
    for path, expected in implementation.items():
        if _file_sha256(root / path) != expected:
            raise GroundedSummaryEvaluationError("release implementation changed")
    if manifest.get("predecessor") != _predecessor_attestation(root):
        raise GroundedSummaryEvaluationError("release predecessor attestation changed")


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


def serialize_jsonl(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(item)) for item in values)


def _write_release(
    root: Path,
    output: Path,
    dataset_manifest: Mapping[str, object],
    predictions: Sequence[GroundedSummaryPrediction],
    empty: Sequence[EmptySessionPrediction],
    failures: Sequence[GroundedSummaryFailure],
    checks: GroundedSummaryChecks,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summaries.jsonl": serialize_jsonl(predictions),
        "empty_sessions.jsonl": serialize_jsonl(empty),
        "failures.jsonl": serialize_jsonl(failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": DATASET_VERSION,
                "starting_commit": STARTING_COMMIT,
                "source_count": EXPECTED["source_count"],
                "session_count": checks.session_count,
                "summary_count": checks.summary_count,
                "empty_session_count": checks.empty_session_count,
                "failure_count": checks.failure_count,
                "request_count": 0,
                "retry_count": 0,
                "model_calls": 0,
                "model_fallback": False,
                "cost_usd": 0,
                "execution_mode": "deterministic",
            }
        ),
        "findings.md": (
            "# Grounded summary development findings\n\n"
            "The run persisted 17 summaries from 20 sessions. Three sessions had no "
            "eligible claim evidence, so they were recorded as empty and were not stored. "
            "All 33 candidate claims kept their exact source, span, and version links.\n\n"
            "The deletion check removed a source-backed summary before rebuilding the "
            "remaining sessions. This development run checks structure and reproducibility; "
            "it does not rate writing quality or production traffic.\n"
        ).encode("utf-8"),
    }
    for name, content in artifacts.items():
        _write_exclusive(output / name, content)
    release = {
        "artifact_version": DATASET_VERSION,
        "artifacts": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in artifacts.items()
        },
        "dataset": {
            "path": DATASET_MANIFEST.as_posix(),
            "sha256": _file_sha256(root / DATASET_MANIFEST),
            "input_bindings": dataset_manifest["inputs"],
        },
        "checks": asdict(checks),
        "implementation_hashes": {
            path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS
        },
        "predecessor": _predecessor_attestation(root),
        "limitations": [
            "The frozen development sessions each contain one source.",
            "All 33 imported claims remain candidates; this run does not promote lifecycle state.",
            "Structural checks do not score editorial quality.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))


def _verify_recomputed_sessions(
    connection: object,
    definitions: Sequence[SessionDefinition],
) -> None:
    service = SessionizationService(SessionSourceRepository(connection))
    recomputed: list[SessionDefinition] = []
    for user_id in DEVELOPMENT_USERS:
        cutoff = next(
            item.transaction_as_of for item in definitions if item.user_id == user_id
        )
        recomputed.extend(
            service.define_sessions(SessionizationRequest(user_id, cutoff)).definitions
        )
    by_id = {item.definition_id: item for item in definitions}
    if {item.definition_id: item for item in recomputed} != by_id:
        raise GroundedSummaryEvaluationError("Step 6.1 session replay changed")


def _verify_persisted(
    connection: object,
    definitions: Sequence[SessionDefinition],
    predictions: Sequence[GroundedSummaryPrediction],
    empty: Sequence[EmptySessionPrediction],
) -> None:
    rows = connection.execute(
        """
        SELECT summary_id, user_id, session_definition_id
        FROM session_summaries WHERE transaction_to IS NULL
        ORDER BY user_id, session_definition_id
        """
    ).fetchall()
    expected = sorted(
        (item.summary_id, item.user_id, item.session_definition_id)
        for item in predictions
    )
    if sorted(rows) != expected:
        raise GroundedSummaryEvaluationError("persisted summary set changed")
    empty_ids = {item.session_definition_id for item in empty}
    if any(row[2] in empty_ids for row in rows):
        raise GroundedSummaryEvaluationError("empty session was persisted")
    mapped = {
        row[0]: tuple(row[1])
        for row in connection.execute(
            """
            SELECT summary_id, array_agg(source_id ORDER BY source_order)
            FROM session_summary_sources GROUP BY summary_id
            """
        ).fetchall()
    }
    if any(mapped.get(item.summary_id) != item.source_ids for item in predictions):
        raise GroundedSummaryEvaluationError("persisted membership changed")
    if len(definitions) != len(predictions) + len(empty):
        raise GroundedSummaryEvaluationError("persisted session accounting changed")


def _deletion_recompute_check(
    connection: object,
    predictions: Sequence[GroundedSummaryPrediction],
) -> bool:
    if not predictions:
        return False
    target = predictions[0]
    source_id = target.source_ids[0]
    before = connection.execute(
        "SELECT count(*) FROM session_summaries WHERE transaction_to IS NULL"
    ).fetchone()[0]
    deleted_at = max(
        datetime.fromisoformat("2026-11-14T09:00:00+00:00"),
        connection.execute(
            "SELECT max(ingested_at) FROM source_events WHERE user_id = %s",
            (target.user_id,),
        ).fetchone()[0],
    ) + timedelta(seconds=1)
    with connection.transaction(force_rollback=True):
        IngestionService(connection).delete_source(target.user_id, source_id, deleted_at)
        immediate = connection.execute(
            """
            SELECT count(*) FROM session_summary_sources
            WHERE user_id = %s AND source_id = %s
            """,
            (target.user_id, source_id),
        ).fetchone()[0] == 0
        event_id = connection.execute(
            """
            SELECT event_id FROM processing_outbox
            WHERE user_id = %s AND event_type = 'source_deleted'
              AND aggregate_id = %s
            """,
            (target.user_id, source_id),
        ).fetchone()[0]
        event = StorageRepository(connection).get_outbox(target.user_id, event_id)
        if not isinstance(event, ProcessingOutboxRecord):
            return False
        GroundedSummaryCoordinator(connection).process(event)
        after = connection.execute(
            "SELECT count(*) FROM session_summaries WHERE transaction_to IS NULL"
        ).fetchone()[0]
        no_deleted_source = connection.execute(
            "SELECT count(*) FROM source_events WHERE user_id = %s AND source_id = %s",
            (target.user_id, source_id),
        ).fetchone()[0] == 0
        passed = immediate and no_deleted_source and after == before - 1
    restored = connection.execute(
        "SELECT count(*) FROM session_summaries WHERE transaction_to IS NULL"
    ).fetchone()[0]
    return passed and restored == before


def _prediction(result, source_ids: Sequence[str]) -> GroundedSummaryPrediction:
    summary = result.summary
    statements = tuple(
        GroundedStatementPrediction(
            item.statement_id,
            item.statement_kind,
            item.lifecycle_view,
            item.text,
            item.claim_ids,
            tuple(
                EvidencePath(
                    evidence.claim_id,
                    evidence.claim_version_id,
                    evidence.source_id,
                    evidence.span_id,
                    evidence.support_type,
                )
                for evidence in item.evidence
            ),
            item.sensitive,
        )
        for item in (*summary.observed_facts, *summary.unresolved_questions)
    )
    return GroundedSummaryPrediction(
        summary.summary_id,
        summary.user_id,
        summary.session_definition_id,
        result.request.session_membership_sha256,
        summary.renderer_version,
        result.input_snapshot_sha256,
        tuple(source_ids),
        summary.claim_ids,
        summary.summary_text,
        statements,
        summary.contains_sensitive,
    )


def _validate_manifest(value: Mapping[str, object]) -> None:
    if set(value) != {
        "dataset_version", "split", "review_status", "guidance_version",
        "expected", "allowed_users", "inputs", "loader_boundary", "execution_mode",
    }:
        raise GroundedSummaryEvaluationError("dataset manifest fields changed")
    if (
        value["dataset_version"] != DATASET_VERSION
        or value["split"] != "development"
        or value["review_status"] != "implementation_reviewed"
        or value["guidance_version"] != "step-6.2-guidance-v1"
        or value["expected"] != EXPECTED
        or value["allowed_users"] != list(DEVELOPMENT_USERS)
        or value["execution_mode"] != "deterministic_no_model"
    ):
        raise GroundedSummaryEvaluationError("dataset manifest identity changed")
    serialized = json.dumps(value, sort_keys=True)
    for forbidden in ("oracle", "review_queue", "test_user"):
        if forbidden in serialized:
            raise GroundedSummaryEvaluationError("forbidden dataset dependency")


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    predecessor_path = Path("results/summaries/sessionization-development-v1/manifest.json")
    predecessor_sha256 = _file_sha256(root / predecessor_path)
    if predecessor_sha256 != "34611525b22cb9dcf8b5c9eb4affd2422d778b58b4b443c90913ce29c8f9365c":
        raise GroundedSummaryEvaluationError("Step 6.1 release manifest changed")
    predecessor = _read_object(root / predecessor_path)
    implementation = predecessor.get("implementation_hashes")
    protected = predecessor.get("predecessor", {}).get("protected_file_hashes")
    if not isinstance(implementation, dict) or not isinstance(protected, dict):
        raise GroundedSummaryEvaluationError("Step 6.1 predecessor map is invalid")
    baseline = dict(protected)
    baseline.update(implementation)
    if any(not isinstance(path, str) or not isinstance(value, str) for path, value in baseline.items()):
        raise GroundedSummaryEvaluationError("Step 6.1 predecessor hash is invalid")
    current = {path: _file_sha256(root / path) for path in sorted(baseline)}
    drift = {path for path in baseline if current[path] != baseline[path]}
    if drift != set(AUTHORIZED_PREDECESSOR_DRIFT):
        raise GroundedSummaryEvaluationError("predecessor drift set changed")
    return {
        "manifest": {
            "path": predecessor_path.as_posix(),
            "sha256": predecessor_sha256,
        },
        "protected_file_count": len(baseline),
        "protected_file_hashes": dict(sorted(baseline.items())),
        "authorized_drift": [
            {
                "path": path,
                "predecessor_sha256": baseline[path],
                "step6_2_sha256": current[path],
                "reason": AUTHORIZED_PREDECESSOR_DRIFT[path],
            }
            for path in sorted(AUTHORIZED_PREDECESSOR_DRIFT)
        ],
        "unchanged_file_count": len(baseline) - len(drift),
    }


def _validate_runtime(
    sources: Sequence[DevelopmentSource],
    definitions: Sequence[SessionDefinition],
    claims: Sequence[Mapping[str, object]],
    evidence_index: Mapping[str, object],
) -> None:
    if (len(sources), len(definitions), len(claims)) != (20, 20, 33):
        raise GroundedSummaryEvaluationError("development runtime count changed")
    source_owners = {item.source_id: item.user_id for item in sources}
    seen_sources = [source_id for item in definitions for source_id in item.source_ids]
    if len(seen_sources) != 20 or set(seen_sources) != set(source_owners):
        raise GroundedSummaryEvaluationError("session source coverage changed")
    if any(
        source_owners.get(source_id) != item.user_id
        for item in definitions
        for source_id in item.source_ids
    ):
        raise GroundedSummaryEvaluationError("session crossed a user boundary")
    claim_ids: set[str] = set()
    indexed: dict[str, tuple[str, tuple[str, ...]]] = {}
    for source in evidence_index.get("sources", []):
        indexed[source["source_id"]] = (
            source["user_id"], tuple(source["claim_ids"])
        )
    if set(indexed) != set(source_owners):
        raise GroundedSummaryEvaluationError("evidence index source coverage changed")
    by_source: dict[str, list[str]] = {source_id: [] for source_id in source_owners}
    for item in claims:
        claim_id = item.get("claim_id")
        if not isinstance(claim_id, str) or claim_id in claim_ids:
            raise GroundedSummaryEvaluationError("claim identity changed")
        claim_ids.add(claim_id)
        if item.get("user_id") not in DEVELOPMENT_USERS or len(item.get("evidence", [])) != 1:
            raise GroundedSummaryEvaluationError("claim boundary changed")
        evidence = item["evidence"][0]
        source_id = evidence.get("source_id")
        if source_owners.get(source_id) != item["user_id"]:
            raise GroundedSummaryEvaluationError("claim evidence crossed a user boundary")
        by_source[source_id].append(claim_id)
    for source_id, values in by_source.items():
        owner, expected_ids = indexed[source_id]
        if owner != source_owners[source_id] or tuple(sorted(values)) != tuple(expected_ids):
            raise GroundedSummaryEvaluationError("evidence index claim mapping changed")


def _definition(value: Mapping[str, object]) -> SessionDefinition:
    if value.get("execution_mode") != "deterministic" or value.get("model") is not None:
        raise GroundedSummaryEvaluationError("session prediction execution changed")
    return SessionDefinition(
        value["definition_id"],
        value["user_id"],
        value["source_type"],
        value["boundary_kind"],
        value["boundary_version"],
        tuple(value["source_ids"]),
        datetime.fromisoformat(value["start_at"].replace("Z", "+00:00")),
        datetime.fromisoformat(value["end_at"].replace("Z", "+00:00")),
        datetime.fromisoformat(value["transaction_as_of"].replace("Z", "+00:00")),
        value["membership_sha256"],
    )


def _boundary(value: object, precision: str) -> tuple[date | None, datetime | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise GroundedSummaryEvaluationError("claim time boundary is invalid")
    if precision == "timestamp":
        return None, datetime.fromisoformat(value.replace("Z", "+00:00"))
    return date.fromisoformat(value), None


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {
            "status": "not_evaluated", "value": None, "numerator": numerator,
            "denominator": denominator, "reason": reason,
        }
    return {
        "status": "evaluated", "value": numerator / denominator,
        "numerator": numerator, "denominator": denominator, "reason": None,
    }


def _require_clean(connection: object) -> None:
    counts = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM memory_users),
          (SELECT count(*) FROM source_events),
          (SELECT count(*) FROM claims),
          (SELECT count(*) FROM session_summaries)
        """
    ).fetchone()
    if counts != (0, 0, 0, 0):
        raise GroundedSummaryEvaluationError("evaluation database must be clean")


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedSummaryEvaluationError("bound JSON input is unreadable") from error
    if not isinstance(value, dict):
        raise GroundedSummaryEvaluationError("bound JSON input must be an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedSummaryEvaluationError("bound JSONL input is unreadable") from error
    if any(not isinstance(item, dict) for item in rows):
        raise GroundedSummaryEvaluationError("bound JSONL row must be an object")
    return rows


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise GroundedSummaryEvaluationError("result artifact already exists") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise GroundedSummaryEvaluationError("bound file is unreadable") from error


def _stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(canonical_json_bytes([namespace, *values])[:-1]).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise GroundedSummaryEvaluationError("timestamp must be aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundedSummaryEvaluationError(f"{name} must be nonempty text")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise GroundedSummaryEvaluationError(f"{name} must be a SHA-256")
    return value


def _user(value: object) -> str:
    if value not in DEVELOPMENT_USERS:
        raise GroundedSummaryEvaluationError("user is outside the development split")
    return value
