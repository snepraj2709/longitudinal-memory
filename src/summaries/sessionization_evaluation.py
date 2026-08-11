"""Gold-free deterministic development evaluation for session boundaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from storage.contracts import MemoryUser, SourceEventRecord, StorageValidationError, safe_json
from storage.migrations import apply_migrations
from storage.repository import StorageRepository

from .contracts import BOUNDARY_VERSION, SessionDefinition, SessionizationRequest
from .repository import SessionSourceRepository
from .sessions import SessionizationService


DATASET_VERSION = "sessionization_development_v1"
DATASET_MANIFEST = Path("data/summaries/sessionization-development-v1/manifest.json")
RESULT_ROOT = Path("results/summaries/sessionization-development-v1")
STEP5_RELEASE = Path("results/conflicts/phase5-conflict-evaluation-development-v1")
STEP5_MANIFEST_SHA256 = "35f37c3ef5f4a45739df200e64053a5d35552dec264336bb6bca3b701fb850ff"
DEVELOPMENT_USERS = ("user_001", "user_002")
SOURCE_SUFFIXES = (
    "conversation_001", "email_001", "chat_001", "calendar_001",
    "conversation_002", "email_002", "chat_002", "calendar_002",
    "conversation_003", "conversation_004",
)
EXPECTED_SOURCE_IDS = tuple(
    f"scaled_{user_id}_{suffix}"
    for user_id in DEVELOPMENT_USERS
    for suffix in SOURCE_SUFFIXES
)
EXPECTED_TYPE_COUNTS = {"calendar": 4, "chat": 4, "conversation": 8, "email": 4}
IMPLEMENTATION_PATHS = (
    "Makefile",
    "configs/summaries/session_boundaries_v1.json",
    "data/summaries/sessionization-development-v1/manifest.json",
    "src/summaries/__init__.py",
    "src/summaries/contracts.py",
    "src/summaries/repository.py",
    "src/summaries/sessionization_evaluation.py",
    "src/summaries/sessions.py",
    "tests/integration/test_phase5_conflict_evaluation.py",
    "tests/integration/test_sessionization.py",
    "tests/unit/test_sessionization.py",
    "tests/unit/test_sessionization_evaluation.py",
)
EXPECTED_INPUT_HASHES = {
    "sources_file": "a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de",
    "sources_prefix": "b0dc2b0dae985b51e2e25e441f98dcbe8d86045fa6a3beae2545f50de9e844d9",
    "users_file": "13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef",
    "users_prefix": "3cb98b28b57ed333c170eb93ba2bdd9cea1c24941df74d3d60e75397a88fbfdd",
    "boundary_config": "d2af3cbfd35e24d1f0b3a10acd148fb46a7fc2b4cd96268182bd12dd8570f20b",
}
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version", "split", "review_status", "source_count", "user_counts",
        "expected_source_order", "expected_session_counts", "inputs", "loader_boundary",
        "evaluation_contract", "guidance_version", "decision_envelope_sha256",
    }
)
TOKEN = re.compile(r"^[a-z0-9_:-]+$")


class SessionizationEvaluationError(ValueError):
    """Reject malformed, leaky, or incomplete evaluation state."""


@dataclass(frozen=True)
class DevelopmentSource:
    source_id: str
    user_id: str
    source_type: str
    produced_at: datetime
    ingested_at: datetime
    raw_content: str
    participants: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _token(self.source_id, "source_id")
        _development_user(self.user_id)
        if self.source_type not in {"conversation", "email", "chat", "calendar"}:
            raise SessionizationEvaluationError("source_type is invalid")
        _aware(self.produced_at, "produced_at")
        _aware(self.ingested_at, "ingested_at")
        if not isinstance(self.raw_content, str) or not self.raw_content:
            raise SessionizationEvaluationError("source content is empty")
        if (
            not self.participants
            or len(self.participants) != len(set(self.participants))
            or self.user_id not in self.participants
        ):
            raise SessionizationEvaluationError("source participants are invalid")
        try:
            metadata = safe_json(self.metadata, "metadata", top_type=dict)
        except StorageValidationError as error:
            raise SessionizationEvaluationError("source metadata is invalid") from error
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class SessionizationPrediction:
    definition_id: str
    user_id: str
    source_type: str
    boundary_kind: str
    boundary_version: str
    source_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    transaction_as_of: datetime
    membership_sha256: str
    execution_mode: str = "deterministic"
    model: None = None

    def __post_init__(self) -> None:
        if not _sha256(self.definition_id) or not _sha256(self.membership_sha256):
            raise SessionizationEvaluationError("prediction hash is invalid")
        _development_user(self.user_id)
        if self.source_type not in EXPECTED_TYPE_COUNTS:
            raise SessionizationEvaluationError("prediction source_type is invalid")
        if self.boundary_kind not in {
            "source", "declared_thread", "declared_thread_or_source", "inactivity"
        }:
            raise SessionizationEvaluationError("prediction boundary kind is invalid")
        if self.boundary_version != BOUNDARY_VERSION:
            raise SessionizationEvaluationError("prediction boundary version changed")
        if (
            not self.source_ids
            or any(not isinstance(item, str) or not item for item in self.source_ids)
            or len(self.source_ids) != len(set(self.source_ids))
        ):
            raise SessionizationEvaluationError("prediction source IDs are invalid")
        _aware(self.start_at, "start_at")
        _aware(self.end_at, "end_at")
        _aware(self.transaction_as_of, "transaction_as_of")
        if self.end_at < self.start_at:
            raise SessionizationEvaluationError("prediction time range is invalid")
        if self.execution_mode != "deterministic" or self.model is not None:
            raise SessionizationEvaluationError("prediction execution mode changed")


@dataclass(frozen=True)
class SessionizationFailure:
    failure_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "user_id", "code", "location"):
            _token(getattr(self, name), name)
        _development_user(self.user_id)


@dataclass(frozen=True)
class SessionizationScorecard:
    dataset_version: str
    source_count: int
    session_count: int
    prediction_count: int
    failure_count: int
    source_coverage: Mapping[str, object]
    failure_rate: Mapping[str, object]
    session_counts_by_type: Mapping[str, int]
    session_counts_by_user: Mapping[str, int]
    invalid_session_count: int
    duplicate_source_count: int
    cross_user_count: int
    deterministic_prediction_count: int
    model_prediction_count: int
    model_fallback: bool


def load_sessionization_runtime(
    manifest_path: str | Path = DATASET_MANIFEST,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], tuple[DevelopmentSource, ...]]:
    root = Path(repo_root).resolve()
    manifest = _read_object(root / manifest_path)
    _validate_manifest(manifest)
    inputs = manifest["inputs"]
    users_binding = inputs["users"]
    source_binding = inputs["sources"]
    config_binding = inputs["boundary_config"]
    if _file_sha256(root / config_binding["path"]) != config_binding["sha256"]:
        raise SessionizationEvaluationError("boundary config hash changed")
    user_rows, user_prefix = _read_prefix(
        root / users_binding["path"], users_binding["development_prefix_records"]
    )
    source_rows, source_prefix = _read_prefix(
        root / source_binding["path"], source_binding["development_prefix_records"]
    )
    if hashlib.sha256(user_prefix).hexdigest() != users_binding["development_prefix_sha256"]:
        raise SessionizationEvaluationError("development user prefix hash changed")
    if hashlib.sha256(source_prefix).hexdigest() != source_binding["development_prefix_sha256"]:
        raise SessionizationEvaluationError("development source prefix hash changed")
    _validate_users(user_rows)
    sources = tuple(_source(row, position) for position, row in enumerate(source_rows))
    _validate_runtime(sources)
    return manifest, sources


def materialize_runtime_sources(connection: object, sources: Sequence[DevelopmentSource]) -> None:
    _validate_runtime(sources)
    repository = StorageRepository(connection)
    for user_id in DEVELOPMENT_USERS:
        first = min(item.produced_at for item in sources if item.user_id == user_id)
        repository.insert_user(MemoryUser(user_id, first))
    for item in sources:
        repository.insert_source_event(
            SourceEventRecord(
                source_id=item.source_id,
                user_id=item.user_id,
                source_type=item.source_type,
                session_id=None,
                idempotency_key=f"sessionization:{item.source_id}",
                produced_at=item.produced_at,
                ingested_at=item.ingested_at,
                raw_content=item.raw_content,
                participants=list(item.participants),
                metadata=item.metadata,
                content_hash=hashlib.sha256(item.raw_content.encode("utf-8")).hexdigest(),
            )
        )


def run_sessionization(
    connection: object,
    sources: Sequence[DevelopmentSource],
    *,
    transaction_as_of: datetime | None = None,
) -> tuple[tuple[SessionizationPrediction, ...], tuple[SessionizationFailure, ...]]:
    _validate_runtime(sources)
    predictions: list[SessionizationPrediction] = []
    failures: list[SessionizationFailure] = []
    repository = SessionSourceRepository(connection)
    service = SessionizationService(repository)
    for user_id in DEVELOPMENT_USERS:
        cutoff = transaction_as_of or max(
            item.ingested_at for item in sources if item.user_id == user_id
        )
        try:
            result = service.define_sessions(
                SessionizationRequest(user_id, cutoff, BOUNDARY_VERSION)
            )
            predictions.extend(_prediction(item) for item in result.definitions)
        except Exception as error:
            code = getattr(error, "code", "sessionization_failed")
            location = getattr(error, "location", "service")
            if not isinstance(code, str) or TOKEN.fullmatch(code) is None:
                code = "sessionization_failed"
            if not isinstance(location, str) or TOKEN.fullmatch(location) is None:
                location = "service"
            failures.append(_failure(user_id, code, location))
    return (
        tuple(sorted(predictions, key=_prediction_order)),
        tuple(sorted(failures, key=lambda item: item.user_id)),
    )


def score_sessionization(
    sources: Sequence[DevelopmentSource],
    predictions: Sequence[SessionizationPrediction],
    failures: Sequence[SessionizationFailure],
) -> SessionizationScorecard:
    _validate_runtime(sources)
    owners = {item.source_id: item.user_id for item in sources}
    seen: list[str] = []
    invalid = 0
    cross_user = 0
    for prediction in predictions:
        member_sources = [item for item in sources if item.source_id in prediction.source_ids]
        if (
            not member_sources
            or any(item.user_id != prediction.user_id for item in member_sources)
            or any(owners.get(source_id) != prediction.user_id for source_id in prediction.source_ids)
        ):
            invalid += 1
            cross_user += 1
        elif (
            any(item.source_type != prediction.source_type for item in member_sources)
            or min(item.produced_at for item in member_sources) != prediction.start_at
            or max(item.produced_at for item in member_sources) != prediction.end_at
        ):
            invalid += 1
        seen.extend(prediction.source_ids)
    duplicate_count = len(seen) - len(set(seen))
    covered = len(set(seen) & set(owners))
    failure_users = [item.user_id for item in failures]
    if len(failure_users) != len(set(failure_users)) or any(
        user_id not in DEVELOPMENT_USERS for user_id in failure_users
    ):
        raise SessionizationEvaluationError("failure accounting is invalid")
    prediction_users = {item.user_id for item in predictions}
    for user_id in DEVELOPMENT_USERS:
        if (user_id in prediction_users) == (user_id in failure_users):
            raise SessionizationEvaluationError("each user requires predictions or one failure")
    return SessionizationScorecard(
        dataset_version=DATASET_VERSION,
        source_count=len(sources),
        session_count=len(predictions),
        prediction_count=len(predictions),
        failure_count=len(failures),
        source_coverage=_metric(covered, len(sources), "no_sources"),
        failure_rate=_metric(len(failures), len(DEVELOPMENT_USERS), "no_users"),
        session_counts_by_type=dict(sorted(Counter(item.source_type for item in predictions).items())),
        session_counts_by_user=dict(sorted(Counter(item.user_id for item in predictions).items())),
        invalid_session_count=invalid,
        duplicate_source_count=duplicate_count,
        cross_user_count=cross_user,
        deterministic_prediction_count=sum(item.execution_mode == "deterministic" for item in predictions),
        model_prediction_count=sum(item.model is not None for item in predictions),
        model_fallback=False,
    )


def execute_sessionization_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> SessionizationScorecard:
    root = Path(repo_root).resolve()
    output = root / output_dir
    if output.exists() and any(output.iterdir()):
        raise SessionizationEvaluationError("result directory must be empty")
    manifest, sources = load_sessionization_runtime(repo_root=root)
    connection = connection_factory()
    try:
        apply_migrations(connection, root / "migrations")
        _require_clean_database(connection)
        materialize_runtime_sources(connection, sources)
        predictions, failures = run_sessionization(connection, sources)
        session_tables = connection.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE %s
            """,
            ("%session%",),
        ).fetchall()
        if session_tables:
            raise SessionizationEvaluationError("session state was persisted")
    finally:
        connection.close()
    scorecard = score_sessionization(sources, predictions, failures)
    _write_release(root, output, manifest, sources, predictions, failures, scorecard)
    return scorecard


def verify_release(output_dir: str | Path = RESULT_ROOT, *, repo_root: str | Path = ".") -> None:
    root = Path(repo_root).resolve()
    output = root / output_dir
    release = _read_object(output / "manifest.json")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "predictions.jsonl", "failures.jsonl", "scores.json", "run.json", "findings.md"
    }:
        raise SessionizationEvaluationError("release artifact map changed")
    for name, expected in artifacts.items():
        if not _sha256(expected) or _file_sha256(output / name) != expected:
            raise SessionizationEvaluationError("release artifact hash changed")
    if release.get("predecessor") != _predecessor_attestation(root):
        raise SessionizationEvaluationError("predecessor attestation changed")
    implementation = release.get("implementation_hashes")
    if not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS):
        raise SessionizationEvaluationError("implementation hash map changed")
    for path, expected in implementation.items():
        if _file_sha256(root / path) != expected:
            raise SessionizationEvaluationError("sessionization implementation changed")


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
    sources: Sequence[DevelopmentSource],
    predictions: Sequence[SessionizationPrediction],
    failures: Sequence[SessionizationFailure],
    scorecard: SessionizationScorecard,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "predictions.jsonl": serialize_jsonl(predictions),
        "failures.jsonl": serialize_jsonl(failures),
        "scores.json": canonical_json_bytes(asdict(scorecard)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": "sessionization_development_v1",
                "dataset_version": DATASET_VERSION,
                "starting_commit": "548b12750142eb07c8749f0a8f7834ba4c7b7c3b",
                "source_count": len(sources),
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "request_count": 0,
                "retry_count": 0,
                "model_calls": 0,
                "cost_usd": 0,
                "execution_mode": "deterministic",
                "persisted_session_rows": 0,
            }
        ),
        "findings.md": (
            "# Sessionization development findings\n\n"
            "All 20 development sources were assigned once, producing 20 deterministic "
            "sessions with no failures or cross-user membership. The frozen sources use "
            "unique declared thread IDs, so the grouping edge cases remain covered by tests.\n\n"
            "This run measures boundary reproducibility and source coverage. It does not "
            "evaluate summary quality or production traffic.\n"
        ).encode("utf-8"),
    }
    for name, content in artifacts.items():
        _write_exclusive(output / name, content)
    release = {
        "artifact_version": "sessionization_development_v1",
        "artifacts": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()},
        "dataset": {
            "manifest_path": DATASET_MANIFEST.as_posix(),
            "manifest_sha256": _file_sha256(root / DATASET_MANIFEST),
            "source_prefix_sha256": dataset_manifest["inputs"]["sources"]["development_prefix_sha256"],
            "user_prefix_sha256": dataset_manifest["inputs"]["users"]["development_prefix_sha256"],
            "boundary_config_sha256": dataset_manifest["inputs"]["boundary_config"]["sha256"],
        },
        "metrics": asdict(scorecard),
        "predecessor": _predecessor_attestation(root),
        "implementation_hashes": {
            path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS
        },
        "limitations": [
            "The 20 frozen development sources have unique declared thread IDs.",
            "Synthetic integration fixtures cover thread grouping, late ingestion, and deletion recomputation.",
            "No summaries or durative memories are evaluated in Step 6.1.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    manifest_path = root / STEP5_RELEASE / "manifest.json"
    if _file_sha256(manifest_path) != STEP5_MANIFEST_SHA256:
        raise SessionizationEvaluationError("Step 5.4 release manifest changed")
    manifest = _read_object(manifest_path)
    artifacts = manifest.get("artifacts")
    step5_implementation = manifest.get("implementation_hashes")
    predecessor = manifest.get("predecessor")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(step5_implementation, dict)
        or not isinstance(predecessor, dict)
    ):
        raise SessionizationEvaluationError("Step 5.4 attestation is invalid")
    for name, expected in artifacts.items():
        if not _sha256(expected) or _file_sha256(root / STEP5_RELEASE / name) != expected:
            raise SessionizationEvaluationError("Step 5.4 artifact changed")
    protected_maps = predecessor.get("protected_maps")
    prior_implementation = predecessor.get("implementation_hashes")
    if not isinstance(protected_maps, dict) or not isinstance(prior_implementation, dict):
        raise SessionizationEvaluationError("Step 5.4 predecessor map is invalid")
    protected = protected_maps.get("protected_file_hashes")
    if not isinstance(protected, dict):
        raise SessionizationEvaluationError("Step 5.4 protected hashes are invalid")
    effective = dict(protected)
    effective.update(prior_implementation)
    for path, expected in step5_implementation.items():
        if path in effective:
            effective[path] = expected
    if len(effective) != 68:
        raise SessionizationEvaluationError("Step 5.4 protected file count changed")
    expected_current = dict(effective)
    expected_current.update(step5_implementation)
    drift_reasons = {
        "Makefile": "adds_step6_1_sessionization_test_target",
        "tests/integration/test_phase5_conflict_evaluation.py": "adapts_step5_4_replay_for_step6_1_makefile_drift",
    }
    drift: list[Mapping[str, object]] = []
    for path, expected in sorted(expected_current.items()):
        current = _file_sha256(root / path)
        if current == expected:
            continue
        reason = drift_reasons.get(path)
        if reason is None:
            raise SessionizationEvaluationError("protected predecessor file changed")
        drift.append(
            {
                "path": path,
                "predecessor_sha256": expected,
                "step6_1_sha256": current,
                "reason": reason,
            }
        )
    if tuple(item["path"] for item in drift) != tuple(sorted(drift_reasons)):
        raise SessionizationEvaluationError("Step 6.1 predecessor drift changed")
    return {
        "manifest": {
            "path": (STEP5_RELEASE / "manifest.json").as_posix(),
            "sha256": STEP5_MANIFEST_SHA256,
        },
        "artifacts": artifacts,
        "protected_file_count": 68,
        "protected_file_hashes": dict(sorted(effective.items())),
        "implementation_hashes": step5_implementation,
        "authorized_drift": drift,
        "unchanged_file_count": len(expected_current) - len(drift),
    }


def _prediction(definition: SessionDefinition) -> SessionizationPrediction:
    return SessionizationPrediction(**asdict(definition))


def _failure(user_id: str, code: str, location: str) -> SessionizationFailure:
    identity = hashlib.sha256(f"{user_id}:{code}:{location}".encode("utf-8")).hexdigest()
    return SessionizationFailure(f"failure_{identity}", user_id, code, location)


def _prediction_order(value: SessionizationPrediction) -> tuple[object, ...]:
    return (value.start_at, value.user_id, value.source_type, value.source_ids[0], value.definition_id)


def _source(row: Mapping[str, object], position: int) -> DevelopmentSource:
    expected_fields = {
        "source_id", "source_type", "user_id", "created_at", "ingested_at",
        "participants", "content", "messages", "metadata",
    }
    if set(row) != expected_fields or row.get("source_id") != EXPECTED_SOURCE_IDS[position]:
        raise SessionizationEvaluationError("development source order or fields changed")
    participants = row.get("participants")
    metadata = row.get("metadata")
    if (
        not isinstance(participants, list)
        or any(not isinstance(item, str) or not item for item in participants)
        or not isinstance(metadata, dict)
        or not isinstance(row.get("content"), str)
        or not isinstance(row.get("user_id"), str)
        or not isinstance(row.get("source_type"), str)
    ):
        raise SessionizationEvaluationError("development source JSON shape changed")
    return DevelopmentSource(
        source_id=str(row["source_id"]),
        user_id=str(row["user_id"]),
        source_type=str(row["source_type"]),
        produced_at=_timestamp(row["created_at"], "created_at"),
        ingested_at=_timestamp(row["ingested_at"], "ingested_at"),
        raw_content=row["content"],
        participants=tuple(participants),
        metadata=metadata,
    )


def _validate_manifest(value: Mapping[str, object]) -> None:
    if set(value) != MANIFEST_FIELDS:
        raise SessionizationEvaluationError("dataset manifest fields changed")
    if (
        value.get("dataset_version") != DATASET_VERSION
        or value.get("split") != "development"
        or value.get("review_status") != "implementation_reviewed"
        or value.get("source_count") != 20
        or value.get("user_counts") != {"user_001": 10, "user_002": 10}
        or tuple(value.get("expected_source_order", ())) != EXPECTED_SOURCE_IDS
        or value.get("expected_session_counts") != {**EXPECTED_TYPE_COUNTS, "total": 20}
        or value.get("loader_boundary") != "parse_exact_development_prefix_only_and_stop_before_test_records"
        or value.get("evaluation_contract") != "deterministic_runtime_source_coverage_no_scorer_inputs_or_persisted_sessions"
        or value.get("guidance_version") != "step-6.1-guidance-v1"
        or value.get("decision_envelope_sha256") != "bdbf7a891369104061b675a9c0754de573444e9eeca27972da6c98b6c863f09f"
    ):
        raise SessionizationEvaluationError("dataset manifest identity changed")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"sources", "users", "boundary_config"}:
        raise SessionizationEvaluationError("dataset input bindings changed")
    expected = {
        "sources": ("data/scaled-v1/runtime/sources.jsonl", 20),
        "users": ("data/scaled-v1/runtime/users.jsonl", 2),
    }
    for name, (path, count) in expected.items():
        binding = inputs[name]
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "frozen_file_sha256", "development_prefix_records", "development_prefix_sha256"}
            or binding.get("path") != path
            or binding.get("development_prefix_records") != count
            or not _sha256(binding.get("frozen_file_sha256"))
            or not _sha256(binding.get("development_prefix_sha256"))
        ):
            raise SessionizationEvaluationError("dataset prefix binding changed")
    if (
        inputs["sources"]["frozen_file_sha256"] != EXPECTED_INPUT_HASHES["sources_file"]
        or inputs["sources"]["development_prefix_sha256"] != EXPECTED_INPUT_HASHES["sources_prefix"]
        or inputs["users"]["frozen_file_sha256"] != EXPECTED_INPUT_HASHES["users_file"]
        or inputs["users"]["development_prefix_sha256"] != EXPECTED_INPUT_HASHES["users_prefix"]
    ):
        raise SessionizationEvaluationError("frozen runtime input hashes changed")
    config = inputs["boundary_config"]
    if (
        not isinstance(config, dict)
        or set(config) != {"path", "sha256"}
        or config.get("path") != "configs/summaries/session_boundaries_v1.json"
        or config.get("sha256") != EXPECTED_INPUT_HASHES["boundary_config"]
    ):
        raise SessionizationEvaluationError("boundary config binding changed")


def _validate_users(rows: Sequence[Mapping[str, object]]) -> None:
    for position, row in enumerate(rows):
        if (
            set(row) != {"display_name", "profile_note", "split", "timezone", "user_id"}
            or row.get("user_id") != DEVELOPMENT_USERS[position]
            or row.get("split") != "development"
        ):
            raise SessionizationEvaluationError("development user order or split changed")


def _validate_runtime(sources: Sequence[DevelopmentSource]) -> None:
    if len(sources) != 20 or tuple(item.source_id for item in sources) != EXPECTED_SOURCE_IDS:
        raise SessionizationEvaluationError("development source order or count changed")
    if len({item.source_id for item in sources}) != 20:
        raise SessionizationEvaluationError("development source IDs are duplicated")
    for position, item in enumerate(sources):
        expected_user = DEVELOPMENT_USERS[position // 10]
        expected_type = SOURCE_SUFFIXES[position % 10].rsplit("_", 1)[0]
        if item.user_id != expected_user or item.source_type != expected_type:
            raise SessionizationEvaluationError("development source ownership or type changed")


def _read_prefix(path: Path, count: int) -> tuple[list[Mapping[str, object]], bytes]:
    rows: list[Mapping[str, object]] = []
    payload = bytearray()
    try:
        with path.open("rb") as handle:
            for _ in range(count):
                line = handle.readline()
                if not line:
                    raise SessionizationEvaluationError("development prefix is incomplete")
                payload.extend(line)
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SessionizationEvaluationError("development prefix record is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionizationEvaluationError("development prefix is unreadable") from error
    return rows, bytes(payload)


def _require_clean_database(connection: object) -> None:
    for table in ("memory_users", "source_events", "source_tombstones"):
        if connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != 0:
            raise SessionizationEvaluationError("evaluation database must be empty")


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionizationEvaluationError("JSON object is unreadable") from error
    if not isinstance(value, dict):
        raise SessionizationEvaluationError("JSON value must be an object")
    return value


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise SessionizationEvaluationError("result artifact already exists") from error


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise SessionizationEvaluationError(f"{name} must be a timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SessionizationEvaluationError(f"{name} is invalid") from error
    return _aware(result, name)


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise SessionizationEvaluationError(f"{name} must be timezone-aware")
    return value


def _development_user(value: object) -> str:
    if value not in DEVELOPMENT_USERS:
        raise SessionizationEvaluationError("only development users are allowed")
    return str(value)


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or TOKEN.fullmatch(value) is None:
        raise SessionizationEvaluationError(f"{name} must be a sanitized token")
    return value


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {"status": "not_evaluated", "numerator": 0, "denominator": 0, "value": None, "reason": reason}
    return {"status": "evaluated", "numerator": numerator, "denominator": denominator, "value": numerator / denominator, "reason": None}


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("value is not canonical JSON")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SessionizationEvaluationError("bound file is unreadable") from error
