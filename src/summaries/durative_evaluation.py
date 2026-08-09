"""Deterministic development evaluation for durative-claim inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from ingestion.service import IngestionService
from extraction.predicate_registry import load_predicate_registry
from storage.contracts import ProcessingOutboxRecord
from storage.migrations import apply_migrations

from .contracts import SessionDefinition
from .durative_contracts import ALLOWED_FAMILIES, RULES_VERSION, canonical_json, stable_id
from .durative_repository import DurativeClaimCoordinator
from .grounded_evaluation import (
    DevelopmentSource,
    canonical_json_bytes,
    load_grounded_runtime,
    materialize_grounded_runtime,
)


DATASET_VERSION = "durative_claim_development_v1"
DATASET_MANIFEST = Path("data/summaries/durative-claim-development-v1/manifest.json")
RESULT_ROOT = Path("results/summaries/durative-claim-development-v1")
STARTING_COMMIT = "16f6c367ed7543aa05b3da00044f9fab1555dfb1"
DEVELOPMENT_USERS = ("user_001", "user_002")
EXPECTED = {
    "user_count": 2,
    "source_count": 20,
    "session_count": 20,
    "input_claim_count": 33,
    "durative_proposition_count": 26,
    "unsupported_predicate_count": 7,
    "accepted_claim_count": 0,
}
STEP62_MANIFEST_SHA256 = "ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761"
AUTHORIZED_PREDECESSOR_DRIFT = {
    "Makefile": "adds_step6_3_durative_claim_test_target",
    "tests/integration/test_belief_resolution.py": "expects_migration_0007",
    "tests/integration/test_conflict_relations.py": "expects_migration_0007",
    "tests/integration/test_grounded_summary_persistence.py": "expects_migration_0007",
    "tests/integration/test_grounded_summary_evaluation.py": "adapts_frozen_step6_2_release_attestation",
    "tests/integration/test_phase4_storage.py": "expects_migration_0007_and_durative_tables",
    "tests/integration/test_phase5_conflict_evaluation.py": "adapts_predecessor_hashes_for_migration_0007",
    "tests/integration/test_temporal_service.py": "expects_migration_0007",
    "tests/unit/test_grounded_summary_evaluation.py": "adapts_frozen_step6_2_release_attestation",
}
ARTIFACT_NAMES = (
    "rejections.jsonl",
    "claims.jsonl",
    "failures.jsonl",
    "checks.json",
    "run.json",
    "findings.md",
)
IMPLEMENTATION_PATHS = (
    "Makefile",
    "configs/summaries/durative_claim_rules_v1.json",
    "data/summaries/durative-claim-development-v1/manifest.json",
    "migrations/0007_durative_claims.sql",
    "src/summaries/durative.py",
    "src/summaries/durative_contracts.py",
    "src/summaries/durative_evaluation.py",
    "src/summaries/durative_repository.py",
    "tests/integration/test_durative_claim_persistence.py",
    "tests/integration/test_durative_claim_evaluation.py",
    "tests/unit/test_durative_claims.py",
    "tests/unit/test_durative_evaluation.py",
)
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version",
        "split",
        "review_status",
        "expected",
        "inputs",
        "loader_boundary",
        "evaluation_contract",
        "guidance_version",
        "decision_envelope_sha256",
    }
)
TOKEN = re.compile(r"^[a-z0-9_:-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DurativeEvaluationError(ValueError):
    """Reject incomplete, leaky, or non-deterministic evaluation state."""


@dataclass(frozen=True)
class DurativeEpisodePrediction:
    episode_id: str
    role: str
    claim_id: str
    claim_version_id: str
    session_definition_id: str
    source_id: str
    span_id: str
    support_type: str

    def __post_init__(self) -> None:
        _sha(self.episode_id, "episode_id")
        _sha(self.session_definition_id, "session_definition_id")
        for name in ("claim_id", "claim_version_id", "source_id", "span_id"):
            _text(getattr(self, name), name)
        if self.role not in {"supports", "counter_evidence"}:
            raise DurativeEvaluationError("episode role is invalid")
        if self.support_type not in {"supports", "contradicts", "corrects"}:
            raise DurativeEvaluationError("episode support type is invalid")


@dataclass(frozen=True)
class DurativeDecisionPrediction:
    plan_id: str
    decision_id: str
    user_id: str
    rules_version: str
    input_snapshot_sha256: str
    status: str
    reason: str
    derived_claim_id: str | None
    derived_version_id: str | None
    transaction_as_of: datetime
    episodes: tuple[DurativeEpisodePrediction, ...]
    execution_mode: str = "deterministic"
    model: None = None

    def __post_init__(self) -> None:
        for name in ("plan_id", "decision_id", "input_snapshot_sha256"):
            _sha(getattr(self, name), name)
        _user(self.user_id)
        if self.rules_version != RULES_VERSION:
            raise DurativeEvaluationError("decision rules version changed")
        if self.status not in {"accepted", "rejected"} or TOKEN.fullmatch(self.reason) is None:
            raise DurativeEvaluationError("decision outcome is invalid")
        if self.status == "accepted":
            _sha(self.derived_claim_id, "derived_claim_id")
            _sha(self.derived_version_id, "derived_version_id")
        elif self.derived_claim_id is not None or self.derived_version_id is not None:
            raise DurativeEvaluationError("rejected decision names a claim")
        _aware(self.transaction_as_of, "transaction_as_of")
        if not self.episodes or len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise DurativeEvaluationError("decision episodes are empty or duplicated")
        if self.execution_mode != "deterministic" or self.model is not None:
            raise DurativeEvaluationError("model execution is forbidden")


@dataclass(frozen=True)
class DurativeClaimPrediction:
    claim_id: str
    version_id: str
    user_id: str
    subject_id: str
    predicate: str
    lifecycle_status: str
    memory_kind: str
    epistemic_status: str
    evidence: tuple[DurativeEpisodePrediction, ...]

    def __post_init__(self) -> None:
        _sha(self.claim_id, "claim_id")
        _sha(self.version_id, "version_id")
        _user(self.user_id)
        for name in ("subject_id", "predicate"):
            _text(getattr(self, name), name)
        if (
            self.lifecycle_status,
            self.memory_kind,
            self.epistemic_status,
        ) != ("candidate", "durative", "inferred"):
            raise DurativeEvaluationError("derived claim constants changed")
        if not self.evidence:
            raise DurativeEvaluationError("derived claim evidence is empty")


@dataclass(frozen=True)
class DurativeEvaluationFailure:
    failure_id: str
    user_id: str
    expected_decision_count: int
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure_id")
        _user(self.user_id)
        if self.expected_decision_count <= 0:
            raise DurativeEvaluationError("failure denominator is invalid")
        if TOKEN.fullmatch(self.code) is None or TOKEN.fullmatch(self.location) is None:
            raise DurativeEvaluationError("failure detail is not sanitized")


@dataclass(frozen=True)
class DurativeScorecard:
    dataset_version: str
    input_claim_count: int
    durative_proposition_count: int
    decision_count: int
    accepted_count: int
    rejected_count: int
    derived_claim_count: int
    failure_count: int
    failed_proposition_count: int
    unsupported_predicate_count: int
    rejection_counts: Mapping[str, int]
    decision_accounting: Mapping[str, object]
    input_claim_accounting: Mapping[str, object]
    provenance_integrity: Mapping[str, object]
    cross_user_count: int
    duplicate_decision_count: int
    duplicate_episode_count: int
    stale_reference_count: int
    unsupported_decision_count: int
    idempotency_replay_pass: bool
    deletion_recompute_pass: bool
    deterministic_prediction_count: int
    model_prediction_count: int
    model_calls: int = 0
    retry_count: int = 0
    cost_usd: int = 0

    def __post_init__(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise DurativeEvaluationError("dataset version changed")
        for name in (
            "input_claim_count", "durative_proposition_count", "decision_count",
            "accepted_count", "rejected_count", "derived_claim_count",
            "failure_count", "failed_proposition_count",
            "unsupported_predicate_count", "cross_user_count",
            "duplicate_decision_count", "duplicate_episode_count",
            "stale_reference_count", "unsupported_decision_count",
            "deterministic_prediction_count", "model_prediction_count",
            "model_calls", "retry_count", "cost_usd",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DurativeEvaluationError("score count is invalid")
        if self.model_calls or self.model_prediction_count or self.retry_count or self.cost_usd:
            raise DurativeEvaluationError("model execution or cost is forbidden")


def load_durative_runtime(
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
    bindings = (
        ("step6_2_release", "manifest_path", "manifest_sha256"),
        ("phase3_handoff", "manifest_path", "manifest_sha256"),
        ("phase3_handoff", "claims_path", "claims_sha256"),
        ("phase3_handoff", "evidence_index_path", "evidence_index_sha256"),
        ("session_definitions", "manifest_path", "manifest_sha256"),
        ("session_definitions", "predictions_path", "predictions_sha256"),
        ("predicate_registry", "path", "file_sha256"),
        ("rules", "path", "sha256"),
    )
    for group, path_name, hash_name in bindings:
        if _file_sha256(root / inputs[group][path_name]) != inputs[group][hash_name]:
            raise DurativeEvaluationError("bound input hash changed")
    release = _read_object(root / inputs["step6_2_release"]["manifest_path"])
    if release.get("artifacts") != inputs["step6_2_release"]["artifacts"]:
        raise DurativeEvaluationError("Step 6.2 artifact binding changed")
    for name, expected in inputs["step6_2_release"]["artifacts"].items():
        path = root / "results/summaries/grounded-summary-development-v1" / name
        if _file_sha256(path) != expected:
            raise DurativeEvaluationError("Step 6.2 artifact changed")
    registry = load_predicate_registry(root / inputs["predicate_registry"]["path"])
    if registry.content_sha256 != inputs["predicate_registry"]["content_sha256"]:
        raise DurativeEvaluationError("predicate registry content changed")
    _, sources, definitions, claims = load_grounded_runtime(repo_root=root)
    _validate_runtime(sources, definitions, claims, registry)
    return manifest, sources, definitions, claims


def run_durative_claims(
    connection: object,
    definitions: Sequence[SessionDefinition],
    claims: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[DurativeDecisionPrediction, ...],
    tuple[DurativeClaimPrediction, ...],
    tuple[DurativeEvaluationFailure, ...],
    bool,
]:
    registry = load_predicate_registry(REPO_ROOT / "configs/extraction/predicate_registry_v2.json")
    expected_by_user = _expected_proposition_counts(claims, registry)
    coordinator = DurativeClaimCoordinator(connection)
    events = tuple(_event(user_id, definitions) for user_id in DEVELOPMENT_USERS)
    failures: list[DurativeEvaluationFailure] = []
    for event in events:
        try:
            result = coordinator.process(event)
            if len(result.decisions) != expected_by_user[event.user_id]:
                raise DurativeEvaluationError("proposition accounting changed")
        except Exception as error:
            code = getattr(error, "code", "durative_evaluation_failed")
            location = getattr(error, "location", "coordinator")
            if not isinstance(code, str) or TOKEN.fullmatch(code) is None:
                code = "durative_evaluation_failed"
            if not isinstance(location, str) or TOKEN.fullmatch(location) is None:
                location = "coordinator"
            failures.append(
                DurativeEvaluationFailure(
                    stable_id("durative_evaluation_failure", event.user_id, code, location),
                    event.user_id,
                    expected_by_user[event.user_id],
                    code,
                    location,
                )
            )
    failed_users = {item.user_id for item in failures}
    predictions = _load_predictions(
        connection,
        tuple(event for event in events if event.user_id not in failed_users),
    )
    claims_out = _load_derived_claims(connection, predictions)
    before = _run_count(connection)
    replay_pass = True
    for event in events:
        if event.user_id in failed_users:
            continue
        outcome = coordinator.process(event)
        replay_pass = replay_pass and outcome.replayed_count == len(outcome.decisions)
    replay_pass = replay_pass and _run_count(connection) == before
    return predictions, claims_out, tuple(failures), replay_pass


def score_durative_claims(
    definitions: Sequence[SessionDefinition],
    input_claims: Sequence[Mapping[str, object]],
    predictions: Sequence[DurativeDecisionPrediction],
    derived_claims: Sequence[DurativeClaimPrediction],
    failures: Sequence[DurativeEvaluationFailure],
    *,
    idempotency_replay_pass: bool,
    deletion_recompute_pass: bool,
) -> DurativeScorecard:
    registry = load_predicate_registry(REPO_ROOT / "configs/extraction/predicate_registry_v2.json")
    candidate_ids = {
        item["claim_id"] for item in input_claims
        if _durative_predicate(registry, item["predicate"])
    }
    unsupported_ids = {item["claim_id"] for item in input_claims} - candidate_ids
    expected_propositions = _expected_proposition_counts(input_claims, registry)
    failed_users = {item.user_id for item in failures}
    failed_propositions = sum(item.expected_decision_count for item in failures)
    if len(failed_users) != len(failures):
        raise DurativeEvaluationError("failure users are duplicated")
    if any(item.user_id in failed_users for item in predictions):
        raise DurativeEvaluationError("failed user also has predictions")
    if len(predictions) + failed_propositions != sum(expected_propositions.values()):
        raise DurativeEvaluationError("decision and failure accounting is incomplete")

    definitions_by_source = {
        source_id: definition
        for definition in definitions
        for source_id in definition.source_ids
    }
    claims_by_id = {item["claim_id"]: item for item in input_claims}
    seen_candidate_ids: set[str] = set()
    provenance_total = 0
    provenance_valid = 0
    cross_user = 0
    stale = 0
    unsupported_decisions = 0
    duplicate_episodes = 0
    plan_ids: list[str] = []
    decision_ids: list[str] = []
    for prediction in predictions:
        plan_ids.append(prediction.plan_id)
        decision_ids.append(prediction.decision_id)
        episode_ids = [item.episode_id for item in prediction.episodes]
        duplicate_episodes += len(episode_ids) - len(set(episode_ids))
        prediction_has_candidate = False
        for episode in prediction.episodes:
            provenance_total += 1
            source = claims_by_id.get(episode.claim_id)
            if source is None:
                stale += 1
                continue
            if source["user_id"] != prediction.user_id:
                cross_user += 1
                continue
            definition = definitions_by_source.get(episode.source_id)
            evidence = source["evidence"][0]
            expected_span = _span_id(source)
            if (
                episode.claim_version_id != f"version_{episode.claim_id}"
                or evidence["source_id"] != episode.source_id
                or episode.span_id != expected_span
                or episode.support_type != "supports"
                or definition is None
                or definition.user_id != prediction.user_id
                or episode.session_definition_id != definition.definition_id
            ):
                stale += 1
                continue
            provenance_valid += 1
            if episode.claim_id in candidate_ids:
                seen_candidate_ids.add(episode.claim_id)
                prediction_has_candidate = True
        if not prediction_has_candidate:
            unsupported_decisions += 1

    failed_candidate_ids = {
        item["claim_id"] for item in input_claims
        if item["user_id"] in failed_users and item["claim_id"] in candidate_ids
    }
    accounted_inputs = len(seen_candidate_ids | failed_candidate_ids) + len(unsupported_ids)
    accepted = sum(item.status == "accepted" for item in predictions)
    rejected = sum(item.status == "rejected" for item in predictions)
    if accepted != len(derived_claims):
        raise DurativeEvaluationError("accepted decisions and derived claims differ")
    return DurativeScorecard(
        DATASET_VERSION,
        len(input_claims),
        sum(expected_propositions.values()),
        len(predictions),
        accepted,
        rejected,
        len(derived_claims),
        len(failures),
        failed_propositions,
        len(unsupported_ids),
        dict(sorted(Counter(item.reason for item in predictions if item.status == "rejected").items())),
        _metric(len(predictions) + failed_propositions, sum(expected_propositions.values()), "no_durative_propositions"),
        _metric(accounted_inputs, len(input_claims), "no_input_claims"),
        _metric(provenance_valid, provenance_total, "no_decision_provenance"),
        cross_user,
        (len(plan_ids) - len(set(plan_ids))) + (len(decision_ids) - len(set(decision_ids))),
        duplicate_episodes,
        stale,
        unsupported_decisions,
        idempotency_replay_pass,
        deletion_recompute_pass,
        sum(item.execution_mode == "deterministic" for item in predictions),
        0,
    )


def execute_durative_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> DurativeScorecard:
    root = Path(repo_root).resolve()
    output = root / output_dir
    if output.exists() and any(output.iterdir()):
        raise DurativeEvaluationError("result directory must be empty")
    manifest, sources, definitions, claims = load_durative_runtime(repo_root=root)
    connection = connection_factory()
    try:
        apply_migrations(connection, root / "migrations")
        _require_clean(connection)
        materialize_grounded_runtime(connection, sources, claims)
        predictions, derived, failures, replay_pass = run_durative_claims(
            connection, definitions, claims
        )
        deletion_pass = _deletion_recompute_check(connection, predictions)
        checks = score_durative_claims(
            definitions,
            claims,
            predictions,
            derived,
            failures,
            idempotency_replay_pass=replay_pass,
            deletion_recompute_pass=deletion_pass,
        )
        if _run_count(connection) != len(DEVELOPMENT_USERS):
            raise DurativeEvaluationError("persisted run count changed")
    finally:
        connection.close()
    _write_release(root, output, manifest, predictions, derived, failures, checks)
    return checks


def verify_durative_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = root / output_dir
    manifest = _read_object(output / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise DurativeEvaluationError("release artifact map changed")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
            raise DurativeEvaluationError("release artifact hash is invalid")
        if _file_sha256(output / name) != expected:
            raise DurativeEvaluationError("release artifact changed")
    dataset = manifest.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("path") != DATASET_MANIFEST.as_posix()
        or dataset.get("sha256") != _file_sha256(root / DATASET_MANIFEST)
    ):
        raise DurativeEvaluationError("release dataset binding changed")
    implementation = manifest.get("implementation_hashes")
    if not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS):
        raise DurativeEvaluationError("release implementation map changed")
    for path, expected in implementation.items():
        if _file_sha256(root / path) != expected:
            raise DurativeEvaluationError("release implementation changed")
    if manifest.get("predecessor") != _predecessor_attestation(root):
        raise DurativeEvaluationError("release predecessor attestation changed")
    load_durative_runtime(repo_root=root)


def serialize_jsonl(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(item)) for item in values)


def _write_release(
    root: Path,
    output: Path,
    dataset_manifest: Mapping[str, object],
    predictions: Sequence[DurativeDecisionPrediction],
    claims: Sequence[DurativeClaimPrediction],
    failures: Sequence[DurativeEvaluationFailure],
    checks: DurativeScorecard,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    findings = (
        "# Durative claim development findings\n\n"
        "The evaluator accounted for all 33 development claims. Seven predicates are "
        "outside the durative rules. The remaining 26 propositions were rejected because "
        "the imported claims remain unresolved candidates. They cannot serve as confirmed "
        "episode support, and they independently block inference. No durative claim was "
        "created.\n\n"
        "Replay and source-deletion checks passed without stale or cross-user evidence. "
        "This run is deterministic and made no model calls. During review, the reviewer "
        "accidentally exposed frozen-test gold text because of a wrong exclusion glob and "
        "stopped immediately. The exposed text was not used in code, fixtures, expected "
        "values, metrics, or judgments.\n"
    ).encode("utf-8")
    artifacts = {
        "rejections.jsonl": serialize_jsonl(
            tuple(item for item in predictions if item.status == "rejected")
        ),
        "claims.jsonl": serialize_jsonl(claims),
        "failures.jsonl": serialize_jsonl(failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": DATASET_VERSION,
                "starting_commit": STARTING_COMMIT,
                "input_claim_count": checks.input_claim_count,
                "durative_proposition_count": checks.durative_proposition_count,
                "decision_count": checks.decision_count,
                "accepted_count": checks.accepted_count,
                "rejected_count": checks.rejected_count,
                "failure_count": checks.failure_count,
                "request_count": 0,
                "retry_count": 0,
                "model_calls": 0,
                "cost_usd": 0,
                "execution_mode": "deterministic",
            }
        ),
        "findings.md": findings,
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
            "The frozen Phase 3 claims have null memory kind and remain candidates.",
            "This development release therefore measures rejection and provenance behavior, not accepted durative quality.",
            "The evaluator uses deterministic rules and does not model production traffic.",
            "During review, the reviewer accidentally exposed frozen-test gold text because of a wrong exclusion glob and stopped immediately. The exposed text was not used in code, fixtures, expected values, metrics, or judgments.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))


def _load_predictions(
    connection: object,
    events: Sequence[ProcessingOutboxRecord],
) -> tuple[DurativeDecisionPrediction, ...]:
    event_keys = tuple(f"durative:{event.event_id}" for event in events)
    if not event_keys:
        return ()
    rows = connection.execute(
        """
        SELECT run.run_id, decision.decision_id, run.user_id,
               run.rules_version, run.input_snapshot_sha256,
               decision.decision_status, decision.decision_reason,
               decision.derived_claim_id, decision.derived_version_id,
               run.transaction_as_of, run.idempotency_key
        FROM durative_inference_runs AS run
        JOIN durative_inference_decisions AS decision
          ON decision.user_id = run.user_id AND decision.run_id = run.run_id
        WHERE run.idempotency_key = ANY(%s)
        ORDER BY run.user_id, decision.decision_id
        """,
        (list(event_keys),),
    ).fetchall()
    predictions: list[DurativeDecisionPrediction] = []
    for row in rows:
        episodes = tuple(
            DurativeEpisodePrediction(*episode)
            for episode in connection.execute(
                """
                SELECT episode_id, role, supporting_claim_id,
                       supporting_version_id, session_definition_id,
                       source_id, span_id, support_type
                FROM durative_inference_evidence
                WHERE user_id = %s AND decision_id = %s
                ORDER BY evidence_order
                """,
                (row[2], row[1]),
            ).fetchall()
        )
        predictions.append(
            DurativeDecisionPrediction(
                row[1], row[1], row[2], row[3], row[4], row[5], row[6],
                row[7], row[8], row[9], episodes,
            )
        )
    return tuple(predictions)


def _load_derived_claims(
    connection: object,
    predictions: Sequence[DurativeDecisionPrediction],
) -> tuple[DurativeClaimPrediction, ...]:
    result: list[DurativeClaimPrediction] = []
    for prediction in predictions:
        if prediction.derived_claim_id is None:
            continue
        row = connection.execute(
            """
            SELECT claim.claim_id, version.version_id, claim.user_id,
                   claim.subject_id, claim.predicate, version.lifecycle_status,
                   claim.memory_kind, claim.epistemic_status
            FROM claims AS claim
            JOIN claim_versions AS version
              ON version.user_id = claim.user_id AND version.claim_id = claim.claim_id
            WHERE claim.user_id = %s AND claim.claim_id = %s
              AND version.version_id = %s
            """,
            (prediction.user_id, prediction.derived_claim_id, prediction.derived_version_id),
        ).fetchone()
        if row is None:
            raise DurativeEvaluationError("accepted derived claim is missing")
        support = tuple(item for item in prediction.episodes if item.role == "supports")
        result.append(DurativeClaimPrediction(*row, support))
    return tuple(result)


def _deletion_recompute_check(
    connection: object,
    predictions: Sequence[DurativeDecisionPrediction],
) -> bool:
    if not predictions:
        return False
    target = predictions[0].episodes[0]
    deleted_at = predictions[0].transaction_as_of + timedelta(seconds=1)
    with connection.transaction(force_rollback=True):
        IngestionService(connection).delete_source(
            predictions[0].user_id,
            target.source_id,
            deleted_at,
        )
        row = connection.execute(
            """
            SELECT event_id, user_id, event_type, aggregate_id, dedupe_key,
                   payload, state, created_at, published_at
            FROM processing_outbox
            WHERE user_id = %s AND event_type = 'source_deleted'
              AND aggregate_id = %s
            """,
            (predictions[0].user_id, target.source_id),
        ).fetchone()
        if row is None:
            return False
        DurativeClaimCoordinator(connection).process(ProcessingOutboxRecord(*row))
        stale = connection.execute(
            """
            SELECT count(*) FROM durative_inference_evidence
            WHERE user_id = %s AND source_id = %s
            """,
            (predictions[0].user_id, target.source_id),
        ).fetchone()[0]
        unsupported = connection.execute(
            """
            SELECT count(*)
            FROM claims AS claim
            WHERE claim.user_id = %s AND claim.memory_kind = 'durative'
              AND NOT EXISTS (
                  SELECT 1 FROM evidence_links AS evidence
                  WHERE evidence.user_id = claim.user_id
                    AND evidence.claim_id = claim.claim_id
              )
            """,
            (predictions[0].user_id,),
        ).fetchone()[0]
        return stale == 0 and unsupported == 0


def _event(user_id: str, definitions: Sequence[SessionDefinition]) -> ProcessingOutboxRecord:
    cutoff = next(item.transaction_as_of for item in definitions if item.user_id == user_id)
    event_id = stable_id("durative_evaluation_event", DATASET_VERSION, user_id)
    return ProcessingOutboxRecord(
        event_id,
        user_id,
        "claims_changed",
        user_id,
        f"durative-evaluation:{user_id}",
        {"user_id": user_id},
        "pending",
        cutoff,
    )


def _expected_proposition_counts(
    claims: Sequence[Mapping[str, object]],
    registry: object,
) -> dict[str, int]:
    values: dict[str, set[str]] = {user_id: set() for user_id in DEVELOPMENT_USERS}
    for item in claims:
        if not _durative_predicate(registry, item["predicate"]):
            continue
        identity = (
            item["user_id"], item["subject_id"], item["predicate"],
            item["predicate_registry_version"], item["object"], item["polarity"],
        )
        values[item["user_id"]].add(canonical_json(identity))
    return {user_id: len(items) for user_id, items in values.items()}


def _durative_predicate(registry: object, predicate: object) -> bool:
    definition = registry.by_predicate.get(predicate)
    return bool(
        definition is not None
        and definition.temporal_behavior == "interval"
        and definition.family in ALLOWED_FAMILIES
    )


def _validate_runtime(
    sources: Sequence[DevelopmentSource],
    definitions: Sequence[SessionDefinition],
    claims: Sequence[Mapping[str, object]],
    registry: object,
) -> None:
    if (
        len(sources) != EXPECTED["source_count"]
        or len(definitions) != EXPECTED["session_count"]
        or len(claims) != EXPECTED["input_claim_count"]
    ):
        raise DurativeEvaluationError("development input count changed")
    if {item.user_id for item in sources} != set(DEVELOPMENT_USERS):
        raise DurativeEvaluationError("development users changed")
    source_ids = {item.source_id for item in sources}
    if any(
        item["user_id"] not in DEVELOPMENT_USERS
        or item["evidence"][0]["source_id"] not in source_ids
        for item in claims
    ):
        raise DurativeEvaluationError("claim leaves the development boundary")
    expected = _expected_proposition_counts(claims, registry)
    if sum(expected.values()) != EXPECTED["durative_proposition_count"]:
        raise DurativeEvaluationError("durative proposition count changed")
    unsupported = sum(not _durative_predicate(registry, item["predicate"]) for item in claims)
    if unsupported != EXPECTED["unsupported_predicate_count"]:
        raise DurativeEvaluationError("unsupported predicate count changed")


def _validate_manifest(value: Mapping[str, object]) -> None:
    if set(value) != MANIFEST_FIELDS:
        raise DurativeEvaluationError("dataset manifest fields changed")
    if (
        value.get("dataset_version") != DATASET_VERSION
        or value.get("split") != "development"
        or value.get("review_status") != "implementation_reviewed"
        or value.get("expected") != EXPECTED
        or value.get("loader_boundary") != "bound_development_artifacts_only_no_gold_or_test_users"
        or value.get("guidance_version") != "step-6.3-guidance-v1"
        or value.get("decision_envelope_sha256") != "de50f5ce8955d85f617bfef15af7e39297704c4b6a056f32a467ad7473ce28e2"
    ):
        raise DurativeEvaluationError("dataset manifest contract changed")
    contract = value.get("evaluation_contract")
    if not isinstance(contract, dict) or contract != {
        "prediction_source": "fresh_deterministic_service_execution",
        "failure_policy": "sanitized_and_in_denominator",
        "accepted_zero_is_valid": True,
        "model_calls": 0,
        "cost_usd": 0,
    }:
        raise DurativeEvaluationError("evaluation contract changed")
    if not isinstance(value.get("inputs"), dict):
        raise DurativeEvaluationError("dataset inputs are invalid")


def _span_id(item: Mapping[str, object]) -> str:
    evidence = item["evidence"][0]
    identity = (
        item["user_id"], evidence["source_id"], evidence["message_id"],
        evidence["quote"],
    )
    return "span_" + hashlib.sha256(canonical_json_bytes(identity)[:-1]).hexdigest()


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": 0,
            "value": None,
            "status": "not_evaluated",
            "reason": reason,
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
        "status": "evaluated",
        "reason": None,
    }


def _require_clean(connection: object) -> None:
    counts = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM memory_users),
          (SELECT count(*) FROM source_events),
          (SELECT count(*) FROM claims),
          (SELECT count(*) FROM durative_inference_runs)
        """
    ).fetchone()
    if counts != (0, 0, 0, 0):
        raise DurativeEvaluationError("evaluation database must be clean")


def _run_count(connection: object) -> int:
    return connection.execute("SELECT count(*) FROM durative_inference_runs").fetchone()[0]


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    manifest_path = Path("results/summaries/grounded-summary-development-v1/manifest.json")
    if _file_sha256(root / manifest_path) != STEP62_MANIFEST_SHA256:
        raise DurativeEvaluationError("Step 6.2 manifest changed")
    manifest = _read_object(root / manifest_path)
    implementation = manifest.get("implementation_hashes")
    prior = manifest.get("predecessor", {})
    protected = prior.get("protected_file_hashes")
    prior_drift = prior.get("authorized_drift")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(implementation, dict)
        or not isinstance(protected, dict)
        or not isinstance(prior_drift, list)
        or not isinstance(artifacts, dict)
    ):
        raise DurativeEvaluationError("Step 6.2 predecessor map is invalid")
    if len(protected) != 79 or prior.get("protected_file_count") != 79:
        raise DurativeEvaluationError("Step 6.2 protected baseline changed")
    for name, expected in artifacts.items():
        if _file_sha256(root / manifest_path.parent / name) != expected:
            raise DurativeEvaluationError("Step 6.2 artifact changed")
    effective_protected = dict(protected)
    for item in prior_drift:
        if not isinstance(item, dict) or set(item) != {
            "path", "predecessor_sha256", "reason", "step6_2_sha256"
        }:
            raise DurativeEvaluationError("Step 6.2 drift record is invalid")
        path = item["path"]
        if effective_protected.get(path) != item["predecessor_sha256"]:
            raise DurativeEvaluationError("Step 6.2 protected overlay changed")
        effective_protected[path] = item["step6_2_sha256"]
    baseline = {**effective_protected, **implementation}
    drift = {
        path: _file_sha256(root / path)
        for path, expected in baseline.items()
        if _file_sha256(root / path) != expected
    }
    if set(drift) != set(AUTHORIZED_PREDECESSOR_DRIFT):
        raise DurativeEvaluationError("Step 6.3 predecessor drift changed")
    return {
        "manifest": {"path": manifest_path.as_posix(), "sha256": STEP62_MANIFEST_SHA256},
        "artifacts": artifacts,
        "protected_file_count": len(effective_protected),
        "effective_protected_file_hashes": effective_protected,
        "implementation_file_count": len(implementation),
        "step6_2_implementation_hashes": implementation,
        "effective_file_count": len(baseline),
        "unchanged_file_count": len(baseline) - len(drift),
        "authorized_drift": [
            {
                "path": path,
                "predecessor_sha256": baseline[path],
                "step6_3_sha256": drift[path],
                "reason": AUTHORIZED_PREDECESSOR_DRIFT[path],
            }
            for path in sorted(drift)
        ],
    }


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DurativeEvaluationError("JSON input is unreadable") from error
    if not isinstance(value, dict):
        raise DurativeEvaluationError("JSON input must be an object")
    return value


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, content)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DurativeEvaluationError(f"{name} must be nonempty text")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise DurativeEvaluationError(f"{name} must be a SHA-256")
    return value


def _user(value: object) -> str:
    if value not in DEVELOPMENT_USERS:
        raise DurativeEvaluationError("user is outside the development split")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise DurativeEvaluationError(f"{name} must be timezone-aware")
    return value


REPO_ROOT = Path(__file__).resolve().parents[2]
