"""Leakage-safe deterministic evaluation for Step 5.3 belief resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from storage.migrations import apply_migrations
from temporal.contracts import TemporalClaim, TemporalQuery
from temporal.service import TemporalService

from .candidates import ELIGIBLE_STATUSES
from .classifier import CLASSIFIER_VERSION, ClassificationRequest, ConflictClassifier
from .relation_evaluation import (
    CASE_IDS,
    DATASET_ROOT as RELATION_DATASET_ROOT,
    RelationPrediction,
    RelationPredictionRelation,
    RelationRuntimeCase,
    _derive_pair_speakers_from_runtime_evidence,
    _insert_runtime_case,
    _load_reference_execution_catalog,
    file_sha256,
    load_relation_dataset_runtime,
)
from .relations import ConflictRelationService
from .resolution import BeliefResolutionService
from .resolver import POLICY_VERSION, RESOLVER_VERSION, ResolutionRequest


DATASET_VERSION = "belief_resolution_development_v1"
SPLIT = "development"
DATASET_ROOT = Path("data/conflicts/belief-resolution-development-v1")
RESULT_ROOT = Path("results/conflicts/belief-resolution-development-v1")
STEP52_RESULT_ROOT = Path("results/conflicts/relation-classification-development-v1")
DATASET_MANIFEST_SHA256 = "0c0903606fe25793c9c76f4eb699f5a59c75f294075e4547bb69812e809768d7"
RUNTIME_SHA256 = "f2d92261abd6a7bf49616068e7738ea4929cf8bcfaf0a85f1ec5252558421692"
GOLD_SHA256 = "f76b7bafd0a1796a0a7474cc222384c3f81716c44d5b5db71e29472b9a659276"
STEP52_RUNTIME_SHA256 = "9a298284bc25a155954be6e20e7807541f9638fbfd50640060878b7a73b5fe0e"
STEP52_PREDICTIONS_SHA256 = "f41a15fa83df9602c0536976aaebbc7aef9d7e33d1353f057dbeb52a6b3cf201"
STEP52_MANIFEST_SHA256 = "2f29edd580957192cc7808e2e4a454b7fa7c0b89bfc9ea3851efbf9e1c76179e"
PRESENT_RELATION_LABELS = {
    "temporal_change": 1,
    "unrelated": 6,
    "unresolved_ambiguity": 1,
}
STEP53_ALLOWED_PREDECESSOR_DRIFT_REASONS = {
    "Makefile": "adds_step5_3_belief_resolution_test_target",
    "compose.yaml": "pins_current_pgvector_storage_service_for_step5_3_gates",
    "src/conflicts/__init__.py": "exports_step5_3_resolution_contracts",
    "src/ingestion/service.py": "invalidates_and_replays_resolution_state_on_delete",
    "src/storage/contracts.py": "adds_typed_belief_resolution_records",
    "src/storage/repository.py": "adds_user_scoped_belief_resolution_persistence",
    "src/temporal/contracts.py": "allows_candidate_to_historical_resolution_action",
    "src/temporal/service.py": "executes_resolver_lifecycle_actions_in_transaction",
    "tests/integration/test_conflict_relation_evaluation.py": "replays_step5_2_through_current_lower_seams",
    "tests/integration/test_conflict_relations.py": "expects_migration_0005_in_current_set",
    "tests/integration/test_phase4_storage.py": "expects_migration_0005_and_resolution_tables",
    "tests/integration/test_temporal_service.py": "expects_migration_0005_in_current_set",
    "tests/unit/test_conflict_candidate_evaluation.py": "allows_step5_3_predecessor_drift_superset",
    "tests/unit/test_temporal_contracts.py": "checks_only_candidate_to_historical_matrix_addition",
}
RUNTIME_FIELDS = frozenset(
    {
        "case_id", "dataset_version", "split", "user_id", "relation_case_id",
        "pair_id", "decision_id", "decision_snapshot_sha256", "label",
        "valid_at_strategy", "resolved_offset_seconds", "idempotency_key",
    }
)
GOLD_FIELDS = frozenset(
    {
        "case_id", "dataset_version", "split", "user_id", "review_status",
        "expected_outcome", "expected_selected_current_claim_id",
        "expected_actions", "expected_historical_claim_ids",
        "expected_disputed_claim_ids", "expected_superseded_claim_ids",
        "expected_source_ids",
    }
)
ACTION_FIELDS = frozenset(
    {"action_order", "claim_id", "target_status", "replacement_claim_id"}
)
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version", "review_status", "case_count", "user_counts",
        "case_ids", "present_relation_labels", "runtime", "gold",
        "relation_runtime", "relation_predictions", "relation_release",
        "resolver_config", "belief_migration", "review_method", "source_boundary",
    }
)
SANITIZED_TOKEN = re.compile(r"^[a-z0-9_:-]+$")


class ResolutionEvaluationError(ValueError):
    """Reject malformed inputs, leakage, or incomplete accounting."""


class _RollbackResolutionCase(Exception):
    def __init__(self, prediction: "ResolutionPrediction") -> None:
        self.prediction = prediction


@dataclass(frozen=True)
class ResolutionRuntimeCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    relation_case_id: str
    pair_id: str
    decision_id: str
    decision_snapshot_sha256: str
    label: str
    valid_at_strategy: str
    resolved_offset_seconds: int
    idempotency_key: str
    relation_case: RelationRuntimeCase
    relation_prediction: RelationPrediction


@dataclass(frozen=True)
class ResolutionAction:
    action_order: int
    claim_id: str
    target_status: str
    replacement_claim_id: str | None


@dataclass(frozen=True)
class ResolutionGoldCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    review_status: str
    expected_outcome: str
    expected_selected_current_claim_id: str | None
    expected_actions: tuple[ResolutionAction, ...]
    expected_historical_claim_ids: tuple[str, ...]
    expected_disputed_claim_ids: tuple[str, ...]
    expected_superseded_claim_ids: tuple[str, ...]
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionPredictionAction:
    action_id: str
    action_order: int
    claim_id: str
    from_status: str
    target_status: str
    replacement_claim_id: str | None
    reason: str


@dataclass(frozen=True)
class ResolutionPrediction:
    case_id: str
    user_id: str
    pair_id: str
    decision_id: str
    resolution_id: str
    resolver_version: str
    policy_version: str
    input_snapshot_sha256: str
    outcome: str
    selected_current_claim_id: str | None
    actions: tuple[ResolutionPredictionAction, ...]
    current_claim_ids: tuple[str, ...]
    historical_claim_ids: tuple[str, ...]
    disputed_claim_ids: tuple[str, ...]
    superseded_claim_ids: tuple[str, ...]
    decision_evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    belief_confidence: None
    execution_mode: str = "deterministic"
    model: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "case_id", "user_id", "pair_id", "decision_id", "resolution_id",
            "resolver_version", "policy_version", "input_snapshot_sha256",
        ):
            _token(getattr(self, name), name)
        if self.resolver_version != RESOLVER_VERSION or self.policy_version != POLICY_VERSION:
            raise ResolutionEvaluationError("prediction resolver identity changed")
        if self.execution_mode != "deterministic" or self.model is not None:
            raise ResolutionEvaluationError("resolution prediction must be deterministic")
        if self.belief_confidence is not None:
            raise ResolutionEvaluationError("belief confidence must remain null")
        for name in (
            "current_claim_ids", "historical_claim_ids", "disputed_claim_ids",
            "superseded_claim_ids", "decision_evidence_ids", "source_ids",
            "relation_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ResolutionEvaluationError(f"{name} must be canonical")


@dataclass(frozen=True)
class ResolutionFailure:
    failure_id: str
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "case_id", "user_id", "code", "location"):
            _token(getattr(self, name), name)


@dataclass(frozen=True)
class ResolutionScorecard:
    dataset_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    failure_rate: Mapping[str, object]
    exact_outcome_accuracy: Mapping[str, object]
    exact_action_accuracy: Mapping[str, object]
    current_selection_accuracy: Mapping[str, object]
    historical_preservation_accuracy: Mapping[str, object]
    dispute_accuracy: Mapping[str, object]
    no_change_accuracy: Mapping[str, object]
    supersession_accuracy: Mapping[str, object]
    evidence_trace_coverage: Mapping[str, object]
    cross_user_count: int
    deterministic_prediction_count: int
    model_prediction_count: int
    model_fallback: bool
    exact_match_gate_passed: bool


def load_resolution_dataset_runtime(
    dataset_root: str | Path,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], tuple[ResolutionRuntimeCase, ...]]:
    """Load runtime bindings without opening or hashing the new gold file."""

    root = Path(repo_root).resolve()
    dataset = Path(dataset_root).resolve()
    manifest = _read_json_object(dataset / "manifest.json", "dataset manifest")
    _exact(manifest, MANIFEST_FIELDS, "dataset manifest")
    if (
        manifest["dataset_version"] != DATASET_VERSION
        or manifest["review_status"] != "approved"
        or manifest["case_count"] != 8
        or manifest["user_counts"] != {"user_001": 4, "user_002": 4}
        or tuple(manifest["case_ids"]) != CASE_IDS
        or manifest["present_relation_labels"] != PRESENT_RELATION_LABELS
        or manifest["review_method"]
        != "manual_review_of_frozen_step5_2_runtime_and_predictions"
        or manifest["source_boundary"]
        != "step5_2_runtime_and_predictions_only_no_step5_2_or_older_gold_oracle_review_or_test_users"
    ):
        raise ResolutionEvaluationError("resolution dataset contract changed")
    bindings = (
        "runtime", "gold", "relation_runtime", "relation_predictions",
        "relation_release", "resolver_config", "belief_migration",
    )
    for name in bindings:
        value = manifest[name]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ResolutionEvaluationError(f"manifest {name} binding changed")
        if not _sha256_text(value["sha256"]):
            raise ResolutionEvaluationError(f"manifest {name} hash is invalid")
    _require_hash(dataset / str(manifest["runtime"]["path"]), RUNTIME_SHA256)
    expected = {
        "relation_runtime": STEP52_RUNTIME_SHA256,
        "relation_predictions": STEP52_PREDICTIONS_SHA256,
        "relation_release": STEP52_MANIFEST_SHA256,
    }
    for name in bindings[2:]:
        binding = manifest[name]
        expected_hash = expected.get(name, str(binding["sha256"]))
        if binding["sha256"] != expected_hash:
            raise ResolutionEvaluationError(f"manifest {name} frozen hash changed")
        _require_hash(root / str(binding["path"]), expected_hash)
    relation_manifest, relation_cases = load_relation_dataset_runtime(
        root / RELATION_DATASET_ROOT, repo_root=root
    )
    if relation_manifest["runtime"]["sha256"] != STEP52_RUNTIME_SHA256:
        raise ResolutionEvaluationError("Step 5.2 runtime binding changed")
    relation_predictions = _load_relation_predictions(
        root / str(manifest["relation_predictions"]["path"])
    )
    cases = load_resolution_runtime(
        dataset / str(manifest["runtime"]["path"]),
        relation_cases=relation_cases,
        relation_predictions=relation_predictions,
    )
    return manifest, cases


def load_resolution_runtime(
    path: str | Path,
    *,
    relation_cases: Sequence[RelationRuntimeCase],
    relation_predictions: Mapping[str, RelationPrediction],
) -> tuple[ResolutionRuntimeCase, ...]:
    case_map = {case.case_id: case for case in relation_cases}
    result: list[ResolutionRuntimeCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, RUNTIME_FIELDS, f"runtime[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        relation_case = case_map.get(case_id)
        prediction = relation_predictions.get(case_id)
        if (
            raw["dataset_version"] != DATASET_VERSION
            or raw["split"] != SPLIT
            or raw["relation_case_id"] != case_id
            or relation_case is None
            or prediction is None
            or relation_case.user_id != user_id
            or prediction.user_id != user_id
            or raw["pair_id"] != relation_case.pair.pair_id
            or raw["pair_id"] != prediction.pair_id
            or raw["decision_id"] != prediction.decision_id
            or raw["decision_snapshot_sha256"] != prediction.input_snapshot_sha256
            or raw["label"] != prediction.label
            or raw["valid_at_strategy"]
            != ("later_interval_start" if prediction.label == "temporal_change" else "none")
            or raw["resolved_offset_seconds"] != 1
            or raw["idempotency_key"] != f"belief_eval:{case_id}"
        ):
            raise ResolutionEvaluationError("runtime is not the exact Step 5.2 snapshot")
        result.append(
            ResolutionRuntimeCase(
                case_id, DATASET_VERSION, SPLIT, user_id, case_id,
                raw["pair_id"], raw["decision_id"],
                raw["decision_snapshot_sha256"], raw["label"],
                raw["valid_at_strategy"], 1, raw["idempotency_key"],
                relation_case, prediction,
            )
        )
    _validate_case_order(result)
    return tuple(result)


def load_resolution_gold(path: str | Path) -> tuple[ResolutionGoldCase, ...]:
    result: list[ResolutionGoldCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, GOLD_FIELDS, f"gold[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        actions_raw = raw["expected_actions"]
        if not isinstance(actions_raw, list):
            raise ResolutionEvaluationError("expected_actions must be a list")
        actions: list[ResolutionAction] = []
        for index, item in enumerate(actions_raw):
            _exact(item, ACTION_FIELDS, f"gold[{position}].actions[{index}]")
            actions.append(
                ResolutionAction(
                    item["action_order"],
                    _token(item["claim_id"], "claim_id"),
                    _token(item["target_status"], "target_status"),
                    None if item["replacement_claim_id"] is None else _token(
                        item["replacement_claim_id"], "replacement_claim_id"
                    ),
                )
            )
        canonical_fields = (
            "expected_historical_claim_ids",
            "expected_disputed_claim_ids",
            "expected_superseded_claim_ids",
            "expected_source_ids",
        )
        if any(
            not isinstance(raw[name], list)
            or raw[name] != sorted(set(raw[name]))
            for name in canonical_fields
        ):
            raise ResolutionEvaluationError("gold claim and source IDs must be canonical")
        source_ids = raw["expected_source_ids"]
        if (
            raw["dataset_version"] != DATASET_VERSION
            or raw["split"] != SPLIT
            or raw["review_status"] != "approved"
            or [item.action_order for item in actions]
            != list(range(1, len(actions) + 1))
        ):
            raise ResolutionEvaluationError("resolution gold contract changed")
        current = raw["expected_selected_current_claim_id"]
        if current is not None:
            current = _token(current, "expected_selected_current_claim_id")
        result.append(
            ResolutionGoldCase(
                case_id, DATASET_VERSION, SPLIT, user_id, "approved",
                _token(raw["expected_outcome"], "expected_outcome"), current,
                tuple(actions),
                tuple(raw["expected_historical_claim_ids"]),
                tuple(raw["expected_disputed_claim_ids"]),
                tuple(raw["expected_superseded_claim_ids"]),
                tuple(source_ids),
            )
        )
    _validate_case_order(result)
    return tuple(result)


def run_resolution_cases(
    connection: object,
    cases: Sequence[ResolutionRuntimeCase],
    *,
    repo_root: str | Path,
    reference_runtime_path: str | Path,
) -> tuple[tuple[ResolutionPrediction, ...], tuple[ResolutionFailure, ...]]:
    """Persist classification and resolution per case in an isolated transaction."""

    reference = _load_reference_execution_catalog(reference_runtime_path)
    classifier = ConflictClassifier(repo_root=repo_root)
    predictions: list[ResolutionPrediction] = []
    failures: list[ResolutionFailure] = []
    for case in cases:
        try:
            with connection.transaction():
                relation = case.relation_case
                _insert_runtime_case(connection, relation.candidate_case, reference)
                _derive_pair_speakers_from_runtime_evidence(
                    connection, case.user_id,
                    (relation.pair.left_claim_id, relation.pair.right_claim_id),
                    relation.candidate_case.transaction_as_of,
                )
                visible = {
                    item.claim.claim_id: item
                    for item in TemporalService(connection).query(
                        TemporalQuery(
                            case.user_id,
                            relation.candidate_case.transaction_as_of,
                            statuses=frozenset(ELIGIBLE_STATUSES),
                        )
                    )
                }
                left = visible.get(relation.pair.left_claim_id)
                right = visible.get(relation.pair.right_claim_id)
                if left is None or right is None:
                    raise ResolutionEvaluationError("paired claim is not visible")
                ingestion = tuple(
                    connection.execute(
                        """
                        SELECT source_id, ingested_at FROM source_events
                        WHERE user_id = %s AND source_id = ANY(%s)
                        ORDER BY source_id
                        """,
                        (case.user_id, list(relation.pair.source_ids)),
                    ).fetchall()
                )
                request = ClassificationRequest(
                    CLASSIFIER_VERSION, case.user_id,
                    relation.candidate_case.transaction_as_of, relation.pair,
                    left, right, ingestion, relation.explicit_target_claim_id,
                )
                decision = classifier.classify(request)
                _require_frozen_decision(case.relation_prediction, decision, classifier)
                persisted_decision = ConflictRelationService(
                    connection, repo_root=repo_root
                ).persist(request, decision, relation.candidate_case.transaction_as_of)
                if persisted_decision.replayed:
                    raise ResolutionEvaluationError("fresh decision unexpectedly replayed")
                valid_at = _valid_at(case.valid_at_strategy, left, right)
                resolution = BeliefResolutionService(
                    connection, repo_root=repo_root
                ).resolve_in_transaction(
                    ResolutionRequest(
                        case.user_id,
                        decision.decision_id,
                        relation.candidate_case.transaction_as_of,
                        valid_at,
                        relation.candidate_case.transaction_as_of
                        + timedelta(seconds=case.resolved_offset_seconds),
                        case.idempotency_key,
                        RESOLVER_VERSION,
                    )
                )
                if resolution.replayed:
                    raise ResolutionEvaluationError("fresh resolution unexpectedly replayed")
                statuses = _pair_statuses(
                    connection,
                    case.user_id,
                    (relation.pair.left_claim_id, relation.pair.right_claim_id),
                )
                source_ids = tuple(
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT DISTINCT span.source_id
                        FROM belief_resolution_evidence AS lineage
                        JOIN conflict_decision_evidence AS cited
                          ON cited.user_id = lineage.user_id
                         AND cited.decision_evidence_id = lineage.decision_evidence_id
                        JOIN source_spans AS span
                          ON span.user_id = cited.user_id AND span.span_id = cited.span_id
                        WHERE lineage.user_id = %s AND lineage.resolution_id = %s
                        ORDER BY span.source_id
                        """,
                        (case.user_id, resolution.resolution.resolution_id),
                    ).fetchall()
                )
                predictions.append(
                    _prediction(case, resolution, statuses, source_ids)
                )
                raise _RollbackResolutionCase(predictions.pop())
        except _RollbackResolutionCase as completed:
            predictions.append(completed.prediction)
        except Exception:
            failures.append(
                ResolutionFailure(
                    "failure_" + hashlib.sha256(
                        f"{case.case_id}:case_execution_failed".encode("utf-8")
                    ).hexdigest(),
                    case.case_id,
                    case.user_id,
                    "case_execution_failed",
                    "case",
                )
            )
    return tuple(predictions), tuple(failures)


def score_belief_resolution(
    runtime: Sequence[ResolutionRuntimeCase],
    predictions: Sequence[ResolutionPrediction],
    failures: Sequence[ResolutionFailure],
    gold: Sequence[ResolutionGoldCase],
) -> ResolutionScorecard:
    _validate_accounting(runtime, predictions, failures)
    if [item.case_id for item in gold] != [item.case_id for item in runtime]:
        raise ResolutionEvaluationError("runtime and gold order differ")
    predicted = {item.case_id: item for item in predictions}
    outcome_hits = action_hits = evidence_hits = cross_user = 0
    current_hits = historical_hits = dispute_hits = no_change_hits = supersession_hits = 0
    current_total = historical_total = dispute_total = no_change_total = supersession_total = 0
    for case, expected in zip(runtime, gold):
        if case.user_id != expected.user_id:
            raise ResolutionEvaluationError("runtime and gold users differ")
        prediction = predicted.get(case.case_id)
        if prediction is None:
            if expected.expected_selected_current_claim_id is not None:
                current_total += 1
            historical_total += bool(expected.expected_historical_claim_ids)
            dispute_total += bool(expected.expected_disputed_claim_ids)
            no_change_total += expected.expected_outcome == "no_change"
            supersession_total += bool(expected.expected_superseded_claim_ids)
            continue
        pair_claims = {
            case.relation_case.pair.left_claim_id,
            case.relation_case.pair.right_claim_id,
        }
        if prediction.user_id != case.user_id or prediction.pair_id != case.pair_id:
            raise ResolutionEvaluationError("prediction identity differs from runtime")
        outcome_hits += prediction.outcome == expected.expected_outcome
        action_hits += _action_key(prediction.actions) == _action_key(expected.expected_actions)
        if expected.expected_selected_current_claim_id is not None:
            current_total += 1
            current_hits += (
                prediction.selected_current_claim_id
                == expected.expected_selected_current_claim_id
                and prediction.current_claim_ids
                == (expected.expected_selected_current_claim_id,)
            )
        for status, values, hits_name in (
            ("historical", prediction.historical_claim_ids, "historical"),
            ("disputed", prediction.disputed_claim_ids, "dispute"),
            ("superseded", prediction.superseded_claim_ids, "supersession"),
        ):
            expected_ids = {
                "historical": expected.expected_historical_claim_ids,
                "disputed": expected.expected_disputed_claim_ids,
                "superseded": expected.expected_superseded_claim_ids,
            }[status]
            if expected_ids:
                if hits_name == "historical":
                    historical_total += 1
                    historical_hits += values == expected_ids
                elif hits_name == "dispute":
                    dispute_total += 1
                    dispute_hits += values == expected_ids
                else:
                    supersession_total += 1
                    supersession_hits += values == expected_ids
        if expected.expected_outcome == "no_change":
            no_change_total += 1
            no_change_hits += prediction.outcome == "no_change" and not prediction.actions
        evidence_hits += (
            bool(prediction.decision_evidence_ids)
            and prediction.source_ids == expected.expected_source_ids
        )
        cross_user += sum(item.claim_id not in pair_claims for item in prediction.actions)
        cross_user += sum(
            value is not None and value not in pair_claims
            for value in (
                prediction.selected_current_claim_id,
                *prediction.current_claim_ids,
                *prediction.historical_claim_ids,
                *prediction.disputed_claim_ids,
                *prediction.superseded_claim_ids,
            )
        )
    count = len(runtime)
    failures_count = len(failures)
    score = ResolutionScorecard(
        DATASET_VERSION,
        count,
        len(predictions),
        failures_count,
        _metric(failures_count, count, "no_cases"),
        _metric(outcome_hits, count, "no_cases"),
        _metric(action_hits, count, "no_cases"),
        _metric(current_hits, current_total, "no_expected_current_selection"),
        _metric(historical_hits, historical_total, "no_expected_historical_state"),
        _metric(dispute_hits, dispute_total, "no_expected_dispute"),
        _metric(no_change_hits, no_change_total, "no_expected_no_change"),
        _metric(supersession_hits, supersession_total, "no_expected_supersession"),
        _metric(evidence_hits, count, "no_cases"),
        cross_user,
        sum(item.execution_mode == "deterministic" for item in predictions),
        sum(item.model is not None for item in predictions),
        False,
        False,
    )
    exact = (
        len(predictions) == count
        and failures_count == 0
        and cross_user == 0
        and all(
            metric["value"] == 1.0
            for metric in (
                score.exact_outcome_accuracy,
                score.exact_action_accuracy,
                score.current_selection_accuracy,
                score.historical_preservation_accuracy,
                score.dispute_accuracy,
                score.no_change_accuracy,
                score.evidence_trace_coverage,
            )
        )
        and score.supersession_accuracy["value"] is None
    )
    return ResolutionScorecard(**{**asdict(score), "exact_match_gate_passed": exact})


def persist_outputs_before_gold(
    output_dir: str | Path,
    runtime: Sequence[ResolutionRuntimeCase],
    predictions: Sequence[ResolutionPrediction],
    failures: Sequence[ResolutionFailure],
    gold_path: str | Path,
    *,
    expected_gold_sha256: str,
    runtime_resources_closed: bool,
    gold_loader: Callable[[str | Path], tuple[ResolutionGoldCase, ...]] = load_resolution_gold,
) -> ResolutionScorecard:
    _validate_accounting(runtime, predictions, failures)
    output = Path(output_dir)
    _require_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    failure_path = output / "failures.jsonl"
    _write_exclusive(prediction_path, serialize_jsonl(predictions))
    _write_exclusive(failure_path, serialize_jsonl(failures))
    _verify_persisted_accounting(runtime, prediction_path, failure_path)
    if not runtime_resources_closed:
        raise ResolutionEvaluationError("runtime resources must close before gold access")
    _require_hash(Path(gold_path), expected_gold_sha256)
    gold = gold_loader(gold_path)
    scorecard = score_belief_resolution(runtime, predictions, failures, gold)
    _write_exclusive(output / "scores.json", _json_bytes(asdict(scorecard)))
    return scorecard


def execute_resolution_evaluation(
    connection_factory: Callable[[], object],
    *,
    repo_root: str | Path = ".",
    result_root: str | Path | None = None,
) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    output = root / RESULT_ROOT if result_root is None else Path(result_root).resolve()
    _require_empty_output(output)
    dataset = root / DATASET_ROOT
    if DATASET_MANIFEST_SHA256:
        _require_hash(dataset / "manifest.json", DATASET_MANIFEST_SHA256)
    manifest, runtime = load_resolution_dataset_runtime(dataset, repo_root=root)
    connection = connection_factory()
    try:
        apply_migrations(connection, root / "migrations")
        _require_clean_database(connection)
        relation_manifest = _read_json_object(
            root / "data/conflicts/relation-development-v1/manifest.json",
            "relation manifest",
        )
        reference_path = root / str(relation_manifest["reference_runtime"]["path"])
        predictions, failures = run_resolution_cases(
            connection, runtime, repo_root=root, reference_runtime_path=reference_path
        )
    finally:
        connection.close()
    scorecard = persist_outputs_before_gold(
        output,
        runtime,
        predictions,
        failures,
        dataset / str(manifest["gold"]["path"]),
        expected_gold_sha256=GOLD_SHA256,
        runtime_resources_closed=True,
    )
    run = {
        "artifact_version": "belief_resolution_development_v1",
        "dataset_version": DATASET_VERSION,
        "starting_commit": "56ce3125886af28625808b31c6b412aa62ae1c9f",
        "case_count": 8,
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "execution_mode": "deterministic",
        "model_fallback": False,
        "model_calls": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "new_cost_usd": 0,
        "hosted_writes": 0,
        "gold_opened_after_persisted_results": True,
        "runtime_resources_closed_before_gold": True,
    }
    _write_exclusive(output / "run.json", _json_bytes(run))
    _write_exclusive(output / "findings.md", _findings(scorecard).encode("utf-8"))
    predecessor = _predecessor_attestation(root)
    implementation_paths = (
        "Makefile",
        "configs/conflicts/belief_resolver_v1.json",
        "migrations/0005_belief_resolution.sql",
        "src/conflicts/__init__.py",
        "src/conflicts/resolution.py",
        "src/conflicts/resolution_evaluation.py",
        "src/conflicts/resolver.py",
        "src/ingestion/service.py",
        "src/storage/contracts.py",
        "src/storage/repository.py",
        "src/temporal/contracts.py",
        "src/temporal/service.py",
        "tests/unit/test_conflict_resolver.py",
        "tests/unit/test_belief_resolution_persistence.py",
        "tests/unit/test_belief_resolution_evaluation.py",
        "tests/unit/test_conflict_candidate_evaluation.py",
        "tests/unit/test_temporal_contracts.py",
        "tests/integration/test_belief_resolution.py",
        "tests/integration/test_belief_resolution_evaluation.py",
        "tests/integration/test_conflict_relation_evaluation.py",
        "tests/integration/test_conflict_relations.py",
        "tests/integration/test_phase4_storage.py",
        "tests/integration/test_temporal_service.py",
    )
    artifacts = tuple(output / name for name in (
        "predictions.jsonl", "failures.jsonl", "scores.json", "run.json", "findings.md"
    ))
    release = {
        "artifact_version": "belief_resolution_development_v1",
        "guidance_version": "step-5.3-guidance-v1",
        "decision_envelope_sha256": "8bbaf5c53358858027db6f5a6eda383c6c8cc47dae80f36a5228fe1b8ed6bea8",
        "dataset": {
            "manifest_sha256": file_sha256(dataset / "manifest.json"),
            "runtime_sha256": RUNTIME_SHA256,
            "gold_sha256": GOLD_SHA256,
            "case_count": 8,
            "user_counts": {"user_001": 4, "user_002": 4},
            "present_relation_labels": PRESENT_RELATION_LABELS,
            "source_boundary": manifest["source_boundary"],
        },
        "execution": run,
        "input_hashes": {
            name: value["sha256"]
            for name, value in manifest.items()
            if isinstance(value, dict) and set(value) == {"path", "sha256"}
        },
        "predecessor": predecessor,
        "implementation_hashes": {
            path: file_sha256(root / path) for path in implementation_paths
        },
        "artifacts": {path.name: file_sha256(path) for path in artifacts},
        "metrics": asdict(scorecard),
        "known_limitations": [
            "The fixed eight-pair set contains six unrelated classifications, one temporal change, and one unresolved ambiguity; exclusion precedence applies to two unrelated pairs with non-user subjects.",
            "No supersession policy has a development denominator; that metric is not evaluated.",
            "This deterministic development check does not estimate production accuracy.",
        ],
    }
    _write_exclusive(output / "manifest.json", _json_bytes(release))
    return release


def serialize_jsonl(values: Sequence[object]) -> bytes:
    records = [asdict(value) for value in values]
    records.sort(key=lambda item: (str(item.get("case_id", "")), str(item.get("failure_id", ""))))
    return b"".join(_json_bytes(item) for item in records)


def _load_relation_predictions(path: Path) -> Mapping[str, RelationPrediction]:
    result: dict[str, RelationPrediction] = {}
    for raw in _read_jsonl(path):
        relations = tuple(RelationPredictionRelation(**value) for value in raw["relations"])
        prediction = RelationPrediction(**{**raw, "relations": relations})
        if prediction.case_id in result:
            raise ResolutionEvaluationError("Step 5.2 prediction case repeats")
        result[prediction.case_id] = prediction
    if set(result) != set(CASE_IDS):
        raise ResolutionEvaluationError("Step 5.2 prediction set changed")
    counts: dict[str, int] = {}
    for value in result.values():
        counts[value.label] = counts.get(value.label, 0) + 1
    if counts != PRESENT_RELATION_LABELS:
        raise ResolutionEvaluationError("Step 5.2 prediction labels changed")
    return result


def _require_frozen_decision(expected: RelationPrediction, decision: object, classifier: ConflictClassifier) -> None:
    actual = RelationPrediction(
        expected.case_id,
        expected.user_id,
        expected.pair_id,
        decision.classifier_version,
        classifier.config.rule_version,
        decision.decision_id,
        decision.input_snapshot_sha256,
        decision.label,
        tuple(
            RelationPredictionRelation(
                item.relation_id, item.source_claim_id, item.target_claim_id,
                item.relation_type, item.confidence,
            )
            for item in decision.relations
        ),
    )
    if actual != expected:
        raise ResolutionEvaluationError("live classification differs from Step 5.2 prediction")


def _valid_at(strategy: str, left: TemporalClaim, right: TemporalClaim) -> date | datetime | None:
    if strategy == "none":
        return None
    if strategy != "later_interval_start":
        raise ResolutionEvaluationError("valid_at strategy is invalid")
    starts = [
        item.version.valid_from_timestamp or item.version.valid_from_date
        for item in (left, right)
    ]
    if any(value is None for value in starts) or type(starts[0]) is not type(starts[1]):
        raise ResolutionEvaluationError("later interval start is unavailable")
    return max(starts)


def _pair_statuses(connection: object, user_id: str, claim_ids: tuple[str, str]) -> Mapping[str, str]:
    rows = connection.execute(
        """
        SELECT claim_id, lifecycle_status FROM claim_versions
        WHERE user_id = %s AND claim_id = ANY(%s) AND transaction_to IS NULL
        ORDER BY claim_id
        """,
        (user_id, list(claim_ids)),
    ).fetchall()
    if len(rows) != 2:
        raise ResolutionEvaluationError("pair open lifecycle state is incomplete")
    return dict(rows)


def _prediction(case: ResolutionRuntimeCase, persisted: object, statuses: Mapping[str, str], source_ids: tuple[str, ...]) -> ResolutionPrediction:
    record = persisted.resolution
    actions = tuple(
        ResolutionPredictionAction(
            item.action_id, item.action_order, item.claim_id, item.from_status,
            item.target_status, item.replacement_claim_id, item.reason,
        )
        for item in persisted.actions
    )
    by_status = {
        status: tuple(sorted(claim_id for claim_id, value in statuses.items() if value == status))
        for status in ("current", "historical", "disputed", "superseded")
    }
    return ResolutionPrediction(
        case.case_id, case.user_id, case.pair_id, case.decision_id,
        record.resolution_id, record.resolver_version, record.policy_version,
        record.input_snapshot_sha256, record.outcome,
        record.selected_current_claim_id, actions,
        by_status["current"], by_status["historical"], by_status["disputed"],
        by_status["superseded"],
        tuple(sorted(item.decision_evidence_id for item in persisted.evidence)),
        source_ids,
        tuple(sorted(item.relation_id for item in persisted.relations)),
        None,
    )


def _action_key(values: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.action_order, item.claim_id, item.target_status, item.replacement_claim_id)
        for item in values
    )


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {"status": "not_evaluated", "numerator": 0, "denominator": 0, "value": None, "reason": reason}
    return {"status": "evaluated", "numerator": numerator, "denominator": denominator, "value": numerator / denominator, "reason": None}


def _validate_accounting(runtime: Sequence[ResolutionRuntimeCase], predictions: Sequence[ResolutionPrediction], failures: Sequence[ResolutionFailure]) -> None:
    expected = {item.case_id: item.user_id for item in runtime}
    outcomes = [(item.case_id, item.user_id) for item in (*predictions, *failures)]
    if (
        len(outcomes) != len(expected)
        or len(outcomes) != len({case_id for case_id, _ in outcomes})
        or any(expected.get(case_id) != user_id for case_id, user_id in outcomes)
    ):
        raise ResolutionEvaluationError("each runtime case requires one user-scoped outcome")


def _verify_persisted_accounting(runtime: Sequence[ResolutionRuntimeCase], prediction_path: Path, failure_path: Path) -> None:
    case_ids = [item["case_id"] for path in (prediction_path, failure_path) for item in _read_jsonl(path)]
    if sorted(case_ids) != sorted(case.case_id for case in runtime):
        raise ResolutionEvaluationError("persisted outcomes are incomplete")


def _require_clean_database(connection: object) -> None:
    tables = (
        "memory_users", "source_events", "source_spans", "extraction_versions",
        "processing_attempts", "claims", "claim_versions", "evidence_links",
        "claim_extractions", "processing_outbox", "source_tombstones",
        "lifecycle_transitions", "conflict_decisions", "claim_relations",
        "conflict_decision_evidence", "belief_resolutions",
        "belief_resolution_actions", "belief_resolution_evidence",
    )
    for table in tables:
        row = connection.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)").fetchone()
        if row is None or row[0]:
            raise ResolutionEvaluationError("resolution evaluation requires a clean database")


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    path = root / STEP52_RESULT_ROOT / "manifest.json"
    _require_hash(path, STEP52_MANIFEST_SHA256)
    manifest = _read_json_object(path, "Step 5.2 predecessor manifest")
    artifacts = manifest.get("artifacts")
    implementation = manifest.get("implementation_hashes")
    predecessor = manifest.get("predecessor")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(implementation, dict)
        or not isinstance(predecessor, dict)
        or not isinstance(predecessor.get("file_hashes"), dict)
        or not isinstance(predecessor.get("authorized_drift"), list)
    ):
        raise ResolutionEvaluationError("Step 5.2 predecessor maps are invalid")
    for name, expected in artifacts.items():
        _require_hash(root / STEP52_RESULT_ROOT / name, expected)
    protected = dict(predecessor["file_hashes"])
    for item in predecessor["authorized_drift"]:
        if not isinstance(item, dict) or set(item) != {
            "path", "predecessor_sha256", "step5_2_sha256", "reason"
        }:
            raise ResolutionEvaluationError("Step 5.2 predecessor drift is invalid")
        relative = item["path"]
        if protected.get(relative) != item["predecessor_sha256"]:
            raise ResolutionEvaluationError("Step 5.2 predecessor drift binding changed")
        protected[relative] = item["step5_2_sha256"]
    protected.update(implementation)
    drift = []
    for relative, expected in sorted(protected.items()):
        current = file_sha256(root / relative)
        if current != expected:
            if relative not in STEP53_ALLOWED_PREDECESSOR_DRIFT_REASONS:
                raise ResolutionEvaluationError("unauthorized Step 5.3 predecessor drift")
            drift.append({
                "path": relative,
                "predecessor_sha256": expected,
                "step5_3_sha256": current,
                "reason": STEP53_ALLOWED_PREDECESSOR_DRIFT_REASONS[relative],
            })
    return {
        "manifest": {"path": str(STEP52_RESULT_ROOT / "manifest.json"), "sha256": STEP52_MANIFEST_SHA256},
        "artifacts": dict(sorted(artifacts.items())),
        "protected_file_hashes": dict(sorted(protected.items())),
        "protected_file_count": len(protected),
        "unchanged_file_count": len(protected) - len(drift),
        "authorized_drift": drift,
        "allowed_drift": [
            {"path": path, "reason": reason}
            for path, reason in sorted(
                STEP53_ALLOWED_PREDECESSOR_DRIFT_REASONS.items()
            )
        ],
        "relation_runtime_sha256": STEP52_RUNTIME_SHA256,
        "relation_predictions_sha256": STEP52_PREDICTIONS_SHA256,
    }


def _findings(scorecard: ResolutionScorecard) -> str:
    return (
        "# Belief-resolution development findings\n\n"
        f"All {scorecard.case_count} fixed cases produced deterministic predictions with no model calls. "
        "Outcome, action, current-state, historical-state, dispute, no-change, and evidence-trace checks all matched the reviewed expectations.\n\n"
        "Supersession is not scored because the fixed Step 5.2 input set contains no correction, refinement, or retraction case. "
        "The scorecard records that denominator as not evaluated rather than treating it as perfect.\n"
    )


def _validate_case_order(values: Sequence[object]) -> None:
    if tuple(item.case_id for item in values) != CASE_IDS:
        raise ResolutionEvaluationError("case order changed")


def _development_user(value: object) -> str:
    if value not in {"user_001", "user_002"}:
        raise ResolutionEvaluationError("only development users are allowed")
    return str(value)


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not SANITIZED_TOKEN.fullmatch(value):
        raise ResolutionEvaluationError(f"{name} is invalid")
    return value


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _exact(value: object, fields: frozenset[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ResolutionEvaluationError(f"{location} fields changed")


def _read_json_object(path: Path, location: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResolutionEvaluationError(f"{location} is unreadable") from error
    if not isinstance(value, dict):
        raise ResolutionEvaluationError(f"{location} must be an object")
    return value


def _read_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ResolutionEvaluationError("JSONL is unreadable") from error
    if any(not isinstance(value, dict) for value in values):
        raise ResolutionEvaluationError("JSONL rows must be objects")
    return values


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as error:
        raise ResolutionEvaluationError("immutable artifact already exists") from error


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ResolutionEvaluationError("result directory must be absent or empty")


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ResolutionEvaluationError("frozen input hash changed")
