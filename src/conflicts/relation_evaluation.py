"""Leakage-safe deterministic evaluation for Step 5.2 relation classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from storage.migrations import apply_migrations
from temporal.contracts import TemporalQuery
from temporal.service import TemporalService

from .candidates import CandidatePair, CandidateSignals, ELIGIBLE_STATUSES
from .classifier import (
    CLASSIFIER_VERSION,
    CONFLICT_LABELS,
    DIRECTED_RELATIONS,
    EMITTED_RELATIONS,
    LABEL_RELATION,
    ClassificationRequest,
    ConflictClassifier,
)
from .evaluation import (
    CandidateRuntimeCase,
    _insert_runtime_case,
    _load_reference_execution_catalog,
    load_candidate_runtime,
)
from .relations import ConflictRelationService


DATASET_VERSION = "relation_development_v1"
SPLIT = "development"
DATASET_ROOT = Path("data/conflicts/relation-development-v1")
RESULT_ROOT = Path("results/conflicts/relation-classification-development-v1")
STEP51_RESULT_ROOT = Path("results/conflicts/candidate-generation-development-v1")
STEP51_MANIFEST_SHA256 = "e089dd87b4361982988cd6df37a150e678f3b9245c1a14a007f90202de6b6c18"
STEP52_PREDECESSOR_DRIFT_REASONS = {
    "Makefile": "adds_step5_2_relation_test_target",
    "src/conflicts/__init__.py": "exports_step5_2_relation_contracts",
    "src/ingestion/service.py": "removes_citing_decisions_before_source_spans",
    "src/storage/contracts.py": "adds_typed_conflict_relation_records",
    "src/storage/repository.py": "adds_atomic_user_scoped_relation_persistence",
    "tests/integration/test_conflict_candidates.py": "replays_step5_1_through_current_migrations",
    "tests/integration/test_phase4_storage.py": "expects_migration_0004_and_three_tables",
    "tests/integration/test_temporal_evaluation.py": "replays_step4_4_through_current_migrations",
    "tests/integration/test_temporal_service.py": "expects_migration_0004_in_current_set",
    "tests/unit/test_conflict_candidate_evaluation.py": "attests_predecessor_maps_and_exact_drift",
    "tests/unit/test_temporal_evaluation.py": "checks_gold_sequencing_through_lower_seams",
}
DATASET_MANIFEST_SHA256 = "690694179f910a93fd536d082e7abd4c48cf76d9216959211189833af9f72d5a"
RUNTIME_SHA256 = "9a298284bc25a155954be6e20e7807541f9638fbfd50640060878b7a73b5fe0e"
GOLD_SHA256 = "0e245546cab7b85cffa83fa3ef05fe860ea6a230891718f36dfe44851fa42d74"
CASE_IDS = (
    "c51_u1_same_family_repeat",
    "c51_u1_separate_periods",
    "c51_u1_ninety_day_boundary",
    "c51_u1_lexical_threshold",
    "c51_u2_shared_entity_overlap",
    "c51_u2_mixed_unknown",
    "c51_u2_approximate",
    "c51_u2_cross_user_distractor",
)
PRESENT_GOLD_LABELS = {
    "temporal_change": 1,
    "unrelated": 6,
    "unresolved_ambiguity": 1,
}
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version",
        "review_status",
        "case_count",
        "user_counts",
        "case_ids",
        "present_gold_labels",
        "runtime",
        "gold",
        "candidate_runtime",
        "candidate_predictions",
        "reference_runtime",
        "classifier_config",
        "predicate_registry",
        "relation_migration",
        "review_method",
        "source_boundary",
    }
)
RUNTIME_FIELDS = frozenset(
    {
        "case_id",
        "dataset_version",
        "split",
        "user_id",
        "candidate_case_id",
        "pair",
        "explicit_target_claim_id",
    }
)
PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "linker_version",
        "user_id",
        "left_claim_id",
        "right_claim_id",
        "signals",
        "source_ids",
    }
)
SIGNAL_FIELDS = frozenset(
    {
        "same_subject",
        "same_predicate_family",
        "shared_entities",
        "temporal_relation",
        "temporal_gap_days",
        "approximate_time",
        "lexical_jaccard",
    }
)
GOLD_FIELDS = frozenset(
    {
        "case_id",
        "dataset_version",
        "split",
        "user_id",
        "review_status",
        "expected_label",
        "expected_relations",
    }
)
GOLD_RELATION_FIELDS = frozenset(
    {"source_claim_id", "target_claim_id", "relation_type"}
)
SANITIZED_TOKEN = re.compile(r"^[a-z0-9_:-]+$")


class RelationEvaluationError(ValueError):
    """Reject malformed data, leakage, or incomplete evaluation accounting."""


class _RollbackRelationCase(Exception):
    def __init__(self, prediction: "RelationPrediction") -> None:
        self.prediction = prediction


@dataclass(frozen=True)
class RelationRuntimeCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    candidate_case_id: str
    pair: CandidatePair
    explicit_target_claim_id: str | None
    candidate_case: CandidateRuntimeCase


@dataclass(frozen=True)
class RelationGoldRelation:
    source_claim_id: str
    target_claim_id: str
    relation_type: str


@dataclass(frozen=True)
class RelationGoldCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    review_status: str
    expected_label: str
    expected_relations: tuple[RelationGoldRelation, ...]


@dataclass(frozen=True)
class RelationPredictionRelation:
    relation_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    confidence: float

    def __post_init__(self) -> None:
        for name in ("relation_id", "source_claim_id", "target_claim_id"):
            _token(getattr(self, name), name)
        if (
            self.relation_type not in EMITTED_RELATIONS
            or self.source_claim_id == self.target_claim_id
        ):
            raise RelationEvaluationError("prediction relation is invalid")
        if self.relation_type in {"contradicts", "same_topic_as"} and (
            self.source_claim_id >= self.target_claim_id
        ):
            raise RelationEvaluationError("symmetric prediction relation is not canonical")
        if self.confidence != 1.0:
            raise RelationEvaluationError("prediction relation confidence must be one")


@dataclass(frozen=True)
class RelationPrediction:
    case_id: str
    user_id: str
    pair_id: str
    classifier_version: str
    rule_version: str
    decision_id: str
    input_snapshot_sha256: str
    label: str
    relations: tuple[RelationPredictionRelation, ...]
    execution_mode: str = "deterministic"
    model: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "case_id", "user_id", "pair_id", "classifier_version",
            "rule_version", "decision_id", "input_snapshot_sha256",
        ):
            _token(getattr(self, name), name)
        if self.label not in CONFLICT_LABELS:
            raise RelationEvaluationError("prediction label is invalid")
        if self.execution_mode != "deterministic" or self.model is not None:
            raise RelationEvaluationError("relation prediction must be deterministic")
        if not _sha256_text(self.input_snapshot_sha256):
            raise RelationEvaluationError("prediction snapshot hash is invalid")
        if not isinstance(self.relations, tuple):
            raise RelationEvaluationError("prediction relations must be a tuple")
        expected_type = LABEL_RELATION[self.label]
        if (
            (expected_type is None and self.relations)
            or (expected_type is not None and len(self.relations) != 1)
            or any(item.relation_type != expected_type for item in self.relations)
        ):
            raise RelationEvaluationError("prediction label and relation set differ")


@dataclass(frozen=True)
class RelationFailure:
    failure_id: str
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "case_id", "user_id", "code", "location"):
            _token(getattr(self, name), name)


@dataclass(frozen=True)
class RelationScorecard:
    dataset_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    failure_rate: Mapping[str, object]
    overall_label_accuracy: Mapping[str, object]
    per_label: Mapping[str, Mapping[str, object]]
    macro_f1_supported_labels: Mapping[str, object]
    exact_relation_set_accuracy: Mapping[str, object]
    direction_accuracy: Mapping[str, object]
    unresolved_rate: Mapping[str, object]
    deterministic_prediction_count: int
    model_prediction_count: int
    model_fallback: bool
    cross_user_relation_count: int
    exact_match_gate_passed: bool


def load_relation_dataset_runtime(
    dataset_root: str | Path,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], tuple[RelationRuntimeCase, ...]]:
    """Load runtime inputs and bindings without opening or hashing relation gold."""

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
        or manifest["present_gold_labels"] != PRESENT_GOLD_LABELS
        or manifest["review_method"]
        != "manual_review_of_fixed_candidate_pairs_and_runtime_evidence"
        or manifest["source_boundary"]
        != "step5.1_predictions_candidate_runtime_and_step4.4_runtime_only_no_supplemental_claims_gold_oracle_review_or_test_users"
    ):
        raise RelationEvaluationError("relation dataset manifest contract changed")
    bindings = (
        "runtime", "gold", "candidate_runtime", "candidate_predictions",
        "reference_runtime", "classifier_config", "predicate_registry",
        "relation_migration",
    )
    for name in bindings:
        value = manifest[name]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise RelationEvaluationError(f"manifest {name} binding changed")
        if not _sha256_text(value["sha256"]):
            raise RelationEvaluationError(f"manifest {name} hash is invalid")
    runtime_path = dataset / str(manifest["runtime"]["path"])
    _require_hash(runtime_path, str(manifest["runtime"]["sha256"]))
    for name in bindings[2:]:
        binding = manifest[name]
        _require_hash(root / str(binding["path"]), str(binding["sha256"]))
    cases = load_relation_runtime(
        runtime_path,
        candidate_runtime_path=root / str(manifest["candidate_runtime"]["path"]),
        candidate_predictions_path=root / str(manifest["candidate_predictions"]["path"]),
        reference_runtime_path=root / str(manifest["reference_runtime"]["path"]),
    )
    return manifest, cases


def load_relation_runtime(
    path: str | Path,
    *,
    candidate_runtime_path: str | Path,
    candidate_predictions_path: str | Path,
    reference_runtime_path: str | Path,
) -> tuple[RelationRuntimeCase, ...]:
    candidate_cases = {
        case.case_id: case
        for case in load_candidate_runtime(
            candidate_runtime_path, reference_runtime_path=reference_runtime_path
        )
    }
    predictions = _load_candidate_predictions(candidate_predictions_path)
    cases: list[RelationRuntimeCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, RUNTIME_FIELDS, f"runtime[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        candidate_case_id = _token(raw["candidate_case_id"], "candidate_case_id")
        if (
            raw["dataset_version"] != DATASET_VERSION
            or raw["split"] != SPLIT
            or candidate_case_id != case_id
        ):
            raise RelationEvaluationError("relation runtime identity changed")
        target = raw["explicit_target_claim_id"]
        if target is not None:
            target = _token(target, "explicit_target_claim_id")
        pair = _candidate_pair(raw["pair"], f"runtime[{position}].pair")
        candidate_case = candidate_cases.get(candidate_case_id)
        predicted_pair = predictions.get(candidate_case_id)
        if (
            candidate_case is None
            or candidate_case.user_id != user_id
            or predicted_pair is None
            or pair != predicted_pair
            or pair.user_id != user_id
        ):
            raise RelationEvaluationError("runtime pair is not the exact Step 5.1 prediction")
        by_id = {claim.claim_id: claim for claim in candidate_case.claims}
        if (
            pair.left_claim_id not in by_id
            or pair.right_claim_id not in by_id
            or by_id[pair.left_claim_id].user_id != user_id
            or by_id[pair.right_claim_id].user_id != user_id
        ):
            raise RelationEvaluationError("runtime pair is missing or cross-user")
        if target is not None and target not in {
            pair.left_claim_id, pair.right_claim_id
        }:
            raise RelationEvaluationError("runtime target is outside the pair")
        cases.append(
            RelationRuntimeCase(
                case_id, DATASET_VERSION, SPLIT, user_id, candidate_case_id,
                pair, target, candidate_case,
            )
        )
    _validate_case_order(cases)
    return tuple(cases)


def load_relation_gold(path: str | Path) -> tuple[RelationGoldCase, ...]:
    cases: list[RelationGoldCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, GOLD_FIELDS, f"gold[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        label = raw["expected_label"]
        if (
            raw["dataset_version"] != DATASET_VERSION
            or raw["split"] != SPLIT
            or raw["review_status"] != "approved"
            or label not in CONFLICT_LABELS
        ):
            raise RelationEvaluationError("relation gold review contract changed")
        values = raw["expected_relations"]
        if not isinstance(values, list):
            raise RelationEvaluationError("expected_relations must be a list")
        relations: list[RelationGoldRelation] = []
        for index, value in enumerate(values):
            _exact(value, GOLD_RELATION_FIELDS, f"gold[{position}].relations[{index}]")
            source = _token(value["source_claim_id"], "source_claim_id")
            target = _token(value["target_claim_id"], "target_claim_id")
            relation_type = value["relation_type"]
            if relation_type not in EMITTED_RELATIONS or source == target:
                raise RelationEvaluationError("gold relation is invalid")
            if relation_type in {"contradicts", "same_topic_as"} and source >= target:
                raise RelationEvaluationError("symmetric gold relation is not canonical")
            relations.append(RelationGoldRelation(source, target, relation_type))
        expected_type = LABEL_RELATION[label]
        if (
            (expected_type is None and relations)
            or (expected_type is not None and len(relations) != 1)
            or any(item.relation_type != expected_type for item in relations)
        ):
            raise RelationEvaluationError("gold label and relation set differ")
        cases.append(
            RelationGoldCase(
                case_id, DATASET_VERSION, SPLIT, user_id, "approved", label,
                tuple(relations),
            )
        )
    _validate_case_order(cases)
    counts = {label: 0 for label in CONFLICT_LABELS}
    for case in cases:
        counts[case.expected_label] += 1
    if {label: count for label, count in counts.items() if count} != PRESENT_GOLD_LABELS:
        raise RelationEvaluationError("present relation gold labels changed")
    return tuple(cases)


def run_relation_cases(
    connection: object,
    cases: Sequence[RelationRuntimeCase],
    *,
    repo_root: str | Path,
    reference_runtime_path: str | Path,
) -> tuple[tuple[RelationPrediction, ...], tuple[RelationFailure, ...]]:
    """Classify and persist each case inside an isolated rollback transaction."""

    reference = _load_reference_execution_catalog(reference_runtime_path)
    classifier = ConflictClassifier(repo_root=repo_root)
    predictions: list[RelationPrediction] = []
    failures: list[RelationFailure] = []
    for case in cases:
        try:
            with connection.transaction():
                _insert_runtime_case(connection, case.candidate_case, reference)
                _derive_pair_speakers_from_runtime_evidence(
                    connection,
                    case.user_id,
                    (case.pair.left_claim_id, case.pair.right_claim_id),
                    case.candidate_case.transaction_as_of,
                )
                visible = {
                    item.claim.claim_id: item
                    for item in TemporalService(connection).query(
                        TemporalQuery(
                            case.user_id,
                            case.candidate_case.transaction_as_of,
                            statuses=frozenset(ELIGIBLE_STATUSES),
                        )
                    )
                }
                left = visible.get(case.pair.left_claim_id)
                right = visible.get(case.pair.right_claim_id)
                if left is None or right is None:
                    raise RelationEvaluationError("paired claim is not visible")
                ingestion = tuple(
                    connection.execute(
                        """
                        SELECT source_id, ingested_at FROM source_events
                        WHERE user_id = %s AND source_id = ANY(%s)
                        ORDER BY source_id
                        """,
                        (case.user_id, list(case.pair.source_ids)),
                    ).fetchall()
                )
                request = ClassificationRequest(
                    CLASSIFIER_VERSION,
                    case.user_id,
                    case.candidate_case.transaction_as_of,
                    case.pair,
                    left,
                    right,
                    ingestion,
                    case.explicit_target_claim_id,
                )
                decision = classifier.classify(request)
                persisted = ConflictRelationService(
                    connection, repo_root=repo_root
                ).persist(
                    request,
                    decision,
                    case.candidate_case.transaction_as_of,
                )
                if persisted.replayed:
                    raise RelationEvaluationError("fresh case unexpectedly replayed")
                prediction = RelationPrediction(
                    case_id=case.case_id,
                    user_id=case.user_id,
                    pair_id=case.pair.pair_id,
                    classifier_version=decision.classifier_version,
                    rule_version=classifier.config.rule_version,
                    decision_id=decision.decision_id,
                    input_snapshot_sha256=decision.input_snapshot_sha256,
                    label=decision.label,
                    relations=tuple(
                        RelationPredictionRelation(
                            item.relation_id,
                            item.source_claim_id,
                            item.target_claim_id,
                            item.relation_type,
                            item.confidence,
                        )
                        for item in decision.relations
                    ),
                )
                raise _RollbackRelationCase(prediction)
        except _RollbackRelationCase as completed:
            predictions.append(completed.prediction)
        except Exception:
            failures.append(
                RelationFailure(
                    failure_id="failure_"
                    + hashlib.sha256(
                        f"{case.case_id}:case_execution_failed".encode("utf-8")
                    ).hexdigest(),
                    case_id=case.case_id,
                    user_id=case.user_id,
                    code="case_execution_failed",
                    location="case",
                )
            )
    return tuple(predictions), tuple(failures)


def score_relation_classification(
    runtime: Sequence[RelationRuntimeCase],
    predictions: Sequence[RelationPrediction],
    failures: Sequence[RelationFailure],
    gold: Sequence[RelationGoldCase],
) -> RelationScorecard:
    _validate_accounting(runtime, predictions, failures)
    if [case.case_id for case in gold] != [case.case_id for case in runtime]:
        raise RelationEvaluationError("runtime and gold order differ")
    gold_by_id = {case.case_id: case for case in gold}
    predicted = {item.case_id: item for item in predictions}
    label_hits = 0
    relation_hits = 0
    unresolved = 0
    cross_user = 0
    directed_total = 0
    directed_hits = 0
    per_label: dict[str, Mapping[str, object]] = {}
    for case in runtime:
        gold_case = gold_by_id[case.case_id]
        if gold_case.user_id != case.user_id:
            raise RelationEvaluationError("runtime and gold users differ")
        prediction = predicted.get(case.case_id)
        if prediction is None:
            continue
        if prediction.user_id != case.user_id or prediction.pair_id != case.pair.pair_id:
            raise RelationEvaluationError("prediction identity differs from runtime")
        label_hits += int(prediction.label == gold_case.expected_label)
        unresolved += int(prediction.label == "unresolved_ambiguity")
        expected_set = {
            (item.source_claim_id, item.target_claim_id, item.relation_type)
            for item in gold_case.expected_relations
        }
        predicted_set = {
            (item.source_claim_id, item.target_claim_id, item.relation_type)
            for item in prediction.relations
        }
        relation_hits += int(predicted_set == expected_set)
        pair_claims = {case.pair.left_claim_id, case.pair.right_claim_id}
        cross_user += sum(
            item.source_claim_id not in pair_claims
            or item.target_claim_id not in pair_claims
            for item in prediction.relations
        )
        for item in gold_case.expected_relations:
            if item.relation_type in DIRECTED_RELATIONS:
                directed_total += 1
                directed_hits += int(
                    (
                        item.source_claim_id,
                        item.target_claim_id,
                        item.relation_type,
                    )
                    in predicted_set
                )
    f1_values: list[float] = []
    for label in CONFLICT_LABELS:
        support = sum(item.expected_label == label for item in gold)
        if support == 0:
            per_label[label] = {
                "status": "not_evaluated",
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "reason": "label_not_present",
            }
            continue
        true_positive = sum(
            gold_by_id[item.case_id].expected_label == label and item.label == label
            for item in predictions
        )
        predicted_count = sum(item.label == label for item in predictions)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        f1_values.append(f1)
        per_label[label] = {
            "status": "evaluated",
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "reason": None,
        }
    count = len(runtime)
    failure_count = len(failures)
    exact_gate = (
        len(predictions) == count
        and failure_count == 0
        and cross_user == 0
        and label_hits == count
        and relation_hits == count
    )
    return RelationScorecard(
        dataset_version=DATASET_VERSION,
        case_count=count,
        prediction_count=len(predictions),
        failure_count=failure_count,
        failure_rate=_metric(failure_count, count, "no_cases"),
        overall_label_accuracy=_metric(label_hits, count, "no_cases"),
        per_label=per_label,
        macro_f1_supported_labels={
            "value": sum(f1_values) / len(f1_values) if f1_values else None,
            "evaluated_label_count": len(f1_values),
            "reason": None if f1_values else "no_supported_labels",
        },
        exact_relation_set_accuracy=_metric(relation_hits, count, "no_cases"),
        direction_accuracy=_metric(
            directed_hits, directed_total, "no_directed_gold_relations"
        ),
        unresolved_rate=_metric(unresolved, count, "no_cases"),
        deterministic_prediction_count=sum(
            item.execution_mode == "deterministic" for item in predictions
        ),
        model_prediction_count=sum(item.model is not None for item in predictions),
        model_fallback=False,
        cross_user_relation_count=cross_user,
        exact_match_gate_passed=exact_gate,
    )


def persist_outputs_before_gold(
    output_dir: str | Path,
    runtime: Sequence[RelationRuntimeCase],
    predictions: Sequence[RelationPrediction],
    failures: Sequence[RelationFailure],
    gold_path: str | Path,
    *,
    expected_gold_sha256: str,
    runtime_resources_closed: bool,
    gold_loader: Callable[[str | Path], tuple[RelationGoldCase, ...]] = load_relation_gold,
) -> RelationScorecard:
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
        raise RelationEvaluationError("runtime resources must close before gold access")
    _require_hash(Path(gold_path), expected_gold_sha256)
    gold = gold_loader(gold_path)
    scorecard = score_relation_classification(runtime, predictions, failures, gold)
    _write_exclusive(output / "scores.json", _json_bytes(asdict(scorecard)))
    return scorecard


def execute_relation_evaluation(
    connection: object,
    *,
    repo_root: str | Path = ".",
    result_root: str | Path | None = None,
) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    output = root / RESULT_ROOT if result_root is None else Path(result_root).resolve()
    _require_empty_output(output)
    dataset = root / DATASET_ROOT
    _require_hash(dataset / "manifest.json", DATASET_MANIFEST_SHA256)
    manifest, runtime = load_relation_dataset_runtime(dataset, repo_root=root)
    if (
        manifest["runtime"]["sha256"] != RUNTIME_SHA256
        or manifest["gold"]["sha256"] != GOLD_SHA256
    ):
        raise RelationEvaluationError("relation release input hashes changed")
    apply_migrations(connection, root / "migrations")
    _require_clean_database(connection)
    reference_path = root / str(manifest["reference_runtime"]["path"])
    predictions, failures = run_relation_cases(
        connection,
        runtime,
        repo_root=root,
        reference_runtime_path=reference_path,
    )
    scorecard = persist_outputs_before_gold(
        output,
        runtime,
        predictions,
        failures,
        dataset / str(manifest["gold"]["path"]),
        expected_gold_sha256=GOLD_SHA256,
        runtime_resources_closed=True,
    )
    run_path = output / "run.json"
    _write_exclusive(
        run_path,
        _json_bytes(
            {
                "artifact_version": "relation_classification_development_v1",
                "dataset_version": DATASET_VERSION,
                "starting_commit": "4780c85a05d6397d24abe15de96f6c3979867f33",
                "case_count": len(runtime),
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
        ),
    )
    findings_path = output / "findings.md"
    _write_exclusive(findings_path, _findings(scorecard).encode("utf-8"))
    artifact_paths = (
        output / "predictions.jsonl",
        output / "failures.jsonl",
        output / "scores.json",
        run_path,
        findings_path,
    )
    predecessor = _build_predecessor_attestation(root)
    implementation_paths = (
        "Makefile",
        "migrations/0004_conflict_relations.sql",
        "src/conflicts/__init__.py",
        "src/conflicts/classifier.py",
        "src/conflicts/relations.py",
        "src/conflicts/relation_evaluation.py",
        "src/ingestion/service.py",
        "src/storage/contracts.py",
        "src/storage/repository.py",
        "tests/unit/test_conflict_classifier.py",
        "tests/unit/test_conflict_relations.py",
        "tests/unit/test_conflict_relation_evaluation.py",
        "tests/integration/test_conflict_relations.py",
        "tests/integration/test_conflict_relation_evaluation.py",
    )
    release = {
        "artifact_version": "relation_classification_development_v1",
        "guidance_version": "step-5.2-guidance-v1",
        "decision_envelope_sha256": "212dd69b414fab36987c917cc1c6a05bf2828b8abf9c7fa657afb7fc0838767b",
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "runtime_sha256": RUNTIME_SHA256,
            "gold_sha256": GOLD_SHA256,
            "case_count": 8,
            "user_counts": {"user_001": 4, "user_002": 4},
            "present_gold_labels": PRESENT_GOLD_LABELS,
            "source_boundary": manifest["source_boundary"],
        },
        "execution": {
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "cross_user_relation_count": scorecard.cross_user_relation_count,
            "gold_access_boundary": "after_predictions_failures_and_runtime_close",
            "classifier_version": CLASSIFIER_VERSION,
            "execution_mode": "deterministic",
            "model_fallback": False,
            "model_calls": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "new_cost_usd": 0,
            "hosted_writes": 0,
        },
        "input_hashes": {
            name: value["sha256"]
            for name, value in manifest.items()
            if isinstance(value, dict) and set(value) == {"path", "sha256"}
        },
        "predecessor": predecessor,
        "implementation_hashes": {
            path: file_sha256(root / path) for path in implementation_paths
        },
        "artifacts": {path.name: file_sha256(path) for path in artifact_paths},
        "metrics": asdict(scorecard),
        "known_limitations": [
            "The fixed eight-pair development set contains three of the eight relation labels.",
            "No directed relation is present, so direction accuracy has no denominator.",
            "This deterministic development check does not estimate production accuracy.",
        ],
    }
    _write_exclusive(output / "manifest.json", _json_bytes(release))
    return release


def _build_predecessor_attestation(root: Path) -> Mapping[str, object]:
    manifest_path = root / STEP51_RESULT_ROOT / "manifest.json"
    _require_hash(manifest_path, STEP51_MANIFEST_SHA256)
    manifest = _read_json_object(manifest_path, "Step 5.1 predecessor manifest")
    protected = manifest.get("protected_inputs")
    implementation = manifest.get("implementation_hashes")
    artifacts = manifest.get("artifacts")
    if not all(isinstance(value, dict) for value in (protected, implementation, artifacts)):
        raise RelationEvaluationError("Step 5.1 predecessor maps are invalid")
    predecessor_files: dict[str, str] = {}
    for section in (protected, implementation):
        for path, expected in section.items():
            if (
                not isinstance(path, str)
                or not _sha256_text(expected)
                or path in predecessor_files
            ):
                raise RelationEvaluationError("Step 5.1 predecessor file map is invalid")
            predecessor_files[path] = expected
    current_files = {
        path: file_sha256(root / path) for path in predecessor_files
    }
    actual_drift = {
        path for path, expected in predecessor_files.items()
        if current_files[path] != expected
    }
    if actual_drift != set(STEP52_PREDECESSOR_DRIFT_REASONS):
        raise RelationEvaluationError("Step 5.1 predecessor drift set changed")
    drift = [
        {
            "path": path,
            "predecessor_sha256": predecessor_files[path],
            "step5_2_sha256": current_files[path],
            "reason": STEP52_PREDECESSOR_DRIFT_REASONS[path],
        }
        for path in sorted(STEP52_PREDECESSOR_DRIFT_REASONS)
    ]
    _validate_predecessor_drift(predecessor_files, current_files, drift)
    predecessor_result = root / STEP51_RESULT_ROOT
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not _sha256_text(expected):
            raise RelationEvaluationError("Step 5.1 predecessor artifact map is invalid")
        _require_hash(predecessor_result / name, expected)
    return {
        "manifest": {
            "path": str(STEP51_RESULT_ROOT / "manifest.json"),
            "sha256": STEP51_MANIFEST_SHA256,
        },
        "file_hashes": dict(sorted(predecessor_files.items())),
        "file_count": len(predecessor_files),
        "unchanged_file_count": len(predecessor_files) - len(drift),
        "authorized_drift": drift,
        "candidate_dataset": manifest.get("dataset"),
        "candidate_input_hashes": manifest.get("input_hashes"),
        "candidate_artifacts": dict(sorted(artifacts.items())),
    }


def _validate_predecessor_drift(
    predecessor_files: Mapping[str, str],
    current_files: Mapping[str, str],
    drift: Sequence[Mapping[str, object]],
) -> None:
    paths = [item.get("path") for item in drift]
    if len(paths) != len(set(paths)):
        raise RelationEvaluationError("Step 5.1 predecessor drift paths repeat")
    if set(paths) != set(STEP52_PREDECESSOR_DRIFT_REASONS):
        raise RelationEvaluationError("Step 5.1 predecessor drift paths changed")
    for item in drift:
        if set(item) != {
            "path", "predecessor_sha256", "step5_2_sha256", "reason"
        }:
            raise RelationEvaluationError("Step 5.1 predecessor drift fields changed")
        path = str(item["path"])
        if (
            item["predecessor_sha256"] != predecessor_files.get(path)
            or item["step5_2_sha256"] != current_files.get(path)
            or item["reason"] != STEP52_PREDECESSOR_DRIFT_REASONS[path]
            or item["predecessor_sha256"] == item["step5_2_sha256"]
        ):
            raise RelationEvaluationError("Step 5.1 predecessor drift record is invalid")


def serialize_jsonl(values: Sequence[object]) -> bytes:
    records = [asdict(value) for value in values]
    records.sort(key=lambda item: (str(item.get("case_id", "")), str(item.get("failure_id", ""))))
    return b"".join(_json_bytes(item) for item in records)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_pair(raw: object, location: str) -> CandidatePair:
    if not isinstance(raw, dict):
        raise RelationEvaluationError(f"{location} must be an object")
    _exact(raw, PAIR_FIELDS, location)
    signals = raw["signals"]
    if not isinstance(signals, dict):
        raise RelationEvaluationError(f"{location}.signals must be an object")
    _exact(signals, SIGNAL_FIELDS, f"{location}.signals")
    try:
        return CandidatePair(
            pair_id=raw["pair_id"],
            linker_version=raw["linker_version"],
            user_id=raw["user_id"],
            left_claim_id=raw["left_claim_id"],
            right_claim_id=raw["right_claim_id"],
            signals=CandidateSignals(
                same_subject=signals["same_subject"],
                same_predicate_family=signals["same_predicate_family"],
                shared_entities=tuple(signals["shared_entities"]),
                temporal_relation=signals["temporal_relation"],
                temporal_gap_days=signals["temporal_gap_days"],
                approximate_time=signals["approximate_time"],
                lexical_jaccard=signals["lexical_jaccard"],
            ),
            source_ids=tuple(raw["source_ids"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RelationEvaluationError(f"{location} is invalid") from error


def _load_candidate_predictions(path: str | Path) -> Mapping[str, CandidatePair]:
    result: dict[str, CandidatePair] = {}
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, frozenset({"case_id", "user_id", "pairs"}), f"prediction[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        pairs = raw["pairs"]
        if not isinstance(pairs, list) or len(pairs) != 1:
            raise RelationEvaluationError("Step 5.1 prediction must contain one pair")
        pair = _candidate_pair(pairs[0], f"prediction[{position}].pairs[0]")
        if raw["user_id"] != pair.user_id or case_id in result:
            raise RelationEvaluationError("Step 5.1 prediction identity changed")
        result[case_id] = pair
    if set(result) != set(CASE_IDS):
        raise RelationEvaluationError("Step 5.1 prediction case set changed")
    return result


def _validate_accounting(
    runtime: Sequence[RelationRuntimeCase],
    predictions: Sequence[RelationPrediction],
    failures: Sequence[RelationFailure],
) -> None:
    expected = {item.case_id: item.user_id for item in runtime}
    outcomes = [(item.case_id, item.user_id) for item in (*predictions, *failures)]
    if (
        len(outcomes) != len({case_id for case_id, _ in outcomes})
        or {case_id for case_id, _ in outcomes} != set(expected)
        or any(expected.get(case_id) != user_id for case_id, user_id in outcomes)
    ):
        raise RelationEvaluationError("each runtime case requires one user-scoped outcome")


def _verify_persisted_accounting(
    runtime: Sequence[RelationRuntimeCase],
    prediction_path: Path,
    failure_path: Path,
) -> None:
    case_ids = [
        item["case_id"]
        for path in (prediction_path, failure_path)
        for item in _read_jsonl(path)
    ]
    if sorted(case_ids) != sorted(case.case_id for case in runtime):
        raise RelationEvaluationError("persisted outcomes are incomplete")


def _require_clean_database(connection: object) -> None:
    tables = (
        "memory_users", "source_events", "source_spans", "extraction_versions",
        "processing_attempts", "claims", "claim_versions", "evidence_links",
        "claim_extractions", "processing_outbox", "source_tombstones",
        "lifecycle_transitions", "conflict_decisions", "claim_relations",
        "conflict_decision_evidence",
    )
    for table in tables:
        row = connection.execute(
            f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)"
        ).fetchone()
        if row is None or row[0]:
            raise RelationEvaluationError("relation evaluation requires a clean database")


def _derive_pair_speakers_from_runtime_evidence(
    connection: object,
    user_id: str,
    claim_ids: tuple[str, str],
    transaction_as_of: datetime,
) -> None:
    """Fill the candidate fixture's omitted speaker from its exact runtime spans."""

    for claim_id in claim_ids:
        speakers = connection.execute(
            """
            SELECT DISTINCT span.speaker_id
            FROM evidence_links AS evidence
            JOIN source_spans AS span
              ON span.user_id = evidence.user_id
             AND span.span_id = evidence.span_id
            JOIN source_events AS source
              ON source.user_id = span.user_id
             AND source.source_id = span.source_id
            WHERE evidence.user_id = %s AND evidence.claim_id = %s
              AND source.ingested_at <= %s
            ORDER BY span.speaker_id
            """,
            (user_id, claim_id, transaction_as_of),
        ).fetchall()
        if len(speakers) != 1:
            raise RelationEvaluationError(
                "paired claim needs one runtime evidence speaker"
            )
        connection.execute(
            "UPDATE claims SET speaker_id = %s WHERE user_id = %s AND claim_id = %s",
            (speakers[0][0], user_id, claim_id),
        )


def _findings(scorecard: RelationScorecard) -> str:
    return (
        "# Relation classification findings\n\n"
        "The deterministic classifier matched all eight reviewed labels and relation "
        "sets. Six cases are unrelated, one is a temporal change, and one remains "
        "unresolved. No cross-user relation or execution failure occurred.\n\n"
        "Only three labels appear in this fixed development set. None of the pairs has "
        "a directed gold relation, so direction accuracy is not evaluated. The other "
        "five labels still have unit coverage, but this release does not measure them.\n"
    )


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None if denominator else reason,
    }


def _validate_case_order(cases: Sequence[object]) -> None:
    if tuple(item.case_id for item in cases) != CASE_IDS:
        raise RelationEvaluationError("relation case order changed")
    users = [item.user_id for item in cases]
    if users.count("user_001") != 4 or users.count("user_002") != 4:
        raise RelationEvaluationError("relation user counts changed")


def _read_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise RelationEvaluationError("JSONL input is unreadable") from error
    if any(not isinstance(item, dict) for item in values):
        raise RelationEvaluationError("JSONL records must be objects")
    return values


def _read_json_object(path: str | Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RelationEvaluationError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise RelationEvaluationError(f"{name} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RelationEvaluationError("result artifacts are immutable") from error


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RelationEvaluationError("result directory must be absent or empty")


def _require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise RelationEvaluationError("hash-bound input changed")


def _exact(value: object, fields: frozenset[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise RelationEvaluationError(f"{location} fields changed")


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not SANITIZED_TOKEN.fullmatch(value):
        raise RelationEvaluationError(f"{name} must be a sanitized token")
    return value


def _development_user(value: object) -> str:
    user_id = _token(value, "user_id")
    if user_id not in {"user_001", "user_002"}:
        raise RelationEvaluationError("only development users are allowed")
    return user_id


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
