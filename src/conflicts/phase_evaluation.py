"""Isolated deterministic Phase 5 conflict evaluation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, Sequence

from .candidates import CandidatePair, CandidateSignals
from .classifier import CONFLICT_LABELS, EMITTED_RELATIONS
from .evaluation import CandidatePrediction, load_candidate_gold
from .relation_evaluation import RelationPrediction, load_relation_gold
from .resolution_evaluation import ResolutionPrediction, load_resolution_gold


DATASET_VERSION = "phase5_evaluation_development_v1"
SPLIT = "development"
DATASET_ROOT = Path("data/conflicts/phase5-evaluation-development-v1")
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
CASE_USERS = ("user_001",) * 4 + ("user_002",) * 4
RUNTIME_PATHS = {
    "candidate": "data/conflicts/candidate-development-v1/runtime/cases.jsonl",
    "relation": "data/conflicts/relation-development-v1/runtime/cases.jsonl",
    "resolution": "data/conflicts/belief-resolution-development-v1/runtime/cases.jsonl",
}
GOLD_PATHS = {
    "candidate": "data/conflicts/candidate-development-v1/gold/required_pairs.jsonl",
    "relation": "data/conflicts/relation-development-v1/gold/cases.jsonl",
    "resolution": "data/conflicts/belief-resolution-development-v1/gold/cases.jsonl",
}
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version", "review_status", "case_count", "user_counts",
        "case_ids", "runtime_inputs", "scorer_gold", "execution_contract",
        "gold_access_boundary", "source_boundary", "guidance_version",
        "decision_envelope_sha256",
    }
)
SANITIZED_TOKEN = re.compile(r"^[a-z0-9_:-]+$")


class Phase5EvaluationError(ValueError):
    """Reject malformed, leaky, or incomplete phase evaluation data."""


class _StageError(Exception):
    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location


@dataclass(frozen=True)
class Phase5RuntimeCase:
    case_id: str
    user_id: str
    transaction_as_of: datetime
    incoming_claim_ids: tuple[str, ...]
    claim_owners: tuple[tuple[str, str], ...]
    pair: CandidatePair
    explicit_target_claim_id: str | None
    decision_id: str
    decision_snapshot_sha256: str
    relation_label: str
    valid_at_strategy: str
    resolved_offset_seconds: int
    resolution_idempotency_key: str

    def __post_init__(self) -> None:
        _token(self.case_id, "case_id")
        _development_user(self.user_id)
        if self.transaction_as_of.utcoffset() is None:
            raise Phase5EvaluationError("transaction_as_of must be timezone-aware")
        if not self.incoming_claim_ids:
            raise Phase5EvaluationError("incoming claim IDs are required")
        if tuple(sorted(set(self.claim_owners))) != self.claim_owners:
            raise Phase5EvaluationError("claim ownership must be canonical")
        if self.pair.user_id != self.user_id:
            raise Phase5EvaluationError("runtime pair is cross-user")
        owners = dict(self.claim_owners)
        if any(owners.get(item) != self.user_id for item in self.incoming_claim_ids):
            raise Phase5EvaluationError("incoming claim is missing or cross-user")
        if any(owners.get(item) != self.user_id for item in (self.pair.left_claim_id, self.pair.right_claim_id)):
            raise Phase5EvaluationError("runtime pair claim is missing or cross-user")
        _token(self.decision_id, "decision_id")
        if not _sha256_text(self.decision_snapshot_sha256):
            raise Phase5EvaluationError("decision snapshot hash is invalid")
        if self.relation_label not in CONFLICT_LABELS:
            raise Phase5EvaluationError("runtime relation label is invalid")
        _token(self.valid_at_strategy, "valid_at_strategy")
        if not isinstance(self.resolved_offset_seconds, int) or self.resolved_offset_seconds < 0:
            raise Phase5EvaluationError("resolved offset is invalid")
        _token(self.resolution_idempotency_key, "resolution_idempotency_key")


@dataclass(frozen=True)
class Phase5Relation:
    source_claim_id: str
    target_claim_id: str
    relation_type: str

    def __post_init__(self) -> None:
        _token(self.source_claim_id, "source_claim_id")
        _token(self.target_claim_id, "target_claim_id")
        if self.source_claim_id == self.target_claim_id or self.relation_type not in EMITTED_RELATIONS:
            raise Phase5EvaluationError("relation is invalid")


@dataclass(frozen=True)
class Phase5Action:
    action_order: int
    claim_id: str
    target_status: str
    replacement_claim_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.action_order, int) or self.action_order < 1:
            raise Phase5EvaluationError("action order is invalid")
        _token(self.claim_id, "action claim_id")
        _token(self.target_status, "action target_status")
        if self.replacement_claim_id is not None:
            _token(self.replacement_claim_id, "replacement_claim_id")


@dataclass(frozen=True)
class Phase5GoldCase:
    case_id: str
    user_id: str
    required_pair: tuple[str, str]
    expected_label: str
    expected_relations: tuple[Phase5Relation, ...]
    expected_outcome: str
    expected_selected_current_claim_id: str | None
    expected_actions: tuple[Phase5Action, ...]
    expected_historical_claim_ids: tuple[str, ...]
    expected_disputed_claim_ids: tuple[str, ...]
    expected_superseded_claim_ids: tuple[str, ...]
    expected_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _token(self.case_id, "case_id")
        _development_user(self.user_id)
        if len(self.required_pair) != 2 or self.required_pair[0] >= self.required_pair[1]:
            raise Phase5EvaluationError("required pair is not canonical")
        if self.expected_label not in CONFLICT_LABELS:
            raise Phase5EvaluationError("gold label is invalid")
        allowed = set(self.required_pair)
        if any(
            item.source_claim_id not in allowed or item.target_claim_id not in allowed
            for item in self.expected_relations
        ):
            raise Phase5EvaluationError("gold relation is outside the pair")
        if any(item.claim_id not in allowed for item in self.expected_actions):
            raise Phase5EvaluationError("gold action is outside the pair")
        for name in (
            "expected_historical_claim_ids", "expected_disputed_claim_ids",
            "expected_superseded_claim_ids", "expected_source_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise Phase5EvaluationError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class Phase5Prediction:
    case_id: str
    user_id: str
    pair_id: str
    left_claim_id: str
    right_claim_id: str
    candidate_source_ids: tuple[str, ...]
    decision_id: str
    decision_snapshot_sha256: str
    relation_label: str
    relations: tuple[Phase5Relation, ...]
    resolution_id: str
    resolution_outcome: str
    selected_current_claim_id: str | None
    actions: tuple[Phase5Action, ...]
    current_claim_ids: tuple[str, ...]
    historical_claim_ids: tuple[str, ...]
    disputed_claim_ids: tuple[str, ...]
    superseded_claim_ids: tuple[str, ...]
    decision_evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    execution_mode: str = "deterministic"
    model: None = None

    def __post_init__(self) -> None:
        for name in (
            "case_id", "user_id", "pair_id", "left_claim_id", "right_claim_id",
            "decision_id", "resolution_id", "resolution_outcome",
        ):
            _token(getattr(self, name), name)
        if self.left_claim_id >= self.right_claim_id:
            raise Phase5EvaluationError("prediction claim IDs must be canonical")
        if not _sha256_text(self.decision_snapshot_sha256):
            raise Phase5EvaluationError("prediction snapshot hash is invalid")
        if self.relation_label not in CONFLICT_LABELS:
            raise Phase5EvaluationError("prediction relation label is invalid")
        if self.execution_mode != "deterministic" or self.model is not None:
            raise Phase5EvaluationError("phase prediction must be deterministic")
        allowed = {self.left_claim_id, self.right_claim_id}
        if any(
            item.source_claim_id not in allowed or item.target_claim_id not in allowed
            for item in self.relations
        ):
            raise Phase5EvaluationError("prediction relation is outside the pair")
        if any(item.claim_id not in allowed for item in self.actions):
            raise Phase5EvaluationError("prediction action is outside the pair")
        for name in (
            "candidate_source_ids", "current_claim_ids", "historical_claim_ids",
            "disputed_claim_ids", "superseded_claim_ids", "decision_evidence_ids",
            "source_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise Phase5EvaluationError(f"{name} must be sorted and unique")
        for name in ("current_claim_ids", "historical_claim_ids", "disputed_claim_ids", "superseded_claim_ids"):
            if any(item not in allowed for item in getattr(self, name)):
                raise Phase5EvaluationError(f"{name} contains a claim outside the pair")


@dataclass(frozen=True)
class Phase5Failure:
    failure_id: str
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "case_id", "user_id", "code", "location"):
            _token(getattr(self, name), name)


@dataclass(frozen=True)
class Phase5Scorecard:
    dataset_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    failure_rate: Mapping[str, object]
    candidate_recall: Mapping[str, object]
    conflict_pair_precision: Mapping[str, object]
    conflict_pair_recall: Mapping[str, object]
    conflict_pair_f1: Mapping[str, object]
    conflict_type_accuracy: Mapping[str, object]
    false_contradiction_rate: Mapping[str, object]
    correction_link_accuracy: Mapping[str, object]
    current_belief_selection_accuracy: Mapping[str, object]
    historical_preservation_accuracy: Mapping[str, object]
    superseded_preservation_accuracy: Mapping[str, object]
    unresolved_dispute_accuracy: Mapping[str, object]
    evidence_trace_coverage: Mapping[str, object]
    cross_user_count: int
    deterministic_prediction_count: int
    model_prediction_count: int
    model_fallback: bool


class FreshPhase5Pipeline(Protocol):
    """Stage seam implemented by fresh production services in the live runner."""

    def generate_candidates(self, case: Phase5RuntimeCase) -> CandidatePrediction: ...

    def classify_relation(
        self, case: Phase5RuntimeCase, pair: CandidatePair
    ) -> RelationPrediction: ...

    def resolve_belief(
        self, case: Phase5RuntimeCase, decision: RelationPrediction
    ) -> ResolutionPrediction: ...


def load_phase5_runtime(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], tuple[Phase5RuntimeCase, ...]]:
    """Load and hash only the three runtime inputs."""

    root = Path(repo_root).resolve()
    manifest = _read_json_object(manifest_path, "phase manifest")
    if set(manifest) != MANIFEST_FIELDS:
        raise Phase5EvaluationError("phase manifest fields changed")
    if (
        manifest["dataset_version"] != DATASET_VERSION
        or manifest["review_status"] != "approved"
        or manifest["case_count"] != 8
        or manifest["user_counts"] != {"user_001": 4, "user_002": 4}
        or tuple(manifest["case_ids"]) != CASE_IDS
        or manifest["execution_contract"] != "fresh_production_services_no_predecessor_predictions"
        or manifest["gold_access_boundary"] != "after_all_fresh_predictions_and_failures_are_persisted_and_runtime_resources_close"
        or manifest["source_boundary"] != "three_step5_runtime_inputs_and_three_scorer_only_step5_gold_files_no_older_gold_oracle_review_or_test_users"
        or manifest["guidance_version"] != "step-5.4-guidance-v1"
        or manifest["decision_envelope_sha256"] != "512dcbffc4d6b8dab30ba9c55045bb0b22eb8fea16942f9b9c50ea61896043e3"
    ):
        raise Phase5EvaluationError("phase manifest contract changed")
    runtime_bindings = _bindings(manifest["runtime_inputs"], RUNTIME_PATHS, "runtime")
    _bindings(manifest["scorer_gold"], GOLD_PATHS, "gold", verify_files=False)
    runtime_rows: dict[str, tuple[Mapping[str, object], ...]] = {}
    for name, binding in runtime_bindings.items():
        path = root / str(binding["path"])
        _require_hash(path, str(binding["sha256"]))
        runtime_rows[name] = tuple(_read_jsonl(path))
    return manifest, _join_runtime(runtime_rows)


def run_phase5_cases(
    cases: Sequence[Phase5RuntimeCase],
    pipeline: FreshPhase5Pipeline,
) -> tuple[tuple[Phase5Prediction, ...], tuple[Phase5Failure, ...]]:
    """Run each case through fresh candidate, relation, and resolver services."""

    _validate_runtime_order(cases)
    predictions: list[Phase5Prediction] = []
    failures: list[Phase5Failure] = []
    for case in cases:
        try:
            try:
                candidate = pipeline.generate_candidates(case)
            except Exception as error:
                raise _StageError("candidate_execution_failed", "candidate") from error
            if candidate.case_id != case.case_id or candidate.user_id != case.user_id:
                raise _StageError("candidate_identity_mismatch", "candidate")
            if len(candidate.pairs) != 1:
                raise _StageError("candidate_pair_count_mismatch", "candidate")
            pair = candidate.pairs[0]
            if pair != case.pair:
                raise _StageError("candidate_pair_mismatch", "candidate")
            try:
                relation = pipeline.classify_relation(case, pair)
            except Exception as error:
                raise _StageError("relation_execution_failed", "relation") from error
            if (
                relation.case_id != case.case_id
                or relation.user_id != case.user_id
                or relation.pair_id != pair.pair_id
                or relation.decision_id != case.decision_id
                or relation.label != case.relation_label
            ):
                raise _StageError("relation_identity_mismatch", "relation")
            if relation.input_snapshot_sha256 != case.decision_snapshot_sha256:
                raise _StageError("relation_snapshot_mismatch", "relation")
            try:
                resolution = pipeline.resolve_belief(case, relation)
            except Exception as error:
                raise _StageError("resolution_execution_failed", "resolution") from error
            if (
                resolution.case_id != case.case_id
                or resolution.user_id != case.user_id
                or resolution.pair_id != pair.pair_id
                or resolution.decision_id != relation.decision_id
            ):
                raise _StageError("resolution_identity_mismatch", "resolution")
            predictions.append(_prediction(case, pair, relation, resolution))
        except _StageError as error:
            failures.append(_failure(case, error.code, error.location))
    return tuple(predictions), tuple(failures)


def load_phase5_gold(
    manifest: Mapping[str, object],
    *,
    repo_root: str | Path = ".",
    runtime_resources_closed: bool,
) -> tuple[Phase5GoldCase, ...]:
    """Open the three authorized scorer files only after runtime has closed."""

    if not runtime_resources_closed:
        raise Phase5EvaluationError("runtime resources must close before gold access")
    root = Path(repo_root).resolve()
    bindings = _bindings(manifest.get("scorer_gold"), GOLD_PATHS, "gold", verify_files=False)
    paths = {name: root / str(value["path"]) for name, value in bindings.items()}
    for name, path in paths.items():
        _require_hash(path, str(bindings[name]["sha256"]))
    candidate = load_candidate_gold(paths["candidate"])
    relation = load_relation_gold(paths["relation"])
    resolution = load_resolution_gold(paths["resolution"])
    if not (
        [item.case_id for item in candidate]
        == [item.case_id for item in relation]
        == [item.case_id for item in resolution]
        == list(CASE_IDS)
    ):
        raise Phase5EvaluationError("gold case order changed")
    result: list[Phase5GoldCase] = []
    for c, r, b in zip(candidate, relation, resolution, strict=True):
        if c.user_id != r.user_id or c.user_id != b.user_id:
            raise Phase5EvaluationError("gold users differ across stages")
        if len(c.required_pairs) != 1:
            raise Phase5EvaluationError("phase gold requires one candidate pair")
        result.append(
            Phase5GoldCase(
                c.case_id,
                c.user_id,
                c.required_pairs[0],
                r.expected_label,
                tuple(Phase5Relation(x.source_claim_id, x.target_claim_id, x.relation_type) for x in r.expected_relations),
                b.expected_outcome,
                b.expected_selected_current_claim_id,
                tuple(Phase5Action(x.action_order, x.claim_id, x.target_status, x.replacement_claim_id) for x in b.expected_actions),
                b.expected_historical_claim_ids,
                b.expected_disputed_claim_ids,
                b.expected_superseded_claim_ids,
                b.expected_source_ids,
            )
        )
    return tuple(result)


def score_phase5_conflicts(
    runtime: Sequence[Phase5RuntimeCase],
    predictions: Sequence[Phase5Prediction],
    failures: Sequence[Phase5Failure],
    gold: Sequence[Phase5GoldCase],
) -> Phase5Scorecard:
    """Score each Phase 5 component separately. There is no composite score."""

    _validate_accounting(runtime, predictions, failures)
    if [item.case_id for item in gold] != [item.case_id for item in runtime]:
        raise Phase5EvaluationError("runtime and gold order differ")
    by_prediction = {item.case_id: item for item in predictions}
    by_gold = {item.case_id: item for item in gold}
    candidate_hits = pair_tp = pair_fp = pair_fn = type_hits = 0
    false_contradictions = nonhard = correction_hits = correction_total = 0
    current_hits = current_total = historical_hits = historical_total = 0
    superseded_hits = superseded_total = unresolved_hits = unresolved_total = 0
    evidence_hits = cross_user = 0
    for case in runtime:
        expected = by_gold[case.case_id]
        if expected.user_id != case.user_id:
            raise Phase5EvaluationError("runtime and gold users differ")
        prediction = by_prediction.get(case.case_id)
        gold_positive = expected.expected_label != "unrelated"
        predicted_positive = prediction is not None and prediction.relation_label != "unrelated"
        pair_tp += int(gold_positive and predicted_positive)
        pair_fp += int(not gold_positive and predicted_positive)
        pair_fn += int(gold_positive and not predicted_positive)
        nonhard += int(expected.expected_label != "hard_contradiction")
        correction_expected = tuple(x for x in expected.expected_relations if x.relation_type == "corrects")
        correction_total += len(correction_expected)
        if prediction is None:
            current_total += int(expected.expected_selected_current_claim_id is not None)
            historical_total += int(bool(expected.expected_historical_claim_ids))
            superseded_total += int(bool(expected.expected_superseded_claim_ids))
            unresolved_total += int(expected.expected_label == "unresolved_ambiguity")
            continue
        pair = (prediction.left_claim_id, prediction.right_claim_id)
        candidate_hits += int(pair == expected.required_pair)
        type_hits += int(gold_positive and prediction.relation_label == expected.expected_label)
        predicted_relations = set(prediction.relations)
        false_contradictions += int(
            expected.expected_label != "hard_contradiction"
            and any(item.relation_type == "contradicts" for item in prediction.relations)
        )
        correction_hits += sum(item in predicted_relations for item in correction_expected)
        if expected.expected_selected_current_claim_id is not None:
            current_total += 1
            current_hits += int(prediction.selected_current_claim_id == expected.expected_selected_current_claim_id)
        if expected.expected_historical_claim_ids:
            historical_total += 1
            historical_hits += int(prediction.historical_claim_ids == expected.expected_historical_claim_ids)
        if expected.expected_superseded_claim_ids:
            superseded_total += 1
            superseded_hits += int(prediction.superseded_claim_ids == expected.expected_superseded_claim_ids)
        if expected.expected_label == "unresolved_ambiguity":
            unresolved_total += 1
            unresolved_hits += int(
                prediction.relation_label == "unresolved_ambiguity"
                and prediction.disputed_claim_ids == expected.expected_disputed_claim_ids
            )
        evidence_hits += int(
            bool(prediction.decision_evidence_ids)
            and prediction.source_ids == expected.expected_source_ids
        )
        cross_user += int(prediction.user_id != case.user_id)
        cross_user += int(pair != (case.pair.left_claim_id, case.pair.right_claim_id))
    precision = _metric(pair_tp, pair_tp + pair_fp, "no_predicted_conflict_pairs")
    recall = _metric(pair_tp, pair_tp + pair_fn, "no_required_conflict_pairs")
    f1 = _f1_metric(pair_tp, pair_fp, pair_fn)
    return Phase5Scorecard(
        dataset_version=DATASET_VERSION,
        case_count=len(runtime),
        prediction_count=len(predictions),
        failure_count=len(failures),
        failure_rate=_metric(len(failures), len(runtime), "no_cases"),
        candidate_recall=_metric(candidate_hits, len(runtime), "no_required_candidate_pairs"),
        conflict_pair_precision=precision,
        conflict_pair_recall=recall,
        conflict_pair_f1=f1,
        conflict_type_accuracy=_metric(type_hits, pair_tp + pair_fn, "no_required_conflict_pairs"),
        false_contradiction_rate=_metric(false_contradictions, nonhard, "no_nonhard_cases"),
        correction_link_accuracy=_metric(
            correction_hits,
            correction_total,
            "no_reviewed_correction_links",
        ),
        current_belief_selection_accuracy=_metric(current_hits, current_total, "no_current_selection_cases"),
        historical_preservation_accuracy=_metric(historical_hits, historical_total, "no_historical_cases"),
        superseded_preservation_accuracy=_metric(
            superseded_hits,
            superseded_total,
            "no_reviewed_superseded_claims",
        ),
        unresolved_dispute_accuracy=_metric(unresolved_hits, unresolved_total, "no_unresolved_cases"),
        evidence_trace_coverage=_metric(evidence_hits, len(runtime), "no_cases"),
        cross_user_count=cross_user,
        deterministic_prediction_count=sum(item.execution_mode == "deterministic" for item in predictions),
        model_prediction_count=sum(item.model is not None for item in predictions),
        model_fallback=False,
    )


def persist_outputs_before_gold(
    output_dir: str | Path,
    manifest: Mapping[str, object],
    runtime: Sequence[Phase5RuntimeCase],
    predictions: Sequence[Phase5Prediction],
    failures: Sequence[Phase5Failure],
    *,
    repo_root: str | Path = ".",
    runtime_resources_closed: bool,
    gold_loader: Callable[..., tuple[Phase5GoldCase, ...]] = load_phase5_gold,
) -> Phase5Scorecard:
    """Persist every fresh outcome before scorer-only gold is touched."""

    _validate_accounting(runtime, predictions, failures)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise Phase5EvaluationError("result directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output / "predictions.jsonl", serialize_jsonl(predictions))
    _write_exclusive(output / "failures.jsonl", serialize_jsonl(failures))
    if not runtime_resources_closed:
        raise Phase5EvaluationError("runtime resources must close before gold access")
    gold = gold_loader(
        manifest,
        repo_root=repo_root,
        runtime_resources_closed=True,
    )
    scorecard = score_phase5_conflicts(runtime, predictions, failures, gold)
    _write_exclusive(output / "scores.json", canonical_json_bytes(asdict(scorecard)))
    return scorecard


def serialize_jsonl(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(value)) for value in values)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _join_runtime(rows: Mapping[str, tuple[Mapping[str, object], ...]]) -> tuple[Phase5RuntimeCase, ...]:
    if any(len(rows[name]) != 8 for name in RUNTIME_PATHS):
        raise Phase5EvaluationError("each runtime input must contain eight cases")
    result: list[Phase5RuntimeCase] = []
    for position, (candidate, relation, resolution) in enumerate(
        zip(rows["candidate"], rows["relation"], rows["resolution"], strict=True)
    ):
        case_id = _token(candidate.get("case_id"), "case_id")
        user_id = _development_user(candidate.get("user_id"))
        if case_id != CASE_IDS[position] or user_id != CASE_USERS[position]:
            raise Phase5EvaluationError("runtime case order or user changed")
        if candidate.get("dataset_version") != "candidate_development_v1" or candidate.get("split") != SPLIT:
            raise Phase5EvaluationError("candidate runtime identity changed")
        if set(candidate) != {"case_id", "dataset_version", "split", "user_id", "transaction_as_of", "incoming_claim_ids", "tags", "claims"}:
            raise Phase5EvaluationError("candidate runtime fields changed")
        if set(relation) != {"case_id", "dataset_version", "split", "user_id", "candidate_case_id", "pair", "explicit_target_claim_id"}:
            raise Phase5EvaluationError("relation runtime fields changed")
        if set(resolution) != {"case_id", "dataset_version", "split", "user_id", "relation_case_id", "pair_id", "decision_id", "decision_snapshot_sha256", "label", "valid_at_strategy", "resolved_offset_seconds", "idempotency_key"}:
            raise Phase5EvaluationError("resolution runtime fields changed")
        if any(value.get("case_id") != case_id or value.get("user_id") != user_id or value.get("split") != SPLIT for value in (relation, resolution)):
            raise Phase5EvaluationError("cross-stage case identity changed")
        if relation.get("candidate_case_id") != case_id or resolution.get("relation_case_id") != case_id:
            raise Phase5EvaluationError("cross-stage case continuity changed")
        pair = _pair(relation.get("pair"))
        if resolution.get("pair_id") != pair.pair_id or resolution.get("label") not in CONFLICT_LABELS:
            raise Phase5EvaluationError("cross-stage pair or label changed")
        claims = candidate.get("claims")
        if not isinstance(claims, list):
            raise Phase5EvaluationError("candidate runtime claims must be a list")
        owners: list[tuple[str, str]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise Phase5EvaluationError("candidate runtime claim is invalid")
            owners.append((_token(claim.get("claim_id"), "claim_id"), _development_user(claim.get("user_id"))))
        incoming = _string_tuple(candidate.get("incoming_claim_ids"), "incoming_claim_ids")
        transaction_as_of = _aware(candidate.get("transaction_as_of"), "transaction_as_of")
        target = relation.get("explicit_target_claim_id")
        if target is not None:
            target = _token(target, "explicit_target_claim_id")
            if target not in {pair.left_claim_id, pair.right_claim_id}:
                raise Phase5EvaluationError("explicit target is outside the pair")
        result.append(
            Phase5RuntimeCase(
                case_id,
                user_id,
                transaction_as_of,
                incoming,
                tuple(sorted(owners)),
                pair,
                target,
                _token(resolution.get("decision_id"), "decision_id"),
                str(resolution.get("decision_snapshot_sha256")),
                str(resolution.get("label")),
                _token(resolution.get("valid_at_strategy"), "valid_at_strategy"),
                int(resolution.get("resolved_offset_seconds")),
                _token(resolution.get("idempotency_key"), "idempotency_key"),
            )
        )
    _validate_runtime_order(result)
    return tuple(result)


def _prediction(
    case: Phase5RuntimeCase,
    pair: CandidatePair,
    relation: RelationPrediction,
    resolution: ResolutionPrediction,
) -> Phase5Prediction:
    return Phase5Prediction(
        case.case_id,
        case.user_id,
        pair.pair_id,
        pair.left_claim_id,
        pair.right_claim_id,
        pair.source_ids,
        relation.decision_id,
        relation.input_snapshot_sha256,
        relation.label,
        tuple(Phase5Relation(x.source_claim_id, x.target_claim_id, x.relation_type) for x in relation.relations),
        resolution.resolution_id,
        resolution.outcome,
        resolution.selected_current_claim_id,
        tuple(Phase5Action(x.action_order, x.claim_id, x.target_status, x.replacement_claim_id) for x in resolution.actions),
        resolution.current_claim_ids,
        resolution.historical_claim_ids,
        resolution.disputed_claim_ids,
        resolution.superseded_claim_ids,
        resolution.decision_evidence_ids,
        resolution.source_ids,
    )


def _failure(case: Phase5RuntimeCase, code: str, location: str) -> Phase5Failure:
    return Phase5Failure(
        "failure_" + hashlib.sha256(f"{case.case_id}:{code}:{location}".encode("utf-8")).hexdigest(),
        case.case_id,
        case.user_id,
        code,
        location,
    )


def _pair(value: object) -> CandidatePair:
    if not isinstance(value, dict) or set(value) != {"pair_id", "linker_version", "user_id", "left_claim_id", "right_claim_id", "signals", "source_ids"}:
        raise Phase5EvaluationError("runtime pair fields changed")
    signals = value["signals"]
    if not isinstance(signals, dict) or set(signals) != {"same_subject", "same_predicate_family", "shared_entities", "temporal_relation", "temporal_gap_days", "approximate_time", "lexical_jaccard"}:
        raise Phase5EvaluationError("runtime signal fields changed")
    if not isinstance(signals["same_subject"], bool) or not isinstance(signals["same_predicate_family"], bool) or not isinstance(signals["approximate_time"], bool):
        raise Phase5EvaluationError("runtime signal booleans are invalid")
    try:
        return CandidatePair(
            str(value["pair_id"]),
            str(value["linker_version"]),
            _development_user(value["user_id"]),
            _token(value["left_claim_id"], "left_claim_id"),
            _token(value["right_claim_id"], "right_claim_id"),
            CandidateSignals(
                signals["same_subject"],
                signals["same_predicate_family"],
                _sorted_strings(signals["shared_entities"], "shared_entities"),
                str(signals["temporal_relation"]),
                signals["temporal_gap_days"],
                signals["approximate_time"],
                signals["lexical_jaccard"],
            ),
            _string_tuple(value["source_ids"], "source_ids"),
        )
    except (TypeError, ValueError) as error:
        raise Phase5EvaluationError("runtime pair is invalid") from error


def _bindings(
    value: object,
    expected_paths: Mapping[str, str],
    name: str,
    *,
    verify_files: bool = True,
) -> Mapping[str, Mapping[str, object]]:
    del verify_files
    if not isinstance(value, dict) or set(value) != set(expected_paths):
        raise Phase5EvaluationError(f"{name} bindings changed")
    result: dict[str, Mapping[str, object]] = {}
    for key, expected_path in expected_paths.items():
        binding = value[key]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise Phase5EvaluationError(f"{name} binding changed")
        if binding["path"] != expected_path or not _sha256_text(binding["sha256"]):
            raise Phase5EvaluationError(f"{name} binding identity changed")
        result[key] = binding
    return result


def _validate_runtime_order(values: Sequence[Phase5RuntimeCase]) -> None:
    if tuple(item.case_id for item in values) != CASE_IDS or tuple(item.user_id for item in values) != CASE_USERS:
        raise Phase5EvaluationError("runtime case order or users changed")


def _validate_accounting(
    runtime: Sequence[Phase5RuntimeCase],
    predictions: Sequence[Phase5Prediction],
    failures: Sequence[Phase5Failure],
) -> None:
    _validate_runtime_order(runtime)
    expected = {item.case_id: item.user_id for item in runtime}
    outcomes = [(item.case_id, item.user_id) for item in (*predictions, *failures)]
    if (
        len(outcomes) != len(expected)
        or len({case_id for case_id, _ in outcomes}) != len(outcomes)
        or any(expected.get(case_id) != user_id for case_id, user_id in outcomes)
    ):
        raise Phase5EvaluationError("each runtime case requires one user-scoped outcome")


def _metric(numerator: int, denominator: int, reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {"status": "not_evaluated", "numerator": 0, "denominator": 0, "value": None, "reason": reason}
    return {"status": "evaluated", "numerator": numerator, "denominator": denominator, "value": numerator / denominator, "reason": None}


def _f1_metric(true_positive: int, false_positive: int, false_negative: int) -> Mapping[str, object]:
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return {"status": "not_evaluated", "numerator": 0, "denominator": 0, "value": None, "reason": "no_required_or_predicted_conflict_pairs"}
    return {"status": "evaluated", "numerator": 2 * true_positive, "denominator": denominator, "value": 2 * true_positive / denominator, "reason": None}


def _read_json_object(path: str | Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase5EvaluationError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise Phase5EvaluationError(f"{name} must be an object")
    return value


def _read_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase5EvaluationError("JSONL record must be an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Phase5EvaluationError("runtime JSONL is unreadable") from error
    return rows


def _require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise Phase5EvaluationError(f"hash mismatch for {path.name}")


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise Phase5EvaluationError("result artifact already exists") from error


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise Phase5EvaluationError(f"{name} must be a timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase5EvaluationError(f"{name} is invalid") from error
    if result.utcoffset() is None:
        raise Phase5EvaluationError(f"{name} must be timezone-aware")
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase5EvaluationError(f"{name} must be a list")
    result = tuple(_token(item, name) for item in value)
    if tuple(sorted(set(result))) != result:
        raise Phase5EvaluationError(f"{name} must be sorted and unique")
    return result


def _sorted_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise Phase5EvaluationError(f"{name} must be a string list")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise Phase5EvaluationError(f"{name} must be sorted and unique")
    return result


def _development_user(value: object) -> str:
    if value not in {"user_001", "user_002"}:
        raise Phase5EvaluationError("only development users are allowed")
    return str(value)


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or SANITIZED_TOKEN.fullmatch(value) is None:
        raise Phase5EvaluationError(f"{name} must be a sanitized token")
    return value


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
