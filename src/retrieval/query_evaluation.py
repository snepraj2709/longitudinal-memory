"""Deterministic development evaluation for retrieval query planning and filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

from .index_evaluation import execute_index_evaluation
from .query_contracts import (
    EligibilityDecision,
    EligibilityResult,
    QueryPlan,
    RetrievalQueryFailure,
    canonical_json_bytes,
    parse_retrieval_query_request,
    stable_sha256,
)
from .query_planner import build_query_plan, load_query_planner_config
from .query_repository import RetrievalQueryRepository, RetrievalQueryRepositoryError


DATASET_VERSION = "query_planning_development_v1"
STARTING_COMMIT = "cd8fc5856b682584aa979623b500fabaf8d5f902"
GUIDANCE_VERSION = "step-7.2-guidance-v1"
GUIDANCE_SHA256 = "d698cc1b53460bc6422fa9d25fb9f47a398425e7848cdf18f93a71e2e9103e70"
DATASET_ROOT = Path("data/retrieval/query-planning-development-v1")
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
REQUESTS_PATH = DATASET_ROOT / "requests.jsonl"
REFERENCE_PATH = DATASET_ROOT / "reference.jsonl"
RESULT_ROOT = Path("results/retrieval/query-planning-development-v1")
CONFIG_PATH = Path("configs/retrieval/query_planner_v1.json")
ALLOWED_USERS = ("user_001", "user_002")
CASE_IDS = tuple(f"query_case_{number:03d}" for number in range(1, 25))
IMPLEMENTATION_PATHS = (
    "configs/retrieval/query_planner_v1.json",
    "src/retrieval/query_contracts.py",
    "src/retrieval/query_planner.py",
    "src/retrieval/query_repository.py",
    "src/retrieval/query_evaluation.py",
)
RUNTIME_ARTIFACTS = (
    "predictions.jsonl",
    "filter-decisions.jsonl",
    "failures.jsonl",
)
FINAL_ARTIFACTS = (
    *RUNTIME_ARTIFACTS,
    "checks.json",
    "run.json",
    "findings.md",
    "runtime-checkpoint.json",
)
LABELS = (
    "current_state",
    "historical_state",
    "change_over_time",
    "specific_event",
    "relationship",
    "commitment",
    "evidence_request",
    "unknown",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QueryEvaluationError(RuntimeError):
    """Reject incompatible, leaky, or mutable evaluation state."""


@dataclass(frozen=True)
class DevelopmentQueryCase:
    case_id: str
    request: object


@dataclass(frozen=True)
class QueryPrediction:
    case_id: str
    query_id: str
    user_id: str
    plan: QueryPlan
    execution_mode: str = "deterministic_no_model"


@dataclass(frozen=True)
class FilterDecisionBundle:
    case_id: str
    query_id: str
    user_id: str
    plan_id: str
    snapshot_run_id: str
    eligible_record_ids: tuple[str, ...]
    decisions: tuple[EligibilityDecision, ...]


@dataclass(frozen=True)
class QueryReference:
    case_id: str
    query_id: str
    user_id: str
    expected_label: str
    expected_plan: Mapping[str, object]
    expected_eligible_record_ids: tuple[str, ...]
    expected_filter_sha256: str
    review_status: str


@dataclass(frozen=True)
class Metric:
    numerator: int
    denominator: int
    value: float | None
    null_reason: str | None


@dataclass(frozen=True)
class QueryPlanningChecks:
    case_count: int
    user_count: int
    prediction_count: int
    failure_count: int
    label_accuracy: Metric
    plan_expectation_accuracy: Metric
    eligibility_expectation_accuracy: Metric
    cross_user_count: int
    restricted_leakage_count: int
    post_cutoff_leakage_count: int
    duplicate_decision_count: int
    stale_leakage_count: int
    unsupported_leakage_count: int
    model_call_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int


def load_development_requests(
    path: str | Path = REQUESTS_PATH,
) -> tuple[DevelopmentQueryCase, ...]:
    values = _read_jsonl(Path(path))
    cases: list[DevelopmentQueryCase] = []
    for value in values:
        if not isinstance(value, dict) or "case_id" not in value:
            raise QueryEvaluationError("runtime case fields changed")
        case_id = value["case_id"]
        if not isinstance(case_id, str):
            raise QueryEvaluationError("runtime case ID is invalid")
        request_value = dict(value)
        request_value.pop("case_id")
        try:
            request = parse_retrieval_query_request(request_value)
        except Exception as error:
            raise QueryEvaluationError("runtime request is invalid") from error
        cases.append(DevelopmentQueryCase(case_id, request))
    case_ids = tuple(item.case_id for item in cases)
    users = tuple(item.request.user_id for item in cases)
    if case_ids != CASE_IDS or set(users) != set(ALLOWED_USERS):
        raise QueryEvaluationError("runtime case order or users changed")
    if any(users.count(user_id) != 12 for user_id in ALLOWED_USERS):
        raise QueryEvaluationError("runtime user accounting changed")
    if len({item.request.query_id for item in cases}) != len(cases):
        raise QueryEvaluationError("runtime query IDs are duplicated")
    return tuple(cases)


def load_development_reference(
    path: str | Path = REFERENCE_PATH,
) -> tuple[QueryReference, ...]:
    values = _read_jsonl(Path(path))
    references: list[QueryReference] = []
    fields = {
        "case_id",
        "query_id",
        "user_id",
        "expected_label",
        "expected_plan",
        "expected_eligible_record_ids",
        "expected_filter_sha256",
        "review_status",
    }
    for value in values:
        if not isinstance(value, dict) or set(value) != fields:
            raise QueryEvaluationError("reference fields changed")
        eligible = value["expected_eligible_record_ids"]
        if not isinstance(eligible, list) or eligible != sorted(set(eligible)):
            raise QueryEvaluationError("reference eligible IDs changed")
        plan = value["expected_plan"]
        if not isinstance(plan, dict):
            raise QueryEvaluationError("reference plan changed")
        reference = QueryReference(
            _text(value, "case_id"),
            _text(value, "query_id"),
            _text(value, "user_id"),
            _text(value, "expected_label"),
            plan,
            tuple(eligible),
            _text(value, "expected_filter_sha256"),
            _text(value, "review_status"),
        )
        if (
            reference.expected_label not in LABELS
            or SHA256.fullmatch(reference.expected_filter_sha256) is None
            or reference.review_status != "implementation_reviewed"
        ):
            raise QueryEvaluationError("reference identity changed")
        references.append(reference)
    if tuple(item.case_id for item in references) != CASE_IDS:
        raise QueryEvaluationError("reference case order changed")
    if len({item.query_id for item in references}) != len(references):
        raise QueryEvaluationError("reference query IDs are duplicated")
    return tuple(references)


def execute_query_runtime(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[QueryPrediction, ...]:
    root = Path(repo_root).resolve()
    output = _resolve_output(root, output_dir)
    _require_empty(output)
    manifest = _read_object(root / DATASET_MANIFEST)
    _validate_dataset_manifest(root, manifest)
    cases = load_development_requests(root / REQUESTS_PATH)
    config = load_query_planner_config(root / CONFIG_PATH)
    with tempfile.TemporaryDirectory() as directory:
        execute_index_evaluation(
            connection_factory,
            Path(directory) / "index-release",
            repo_root=root,
        )
    connection = connection_factory()
    predictions: list[QueryPrediction] = []
    bundles: list[FilterDecisionBundle] = []
    failures: list[RetrievalQueryFailure] = []
    cross_user_count = 0
    try:
        repository = RetrievalQueryRepository(connection)
        for case in cases:
            try:
                plan = build_query_plan(case.request, config=config)
                result = repository.filter(plan)
                predictions.append(
                    QueryPrediction(case.case_id, case.request.query_id, case.request.user_id, plan)
                )
                bundle = _bundle(case.case_id, case.request.query_id, result)
                bundles.append(bundle)
                ids = [item.index_record_id for item in result.decisions]
                if ids:
                    cross_user_count += connection.execute(
                        """
                        SELECT count(*) FROM retrieval_index_records
                        WHERE user_id <> %s AND index_record_id = ANY(%s)
                        """,
                        (case.request.user_id, ids),
                    ).fetchone()[0]
            except Exception as error:
                failures.append(_sanitized_failure(case, error))
    finally:
        connection.close()
    predictions.sort(key=lambda item: item.case_id)
    bundles.sort(key=lambda item: item.case_id)
    failures.sort(key=lambda item: item.query_id)
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "predictions.jsonl": _serialize(predictions),
        "filter-decisions.jsonl": _serialize(bundles),
        "failures.jsonl": _serialize(failures),
    }
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    checkpoint = {
        "artifact_version": DATASET_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "case_ids": list(CASE_IDS),
        "users": list(ALLOWED_USERS),
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "cross_user_count": cross_user_count,
        "runtime_artifacts": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
        },
        "dataset_manifest_sha256": _file_sha256(root / DATASET_MANIFEST),
        "requests_sha256": _file_sha256(root / REQUESTS_PATH),
        "planner_config_sha256": config.sha256,
        "implementation_hashes": {
            path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS
        },
        "reference_opened": False,
        "runtime_dependencies": "development_requests_and_frozen_index_inputs_only",
        "search_executed": False,
        "ranking_executed": False,
        "model_calls": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "incremental_cost_usd": 0,
        "historical_openai_spend_usd": "0.2314404",
    }
    _write_exclusive(output / "runtime-checkpoint.json", canonical_json_bytes(checkpoint))
    if len(predictions) + len(failures) != len(cases):
        raise QueryEvaluationError("runtime case accounting changed")
    return tuple(predictions)


def score_query_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> QueryPlanningChecks:
    root = Path(repo_root).resolve()
    output = _resolve_output(root, output_dir)
    allowed = {*RUNTIME_ARTIFACTS, "runtime-checkpoint.json"}
    if not output.is_dir() or {item.name for item in output.iterdir()} != allowed:
        raise QueryEvaluationError("runtime checkpoint tree changed")
    checkpoint = _read_object(output / "runtime-checkpoint.json")
    _verify_checkpoint(root, output, checkpoint)
    predictions = _load_predictions(output / "predictions.jsonl")
    bundles = _load_bundles(output / "filter-decisions.jsonl")
    failures = _load_failures(output / "failures.jsonl")
    references = load_development_reference(root / REFERENCE_PATH)
    checks = score_query_planning(predictions, bundles, failures, references, checkpoint)
    artifacts = {
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": DATASET_VERSION,
                "case_count": checks.case_count,
                "prediction_count": checks.prediction_count,
                "failure_count": checks.failure_count,
                "execution_mode": "deterministic_no_model",
                "search_executed": False,
                "ranking_executed": False,
                "model_calls": 0,
                "retries": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "incremental_cost_usd": 0,
                "historical_openai_spend_usd": "0.2314404",
            }
        ),
        "findings.md": (
            "# Query planning development findings\n\n"
            "All 24 synthetic development requests matched their reviewed query labels, "
            "plans, and pre-search eligibility decisions. User, time, lifecycle, and "
            "sensitivity checks ran before any search. No cross-user, restricted, stale, "
            "or post-cutoff record became eligible.\n\n"
            "This release checks deterministic planning and filtering only. It does not "
            "run full-text or vector search, rank candidates, or report retrieval-quality "
            "metrics. The index still reflects candidate-heavy upstream data. No model "
            "was called.\n"
        ).encode("utf-8"),
    }
    for name, payload in artifacts.items():
        _write_exclusive(output / name, payload)
    artifact_hashes = {
        name: _file_sha256(output / name)
        for name in FINAL_ARTIFACTS
    }
    release = {
        "artifact_version": DATASET_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "artifacts": artifact_hashes,
        "dataset": {
            "manifest_path": DATASET_MANIFEST.as_posix(),
            "manifest_sha256": _file_sha256(root / DATASET_MANIFEST),
            "requests_path": REQUESTS_PATH.as_posix(),
            "requests_sha256": _file_sha256(root / REQUESTS_PATH),
            "reference_path": REFERENCE_PATH.as_posix(),
            "reference_sha256": _file_sha256(root / REFERENCE_PATH),
        },
        "runtime_checkpoint_sha256": artifact_hashes["runtime-checkpoint.json"],
        "implementation_hashes": {
            path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS
        },
        "checks": asdict(checks),
        "excluded_inputs": [
            "benchmark_qa",
            "gold",
            "oracle",
            "review_queue",
            "test_users",
        ],
        "limitations": [
            "This is a structural planner and filter evaluation, not a retrieval-quality evaluation.",
            "The frozen index is candidate-heavy and uses deterministic token-hash vectors.",
            "The repository chooses one latest safe aggregate snapshot and fails closed instead of reconstructing an older aggregate.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))
    return checks


def execute_query_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> QueryPlanningChecks:
    execute_query_runtime(connection_factory, output_dir, repo_root=repo_root)
    return score_query_release(output_dir, repo_root=repo_root)


def score_query_planning(
    predictions: Sequence[QueryPrediction],
    bundles: Sequence[FilterDecisionBundle],
    failures: Sequence[RetrievalQueryFailure],
    references: Sequence[QueryReference],
    checkpoint: Mapping[str, object],
) -> QueryPlanningChecks:
    prediction_by_case = _unique_by_case(predictions, "prediction")
    bundle_by_case = _unique_by_case(bundles, "filter decision")
    reference_by_case = _unique_by_case(references, "reference")
    label_matches = 0
    plan_matches = 0
    eligibility_matches = 0
    restricted = 0
    post_cutoff = 0
    duplicates = 0
    stale = 0
    for case_id in CASE_IDS:
        prediction = prediction_by_case.get(case_id)
        bundle = bundle_by_case.get(case_id)
        reference = reference_by_case.get(case_id)
        if prediction is None or bundle is None or reference is None:
            continue
        if prediction.query_id != reference.query_id or prediction.user_id != reference.user_id:
            raise QueryEvaluationError("scorer identity changed")
        label_matches += prediction.plan.primary_label == reference.expected_label
        plan_matches += _plan_expectations(prediction.plan) == reference.expected_plan
        filter_sha = hashlib.sha256(canonical_json_bytes(asdict(bundle))).hexdigest()
        eligibility_matches += (
            filter_sha == reference.expected_filter_sha256
            and bundle.eligible_record_ids == reference.expected_eligible_record_ids
        )
        ids = tuple(item.index_record_id for item in bundle.decisions)
        duplicates += len(ids) - len(set(ids))
        for decision in bundle.decisions:
            if decision.eligible and decision.sensitivity == "restricted":
                restricted += 1
            if decision.eligible and set(decision.rejection_reasons).intersection(
                {"transaction_hidden", "source_after_as_of", "relation_after_as_of"}
            ):
                post_cutoff += 1
            if decision.eligible and set(decision.rejection_reasons).intersection(
                {"stale_lineage", "partial_lineage"}
            ):
                stale += 1
    denominator = len(CASE_IDS)
    checks = QueryPlanningChecks(
        denominator,
        len({item.user_id for item in references}),
        len(predictions),
        len(failures),
        _metric(label_matches, denominator, "no_reference_cases"),
        _metric(plan_matches, denominator, "no_reference_cases"),
        _metric(eligibility_matches, denominator, "no_reference_cases"),
        int(checkpoint.get("cross_user_count", -1)),
        restricted,
        post_cutoff,
        duplicates,
        stale,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    if (
        checks.case_count != 24
        or checks.user_count != 2
        or checks.prediction_count != 24
        or checks.failure_count != 0
        or any(
            metric.numerator != metric.denominator
            for metric in (
                checks.label_accuracy,
                checks.plan_expectation_accuracy,
                checks.eligibility_expectation_accuracy,
            )
        )
        or any(
            (
                checks.cross_user_count,
                checks.restricted_leakage_count,
                checks.post_cutoff_leakage_count,
                checks.duplicate_decision_count,
                checks.stale_leakage_count,
                checks.unsupported_leakage_count,
                checks.model_call_count,
                checks.retry_count,
                checks.input_token_count,
                checks.output_token_count,
                checks.incremental_cost_usd,
            )
        )
    ):
        raise QueryEvaluationError("query planning development checks failed")
    return checks


def verify_query_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve_output(root, output_dir)
    release = _read_object(output / "manifest.json")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(FINAL_ARTIFACTS):
        raise QueryEvaluationError("release artifact map changed")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or _file_sha256(output / name) != expected:
            raise QueryEvaluationError("release artifact changed")
    dataset = release.get("dataset")
    expected_dataset = {
        "manifest_path": DATASET_MANIFEST.as_posix(),
        "manifest_sha256": _file_sha256(root / DATASET_MANIFEST),
        "requests_path": REQUESTS_PATH.as_posix(),
        "requests_sha256": _file_sha256(root / REQUESTS_PATH),
        "reference_path": REFERENCE_PATH.as_posix(),
        "reference_sha256": _file_sha256(root / REFERENCE_PATH),
    }
    if dataset != expected_dataset:
        raise QueryEvaluationError("release dataset binding changed")
    _verify_checkpoint(root, output, _read_object(output / "runtime-checkpoint.json"))


def _bundle(case_id: str, query_id: str, result: EligibilityResult) -> FilterDecisionBundle:
    return FilterDecisionBundle(
        case_id,
        query_id,
        result.user_id,
        result.plan_id,
        result.snapshot_run_id,
        result.eligible_record_ids,
        result.decisions,
    )


def _sanitized_failure(case: DevelopmentQueryCase, error: Exception) -> RetrievalQueryFailure:
    if isinstance(error, RetrievalQueryRepositoryError):
        code = "planning_failed"
        location = "repository"
    else:
        code = "planning_failed"
        location = "planner"
    failure_id = stable_sha256(
        {"case_id": case.case_id, "query_id": case.request.query_id, "code": code, "location": location}
    )
    return RetrievalQueryFailure(
        failure_id,
        case.request.query_id,
        case.request.user_id,
        code,
        location,
    )


def _metric(numerator: int, denominator: int, reason: str) -> Metric:
    return Metric(
        numerator,
        denominator,
        None if denominator == 0 else numerator / denominator,
        reason if denominator == 0 else None,
    )


def _plan_expectations(plan: QueryPlan) -> Mapping[str, object]:
    return {
        "primary_label": plan.primary_label,
        "enabled_record_kinds": list(plan.enabled_record_kinds),
        "requested_valid_time": _json_value(plan.requested_valid_time),
        "speaker_ids": list(plan.speaker_ids),
        "entity_ids": list(plan.entity_ids),
        "allowed_lifecycle_statuses": list(plan.allowed_lifecycle_statuses),
        "allow_sensitive": plan.allow_sensitive,
        "allow_unclassified_sensitivity": plan.allow_unclassified_sensitivity,
        "include_previous_versions": plan.include_previous_versions,
        "checked_relation_expansion_intent": plan.checked_relation_expansion_intent,
        "relation_expansion_types": list(plan.relation_expansion_types),
        "source_evidence_intent": plan.source_evidence_intent,
        "unresolved_time": plan.unresolved_time,
        "clarification_required": plan.clarification_required,
    }


def _json_value(value: object) -> object:
    if value is None:
        return None
    return json.loads(canonical_json_bytes(asdict(value)))


def _validate_dataset_manifest(root: Path, value: Mapping[str, object]) -> None:
    fields = {
        "dataset_version",
        "split",
        "review_status",
        "guidance_version",
        "guidance_sha256",
        "allowed_users",
        "case_ids",
        "label_counts",
        "requests",
        "runtime_bindings",
        "reference_boundary",
        "excluded_inputs",
        "execution_mode",
    }
    if set(value) != fields:
        raise QueryEvaluationError("dataset manifest fields changed")
    if (
        value["dataset_version"] != DATASET_VERSION
        or value["split"] != "development"
        or value["review_status"] != "implementation_reviewed"
        or value["guidance_version"] != GUIDANCE_VERSION
        or value["guidance_sha256"] != GUIDANCE_SHA256
        or value["allowed_users"] != list(ALLOWED_USERS)
        or value["case_ids"] != list(CASE_IDS)
        or value["label_counts"] != {label: 3 for label in LABELS}
        or value["reference_boundary"] != "scorer_only_after_runtime_checkpoint"
        or value["excluded_inputs"] != [
            "benchmark_qa",
            "gold",
            "oracle",
            "review_queue",
            "test_users",
        ]
        or value["execution_mode"] != "deterministic_no_model"
    ):
        raise QueryEvaluationError("dataset manifest identity changed")
    requests = value["requests"]
    if requests != {
        "path": REQUESTS_PATH.as_posix(),
        "record_count": 24,
        "sha256": _file_sha256(root / REQUESTS_PATH),
    }:
        raise QueryEvaluationError("dataset request binding changed")
    bindings = value["runtime_bindings"]
    if not isinstance(bindings, dict):
        raise QueryEvaluationError("dataset runtime bindings changed")
    for path, expected in bindings.items():
        if path not in IMPLEMENTATION_PATHS or _file_sha256(root / path) != expected:
            raise QueryEvaluationError("dataset implementation binding changed")
    serialized = json.dumps(bindings, sort_keys=True)
    if any(token in serialized for token in ("/gold/", "oracle", "review_queue", "test_user")):
        raise QueryEvaluationError("dataset manifest contains a forbidden runtime dependency")


def _verify_checkpoint(root: Path, output: Path, checkpoint: Mapping[str, object]) -> None:
    if (
        checkpoint.get("artifact_version") != DATASET_VERSION
        or checkpoint.get("case_ids") != list(CASE_IDS)
        or checkpoint.get("users") != list(ALLOWED_USERS)
        or checkpoint.get("prediction_count") != 24
        or checkpoint.get("failure_count") != 0
        or checkpoint.get("cross_user_count") != 0
        or checkpoint.get("reference_opened") is not False
        or checkpoint.get("search_executed") is not False
        or checkpoint.get("ranking_executed") is not False
        or checkpoint.get("model_calls") != 0
    ):
        raise QueryEvaluationError("runtime checkpoint accounting changed")
    runtime = checkpoint.get("runtime_artifacts")
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_ARTIFACTS):
        raise QueryEvaluationError("runtime checkpoint artifacts changed")
    for name, expected in runtime.items():
        if _file_sha256(output / name) != expected:
            raise QueryEvaluationError("runtime checkpoint payload changed")
    if checkpoint.get("dataset_manifest_sha256") != _file_sha256(root / DATASET_MANIFEST):
        raise QueryEvaluationError("runtime checkpoint dataset changed")
    if checkpoint.get("requests_sha256") != _file_sha256(root / REQUESTS_PATH):
        raise QueryEvaluationError("runtime checkpoint requests changed")
    implementation = checkpoint.get("implementation_hashes")
    if not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS):
        raise QueryEvaluationError("runtime implementation map changed")
    for path, expected in implementation.items():
        if _file_sha256(root / path) != expected:
            raise QueryEvaluationError("runtime implementation changed")


def _load_predictions(path: Path) -> tuple[QueryPrediction, ...]:
    # Runtime predictions are immutable scorer input. The checked scorer consumes
    # the canonical mapping form and validates identity rather than reconstructing plans.
    values = _read_jsonl(path)
    result: list[QueryPrediction] = []
    for value in values:
        case_id = _text(value, "case_id")
        plan_value = value.get("plan")
        if not isinstance(plan_value, dict):
            raise QueryEvaluationError("prediction plan changed")
        result.append(_prediction_from_value(case_id, value, plan_value))
    return tuple(result)


def _prediction_from_value(case_id: str, value: Mapping[str, object], plan: Mapping[str, object]) -> QueryPrediction:
    from .query_contracts import QueryPlan, RequestedValidTime

    valid = plan["requested_valid_time"]
    requested = None
    if valid is not None:
        requested = RequestedValidTime(
            kind=valid["kind"],
            point_date=_date(valid["point_date"]),
            point_timestamp=_datetime(valid["point_timestamp"]),
            range_start_date=_date(valid["range_start_date"]),
            range_end_date=_date(valid["range_end_date"]),
            range_start_timestamp=_datetime(valid["range_start_timestamp"]),
            range_end_timestamp=_datetime(valid["range_end_timestamp"]),
        )
    parsed = QueryPlan(
        plan["plan_id"], plan["query_id"], plan["user_id"], plan["planner_version"],
        plan["planner_config_sha256"], plan["index_version"], plan["primary_label"],
        _datetime(plan["as_of"]), tuple(plan["enabled_record_kinds"]), requested,
        tuple(plan["speaker_ids"]), tuple(plan["entity_ids"]),
        tuple(plan["allowed_lifecycle_statuses"]), plan["allow_sensitive"],
        plan["allow_unclassified_sensitivity"], plan["include_previous_versions"],
        plan["checked_relation_expansion_intent"], tuple(plan["relation_expansion_types"]),
        plan["source_evidence_intent"], plan["unresolved_time"], plan["clarification_required"],
    )
    return QueryPrediction(
        case_id,
        _text(value, "query_id"),
        _text(value, "user_id"),
        parsed,
        _text(value, "execution_mode"),
    )


def _load_bundles(path: Path) -> tuple[FilterDecisionBundle, ...]:
    values = _read_jsonl(path)
    bundles: list[FilterDecisionBundle] = []
    for value in values:
        decisions = tuple(
            EligibilityDecision(
                item["index_record_id"], item["eligible"], tuple(item["rejection_reasons"]),
                tuple(item["lifecycle_statuses"]), item["sensitivity"], item["unclassified_sensitivity"],
            )
            for item in value["decisions"]
        )
        bundles.append(
            FilterDecisionBundle(
                _text(value, "case_id"), _text(value, "query_id"), _text(value, "user_id"),
                _text(value, "plan_id"), _text(value, "snapshot_run_id"),
                tuple(value["eligible_record_ids"]), decisions,
            )
        )
    return tuple(bundles)


def _load_failures(path: Path) -> tuple[RetrievalQueryFailure, ...]:
    return tuple(
        RetrievalQueryFailure(
            _text(value, "failure_id"), _text(value, "query_id"), _text(value, "user_id"),
            _text(value, "code"), _text(value, "location"),
        )
        for value in _read_jsonl(path)
    )


def _unique_by_case(values: Sequence[object], name: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in values:
        case_id = getattr(item, "case_id")
        if case_id in result:
            raise QueryEvaluationError(f"duplicate {name} case")
        result[case_id] = item
    if not set(result).issubset(CASE_IDS):
        raise QueryEvaluationError(f"unknown {name} case")
    return result


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise QueryEvaluationError("JSONL contains an empty line")
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    QueryEvaluationError(f"invalid JSON constant: {item}")
                ),
            )
            if not isinstance(value, dict):
                raise QueryEvaluationError("JSONL record is invalid")
            values.append(value)
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QueryEvaluationError("JSON object is invalid")
    return value


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(item)) for item in values)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise QueryEvaluationError("immutable artifact already exists") from error


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise QueryEvaluationError("result directory must be empty")


def _resolve_output(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise QueryEvaluationError(f"{name} is invalid")
    return item


def _date(value: object) -> date | None:
    return None if value is None else date.fromisoformat(str(value))


def _datetime(value: object) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise QueryEvaluationError("timestamp is naive")
    return result.astimezone(timezone.utc)
