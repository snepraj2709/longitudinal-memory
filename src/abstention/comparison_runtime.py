"""Freeze matched four-case B6/B7 runtime inputs without scoring or provider use."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from answering.answer_quality_evaluation import verify_answer_quality_release
from answering.contracts import (
    CONFIG_VERSION as PACKAGE_CONFIG_VERSION,
    INPUT_RELEASE_VERSION as PACKAGE_INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION as PACKAGE_RUNTIME_VERSION,
    SCHEMA_VERSION as PACKAGE_SCHEMA_VERSION,
    EvidencePackage,
    EvidencePackageBuildRequest,
    canonical_json_bytes as package_json_bytes,
    stable_sha256 as package_sha256,
)
from answering.evidence_package import build_evidence_package, load_evidence_package_config
from answering.evaluation import _parse_package, _parse_result
from answering.memory_answer import build_memory_answer
from answering.repository import EvidencePackageRepository
from retrieval.baseline_contracts import BaselineRetrievalResult
from retrieval.baselines import load_baseline_config
from retrieval.index_evaluation import execute_index_evaluation
from retrieval.query_planner import build_query_plan, load_query_planner_config
from retrieval.search_repository import RetrievalSearchRepository

from .calibration import load_threshold_config
from .calibration_release_v2 import verify_answerability_threshold_release_v2
from .contracts import answerability_decision_from_mapping
from .evaluation import verify_answerability_release
from .interactive_contracts_v2 import (
    InteractiveExecutionPlan,
    canonical_json_bytes as interactive_json_bytes,
    stable_sha256 as interactive_sha256,
)
from .interactive_evaluation_v2 import FINAL_ROOT as STEP93_FINAL_ROOT
from .interactive_runtime_v2 import (
    ABSTENTION_CONFIG,
    ABSTENTION_POLICY,
    BASELINE_CHECKPOINT_SHA256,
    BASELINE_MANIFEST_SHA256,
    BASELINE_RESULTS_SHA256,
    RUNTIME_ROOT as STEP93_RUNTIME_ROOT,
    THRESHOLD_CONFIG,
    _answer_view,
    _answerability_request,
    _frozen_eligibility,
    _load_cases,
    _load_requirements,
    _threshold_application,
    query_request_for_case,
    verify_interactive_runtime_checkpoint,
)
from .policy import decide_answerability
from .comparison_contracts import (
    COMPARISON_VERSION,
    CONFIG_VERSION,
    DATASET_VERSION,
    DIFFERENCE_WHITELIST,
    PACKAGE_BASELINE,
    RUNTIME_RELEASE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    ComparableCheckpoint,
    ComparablePair,
    ComparablePrediction,
    ComparablePreflight,
    ComparableRuntimeInput,
    ComparisonPrerequisiteError,
    canonical_json_bytes,
    checkpoint_from_mapping,
    pair_from_mapping,
    prediction_from_mapping,
    preflight_from_mapping,
    stable_sha256,
)


STARTING_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
GUIDANCE_SHA256 = "9823e973c651cd2d8836035df07cb5fa0ef69336c18e542955afda05227b9edd"
CONFIG_PATH = Path("configs/abstention/b6_b7_comparable_runtime_v1.json")
DATASET_MANIFEST = Path("data/abstention/b6-b7-comparable-development-v1/runtime/manifest.json")
RESULT_ROOT = Path("results/abstention/b6-b7-comparable-development-runtime-v1")
STEP93_CASES = Path("data/abstention/interactive-answering-development-v2/runtime/cases.jsonl")
STEP93_REQUIREMENTS = Path("data/abstention/interactive-answering-development-v2/runtime/requirements.jsonl")
STEP93_FINAL_MANIFEST = STEP93_FINAL_ROOT / "manifest.json"
STEP83_ROOT = Path("results/answering/memory-answer-quality-development-v1")
STEP91_ROOT = Path("results/abstention/answerability-development-v1")
STEP92_ROOT = Path("results/abstention/answerability-threshold-development-v2")
STEP93_HASHES = {
    "manifest.json": "c9ac70ba2b25ef5639288b39eb0aa997b59957dab6195060d12e6943eec10d7b",
    "checkpoint_manifest.json": "423790b9eb4d0fa5e33aeeafd870cb75f8d78c259c307322d25e276615a199d4",
    "plans.jsonl": "02e2afd4001d96fa421c57b2ad5fd329dda1d581e90124cc0de768b496e5f94a",
    "retrieval-results.jsonl": "5b4f5199cff3f73a83deab9f394aff5340f307a7aac7581c5d7a524424e2f203",
    "packages.jsonl": "ccdf6a13fbba02125442cae803c47298e73be88407267c4c32d2367562beefe4",
    "decisions.jsonl": "be32a1d12084a3bba57fc48d76ae5f15e43a70ddc50aa6ff176fef4dff899e0a",
    "responses.jsonl": "babf748e7014d7d89b418986b3e3b9daf3d191f04e1a93d4cca8b877a276a977",
}
STEP91_MANIFEST_SHA256 = "15667096c215de19fcf2ac6d897237791b62eef30e37d505adac2e5de4523de3"
STEP92_MANIFEST_SHA256 = "ef957226d62beb44a8cb117e786a2c59f23f96815d8b5daa48b9d22d80c6ef53"
MEMORY_CONFIG = Path("configs/answering/memory_answer_v1.json")
MEMORY_CONFIG_SHA256 = "98b743d593e17a88216ac74cf17693f08898537027e0d91672d45aaeb4dfcc10"
PROMPT_SHA256 = "69dd688430c55fc35a16369201ef8eea7d470d10f610edec254f5cbd59bf14c1"
ARTIFACTS = (
    "preflight.json", "pairs.jsonl", "b6-predictions.jsonl",
    "b7-predictions.jsonl", "failures.jsonl",
)
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/abstention/comparison_contracts.py"),
    Path("src/abstention/comparison_runtime.py"),
    DATASET_MANIFEST,
)
ADAPTER_DRIFT = (
    (
        "tests/integration/test_answer_quality_evaluation.py",
        "fff42248ee2c3f361caea561341b4d0e28728be98166d11f6f8f9d138694ba2d",
        "ee33eecc22993e4ddc12edb59eabb74d1908d050a18d0a495f55446c5e871039",
        "commit_aware_step9_3_attestation_and_exact_prerequisite_topology",
    ),
    (
        "tests/integration/test_answerability.py",
        "73aeedc3f24f76e3f357e99793ef6ed85bc8c7610caf9b87fd298a7e2fb3c9f4",
        "8db8c29f08fb8ac687621f58533aeb4f71e1bfbf4ced0e577a0cc77017c655b7",
        "commit_aware_step9_3_attestation_and_exact_prerequisite_topology",
    ),
    (
        "tests/integration/test_memory_answer.py",
        "1ca97a9ff1df5f11d5210b5434f5ded6830205df0b420c4d88570fdaa5f05f1d",
        "ba44d6aad300712233d5ecef6b81e840a3d933e4dbec496d0bbd8dc8800984ff",
        "commit_aware_step9_3_attestation_and_exact_prerequisite_topology",
    ),
)


def load_comparison_config(
    path: str | Path = CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], str]:
    root = Path(repo_root).resolve()
    target = _resolve(root, path)
    raw = target.read_bytes()
    value = json.loads(raw)
    expected = {
        "comparison_version", "schema_version", "config_version", "runtime_version",
        "dataset_version", "runtime_release_version", "development_case_count",
        "roadmap_target_case_count", "frozen_test_deferred_count",
        "wrapper_baseline_order", "package_baseline_id", "b6_capabilities",
        "b7_capabilities", "selected_threshold_profile",
        "configured_future_answer_model", "memory_answer_config_sha256",
        "prompt_version", "prompt_sha256", "provider_execution_enabled",
        "max_output_tokens", "incremental_cost_usd", "historical_openai_spend_usd",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ComparisonPrerequisiteError("comparison config fields changed")
    if (
        value["comparison_version"] != COMPARISON_VERSION
        or value["schema_version"] != SCHEMA_VERSION
        or value["config_version"] != CONFIG_VERSION
        or value["runtime_version"] != RUNTIME_VERSION
        or value["dataset_version"] != DATASET_VERSION
        or value["runtime_release_version"] != RUNTIME_RELEASE_VERSION
        or value["development_case_count"] != 4
        or value["roadmap_target_case_count"] != 20
        or value["frozen_test_deferred_count"] != 16
        or value["wrapper_baseline_order"] != ["B6", "B7"]
        or value["package_baseline_id"] != PACKAGE_BASELINE
        or value["b6_capabilities"] != ["conflict_resolution", "dual_retrieval", "temporal_versioning"]
        or value["b7_capabilities"] != ["answerability_v1", "conflict_resolution", "dual_retrieval", "ordinary_1_trait_2", "temporal_versioning"]
        or value["selected_threshold_profile"] != "ordinary_1_trait_2"
        or value["configured_future_answer_model"] != "gpt-4.1-2025-04-14"
        or value["memory_answer_config_sha256"] != MEMORY_CONFIG_SHA256
        or value["prompt_version"] != "memory_answer_prompt_v1"
        or value["prompt_sha256"] != PROMPT_SHA256
        or value["provider_execution_enabled"] is not False
        or value["max_output_tokens"] != 0
        or value["incremental_cost_usd"] != 0
        or value["historical_openai_spend_usd"] != "0.2314404"
    ):
        raise ComparisonPrerequisiteError("comparison config policy changed")
    return value, hashlib.sha256(raw).hexdigest()


def execute_comparison_runtime(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[ComparablePair, ...]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    _verify_predecessors(root)
    config, config_sha256 = load_comparison_config(repo_root=root)
    _verify_dataset_manifest(root, config_sha256)
    cases = _load_cases(root / STEP93_CASES)
    requirements = _load_requirements(root / STEP93_REQUIREMENTS)
    if tuple(item.case_id for item in cases) != tuple(item.case_id for item in requirements):
        raise ComparisonPrerequisiteError("case/requirement order changed")
    planner_config = load_query_planner_config(root / "configs/retrieval/query_planner_v1.json")
    baseline_config = load_baseline_config(root / "configs/retrieval/baseline_v1.json")
    _, package_config_sha = load_evidence_package_config(root / "configs/answering/evidence_package_v1.json")
    threshold_config = load_threshold_config(root / THRESHOLD_CONFIG, repo_root=root)
    if threshold_config.selected_profile != "ordinary_1_trait_2":
        raise ComparisonPrerequisiteError("selected threshold profile changed")
    answerability_config_sha = _sha(root / ABSTENTION_CONFIG)
    answerability_policy_sha = _sha(root / ABSTENTION_POLICY)
    if _sha(root / MEMORY_CONFIG) != MEMORY_CONFIG_SHA256:
        raise ComparisonPrerequisiteError("memory answer config changed")

    with tempfile.TemporaryDirectory() as directory:
        execute_index_evaluation(
            connection_factory,
            Path(directory) / "index-release",
            repo_root=root,
        )

    runtime_inputs: list[ComparableRuntimeInput] = []
    plans: list[InteractiveExecutionPlan] = []
    results: list[BaselineRetrievalResult] = []
    packages: list[EvidencePackage] = []
    decisions = []
    b6_predictions: list[ComparablePrediction] = []
    b7_predictions: list[ComparablePrediction] = []
    pairs: list[ComparablePair] = []
    connection = connection_factory()
    try:
        search = RetrievalSearchRepository(connection)
        package_repository = EvidencePackageRepository(connection)
        for case, requirement_set in zip(cases, requirements, strict=True):
            request = query_request_for_case(case)
            plan = build_query_plan(request, config=planner_config)
            result = search.retrieve(
                request, PACKAGE_BASELINE,
                planner_config=planner_config, baseline_config=baseline_config,
            )
            if result.plan_id != plan.plan_id or result.baseline_id != PACKAGE_BASELINE:
                raise ComparisonPrerequisiteError("shared B4 retrieval identity changed")
            eligibility = _frozen_eligibility(request.index_version, result)
            package_request = EvidencePackageBuildRequest(
                PACKAGE_VERSION, PACKAGE_SCHEMA_VERSION, PACKAGE_CONFIG_VERSION,
                package_config_sha, PACKAGE_RUNTIME_VERSION, PACKAGE_INPUT_RELEASE_VERSION,
                BASELINE_MANIFEST_SHA256, BASELINE_CHECKPOINT_SHA256,
                BASELINE_RESULTS_SHA256, request, plan, eligibility, result,
                package_sha256(result),
            )
            package = build_evidence_package(
                package_request, package_repository.hydrate(package_request),
            )
            if package.answer_allowed:
                raise ComparisonPrerequisiteError(
                    "b6_provider_eligibility_requires_new_approval"
                )
            package_hash = hashlib.sha256(package_json_bytes(package)).hexdigest()
            required_predicates = tuple(sorted({
                part.required_predicate for part in requirement_set.parts
                if part.required_predicate is not None
            }))
            execution_fields = {
                "case_id": case.case_id,
                "query_id": request.query_id,
                "requirement_set_id": requirement_set.requirement_set_id,
                "requirement_set_sha256": hashlib.sha256(
                    interactive_json_bytes(requirement_set)
                ).hexdigest(),
                "required_predicates": required_predicates,
                "user_id": case.user_id,
                "request_sha256": hashlib.sha256(package_json_bytes(request)).hexdigest(),
                "plan_id": plan.plan_id,
                "primary_label": plan.primary_label,
                "snapshot_run_id": result.snapshot_run_id,
                "retrieval_execution_id": result.execution_id,
            }
            execution_plan = InteractiveExecutionPlan(
                execution_plan_id=interactive_sha256(execution_fields), **execution_fields,
            )
            runtime_input = _runtime_input(
                case, requirement_set, execution_plan, result, package, package_hash,
            )
            view = _answer_view(package, package_hash)
            b6_answer = build_memory_answer(view, config_path=root / MEMORY_CONFIG)
            decision = decide_answerability(
                _answerability_request(
                    requirement_set, package, package_hash,
                    answerability_config_sha, answerability_policy_sha,
                ),
                package,
            )
            threshold = _threshold_application(decision)
            if threshold.output_decision == "answerable" or threshold.generation_allowed:
                raise ComparisonPrerequisiteError(
                    "b7_provider_eligibility_requires_new_approval"
                )
            if threshold.output_decision != "abstain":
                raise ComparisonPrerequisiteError("B7 development status distribution changed")
            b7_answer = build_memory_answer(view, config_path=root / MEMORY_CONFIG)
            b6 = _prediction(runtime_input, config, "B6", b6_answer)
            b7 = _prediction(runtime_input, config, "B7", b7_answer, decision, threshold)
            pair = _pair(runtime_input, b6, b7)
            runtime_inputs.append(runtime_input)
            plans.append(execution_plan)
            results.append(result)
            packages.append(package)
            decisions.append(decision)
            b6_predictions.append(b6)
            b7_predictions.append(b7)
            pairs.append(pair)
    finally:
        connection.close()

    _require_recomputed_step93(root, plans, results, packages, decisions)
    if len(pairs) != 4:
        raise ComparisonPrerequisiteError("matched runtime accounting changed")
    preflight = ComparablePreflight(
        comparison_version=COMPARISON_VERSION,
        runtime_release_version=RUNTIME_RELEASE_VERSION,
        guidance_sha256=GUIDANCE_SHA256,
        starting_commit=STARTING_COMMIT,
        development_case_count=4,
        roadmap_target_case_count=20,
        frozen_test_deferred_count=16,
        step9_4_evaluated=False,
        step9_4_reference_absent=True,
        step9_4_scorer_absent=True,
        step9_4_final_absent=True,
        step9_3_gold_opened=False,
        step9_3_reference_opened=False,
        runtime_rows_requested=4,
        record_five_requested=False,
        development_user_count=2,
        development_source_count=20,
        next_user_or_source_requested=False,
        implementer_pilot_answer_reference_exposure=True,
        implementer_pilot_answer_reference_used=False,
        reviewer_frozen_snippet_exposure=True,
        reviewer_frozen_snippet_used=False,
        root_contract_deriver_prohibited_exposure=False,
        prompt_rendered=False,
        provider_initialized=False,
        credential_or_environment_opened=False,
        network_opened=False,
        provider_request_count=0,
        retry_count=0,
        input_token_count=0,
        output_token_count=0,
        incremental_cost_usd=0,
    )
    payloads = {
        "preflight.json": canonical_json_bytes(preflight),
        "pairs.jsonl": _serialize(pairs),
        "b6-predictions.jsonl": _serialize(b6_predictions),
        "b7-predictions.jsonl": _serialize(b7_predictions),
        "failures.jsonl": b"",
    }
    checkpoint = ComparableCheckpoint(
        COMPARISON_VERSION, RUNTIME_RELEASE_VERSION, GUIDANCE_SHA256, config_sha256,
        4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 0, 0, 20, 4, 16,
        False, False,
        tuple(sorted((name, hashlib.sha256(data).hexdigest()) for name, data in payloads.items())),
        tuple(sorted((path.as_posix(), _sha(root / path)) for path in IMPLEMENTATION_PATHS)),
        ADAPTER_DRIFT,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        _write_exclusive(output / name, payloads[name])
    _write_exclusive(output / "checkpoint_manifest.json", canonical_json_bytes(checkpoint))
    verify_comparison_runtime(output, repo_root=root)
    return tuple(pairs)


def verify_comparison_runtime(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> ComparableCheckpoint:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _verify_predecessors(root)
    config, config_sha256 = load_comparison_config(repo_root=root)
    _verify_dataset_manifest(root, config_sha256)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {
        *ARTIFACTS, "checkpoint_manifest.json",
    }:
        raise ComparisonPrerequisiteError("runtime artifact set changed")
    preflight = preflight_from_mapping(_read_object(output / "preflight.json"))
    b6 = tuple(prediction_from_mapping(item) for item in _read_jsonl(output / "b6-predictions.jsonl"))
    b7 = tuple(prediction_from_mapping(item) for item in _read_jsonl(output / "b7-predictions.jsonl"))
    pairs = tuple(pair_from_mapping(item) for item in _read_jsonl(output / "pairs.jsonl"))
    if _read_jsonl(output / "failures.jsonl"):
        raise ComparisonPrerequisiteError("runtime failures are not empty")
    checkpoint = checkpoint_from_mapping(_read_object(output / "checkpoint_manifest.json"))
    if (
        preflight.guidance_sha256 != GUIDANCE_SHA256
        or preflight.starting_commit != STARTING_COMMIT
        or checkpoint.guidance_sha256 != GUIDANCE_SHA256
        or checkpoint.config_sha256 != config_sha256
    ):
        raise ComparisonPrerequisiteError("runtime authority changed")
    expected_artifact_names = tuple(sorted(ARTIFACTS))
    if tuple(name for name, _ in checkpoint.artifacts) != expected_artifact_names:
        raise ComparisonPrerequisiteError("runtime artifact binding set changed")
    for name, expected in checkpoint.artifacts:
        if _sha(output / name) != expected:
            raise ComparisonPrerequisiteError("runtime artifact hash changed")
    expected_implementation = tuple(sorted(
        (path.as_posix(), _sha(root / path)) for path in IMPLEMENTATION_PATHS
    ))
    if checkpoint.implementation != expected_implementation:
        raise ComparisonPrerequisiteError("runtime implementation binding set changed")
    for name, expected in checkpoint.implementation:
        if _sha(root / name) != expected:
            raise ComparisonPrerequisiteError("runtime implementation changed")
    if checkpoint.predecessor_drift != ADAPTER_DRIFT:
        raise ComparisonPrerequisiteError("predecessor drift binding changed")
    if len(b6) != 4 or len(b7) != 4 or len(pairs) != 4:
        raise ComparisonPrerequisiteError("runtime pair accounting changed")
    expected_case_order = tuple(item.case_id for item in _load_cases(root / STEP93_CASES))
    if len({item.case_id for item in pairs}) != 4 or tuple(item.case_id for item in pairs) != expected_case_order:
        raise ComparisonPrerequisiteError("runtime pair order changed")
    expected_b6, expected_b7, expected_pairs = _expected_runtime_outputs(root, config)
    if (b6, b7, pairs) != (expected_b6, expected_b7, expected_pairs):
        raise ComparisonPrerequisiteError("frozen Step 9.3 runtime identity changed")
    b6_index = {item.prediction_id: item for item in b6}
    b7_index = {item.prediction_id: item for item in b7}
    if len(b6_index) != 4 or len(b7_index) != 4:
        raise ComparisonPrerequisiteError("runtime prediction IDs are duplicated")
    for pair in pairs:
        left = b6_index.get(pair.b6_prediction_id)
        right = b7_index.get(pair.b7_prediction_id)
        if left is None or right is None:
            raise ComparisonPrerequisiteError("pair prediction is missing")
        if (pair.input_id, pair.case_id, pair.user_id) != (
            left.input_id, left.case_id, left.user_id,
        ) or (pair.input_id, pair.case_id, pair.user_id) != (
            right.input_id, right.case_id, right.user_id,
        ):
            raise ComparisonPrerequisiteError("pair identity differs from predictions")
        if (
            hashlib.sha256(canonical_json_bytes(left)).hexdigest() != pair.b6_prediction_sha256
            or hashlib.sha256(canonical_json_bytes(right)).hexdigest() != pair.b7_prediction_sha256
        ):
            raise ComparisonPrerequisiteError("pair prediction hash changed")
        _assert_matched(left, right)
    return checkpoint


def _expected_runtime_outputs(root: Path, config: Mapping[str, object]):
    cases = _load_cases(root / STEP93_CASES)
    requirements = _load_requirements(root / STEP93_REQUIREMENTS)
    plan_rows = _read_jsonl(root / STEP93_RUNTIME_ROOT / "plans.jsonl")
    result_rows = _read_jsonl(root / STEP93_RUNTIME_ROOT / "retrieval-results.jsonl")
    package_rows = _read_jsonl(root / STEP93_RUNTIME_ROOT / "packages.jsonl")
    decision_rows = _read_jsonl(root / STEP93_RUNTIME_ROOT / "decisions.jsonl")
    if not all(len(values) == 4 for values in (
        cases, requirements, plan_rows, result_rows, package_rows, decision_rows,
    )):
        raise ComparisonPrerequisiteError("frozen Step 9.3 accounting changed")
    b6_predictions = []
    b7_predictions = []
    pairs = []
    for case, requirement_set, plan_row, result_row, package_row, decision_row in zip(
        cases, requirements, plan_rows, result_rows, package_rows, decision_rows, strict=True,
    ):
        plan_values = dict(plan_row)
        required_predicates = plan_values.get("required_predicates")
        if not isinstance(required_predicates, list):
            raise ComparisonPrerequisiteError("frozen Step 9.3 plan predicates changed")
        plan_values["required_predicates"] = tuple(required_predicates)
        execution_plan = InteractiveExecutionPlan(**plan_values)
        result = _parse_result(result_row)
        package = _parse_package(package_row)
        decision = answerability_decision_from_mapping(decision_row)
        requirement_sha256 = hashlib.sha256(
            interactive_json_bytes(requirement_set)
        ).hexdigest()
        result_sha256 = package_sha256(result)
        package_hash = hashlib.sha256(package_json_bytes(package)).hexdigest()
        if (
            (
                execution_plan.case_id, execution_plan.query_id,
                execution_plan.requirement_set_id, execution_plan.requirement_set_sha256,
                execution_plan.user_id, execution_plan.plan_id,
                execution_plan.snapshot_run_id, execution_plan.retrieval_execution_id,
            )
            != (
                case.case_id, result.query_id, requirement_set.requirement_set_id,
                requirement_sha256, case.user_id, result.plan_id,
                result.snapshot_run_id, result.execution_id,
            )
            or result.baseline_id != PACKAGE_BASELINE
            or (
                package.user_id, package.query.query_id, package.baseline_id,
                package.execution_id, package.plan.plan_id, package.snapshot_run_id,
                package.as_of, package.retrieval_result_sha256,
            )
            != (
                case.user_id, result.query_id, PACKAGE_BASELINE, result.execution_id,
                result.plan_id, result.snapshot_run_id, case.as_of, result_sha256,
            )
            or (
                decision.user_id, decision.query_id, decision.baseline_id,
                decision.execution_id, decision.plan_id, decision.snapshot_run_id,
                decision.package_id, decision.package_sha256,
                decision.decision, decision.generation_allowed,
            )
            != (
                case.user_id, result.query_id, PACKAGE_BASELINE, result.execution_id,
                result.plan_id, result.snapshot_run_id, package.package_id, package_hash,
                "abstain", False,
            )
        ):
            raise ComparisonPrerequisiteError("frozen Step 9.3 identity chain changed")
        runtime_input = _runtime_input(
            case, requirement_set, execution_plan, result, package, package_hash,
        )
        answer = build_memory_answer(
            _answer_view(package, package_hash), config_path=root / MEMORY_CONFIG,
        )
        threshold = _threshold_application(decision)
        b6 = _prediction(runtime_input, config, "B6", answer)
        b7 = _prediction(runtime_input, config, "B7", answer, decision, threshold)
        b6_predictions.append(b6)
        b7_predictions.append(b7)
        pairs.append(_pair(runtime_input, b6, b7))
    return tuple(b6_predictions), tuple(b7_predictions), tuple(pairs)


def _runtime_input(case, requirement_set, execution_plan, result, package, package_hash):
    fields = {
        "case_id": case.case_id,
        "user_id": case.user_id,
        "query_id": result.query_id,
        "as_of": case.as_of,
        "requirement_set_id": requirement_set.requirement_set_id,
        "requirement_set_sha256": hashlib.sha256(interactive_json_bytes(requirement_set)).hexdigest(),
        "execution_plan_id": execution_plan.execution_plan_id,
        "plan_id": result.plan_id,
        "snapshot_run_id": result.snapshot_run_id,
        "retrieval_execution_id": result.execution_id,
        "retrieval_result_sha256": package_sha256(result),
        "package_id": package.package_id,
        "package_sha256": package_hash,
        "package_baseline_id": package.baseline_id,
        "accepted_index_record_ids": tuple(sorted(item.index_record_id for item in result.accepted)),
        "relevant_source_ids": tuple(sorted(item.source_id for item in package.relevant_sources)),
    }
    return ComparableRuntimeInput(input_id=stable_sha256(fields), **fields)


def _prediction(runtime_input, config, wrapper, answer, decision=None, threshold=None):
    fields = {
        "comparison_version": COMPARISON_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "runtime_release_version": RUNTIME_RELEASE_VERSION,
        "input_id": runtime_input.input_id,
        "case_id": runtime_input.case_id,
        "user_id": runtime_input.user_id,
        "query_id": runtime_input.query_id,
        "as_of": runtime_input.as_of,
        "requirement_set_id": runtime_input.requirement_set_id,
        "execution_plan_id": runtime_input.execution_plan_id,
        "plan_id": runtime_input.plan_id,
        "snapshot_run_id": runtime_input.snapshot_run_id,
        "retrieval_execution_id": runtime_input.retrieval_execution_id,
        "retrieval_result_sha256": runtime_input.retrieval_result_sha256,
        "package_id": runtime_input.package_id,
        "package_sha256": runtime_input.package_sha256,
        "package_baseline_id": runtime_input.package_baseline_id,
        "wrapper_baseline_id": wrapper,
        "gate_applied": wrapper == "B7",
        "answerability_decision_id": None if decision is None else decision.decision_id,
        "answerability_decision_sha256": None if decision is None else hashlib.sha256(package_json_bytes(decision)).hexdigest(),
        "threshold_profile": None if threshold is None else threshold.profile_id,
        "threshold_application_id": None if threshold is None else threshold.application_id,
        "configured_future_answer_model": config["configured_future_answer_model"],
        "prompt_version": config["prompt_version"],
        "prompt_sha256": config["prompt_sha256"],
        "memory_answer_config_sha256": config["memory_answer_config_sha256"],
        "response_action": "abstained",
        "response_status": answer.status,
        "response_text_sha256": hashlib.sha256(answer.answer.encode("utf-8")).hexdigest(),
        "underlying_answer_id": answer.answer_id,
        "generation_allowed": False,
        "provider_eligible": False,
        "provider_returned_model": None,
        "provider_request_count": 0,
        "retry_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": 0,
    }
    return ComparablePrediction(prediction_id=stable_sha256(fields), **fields)


def _pair(runtime_input, b6, b7):
    _assert_matched(b6, b7)
    fields = {
        "input_id": runtime_input.input_id,
        "case_id": runtime_input.case_id,
        "user_id": runtime_input.user_id,
        "b6_prediction_id": b6.prediction_id,
        "b6_prediction_sha256": hashlib.sha256(canonical_json_bytes(b6)).hexdigest(),
        "b7_prediction_id": b7.prediction_id,
        "b7_prediction_sha256": hashlib.sha256(canonical_json_bytes(b7)).hexdigest(),
        "shared_identity_sha256": runtime_input.input_id,
        "difference_whitelist": DIFFERENCE_WHITELIST,
        "response_identity_match": True,
    }
    return ComparablePair(pair_id=stable_sha256(fields), **fields)


def _assert_matched(b6: ComparablePrediction, b7: ComparablePrediction) -> None:
    left = asdict(b6)
    right = asdict(b7)
    for name in DIFFERENCE_WHITELIST:
        left.pop(name)
        right.pop(name)
    if left != right:
        raise ComparisonPrerequisiteError("B6/B7 shared identity differs")
    if (
        b6.response_action, b6.response_status, b6.response_text_sha256,
        b6.underlying_answer_id,
    ) != (
        b7.response_action, b7.response_status, b7.response_text_sha256,
        b7.underlying_answer_id,
    ):
        raise ComparisonPrerequisiteError("B6/B7 structural response differs")


def _require_recomputed_step93(root, plans, results, packages, decisions) -> None:
    payloads = {
        "plans.jsonl": _serialize_with(plans, interactive_json_bytes),
        "retrieval-results.jsonl": _serialize_with(results, interactive_json_bytes),
        "packages.jsonl": _serialize_with(packages, interactive_json_bytes),
        "decisions.jsonl": _serialize_with(decisions, interactive_json_bytes),
    }
    for name, data in payloads.items():
        committed = root / STEP93_RUNTIME_ROOT / name
        if data != committed.read_bytes() or hashlib.sha256(data).hexdigest() != STEP93_HASHES[name]:
            raise ComparisonPrerequisiteError(f"recomputed Step 9.3 {name} changed")


def _verify_predecessors(root: Path) -> None:
    verify_answer_quality_release(root / STEP83_ROOT, repo_root=root)
    verify_answerability_release(root / STEP91_ROOT, repo_root=root)
    verify_answerability_threshold_release_v2(root / STEP92_ROOT, repo_root=root)
    verify_interactive_runtime_checkpoint(root / STEP93_RUNTIME_ROOT, repo_root=root)
    exact = (
        (root / STEP91_ROOT / "manifest.json", STEP91_MANIFEST_SHA256),
        (root / STEP92_ROOT / "manifest.json", STEP92_MANIFEST_SHA256),
        (root / STEP93_FINAL_MANIFEST, STEP93_HASHES["manifest.json"]),
        (root / STEP93_RUNTIME_ROOT / "checkpoint_manifest.json", STEP93_HASHES["checkpoint_manifest.json"]),
    )
    for path, expected in exact:
        if _sha(path) != expected:
            raise ComparisonPrerequisiteError("predecessor authority changed")
    for name in ("plans.jsonl", "retrieval-results.jsonl", "packages.jsonl", "decisions.jsonl", "responses.jsonl"):
        if _sha(root / STEP93_RUNTIME_ROOT / name) != STEP93_HASHES[name]:
            raise ComparisonPrerequisiteError("Step 9.3 runtime artifact changed")


def _verify_dataset_manifest(root: Path, config_sha256: str) -> None:
    value = _read_object(root / DATASET_MANIFEST)
    if value != {
        "dataset_version": DATASET_VERSION,
        "runtime_release_version": RUNTIME_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "config_sha256": config_sha256,
        "development_case_count": 4,
        "roadmap_target_case_count": 20,
        "frozen_test_deferred_count": 16,
        "step9_3_manifest_sha256": STEP93_HASHES["manifest.json"],
        "step9_3_checkpoint_sha256": STEP93_HASHES["checkpoint_manifest.json"],
        "step9_3_cases_sha256": _sha(root / STEP93_CASES),
        "step9_3_requirements_sha256": _sha(root / STEP93_REQUIREMENTS),
        "step9_1_manifest_sha256": STEP91_MANIFEST_SHA256,
        "step9_2_manifest_sha256": STEP92_MANIFEST_SHA256,
        "package_baseline_id": PACKAGE_BASELINE,
        "wrapper_baselines": ["B6", "B7"],
        "reference_available": False,
        "step9_4_scorer_available": False,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
        "excluded_inputs": [
            "environment and credentials", "frozen test records", "network and providers",
            "oracle and review queues", "pilot answer references", "Step 9.3 gold and reference",
            "Step 9.3 scorecard and per-case report", "Step 9.4 scorer and reference",
        ],
    }:
        raise ComparisonPrerequisiteError("dataset manifest changed")


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(item) for item in values)


def _serialize_with(values: Sequence[object], encoder) -> bytes:
    return b"".join(encoder(item) for item in values)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ComparisonPrerequisiteError("JSONL row lacks newline")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ComparisonPrerequisiteError("JSONL row is not an object")
        rows.append(value)
    return tuple(rows)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ComparisonPrerequisiteError("JSON object is invalid")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ComparisonPrerequisiteError("output directory must be empty")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
