"""Deterministic no-call execution of the four development interactive cases."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from answering.answer_contracts import (
    INPUT_RELEASE_VERSION as ANSWER_INPUT_RELEASE_VERSION,
    AnswerPackageClaim,
    AnswerPackageSpan,
    AnswerPackageView,
)
from answering.contracts import (
    CONFIG_VERSION as PACKAGE_CONFIG_VERSION,
    INPUT_RELEASE_VERSION as PACKAGE_INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION as PACKAGE_RUNTIME_VERSION,
    SCHEMA_VERSION as PACKAGE_SCHEMA_VERSION,
    EvidencePackage,
    EvidencePackageBuildRequest,
    FrozenEligibilitySnapshot,
    canonical_json_bytes as package_json_bytes,
    stable_sha256 as package_sha256,
)
from answering.evidence_package import build_evidence_package, load_evidence_package_config
from answering.memory_answer import build_memory_answer
from answering.repository import EvidencePackageRepository
from retrieval.baseline_contracts import BASELINE_RECORD_KINDS, BaselineRetrievalResult
from retrieval.baselines import load_baseline_config
from retrieval.index_evaluation import execute_index_evaluation
from retrieval.query_contracts import INDEX_VERSION, RetrievalQueryRequest, request_canonical_value
from retrieval.query_planner import build_query_plan, load_query_planner_config
from retrieval.search_repository import RetrievalSearchRepository

from .calibration import load_threshold_config
from .calibration_contracts import (
    THRESHOLD_VERSION,
    ThresholdApplication,
    canonical_json_bytes as calibration_json_bytes,
    stable_sha256 as calibration_sha256,
)
from .contracts import (
    ANSWERABILITY_VERSION,
    CONFIG_VERSION as ANSWERABILITY_CONFIG_VERSION,
    INPUT_RELEASE_VERSION as ANSWERABILITY_INPUT_RELEASE_VERSION,
    POLICY_VERSION,
    RUNTIME_VERSION as ANSWERABILITY_RUNTIME_VERSION,
    SCHEMA_VERSION as ANSWERABILITY_SCHEMA_VERSION,
    AnswerabilityRequest,
    AnswerabilityRequirement,
    stable_sha256 as answerability_sha256,
)
from .evaluation import verify_answerability_release
from .input import STEP81_MANIFEST_SHA256, STEP81_PACKAGES_SHA256
from .calibration_release_v2 import verify_answerability_threshold_release_v2
from .interactive_contracts_v2 import (
    CONFIG_VERSION,
    DERIVATION_INPUTS,
    FINAL_RELEASE_VERSION,
    INTERACTIVE_VERSION,
    PREDICATE_REGISTRY_SHA256,
    PREDICATE_REGISTRY_VERSION,
    REQUIREMENT_REVIEW_STATUS,
    RUNTIME_RELEASE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    SYSTEM_VARIANT,
    InteractiveAnsweringError,
    InteractiveExecutionPlan,
    InteractivePrediction,
    InteractiveRequirementPart,
    InteractiveRequirementSet,
    InteractiveRuntimeCase,
    InteractiveTurn,
    canonical_json_bytes,
    stable_sha256,
)
from .policy import decide_answerability


CONFIG_PATH = Path("configs/abstention/interactive_answering_v2.json")
RUNTIME_CASES = Path("data/abstention/interactive-answering-development-v2/runtime/cases.jsonl")
RUNTIME_REQUIREMENTS = Path("data/abstention/interactive-answering-development-v2/runtime/requirements.jsonl")
RUNTIME_MANIFEST = Path("data/abstention/interactive-answering-development-v2/runtime/manifest.json")
RUNTIME_ROOT = Path("results/abstention/interactive-answering-development-runtime-v2")
GUIDANCE_SHA256 = "865223b8aea8665b8df8b10839f9d1f1cdb9c9c5a31d17f97e6f47ad27606a77"
CORRECTION_1_SHA256 = "34a82610f0287d6f5321d7be6c1b75fb6fe475a984c6418cfb67d4b31b160880"
CORRECTION_2_SHA256 = "5480ebcbe58fbeeb17bfbdb3e30c0aa4a7e9ed41477976233c0efe19a72733a1"
INVALID_V1_CHECKPOINT_SHA256 = "0b7be7fe41bb9275115a32ef8998862707a7a0a945e56754bdba77556b00017e"
INVALID_V1_BACKUP_MAP_SHA256 = "ee40e76025fa0d7ae92ee565460c2c49dae1e0134aca4ab9da01319298659748"
BASELINE_MANIFEST_SHA256 = "ab45d4a51766d9d0edcc39c77f8b0ccd4abbcb1254437153f7759418cfe01703"
BASELINE_CHECKPOINT_SHA256 = "d99a2ee8720f16fb40867f92d1740582536ecced84c7eadfffc5f1907065e3b4"
BASELINE_RESULTS_SHA256 = "e1e69fe8ac64a81f27e171cdd7082b7648e2a339659f11e6e4f2a1702a1dbb60"
ABSTENTION_CONFIG = Path("configs/abstention/answerability_v1.json")
ABSTENTION_POLICY = Path("src/abstention/policy.py")
THRESHOLD_CONFIG = Path("configs/abstention/answerability_thresholds_v1.json")
PREDICATE_REGISTRY = Path("configs/extraction/predicate_registry_v2.json")
STEP91_ROOT = Path("results/abstention/answerability-development-v1")
STEP92_ROOT = Path("results/abstention/answerability-threshold-development-v2")


def load_interactive_config(path: str | Path = CONFIG_PATH) -> tuple[Mapping[str, object], str]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InteractiveAnsweringError("interactive config is invalid") from error
    expected = {
        "configured_future_answer_model", "config_version", "development_case_count",
        "final_release_version", "fixed_clarification_text", "frozen_test_deferred_count",
        "incremental_cost_usd", "interactive_version", "max_output_tokens",
        "predicate_registry_path", "predicate_registry_sha256", "predicate_registry_version",
        "provider_execution_enabled", "roadmap_target_case_count", "runtime_release_version",
        "runtime_version", "schema_version", "scorer_version",
        "selected_threshold_profile", "system_variant",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise InteractiveAnsweringError("interactive config fields changed")
    if (
        value["interactive_version"] != INTERACTIVE_VERSION
        or value["schema_version"] != SCHEMA_VERSION
        or value["config_version"] != CONFIG_VERSION
        or value["runtime_version"] != RUNTIME_VERSION
        or value["runtime_release_version"] != RUNTIME_RELEASE_VERSION
        or value["final_release_version"] != FINAL_RELEASE_VERSION
        or value["system_variant"] != SYSTEM_VARIANT
        or value["selected_threshold_profile"] != "ordinary_1_trait_2"
        or value["scorer_version"] != "interactive_answering_scorer_v2"
        or value["configured_future_answer_model"] != "gpt-4.1-2025-04-14"
        or value["predicate_registry_path"] != str(PREDICATE_REGISTRY)
        or value["predicate_registry_sha256"] != PREDICATE_REGISTRY_SHA256
        or value["predicate_registry_version"] != PREDICATE_REGISTRY_VERSION
        or value["roadmap_target_case_count"] != 20
        or value["development_case_count"] != 4
        or value["frozen_test_deferred_count"] != 16
        or value["provider_execution_enabled"] is not False
        or value["max_output_tokens"] != 0
        or value["incremental_cost_usd"] != 0
    ):
        raise InteractiveAnsweringError("interactive policy changed")
    return value, hashlib.sha256(raw).hexdigest()


def query_request_for_case(case: InteractiveRuntimeCase) -> RetrievalQueryRequest:
    return RetrievalQueryRequest(
        query_id=f"interactive_query_{case.case_id}",
        user_id=case.user_id,
        query_text=case.initial_user_message,
        as_of=case.as_of,
        index_version=INDEX_VERSION,
        enabled_record_kinds=BASELINE_RECORD_KINDS["B4"],
        requested_valid_time=None,
        speaker_ids=(),
        entity_ids=(case.user_id,),
        sensitivity_scope="sensitive",
        allow_unclassified_sensitivity=True,
    )


def derive_requirement_set(
    case: InteractiveRuntimeCase,
    *,
    repo_root: str | Path = ".",
) -> InteractiveRequirementSet:
    root = Path(repo_root).resolve()
    request = query_request_for_case(case)
    planner = load_query_planner_config(root / "configs/retrieval/query_planner_v1.json")
    plan = build_query_plan(request, config=planner)
    semantic = {
        "scaled_user_001_interactive_extraction_001": (
            ("has_mentor", "relationship", False),
            ("leads_project", "fact", False),
        ),
        "scaled_user_001_interactive_temporal_reasoning_002": (
            ("career_goal", "change_over_time", False),
        ),
        "scaled_user_002_interactive_conflict_detection_001": (
            ("job_start_date", "evidence_request", True),
        ),
        "scaled_user_002_interactive_abstention_002": (
            ("office_base", "current_state", False),
        ),
    }.get(case.case_id)
    if semantic is None:
        raise InteractiveAnsweringError("runtime case has no reviewed requirement mapping")
    registered = _registered_predicates(root / PREDICATE_REGISTRY)
    required_predicates = {item[0] for item in semantic}
    if not required_predicates.issubset(registered):
        raise InteractiveAnsweringError("runtime requirement predicate is not registered")
    if registered.intersection({"project_assignment", "career_direction", "start_date"}):
        raise InteractiveAnsweringError("aggregate predicate alias entered the registry")
    parts = []
    for predicate, information_kind, conflict_requested in semantic:
        fields = {
            "information_kind": information_kind,
            "target_subject_ids": (case.user_id,),
            "permitted_speaker_ids": (),
            "requested_valid_time": None,
            "required_authority_class": "exact_supported",
            "required_predicate": predicate,
            "required_object_json": None,
            "required_polarity": None,
            "complete_subpart_coverage": True,
            "conflict_reporting_requested": conflict_requested,
            "partial_response_requested": False,
            "subrequirement_ids": (),
        }
        parts.append(InteractiveRequirementPart(
            requirement_id=stable_sha256(fields), **fields,
        ))
    set_fields = {
        "case_id": case.case_id,
        "query_id": request.query_id,
        "query_type": plan.primary_label,
        "parts": tuple(sorted(parts, key=lambda item: item.requirement_id)),
        "predicate_registry_version": PREDICATE_REGISTRY_VERSION,
        "predicate_registry_sha256": PREDICATE_REGISTRY_SHA256,
        "derivation_inputs": DERIVATION_INPUTS,
        "prior_development_gold_exposure": True,
        "development_gold_used": False,
        "review_status": REQUIREMENT_REVIEW_STATUS,
    }
    return InteractiveRequirementSet(
        requirement_set_id=stable_sha256(set_fields), **set_fields,
    )


def _registered_predicates(path: Path) -> frozenset[str]:
    if _sha(path) != PREDICATE_REGISTRY_SHA256:
        raise InteractiveAnsweringError("predicate registry bytes changed")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("registry_version") != PREDICATE_REGISTRY_VERSION:
        raise InteractiveAnsweringError("predicate registry version changed")
    rows = value.get("predicates")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("predicate"), str) for row in rows
    ):
        raise InteractiveAnsweringError("predicate registry rows are invalid")
    predicates = tuple(row["predicate"] for row in rows)
    if predicates != tuple(sorted(set(predicates))):
        raise InteractiveAnsweringError("predicate registry order changed")
    return frozenset(predicates)


def execute_interactive_runtime(
    connection_factory,
    output_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[InteractivePrediction, ...]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    _require_runtime_only_tree(root)
    verify_answerability_release(root / STEP91_ROOT, repo_root=root)
    verify_answerability_threshold_release_v2(root / STEP92_ROOT, repo_root=root)
    config, config_sha = load_interactive_config(root / CONFIG_PATH)
    cases = _load_cases(root / RUNTIME_CASES)
    requirements = _load_requirements(root / RUNTIME_REQUIREMENTS)
    if tuple(item.case_id for item in cases) != tuple(item.case_id for item in requirements):
        raise InteractiveAnsweringError("case and requirement order differs")
    derived = tuple(derive_requirement_set(case, repo_root=root) for case in cases)
    if requirements != derived:
        raise InteractiveAnsweringError("frozen requirement sets do not recompute")
    planner_config = load_query_planner_config(root / "configs/retrieval/query_planner_v1.json")
    baseline_config = load_baseline_config(root / "configs/retrieval/baseline_v1.json")
    _, package_config_sha = load_evidence_package_config(root / "configs/answering/evidence_package_v1.json")
    threshold_config = load_threshold_config(root / THRESHOLD_CONFIG, repo_root=root)
    if threshold_config.selected_profile != "ordinary_1_trait_2":
        raise InteractiveAnsweringError("selected threshold profile changed")
    answerability_config_sha = _sha(root / ABSTENTION_CONFIG)
    answerability_policy_sha = _sha(root / ABSTENTION_POLICY)

    preflight = {
        "interactive_version": INTERACTIVE_VERSION,
        "runtime_release_version": RUNTIME_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "correction_1_sha256": CORRECTION_1_SHA256,
        "correction_2_sha256": CORRECTION_2_SHA256,
        "config_sha256": config_sha,
        "development_case_count": 4,
        "roadmap_target_case_count": 20,
        "frozen_test_deferred_count": 16,
        "invalid_v1_checkpoint_sha256": INVALID_V1_CHECKPOINT_SHA256,
        "invalid_v1_backup_map_sha256": INVALID_V1_BACKUP_MAP_SHA256,
        "invalid_v1_repository_paths_present": False,
        "prior_development_gold_exposure": True,
        "development_gold_used_for_v2_runtime": False,
        "reviewer_frozen_snippet_exposure_disclosed": True,
        "reviewer_frozen_snippet_used": False,
        "runtime_case_records_requested": 4,
        "record_five_requested": False,
        "gold_paths_absent": True,
        "scorer_paths_absent": True,
        "final_result_paths_absent": True,
        "oracle_opened": False,
        "review_queue_opened": False,
        "frozen_test_opened": False,
        "credential_or_environment_opened": False,
        "provider_path_opened": False,
        "provider_requests": 0,
    }

    with tempfile.TemporaryDirectory() as directory:
        execute_index_evaluation(
            connection_factory,
            Path(directory) / "index-release",
            repo_root=root,
        )

    plans: list[InteractiveExecutionPlan] = []
    results: list[BaselineRetrievalResult] = []
    packages: list[EvidencePackage] = []
    decisions = []
    responses: list[InteractivePrediction] = []
    connection = connection_factory()
    try:
        search = RetrievalSearchRepository(connection)
        package_repository = EvidencePackageRepository(connection)
        for case, requirement_set in zip(cases, requirements, strict=True):
            request = query_request_for_case(case)
            plan = build_query_plan(request, config=planner_config)
            if (requirement_set.query_id, requirement_set.query_type) != (
                request.query_id, plan.primary_label,
            ):
                raise InteractiveAnsweringError("frozen requirement differs from planner input")
            required_predicates = tuple(sorted({
                part.required_predicate for part in requirement_set.parts
                if part.required_predicate is not None
            }))
            result = search.retrieve(
                request, "B4", planner_config=planner_config, baseline_config=baseline_config,
            )
            if result.plan_id != plan.plan_id:
                raise InteractiveAnsweringError("runtime plan differs from retrieval plan")
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
            package_hash = hashlib.sha256(package_json_bytes(package)).hexdigest()
            answerability_request = _answerability_request(
                requirement_set, package, package_hash,
                answerability_config_sha, answerability_policy_sha,
            )
            decision = decide_answerability(answerability_request, package)
            threshold = _threshold_application(decision)
            if threshold.generation_allowed or threshold.output_decision == "answerable":
                raise InteractiveAnsweringError("generation_allowed_stop_before_prompt")
            view = _answer_view(package, package_hash)
            answer = build_memory_answer(
                view, config_path=root / "configs/answering/memory_answer_v1.json",
            )
            action = "clarification_requested" if threshold.output_decision == "clarify" else "abstained"
            text = (
                str(config["fixed_clarification_text"])
                if action == "clarification_requested" else answer.answer
            )
            turn = InteractiveTurn(1, "assistant", action, text, answer.answer_id)
            execution_fields = {
                "case_id": case.case_id,
                "query_id": request.query_id,
                "requirement_set_id": requirement_set.requirement_set_id,
                "requirement_set_sha256": hashlib.sha256(
                    canonical_json_bytes(requirement_set)
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
                execution_plan_id=stable_sha256(execution_fields), **execution_fields,
            )
            prediction_fields = {
                "interactive_version": INTERACTIVE_VERSION,
                "schema_version": SCHEMA_VERSION,
                "runtime_version": RUNTIME_VERSION,
                "case_id": case.case_id,
                "benchmark_version": case.benchmark_version,
                "split": case.split,
                "user_id": case.user_id,
                "capability": case.capability,
                "difficulty": case.difficulty,
                "as_of": case.as_of,
                "allowed_turns": case.allowed_turns,
                "system_variant": SYSTEM_VARIANT,
                "query_id": request.query_id,
                "requirement_set_id": requirement_set.requirement_set_id,
                "requirement_set_sha256": hashlib.sha256(
                    canonical_json_bytes(requirement_set)
                ).hexdigest(),
                "required_predicates": required_predicates,
                "execution_plan_id": execution_plan.execution_plan_id,
                "plan_id": plan.plan_id,
                "snapshot_run_id": result.snapshot_run_id,
                "retrieval_execution_id": result.execution_id,
                "retrieval_result_sha256": package_sha256(result),
                "package_id": package.package_id,
                "package_sha256": package_hash,
                "answerability_decision_id": decision.decision_id,
                "answerability_decision_sha256": hashlib.sha256(
                    package_json_bytes(decision)
                ).hexdigest(),
                "threshold_profile": "ordinary_1_trait_2",
                "threshold_application_id": threshold.application_id,
                "response_action": action,
                "decision": threshold.output_decision,
                "generation_allowed": False,
                "turn_count": 1,
                "turns": (turn,),
                "predicted_behaviours": ("clarification",) if action == "clarification_requested" else ("abstention",),
                "predicted_claim_ids": (),
                "predicted_evidence_tuples": (),
                "factual_statement_count": 0,
                "citation_count": 0,
                "provider_eligible": False,
            }
            prediction = InteractivePrediction(
                prediction_id=stable_sha256(prediction_fields), **prediction_fields,
            )
            plans.append(execution_plan)
            results.append(result)
            packages.append(package)
            decisions.append(decision)
            responses.append(prediction)
    finally:
        connection.close()

    if len(responses) != 4 or any(item.response_action != "abstained" for item in responses):
        raise InteractiveAnsweringError("development status distribution changed")
    payloads = {
        "preflight.json": canonical_json_bytes(preflight),
        "plans.jsonl": _serialize(plans),
        "retrieval-results.jsonl": _serialize(results),
        "packages.jsonl": _serialize(packages),
        "decisions.jsonl": _serialize(decisions),
        "responses.jsonl": _serialize(responses),
        "failures.jsonl": b"",
    }
    checkpoint = {
        "interactive_version": INTERACTIVE_VERSION,
        "runtime_release_version": RUNTIME_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "config_sha256": config_sha,
        "runtime_case_count": 4,
        "requirement_set_count": 4,
        "requirement_part_count": 5,
        "response_count": 4,
        "failure_count": 0,
        "provider_eligible_case_count": 0,
        "provider_request_count": 0,
        "retry_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": 0,
        "gold_opened": False,
        "prior_development_gold_exposure": True,
        "development_gold_used_for_v2_runtime": False,
        "reviewer_frozen_snippet_exposure_disclosed": True,
        "reviewer_frozen_snippet_used": False,
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        "implementation": {
            str(path): _sha(root / path) for path in (
                CONFIG_PATH,
                Path("src/abstention/interactive_contracts_v2.py"),
                Path("src/abstention/interactive_input_v2.py"),
                Path("src/abstention/interactive_runtime_v2.py"),
                RUNTIME_MANIFEST,
                RUNTIME_CASES,
                RUNTIME_REQUIREMENTS,
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        _write_exclusive(output / name, data)
    _write_exclusive(output / "checkpoint_manifest.json", canonical_json_bytes(checkpoint))
    verify_interactive_runtime_checkpoint(output, repo_root=root)
    return tuple(responses)


def verify_interactive_runtime_checkpoint(
    output_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    checkpoint = json.loads((output / "checkpoint_manifest.json").read_bytes())
    if (
        checkpoint.get("interactive_version") != INTERACTIVE_VERSION
        or checkpoint.get("runtime_release_version") != RUNTIME_RELEASE_VERSION
        or checkpoint.get("guidance_sha256") != GUIDANCE_SHA256
        or checkpoint.get("runtime_case_count") != 4
        or checkpoint.get("requirement_set_count") != 4
        or checkpoint.get("requirement_part_count") != 5
        or checkpoint.get("response_count") != 4
        or checkpoint.get("failure_count") != 0
        or checkpoint.get("provider_eligible_case_count") != 0
        or checkpoint.get("gold_opened") is not False
        or checkpoint.get("prior_development_gold_exposure") is not True
        or checkpoint.get("development_gold_used_for_v2_runtime") is not False
        or checkpoint.get("reviewer_frozen_snippet_exposure_disclosed") is not True
        or checkpoint.get("reviewer_frozen_snippet_used") is not False
    ):
        raise InteractiveAnsweringError("runtime checkpoint accounting changed")
    artifacts = checkpoint.get("artifacts")
    implementations = checkpoint.get("implementation")
    if not isinstance(artifacts, dict) or not isinstance(implementations, dict):
        raise InteractiveAnsweringError("runtime checkpoint map is invalid")
    for name, expected in artifacts.items():
        if _sha(output / name) != expected:
            raise InteractiveAnsweringError("runtime artifact changed")
    for name, expected in implementations.items():
        if _sha(root / name) != expected:
            raise InteractiveAnsweringError("runtime implementation changed")
    rows = _read_jsonl(output / "responses.jsonl")
    predictions = tuple(_prediction_from_row(item) for item in rows)
    if len(predictions) != 4 or any(
        item.response_action != "abstained" or item.provider_eligible for item in predictions
    ):
        raise InteractiveAnsweringError("runtime predictions changed")


def _answerability_request(
    requirement_set: InteractiveRequirementSet,
    package: EvidencePackage,
    package_hash: str,
    config_sha: str,
    policy_sha: str,
) -> AnswerabilityRequest:
    answer_requirements = tuple(AnswerabilityRequirement(
        requirement_id=part.requirement_id,
        information_kind=part.information_kind,
        target_subject_ids=part.target_subject_ids,
        permitted_speaker_ids=part.permitted_speaker_ids,
        requested_valid_time=part.requested_valid_time,
        required_authority_class=part.required_authority_class,
        required_predicate=part.required_predicate,
        required_object_json=part.required_object_json,
        required_polarity=part.required_polarity,
        complete_subpart_coverage=part.complete_subpart_coverage,
        conflict_reporting_requested=part.conflict_reporting_requested,
        partial_response_requested=part.partial_response_requested,
        subrequirement_ids=part.subrequirement_ids,
    ) for part in requirement_set.parts)
    subjects = tuple(sorted({
        subject for part in requirement_set.parts for subject in part.target_subject_ids
    }))
    speakers = tuple(sorted({
        speaker for part in requirement_set.parts for speaker in part.permitted_speaker_ids
    }))
    request_fields = {
        "answerability_version": ANSWERABILITY_VERSION,
        "schema_version": ANSWERABILITY_SCHEMA_VERSION,
        "config_version": ANSWERABILITY_CONFIG_VERSION,
        "config_sha256": config_sha,
        "policy_version": POLICY_VERSION,
        "policy_sha256": policy_sha,
        "runtime_version": ANSWERABILITY_RUNTIME_VERSION,
        "input_release_version": ANSWERABILITY_INPUT_RELEASE_VERSION,
        "input_release_manifest_sha256": STEP81_MANIFEST_SHA256,
        "package_id": package.package_id,
        "package_sha256": package_hash,
        "user_id": package.user_id,
        "query_id": package.query.query_id,
        "baseline_id": package.baseline_id,
        "execution_id": package.execution_id,
        "plan_id": package.plan.plan_id,
        "snapshot_run_id": package.snapshot_run_id,
        "index_version": package.index_version,
        "as_of": package.as_of,
        "requested_valid_time": package.requested_valid_time,
        "query_type": package.query_type,
        "subject_scope": subjects,
        "speaker_scope": speakers,
        "sensitivity_scope": "sensitive",
        "requirements": answer_requirements,
    }
    return AnswerabilityRequest(request_id=answerability_sha256(request_fields), **request_fields)


def _threshold_application(decision) -> ThresholdApplication:
    fields = {
        "threshold_version": THRESHOLD_VERSION,
        "profile_id": "ordinary_1_trait_2",
        "input_decision_id": decision.decision_id,
        "input_decision_sha256": hashlib.sha256(package_json_bytes(decision)).hexdigest(),
        "package_id": decision.package_id,
        "user_id": decision.user_id,
        "query_id": decision.query_id,
        "baseline_id": decision.baseline_id,
        "input_decision": decision.decision,
        "output_decision": decision.decision,
        "primary_reason": decision.primary_reason,
        "reasons": decision.reasons,
        "generation_allowed": decision.generation_allowed,
        "accepted_evidence_count": len(decision.accepted_evidence),
        "accepted_evidence_sha256": hashlib.sha256(
            calibration_json_bytes(decision.accepted_evidence)
        ).hexdigest(),
        "rejected_evidence_count": len(decision.rejected_evidence),
        "rejected_evidence_sha256": hashlib.sha256(
            calibration_json_bytes(decision.rejected_evidence)
        ).hexdigest(),
        "withheld_by_threshold": False,
        "confidence": {
            "value": None, "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        },
    }
    return ThresholdApplication(application_id=calibration_sha256(fields), **fields)


def _answer_view(package: EvidencePackage, package_hash: str) -> AnswerPackageView:
    claims = []
    for category, values in (
        ("current_claims", package.current_claims),
        ("historical_claims", package.historical_claims),
        ("conflicting_claims", package.conflicting_claims),
    ):
        for claim in values:
            claims.append(AnswerPackageClaim(
                category, claim.user_id, claim.claim_id, claim.claim_version_id,
                claim.subject_id, claim.speaker_id, claim.predicate, claim.object_json,
                claim.polarity, claim.epistemic_status, claim.lifecycle_status,
                claim.valid_from_date, claim.valid_from_timestamp, claim.valid_to_date,
                claim.valid_to_timestamp, claim.time_precision, claim.evidence_ids,
            ))
    spans = []
    for source in package.relevant_sources:
        for span in source.evidence_spans:
            spans.append(AnswerPackageSpan(
                span.evidence_id, package.user_id, span.claim_id, span.claim_version_id,
                span.source_id, span.span_id, span.message_id, span.verbatim_quote,
            ))
    return AnswerPackageView(
        ANSWER_INPUT_RELEASE_VERSION, STEP81_MANIFEST_SHA256, STEP81_PACKAGES_SHA256,
        package.package_id, package_hash, package.user_id, package.query.query_id,
        package.query.query_text, package.query_type, package.baseline_id,
        package.execution_id, package.plan.plan_id, package.snapshot_run_id,
        package.index_version, package.as_of, package.requested_valid_time,
        package.answer_allowed, package.structural_blockers,
        tuple(sorted(claims, key=lambda item: (
            ("current_claims", "historical_claims", "conflicting_claims").index(item.category),
            item.claim_id, item.claim_version_id,
        ))),
        tuple(sorted(spans, key=lambda item: (
            item.claim_id, item.claim_version_id, item.evidence_id, item.source_id, item.span_id,
        ))),
    )


def _frozen_eligibility(index_version: str, result: BaselineRetrievalResult) -> FrozenEligibilitySnapshot:
    eligible = tuple(sorted({
        *(item.index_record_id for item in result.accepted),
        *(item.index_record_id for item in result.rejected if item.stage == "post_rank"),
    }))
    pre = tuple(sorted(
        (item for item in result.rejected if item.stage == "pre_filter"),
        key=lambda item: item.index_record_id,
    ))
    value = {
        "plan_id": result.plan_id,
        "user_id": result.user_id,
        "index_version": index_version,
        "snapshot_run_id": result.snapshot_run_id,
        "eligible_record_ids": eligible,
        "pre_filter_rejections": pre,
    }
    return FrozenEligibilitySnapshot(
        result.plan_id, result.user_id, index_version, result.snapshot_run_id,
        eligible, pre, package_sha256(value),
    )


def _load_cases(path: Path) -> tuple[InteractiveRuntimeCase, ...]:
    from .interactive_contracts_v2 import runtime_case_from_mapping
    rows = tuple(runtime_case_from_mapping(item) for item in _read_jsonl(path))
    if len(rows) != 4 or len({item.case_id for item in rows}) != 4:
        raise InteractiveAnsweringError("runtime case accounting changed")
    return rows


def _load_requirements(path: Path) -> tuple[InteractiveRequirementSet, ...]:
    from .interactive_contracts_v2 import requirement_set_from_mapping
    rows = tuple(requirement_set_from_mapping(item) for item in _read_jsonl(path))
    if (
        len(rows) != 4
        or len({item.requirement_set_id for item in rows}) != 4
        or sum(len(item.parts) for item in rows) != 5
    ):
        raise InteractiveAnsweringError("runtime requirement accounting changed")
    return rows


def _prediction_from_row(value: object) -> InteractivePrediction:
    from .interactive_contracts_v2 import prediction_from_mapping
    return prediction_from_mapping(value)


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(value) for value in values)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise InteractiveAnsweringError("JSONL row lacks newline")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise InteractiveAnsweringError("JSONL row is not an object")
        rows.append(value)
    return tuple(rows)


def _require_runtime_only_tree(root: Path) -> None:
    forbidden = (
        root / "data/abstention/interactive-answering-development-v2/gold",
        root / "data/abstention/interactive-answering-development-v2/manifest.json",
        root / "src/abstention/interactive_evaluation_v2.py",
        root / "results/abstention/interactive-answering-development-v2",
        root / "configs/abstention/interactive_answering_v1.json",
        root / "data/abstention/interactive-answering-development-v1",
        root / "results/abstention/interactive-answering-development-runtime-v1",
        root / "results/abstention/interactive-answering-development-v1",
        root / "src/abstention/interactive_contracts.py",
        root / "src/abstention/interactive_input.py",
        root / "src/abstention/interactive_runtime.py",
        root / "src/abstention/interactive_evaluation.py",
    )
    if any(path.exists() for path in forbidden):
        raise InteractiveAnsweringError("runtime-only or invalid-v1 path boundary changed")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise InteractiveAnsweringError("output directory is not empty")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
