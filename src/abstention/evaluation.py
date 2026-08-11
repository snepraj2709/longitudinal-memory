"""Deterministic development release for the answerability policy."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import (
    OUTPUT_RELEASE_VERSION,
    AnswerabilityChecks,
    AnswerabilityDecision,
    AnswerabilityFailure,
    answerability_decision_from_mapping,
    answerability_failure_from_mapping,
    canonical_json_bytes,
    stable_sha256,
)
from .input import (
    STEP81_CHECKS_SHA256,
    STEP81_DATASET_SHA256,
    STEP81_FAILURES_SHA256,
    STEP81_MANIFEST_SHA256,
    STEP81_PACKAGES_SHA256,
    STEP81_RUN_SHA256,
    STEP82_MANIFEST_SHA256,
    STEP83_MANIFEST_SHA256,
    load_answerability_inputs,
)
from .policy import AnswerabilityPolicyError, decide_answerability


GUIDANCE_VERSION = "step-9.1-guidance-v1"
GUIDANCE_SHA256 = "9cc8c86bf9725d04c71d2033df14939e392166377ecd6fc5a70955f8c717f48b"
START_COMMIT = "a18501a27708c259cccce8bf87948962e672bd41"
DATASET_MANIFEST = Path("data/abstention/answerability-development-v1/manifest.json")
RESULT_ROOT = Path("results/abstention/answerability-development-v1")
CONFIG_PATH = Path("configs/abstention/answerability_v1.json")
IMPLEMENTATION_PATHS = (
    Path("src/abstention/__init__.py"),
    Path("src/abstention/contracts.py"),
    Path("src/abstention/input.py"),
    Path("src/abstention/policy.py"),
    Path("src/abstention/evaluation.py"),
)
ARTIFACTS = ("decisions.jsonl", "failures.jsonl", "checks.json", "run.json", "findings.md")
BASELINE_ORDER = {"B2": 0, "B3": 1, "B4": 2}
COMPATIBILITY_RULINGS = (
    "7f022204628f60e1079d4dbe1c70fdea3aa30d1b891f1027ee3e131231b04def",
    "6faffe095dc298db3234e8d66d47ddd3826062d65e41c3d3411cf10927f87172",
    "6308f6c27642fbe3e0172d4b28140dcb3bb1b2091d94abcafd50a64712fe6eaf",
    "15e90d9f1132bee9d902d9591d67204cb9fb7c87da4260ea27692bba9ff74eb9",
)
PREDECESSOR_DRIFT = (
    {
        "path": "tests/integration/test_answer_quality_evaluation.py",
        "old_sha256": "6f2a30fd38b3d880fd4083ab88006dc708ea966554961b1cb91cb395ceb90a28",
        "new_sha256": "933a7b9378d99957295004bbc81926707e79e4c7cf6e5f68d3f89d8f76421dc4",
        "reason": "compatibility_rulings_1_3_4_step91_topology_only",
    },
    {
        "path": "tests/integration/test_memory_answer.py",
        "old_sha256": "bcb41709723cdf8e9fc6617acd16903ce8202979016d3ac2d7e8e333e76466ea",
        "new_sha256": "5b68e3cb8037c78e841ce7293498859fc2c1d5397f5fe55b087b1127d3a40818",
        "reason": "compatibility_ruling_2_step91_topology_only",
    },
)


class AnswerabilityEvaluationError(RuntimeError):
    """Fail a release that does not recompute exactly."""


def execute_answerability_evaluation(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> AnswerabilityChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    dataset = _load_dataset(root)
    config_sha = _sha(root / CONFIG_PATH)
    policy_sha = _sha(root / "src/abstention/policy.py")
    pairs = load_answerability_inputs(root, config_sha256=config_sha, policy_sha256=policy_sha)
    decisions: list[AnswerabilityDecision] = []
    failures: list[AnswerabilityFailure] = []
    for request, package in pairs:
        try:
            decisions.append(decide_answerability(request, package))
        except Exception as error:
            failures.append(_failure(request.package_id, request.query_id, request.baseline_id, error))
    decisions.sort(key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id]))
    failures.sort(key=lambda item: (item.query_id, BASELINE_ORDER.get(item.baseline_id, 99)))
    checks = score_answerability(pairs, decisions, failures)
    payloads = _artifact_payloads(checks, decisions, failures)
    manifest = _manifest(root, dataset, checks, payloads)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    _write_exclusive(output / "manifest.json", canonical_json_bytes(manifest))
    verify_answerability_release(output, repo_root=root)
    return checks


def score_answerability(
    pairs: Sequence[tuple[object, object]],
    decisions: Sequence[AnswerabilityDecision],
    failures: Sequence[AnswerabilityFailure],
) -> AnswerabilityChecks:
    if len(decisions) + len(failures) != len(pairs):
        raise AnswerabilityEvaluationError("decision accounting changed")
    by_package = {decision.package_id: decision for decision in decisions}
    if len(by_package) != len(decisions):
        raise AnswerabilityEvaluationError("decision package IDs are duplicated")
    for request, package in pairs:
        decision = by_package.get(request.package_id)
        if decision is None:
            continue
        if (
            decision.request_id != request.request_id
            or decision.user_id != request.user_id
            or decision.query_id != request.query_id
            or decision.baseline_id != request.baseline_id
            or decision.package_sha256 != request.package_sha256
            or tuple(
                (
                    item.rejection_id, item.index_record_id, item.retrieval_rank,
                    item.claim_id, item.claim_version_id, item.stage, item.reasons,
                )
                for item in decision.rejected_evidence
            ) != tuple(
                (
                    item.rejection_id, item.index_record_id, item.retrieval_rank,
                    item.claim_id, item.claim_version_id, item.stage, item.reasons,
                )
                for item in package.rejected_evidence
            )
        ):
            raise AnswerabilityEvaluationError("decision identity or carried rejection changed")
    checks = AnswerabilityChecks(
        len(pairs),
        len(decisions),
        sum(item.baseline_id == "B2" for item in decisions),
        sum(item.baseline_id == "B3" for item in decisions),
        sum(item.baseline_id == "B4" for item in decisions),
        sum(item.decision == "answerable" for item in decisions),
        sum(item.decision == "abstain" for item in decisions),
        sum(item.decision == "clarify" for item in decisions),
        sum(item.generation_allowed for item in decisions),
        sum(len(item.accepted_evidence) for item in decisions),
        sum(len(item.rejected_evidence) for item in decisions),
        sum(
            rejection.stage == "package_validation" and rejection.reasons == ("candidate_not_promoted",)
            for item in decisions for rejection in item.rejected_evidence
        ),
        sum(
            rejection.stage in {"pre_filter", "post_rank"}
            for item in decisions for rejection in item.rejected_evidence
        ),
        sum(item.primary_reason == "no_promoted_claims" for item in decisions),
        sum(
            item.confidence.value is None
            and item.confidence.calibration_status == "not_calibrated"
            and item.confidence.null_reason == "step_9_2_not_run"
            for item in decisions
        ),
        len(failures),
        0,
    )
    if asdict(checks) != {
        "package_count": 24,
        "decision_count": 24,
        "b2_count": 8,
        "b3_count": 8,
        "b4_count": 8,
        "answerable_count": 0,
        "abstain_count": 24,
        "clarify_count": 0,
        "generation_allowed_count": 0,
        "accepted_evidence_count": 0,
        "rejected_evidence_count": 491,
        "candidate_rejection_count": 295,
        "retrieval_rejection_count": 196,
        "no_promoted_claims_count": 24,
        "uncalibrated_confidence_count": 24,
        "failure_count": 0,
        "provider_request_count": 0,
    }:
        raise AnswerabilityEvaluationError("development answerability checks failed")
    return checks


def verify_answerability_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {*ARTIFACTS, "manifest.json"}:
        raise AnswerabilityEvaluationError("release artifact set changed")
    dataset = _load_dataset(root)
    config_sha = _sha(root / CONFIG_PATH)
    policy_sha = _sha(root / "src/abstention/policy.py")
    pairs = load_answerability_inputs(root, config_sha256=config_sha, policy_sha256=policy_sha)
    decision_values = _read_jsonl(output / "decisions.jsonl")
    failure_values = _read_jsonl(output / "failures.jsonl")
    decisions = tuple(answerability_decision_from_mapping(value) for value in decision_values)
    failures = tuple(answerability_failure_from_mapping(value) for value in failure_values)
    expected_order = tuple(sorted(decisions, key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id])))
    if decisions != expected_order:
        raise AnswerabilityEvaluationError("decision order changed")
    recomputed = tuple(decide_answerability(request, package) for request, package in pairs)
    if decisions != recomputed:
        raise AnswerabilityEvaluationError("decisions do not recompute")
    checks = score_answerability(pairs, decisions, failures)
    payloads = _artifact_payloads(checks, decisions, failures)
    for name, payload in payloads.items():
        if (output / name).read_bytes() != payload:
            raise AnswerabilityEvaluationError(f"{name} does not recompute")
    expected_manifest = _manifest(root, dataset, checks, payloads)
    if _read_object(output / "manifest.json") != expected_manifest:
        raise AnswerabilityEvaluationError("manifest does not recompute")


def _artifact_payloads(checks, decisions, failures) -> dict[str, bytes]:
    return {
        "decisions.jsonl": b"".join(canonical_json_bytes(item) for item in decisions),
        "failures.jsonl": b"".join(canonical_json_bytes(item) for item in failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes({
            "guidance_version": GUIDANCE_VERSION,
            "runtime_version": "answerability_runtime_v1",
            "input_release_version": "evidence_package_development_v1",
            "output_release_version": OUTPUT_RELEASE_VERSION,
            "package_count": checks.package_count,
            "decision_count": checks.decision_count,
            "failure_count": checks.failure_count,
            "model": "none",
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        }),
        "findings.md": _findings(checks).encode("utf-8"),
    }


def _manifest(root, dataset, checks, payloads) -> Mapping[str, object]:
    return {
        "release_version": OUTPUT_RELEASE_VERSION,
        "guidance": {
            "version": GUIDANCE_VERSION,
            "sha256": GUIDANCE_SHA256,
            "starting_commit": START_COMMIT,
            "contamination_boundary": "clean_room_no_rejected_prompt_content",
        },
        "compatibility_ruling_sha256": list(COMPATIBILITY_RULINGS),
        "dataset": {"path": DATASET_MANIFEST.as_posix(), "sha256": _sha(root / DATASET_MANIFEST)},
        "configuration": {"path": CONFIG_PATH.as_posix(), "sha256": _sha(root / CONFIG_PATH)},
        "implementation": {
            path.as_posix(): _sha(root / path) for path in IMPLEMENTATION_PATHS
        },
        "inputs": dataset["inputs"],
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "counts": asdict(checks),
        "reason_precedence": dataset["reason_precedence"],
        "confidence_policy": {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "step_9_2_not_run",
        },
        "excluded_inputs": dataset["excluded_inputs"],
        "predecessor_drift": list(PREDECESSOR_DRIFT),
        "model_usage": {
            "model": "none",
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
        "limitations": [
            "All development claim versions remain candidates.",
            "Positive semantic branches are exercised only with invented controlled fixtures.",
            "The release checks deterministic policy behavior; it does not measure answerability accuracy.",
            "Decision confidence remains uncalibrated until Step 9.2.",
        ],
    }


def _load_dataset(root: Path) -> Mapping[str, object]:
    value = _read_object(root / DATASET_MANIFEST)
    if (
        value.get("dataset_version") != "answerability-development-v1"
        or value.get("guidance_sha256") != GUIDANCE_SHA256
        or value.get("allowed_users") != ["user_001", "user_002"]
        or value.get("expected_package_count") != 24
        or value.get("predecessor_drift") != []
    ):
        raise AnswerabilityEvaluationError("dataset manifest identity changed")
    expected_inputs = {
        "step_8_1_dataset_manifest": STEP81_DATASET_SHA256,
        "step_8_1_result_manifest": STEP81_MANIFEST_SHA256,
        "step_8_1_packages": STEP81_PACKAGES_SHA256,
        "step_8_1_checks": STEP81_CHECKS_SHA256,
        "step_8_1_run": STEP81_RUN_SHA256,
        "step_8_1_failures": STEP81_FAILURES_SHA256,
        "step_8_2_prerequisite_manifest": STEP82_MANIFEST_SHA256,
        "step_8_3_prerequisite_manifest": STEP83_MANIFEST_SHA256,
    }
    if {name: binding.get("sha256") for name, binding in value.get("inputs", {}).items()} != expected_inputs:
        raise AnswerabilityEvaluationError("dataset input bindings changed")
    if value.get("config_sha256") != _sha(root / CONFIG_PATH):
        raise AnswerabilityEvaluationError("dataset config binding changed")
    return value


def _failure(package_id, query_id, baseline_id, error) -> AnswerabilityFailure:
    code, location = "invariant_violation", "policy"
    if isinstance(error, AnswerabilityPolicyError):
        code, location = error.code, error.location
    fields = {
        "package_id": package_id,
        "query_id": query_id,
        "baseline_id": baseline_id,
        "code": code,
        "location": location,
    }
    return AnswerabilityFailure(failure_id=stable_sha256(fields), **fields)


def _findings(checks: AnswerabilityChecks) -> str:
    return (
        "# Answerability development findings\n\n"
        f"The policy evaluated {checks.package_count} evidence packages without a runtime failure. "
        "Every package abstained because its retrieved claims remain candidates.\n\n"
        f"The decisions preserve all {checks.rejected_evidence_count} upstream rejection rows. "
        "No evidence was accepted for generation, and confidence remains uncalibrated until Step 9.2.\n\n"
        "Positive authority, time, conflict, trait, and causal branches are covered by invented fixtures only. "
        "The candidate-only development release does not exercise them.\n\n"
        "This release checks deterministic policy behavior. It does not measure whether the abstentions are correct. "
        "No model or provider was used.\n"
    )


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise AnswerabilityEvaluationError("JSONL record is invalid")
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AnswerabilityEvaluationError("JSON object is invalid")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AnswerabilityEvaluationError("result directory must be empty")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
