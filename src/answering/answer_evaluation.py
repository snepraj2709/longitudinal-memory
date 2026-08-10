"""Immutable development release for the memory-answer contract."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .answer_contracts import (
    BASELINE_ORDER,
    DORMANT_MODEL,
    OUTPUT_RELEASE_VERSION,
    MemoryAnswer,
    MemoryAnswerChecks,
    MemoryAnswerError,
    MemoryAnswerFailure,
)
from .answer_input import (
    STEP81_DATASET_SHA256,
    STEP81_CHECKS_SHA256,
    STEP81_FAILURES_SHA256,
    STEP81_MANIFEST_SHA256,
    STEP81_PACKAGES_SHA256,
    STEP81_RUN_SHA256,
    load_answer_package_views,
)
from .contracts import canonical_json_bytes, stable_sha256
from .memory_answer import (
    ABSTENTION_REASONS,
    build_memory_answer,
    load_memory_answer_config,
    memory_answer_from_mapping,
)


GUIDANCE_VERSION = "step-8.2-guidance-v1"
GUIDANCE_SHA256 = "91593dde89451249910fdbc00b0f048db4964d499b0c83d3c60deba5c3a2033d"
STARTING_COMMIT = "8fec075d754dff7f12821947919d5c01f867d949"
DATASET_MANIFEST = Path("data/answering/memory-answer-contract-development-v1/manifest.json")
RESULT_ROOT = Path("results/answering/memory-answer-contract-development-v1")
ARTIFACTS = ("answers.jsonl", "failures.jsonl", "checks.json", "run.json", "findings.md")
IMPLEMENTATION_PATHS = (
    "configs/answering/memory_answer_v1.json",
    "src/answering/answer_contracts.py",
    "src/answering/answer_input.py",
    "src/answering/memory_answer.py",
    "src/answering/answer_evaluation.py",
)


class MemoryAnswerEvaluationError(RuntimeError):
    """A sanitized release failure."""


def execute_memory_answer_evaluation(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> MemoryAnswerChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    dataset = _dataset(root)
    views = load_answer_package_views(root)
    answers: list[MemoryAnswer] = []
    failures: list[MemoryAnswerFailure] = []
    for view in views:
        try:
            answers.append(build_memory_answer(
                view, config_path=root / "configs/answering/memory_answer_v1.json"
            ))
        except Exception as error:
            failures.append(_failure(view, error))
    answers.sort(key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id]))
    failures.sort(key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id]))
    checks = score_memory_answers(views, answers, failures)
    payloads = _artifact_payloads(checks, answers, failures)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        _write_exclusive(output / name, payloads[name])
    _write_exclusive(
        output / "manifest.json",
        canonical_json_bytes(_manifest(root, dataset, checks, payloads)),
    )
    verify_memory_answer_release(output, repo_root=root)
    return checks


def score_memory_answers(
    views,
    answers: Sequence[MemoryAnswer],
    failures: Sequence[MemoryAnswerFailure],
) -> MemoryAnswerChecks:
    view_keys = {(item.package_id, item.query_id, item.baseline_id) for item in views}
    answer_keys = {(item.package_id, item.query_id, item.baseline_id) for item in answers}
    failure_keys = {(item.package_id, item.query_id, item.baseline_id) for item in failures}
    checks = MemoryAnswerChecks(
        OUTPUT_RELEASE_VERSION,
        len(views),
        len(answers),
        sum(item.status == "abstained" for item in answers),
        sum(item.status == "answered" for item in answers),
        sum(item.status == "disputed" for item in answers),
        sum(item.status == "partially_answered" for item in answers),
        sum(item.structural_blockers == ("no_promoted_claims",) for item in answers),
        sum(len(item.statements) for item in answers),
        sum(len(statement.citations) for item in answers for statement in item.statements),
        len(failures),
        len(answers) + len(failures) - len(answer_keys | failure_keys),
        sum(item.user_id != view.user_id for item in answers for view in views if item.package_id == view.package_id),
        sum((item.package_id, item.query_id, item.baseline_id) not in view_keys for item in answers),
        0,
        0,
        0,
        0,
        0,
    )
    expected = (
        checks.package_count, checks.answer_count, checks.abstained_count,
        checks.no_promoted_claims_count,
    )
    zeros = (
        checks.answered_count, checks.disputed_count, checks.partially_answered_count,
        checks.statement_count, checks.citation_count, checks.failure_count,
        checks.duplicate_count, checks.cross_user_count, checks.invalid_provenance_count,
        checks.provider_request_count, checks.retry_count, checks.input_token_count,
        checks.output_token_count, checks.incremental_cost_usd,
    )
    if expected != (24, 24, 24, 24) or any(zeros) or answer_keys != view_keys:
        raise MemoryAnswerEvaluationError("memory answer structural checks failed")
    return checks


def verify_memory_answer_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    dataset = _dataset(root)
    views = load_answer_package_views(root)
    manifest = _read_object(output / "manifest.json")
    for name in ARTIFACTS:
        if manifest.get("artifacts", {}).get(name) != _sha(output / name):
            raise MemoryAnswerEvaluationError("release artifact hash changed")
    answers = tuple(
        memory_answer_from_mapping(value) for value in _read_jsonl(output / "answers.jsonl")
    )
    failures = tuple(_failure_from_mapping(value) for value in _read_jsonl(output / "failures.jsonl"))
    checks = score_memory_answers(views, answers, failures)
    if _read_object(output / "checks.json") != asdict(checks):
        raise MemoryAnswerEvaluationError("release checks do not recompute")
    expected_answers = tuple(build_memory_answer(
        view, config_path=root / "configs/answering/memory_answer_v1.json"
    ) for view in views)
    if answers != expected_answers:
        raise MemoryAnswerEvaluationError("release answers do not recompute")
    payloads = _artifact_payloads(checks, answers, failures)
    for name in ARTIFACTS:
        if (output / name).read_bytes() != payloads[name]:
            raise MemoryAnswerEvaluationError("release payload does not recompute")
    expected_manifest = _manifest(root, dataset, checks, payloads)
    if manifest != expected_manifest:
        raise MemoryAnswerEvaluationError("release manifest does not recompute")


def _artifact_payloads(checks, answers, failures) -> dict[str, bytes]:
    return {
        "answers.jsonl": b"".join(canonical_json_bytes(item) for item in answers),
        "failures.jsonl": b"".join(canonical_json_bytes(item) for item in failures),
        "checks.json": canonical_json_bytes(checks),
        "run.json": canonical_json_bytes({
            "output_release_version": OUTPUT_RELEASE_VERSION,
            "input_release_version": "evidence_package_development_v1",
            "package_count": checks.package_count,
            "answer_count": checks.answer_count,
            "failure_count": checks.failure_count,
            "execution_mode": "deterministic_structural_abstention",
            "requested_model": DORMANT_MODEL,
            "resolved_model": DORMANT_MODEL,
            "model_state": "dormant_no_call",
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        }),
        "findings.md": _findings().encode("utf-8"),
    }


def _manifest(root, dataset, checks, payloads) -> Mapping[str, object]:
    return {
        "output_release_version": OUTPUT_RELEASE_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "dataset": dataset,
        "dataset_manifest_sha256": _sha(root / DATASET_MANIFEST),
        "input_authorities": {
            "dataset_manifest_sha256": STEP81_DATASET_SHA256,
            "result_manifest_sha256": STEP81_MANIFEST_SHA256,
            "packages_sha256": STEP81_PACKAGES_SHA256,
            "checks_sha256": STEP81_CHECKS_SHA256,
            "run_sha256": STEP81_RUN_SHA256,
            "failures_sha256": STEP81_FAILURES_SHA256,
        },
        "implementation_hashes": {
            path: _sha(root / path) for path in IMPLEMENTATION_PATHS
        },
        "artifacts": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        "checks": asdict(checks),
        "model_usage": {
            "provider_requests": 0, "retries": 0, "input_tokens": 0,
            "output_tokens": 0, "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
        "excluded_inputs": dataset["excluded_inputs"],
        "limitations": [
            "All development packages contain only candidate claims.",
            "This release checks answer structure and abstention behavior, not answer correctness.",
        ],
        "predecessor_drift": [],
    }


def _dataset(root: Path) -> Mapping[str, object]:
    value = _read_object(root / DATASET_MANIFEST)
    expected = {
        "dataset_version": OUTPUT_RELEASE_VERSION,
        "split": "development",
        "allowed_users": ["user_001", "user_002"],
        "expected_package_count": 24,
        "expected_answer_count": 24,
        "expected_status": "abstained",
        "input_release": {
            "version": "evidence_package_development_v1",
            "dataset_manifest_sha256": STEP81_DATASET_SHA256,
            "result_manifest_sha256": STEP81_MANIFEST_SHA256,
            "packages_sha256": STEP81_PACKAGES_SHA256,
            "checks_sha256": STEP81_CHECKS_SHA256,
            "run_sha256": STEP81_RUN_SHA256,
            "failures_sha256": STEP81_FAILURES_SHA256,
        },
        "excluded_inputs": [
            "answer gold", "retrieval relevance gold", "oracle data", "review queues",
            "Step 6.4 gold", "users user_003 through user_010", ".env",
            "provider credentials",
        ],
        "model_execution": "dormant_no_call",
        "provider_requests": 0,
        "incremental_cost_usd": 0,
    }
    if value != expected:
        raise MemoryAnswerEvaluationError("dataset manifest is invalid")
    return value


def _failure(view, error: Exception) -> MemoryAnswerFailure:
    code = "runtime_failure"
    location = "runtime"
    if isinstance(error, MemoryAnswerError):
        code, location = "input_invalid", "input"
    payload = {
        "package_id": view.package_id,
        "query_id": view.query_id,
        "baseline_id": view.baseline_id,
        "code": code,
        "location": location,
    }
    return MemoryAnswerFailure(stable_sha256(payload), None, **payload)


def _failure_from_mapping(value: object) -> MemoryAnswerFailure:
    if not isinstance(value, dict) or set(value) != {
        "failure_id", "answer_id", "package_id", "query_id", "baseline_id", "code", "location"
    }:
        raise MemoryAnswerEvaluationError("failure record is invalid")
    return MemoryAnswerFailure(**value)


def _findings() -> str:
    return (
        "# Memory answer contract development findings\n\n"
        "All 24 packages were structurally blocked because they contain only candidate claims. "
        "Each package produced the configured abstention with no prompt or provider call.\n\n"
        "This release checks the answer schema and provenance rules. It does not measure answer "
        "correctness or abstention accuracy.\n"
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise MemoryAnswerEvaluationError("result directory must be empty")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise MemoryAnswerEvaluationError("JSONL record is invalid")
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MemoryAnswerEvaluationError("JSON object is invalid")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
