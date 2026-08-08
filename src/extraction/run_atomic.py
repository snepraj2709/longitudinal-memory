"""Run atomic extraction with frozen inputs, budgets, checkpoints, and safe resume."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence, TextIO

from evaluation.openai_client import (
    OpenAIModelMismatchError,
    OpenAIRateLimitMetadata,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)
from evaluation.run_config import assert_no_secrets, canonical_sha256

from .atomic import AtomicExtractionResult, AtomicExtractionValidationError, validate_atomic_response
from .predicate_registry import (
    DEFAULT_PREDICATE_REGISTRY_PATH,
    PredicateRegistryError,
    load_predicate_registry,
)
from .gold import ATOMIC_GOLD_PATH, AtomicGoldCase, load_atomic_gold
from .prompt import (
    ATOMIC_EXTRACTION_PROMPT_VERSION,
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
)
from .run_safety import (
    DEFAULT_CONFIG_PATH,
    AtomicRunConfig,
    AtomicRunConfigError,
    cost_text,
    load_atomic_run_config,
    token_cost,
)
from .scoring import ATOMIC_SCORING_VERSION, score_atomic_extraction
from .source import PILOT_SOURCE_DIR, ExtractionSource, load_pilot_sources


DEFAULT_OUTPUT_DIR = Path("results/phase3/atomic-extraction-v3-safety-v1")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4_000
FROZEN_ATOMIC_GOLD_SHA256 = (
    "e802835dcb3e4278ec68e45474d094896dcad9a780c4115a01e0f0bc44f07f63"
)
ATOMIC_CASE_REFS = (
    ("atomic_cal_001", "cal_001"),
    ("atomic_conv_002", "conv_002"),
    ("atomic_conv_003", "conv_003"),
    ("atomic_cal_002", "cal_002"),
    ("atomic_conv_004", "conv_004"),
    ("atomic_cal_003", "cal_003"),
    ("atomic_email_003", "email_003"),
    ("atomic_email_005", "email_005"),
    ("atomic_conv_010", "conv_010"),
    ("atomic_cal_006", "cal_006"),
)
_SOURCE_FILENAMES = ("calendar.jsonl", "conversations.jsonl", "emails.jsonl")
_CHECKPOINT_FILES = {"run.json", "predictions.jsonl", "failures.jsonl"}
_RUN_RECORD_FIELDS = {
    "run_format_version", "run_status", "started_at", "completed_at",
    "repository_commit", "repository_dirty", "resume_count",
    "configuration_version", "configuration_sha256", "provider",
    "dataset_version", "dataset_split", "runtime_dataset_sha256",
    "requested_model", "resolved_model", "temperature", "generation_settings",
    "prompt_version", "prompt_sha256", "scoring_version",
    "predicate_registry_version", "predicate_registry_sha256", "case_order",
    "source_file_sha256", "gold_file_sha256", "planned_request_count",
    "maximum_request_attempts", "maximum_retry_requests",
    "provider_requests_attempted", "successful_cases", "failed_cases",
    "remaining_cases", "budget_input_tokens", "budget_output_tokens",
    "estimated_cost_usd", "hard_cost_cap_usd", "prior_provider_failures",
    "attempts", "output_file_sha256",
}
_RATE_LIMIT_FIELDS = {
    "remaining_requests", "remaining_tokens", "reset_requests_seconds",
    "reset_tokens_seconds", "remaining_project_tokens",
    "reset_project_tokens_seconds",
}


class AtomicPipelineError(RuntimeError):
    """Raised when the pilot cannot run safely."""


@dataclass(frozen=True)
class AtomicRunPlan:
    """Source-only inputs prepared before extraction starts."""

    case_refs: tuple[tuple[str, str], ...]
    selected_sources: tuple[ExtractionSource, ...]
    all_sources: tuple[ExtractionSource, ...]
    prompts: tuple[str, ...]
    prompt_input_token_upper_bounds: tuple[int, ...]
    gold_file_sha256: str
    source_file_sha256: Mapping[str, str]
    config: AtomicRunConfig


@dataclass(frozen=True)
class AtomicDryRun:
    """Deterministic provider-free preflight summary."""

    plan: AtomicRunPlan
    record: Mapping[str, object]


@dataclass(frozen=True)
class AtomicCaseExecution:
    position: int
    case_id: str
    source_id: str
    attempted: bool
    passed: bool
    claims: tuple[object, ...]
    provider_metadata: OpenAIResponseMetadata | None
    failure_stage: str | None
    error: str | None
    budget_input_tokens: int
    budget_output_tokens: int


def prepare_atomic_run(
    *,
    repo_root: str | Path = ".",
    source_groups: Sequence[ExtractionSource] | None = None,
    validate_gold: bool = False,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AtomicRunPlan:
    """Prepare and verify source-only inputs without parsing scorer-only gold."""

    root = Path(repo_root).resolve()
    try:
        config = load_atomic_run_config(root, config_path)
    except AtomicRunConfigError as error:
        raise AtomicPipelineError(str(error)) from error
    if config.case_order != ATOMIC_CASE_REFS:
        raise AtomicPipelineError("frozen case order is incompatible with the runner")
    if config.prompt_version != ATOMIC_EXTRACTION_PROMPT_VERSION:
        raise AtomicPipelineError("the extraction prompt version drifted")
    try:
        registry = load_predicate_registry(
            root / DEFAULT_PREDICATE_REGISTRY_PATH
        )
    except PredicateRegistryError as error:
        raise AtomicPipelineError(str(error)) from error
    if config.predicate_registry_version != registry.registry_version:
        raise AtomicPipelineError("the predicate registry version drifted")
    if registry.content_sha256 != config.predicate_registry_sha256:
        raise AtomicPipelineError("the predicate registry changed")

    gold_hash = _file_sha256(root / ATOMIC_GOLD_PATH)
    if gold_hash != FROZEN_ATOMIC_GOLD_SHA256 or gold_hash != config.gold_file_sha256:
        raise AtomicPipelineError("the atomic gold file does not match its frozen hash")
    source_hashes = _source_hashes(root)
    if source_hashes != dict(config.source_file_sha256):
        raise AtomicPipelineError("the runtime source files changed")
    runtime_dataset_hash = canonical_sha256({
        "source_file_sha256": source_hashes,
        "case_order": [
            {"case_id": case_id, "source_id": source_id}
            for case_id, source_id in ATOMIC_CASE_REFS
        ],
    })
    if runtime_dataset_hash != config.runtime_dataset_sha256:
        raise AtomicPipelineError("the runtime dataset fingerprint changed")

    all_sources = tuple(source_groups) if source_groups is not None else load_pilot_sources()
    sources_by_id = {source.source_id: source for source in all_sources}
    missing = [source_id for _, source_id in ATOMIC_CASE_REFS if source_id not in sources_by_id]
    if missing:
        raise AtomicPipelineError("missing atomic source IDs: " + ", ".join(missing))
    selected = tuple(sources_by_id[source_id] for _, source_id in ATOMIC_CASE_REFS)
    prompts = tuple(build_atomic_extraction_prompt(source) for source in selected)
    prompt_payload = {
        "system_prompt": ATOMIC_EXTRACTION_SYSTEM_PROMPT,
        "cases": [
            {"case_id": case_id, "source_id": source_id, "prompt": prompt}
            for (case_id, source_id), prompt in zip(ATOMIC_CASE_REFS, prompts)
        ],
    }
    if canonical_sha256(prompt_payload) != config.prompt_sha256:
        raise AtomicPipelineError("the rendered extraction prompts changed")
    _assert_source_only_prompts(prompts)

    if validate_gold:
        gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, all_sources)
        _require_frozen_case_order(gold_cases)

    upper_bounds = tuple(
        len((ATOMIC_EXTRACTION_SYSTEM_PROMPT + prompt).encode("utf-8"))
        for prompt in prompts
    )
    if sum(upper_bounds) * 2 != config.maximum_input_tokens:
        raise AtomicPipelineError("the frozen input-token ceiling no longer covers the prompts")
    return AtomicRunPlan(
        case_refs=ATOMIC_CASE_REFS,
        selected_sources=selected,
        all_sources=all_sources,
        prompts=prompts,
        prompt_input_token_upper_bounds=upper_bounds,
        gold_file_sha256=gold_hash,
        source_file_sha256=source_hashes,
        config=config,
    )


def dry_run_atomic(
    *, repo_root: str | Path = ".", stdout: TextIO | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> AtomicDryRun:
    """Render and validate the plan with no provider calls or output writes."""

    stdout = stdout or sys.stdout
    plan = prepare_atomic_run(repo_root=repo_root, validate_gold=False, config_path=config_path)
    config = plan.config
    estimated_input = sum(plan.prompt_input_token_upper_bounds)
    expected_cost = token_cost(config, estimated_input, config.expected_output_tokens)
    maximum_cost = token_cost(
        config, config.maximum_input_tokens, config.maximum_output_tokens
    )
    record: dict[str, object] = {
        "dry_run": True,
        "provider_calls": 0,
        "output_writes": 0,
        "configuration_version": config.configuration_version,
        "configuration_sha256": config.configuration_sha256,
        "requested_model": config.requested_model,
        "resolved_model": config.resolved_model,
        "dataset_version": config.dataset_version,
        "dataset_split": config.dataset_split,
        "runtime_dataset_sha256": config.runtime_dataset_sha256,
        "prompt_version": config.prompt_version,
        "prompt_sha256": config.prompt_sha256,
        "predicate_registry_version": config.predicate_registry_version,
        "predicate_registry_sha256": config.predicate_registry_sha256,
        "case_order": [
            {"position": position, "case_id": case_id, "source_id": source_id}
            for position, (case_id, source_id) in enumerate(plan.case_refs, 1)
        ],
        "planned_request_count": config.planned_request_count,
        "maximum_retry_requests": config.maximum_retry_requests,
        "estimated_input_tokens_upper_bound": estimated_input,
        "expected_output_tokens": config.expected_output_tokens,
        "expected_cost_usd": cost_text(expected_cost),
        "maximum_request_attempts": config.maximum_request_attempts,
        "maximum_input_tokens": config.maximum_input_tokens,
        "maximum_output_tokens": config.maximum_output_tokens,
        "maximum_cost_usd": cost_text(maximum_cost),
        "hard_cost_cap_usd": cost_text(config.hard_cost_cap_usd),
        "input_usd_per_million_tokens": str(config.input_usd_per_million_tokens),
        "output_usd_per_million_tokens": str(config.output_usd_per_million_tokens),
        "generation_settings": dict(config.generation_settings),
        "source_file_sha256": dict(config.source_file_sha256),
        "gold_file_sha256": config.gold_file_sha256,
        "output_directory": Path(output_dir).as_posix(),
        "checkpoint_after_each_provider_response": True,
        "resume_retries_only_missing_or_provider_failures_without_output": True,
        "source_only_inputs": True,
        "sensitive_or_restricted_fields_present": False,
        "provider_access_reuse_requires_approval": True,
        "paid_execution": True,
        "source_records_transmitted_on_execute": True,
        "oracle_or_gold_fields_in_prompts": False,
    }
    assert_no_secrets(record, "atomic dry run")
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=stdout)
    return AtomicDryRun(plan=plan, record=record)


def execute_atomic_pipeline(
    *, client: object, requested_model: str, output_dir: str | Path,
    repo_root: str | Path = ".", source_groups: Sequence[ExtractionSource] | None = None,
    now: Callable[[], datetime] | None = None, resume: bool = False,
    max_cost_usd: Decimal | str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, object]:
    """Run with a checkpoint after each response and resume only safe work."""

    root = Path(repo_root).resolve()
    plan = prepare_atomic_run(
        repo_root=root, source_groups=source_groups, validate_gold=False,
        config_path=config_path,
    )
    config = plan.config
    if requested_model != config.requested_model:
        raise AtomicPipelineError("requested model does not match the frozen snapshot")
    cap = _runtime_cost_cap(max_cost_usd, config)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root / output_path
    _require_output_directory(output_path, resume=resume)
    clock = now or (lambda: datetime.now(timezone.utc))
    repository_commit = _git_commit(root)
    repository_dirty = _git_dirty(root)

    if resume:
        checkpoint = _load_resume_checkpoint(output_path, plan, cap)
        started_at = checkpoint["started_at"]
        executions = list(checkpoint["executions"])
        prior_provider_failures = list(checkpoint["prior_provider_failures"])
        resume_count = checkpoint["resume_count"] + 1
    else:
        started_at = _utc_text(clock())
        executions = []
        prior_provider_failures = []
        resume_count = 0

    last_metadata = next(
        (item.provider_metadata for item in reversed(executions) if item.provider_metadata),
        None,
    )
    restore = getattr(client, "restore_pacing_metadata", None)
    if resume and last_metadata is not None and callable(restore):
        restore(last_metadata)

    output_path.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(
        output_path, plan, executions, prior_provider_failures, started_at,
        None, "running", cap, resume_count, repository_commit, repository_dirty,
    )

    completed_positions = {item.position for item in executions if item.passed}
    for position, ((case_id, source_id), source, prompt, prompt_upper) in enumerate(
        zip(plan.case_refs, plan.selected_sources, plan.prompts,
            plan.prompt_input_token_upper_bounds), 1
    ):
        if position in completed_positions:
            continue
        stop_reason = _request_limit_failure(
            plan, executions, prior_provider_failures, position, prompt_upper, cap
        )
        if stop_reason is not None:
            executions.append(AtomicCaseExecution(
                position, case_id, source_id, False, False, (), None,
                "safety_cap", stop_reason, 0, 0,
            ))
            return _write_checkpoint(
                output_path, plan, executions, prior_provider_failures, started_at,
                _utc_text(clock()), "stopped_safety_cap", cap, resume_count,
                repository_commit, repository_dirty,
            )

        pending = AtomicCaseExecution(
            position, case_id, source_id, True, False, (), None,
            "provider_pending",
            "No usable provider response has been checkpointed for this case.",
            prompt_upper, MAX_OUTPUT_TOKENS,
        )
        executions.append(pending)
        _write_checkpoint(
            output_path, plan, executions, prior_provider_failures, started_at,
            None, "running", cap, resume_count, repository_commit, repository_dirty,
        )
        try:
            raw_response, metadata = getattr(client, "complete_with_metadata")(
                system_prompt=ATOMIC_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        except Exception as error:
            stage = "provider_model_mismatch" if isinstance(error, OpenAIModelMismatchError) else "provider"
            message = (
                "The provider returned a different model snapshot."
                if stage == "provider_model_mismatch"
                else "The provider request failed."
            )
            executions.pop()
            execution = AtomicCaseExecution(
                position, case_id, source_id, True, False, (), None, stage, message,
                prompt_upper, MAX_OUTPUT_TOKENS,
            )
            executions.append(execution)
            status = "failed_model_mismatch" if stage != "provider" else "completed_with_provider_failures"
            return _write_checkpoint(
                output_path, plan, executions, prior_provider_failures, started_at,
                _utc_text(clock()), status, cap, resume_count,
                repository_commit, repository_dirty,
            )

        try:
            _validate_metadata(metadata, config)
        except AtomicPipelineError:
            executions.pop()
            safe_metadata = metadata if isinstance(metadata, OpenAIResponseMetadata) else None
            executions.append(AtomicCaseExecution(
                position, case_id, source_id, True, False, (), safe_metadata,
                "provider_model_mismatch",
                "The provider returned incompatible response metadata.",
                (
                    safe_metadata.input_tokens
                    if safe_metadata is not None and safe_metadata.input_tokens is not None
                    else prompt_upper
                ),
                (
                    safe_metadata.output_tokens
                    if safe_metadata is not None and safe_metadata.output_tokens is not None
                    else MAX_OUTPUT_TOKENS
                ),
            ))
            return _write_checkpoint(
                output_path, plan, executions, prior_provider_failures, started_at,
                _utc_text(clock()), "failed_model_mismatch", cap, resume_count,
                repository_commit, repository_dirty,
            )
        try:
            result = validate_atomic_response(source, raw_response, metadata)
        except AtomicExtractionValidationError:
            executions.pop()
            executions.append(AtomicCaseExecution(
                position, case_id, source_id, True, False, (), metadata,
                "validation", "The extracted claims failed validation.",
                metadata.input_tokens if metadata.input_tokens is not None else prompt_upper,
                metadata.output_tokens if metadata.output_tokens is not None else MAX_OUTPUT_TOKENS,
            ))
            return _write_checkpoint(
                output_path, plan, executions, prior_provider_failures, started_at,
                _utc_text(clock()), "failed_validation", cap, resume_count,
                repository_commit, repository_dirty,
            )
        executions.pop()
        executions.append(_successful_execution(position, case_id, result, prompt_upper))
        _write_checkpoint(
            output_path, plan, executions, prior_provider_failures, started_at,
            None, "running", cap, resume_count, repository_commit, repository_dirty,
        )

    if len([item for item in executions if item.passed]) != len(plan.case_refs):
        raise AtomicPipelineError("runtime predictions are incomplete")

    # Gold remains hash-only until every source-only prediction is complete.
    gold_path = root / ATOMIC_GOLD_PATH
    if _file_sha256(gold_path) != plan.gold_file_sha256:
        raise AtomicPipelineError("the atomic gold file changed before scoring")
    gold_cases = load_atomic_gold(gold_path, plan.all_sources)
    _require_frozen_case_order(gold_cases)
    predictions_by_case = {
        item.case_id: item.claims for item in executions if item.passed
    }
    scores = score_atomic_extraction(gold_cases, predictions_by_case)
    if _file_sha256(gold_path) != plan.gold_file_sha256:
        raise AtomicPipelineError("the atomic gold file changed during scoring")
    _write_jsonl(output_path / "case_scores.jsonl", scores["case_results"])
    _write_json(output_path / "scores.json", scores)
    return _write_checkpoint(
        output_path, plan, executions, prior_provider_failures, started_at,
        _utc_text(clock()), "completed", cap, resume_count,
        repository_commit, repository_dirty,
    )


def _successful_execution(
    position: int, case_id: str, result: AtomicExtractionResult, prompt_upper: int
) -> AtomicCaseExecution:
    metadata = result.response_metadata
    return AtomicCaseExecution(
        position, case_id, result.source_id, True, True, tuple(result.claims), metadata,
        None, None,
        metadata.input_tokens if metadata.input_tokens is not None else prompt_upper,
        metadata.output_tokens if metadata.output_tokens is not None else MAX_OUTPUT_TOKENS,
    )


def _request_limit_failure(
    plan: AtomicRunPlan, executions: Sequence[AtomicCaseExecution],
    prior_failures: Sequence[Mapping[str, object]], next_position: int,
    next_input: int, cap: Decimal,
) -> str | None:
    attempts = sum(item.attempted for item in executions) + len(prior_failures)
    attempted_positions = {
        item.position for item in executions if item.attempted
    } | {int(item["position"]) for item in prior_failures}
    retries = attempts - len(attempted_positions)
    used_input = sum(item.budget_input_tokens for item in executions) + sum(
        int(item["budget_input_tokens"]) for item in prior_failures
    )
    used_output = sum(item.budget_output_tokens for item in executions) + sum(
        int(item["budget_output_tokens"]) for item in prior_failures
    )
    if attempts + 1 > plan.config.maximum_request_attempts:
        return "The request-count ceiling was reached before this case."
    if (
        next_position in attempted_positions
        and retries + 1 > plan.config.maximum_retry_requests
    ):
        return "The retry allowance was reached before this case."
    if used_input + next_input > plan.config.maximum_input_tokens:
        return "The input-token ceiling was reached before this case."
    if used_output + MAX_OUTPUT_TOKENS > plan.config.maximum_output_tokens:
        return "The output-token ceiling was reached before this case."
    if token_cost(plan.config, used_input + next_input, used_output + MAX_OUTPUT_TOKENS) > cap:
        return "The approved cost cap was reached before this case."
    return None


def _write_checkpoint(
    output_path: Path, plan: AtomicRunPlan, executions: Sequence[AtomicCaseExecution],
    prior_provider_failures: Sequence[Mapping[str, object]], started_at: str,
    completed_at: str | None, status: str, cap: Decimal, resume_count: int,
    repository_commit: str, repository_dirty: bool,
) -> dict[str, object]:
    ordered = sorted(executions, key=lambda item: item.position)
    predictions = [_prediction_record(item) for item in ordered if item.passed]
    failures = [_failure_record(item) for item in ordered if not item.passed]
    _write_jsonl(output_path / "predictions.jsonl", predictions)
    _write_jsonl(output_path / "failures.jsonl", failures)
    output_names = ["predictions.jsonl", "failures.jsonl"]
    for name in ("scores.json", "case_scores.jsonl"):
        if (output_path / name).exists():
            output_names.append(name)
    provider_requests = sum(item.attempted for item in ordered) + len(prior_provider_failures)
    used_input = sum(item.budget_input_tokens for item in ordered) + sum(
        int(item["budget_input_tokens"]) for item in prior_provider_failures
    )
    used_output = sum(item.budget_output_tokens for item in ordered) + sum(
        int(item["budget_output_tokens"]) for item in prior_provider_failures
    )
    returned_models = sorted({
        item.provider_metadata.returned_model for item in ordered
        if item.provider_metadata is not None
    })
    run: dict[str, object] = {
        "run_format_version": plan.config.run_format_version,
        "run_status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "repository_commit": repository_commit,
        "repository_dirty": repository_dirty,
        "resume_count": resume_count,
        "configuration_version": plan.config.configuration_version,
        "configuration_sha256": plan.config.configuration_sha256,
        "provider": plan.config.provider,
        "dataset_version": plan.config.dataset_version,
        "dataset_split": plan.config.dataset_split,
        "runtime_dataset_sha256": plan.config.runtime_dataset_sha256,
        "provider": plan.config.provider,
        "requested_model": plan.config.requested_model,
        "resolved_model": returned_models[0] if len(returned_models) == 1 else None,
        "temperature": plan.config.temperature,
        "generation_settings": dict(plan.config.generation_settings),
        "prompt_version": plan.config.prompt_version,
        "prompt_sha256": plan.config.prompt_sha256,
        "scoring_version": ATOMIC_SCORING_VERSION,
        "predicate_registry_version": plan.config.predicate_registry_version,
        "predicate_registry_sha256": plan.config.predicate_registry_sha256,
        "case_order": [
            {"case_id": case_id, "source_id": source_id}
            for case_id, source_id in plan.case_refs
        ],
        "source_file_sha256": dict(plan.source_file_sha256),
        "gold_file_sha256": plan.gold_file_sha256,
        "planned_request_count": plan.config.planned_request_count,
        "maximum_request_attempts": plan.config.maximum_request_attempts,
        "maximum_retry_requests": plan.config.maximum_retry_requests,
        "provider_requests_attempted": provider_requests,
        "successful_cases": sum(item.passed for item in ordered),
        "failed_cases": sum(item.attempted and not item.passed for item in ordered),
        "remaining_cases": len(plan.case_refs) - sum(item.passed for item in ordered),
        "budget_input_tokens": used_input,
        "budget_output_tokens": used_output,
        "estimated_cost_usd": cost_text(token_cost(plan.config, used_input, used_output)),
        "hard_cost_cap_usd": cost_text(cap),
        "prior_provider_failures": list(prior_provider_failures),
        "attempts": [
            _attempt_record(item, plan.config.requested_model) for item in ordered
        ],
        "output_file_sha256": {
            name: _file_sha256(output_path / name) for name in output_names
        },
    }
    assert_no_secrets(run, "atomic run checkpoint")
    _write_json(output_path / "run.json", run)
    return run


def _load_resume_checkpoint(
    output_path: Path, plan: AtomicRunPlan, cap: Decimal
) -> dict[str, object]:
    try:
        record = json.loads((output_path / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AtomicPipelineError("the resume checkpoint is unreadable") from error
    if not isinstance(record, dict):
        raise AtomicPipelineError("the resume checkpoint must be an object")
    if set(record) != _RUN_RECORD_FIELDS:
        raise AtomicPipelineError("resume checkpoint fields changed")
    expected = {
        "run_format_version": plan.config.run_format_version,
        "configuration_version": plan.config.configuration_version,
        "configuration_sha256": plan.config.configuration_sha256,
        "dataset_version": plan.config.dataset_version,
        "dataset_split": plan.config.dataset_split,
        "runtime_dataset_sha256": plan.config.runtime_dataset_sha256,
        "requested_model": plan.config.requested_model,
        "temperature": plan.config.temperature,
        "generation_settings": dict(plan.config.generation_settings),
        "prompt_version": plan.config.prompt_version,
        "prompt_sha256": plan.config.prompt_sha256,
        "scoring_version": ATOMIC_SCORING_VERSION,
        "predicate_registry_version": plan.config.predicate_registry_version,
        "predicate_registry_sha256": plan.config.predicate_registry_sha256,
        "case_order": [{"case_id": a, "source_id": b} for a, b in plan.case_refs],
        "source_file_sha256": dict(plan.source_file_sha256),
        "gold_file_sha256": plan.gold_file_sha256,
        "planned_request_count": plan.config.planned_request_count,
        "maximum_request_attempts": plan.config.maximum_request_attempts,
        "maximum_retry_requests": plan.config.maximum_retry_requests,
        "hard_cost_cap_usd": cost_text(cap),
    }
    mismatches = [name for name, value in expected.items() if record.get(name) != value]
    if mismatches:
        raise AtomicPipelineError("resume checkpoint is incompatible: " + ", ".join(mismatches))
    if record.get("run_status") not in {"running", "completed_with_provider_failures"}:
        raise AtomicPipelineError("resume requires an incomplete provider-failed checkpoint")
    hashes = record.get("output_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        "predictions.jsonl", "failures.jsonl"
    }:
        raise AtomicPipelineError("resume checkpoint output hashes are malformed")
    for filename, expected_hash in hashes.items():
        if not isinstance(expected_hash, str) or _file_sha256(output_path / filename) != expected_hash:
            raise AtomicPipelineError("resume checkpoint output hash mismatch")
    attempts_raw = record.get("attempts")
    prior_raw = record.get("prior_provider_failures")
    if not isinstance(attempts_raw, list) or not isinstance(prior_raw, list):
        raise AtomicPipelineError("resume checkpoint attempt data is malformed")
    executions = [_execution_from_record(item, plan) for item in attempts_raw]
    positions = [item.position for item in executions]
    if positions != sorted(set(positions)):
        raise AtomicPipelineError("resume checkpoint cases are duplicated or reordered")
    failures = [item for item in executions if not item.passed]
    for item in failures:
        if (
            item.failure_stage not in {"provider", "provider_pending"}
            or item.provider_metadata is not None
        ):
            raise AtomicPipelineError("resume may retry only provider failures with no usable output")
    prior = [_validated_prior_failure(item, plan) for item in prior_raw]
    prior.extend(_failure_budget_record(item) for item in failures)
    saved_attempts = sum(item.attempted for item in executions) + len(prior_raw)
    saved_positions = {
        item.position for item in executions if item.attempted
    } | {int(item["position"]) for item in prior_raw}
    if saved_attempts - len(saved_positions) > plan.config.maximum_retry_requests:
        raise AtomicPipelineError("resume checkpoint exceeds the retry allowance")
    successful = [item for item in executions if item.passed]
    started_at = record.get("started_at")
    resume_count = record.get("resume_count")
    if not isinstance(started_at, str) or not started_at.endswith("Z"):
        raise AtomicPipelineError("resume checkpoint start time is invalid")
    if not isinstance(resume_count, int) or isinstance(resume_count, bool) or resume_count < 0:
        raise AtomicPipelineError("resume count is invalid")
    _validate_checkpoint_summary(record, executions, prior_raw, plan)
    return {
        "started_at": started_at, "resume_count": resume_count,
        "executions": tuple(successful), "prior_provider_failures": tuple(prior),
    }


def _validate_checkpoint_summary(
    record: Mapping[str, object],
    executions: Sequence[AtomicCaseExecution],
    prior_failures: Sequence[Mapping[str, object]],
    plan: AtomicRunPlan,
) -> None:
    status = record["run_status"]
    current_failures = [item for item in executions if not item.passed]
    if status == "running" and (
        len(current_failures) > 1
        or any(item.failure_stage != "provider_pending" for item in current_failures)
    ):
        raise AtomicPipelineError("running checkpoint contains an invalid pending request")
    if status == "completed_with_provider_failures" and len(current_failures) != 1:
        raise AtomicPipelineError("provider-failed checkpoint has invalid failure state")
    if status == "completed_with_provider_failures" and any(
        item.failure_stage != "provider" for item in current_failures
    ):
        raise AtomicPipelineError("provider-failed checkpoint has invalid failure stage")
    completed_at = record["completed_at"]
    if status == "running":
        if completed_at is not None:
            raise AtomicPipelineError("running checkpoint has a completion time")
    elif not isinstance(completed_at, str) or not completed_at.endswith("Z"):
        raise AtomicPipelineError("provider-failed checkpoint completion time is invalid")
    if not isinstance(record["repository_commit"], str) or not record["repository_commit"]:
        raise AtomicPipelineError("resume repository commit is invalid")
    if not isinstance(record["repository_dirty"], bool):
        raise AtomicPipelineError("resume worktree state is invalid")

    attempted = sum(item.attempted for item in executions) + len(prior_failures)
    successful = sum(item.passed for item in executions)
    failed = sum(item.attempted and not item.passed for item in executions)
    input_tokens = sum(item.budget_input_tokens for item in executions) + sum(
        int(item["budget_input_tokens"]) for item in prior_failures
    )
    output_tokens = sum(item.budget_output_tokens for item in executions) + sum(
        int(item["budget_output_tokens"]) for item in prior_failures
    )
    returned_models = sorted({
        item.provider_metadata.returned_model
        for item in executions
        if item.provider_metadata is not None
    })
    expected = {
        "provider_requests_attempted": attempted,
        "successful_cases": successful,
        "failed_cases": failed,
        "remaining_cases": len(plan.case_refs) - successful,
        "budget_input_tokens": input_tokens,
        "budget_output_tokens": output_tokens,
        "estimated_cost_usd": cost_text(
            token_cost(plan.config, input_tokens, output_tokens)
        ),
        "resolved_model": returned_models[0] if len(returned_models) == 1 else None,
    }
    mismatches = [name for name, value in expected.items() if record.get(name) != value]
    if mismatches:
        raise AtomicPipelineError(
            "resume checkpoint summary is inconsistent: " + ", ".join(mismatches)
        )


def _execution_from_record(item: object, plan: AtomicRunPlan) -> AtomicCaseExecution:
    if not isinstance(item, dict):
        raise AtomicPipelineError("resume attempt must be an object")
    required = {
        "position", "case_id", "source_id", "attempted", "passed", "claims",
        "provider_metadata", "failure_stage", "error", "budget_input_tokens",
        "budget_output_tokens",
    }
    if set(item) != required:
        raise AtomicPipelineError("resume attempt fields changed")
    position = item["position"]
    if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= len(plan.case_refs):
        raise AtomicPipelineError("resume attempt position is invalid")
    if (item["case_id"], item["source_id"]) != plan.case_refs[position - 1]:
        raise AtomicPipelineError("resume attempt does not match frozen case order")
    metadata = _metadata_from_record(
        item["provider_metadata"], plan.config.requested_model
    )
    if not all(isinstance(item[name], bool) for name in ("attempted", "passed")):
        raise AtomicPipelineError("resume attempt flags are invalid")
    if item["passed"] and not item["attempted"]:
        raise AtomicPipelineError("successful resume attempt was not marked attempted")
    if not isinstance(item["claims"], list):
        raise AtomicPipelineError("resume claims are malformed")
    if item["passed"]:
        source = plan.selected_sources[position - 1]
        synthetic = json.dumps({"claims": item["claims"]}, ensure_ascii=False)
        if metadata is None:
            raise AtomicPipelineError("successful resume attempt lacks provider metadata")
        try:
            claims = validate_atomic_response(source, synthetic, metadata).claims
        except AtomicExtractionValidationError as error:
            raise AtomicPipelineError("saved resume claims no longer validate") from error
        if item["failure_stage"] is not None or item["error"] is not None:
            raise AtomicPipelineError("successful resume attempt has failure data")
    else:
        claims = ()
        if item["claims"]:
            raise AtomicPipelineError("failed resume attempt contains claims")
        if item["failure_stage"] in {"provider", "provider_pending"} and not item["attempted"]:
            raise AtomicPipelineError("provider failure was not marked attempted")
    budgets = []
    for name in ("budget_input_tokens", "budget_output_tokens"):
        value = item[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AtomicPipelineError("resume token budget is invalid")
        budgets.append(value)
    return AtomicCaseExecution(
        position, item["case_id"], item["source_id"], item["attempted"], item["passed"],
        tuple(claims), metadata, item["failure_stage"], item["error"], budgets[0], budgets[1],
    )


def _attempt_record(
    item: AtomicCaseExecution, requested_model: str
) -> dict[str, object]:
    return {
        "position": item.position, "case_id": item.case_id, "source_id": item.source_id,
        "attempted": item.attempted, "passed": item.passed,
        "claims": [asdict(claim) for claim in item.claims],
        "provider_metadata": _metadata_record(item.provider_metadata, requested_model),
        "failure_stage": item.failure_stage, "error": item.error,
        "budget_input_tokens": item.budget_input_tokens,
        "budget_output_tokens": item.budget_output_tokens,
    }


def _prediction_record(item: AtomicCaseExecution) -> dict[str, object]:
    return {
        "case_id": item.case_id, "source_id": item.source_id,
        "claims": [asdict(claim) for claim in item.claims],
    }


def _failure_record(item: AtomicCaseExecution) -> dict[str, object]:
    return {
        "case_id": item.case_id, "source_id": item.source_id,
        "failure_stage": item.failure_stage, "error": item.error,
    }


def _failure_budget_record(item: AtomicCaseExecution) -> dict[str, object]:
    return {
        "position": item.position, "case_id": item.case_id, "source_id": item.source_id,
        "failure_stage": "provider", "error": "The provider request failed.",
        "budget_input_tokens": item.budget_input_tokens,
        "budget_output_tokens": item.budget_output_tokens,
    }


def _validated_prior_failure(item: object, plan: AtomicRunPlan) -> dict[str, object]:
    if not isinstance(item, dict) or set(item) != {
        "position", "case_id", "source_id", "failure_stage", "error",
        "budget_input_tokens", "budget_output_tokens",
    }:
        raise AtomicPipelineError("prior provider failure is malformed")
    position = item["position"]
    if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= len(plan.case_refs):
        raise AtomicPipelineError("prior provider failure position is invalid")
    if (item["case_id"], item["source_id"]) != plan.case_refs[position - 1]:
        raise AtomicPipelineError("prior provider failure order changed")
    if item["failure_stage"] != "provider" or item["error"] != "The provider request failed.":
        raise AtomicPipelineError("prior provider failure data changed")
    for name in ("budget_input_tokens", "budget_output_tokens"):
        if not isinstance(item[name], int) or isinstance(item[name], bool) or item[name] < 0:
            raise AtomicPipelineError("prior provider failure budget is invalid")
    return dict(item)


def _metadata_record(
    metadata: OpenAIResponseMetadata | None, requested_model: str
) -> dict[str, object] | None:
    if metadata is None:
        return None
    limits = metadata.rate_limits
    return {
        "response_id": metadata.response_id,
        "requested_model": requested_model,
        "returned_model": metadata.returned_model,
        "request_id": metadata.request_id,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "total_tokens": metadata.total_tokens,
        "rate_limits": asdict(limits) if limits is not None else None,
        "pacing_delay_seconds": metadata.pacing_delay_seconds,
    }


def _metadata_from_record(
    value: object, expected_requested_model: str
) -> OpenAIResponseMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "response_id", "requested_model", "returned_model", "request_id", "input_tokens",
        "output_tokens", "total_tokens", "rate_limits", "pacing_delay_seconds",
    }:
        raise AtomicPipelineError("resume provider metadata fields changed")
    if value["requested_model"] != expected_requested_model:
        raise AtomicPipelineError("resume requested-model metadata changed")
    limits_raw = value["rate_limits"]
    limits = None
    if limits_raw is not None:
        if not isinstance(limits_raw, dict) or set(limits_raw) != _RATE_LIMIT_FIELDS:
            raise AtomicPipelineError("resume rate-limit metadata is malformed")
        try:
            limits = OpenAIRateLimitMetadata(**limits_raw)
        except (TypeError, ValueError):
            raise AtomicPipelineError("resume rate-limit metadata is malformed") from None
    metadata = OpenAIResponseMetadata(
        response_id=value["response_id"], returned_model=value["returned_model"],
        input_tokens=value["input_tokens"], output_tokens=value["output_tokens"],
        total_tokens=value["total_tokens"], request_id=value["request_id"],
        rate_limits=limits, pacing_delay_seconds=value["pacing_delay_seconds"],
    )
    _validate_metadata_shape(metadata)
    return metadata


def _validate_metadata(metadata: object, config: AtomicRunConfig) -> None:
    if not isinstance(metadata, OpenAIResponseMetadata):
        raise AtomicPipelineError("provider metadata is missing or malformed")
    _validate_metadata_shape(metadata)
    if metadata.returned_model != config.resolved_model:
        raise AtomicPipelineError("provider metadata reports a different model snapshot")


def _validate_metadata_shape(metadata: OpenAIResponseMetadata) -> None:
    if not isinstance(metadata.response_id, str) or not metadata.response_id:
        raise AtomicPipelineError("provider response ID is missing")
    if not isinstance(metadata.returned_model, str) or not metadata.returned_model:
        raise AtomicPipelineError("returned model metadata is missing")
    if metadata.request_id is not None and (
        not isinstance(metadata.request_id, str) or not metadata.request_id
    ):
        raise AtomicPipelineError("provider request ID is invalid")
    for value in (metadata.input_tokens, metadata.output_tokens, metadata.total_tokens):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise AtomicPipelineError("provider token metadata is invalid")
    if (
        not isinstance(metadata.pacing_delay_seconds, (int, float))
        or isinstance(metadata.pacing_delay_seconds, bool)
        or not math.isfinite(metadata.pacing_delay_seconds)
        or metadata.pacing_delay_seconds < 0
    ):
        raise AtomicPipelineError("provider pacing metadata is invalid")
    if metadata.rate_limits is not None:
        for value in (
            metadata.rate_limits.remaining_requests,
            metadata.rate_limits.remaining_tokens,
            metadata.rate_limits.remaining_project_tokens,
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise AtomicPipelineError("provider rate-limit metadata is invalid")
        for value in (
            metadata.rate_limits.reset_requests_seconds,
            metadata.rate_limits.reset_tokens_seconds,
            metadata.rate_limits.reset_project_tokens_seconds,
        ):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise AtomicPipelineError("provider rate-limit metadata is invalid")


def _assert_source_only_prompts(prompts: Sequence[str]) -> None:
    forbidden = ("expected_claims", "gold_answer", "oracle_truth", "scorer_only")
    combined = "\n".join(prompts).lower()
    leaked = [field for field in forbidden if field in combined]
    if leaked:
        raise AtomicPipelineError("a scorer-only field appeared in a runtime prompt")


def _runtime_cost_cap(value: Decimal | str | None, config: AtomicRunConfig) -> Decimal:
    try:
        cap = config.hard_cost_cap_usd if value is None else Decimal(value)
    except (InvalidOperation, TypeError):
        raise AtomicPipelineError("the cost cap must be a decimal value") from None
    if cap <= 0 or cap > config.hard_cost_cap_usd:
        raise AtomicPipelineError("the cost cap must be positive and no higher than the frozen cap")
    return cap


def _require_output_directory(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_dir() or not (path / "run.json").is_file():
            raise AtomicPipelineError("no resume checkpoint exists")
        names = {item.name for item in path.iterdir()}
        if names != _CHECKPOINT_FILES:
            raise AtomicPipelineError("resume directory contains unexpected files")
    elif path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise AtomicPipelineError(f"refusing to overwrite non-empty output: {path}")


def _require_frozen_case_order(gold_cases: Sequence[AtomicGoldCase]) -> None:
    if tuple((case.case_id, case.source_id) for case in gold_cases) != ATOMIC_CASE_REFS:
        raise AtomicPipelineError("atomic gold case IDs or source order changed")


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        (PILOT_SOURCE_DIR / name).as_posix(): _file_sha256(repo_root / PILOT_SOURCE_DIR / name)
        for name in _SOURCE_FILENAMES
    }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise AtomicPipelineError(f"could not hash {path.name}") from error


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AtomicPipelineError("run timestamps must include a UTC offset")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    assert_no_secrets(list(records), str(path))
    content = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in records)
    _write_text(path, content + ("\n" if content else ""))


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    assert_no_secrets(record, str(path))
    _write_text(path, json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ten-case atomic-extraction pilot safely.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--cost-cap-usd")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Validate a provider-free plan or execute an explicitly approved run."""

    stdout = stdout or sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.resume:
            parser.error("--resume requires --execute")
        dry_run_atomic(stdout=stdout, config_path=args.config, output_dir=args.output_dir)
        return 0
    if not isinstance(args.model, str) or not args.model.strip():
        parser.error("--execute requires --model")
    try:
        config = load_atomic_run_config(".", args.config)
        prepare_atomic_run(config_path=args.config, validate_gold=False)
        if args.model != config.requested_model:
            raise AtomicPipelineError(
                "requested model does not match the frozen snapshot"
            )
        _runtime_cost_cap(args.cost_cap_usd, config)
        _require_output_directory(args.output_dir, resume=args.resume)
        api_key = load_env_value(args.env_file, "OPENAI_API_KEY")
        client = OpenAIResponsesClient(
            api_key=api_key, model=args.model, temperature=config.temperature,
            max_output_tokens=int(config.generation_settings["max_output_tokens"]),
        )
        run = execute_atomic_pipeline(
            client=client, requested_model=args.model, output_dir=args.output_dir,
            resume=args.resume, max_cost_usd=args.cost_cap_usd, config_path=args.config,
        )
    except Exception as error:
        message = str(error)
        try:
            assert_no_secrets(message, "error")
        except Exception:
            message = "The run failed; a sensitive-looking error was omitted."
        print(f"error={message}", file=sys.stderr)
        return 1
    print(f"run_status={run['run_status']}", file=stdout)
    print(f"provider_requests_attempted={run['provider_requests_attempted']}", file=stdout)
    print(f"successful_cases={run['successful_cases']}", file=stdout)
    print(f"failed_cases={run['failed_cases']}", file=stdout)
    print(f"output_dir={args.output_dir}", file=stdout)
    return 0 if run["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
