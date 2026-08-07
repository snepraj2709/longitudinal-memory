"""Run all 25 pilot questions through the frozen B1 full-history baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_UP
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Protocol, Sequence, TextIO

from .history import (
    EvaluationQuestion,
    HistoryObservation,
    build_history_prompt,
    load_evaluation_questions,
    load_history_observations,
)
from .openai_client import (
    OpenAIModelMismatchError,
    OpenAIRateLimitMetadata,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)
from .prediction import (
    BaselinePrediction,
    PredictionValidationError,
    prediction_to_record,
    validate_prediction,
)
from .run_config import (
    FrozenBaselineConfig,
    RunConfigurationError,
    assert_no_secrets,
    inspect_completed_step3_run,
    load_frozen_config,
    load_run_manifest,
    verify_frozen_content,
    verify_manifest_artifacts,
)
from .smoke import EvidenceValidationError, validate_prediction_evidence


BASELINE_ID = "b1-full-history"
PILOT_CASE_IDS = tuple(
    f"{capability}_{index:03d}"
    for capability in (
        "extraction",
        "temporal",
        "conflict",
        "user_modeling",
        "abstention",
    )
    for index in range(1, 6)
)
DEFAULT_CONFIG_PATH = Path("configs/full_history_baseline_v1.json")
DEFAULT_STEP3_MANIFEST_PATH = Path(
    "results/pilot/full_history/"
    "20260807T083918Z-gpt-4.1-2025-04-14/manifest.json"
)
DEFAULT_QUESTIONS_PATH = Path("data/pilot/evaluation/eval_questions.jsonl")
DEFAULT_GOLD_PATH = Path("data/pilot/evaluation/eval_answer.jsonl")
DEFAULT_SOURCE_DIR = Path("data/pilot/sources")
DEFAULT_OUTPUT_DIR = Path("results/pilot/b1-full-history")
MAX_RUN_COST_USD = Decimal("0.56")
GPT_4_1_INPUT_USD_PER_MILLION = Decimal("2.00")
GPT_4_1_OUTPUT_USD_PER_MILLION = Decimal("8.00")
INITIAL_INPUT_TOKEN_ESTIMATE = 8_000


class B1RunError(RuntimeError):
    """Raised when the full B1 run cannot proceed safely."""


class B1QuestionError(ValueError):
    """Raised when the pilot question set is incomplete or reordered."""


class B1ModelClient(Protocol):
    """Provider seam used for sequential B1 execution."""

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        """Return one raw response and its non-secret provider metadata."""


@dataclass(frozen=True)
class B1Preflight:
    """Inputs proven safe before dry-run or paid execution."""

    config: FrozenBaselineConfig
    questions: tuple[EvaluationQuestion, ...]
    observations: tuple[HistoryObservation, ...]
    observation_counts: tuple[int, ...]


@dataclass(frozen=True)
class B1CaseExecution:
    """Validation state and raw diagnostics for one requested case."""

    case_id: str
    question_position: int
    attempted: bool
    raw_response: str | None
    provider_metadata: OpenAIResponseMetadata | None
    valid_json: bool
    valid_contract: bool
    case_id_matches: bool
    exact_evidence: bool
    prediction: BaselinePrediction | None
    failure_stage: str | None
    error: str | None

    @property
    def passed(self) -> bool:
        return (
            self.attempted
            and self.valid_json
            and self.valid_contract
            and self.case_id_matches
            and self.exact_evidence
            and self.prediction is not None
            and self.failure_stage is None
        )


@dataclass(frozen=True)
class B1Artifacts:
    """The four files written by one completed Step 5 execution."""

    run_path: Path
    predictions_path: Path
    scores_path: Path
    failures_path: Path
    status: str
    calls_attempted: int
    validated_predictions: int
    failures: int


def require_all_pilot_questions(
    questions: Sequence[EvaluationQuestion],
) -> tuple[EvaluationQuestion, ...]:
    """Require the frozen 25 case IDs once each and in file order."""

    case_ids = [question.case_id for question in questions]
    duplicates = sorted(
        case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
    )
    missing = sorted(set(PILOT_CASE_IDS) - set(case_ids))
    unknown = sorted(set(case_ids) - set(PILOT_CASE_IDS))
    errors: list[str] = []
    if len(case_ids) != len(PILOT_CASE_IDS):
        errors.append(
            f"pilot questions must contain exactly 25 cases, got {len(case_ids)}"
        )
    if duplicates:
        errors.append("duplicate case IDs: " + ", ".join(duplicates))
    if missing:
        errors.append("missing case IDs: " + ", ".join(missing))
    if unknown:
        errors.append("unknown case IDs: " + ", ".join(unknown))
    if not errors and tuple(case_ids) != PILOT_CASE_IDS:
        errors.append("pilot question order does not match the frozen file order")
    if errors:
        raise B1QuestionError("; ".join(errors))
    return tuple(questions)


def verify_step3_manual_reviews(report_path: str | Path) -> None:
    """Require a recorded manual pass for every Step 3 smoke case."""

    report_path = Path(report_path)
    try:
        lines = report_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise B1RunError(f"could not read Step 3 report {report_path}: {error}") from error
    smoke_case_ids = (
        "extraction_001",
        "temporal_003",
        "conflict_004",
        "user_modeling_001",
        "abstention_001",
    )
    errors: list[str] = []
    for case_id in smoke_case_ids:
        rows = [line for line in lines if line.startswith(f"| {case_id} |")]
        if len(rows) != 1:
            errors.append(f"Step 3 report must contain one row for {case_id}")
        elif "| yes | yes | yes | Pass" not in rows[0]:
            errors.append(f"Step 3 manual review did not pass for {case_id}")
    if errors:
        raise B1RunError("; ".join(errors))


def verify_completed_run_matches_config(
    config: FrozenBaselineConfig,
    *,
    provider: str,
    requested_model: str,
    returned_model: str,
    temperature: float,
    generation_settings: Mapping[str, object],
) -> None:
    """Reject any Step 3 provider or generation-setting mismatch."""

    comparisons = {
        "provider": (config.provider, provider),
        "requested model": (config.requested_model, requested_model),
        "returned model": (config.resolved_model, returned_model),
        "temperature": (config.temperature, temperature),
        "generation settings": (
            dict(config.generation_settings),
            dict(generation_settings),
        ),
    }
    mismatches = [
        name for name, (expected, actual) in comparisons.items() if expected != actual
    ]
    if mismatches:
        raise B1RunError(
            "Step 3 does not match the frozen configuration: "
            + ", ".join(mismatches)
        )


def ensure_output_directory_safe(
    output_dir: str | Path,
    frozen_configuration_hash: str,
    *,
    allow_compatible_resume: bool = False,
) -> None:
    """Refuse existing output except an explicitly resumed matching checkpoint."""

    output_dir = Path(output_dir)
    if not output_dir.exists():
        if allow_compatible_resume:
            raise B1RunError(f"no resume checkpoint exists in {output_dir}")
        return
    contents = list(output_dir.iterdir())
    if not contents:
        if allow_compatible_resume:
            raise B1RunError(f"no resume checkpoint exists in {output_dir}")
        return
    run_path = output_dir / "run.json"
    existing_hash: object = None
    if run_path.is_file():
        try:
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_record = None
        if isinstance(run_record, dict):
            existing_hash = run_record.get("frozen_configuration_hash")
            existing_status = run_record.get("run_status")
        else:
            existing_status = None
    else:
        existing_status = None
    if (
        allow_compatible_resume
        and existing_hash == frozen_configuration_hash
        and existing_status in {"running", "completed_with_failures"}
    ):
        allowed_names = {
            "run.json",
            "predictions.jsonl",
            "failures.jsonl",
            "scores.json",
        }
        unknown_names = sorted(path.name for path in contents if path.name not in allowed_names)
        if unknown_names:
            raise B1RunError(
                "resume checkpoint contains unexpected files: "
                + ", ".join(unknown_names)
            )
        return
    if existing_hash == frozen_configuration_hash:
        detail = "an existing run uses the same frozen configuration"
    elif existing_hash is None:
        detail = "existing artifacts have no verifiable frozen configuration hash"
    else:
        detail = "existing artifacts use a different frozen configuration hash"
    raise B1RunError(
        f"refusing to overwrite {output_dir}: {detail}"
    )


def preflight_b1_run(
    repo_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    step3_manifest_path: str | Path = DEFAULT_STEP3_MANIFEST_PATH,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
) -> B1Preflight:
    """Prove every Step 3, Step 4, input, and rerun prerequisite."""

    repo_root = Path(repo_root)
    config_path = _under_root(repo_root, config_path)
    manifest_path = _under_root(repo_root, step3_manifest_path)
    questions_path = _under_root(repo_root, questions_path)
    source_dir = _under_root(repo_root, source_dir)
    output_dir = _under_root(repo_root, output_dir)

    config = load_frozen_config(config_path)
    verify_frozen_content(config, repo_root)
    if config.baseline_id != "B1_full_history":
        raise B1RunError("frozen configuration is not the B1 full-history baseline")
    if config.provider != "OpenAI":
        raise B1RunError(f"unsupported frozen provider {config.provider!r}")
    if config.requested_model != config.resolved_model:
        raise B1RunError("frozen requested and resolved model snapshots must match")

    manifest = load_run_manifest(manifest_path, config=config)
    verify_manifest_artifacts(manifest, repo_root)
    completed = inspect_completed_step3_run(manifest_path.parent)
    verify_step3_manual_reviews(manifest_path.parent / "report.md")
    verify_completed_run_matches_config(
        config,
        provider=completed.provider,
        requested_model=completed.requested_model,
        returned_model=completed.returned_model,
        temperature=completed.temperature,
        generation_settings=completed.generation_settings,
    )

    questions = require_all_pilot_questions(load_evaluation_questions(questions_path))
    observations = load_history_observations(source_dir)
    observation_counts = tuple(
        build_history_prompt(question, observations).observation_count
        for question in questions
    )
    if len(observations) != 72 or set(observation_counts) != {72}:
        raise B1RunError(
            "the frozen pilot must provide 72 eligible observations to every question"
        )
    ensure_output_directory_safe(
        output_dir,
        config.configuration_sha256,
        allow_compatible_resume=resume,
    )
    return B1Preflight(
        config=config,
        questions=questions,
        observations=observations,
        observation_counts=observation_counts,
    )


def run_b1_questions(
    questions: Sequence[EvaluationQuestion],
    observations: tuple[HistoryObservation, ...],
    client: B1ModelClient,
    config: FrozenBaselineConfig,
    *,
    prior_executions: Sequence[B1CaseExecution] = (),
    on_case_completed: Callable[[tuple[B1CaseExecution, ...]], None] | None = None,
    max_cost_usd: Decimal = MAX_RUN_COST_USD,
) -> tuple[B1CaseExecution, ...]:
    """Run remaining questions sequentially without repairing or retrying."""

    ordered_questions = require_all_pilot_questions(questions)
    if not max_cost_usd.is_finite() or max_cost_usd <= 0:
        raise B1RunError("cost cap must be a positive finite amount")
    _validate_execution_set(prior_executions, ordered_questions)
    results = {
        execution.question_position: execution for execution in prior_executions
    }
    stop_error: str | None = None

    for position, question in enumerate(ordered_questions, start=1):
        if position in results:
            continue
        if stop_error is not None:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=False,
                    stage="not_attempted",
                    error=stop_error,
                ),
                on_case_completed,
            )
            continue

        prompt = build_history_prompt(question, observations)
        if not _cost_cap_allows_request(tuple(results.values()), config, max_cost_usd):
            stop_error = (
                f"stopped before exceeding the approved ${max_cost_usd} cost cap"
            )
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=False,
                    stage="cost_cap",
                    error=stop_error,
                ),
                on_case_completed,
            )
            continue
        raw_response: str | None = None
        metadata: OpenAIResponseMetadata | None = None
        try:
            raw_response, metadata = client.complete_with_metadata(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
            )
        except OpenAIModelMismatchError as error:
            stop_error = f"stopped after provider model mismatch: {error}"
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="provider_model_mismatch",
                    error=str(error),
                ),
                on_case_completed,
            )
            continue
        except Exception as error:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="provider",
                    error=str(error),
                ),
                on_case_completed,
            )
            continue

        if metadata.returned_model != config.resolved_model:
            stop_error = (
                "stopped after provider model mismatch: returned "
                f"{metadata.returned_model!r}, expected {config.resolved_model!r}"
            )
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="provider_model_mismatch",
                    error=stop_error,
                    raw_response=raw_response,
                    metadata=metadata,
                ),
                on_case_completed,
            )
            continue

        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as error:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="json",
                    error=str(error),
                    raw_response=raw_response,
                    metadata=metadata,
                ),
                on_case_completed,
            )
            continue

        try:
            prediction = validate_prediction(parsed)
        except PredictionValidationError as error:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="prediction_contract",
                    error=str(error),
                    raw_response=raw_response,
                    metadata=metadata,
                    valid_json=True,
                ),
                on_case_completed,
            )
            continue

        if prediction.case_id != question.case_id:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="case_id",
                    error=(
                        f"returned case_id {prediction.case_id!r} does not match "
                        f"requested case {question.case_id!r}"
                    ),
                    raw_response=raw_response,
                    metadata=metadata,
                    valid_json=True,
                    valid_contract=True,
                ),
                on_case_completed,
            )
            continue

        eligible_observations = tuple(
            observation
            for observation in observations
            if observation.observed_at <= question.as_of
        )
        try:
            validate_prediction_evidence(prediction, eligible_observations)
        except EvidenceValidationError as error:
            _record_execution(
                results,
                _failed_execution(
                    question,
                    position,
                    attempted=True,
                    stage="evidence",
                    error=str(error),
                    raw_response=raw_response,
                    metadata=metadata,
                    valid_json=True,
                    valid_contract=True,
                    case_id_matches=True,
                ),
                on_case_completed,
            )
            continue

        _record_execution(
            results,
            B1CaseExecution(
                case_id=question.case_id,
                question_position=position,
                attempted=True,
                raw_response=raw_response,
                provider_metadata=metadata,
                valid_json=True,
                valid_contract=True,
                case_id_matches=True,
                exact_evidence=True,
                prediction=prediction,
                failure_stage=None,
                error=None,
            ),
            on_case_completed,
        )
    return tuple(results[position] for position in sorted(results))


def _record_execution(
    results: dict[int, B1CaseExecution],
    execution: B1CaseExecution,
    callback: Callable[[tuple[B1CaseExecution, ...]], None] | None,
) -> None:
    if execution.question_position in results:
        raise B1RunError(f"case {execution.case_id} was recorded more than once")
    results[execution.question_position] = execution
    if callback is not None:
        callback(tuple(results[position] for position in sorted(results)))


def _validate_execution_set(
    executions: Sequence[B1CaseExecution],
    questions: Sequence[EvaluationQuestion],
) -> None:
    if len(executions) > len(questions):
        raise B1RunError("resume checkpoint has more attempts than pilot questions")
    positions: set[int] = set()
    for execution in executions:
        position = execution.question_position
        if position < 1 or position > len(questions) or position in positions:
            raise B1RunError("resume checkpoint has invalid question positions")
        positions.add(position)
        question = questions[position - 1]
        if execution.question_position != position or execution.case_id != question.case_id:
            raise B1RunError("resume checkpoint does not match pilot question order")


def _cost_cap_allows_request(
    executions: Sequence[B1CaseExecution],
    config: FrozenBaselineConfig,
    max_cost_usd: Decimal,
) -> bool:
    spent = _estimated_standard_cost(executions)
    observed_inputs = [
        item.provider_metadata.input_tokens
        for item in executions
        if item.provider_metadata is not None
        and item.provider_metadata.input_tokens is not None
    ]
    next_input_tokens = max(observed_inputs, default=INITIAL_INPUT_TOKEN_ESTIMATE)
    next_input_tokens = (next_input_tokens * 11 + 9) // 10
    max_output_tokens = config.generation_settings.get("max_output_tokens")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
        raise B1RunError("frozen max_output_tokens must be an integer")
    next_cost = _token_cost(next_input_tokens, max_output_tokens)
    return spent + next_cost <= max_cost_usd


def _estimated_standard_cost(
    executions: Sequence[B1CaseExecution],
) -> Decimal:
    input_tokens = sum(
        item.provider_metadata.input_tokens or 0
        for item in executions
        if item.provider_metadata is not None
    )
    output_tokens = sum(
        item.provider_metadata.output_tokens or 0
        for item in executions
        if item.provider_metadata is not None
    )
    return _token_cost(input_tokens, output_tokens)


def _token_cost(input_tokens: int, output_tokens: int) -> Decimal:
    million = Decimal(1_000_000)
    return (
        Decimal(input_tokens) * GPT_4_1_INPUT_USD_PER_MILLION / million
        + Decimal(output_tokens) * GPT_4_1_OUTPUT_USD_PER_MILLION / million
    )


def _cost_text(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_UP))


def execute_b1_pipeline(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    questions_path: str | Path,
    gold_path: str | Path,
    questions: Sequence[EvaluationQuestion],
    observations: tuple[HistoryObservation, ...],
    config: FrozenBaselineConfig,
    client: B1ModelClient,
    repository_commit: str,
    repository_dirty: bool = False,
    scorer: Callable[[str | Path, str | Path, Sequence[Mapping[str, object]]], dict[str, object]] | None = None,
    now: Callable[[], datetime] | None = None,
    resume: bool = False,
    max_cost_usd: Decimal = MAX_RUN_COST_USD,
) -> B1Artifacts:
    """Checkpoint generation, resume safely, then score the complete run."""

    repo_root = Path(repo_root)
    output_dir = _under_root(repo_root, output_dir)
    questions_path = _under_root(repo_root, questions_path)
    gold_path = _under_root(repo_root, gold_path)
    ensure_output_directory_safe(
        output_dir,
        config.configuration_sha256,
        allow_compatible_resume=resume,
    )
    now = now or (lambda: datetime.now(timezone.utc))
    gold_hash_before = _file_sha256(gold_path)
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.jsonl"
    scores_path = output_dir / "scores.json"
    run_path = output_dir / "run.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "run": _repo_relative(run_path, repo_root),
        "predictions": _repo_relative(predictions_path, repo_root),
        "scores": _repo_relative(scores_path, repo_root),
        "failures": _repo_relative(failures_path, repo_root),
    }

    if resume:
        checkpoint = _load_resume_checkpoint(
            run_path,
            questions=questions,
            observations=observations,
            config=config,
            gold_hash=gold_hash_before,
            max_cost_usd=max_cost_usd,
        )
        started_at = checkpoint["started_at"]
        prior_executions = checkpoint["executions"]
        prior_provider_failures = checkpoint["prior_provider_failures"]
        resume_count = checkpoint["resume_count"] + 1
        if scores_path.exists():
            scores_path.unlink()
    else:
        started_at = _utc_text(now())
        prior_executions = ()
        prior_provider_failures = ()
        resume_count = 0

    if prior_executions:
        restore_pacing = getattr(client, "restore_pacing_metadata", None)
        last_metadata = prior_executions[-1].provider_metadata
        if callable(restore_pacing) and last_metadata is not None:
            restore_pacing(last_metadata)

    def write_checkpoint(executions: tuple[B1CaseExecution, ...]) -> None:
        run_record = _build_run_record(
            executions=executions,
            config=config,
            repository_commit=repository_commit,
            repository_dirty=repository_dirty,
            artifact_paths=artifact_paths,
            started_at=started_at,
            completed_at=None,
            status="running",
            gold_hash_before=gold_hash_before,
            gold_hash_after=None,
            max_cost_usd=max_cost_usd,
            resume_count=resume_count,
            prior_provider_failures=prior_provider_failures,
        )
        _write_json(run_path, run_record)
        _write_execution_jsonl(predictions_path, failures_path, executions)

    write_checkpoint(tuple(prior_executions))
    executions = run_b1_questions(
        questions,
        observations,
        client,
        config,
        prior_executions=prior_executions,
        on_case_completed=write_checkpoint,
        max_cost_usd=max_cost_usd,
    )
    _write_execution_jsonl(predictions_path, failures_path, executions)
    attempt_records = [_attempt_record(item) for item in executions]

    if scorer is None:
        from .scoring import score_b1_results

        scorer = score_b1_results
    scores = scorer(questions_path, gold_path, attempt_records)
    assert_no_secrets(scores, "scores")
    _write_json(scores_path, scores)

    gold_hash_after = _file_sha256(gold_path)
    gold_unchanged = gold_hash_before == gold_hash_after
    completed_at = _utc_text(now())
    successful = sum(item.passed for item in executions)
    model_mismatch = any(
        item.failure_stage == "provider_model_mismatch" for item in executions
    )
    if model_mismatch:
        status = "failed_model_mismatch"
    elif any(item.failure_stage == "cost_cap" for item in executions):
        status = "stopped_cost_cap"
    elif successful == len(executions):
        status = "completed"
    else:
        status = "completed_with_failures"
    run_record = _build_run_record(
        executions=executions,
        config=config,
        repository_commit=repository_commit,
        repository_dirty=repository_dirty,
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        gold_hash_before=gold_hash_before,
        gold_hash_after=gold_hash_after,
        max_cost_usd=max_cost_usd,
        resume_count=resume_count,
        prior_provider_failures=prior_provider_failures,
    )
    assert_no_secrets(run_record, "run")
    _write_json(run_path, run_record)
    if not gold_unchanged:
        raise B1RunError("eval_answer.jsonl changed during the B1 run")
    return B1Artifacts(
        run_path=run_path,
        predictions_path=predictions_path,
        scores_path=scores_path,
        failures_path=failures_path,
        status=status,
        calls_attempted=sum(item.attempted for item in executions),
        validated_predictions=successful,
        failures=len(executions) - successful,
    )


def _build_run_record(
    *,
    executions: Sequence[B1CaseExecution],
    config: FrozenBaselineConfig,
    repository_commit: str,
    repository_dirty: bool,
    artifact_paths: Mapping[str, str],
    started_at: str,
    completed_at: str | None,
    status: str,
    gold_hash_before: str,
    gold_hash_after: str | None,
    max_cost_usd: Decimal,
    resume_count: int,
    prior_provider_failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    returned_models = sorted(
        {
            item.provider_metadata.returned_model
            for item in executions
            if item.provider_metadata is not None
        }
    )
    successful = sum(item.passed for item in executions)
    attempted = sum(item.attempted for item in executions)
    return {
        "baseline_id": BASELINE_ID,
        "run_status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "repository_commit": repository_commit,
        "repository_dirty": repository_dirty,
        "frozen_configuration_version": config.configuration_version,
        "frozen_configuration_hash": config.configuration_sha256,
        "provider": config.provider,
        "requested_model_version": config.requested_model,
        "resolved_model_version": (
            returned_models[0] if len(returned_models) == 1 else None
        ),
        "temperature": config.temperature,
        "generation_settings": dict(config.generation_settings),
        "prompt_version": config.prompt_version,
        "prompt_fingerprint": config.prompt_sha256,
        "dataset_hash": config.dataset_sha256,
        "source_ordering_rule": config.source_ordering_rule,
        "total_questions": len(PILOT_CASE_IDS),
        "recorded_cases": len(executions),
        "remaining_questions": len(PILOT_CASE_IDS) - len(executions),
        "calls_attempted": attempted,
        "provider_requests_attempted": attempted + len(prior_provider_failures),
        "successful_predictions": successful,
        "failed_predictions": (
            sum(item.attempted and not item.passed for item in executions)
            if status == "running"
            else len(executions) - successful
        ),
        "artifact_paths": dict(artifact_paths),
        "gold_answer_hash_before": gold_hash_before,
        "gold_answer_hash_after": gold_hash_after,
        "gold_answer_hash_unchanged": (
            gold_hash_after == gold_hash_before
            if gold_hash_after is not None
            else None
        ),
        "approved_cost_cap_usd": _cost_text(max_cost_usd),
        "estimated_standard_cost_usd": _cost_text(
            _estimated_standard_cost(executions)
        ),
        "resume_count": resume_count,
        "prior_provider_failures": list(prior_provider_failures),
        "provider_usage": _provider_usage(executions),
        "attempts": [_attempt_record(item) for item in executions],
    }


def _write_execution_jsonl(
    predictions_path: Path,
    failures_path: Path,
    executions: Sequence[B1CaseExecution],
) -> None:
    _write_jsonl(
        predictions_path,
        [
            prediction_to_record(item.prediction)
            for item in executions
            if item.passed and item.prediction is not None
        ],
    )
    _write_jsonl(
        failures_path,
        [_failure_record(item) for item in executions if not item.passed],
    )


def _load_resume_checkpoint(
    run_path: Path,
    *,
    questions: Sequence[EvaluationQuestion],
    observations: tuple[HistoryObservation, ...],
    config: FrozenBaselineConfig,
    gold_hash: str,
    max_cost_usd: Decimal,
) -> dict[str, object]:
    try:
        record = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B1RunError(f"could not read resume checkpoint {run_path}: {error}") from error
    if not isinstance(record, dict):
        raise B1RunError("resume checkpoint must be a JSON object")
    expected = {
        "baseline_id": BASELINE_ID,
        "frozen_configuration_version": config.configuration_version,
        "frozen_configuration_hash": config.configuration_sha256,
        "provider": config.provider,
        "requested_model_version": config.requested_model,
        "temperature": config.temperature,
        "generation_settings": dict(config.generation_settings),
        "prompt_version": config.prompt_version,
        "prompt_fingerprint": config.prompt_sha256,
        "dataset_hash": config.dataset_sha256,
        "source_ordering_rule": config.source_ordering_rule,
        "total_questions": len(PILOT_CASE_IDS),
        "gold_answer_hash_before": gold_hash,
        "approved_cost_cap_usd": _cost_text(max_cost_usd),
    }
    mismatches = [name for name, value in expected.items() if record.get(name) != value]
    if mismatches:
        raise B1RunError(
            "resume checkpoint is incompatible: " + ", ".join(mismatches)
        )
    checkpoint_status = record.get("run_status")
    if checkpoint_status not in {"running", "completed_with_failures"}:
        raise B1RunError("resume requires an in-progress or provider-failed run")
    started_at = record.get("started_at")
    if not isinstance(started_at, str) or not started_at.endswith("Z"):
        raise B1RunError("resume checkpoint has an invalid start time")
    resume_count = record.get("resume_count")
    if not isinstance(resume_count, int) or isinstance(resume_count, bool) or resume_count < 0:
        raise B1RunError("resume checkpoint has an invalid resume count")
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        raise B1RunError("resume checkpoint attempts must be a list")
    if len(attempts) > len(questions):
        raise B1RunError("resume checkpoint has too many attempts")
    executions_list: list[B1CaseExecution] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise B1RunError("resume checkpoint attempt must be an object")
        position = attempt.get("question_position")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 1
            or position > len(questions)
        ):
            raise B1RunError("resume checkpoint has an invalid question position")
        executions_list.append(
            _execution_from_attempt_record(
                attempt,
                question=questions[position - 1],
                observations=observations,
                position=position,
            )
        )
    executions = tuple(executions_list)
    _validate_execution_set(executions, questions)
    provider_failures = tuple(
        _failure_record(execution)
        for execution in executions
        if not execution.passed
        and execution.attempted
        and execution.failure_stage == "provider"
        and execution.raw_response is None
        and execution.provider_metadata is None
    )
    failed_executions = tuple(execution for execution in executions if not execution.passed)
    if failed_executions and len(provider_failures) != len(failed_executions):
        raise B1RunError(
            "resume may repeat only provider failures that returned no model answer"
        )
    prior_provider_failures_value = record.get("prior_provider_failures", [])
    if not isinstance(prior_provider_failures_value, list):
        raise B1RunError("resume checkpoint prior provider failures must be a list")
    prior_provider_failures = tuple(prior_provider_failures_value) + provider_failures
    executions = tuple(execution for execution in executions if execution.passed)
    if any(
        execution.provider_metadata is None
        or execution.provider_metadata.returned_model != config.resolved_model
        for execution in executions
    ):
        raise B1RunError("resume checkpoint model metadata does not match")
    return {
        "started_at": started_at,
        "executions": executions,
        "resume_count": resume_count,
        "prior_provider_failures": prior_provider_failures,
    }


def _execution_from_attempt_record(
    value: object,
    *,
    question: EvaluationQuestion,
    observations: tuple[HistoryObservation, ...],
    position: int,
) -> B1CaseExecution:
    if not isinstance(value, dict):
        raise B1RunError(f"resume attempt {position} must be an object")
    expected_fields = {
        "case_id",
        "question_position",
        "attempted",
        "passed",
        "valid_json",
        "valid_contract",
        "case_id_matches",
        "exact_evidence",
        "failure_stage",
        "error",
        "raw_response",
        "provider_attempt_metadata",
        "prediction",
    }
    if set(value) != expected_fields:
        raise B1RunError(f"resume attempt {position} fields changed")
    if value.get("case_id") != question.case_id or value.get("question_position") != position:
        raise B1RunError(f"resume attempt {position} does not match its question")
    boolean_fields = (
        "attempted",
        "passed",
        "valid_json",
        "valid_contract",
        "case_id_matches",
        "exact_evidence",
    )
    if any(not isinstance(value.get(name), bool) for name in boolean_fields):
        raise B1RunError(f"resume attempt {position} has invalid validation flags")
    raw_response = value.get("raw_response")
    if raw_response is not None and not isinstance(raw_response, str):
        raise B1RunError(f"resume attempt {position} has invalid raw response")
    failure_stage = value.get("failure_stage")
    error = value.get("error")
    if failure_stage is not None and not isinstance(failure_stage, str):
        raise B1RunError(f"resume attempt {position} has invalid failure stage")
    if error is not None and not isinstance(error, str):
        raise B1RunError(f"resume attempt {position} has invalid error")
    metadata = _metadata_from_record(value.get("provider_attempt_metadata"), position)
    prediction_value = value.get("prediction")
    prediction = None
    if prediction_value is not None:
        try:
            prediction = validate_prediction(prediction_value)
        except PredictionValidationError as validation_error:
            raise B1RunError(
                f"resume attempt {position} prediction is invalid: {validation_error}"
            ) from validation_error
        if prediction.case_id != question.case_id:
            raise B1RunError(f"resume attempt {position} prediction case ID changed")
        eligible = tuple(
            item for item in observations if item.observed_at <= question.as_of
        )
        try:
            validate_prediction_evidence(prediction, eligible)
        except EvidenceValidationError as validation_error:
            raise B1RunError(
                f"resume attempt {position} evidence is invalid: {validation_error}"
            ) from validation_error
    execution = B1CaseExecution(
        case_id=question.case_id,
        question_position=position,
        attempted=value["attempted"],
        raw_response=raw_response,
        provider_metadata=metadata,
        valid_json=value["valid_json"],
        valid_contract=value["valid_contract"],
        case_id_matches=value["case_id_matches"],
        exact_evidence=value["exact_evidence"],
        prediction=prediction,
        failure_stage=failure_stage,
        error=error,
    )
    if execution.passed != value["passed"]:
        raise B1RunError(f"resume attempt {position} pass state is inconsistent")
    return execution


def _metadata_from_record(value: object, position: int) -> OpenAIResponseMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise B1RunError(f"resume attempt {position} metadata must be an object")
    expected_fields = {
        "response_id",
        "returned_model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "request_id",
        "pacing_delay_seconds",
        "rate_limits",
    }
    if set(value) != expected_fields:
        raise B1RunError(f"resume attempt {position} metadata fields changed")
    response_id = value.get("response_id")
    returned_model = value.get("returned_model")
    if not isinstance(response_id, str) or not isinstance(returned_model, str):
        raise B1RunError(f"resume attempt {position} metadata is incomplete")
    rate_value = value.get("rate_limits")
    rate_limits = None
    if rate_value is not None:
        if not isinstance(rate_value, dict):
            raise B1RunError(f"resume attempt {position} rate limits must be an object")
        expected_rate_fields = {
            "remaining_requests",
            "remaining_tokens",
            "reset_requests_seconds",
            "reset_tokens_seconds",
            "remaining_project_tokens",
            "reset_project_tokens_seconds",
        }
        if set(rate_value) != expected_rate_fields:
            raise B1RunError(f"resume attempt {position} rate-limit fields changed")
        rate_limits = OpenAIRateLimitMetadata(
            remaining_requests=_nullable_nonnegative_integer(
                rate_value["remaining_requests"], position
            ),
            remaining_tokens=_nullable_nonnegative_integer(
                rate_value["remaining_tokens"], position
            ),
            reset_requests_seconds=_nullable_nonnegative_number(
                rate_value["reset_requests_seconds"], position
            ),
            reset_tokens_seconds=_nullable_nonnegative_number(
                rate_value["reset_tokens_seconds"], position
            ),
            remaining_project_tokens=_nullable_nonnegative_integer(
                rate_value["remaining_project_tokens"], position
            ),
            reset_project_tokens_seconds=_nullable_nonnegative_number(
                rate_value["reset_project_tokens_seconds"], position
            ),
        )
    pacing_delay = _nullable_nonnegative_number(
        value.get("pacing_delay_seconds", 0.0), position
    )
    request_id = value.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise B1RunError(f"resume attempt {position} has an invalid request ID")
    return OpenAIResponseMetadata(
        response_id=response_id,
        returned_model=returned_model,
        input_tokens=_nullable_nonnegative_integer(value.get("input_tokens"), position),
        output_tokens=_nullable_nonnegative_integer(value.get("output_tokens"), position),
        total_tokens=_nullable_nonnegative_integer(value.get("total_tokens"), position),
        request_id=request_id,
        rate_limits=rate_limits,
        pacing_delay_seconds=pacing_delay or 0.0,
    )


def _nullable_nonnegative_integer(value: object, position: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise B1RunError(f"resume attempt {position} has invalid token usage")
    return value


def _nullable_nonnegative_number(value: object, position: int) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise B1RunError(f"resume attempt {position} has invalid rate-limit timing")
    return float(value)


def _failed_execution(
    question: EvaluationQuestion,
    position: int,
    *,
    attempted: bool,
    stage: str,
    error: str,
    raw_response: str | None = None,
    metadata: OpenAIResponseMetadata | None = None,
    valid_json: bool = False,
    valid_contract: bool = False,
    case_id_matches: bool = False,
) -> B1CaseExecution:
    return B1CaseExecution(
        case_id=question.case_id,
        question_position=position,
        attempted=attempted,
        raw_response=raw_response,
        provider_metadata=metadata,
        valid_json=valid_json,
        valid_contract=valid_contract,
        case_id_matches=case_id_matches,
        exact_evidence=False,
        prediction=None,
        failure_stage=stage,
        error=error,
    )


def _attempt_record(execution: B1CaseExecution) -> dict[str, object]:
    return {
        "case_id": execution.case_id,
        "question_position": execution.question_position,
        "attempted": execution.attempted,
        "passed": execution.passed,
        "valid_json": execution.valid_json,
        "valid_contract": execution.valid_contract,
        "case_id_matches": execution.case_id_matches,
        "exact_evidence": execution.exact_evidence,
        "failure_stage": execution.failure_stage,
        "error": execution.error,
        "raw_response": execution.raw_response,
        "provider_attempt_metadata": _metadata_record(execution.provider_metadata),
        "prediction": (
            prediction_to_record(execution.prediction)
            if execution.prediction is not None
            else None
        ),
    }


def _failure_record(execution: B1CaseExecution) -> dict[str, object]:
    return {
        "case_id": execution.case_id,
        "question_position": execution.question_position,
        "failure_stage": execution.failure_stage,
        "error": execution.error,
        "raw_response": execution.raw_response,
        "provider_attempt_metadata": _metadata_record(execution.provider_metadata),
    }


def _metadata_record(
    metadata: OpenAIResponseMetadata | None,
) -> dict[str, object] | None:
    if metadata is None:
        return None
    rate_limits = metadata.rate_limits
    return {
        "response_id": metadata.response_id,
        "returned_model": metadata.returned_model,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "total_tokens": metadata.total_tokens,
        "request_id": metadata.request_id,
        "pacing_delay_seconds": metadata.pacing_delay_seconds,
        "rate_limits": (
            {
                "remaining_requests": rate_limits.remaining_requests,
                "remaining_tokens": rate_limits.remaining_tokens,
                "reset_requests_seconds": rate_limits.reset_requests_seconds,
                "reset_tokens_seconds": rate_limits.reset_tokens_seconds,
                "remaining_project_tokens": rate_limits.remaining_project_tokens,
                "reset_project_tokens_seconds": (
                    rate_limits.reset_project_tokens_seconds
                ),
            }
            if rate_limits is not None
            else None
        ),
    }


def _provider_usage(executions: Sequence[B1CaseExecution]) -> dict[str, object]:
    metadata = [
        item.provider_metadata
        for item in executions
        if item.provider_metadata is not None
    ]
    return {
        "responses_with_metadata": len(metadata),
        "input_tokens": sum(item.input_tokens or 0 for item in metadata),
        "output_tokens": sum(item.output_tokens or 0 for item in metadata),
        "total_tokens": sum(item.total_tokens or 0 for item in metadata),
    }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise B1RunError(f"could not hash {path}: {error}") from error


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise B1RunError("run timestamps must include a UTC offset")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise B1RunError("could not determine the repository commit")
    return commit


def _repository_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise B1RunError("could not determine the repository worktree state")
    return bool(result.stdout.strip())


def _under_root(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise B1RunError(f"artifact path is outside the repository: {path}") from error


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    assert_no_secrets(list(records), str(path))
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    _write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    assert_no_secrets(record, str(path))
    _write_text_atomic(
        path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 25-case B1 full-history pilot."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify inputs and print the run plan without calling a model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a matching in-progress checkpoint without repeating calls.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Run the fixed B1 pilot from the repository root."""

    stdout = stdout or sys.stdout
    args = _build_parser().parse_args(argv)
    repo_root = Path.cwd()
    try:
        preflight = preflight_b1_run(repo_root, resume=args.resume)
        max_output_tokens = preflight.config.generation_settings[
            "max_output_tokens"
        ]
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
            raise B1RunError("frozen max_output_tokens must be an integer")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "baseline_id": BASELINE_ID,
                        "provider": preflight.config.provider,
                        "model": preflight.config.resolved_model,
                        "maximum_calls": len(preflight.questions),
                        "question_case_ids": [
                            question.case_id for question in preflight.questions
                        ],
                        "observation_counts": list(preflight.observation_counts),
                        "frozen_configuration_hash": (
                            preflight.config.configuration_sha256
                        ),
                        "approved_cost_cap_usd": _cost_text(MAX_RUN_COST_USD),
                        "resume": args.resume,
                        "output_directory_safe": True,
                    },
                    separators=(",", ":"),
                ),
                file=stdout,
            )
            return 0

        api_key = load_env_value(_under_root(repo_root, args.env_file), "OPENAI_API_KEY")
        client = OpenAIResponsesClient(
            api_key=api_key,
            model=preflight.config.requested_model,
            temperature=preflight.config.temperature,
            max_output_tokens=max_output_tokens,
        )
        artifacts = execute_b1_pipeline(
            repo_root=repo_root,
            output_dir=DEFAULT_OUTPUT_DIR,
            questions_path=DEFAULT_QUESTIONS_PATH,
            gold_path=DEFAULT_GOLD_PATH,
            questions=preflight.questions,
            observations=preflight.observations,
            config=preflight.config,
            client=client,
            repository_commit=_repository_commit(repo_root),
            repository_dirty=_repository_dirty(repo_root),
            resume=args.resume,
            max_cost_usd=MAX_RUN_COST_USD,
        )
    except Exception as error:
        message = str(error)
        try:
            assert_no_secrets(message, "error")
        except RunConfigurationError:
            message = "operation failed; a sensitive-looking error was omitted"
        print(f"error={message}", file=sys.stderr)
        return 1

    print(f"run_status={artifacts.status}", file=stdout)
    print(f"calls_attempted={artifacts.calls_attempted}", file=stdout)
    print(f"validated_predictions={artifacts.validated_predictions}", file=stdout)
    print(f"failures={artifacts.failures}", file=stdout)
    print(f"run={artifacts.run_path}", file=stdout)
    print(f"predictions={artifacts.predictions_path}", file=stdout)
    print(f"scores={artifacts.scores_path}", file=stdout)
    print(f"failures_file={artifacts.failures_path}", file=stdout)
    return 0 if artifacts.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
