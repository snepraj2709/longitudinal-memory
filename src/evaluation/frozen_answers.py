"""Run and verify the 24 approved Step 10.3 B0-B7 answer batches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence, TextIO

from extraction.run_safety import count_step35_request_tokens, step35_request_sha256

from .frozen_answer_contracts import (
    FrozenAnswerFailure,
    FrozenAnswerPrediction,
    TASK_BODY,
    answer_failure_from_mapping,
    answer_prediction_from_mapping,
    validate_answer_output,
)
from .frozen_contexts import (
    BASELINES,
    MAX_MEMORY_TOKENS,
    TASKS,
    FrozenRuntime,
    build_context_records,
    context_evidence_index,
    load_frozen_runtime,
)
from .frozen_preflight import verify_frozen_preflight
from .frozen_run import DEFAULT_OUTPUT as EXTRACTION_OUTPUT, verify_extraction_batch
from .frozen_run_contracts import (
    FrozenRunError,
    ProviderRecord,
    canonical_json_bytes,
    money,
    parse_json_bytes,
    stable_sha256,
)
from .openai_client import (
    OpenAIModelMismatchError,
    OpenAIResponseError,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)


CONFIG_PATH = Path("configs/evaluation/frozen_answer_run_v1.json")
CONFIG_SHA256 = "449652ef79024070b433edffd2fb569fd5653bf37b6203ac7f01c7b20f0a6299"
PREFLIGHT_BATCHES = Path("results/evaluation/frozen-preflight-v1/batches.jsonl")
PREFLIGHT_ESTIMATES = Path("results/evaluation/frozen-preflight-v1/token-estimates.jsonl")
PROMPTS_PATH = Path("configs/evaluation/frozen_prompts_v1.json")
PROMPTS_SHA256 = "681935ec199e6f0a4f7cb37c971c0c28295293af9f792f1f41f6225ce5f468bf"
OUTPUT_ROOT = Path("results/evaluation/frozen-run-v1/batches")
MODEL = "gpt-4.1-2025-04-14"
MAX_OUTPUT_TOKENS = 1000
INPUT_RATE = Decimal("2.00")
OUTPUT_RATE = Decimal("8.00")
GLOBAL_CAP = Decimal("91.2131728")
EXTRACTION_COST = Decimal("0.2036120")
WORKERS = 4
CHECKPOINT_FILES = {"predictions.jsonl", "failures.jsonl", "checkpoint.json"}


def prepare_answer_batch(
    *, repo_root: str | Path = ".", baseline_id: str, task: str,
) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    if baseline_id not in BASELINES or task not in TASKS:
        raise FrozenRunError("unknown answer batch")
    if _sha(root / CONFIG_PATH) != CONFIG_SHA256:
        raise FrozenRunError("frozen answer run config changed")
    config = parse_json_bytes((root / CONFIG_PATH).read_bytes(), location="answer run config")
    _validate_config(config)
    verify_frozen_preflight(repo_root=root)
    verify_extraction_batch(repo_root=root, output_dir=root / EXTRACTION_OUTPUT)
    if _sha(root / PROMPTS_PATH) != PROMPTS_SHA256:
        raise FrozenRunError("frozen answer prompt contract changed")
    prompts = parse_json_bytes((root / PROMPTS_PATH).read_bytes(), location="answer prompts")
    prompt_rows = prompts.get("tasks")
    if not isinstance(prompt_rows, list):
        raise FrozenRunError("answer prompt task list changed")
    prompt = next((item for item in prompt_rows if isinstance(item, Mapping) and item.get("task") == task), None)
    if prompt is None:
        raise FrozenRunError("answer task prompt is missing")
    runtime = load_frozen_runtime(root)
    batches = _jsonl((root / PREFLIGHT_BATCHES).read_bytes(), "batch")
    batch = next((item for item in batches if item.get("baseline_id") == baseline_id and item.get("task") == task), None)
    if batch is None:
        raise FrozenRunError("preflight answer batch is missing")
    estimates = tuple(
        item for item in _jsonl((root / PREFLIGHT_ESTIMATES).read_bytes(), "estimate")
        if item.get("batch_id") == batch["batch_id"]
    )
    cases = runtime.cases[task]
    if len(estimates) != len(cases) or len(cases) != int(batch["request_count"]):
        raise FrozenRunError("answer batch case count changed")
    items = []
    for position, (case, estimate) in enumerate(zip(cases, estimates), 1):
        if (
            estimate.get("position") != position
            or estimate.get("record_id") != case.get("case_id")
            or estimate.get("user_id") != case.get("user_id")
        ):
            raise FrozenRunError("answer batch case order changed")
        records = list(build_context_records(
            runtime, baseline_id=baseline_id, task=task, case=case,
        ))
        system_prompt = str(prompt["system_instruction"])
        maximum_input = int(estimate["maximum_input_tokens"])
        user_prompt, request_tokens = _fit_prompt(
            task, case, records, system_prompt, maximum_input,
        )
        text_format = {"type": "json_object"}
        request_hash = step35_request_sha256(
            model=MODEL, system_prompt=system_prompt, user_prompt=user_prompt,
            text_format=text_format,
            generation_settings={"temperature": 0.0, "max_output_tokens": 1000, "store": False},
        )
        items.append({
            "position": position, "case": case, "case_id": case["case_id"],
            "user_id": case["user_id"], "split": case["split"],
            "context_records": tuple(records),
            "context_sha256": stable_sha256(records), "context_count": len(records),
            "evidence_index": context_evidence_index(records),
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "request_sha256": request_hash, "request_tokens": request_tokens,
            "maximum_input_tokens": maximum_input,
        })
    return {
        "root": root, "config": config, "runtime": runtime, "batch": batch,
        "baseline_id": baseline_id, "task": task, "items": tuple(items),
    }


def run_answer_batch(
    *,
    repo_root: str | Path = ".",
    baseline_id: str,
    task: str,
    output_dir: str | Path | None = None,
    env_file: str | Path = ".env",
    client_factory: Callable[[], object] | None = None,
    resume: bool = False,
    require_prior: bool = True,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Mapping[str, object]:
    """Execute one exact baseline/task batch with no retries."""

    plan = prepare_answer_batch(repo_root=repo_root, baseline_id=baseline_id, task=task)
    root = plan["root"]
    output = Path(output_dir) if output_dir is not None else root / str(plan["batch"]["output_directory"])
    if not output.is_absolute():
        output = root / output
    _require_output(output, resume=resume)
    prior_spend = _prior_spend(plan, require_prior=require_prior)
    now = clock or (lambda: datetime.now(timezone.utc))
    if resume:
        checkpoint, predictions, failures = _load_checkpoint(output, plan)
        if checkpoint["status"] == "completed":
            return checkpoint
        if checkpoint["pending_positions"]:
            raise FrozenRunError("resume is blocked after an ambiguous answer request")
        started_at = str(checkpoint["started_at"])
        resume_count = int(checkpoint["resume_count"]) + 1
    else:
        predictions, failures = [], []
        started_at, resume_count = _utc(now()), 0
    output.mkdir(parents=True, exist_ok=True)
    _write_state(output, plan, predictions, failures, (), "running", started_at, None, resume_count, prior_spend)

    if baseline_id == "B7":
        for item in plan["items"]:
            if int(item["position"]) <= len(predictions):
                continue
            output_record = _deterministic_abstention(task)
            predictions.append(_prediction(plan, item, output_record, None, "deterministic_answerability_gate"))
            _write_state(output, plan, predictions, failures, (), "running", started_at, None, resume_count, prior_spend)
        return _write_state(output, plan, predictions, failures, (), "completed", started_at, _utc(now()), resume_count, prior_spend)

    factory = client_factory
    if factory is None:
        key = load_env_value(root / env_file, "OPENAI_API_KEY")
        factory = lambda: OpenAIResponsesClient(
            api_key=key, model=MODEL, temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text_format={"type": "json_object"}, minimum_request_interval_seconds=0.0,
        )
    completed = {item.position for item in predictions}
    failed_positions = {item.position for item in failures}
    remaining = [
        item for item in plan["items"]
        if int(item["position"]) not in completed | failed_positions
    ]
    batch_cap = Decimal(str(plan["batch"]["maximum_cost_usd"]))
    failed = False
    for offset in range(0, len(remaining), WORKERS):
        if failed:
            break
        group = remaining[offset : offset + WORKERS]
        spent = _provider_cost(predictions)
        reserved = sum(
            (_cost(int(item["maximum_input_tokens"]), MAX_OUTPUT_TOKENS) for item in group),
            Decimal(0),
        )
        if spent + reserved > batch_cap or prior_spend + spent + reserved > GLOBAL_CAP:
            failures.append(_failure(plan, group[0], "cost_cap", "approved_cost_cap_reached", "request"))
            failed = True
            break
        pending = tuple(int(item["position"]) for item in group)
        _write_state(output, plan, predictions, failures, pending, "running", started_at, None, resume_count, prior_spend)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_items = {
                executor.submit(_provider_call, factory, item, monotonic): item for item in group
            }
            active = set(pending)
            for future in as_completed(future_items):
                item = future_items[future]
                active.remove(int(item["position"]))
                try:
                    raw, provider = future.result()
                    parsed = parse_json_bytes(raw.encode("utf-8"), location="provider answer")
                    normalized = validate_answer_output(
                        parsed, task=task, evidence_index=item["evidence_index"],
                    )
                    predictions.append(_prediction(plan, item, normalized, provider, "provider"))
                    predictions.sort(key=lambda value: value.position)
                except Exception as error:
                    stage = "model_mismatch" if isinstance(error, OpenAIModelMismatchError) else (
                        "validation" if isinstance(error, FrozenRunError) else "provider"
                    )
                    code = _safe_failure_code(error, stage)
                    failures.append(_failure(plan, item, stage, code, "response"))
                    failures.sort(key=lambda value: value.position)
                    failed = True
                _write_state(output, plan, predictions, failures, tuple(sorted(active)), "running", started_at, None, resume_count, prior_spend)
        if failed:
            break
    status = "completed" if len(predictions) == len(plan["items"]) and not failures else "failed"
    return _write_state(
        output, plan, predictions, failures, (), status, started_at, _utc(now()),
        resume_count, prior_spend,
    )


def verify_answer_batch(
    *, repo_root: str | Path = ".", baseline_id: str, task: str,
    output_dir: str | Path | None = None,
) -> Mapping[str, object]:
    plan = prepare_answer_batch(repo_root=repo_root, baseline_id=baseline_id, task=task)
    root = plan["root"]
    output = Path(output_dir) if output_dir is not None else root / str(plan["batch"]["output_directory"])
    if not output.is_absolute():
        output = root / output
    checkpoint, predictions, failures = _load_checkpoint(output, plan)
    if checkpoint["status"] != "completed" or failures or len(predictions) != len(plan["items"]):
        raise FrozenRunError("answer batch is incomplete")
    for prediction, item in zip(predictions, plan["items"]):
        if (
            prediction.position != item["position"]
            or prediction.case_id != item["case_id"]
            or prediction.context_sha256 != item["context_sha256"]
            or prediction.request_sha256 != item["request_sha256"]
        ):
            raise FrozenRunError("answer prediction identity changed")
        validate_answer_output(
            prediction.output, task=task, evidence_index=item["evidence_index"],
        )
    _validate_checkpoint(checkpoint, predictions, failures, plan, output)
    return checkpoint


def _fit_prompt(task, case, records, system_prompt, maximum_input):
    while True:
        user_prompt = _render_user_prompt(task, case, records)
        request_tokens = count_step35_request_tokens(
            model=MODEL, system_prompt=system_prompt, user_prompt=user_prompt,
            text_format={"type": "json_object"},
            generation_settings={"temperature": 0.0, "max_output_tokens": 1000, "store": False},
        )
        if request_tokens <= maximum_input:
            return user_prompt, request_tokens
        if not records:
            raise FrozenRunError("answer framing exceeds the frozen input ceiling")
        records.pop()


def _render_user_prompt(task, case, records) -> str:
    fields = {
        "qa": ("case_id", "user_id", "as_of", "question"),
        "summary": ("case_id", "user_id", "as_of", "instruction"),
        "interactive": ("case_id", "user_id", "as_of", "scenario", "initial_user_message"),
    }[task]
    body = TASK_BODY[task]
    payload = {
        "runtime_case": {name: case[name] for name in fields},
        "context_records": records,
        "output_contract": {
            "exact_fields": ["status", body, "confidence", "statements", "citations", "unresolved_parts", "abstention_reason"],
            "statuses": ["answered", "abstained", "disputed", "partially_answered"],
            "citation_fields": ["source_id", "message_id", "quote"],
            "types": f"{body}:string;confidence:number;statements:string[];citations:object[];unresolved_parts:string[];abstention_reason:string|null",
            "rules": "Citations must exactly match supplied evidence. Each statement must appear verbatim in the body. Abstained uses confidence 0, empty statements/citations/unresolved_parts, and a reason. Otherwise use grounded statements and null reason.",
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _provider_call(factory, item, monotonic):
    client = factory()
    began = monotonic()
    raw, metadata = getattr(client, "complete_with_metadata")(
        system_prompt=item["system_prompt"], user_prompt=item["user_prompt"],
    )
    latency = max(0, round((monotonic() - began) * 1000))
    if not isinstance(metadata, OpenAIResponseMetadata):
        raise FrozenRunError("answer provider metadata is missing")
    if metadata.returned_model != MODEL:
        raise OpenAIModelMismatchError("answer provider returned a different model")
    values = (metadata.input_tokens, metadata.output_tokens, metadata.total_tokens)
    if any(type(value) is not int or value < 0 for value in values):
        raise FrozenRunError("answer provider token usage is missing")
    provider = ProviderRecord(
        response_id=metadata.response_id, request_id=metadata.request_id,
        requested_model=MODEL, returned_model=metadata.returned_model,
        input_tokens=metadata.input_tokens, output_tokens=metadata.output_tokens,
        total_tokens=metadata.total_tokens,
        pacing_delay_seconds=str(metadata.pacing_delay_seconds), latency_ms=latency,
    )
    return raw, provider


def _prediction(plan, item, output, provider, mode):
    payload = {
        "batch_id": plan["batch"]["batch_id"], "position": item["position"],
        "baseline_id": plan["baseline_id"], "task": plan["task"],
        "case_id": item["case_id"], "user_id": item["user_id"], "split": item["split"],
        "context_sha256": item["context_sha256"], "context_count": item["context_count"],
        "request_sha256": item["request_sha256"], "generation_mode": mode,
        "output": output, "provider": None if provider is None else asdict(provider),
    }
    return FrozenAnswerPrediction(
        prediction_id=stable_sha256(payload),
        **{**payload, "provider": provider},
    )


def _deterministic_abstention(task):
    body = TASK_BODY[task]
    return {
        "status": "abstained",
        body: "I don't have enough reliable memory to answer that.",
        "confidence": 0,
        "statements": [], "citations": [], "unresolved_parts": [],
        "abstention_reason": "no_promoted_claims",
    }


def _failure(plan, item, stage, code, location):
    payload = {
        "batch_id": plan["batch"]["batch_id"], "position": item["position"],
        "baseline_id": plan["baseline_id"], "task": plan["task"],
        "case_id": item["case_id"], "user_id": item["user_id"],
        "stage": stage, "code": code, "location": location,
    }
    return FrozenAnswerFailure(failure_id=stable_sha256(payload), **payload)


def _prior_spend(plan, *, require_prior):
    batches = _jsonl((plan["root"] / PREFLIGHT_BATCHES).read_bytes(), "batch")
    current = int(plan["batch"]["position"])
    total = EXTRACTION_COST
    for batch in batches[1 : current - 1]:
        path = plan["root"] / str(batch["output_directory"]) / "checkpoint.json"
        if not path.is_file():
            if require_prior:
                raise FrozenRunError(f"prior answer batch is incomplete: {batch['batch_id']}")
            continue
        record = parse_json_bytes(path.read_bytes(), location="prior checkpoint")
        if record.get("status") != "completed" and require_prior:
            raise FrozenRunError(f"prior answer batch is incomplete: {batch['batch_id']}")
        total += Decimal(str(record.get("incremental_cost_usd")))
    return total


def _write_state(output, plan, predictions, failures, pending, status, started_at, completed_at, resume_count, prior_spend):
    predictions = sorted(predictions, key=lambda item: item.position)
    failures = sorted(failures, key=lambda item: item.position)
    prediction_bytes = b"".join(canonical_json_bytes(item) for item in predictions)
    failure_bytes = b"".join(canonical_json_bytes(item) for item in failures)
    _atomic_write(output / "predictions.jsonl", prediction_bytes)
    _atomic_write(output / "failures.jsonl", failure_bytes)
    provider_predictions = [item for item in predictions if item.provider is not None]
    batch_cost = _provider_cost(predictions)
    checkpoint = {
        "run_version": "frozen_answer_run_v1", "schema_version": "frozen_answer_run_schema_v1",
        "batch_id": plan["batch"]["batch_id"], "batch_position": plan["batch"]["position"],
        "baseline_id": plan["baseline_id"], "task": plan["task"], "status": status,
        "started_at": started_at, "completed_at": completed_at,
        "repository_commit": _git_commit(plan["root"]), "repository_dirty": _git_dirty(plan["root"]),
        "resume_count": resume_count, "planned_request_count": len(plan["items"]),
        "successful_count": len(predictions), "failure_count": len(failures),
        "remaining_count": len(plan["items"]) - len(predictions) - len(failures),
        "pending_positions": list(pending),
        "provider_request_count": len(provider_predictions) + len(failures) + len(pending),
        "deterministic_gate_count": sum(item.generation_mode == "deterministic_answerability_gate" for item in predictions),
        "input_token_count": sum(item.provider.input_tokens for item in provider_predictions),
        "output_token_count": sum(item.provider.output_tokens for item in provider_predictions),
        "incremental_cost_usd": money(batch_cost),
        "cumulative_incremental_cost_usd": money(prior_spend + batch_cost),
        "batch_cost_cap_usd": money(Decimal(str(plan["batch"]["maximum_cost_usd"]))),
        "global_cost_cap_usd": money(GLOBAL_CAP), "requested_model": MODEL,
        "returned_models": sorted({item.provider.returned_model for item in provider_predictions}),
        "worker_count": WORKERS, "maximum_retry_requests": 0,
        "gold_opened": False, "oracle_opened": False, "review_queue_opened": False,
        "raw_provider_output_persisted": False,
        "output_sha256": {
            "predictions.jsonl": hashlib.sha256(prediction_bytes).hexdigest(),
            "failures.jsonl": hashlib.sha256(failure_bytes).hexdigest(),
        },
    }
    _atomic_write(output / "checkpoint.json", canonical_json_bytes(checkpoint))
    return checkpoint


def _load_checkpoint(output, plan):
    if not output.is_dir() or {item.name for item in output.iterdir()} != CHECKPOINT_FILES:
        raise FrozenRunError("answer checkpoint file set changed")
    predictions = [answer_prediction_from_mapping(item) for item in _jsonl((output / "predictions.jsonl").read_bytes(), "prediction")]
    failures = [answer_failure_from_mapping(item) for item in _jsonl((output / "failures.jsonl").read_bytes(), "failure")]
    checkpoint = parse_json_bytes((output / "checkpoint.json").read_bytes(), location="checkpoint")
    _validate_checkpoint(checkpoint, predictions, failures, plan, output)
    return checkpoint, predictions, failures


def _validate_checkpoint(checkpoint, predictions, failures, plan, output):
    if checkpoint.get("batch_id") != plan["batch"]["batch_id"] or checkpoint.get("baseline_id") != plan["baseline_id"] or checkpoint.get("task") != plan["task"]:
        raise FrozenRunError("answer checkpoint identity changed")
    if checkpoint.get("successful_count") != len(predictions) or checkpoint.get("failure_count") != len(failures):
        raise FrozenRunError("answer checkpoint counts are inconsistent")
    expected_remaining = len(plan["items"]) - len(predictions) - len(failures)
    legacy_remaining = len(plan["items"]) - len(predictions)
    if checkpoint.get("remaining_count") not in {expected_remaining, legacy_remaining}:
        raise FrozenRunError("answer checkpoint remaining count is inconsistent")
    provider = [item for item in predictions if item.provider is not None]
    pending = checkpoint.get("pending_positions")
    if not isinstance(pending, list):
        raise FrozenRunError("answer checkpoint pending positions are invalid")
    if checkpoint.get("provider_request_count") != len(provider) + len(failures) + len(pending):
        raise FrozenRunError("answer checkpoint provider count is inconsistent")
    if checkpoint.get("input_token_count") != sum(item.provider.input_tokens for item in provider):
        raise FrozenRunError("answer checkpoint input count is inconsistent")
    if checkpoint.get("output_token_count") != sum(item.provider.output_tokens for item in provider):
        raise FrozenRunError("answer checkpoint output count is inconsistent")
    if checkpoint.get("incremental_cost_usd") != money(_provider_cost(predictions)):
        raise FrozenRunError("answer checkpoint cost is inconsistent")
    hashes = {
        "predictions.jsonl": _sha(output / "predictions.jsonl"),
        "failures.jsonl": _sha(output / "failures.jsonl"),
    }
    if checkpoint.get("output_sha256") != hashes:
        raise FrozenRunError("answer checkpoint output hashes changed")
    if any(checkpoint.get(name) for name in ("gold_opened", "oracle_opened", "review_queue_opened", "raw_provider_output_persisted")):
        raise FrozenRunError("answer runtime boundary changed")


def _validate_config(config):
    approval = config.get("approval")
    if not isinstance(approval, Mapping) or not all(
        approval.get(name) is True for name in (
            "credential_reuse_approved", "data_transmission_approved",
            "paid_execution_approved", "provider_execution_authorized",
            "reuse_existing_local_openai_api_key",
        )
    ):
        raise FrozenRunError("answer run approval is incomplete")
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("requested_model") != MODEL or model.get("max_output_tokens") != 1000:
        raise FrozenRunError("answer run model policy changed")
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or execution.get("maximum_retry_requests") != 0 or execution.get("worker_count") != WORKERS:
        raise FrozenRunError("answer execution policy changed")
    cost = config.get("cost")
    if not isinstance(cost, Mapping) or Decimal(str(cost.get("maximum_incremental_cost_usd"))) != GLOBAL_CAP:
        raise FrozenRunError("answer global cost cap changed")


def _safe_failure_code(error: Exception, stage: str) -> str:
    if stage == "model_mismatch":
        return "provider_model_mismatch"
    if stage == "provider":
        message = str(error)
        if "HTTP 400" in message:
            return "provider_http_400"
        if "HTTP 401" in message:
            return "provider_http_401"
        if "HTTP 403" in message:
            return "provider_http_403"
        if "HTTP 404" in message:
            return "provider_http_404"
        if "HTTP 429" in message:
            return "provider_http_429"
        if "timed out" in message.lower():
            return "provider_timeout"
        return "provider_request_failed"
    message = str(error)
    categories = (
        ("not valid JSON", "output_invalid_json"),
        ("fields changed", "output_fields_changed"),
        ("status or body", "output_status_or_body_invalid"),
        ("confidence", "output_confidence_invalid"),
        ("statements", "output_statements_invalid"),
        ("citations", "output_citations_invalid"),
        ("citation", "output_citation_invalid"),
        ("abstention", "output_abstention_invalid"),
        ("unresolved", "output_unresolved_parts_invalid"),
        ("untracked", "output_untracked_statement"),
    )
    return next((code for token, code in categories if token in message), "provider_output_invalid")


def _provider_cost(predictions):
    providers = [item.provider for item in predictions if item.provider is not None]
    return _cost(
        sum(item.input_tokens for item in providers),
        sum(item.output_tokens for item in providers),
    )


def _cost(input_tokens, output_tokens):
    return (Decimal(input_tokens) * INPUT_RATE + Decimal(output_tokens) * OUTPUT_RATE) / Decimal(1_000_000)


def _require_output(path, *, resume):
    if resume:
        if not path.is_dir():
            raise FrozenRunError("no answer checkpoint exists")
    elif path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FrozenRunError("refusing to overwrite a non-empty answer output")


def _jsonl(raw, location):
    return tuple(parse_json_bytes(line, location=f"{location}:{index}") for index, line in enumerate(raw.splitlines(), 1) if line)


def _atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise FrozenRunError("answer timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(root):
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    return result.returncode != 0 or bool(result.stdout.strip())


def _parser():
    parser = argparse.ArgumentParser(description="Run one approved frozen answer batch")
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = _parser().parse_args(argv)
    stream = stdout or sys.stdout
    if args.verify_only:
        result = verify_answer_batch(repo_root=args.repo_root, baseline_id=args.baseline, task=args.task)
    else:
        result = run_answer_batch(
            repo_root=args.repo_root, baseline_id=args.baseline, task=args.task,
            env_file=args.env_file, resume=args.resume,
        )
    print(json.dumps({
        "batch_id": result["batch_id"], "status": result["status"],
        "successful_count": result["successful_count"], "failure_count": result["failure_count"],
        "provider_request_count": result["provider_request_count"],
        "incremental_cost_usd": result["incremental_cost_usd"],
        "cumulative_incremental_cost_usd": result["cumulative_incremental_cost_usd"],
    }, sort_keys=True), file=stream)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
