"""Execute and verify the approved Step 10.3 frozen comparison runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence, TextIO

from extraction.atomic import (
    AtomicExtractionValidationError,
    sanitized_validation_diagnostics,
    validate_atomic_response,
)
from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.run_safety import count_step35_request_tokens, step35_request_sha256
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format

from .frozen_preflight import verify_frozen_preflight
from .frozen_run_contracts import (
    ExtractionFailure,
    ExtractionPrediction,
    FrozenRunError,
    ProviderRecord,
    canonical_json_bytes,
    failure_from_mapping,
    money,
    parse_json_bytes,
    prediction_from_mapping,
    prediction_payload,
    stable_sha256,
)
from .openai_client import (
    OpenAIModelMismatchError,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)


CONFIG_PATH = Path("configs/evaluation/frozen_run_v1.json")
CONFIG_SHA256 = "0868b1e891639eec6523a8d7369176eff69976adaa9ec6d82cd78a9e616df557"
PREFLIGHT_MANIFEST_PATH = Path("results/evaluation/frozen-preflight-v1/manifest.json")
PREFLIGHT_MANIFEST_SHA256 = "b5645e4189df260daa69292d15a95a641834b15b64eb19ce3a273b8c7e030ac5"
PREFLIGHT_BATCHES_PATH = Path("results/evaluation/frozen-preflight-v1/batches.jsonl")
PREFLIGHT_ESTIMATES_PATH = Path("results/evaluation/frozen-preflight-v1/token-estimates.jsonl")
PREFLIGHT_TRANSMISSIONS_PATH = Path("data/evaluation/frozen-preflight-v1/runtime/transmission-plan.jsonl")
DEFAULT_OUTPUT = Path("results/evaluation/frozen-run-v1/batches/batch_01_extraction_all_sources")
CHECKPOINT_FILES = {"predictions.jsonl", "failures.jsonl", "checkpoint.json"}
BATCH_ID = "batch_01_extraction_all_sources"
MODEL = "gpt-4.1-mini-2025-04-14"
MAX_OUTPUT_TOKENS = 1200
INPUT_RATE = Decimal("0.40")
OUTPUT_RATE = Decimal("1.60")
BATCH_CAP = Decimal("0.3927728")
GLOBAL_CAP = Decimal("91.2131728")
NORMALIZATION_VERSION = "source_span_boolean_polarity_v2"
USERS_PATH = Path("data/scaled-v1/runtime/users.jsonl")
USERS_SHA256 = "13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef"
SOURCES_PATH = Path("data/scaled-v1/runtime/sources.jsonl")
SOURCES_SHA256 = "a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de"
REGISTRY_PATH = Path("configs/extraction/predicate_registry_v2.json")
REGISTRY_SHA256 = "15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1"


def run_extraction_batch(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT,
    client: object | None = None,
    env_file: str | Path = ".env",
    resume: bool = False,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Mapping[str, object]:
    """Run exactly the approved 100-source extraction batch with safe checkpoints."""

    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    plan = _prepare_extraction(root)
    _require_output(output, resume=resume)
    now = clock or (lambda: datetime.now(timezone.utc))
    if resume:
        checkpoint, predictions = _load_checkpoint(output, plan)
        if checkpoint["status"] == "completed":
            return checkpoint
        if checkpoint["pending_position"] is not None or checkpoint["failure_count"]:
            raise FrozenRunError("resume is blocked after an ambiguous or failed provider attempt")
        started_at = str(checkpoint["started_at"])
        resume_count = int(checkpoint["resume_count"]) + 1
    else:
        predictions = []
        started_at = _utc(now())
        resume_count = 0
    output.mkdir(parents=True, exist_ok=True)
    _write_state(
        output, plan, predictions, [], status="running", started_at=started_at,
        completed_at=None, pending_position=None, resume_count=resume_count,
    )

    active_client = client
    if active_client is None:
        key = load_env_value(root / env_file, "OPENAI_API_KEY")
        active_client = OpenAIResponsesClient(
            api_key=key,
            model=MODEL,
            temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text_format=plan["text_format"],
            minimum_request_interval_seconds=float(plan["minimum_interval"]),
        )
    completed = {item.position for item in predictions}
    for item in plan["items"]:
        position = int(item["position"])
        if position in completed:
            continue
        spent = _prediction_cost(predictions)
        maximum_next = _cost(int(item["maximum_input_tokens"]), MAX_OUTPUT_TOKENS)
        if spent + maximum_next > BATCH_CAP or spent + maximum_next > GLOBAL_CAP:
            failure = _failure(item, "cost_cap", "approved_cost_cap_reached", "request")
            return _write_state(
                output, plan, predictions, [failure], status="stopped_cost_cap",
                started_at=started_at, completed_at=_utc(now()), pending_position=None,
                resume_count=resume_count,
            )
        _write_state(
            output, plan, predictions,
            [_failure(item, "provider_pending", "provider_response_not_checkpointed", "request")],
            status="running", started_at=started_at, completed_at=None,
            pending_position=position, resume_count=resume_count,
        )
        began = monotonic()
        try:
            raw, metadata = getattr(active_client, "complete_with_metadata")(
                system_prompt=plan["system_prompt"], user_prompt=item["prompt"],
            )
        except Exception as error:
            stage = "model_mismatch" if isinstance(error, OpenAIModelMismatchError) else "provider"
            code = "provider_model_mismatch" if stage == "model_mismatch" else "provider_request_failed"
            failure = _failure(item, stage, code, "request")
            return _write_state(
                output, plan, predictions, [failure], status=f"failed_{stage}",
                started_at=started_at, completed_at=_utc(now()), pending_position=None,
                resume_count=resume_count,
            )
        latency_ms = max(0, round((monotonic() - began) * 1000))
        provider = _provider_record(metadata, latency_ms)
        try:
            result = validate_atomic_response(
                item["source"], raw, metadata,
                evidence_normalization_version=NORMALIZATION_VERSION,
                registry=plan["registry"],
            )
        except AtomicExtractionValidationError as error:
            diagnostics = sanitized_validation_diagnostics(error)
            first = diagnostics[0] if diagnostics else None
            failure = _failure(
                item, "validation",
                first.code if first is not None else "response_validation_failed",
                first.location if first is not None else "response",
            )
            return _write_state(
                output, plan, predictions, [failure], status="failed_validation",
                started_at=started_at, completed_at=_utc(now()), pending_position=None,
                resume_count=resume_count,
            )
        payload = {
            "batch_id": BATCH_ID,
            "position": position,
            "source_id": item["source_id"],
            "user_id": item["user_id"],
            "split": item["split"],
            "source_sha256": item["source_sha256"],
            "request_sha256": item["request_sha256"],
            "claims": tuple(asdict(claim) for claim in result.claims),
            "normalization_diagnostics": tuple(asdict(value) for value in result.normalization_diagnostics),
            "provider": asdict(provider),
        }
        prediction = ExtractionPrediction(
            prediction_id=stable_sha256(payload),
            **{**payload, "provider": provider},
        )
        predictions.append(prediction)
        _write_state(
            output, plan, predictions, [], status="running", started_at=started_at,
            completed_at=None, pending_position=None, resume_count=resume_count,
        )
    return _write_state(
        output, plan, predictions, [], status="completed", started_at=started_at,
        completed_at=_utc(now()), pending_position=None, resume_count=resume_count,
    )


def verify_extraction_batch(
    *, repo_root: str | Path = ".", output_dir: str | Path = DEFAULT_OUTPUT,
) -> Mapping[str, object]:
    """Deeply verify checked extraction bytes without opening scorer-only data."""

    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    plan = _prepare_extraction(root)
    checkpoint, predictions = _load_checkpoint(output, plan)
    if checkpoint["status"] != "completed" or len(predictions) != 100:
        raise FrozenRunError("extraction batch is incomplete")
    if (output / "failures.jsonl").read_bytes() != b"":
        raise FrozenRunError("completed extraction batch contains failures")
    if checkpoint["pending_position"] is not None or checkpoint["remaining_count"] != 0:
        raise FrozenRunError("completed extraction checkpoint is inconsistent")
    expected_positions = list(range(1, 101))
    if [item.position for item in predictions] != expected_positions:
        raise FrozenRunError("extraction prediction order changed")
    for prediction, planned in zip(predictions, plan["items"]):
        if (
            prediction.source_id != planned["source_id"]
            or prediction.user_id != planned["user_id"]
            or prediction.split != planned["split"]
            or prediction.source_sha256 != planned["source_sha256"]
            or prediction.request_sha256 != planned["request_sha256"]
        ):
            raise FrozenRunError("extraction prediction identity changed")
        synthetic = json.dumps({"claims": list(prediction.claims)}, ensure_ascii=False)
        metadata = OpenAIResponseMetadata(
            response_id=prediction.provider.response_id,
            returned_model=prediction.provider.returned_model,
            input_tokens=prediction.provider.input_tokens,
            output_tokens=prediction.provider.output_tokens,
            total_tokens=prediction.provider.total_tokens,
            request_id=prediction.provider.request_id,
            pacing_delay_seconds=float(prediction.provider.pacing_delay_seconds),
        )
        validate_atomic_response(
            planned["source"], synthetic, metadata,
            registry=plan["registry"],
        )
    _validate_checkpoint(checkpoint, predictions, [], plan, output)
    return checkpoint


def _prepare_extraction(root: Path) -> Mapping[str, object]:
    if _sha(root / CONFIG_PATH) != CONFIG_SHA256:
        raise FrozenRunError("frozen run config changed")
    config = parse_json_bytes((root / CONFIG_PATH).read_bytes(), location="frozen run config")
    _validate_config(config)
    if _sha(root / PREFLIGHT_MANIFEST_PATH) != PREFLIGHT_MANIFEST_SHA256:
        raise FrozenRunError("Step 10.2 preflight manifest changed")
    verify_frozen_preflight(repo_root=root)
    raw_users = (root / USERS_PATH).read_bytes()
    raw_sources = (root / SOURCES_PATH).read_bytes()
    if hashlib.sha256(raw_users).hexdigest() != USERS_SHA256:
        raise FrozenRunError("scaled runtime users changed")
    if hashlib.sha256(raw_sources).hexdigest() != SOURCES_SHA256:
        raise FrozenRunError("scaled runtime sources changed")
    users = _jsonl(raw_users, "users")
    sources = _jsonl(raw_sources, "sources")
    if len(users) != 10 or len(sources) != 100:
        raise FrozenRunError("scaled extraction input counts changed")
    user_names = {str(item["user_id"]): str(item["display_name"]) for item in users}
    user_splits = {str(item["user_id"]): str(item["split"]) for item in users}
    if list(user_names) != [f"user_{index:03d}" for index in range(1, 11)]:
        raise FrozenRunError("scaled user order changed")
    if _sha(root / REGISTRY_PATH) != REGISTRY_SHA256:
        raise FrozenRunError("predicate registry changed")
    registry = load_predicate_registry(root / REGISTRY_PATH)
    system_prompt = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    text_format = atomic_extraction_text_format(registry)
    estimates = [
        item for item in _jsonl((root / PREFLIGHT_ESTIMATES_PATH).read_bytes(), "estimates")
        if item.get("batch_id") == BATCH_ID
    ]
    transmissions = [
        item for item in _jsonl((root / PREFLIGHT_TRANSMISSIONS_PATH).read_bytes(), "transmissions")
        if item.get("kind") == "extraction_source"
    ]
    batches = _jsonl((root / PREFLIGHT_BATCHES_PATH).read_bytes(), "batches")
    if len(estimates) != 100 or len(transmissions) != 100 or batches[0].get("batch_id") != BATCH_ID:
        raise FrozenRunError("Step 10.2 extraction plan changed")
    items = []
    for position, (record, estimate, transmission) in enumerate(
        zip(sources, estimates, transmissions), 1
    ):
        user_id = str(record["user_id"])
        source_id = str(record["source_id"])
        if (
            estimate.get("position") != position
            or estimate.get("record_id") != source_id
            or transmission.get("record_id") != source_id
            or estimate.get("user_id") != user_id
            or transmission.get("user_id") != user_id
        ):
            raise FrozenRunError("Step 10.2 extraction item order changed")
        source = _adapt_source(record, (user_id, source_id), user_names)
        prompt = build_atomic_extraction_prompt(source, include_speaker_name=False)
        exact = count_step35_request_tokens(
            model=MODEL, system_prompt=system_prompt, user_prompt=prompt,
            text_format=text_format,
            generation_settings={"temperature": 0.0, "max_output_tokens": 1200, "store": False},
        )
        if exact != estimate.get("exact_runtime_text_tokens"):
            raise FrozenRunError("Step 10.2 extraction token count changed")
        request_hash = step35_request_sha256(
            model=MODEL, system_prompt=system_prompt, user_prompt=prompt,
            text_format=text_format,
            generation_settings={"temperature": 0.0, "max_output_tokens": 1200, "store": False},
        )
        items.append({
            "position": position, "source_id": source_id, "user_id": user_id,
            "split": user_splits[user_id], "source": source, "prompt": prompt,
            "source_sha256": stable_sha256(record), "request_sha256": request_hash,
            "maximum_input_tokens": int(estimate["maximum_input_tokens"]),
        })
    return {
        "config": config, "registry": registry, "system_prompt": system_prompt,
        "text_format": text_format, "items": tuple(items),
        "minimum_interval": config["extraction"]["minimum_request_interval_seconds"],
    }


def _validate_config(config: Mapping[str, object]) -> None:
    expected = {
        "run_version", "schema_version", "starting_commit", "preflight_manifest_path",
        "preflight_manifest_sha256", "scaled_manifest_path", "scaled_manifest_sha256",
        "scaled_dataset_sha256", "predicate_registry_path", "predicate_registry_sha256",
        "approval", "transmission", "cost", "extraction", "runtime",
    }
    if set(config) != expected:
        raise FrozenRunError("frozen run config fields changed")
    approval = config.get("approval")
    if not isinstance(approval, Mapping) or not all(
        approval.get(name) is True for name in (
            "credential_reuse_approved", "data_transmission_approved",
            "paid_execution_approved", "provider_execution_authorized",
            "reuse_existing_local_openai_api_key",
        )
    ):
        raise FrozenRunError("provider approval is incomplete")
    extraction = config.get("extraction")
    if not isinstance(extraction, Mapping) or extraction.get("maximum_retry_requests") != 0:
        raise FrozenRunError("extraction retry policy changed")
    if extraction.get("model") != MODEL or extraction.get("max_output_tokens") != 1200:
        raise FrozenRunError("extraction model policy changed")
    cost = config.get("cost")
    if not isinstance(cost, Mapping) or Decimal(str(cost.get("maximum_incremental_cost_usd"))) != GLOBAL_CAP:
        raise FrozenRunError("global cost cap changed")


def _load_checkpoint(output: Path, plan: Mapping[str, object]):
    if not output.is_dir() or {item.name for item in output.iterdir()} != CHECKPOINT_FILES:
        raise FrozenRunError("extraction checkpoint file set changed")
    prediction_rows = _jsonl((output / "predictions.jsonl").read_bytes(), "prediction")
    failure_rows = _jsonl((output / "failures.jsonl").read_bytes(), "failure")
    predictions = [prediction_from_mapping(item) for item in prediction_rows]
    failures = [failure_from_mapping(item) for item in failure_rows]
    checkpoint = parse_json_bytes((output / "checkpoint.json").read_bytes(), location="checkpoint")
    _validate_checkpoint(checkpoint, predictions, failures, plan, output)
    return checkpoint, predictions


def _validate_checkpoint(checkpoint, predictions, failures, plan, output) -> None:
    expected_fields = {
        "run_version", "schema_version", "batch_id", "status", "started_at",
        "completed_at", "repository_commit", "repository_dirty", "resume_count",
        "planned_request_count", "successful_count", "failure_count", "remaining_count",
        "pending_position", "provider_request_count", "input_token_count",
        "output_token_count", "incremental_cost_usd", "batch_cost_cap_usd",
        "global_cost_cap_usd", "requested_model", "returned_models",
        "maximum_retry_requests", "approval", "gold_opened", "oracle_opened",
        "review_queue_opened", "raw_provider_output_persisted", "output_sha256",
    }
    if set(checkpoint) != expected_fields:
        raise FrozenRunError("checkpoint fields changed")
    if checkpoint["run_version"] != "frozen_run_v1" or checkpoint["batch_id"] != BATCH_ID:
        raise FrozenRunError("checkpoint identity changed")
    if checkpoint["planned_request_count"] != 100:
        raise FrozenRunError("checkpoint request count changed")
    if checkpoint["successful_count"] != len(predictions) or checkpoint["failure_count"] != len(failures):
        raise FrozenRunError("checkpoint result counts are inconsistent")
    if checkpoint["remaining_count"] != 100 - len(predictions):
        raise FrozenRunError("checkpoint remaining count is inconsistent")
    if checkpoint["provider_request_count"] != len(predictions) + len(failures):
        raise FrozenRunError("checkpoint provider count is inconsistent")
    if checkpoint["input_token_count"] != sum(item.provider.input_tokens for item in predictions):
        raise FrozenRunError("checkpoint input tokens are inconsistent")
    if checkpoint["output_token_count"] != sum(item.provider.output_tokens for item in predictions):
        raise FrozenRunError("checkpoint output tokens are inconsistent")
    if checkpoint["incremental_cost_usd"] != money(_prediction_cost(predictions)):
        raise FrozenRunError("checkpoint cost is inconsistent")
    expected_hashes = {
        "predictions.jsonl": _sha(output / "predictions.jsonl"),
        "failures.jsonl": _sha(output / "failures.jsonl"),
    }
    if checkpoint["output_sha256"] != expected_hashes:
        raise FrozenRunError("checkpoint output hashes changed")
    if any(checkpoint[name] for name in ("gold_opened", "oracle_opened", "review_queue_opened", "raw_provider_output_persisted")):
        raise FrozenRunError("runtime boundary changed")
    if [item.position for item in predictions] != list(range(1, len(predictions) + 1)):
        raise FrozenRunError("checkpoint prediction order changed")


def _write_state(
    output, plan, predictions, failures, *, status, started_at, completed_at,
    pending_position, resume_count,
):
    prediction_bytes = b"".join(canonical_json_bytes(item) for item in predictions)
    failure_bytes = b"".join(canonical_json_bytes(item) for item in failures)
    _atomic_write(output / "predictions.jsonl", prediction_bytes)
    _atomic_write(output / "failures.jsonl", failure_bytes)
    input_tokens = sum(item.provider.input_tokens for item in predictions)
    output_tokens = sum(item.provider.output_tokens for item in predictions)
    checkpoint = {
        "run_version": "frozen_run_v1", "schema_version": "frozen_run_schema_v1",
        "batch_id": BATCH_ID, "status": status, "started_at": started_at,
        "completed_at": completed_at, "repository_commit": _git_commit(Path.cwd()),
        "repository_dirty": _git_dirty(Path.cwd()), "resume_count": resume_count,
        "planned_request_count": 100, "successful_count": len(predictions),
        "failure_count": len(failures), "remaining_count": 100 - len(predictions),
        "pending_position": pending_position,
        "provider_request_count": len(predictions) + len(failures),
        "input_token_count": input_tokens, "output_token_count": output_tokens,
        "incremental_cost_usd": money(_cost(input_tokens, output_tokens)),
        "batch_cost_cap_usd": money(BATCH_CAP), "global_cost_cap_usd": money(GLOBAL_CAP),
        "requested_model": MODEL,
        "returned_models": sorted({item.provider.returned_model for item in predictions}),
        "maximum_retry_requests": 0,
        "approval": dict(plan["config"]["approval"]),
        "gold_opened": False, "oracle_opened": False, "review_queue_opened": False,
        "raw_provider_output_persisted": False,
        "output_sha256": {
            "predictions.jsonl": hashlib.sha256(prediction_bytes).hexdigest(),
            "failures.jsonl": hashlib.sha256(failure_bytes).hexdigest(),
        },
    }
    _atomic_write(output / "checkpoint.json", canonical_json_bytes(checkpoint))
    return checkpoint


def _provider_record(metadata: object, latency_ms: int) -> ProviderRecord:
    if not isinstance(metadata, OpenAIResponseMetadata):
        raise FrozenRunError("provider metadata is missing")
    if metadata.returned_model != MODEL:
        raise FrozenRunError("provider returned a different model snapshot")
    values = (metadata.input_tokens, metadata.output_tokens, metadata.total_tokens)
    if any(type(value) is not int or value < 0 for value in values):
        raise FrozenRunError("provider token usage is missing")
    return ProviderRecord(
        response_id=metadata.response_id, request_id=metadata.request_id,
        requested_model=MODEL, returned_model=metadata.returned_model,
        input_tokens=metadata.input_tokens, output_tokens=metadata.output_tokens,
        total_tokens=metadata.total_tokens,
        pacing_delay_seconds=str(metadata.pacing_delay_seconds), latency_ms=latency_ms,
    )


def _failure(item, stage: str, code: str, location: str) -> ExtractionFailure:
    payload = {
        "batch_id": BATCH_ID, "position": int(item["position"]),
        "source_id": str(item["source_id"]), "user_id": str(item["user_id"]),
        "stage": stage, "code": code, "location": location,
    }
    return ExtractionFailure(failure_id=stable_sha256(payload), **payload)


def _prediction_cost(predictions: Sequence[ExtractionPrediction]) -> Decimal:
    return _cost(
        sum(item.provider.input_tokens for item in predictions),
        sum(item.provider.output_tokens for item in predictions),
    )


def _cost(input_tokens: int, output_tokens: int) -> Decimal:
    return (Decimal(input_tokens) * INPUT_RATE + Decimal(output_tokens) * OUTPUT_RATE) / Decimal(1_000_000)


def _require_output(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_dir():
            raise FrozenRunError("no extraction checkpoint exists")
    elif path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FrozenRunError("refusing to overwrite a non-empty extraction output")


def _jsonl(raw: bytes, location: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        parse_json_bytes(line, location=f"{location}:{index}")
        for index, line in enumerate(raw.splitlines(), 1) if line
    )


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FrozenRunError("runtime timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(root: Path) -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    return result.returncode != 0 or bool(result.stdout.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the approved frozen extraction batch")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stream = stdout or sys.stdout
    if args.verify_only:
        result = verify_extraction_batch(repo_root=args.repo_root, output_dir=args.output_dir)
    else:
        result = run_extraction_batch(
            repo_root=args.repo_root, output_dir=args.output_dir,
            env_file=args.env_file, resume=args.resume,
        )
    print(json.dumps({
        "status": result["status"], "successful_count": result["successful_count"],
        "failure_count": result["failure_count"],
        "provider_request_count": result["provider_request_count"],
        "incremental_cost_usd": result["incremental_cost_usd"],
    }, sort_keys=True), file=stream)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
