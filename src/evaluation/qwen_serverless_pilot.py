"""Local runner for the JarvisLabs vLLM Qwen compatibility pilot."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from .frozen_run_contracts import FrozenRunError
from .openai_client import OpenAIResponseMetadata
from .qwen_execution import (
    ClientFactory,
    ExecutionJob,
    QwenExecutionError,
    TransportRetryLedger,
    _canonical_bytes,
    _client_factory,
    _failure_record,
    _load_terminal_records,
    _sha,
    _write_exclusive_atomic,
    _write_or_verify,
)
from .qwen_pipeline import build_compatibility_jobs
from .vllm_client import VLLMModelMismatchError, VLLMResponseError, VLLMTransportError


CONFIG_PATH = Path("configs/evaluation/qwen_serverless_pilot.json")
PAID_SERVERLESS_CONFIRMATION = "qwen3-8b-vllm-pilot-paid-requests-approved"


class QwenServerlessPilotError(RuntimeError):
    """Raised when the vLLM pilot input is not safe to run."""


def run_serverless_pilot(
    *,
    repo_root: Path,
    base_url: str,
    model: str,
    api_key: str,
    deployment_id: str,
    gpu: str,
    output_root: Path | None = None,
    region: str = "IN2",
    framework: str = "vllm",
    storage_gb: int | None = None,
    dashboard_cost_inr: float | None = None,
    paid_serverless_confirmation: str,
    config_path: Path | None = None,
    retry_ledger: TransportRetryLedger | None = None,
    client_factory: ClientFactory | None = None,
    monotonic=time.monotonic,
) -> Mapping[str, object]:
    """Run the approved 12-request OpenAI-compatible vLLM pilot."""

    _require_paid_serverless_confirmation(paid_serverless_confirmation)
    root = repo_root.resolve()
    config = load_serverless_pilot_config(root, config_path or CONFIG_PATH)
    path = _select_path(config, model=model, gpu=gpu, region=region, framework=framework)
    output = _output_root(root, output_root, config)
    output.mkdir(parents=True, exist_ok=True)
    series_id = str(config["pilot_id"])
    jobs = tuple(
        ExecutionJob(**{**job.__dict__, "series_id": series_id})
        for job in build_compatibility_jobs(root)
    )
    expected = int(config["compatibility_request_count"])
    if len(jobs) != expected:
        raise QwenServerlessPilotError(f"compatibility job count changed: {len(jobs)}, expected {expected}")
    started = monotonic()
    normalized_base_url = _normalize_base_url(base_url)
    manifest = _execute_serverless_jobs(
        jobs,
        output_dir=output / "compatibility",
        client_factory=client_factory or _client_factory(
            base_url=normalized_base_url,
            model=model,
            api_key=api_key,
            temperature=float(config["temperature"]),
        ),
        retry_ledger=retry_ledger or TransportRetryLedger(25),
        retryable_response_statuses=tuple(config["retryable_http_status_codes"]),
        monotonic=monotonic,
    )
    receipt = {
        "schema_version": "qwen_vllm_pilot_receipt_v1",
        "status": manifest["status"],
        "series_id": series_id,
        "pilot_id": config["pilot_id"],
        "scored_replacement_for_full_v2": False,
        "provider": config["provider"],
        "deployment": {
            "deployment_id": deployment_id,
            "base_url": normalized_base_url,
            "model": model,
            "framework": framework,
            "gpu": gpu,
            "region": region,
            "path": path["name"],
            "concurrency": config["concurrency"],
            "context_length": config["context_length"],
            "temperature": config["temperature"],
            "storage_gb": storage_gb,
            "dashboard_cost_inr": dashboard_cost_inr,
            "delete_after_pilot": config["delete_after_pilot"],
            "deleted_after_pilot": False,
        },
        "request_scope": {
            "planned_request_count": expected,
            "actual_request_count": manifest["planned_request_count"],
            "provider_request_count": manifest["provider_request_count"],
            "successful_count": manifest["successful_count"],
            "failure_count": manifest["failure_count"],
            "retryable_http_status_codes": list(config["retryable_http_status_codes"]),
        },
        "elapsed_seconds": round(max(0.0, monotonic() - started), 6),
        "execution_manifest": manifest,
        "api_key_persisted": False,
        "scoring_status": "compatibility_only_not_scored",
    }
    _write_or_verify(
        output / "pilot-receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def _execute_serverless_jobs(
    jobs: Sequence[ExecutionJob],
    *,
    output_dir: Path,
    client_factory: ClientFactory,
    retry_ledger: TransportRetryLedger,
    retryable_response_statuses: Sequence[int],
    monotonic=time.monotonic,
) -> Mapping[str, object]:
    if not jobs:
        raise QwenServerlessPilotError("vLLM pilot batch cannot be empty")
    ordered = tuple(sorted(jobs, key=lambda item: item.position))
    if [item.position for item in ordered] != list(range(1, len(ordered) + 1)):
        raise QwenServerlessPilotError("vLLM pilot positions must be contiguous")
    if len({item.request_id for item in ordered}) != len(ordered):
        raise QwenServerlessPilotError("vLLM pilot request IDs must be unique")
    if len({item.series_id for item in ordered}) != 1:
        raise QwenServerlessPilotError("vLLM pilot cannot mix series IDs")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_dir = output_dir / "requests"
    request_dir.mkdir(exist_ok=True)
    prior = _load_terminal_records(request_dir, ordered)
    retryable = frozenset(int(status) for status in retryable_response_statuses)
    for job in ordered:
        if job.request_id in prior:
            continue
        _execute_serverless_one(job, request_dir, client_factory, retry_ledger, retryable, monotonic)
    records = _load_terminal_records(request_dir, ordered)
    if len(records) != len(ordered):
        raise QwenServerlessPilotError("vLLM pilot ended without every terminal record")
    rows = [records[item.request_id] for item in ordered]
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _write_or_verify(output_dir / "responses.jsonl", payload)
    failures = [row for row in rows if row["status"] == "failed"]
    failure_payload = b"".join(_canonical_bytes(row) + b"\n" for row in failures)
    _write_or_verify(output_dir / "failures.jsonl", failure_payload)
    successes = [row for row in rows if row["status"] == "succeeded"]
    manifest = {
        "schema_version": "qwen_vllm_compatibility_batch_v1",
        "series_id": ordered[0].series_id,
        "split": "development",
        "task": "mixed",
        "status": "completed" if not failures else "completed_with_failures",
        "planned_request_count": len(rows),
        "provider_request_count": sum(int(row["provider_attempt_count"]) for row in rows),
        "successful_count": len(successes),
        "failure_count": len(failures),
        "transport_retry_count": sum(int(row["transport_retry_count"]) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in successes),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in successes),
        "worker_count": 1,
        "retryable_http_status_codes": sorted(retryable),
        "raw_invalid_output_persisted": False,
        "gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "responses_sha256": sha256(payload).hexdigest(),
        "failures_sha256": sha256(failure_payload).hexdigest(),
    }
    _write_or_verify(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return manifest


def _execute_serverless_one(
    job: ExecutionJob,
    request_dir: Path,
    client_factory: ClientFactory,
    ledger: TransportRetryLedger,
    retryable_response_statuses: frozenset[int],
    monotonic,
) -> None:
    attempts = 0
    retries = 0
    began = monotonic()
    while True:
        attempts += 1
        try:
            raw, metadata = client_factory(job).complete_with_metadata(
                system_prompt=job.system_prompt,
                user_prompt=job.user_prompt,
            )
            if not isinstance(metadata, OpenAIResponseMetadata):
                raise QwenExecutionError("vLLM provider metadata is missing")
            output = job.validator(raw, metadata)
            record = _record_success(job, attempts, retries, monotonic() - began, output, metadata)
            break
        except VLLMTransportError:
            if retries == 0 and ledger.claim():
                retries = 1
                continue
            record = _failure_record(job, attempts, retries, monotonic() - began, "transport", "no_body_transport_failure")
            break
        except VLLMModelMismatchError:
            record = _failure_record(job, attempts, retries, monotonic() - began, "model_mismatch", "returned_model_mismatch")
            break
        except VLLMResponseError as error:
            code = f"http_{error.status_code}" if error.status_code is not None else "provider_response_error"
            if error.status_code in retryable_response_statuses and retries == 0 and ledger.claim():
                retries = 1
                continue
            stage = "transport" if error.status_code in retryable_response_statuses else "provider"
            record = _failure_record(job, attempts, retries, monotonic() - began, stage, code)
            break
        except (json.JSONDecodeError, ValueError, TypeError, FrozenRunError) as error:
            record = _failure_record(
                job,
                attempts,
                retries,
                monotonic() - began,
                "validation",
                _safe_validation_code(error),
            )
            break
        except Exception as error:
            stage = "validation" if not isinstance(error, QwenExecutionError) else "execution"
            record = _failure_record(job, attempts, retries, monotonic() - began, stage, type(error).__name__)
            break
    path = request_dir / f"{job.position:05d}-{sha256(job.request_id.encode()).hexdigest()[:16]}.json"
    _write_exclusive_atomic(path, json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _record_success(
    job: ExecutionJob,
    attempts: int,
    retries: int,
    seconds: float,
    output: Mapping[str, object],
    metadata: OpenAIResponseMetadata,
) -> Mapping[str, object]:
    return {
        "schema_version": "qwen_execution_record_v2",
        "series_id": job.series_id,
        "request_id": job.request_id,
        "request_sha256": job.request_sha256,
        "position": job.position,
        "split": job.split,
        "task": job.task,
        "record_id": job.record_id,
        "user_id": job.user_id,
        "baseline_id": job.baseline_id,
        "context_sha256": job.context_sha256,
        "context_count": job.context_count,
        "provider_attempt_count": attempts,
        "transport_retry_count": retries,
        "latency_ms": max(0, round(seconds * 1000)),
        "status": "succeeded",
        "generation_mode": "provider",
        "failure_stage": None,
        "failure_code": None,
        "output": output,
        "underlying_answer_sha256": _sha(output) if job.baseline_id == "B6" else None,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "response_id": metadata.response_id,
        "returned_model": metadata.returned_model,
    }


def load_serverless_pilot_config(repo_root: Path, config_path: Path = CONFIG_PATH) -> Mapping[str, object]:
    path = config_path if config_path.is_absolute() else repo_root / config_path
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "qwen_vllm_pilot_config_v1":
        raise QwenServerlessPilotError("unexpected vLLM pilot config schema")
    if config.get("compatibility_request_count") != 12:
        raise QwenServerlessPilotError("vLLM pilot must remain 12 compatibility requests")
    if config.get("concurrency") != 1:
        raise QwenServerlessPilotError("vLLM pilot concurrency must remain 1")
    if config.get("context_length") != 8192:
        raise QwenServerlessPilotError("vLLM pilot context length must remain 8192")
    if config.get("temperature") != 0:
        raise QwenServerlessPilotError("vLLM pilot temperature must remain 0")
    return config


def read_api_key(*, env_name: str, env_file: Path | None = None) -> str:
    values = _read_env_file(env_file) if env_file is not None else {}
    api_key = os.environ.get(env_name) or values.get(env_name)
    if not api_key:
        raise QwenServerlessPilotError(f"missing {env_name} in environment or env file")
    return api_key


def _select_path(
    config: Mapping[str, object],
    *,
    model: str,
    gpu: str,
    region: str,
    framework: str,
) -> Mapping[str, object]:
    if framework != config["framework"]:
        raise QwenServerlessPilotError("vLLM pilot framework must be vllm")
    for path in config["paths"]:
        if not isinstance(path, Mapping):
            continue
        if path.get("model") == model and path.get("region") == region and gpu in path.get("allowed_gpus", ()):
            return path
    raise QwenServerlessPilotError("model, GPU, and region do not match an approved vLLM path")


def _output_root(repo_root: Path, output_root: Path | None, config: Mapping[str, object]) -> Path:
    selected = output_root or Path(str(config["default_output_root"]))
    return selected if selected.is_absolute() else repo_root / selected


def _normalize_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    if not base_url:
        raise QwenServerlessPilotError("vLLM base URL must be non-empty")
    if base_url.endswith("/v1"):
        return base_url[:-3].rstrip("/")
    return base_url


def _require_paid_serverless_confirmation(value: str) -> None:
    if value != PAID_SERVERLESS_CONFIRMATION:
        raise QwenServerlessPilotError(
            f"paid vLLM requests require --confirm-paid-serverless {PAID_SERVERLESS_CONFIRMATION}"
        )


def _safe_validation_code(error: Exception) -> str:
    text = str(error).strip()
    if not text:
        return type(error).__name__
    safe = "".join(char.lower() if char.isalnum() else "_" for char in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:120] or type(error).__name__


def _read_env_file(path: Path | None) -> Mapping[str, str]:
    if path is None:
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise QwenServerlessPilotError("env file has an invalid line")
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip().strip("\"").strip("'")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--region", default="IN2")
    parser.add_argument("--framework", default="vllm")
    parser.add_argument("--storage-gb", type=int)
    parser.add_argument("--dashboard-cost-inr", type=float)
    parser.add_argument("--api-key-env", default="QWEN_VLLM_API_KEY")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--confirm-paid-serverless", required=True)
    args = parser.parse_args(argv)
    api_key = read_api_key(env_name=args.api_key_env, env_file=args.env_file)
    receipt = run_serverless_pilot(
        repo_root=args.repo_root,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        deployment_id=args.deployment_id,
        gpu=args.gpu,
        output_root=args.output_root,
        region=args.region,
        framework=args.framework,
        storage_gb=args.storage_gb,
        dashboard_cost_inr=args.dashboard_cost_inr,
        paid_serverless_confirmation=args.confirm_paid_serverless,
        config_path=args.config,
    )
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(_output_root(args.repo_root.resolve(), args.output_root, load_serverless_pilot_config(args.repo_root.resolve(), args.config)) / "pilot-receipt.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
