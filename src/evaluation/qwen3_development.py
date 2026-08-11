"""Run the Qwen3-8B development B0-B7 benchmark against a live vLLM endpoint."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from .qwen_benchmark import select_runtime
from .qwen_execution import (
    TransportRetryLedger,
    _client_factory,
    build_answer_jobs,
    build_extraction_jobs,
    derive_b7_records,
    execute_jobs,
    extraction_rows_from_records,
    write_derived_b7,
)
from .qwen_materialization import materialize_qwen_contexts, write_materialization
from .qwen_scoring import run_judge, score_sealed_release, seal_logical_predictions
from .qwen_serverless_pilot import _normalize_base_url, read_api_key


CONFIG_PATH = Path("configs/evaluation/qwen3_8b_vllm_development_v1.json")
RESULT_ROOT = Path("results/evaluation/qwen3-8b-vllm-dev-v1")
DATABASE_URL = "postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory"
PAID_RUN_CONFIRMATION = "qwen3-8b-vllm-development-paid-run-approved"
GOLD_PATHS = {
    "claims": Path("data/scaled-v1/gold/claims.jsonl"),
    "qa": Path("data/scaled-v1/gold/qa.jsonl"),
    "summary": Path("data/scaled-v1/gold/summaries.jsonl"),
    "interactive": Path("data/scaled-v1/gold/interactive.jsonl"),
}


class Qwen3DevelopmentError(RuntimeError):
    """Raised when the Qwen3 development run cannot continue safely."""


def run_development(
    *,
    repo_root: Path,
    output_root: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    database_url: str,
    hourly_rate_inr: float,
    setup_seconds: float,
    paid_run_confirmation: str,
    config_path: Path = CONFIG_PATH,
    resume: bool = False,
) -> Mapping[str, object]:
    """Execute development extraction, B0-B7 answers, judge diagnostics, and scores."""

    _require_confirmation(paid_run_confirmation)
    root = repo_root.resolve()
    config = load_config(root, config_path)
    series_id = str(config["series_id"])
    if model != config["model"]["model_alias"]:
        raise Qwen3DevelopmentError("served model alias does not match the Qwen3 config")
    output = output_root if output_root.is_absolute() else root / output_root
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to reuse checkpoints: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base = _normalize_base_url(base_url)
    retry_ledger = TransportRetryLedger(int(config["workload"]["maximum_transport_retries"]))
    factory = _client_factory(base_url=base, model=model, api_key=api_key, temperature=0.0)
    started = time.monotonic()

    extraction_dir = output / "extraction"
    extraction = execute_jobs(
        build_extraction_jobs(root, "development", series_id=series_id),
        output_dir=extraction_dir,
        client_factory=factory,
        retry_ledger=retry_ledger,
        workers=1,
        retryable_http_statuses=tuple(config["workload"]["retryable_http_statuses"]),
    )
    expected_extraction = int(config["workload"]["extraction_provider_requests"])
    if extraction["successful_count"] != expected_extraction:
        raise Qwen3DevelopmentError("development extraction did not fully validate")

    runtime = select_runtime(root, "development")
    contexts_dir = output / "contexts"
    if not (contexts_dir / "manifest.json").exists():
        materialized = _materialize(
            root=root,
            database_url=database_url,
            runtime=runtime,
            extraction_records=_jsonl(extraction_dir / "responses.jsonl"),
            contexts_dir=contexts_dir,
            series_id=series_id,
            config_path=config_path,
        )
    else:
        materialized = {"context_count": _line_count(contexts_dir / "contexts.jsonl"), "failure_count": 0}

    answers_dir = output / "answers-b0-b6"
    answers = execute_jobs(
        build_answer_jobs(
            root,
            "development",
            _load_contexts(contexts_dir / "contexts.jsonl"),
            series_id=series_id,
        ),
        output_dir=answers_dir,
        client_factory=factory,
        retry_ledger=retry_ledger,
        workers=1,
        retryable_http_statuses=tuple(config["workload"]["retryable_http_statuses"]),
    )

    b7_dir = output / "answers-b7"
    if not (b7_dir / "manifest.json").exists():
        answer_records = _jsonl(answers_dir / "responses.jsonl")
        b7 = write_derived_b7(
            b7_dir,
            derive_b7_records(_load_contexts(contexts_dir / "contexts.jsonl"), answer_records, series_id=series_id),
            series_id=series_id,
        )
    else:
        b7 = json.loads((b7_dir / "manifest.json").read_text(encoding="utf-8"))

    predictions_dir = output / "predictions"
    if not (predictions_dir / "manifest.json").exists():
        prediction_seal = seal_logical_predictions(
            answers_dir,
            b7_dir,
            predictions_dir,
            split="development",
            series_id=series_id,
        )
    else:
        prediction_seal = json.loads((predictions_dir / "manifest.json").read_text(encoding="utf-8"))

    references = lambda: _load_gold(root)
    judge_dir = output / "judge"
    judge = run_judge(
        repo_root=root,
        prediction_dir=predictions_dir,
        reference_loader=references,
        output_dir=judge_dir,
        base_url=base,
        api_key=api_key,
        retry_ledger=retry_ledger,
        series_id=series_id,
        model=model,
        temperature=0.0,
        workers=1,
        retryable_http_statuses=tuple(config["workload"]["retryable_http_statuses"]),
    )

    elapsed = time.monotonic() - started
    metadata = _execution_metadata(extraction, answers, judge, hourly_rate_inr, setup_seconds, elapsed)
    scores_dir = output / "scores"
    scorecard = score_sealed_release(
        repo_root=root,
        prediction_dir=predictions_dir,
        contexts_path=contexts_dir / "contexts.jsonl",
        extraction_dir=extraction_dir,
        reference_loader=references,
        execution_metadata=metadata,
        output_dir=scores_dir,
        series_id=series_id,
        config_path=config_path,
    )
    run_manifest = {
        "schema_version": "qwen3_development_run_v1",
        "series_id": series_id,
        "split": "development",
        "status": "completed" if scorecard["execution_failure_count"] == 0 else "completed_with_execution_failures",
        "provider": config["provider"],
        "model": config["model"],
        "runtime": {**config["runtime"], "base_url": base},
        "extraction": extraction,
        "materialization": materialized,
        "answers": answers,
        "b7": b7,
        "prediction_seal": prediction_seal,
        "judge": judge,
        "scorecard": scorecard,
        "execution_metadata": metadata,
        "config_sha256": _file_sha(root / config_path),
        "gold_opened_after_prediction_seal": True,
        "oracle_opened": False,
        "review_opened": False,
    }
    _write_json(output / "run-manifest.json", run_manifest)
    return run_manifest


def load_config(repo_root: Path, config_path: Path = CONFIG_PATH) -> Mapping[str, object]:
    config = json.loads((repo_root / config_path).read_text(encoding="utf-8"))
    if config.get("series_id") != "qwen3-8b-vllm-dev-v1":
        raise Qwen3DevelopmentError("unexpected Qwen3 development series id")
    if config.get("split") != "development":
        raise Qwen3DevelopmentError("Qwen3 development runner only supports development split")
    runtime = config.get("runtime")
    model = config.get("model")
    if not isinstance(runtime, Mapping) or not isinstance(model, Mapping):
        raise Qwen3DevelopmentError("Qwen3 development config is malformed")
    if runtime.get("client_concurrency") != 1 or runtime.get("temperature") != 0:
        raise Qwen3DevelopmentError("Qwen3 development run must use concurrency 1 and temperature 0")
    if model.get("model_alias") != "qwen3-8b-vllm" or model.get("context_length") != 8192:
        raise Qwen3DevelopmentError("Qwen3 model alias or context length changed")
    return config


def _materialize(
    *,
    root: Path,
    database_url: str,
    runtime: Mapping[str, Sequence[Mapping[str, object]]],
    extraction_records: Sequence[Mapping[str, object]],
    contexts_dir: Path,
    series_id: str,
    config_path: Path,
) -> Mapping[str, object]:
    try:
        import psycopg
    except ModuleNotFoundError as error:
        raise Qwen3DevelopmentError("psycopg is required for Qwen3 materialization") from error
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        connection.execute("CREATE SCHEMA public")
        result = materialize_qwen_contexts(
            connection,
            repo_root=root,
            split="development",
            runtime=runtime,
            extraction_rows=extraction_rows_from_records(extraction_records),
            series_id=series_id,
            config_path=config_path,
        )
    write_materialization(result, contexts_dir, series_id=series_id)
    return {"context_count": len(result.contexts), "failure_count": len(result.failures)}


def _execution_metadata(
    extraction: Mapping[str, object],
    answers: Mapping[str, object],
    judge: Mapping[str, object],
    hourly_rate_inr: float,
    setup_seconds: float,
    elapsed_seconds: float,
) -> Mapping[str, object]:
    input_tokens = sum(int(item.get("input_tokens", 0)) for item in (extraction, answers, judge))
    output_tokens = sum(int(item.get("output_tokens", 0)) for item in (extraction, answers, judge))
    provider_requests = sum(int(item.get("provider_request_count", 0)) for item in (extraction, answers, judge))
    retries = sum(int(item.get("transport_retry_count", 0)) for item in (extraction, answers, judge))
    cost = hourly_rate_inr * (setup_seconds + elapsed_seconds) / 3600
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_request_count": provider_requests,
        "transport_retry_count": retries,
        "gpu_cost_inr": round(cost, 4),
        "inference_wall_seconds": round(elapsed_seconds, 6),
        "output_tokens_per_second": output_tokens / elapsed_seconds if elapsed_seconds else None,
    }


def _load_gold(root: Path) -> Mapping[str, Sequence[Mapping[str, object]]]:
    return {name: _jsonl(root / path) for name, path in GOLD_PATHS.items()}


def _load_contexts(path: Path):
    from .qwen_execution import _load_contexts as load

    return load(path)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_confirmation(value: str) -> None:
    if value != PAID_RUN_CONFIRMATION:
        raise Qwen3DevelopmentError(
            f"pass --confirm-paid-run {PAID_RUN_CONFIRMATION!r} after capped Jarvis approval"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-8b-vllm")
    parser.add_argument("--api-key-env", default="QWEN_VLLM_API_KEY")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--database-url", default=os.environ.get("STORAGE_DATABASE_URL", DATABASE_URL))
    parser.add_argument("--hourly-rate-inr", type=float, required=True)
    parser.add_argument("--setup-seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-paid-run", required=True)
    args = parser.parse_args(argv)
    api_key = read_api_key(env_name=args.api_key_env, env_file=args.env_file)
    result = run_development(
        repo_root=args.repo_root,
        output_root=args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        database_url=args.database_url,
        hourly_rate_inr=args.hourly_rate_inr,
        setup_seconds=args.setup_seconds,
        paid_run_confirmation=args.confirm_paid_run,
        config_path=args.config,
        resume=args.resume,
    )
    print(json.dumps({
        "status": result["status"],
        "series_id": result["series_id"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
