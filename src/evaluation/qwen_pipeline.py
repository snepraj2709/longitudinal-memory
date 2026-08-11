"""One-command bounded Qwen v2 execution across Jarvis, PostgreSQL, and scoring."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .frozen_answer_contracts import validate_answer_output
from .qwen_benchmark import PROMPTS, REGISTRY, _render_answer_prompt, select_runtime
from .qwen_compatibility import _answer_schema, _response_format
from .qwen_execution import (
    ExecutionJob,
    TransportRetryLedger,
    _client_factory,
    _evidence_index,
    build_answer_jobs,
    build_extraction_jobs,
    derive_b7_records,
    execute_jobs,
    extraction_rows_from_records,
    write_derived_b7,
)
from .qwen_lifecycle import (
    BudgetWatchdog,
    JarvisManager,
    LiveBudgetController,
    STAGE_LIMITS,
    load_local_secrets,
    stage_one_gate,
    stage_two_gate,
)
from .qwen_materialization import materialize_qwen_contexts, write_materialization
from .qwen_preflight import VLLMTokenizer
from .qwen_scoring import (
    _judge_schema,
    _validate_judge_output,
    run_judge,
    score_sealed_release,
    seal_logical_predictions,
)
from .qwen_v2_contract import SERIES_ID, load_qwen_v2_config, verify_series_manifest


RESULT_ROOT = Path("results/evaluation/qwen35-27b-fp8-v2")
DATABASE_URL = "postgresql://storage_test:storage_test@127.0.0.1:55432/longitudinal_memory"
SERVER_SCRIPT = Path("scripts/run_qwen_vllm.sh")
SECRET_FILE = Path(".qwen-secrets.env")
PAID_EXECUTION_CONFIRMATION = "qwen35-27b-fp8-v2-paid-gpu-approved"
GOLD_PATHS = {
    "claims": Path("data/scaled-v1/gold/claims.jsonl"),
    "qa": Path("data/scaled-v1/gold/qa.jsonl"),
    "summary": Path("data/scaled-v1/gold/summaries.jsonl"),
    "interactive": Path("data/scaled-v1/gold/interactive.jsonl"),
}


class QwenPipelineError(RuntimeError):
    """Raised when a bounded stage cannot continue under the frozen contract."""


def build_compatibility_jobs(repo_root: Path) -> tuple[ExecutionJob, ...]:
    """Build three extraction, B0, B6-context, and judge compatibility calls."""

    root = repo_root.resolve()
    selected = select_runtime(root, "development")
    jobs = [replace(job, position=index) for index, job in enumerate(
        build_extraction_jobs(root, "development")[:3], 1
    )]
    prompt_rows = json.loads((root / PROMPTS).read_text(encoding="utf-8"))["tasks"]
    prompts = {str(row["task"]): str(row["system_instruction"]) for row in prompt_rows}
    position = 3
    b6_outputs = []
    for baseline in ("B0", "B6"):
        for task in ("qa", "summary", "interactive"):
            position += 1
            case = selected[task][0]
            records = () if baseline == "B0" else _compatibility_context(selected, case)
            evidence = _evidence_index(records)

            def validate(raw: str, _metadata, *, current_task=task, index=evidence):
                return validate_answer_output(json.loads(raw), task=current_task, evidence_index=index)

            job = ExecutionJob(
                request_id=f"compat:{baseline}:{task}",
                position=position,
                split="development",
                task=task,
                record_id=str(case["case_id"]),
                user_id=str(case["user_id"]),
                baseline_id=baseline,
                context_sha256=None,
                context_count=len(records),
                system_prompt=prompts[task],
                user_prompt=_render_compatibility_answer_prompt(task, case, records),
                response_format=_compatibility_answer_response_format(task, baseline),
                max_output_tokens=1000,
                validator=validate,
            )
            jobs.append(job)
            if baseline == "B6":
                b6_outputs.append((task, case, records))
    for task, case, records in b6_outputs:
        position += 1
        candidate_id = f"compat_{task}"
        candidate_ids = (candidate_id,)

        def validate_judge(raw: str, _metadata, *, ids=candidate_ids):
            return _validate_judge_output(json.loads(raw), ids)

        prompt = {
            "task": task,
            "candidates": [{
                "candidate_id": candidate_id,
                "task_input": _case_input(task, case),
                "reference": {"compatibility_fixture": "Check structure and grounding only."},
                "response": _compatibility_candidate(task, records),
            }],
            "rules": "Return one structured diagnostic label for this non-scored fixture.",
            "identity_policy": "Candidate and system identities are intentionally hidden.",
        }
        jobs.append(ExecutionJob(
            request_id=f"compat:judge:{task}",
            position=position,
            split="development",
            task="judge",
            record_id=f"compat_judge_{task}",
            user_id=str(case["user_id"]),
            baseline_id=None,
            context_sha256=None,
            context_count=1,
            system_prompt="Evaluate the supplied compatibility fixture and return exact JSON.",
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            response_format=_response_format(f"compat_judge_{task}", _judge_schema(candidate_ids)),
            max_output_tokens=500,
            validator=validate_judge,
        ))
    if len(jobs) != 12 or position != 12:
        raise QwenPipelineError("compatibility job composition changed")
    return tuple(jobs)


def run_pipeline(
    *,
    repo_root: Path,
    secret_file: Path,
    output_root: Path,
    paid_execution_confirmation: str,
) -> Mapping[str, object]:
    """Execute approved stages and guarantee Jarvis cleanup before returning."""

    _require_paid_execution_confirmation(paid_execution_confirmation)
    root = repo_root.resolve()
    output = output_root if output_root.is_absolute() else root / output_root
    _local_preflight(root, output)
    prior_gpu_cost = _historical_gpu_cost(output)
    attempt_id = _next_attempt_id(output)
    attempt_root = output / "attempts" / attempt_id
    secrets = load_local_secrets(secret_file if secret_file.is_absolute() else root / secret_file)
    _docker(root, "up")
    retry_ledger = TransportRetryLedger(25)
    manager = JarvisManager(artifact_dir=output / "lifecycle" / attempt_id)
    summary: dict[str, object] = {
        "schema_version": "qwen_pipeline_run_v2",
        "series_id": SERIES_ID,
        "attempt_id": attempt_id,
        "prior_gpu_cost_inr": prior_gpu_cost,
        "status": "running",
        "stages": {},
    }
    try:
        offer = manager.preflight()
        with manager:
            if manager.started_monotonic is None:
                raise QwenPipelineError("Jarvis billable clock was not recorded")
            config = load_qwen_v2_config(root)
            stage1_budget = _budget(
                manager, 1, prior_gpu_cost, manager.started_monotonic,
                prior_stage_cost=prior_gpu_cost,
            )
            with BudgetWatchdog(stage1_budget, manager.cleanup):
                manager.install_vllm(str(config["runtime"]["vllm_git_revision"]))
                manager.upload_server(
                    secret_file=secret_file if secret_file.is_absolute() else root / secret_file,
                    server_script=root / SERVER_SCRIPT,
                )
                manager.download_model_metadata(
                    model_id=str(config["model"]["hugging_face_id"]),
                    revision=str(config["model"]["revision"]),
                )
                manager.start_server(preflight=True)
                endpoint = manager.endpoint()
                dummy_model = _wait_for_server(
                    endpoint, secrets["VLLM_API_KEY"],
                    expected_alias=str(config["model"]["model_alias"]),
                    budget=stage1_budget, server_alive=manager.server_alive,
                    timeout_seconds=600,
                )
                manager.stop_server()
                manager.events.append({
                    "event": "dummy_engine_preflight_passed",
                    "exact_model_alias": dummy_model["exact_model_alias"],
                })
                manager.download_model(
                    model_id=str(config["model"]["hugging_face_id"]),
                    revision=str(config["model"]["revision"]),
                )
                manager.start_server()
                endpoint = manager.endpoint()
                server = _wait_for_server(
                    endpoint,
                    secrets["VLLM_API_KEY"],
                    expected_alias=SERIES_ID,
                    budget=stage1_budget,
                    server_alive=manager.server_alive,
                )
                remote = manager.collect_remote_receipt()
                compatibility = _run_compatibility(
                    root=root,
                    output=output / "stage1/compatibility-run-001",
                    endpoint=endpoint,
                    api_key=secrets["VLLM_API_KEY"],
                    offer=asdict(offer),
                    budget=stage1_budget,
                    retry_ledger=retry_ledger,
                )
            _validate_remote_gpu(offer.gpu, remote)
            projection = _project_remaining_cost(
                hourly_rate=offer.spot_rate_inr_per_hour,
                measured_wall_seconds=float(compatibility["inference_wall_seconds"]),
                completed=12,
                remaining=4650,
            )
            gate1 = stage_one_gate(
                compatibility,
                projected_remaining_cost_inr=projection,
                exact_model=server["exact_model_alias"],
                no_oom=True,
            )
            _write_stage_receipt(output / "stage1", 1, compatibility, gate1, stage1_budget, remote)
            summary["stages"]["stage1"] = compatibility
            with BudgetWatchdog(stage1_budget, manager.cleanup):
                manager.mirror_artifacts(output / "stage1", "stage1")
                stage1_budget.assert_cleanup_safe()
            if not gate1.passed:
                raise QwenPipelineError("Stage 1 gate failed: " + ", ".join(gate1.reasons))

            stage1_cost = prior_gpu_cost + _manager_cost(manager)
            stage2_started = time.monotonic()
            stage2_budget = _budget(manager, 2, stage1_cost, stage2_started)
            with BudgetWatchdog(stage2_budget, manager.cleanup):
                development = _run_split(
                    root=root,
                    split="development",
                    stage_root=output / "stage2/development-run-001",
                    endpoint=endpoint,
                    api_key=secrets["VLLM_API_KEY"],
                    budget=stage2_budget,
                    retry_ledger=retry_ledger,
                    manager=manager,
                )
            projection3 = _project_stage_three(stage2_budget)
            gate2 = stage_two_gate(
                development["extraction"], development["answers"], development["judge"],
                batch_valid_rates=development["batch_valid_rates"],
                projected_stage_three_cost_inr=projection3,
            )
            _write_stage_receipt(output / "stage2", 2, development, gate2, stage2_budget, remote)
            summary["stages"]["stage2"] = development
            with BudgetWatchdog(stage2_budget, manager.cleanup):
                manager.mirror_artifacts(output / "stage2", "stage2")
                stage2_budget.assert_cleanup_safe()
            if not gate2.passed:
                raise QwenPipelineError("Stage 2 gate failed: " + ", ".join(gate2.reasons))

            prior_cost = prior_gpu_cost + _manager_cost(manager)
            stage3_started = time.monotonic()
            stage3_budget = _budget(manager, 3, prior_cost, stage3_started)
            with BudgetWatchdog(stage3_budget, manager.cleanup):
                frozen = _run_split(
                    root=root,
                    split="test",
                    stage_root=output / "stage3/frozen-run-001",
                    endpoint=endpoint,
                    api_key=secrets["VLLM_API_KEY"],
                    budget=stage3_budget,
                    retry_ledger=retry_ledger,
                    manager=manager,
                )
            _write_stage_receipt(output / "stage3", 3, frozen, None, stage3_budget, remote)
            summary["stages"]["stage3"] = frozen
            with BudgetWatchdog(stage3_budget, manager.cleanup):
                manager.mirror_artifacts(output / "stage3", "stage3")
                stage3_budget.assert_cleanup_safe()
            summary["status"] = "completed_pending_cleanup"
        summary["status"] = "completed"
        summary["lifecycle"] = manager.receipt()
        summary["total_gpu_cost_inr"] = round(prior_gpu_cost + _manager_cost(manager), 6)
        summary["transport_retry_count"] = retry_ledger.used
        _write_json(attempt_root / "pipeline.json", summary)
        _write_json(output / "pipeline.json", summary)
        return summary
    except BaseException as error:
        summary["status"] = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed_or_budget_stopped"
        )
        summary["lifecycle"] = manager.receipt()
        raise
    finally:
        try:
            if manager.machine_id is not None and not manager.receipt()["destruction_verified"]:
                manager.cleanup()
        finally:
            try:
                if manager.machine_id is not None and not (attempt_root / "pipeline.json").exists():
                    summary["lifecycle"] = manager.receipt()
                    current = (
                        summary["lifecycle"].get("observed_account_spend_inr")
                        if summary["lifecycle"].get("observed_account_spend_inr") is not None
                        else summary["lifecycle"].get("gpu_cost_inr")
                    )
                    if isinstance(current, (int, float)):
                        summary["total_gpu_cost_inr"] = round(prior_gpu_cost + float(current), 6)
                    _write_json(attempt_root / "pipeline.json", summary)
            finally:
                _docker(root, "down")


def _run_compatibility(
    *, root, output, endpoint, api_key, offer, budget, retry_ledger,
) -> Mapping[str, object]:
    jobs = build_compatibility_jobs(root)
    tokenizer = VLLMTokenizer(base_url=endpoint, model=SERIES_ID, api_key=api_key)
    token_rows = []
    for job in jobs:
        count = tokenizer.count([
            {"role": "system", "content": job.system_prompt},
            {"role": "user", "content": job.user_prompt},
        ])
        token_rows.append({
            "position": job.position,
            "request_id": job.request_id,
            "token_count": count.count,
            "tokenizer_seconds": round(count.elapsed_seconds, 6),
        })
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "tokenization.jsonl", token_rows)
    inference_started = time.monotonic()
    manifest = execute_jobs(
        jobs,
        output_dir=output / "requests",
        client_factory=_client_factory(base_url=endpoint, model=SERIES_ID, api_key=api_key),
        retry_ledger=retry_ledger,
        before_attempt=budget.before_attempt,
        after_attempt=budget.after_attempt,
    )
    inference_wall_seconds = time.monotonic() - inference_started
    enriched = {
        **manifest,
        "gpu": offer["gpu"],
        "region": offer["region"],
        "hourly_rate_inr": offer["spot_rate_inr_per_hour"],
        "tokenized_input_count": sum(row["token_count"] for row in token_rows),
        "inference_wall_seconds": round(inference_wall_seconds, 6),
        "budget": budget.snapshot(),
    }
    _write_json(output / "stage-manifest.json", enriched)
    return enriched


def _run_split(
    *, root, split, stage_root, endpoint, api_key, budget, retry_ledger, manager,
) -> Mapping[str, object]:
    if stage_root.exists():
        raise FileExistsError(f"immutable stage output exists: {stage_root}")
    stage_root.mkdir(parents=True)
    factory = _client_factory(base_url=endpoint, model=SERIES_ID, api_key=api_key)
    extraction_dir = stage_root / "extraction"
    extraction = execute_jobs(
        build_extraction_jobs(root, split),
        output_dir=extraction_dir,
        client_factory=factory,
        retry_ledger=retry_ledger,
        before_attempt=budget.before_attempt,
        after_attempt=budget.after_attempt,
    )
    manager.mirror_artifacts(extraction_dir, f"{split}-extraction")
    expected_extraction = 20 if split == "development" else 80
    if extraction["successful_count"] != expected_extraction:
        raise QwenPipelineError(f"{split} extraction gate failed before answer spending")
    extraction_records = _jsonl(extraction_dir / "responses.jsonl")
    runtime = select_runtime(root, split)
    contexts_dir = stage_root / "contexts"
    materialized = _materialize(root, split, runtime, extraction_records, contexts_dir)
    answers_dir = stage_root / "answers-b0-b6"
    answers = execute_jobs(
        build_answer_jobs(root, split, materialized.contexts),
        output_dir=answers_dir,
        client_factory=factory,
        retry_ledger=retry_ledger,
        before_attempt=budget.before_attempt,
        after_attempt=budget.after_attempt,
    )
    answer_records = _jsonl(answers_dir / "responses.jsonl")
    b7_dir = stage_root / "answers-b7"
    b7 = write_derived_b7(b7_dir, derive_b7_records(materialized.contexts, answer_records))
    seal_dir = stage_root / "predictions"
    seal = seal_logical_predictions(answers_dir, b7_dir, seal_dir, split=split)
    manager.mirror_artifacts(seal_dir, f"{split}-predictions")
    references = lambda: _load_gold(root, split)
    judge_dir = stage_root / "judge"
    judge = run_judge(
        repo_root=root,
        prediction_dir=seal_dir,
        reference_loader=references,
        output_dir=judge_dir,
        base_url=endpoint,
        api_key=api_key,
        retry_ledger=retry_ledger,
        before_attempt=budget.before_attempt,
        after_attempt=budget.after_attempt,
    )
    metadata = _execution_metadata(extraction, answers, judge, budget)
    score = score_sealed_release(
        repo_root=root,
        prediction_dir=seal_dir,
        contexts_path=contexts_dir / "contexts.jsonl",
        extraction_dir=extraction_dir,
        reference_loader=references,
        execution_metadata=metadata,
        output_dir=stage_root / "scores",
    )
    result = {
        "split": split,
        "status": "completed" if not seal["failure_count"] else "completed_with_failures",
        "extraction": extraction,
        "materialization": {
            "context_count": len(materialized.contexts),
            "failure_count": len(materialized.failures),
        },
        "answers": answers,
        "b7": b7,
        "prediction_seal": seal,
        "judge": judge,
        "scorecard": score,
        "batch_valid_rates": _batch_valid_rates(answer_records),
        "execution_metadata": metadata,
    }
    _write_json(stage_root / "stage-manifest.json", result)
    return result


def _materialize(root, split, runtime, extraction_records, output):
    try:
        import psycopg
    except ModuleNotFoundError as error:
        raise QwenPipelineError("psycopg is required for Qwen materialization") from error
    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        connection.execute("CREATE SCHEMA public")
        rows = extraction_rows_from_records(extraction_records)
        result = materialize_qwen_contexts(
            connection,
            repo_root=root,
            split=split,
            runtime=runtime,
            extraction_rows=rows,
        )
        write_materialization(result, output)
        return result
    finally:
        connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        connection.execute("CREATE SCHEMA public")
        connection.close()


def _compatibility_context(runtime, case):
    user_id = str(case["user_id"])
    source = next(row for row in runtime["sources"] if row["user_id"] == user_id)
    evidence = [{
        "source_id": source["source_id"],
        "message_id": message["message_id"],
        "speaker_id": message["speaker_id"],
        "quote": message["text"],
        "support_type": "compatibility",
    } for message in source["messages"]]
    return ({
        "record_id": f"compat_{source['source_id']}",
        "record_kind": "atomic",
        "content": source["content"],
        "lifecycle_statuses": ["current"],
        "evidence": evidence,
        "relations": [],
    },)


def _render_compatibility_answer_prompt(task, case, records):
    payload = json.loads(_render_answer_prompt(task, case, records))
    body = {"qa": "answer", "summary": "summary", "interactive": "response"}[task]
    abstention = {
        "status": "abstained",
        body: "Not enough reliable memory.",
        "confidence": 0,
        "statements": [],
        "citations": [],
        "unresolved_parts": [],
        "abstention_reason": "insufficient_evidence",
    }
    allowed_citations = [
        {
            "source_id": evidence["source_id"],
            "message_id": evidence["message_id"],
            "quote": evidence["quote"],
        }
        for record in payload["context_records"]
        for evidence in record.get("evidence", [])
    ]
    payload["output_contract"]["rules"] = (
        "Return only the exact fields listed. If context_records is empty, or if "
        "the supplied evidence does not answer the runtime case, return the "
        "abstention_template exactly. For non-abstained outputs, copy every "
        "citation from allowed_citations without changing source_id, message_id, "
        "or quote, and make every statement an exact substring of the answer body."
    )
    payload["output_contract"]["allowed_citations"] = allowed_citations
    payload["output_contract"]["abstention_template"] = abstention
    if allowed_citations and task == "qa":
        statement = "The user accepted the product engineer role at Riverstone Labs."
        payload["output_contract"]["grounded_example"] = {
            "status": "answered",
            body: statement,
            "confidence": 0.8,
            "statements": [statement],
            "citations": [allowed_citations[0]],
            "unresolved_parts": [],
            "abstention_reason": None,
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compatibility_answer_response_format(task: str, baseline: str) -> dict[str, object]:
    if baseline == "B0" or task in {"summary", "interactive"}:
        return _response_format(f"compat_v2_{baseline}_{task}_abstain", _abstention_schema(task))
    return _response_format(f"compat_v2_{baseline}_{task}", _answer_schema(task))


def _abstention_schema(task: str) -> dict[str, object]:
    body = {"qa": "answer", "summary": "summary", "interactive": "response"}[task]
    properties: dict[str, object] = {
        "status": {"type": "string", "enum": ["abstained"]},
        body: {"type": "string", "enum": ["Not enough reliable memory."]},
        "confidence": {"type": "number", "enum": [0]},
        "statements": {"type": "array", "maxItems": 0},
        "citations": {"type": "array", "maxItems": 0},
        "unresolved_parts": {"type": "array", "maxItems": 0},
        "abstention_reason": {"type": "string", "enum": ["insufficient_evidence"]},
    }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _compatibility_candidate(task, records):
    body = {"qa": "answer", "summary": "summary", "interactive": "response"}[task]
    evidence = records[0]["evidence"][0]
    statement = "The compatibility fixture contains one grounded source statement."
    return {
        "status": "answered",
        body: statement,
        "confidence": 0.8,
        "statements": [statement],
        "citations": [{
            "source_id": evidence["source_id"],
            "message_id": evidence["message_id"],
            "quote": evidence["quote"],
        }],
        "unresolved_parts": [],
        "abstention_reason": None,
    }


def _case_input(task, case):
    fields = {
        "qa": ("question",),
        "summary": ("instruction",),
        "interactive": ("scenario", "initial_user_message"),
    }[task]
    return {name: case[name] for name in fields}


def _load_gold(root: Path, split: str):
    # Called only after verify_prediction_seal succeeds in judge/scorer entrypoints.
    return {
        name: [row for row in _jsonl(root / path) if row.get("split") == split]
        for name, path in GOLD_PATHS.items()
    }


def _batch_valid_rates(records):
    groups: dict[str, list[bool]] = {}
    for row in records:
        key = f"{row['baseline_id']}_{row['task']}"
        groups.setdefault(key, []).append(row.get("status") == "succeeded")
    return {key: sum(values) / len(values) for key, values in sorted(groups.items())}


def _execution_metadata(extraction, answers, judge, budget):
    manifests = (extraction, answers, judge)
    snapshot = budget.snapshot()
    output_tokens = sum(int(item.get("output_tokens", 0)) for item in manifests)
    elapsed = float(snapshot["elapsed_billable_seconds"])
    return {
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in manifests),
        "output_tokens": output_tokens,
        "provider_request_count": sum(int(item.get("provider_request_count", 0)) for item in manifests),
        "transport_retry_count": sum(int(item.get("transport_retry_count", 0)) for item in manifests),
        "gpu_cost_inr": snapshot["measured_cost_inr"],
        "inference_wall_seconds": elapsed,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
    }


def _budget(manager, stage, prior_cost, started, *, prior_stage_cost=0.0):
    if manager.offer is None:
        raise QwenPipelineError("Jarvis offer is unavailable")
    limits = STAGE_LIMITS[stage]
    return LiveBudgetController(
        stage=stage,
        hourly_rate_inr=manager.offer.spot_rate_inr_per_hour,
        stage_cap_inr=limits["stage_cap_inr"],
        cumulative_cap_inr=limits["cumulative_cap_inr"],
        prior_cost_inr=prior_cost,
        prior_stage_cost_inr=prior_stage_cost,
        started_monotonic=started,
    )


def _project_remaining_cost(*, hourly_rate, measured_wall_seconds, completed, remaining):
    if completed <= 0:
        return float("inf")
    projected_seconds = measured_wall_seconds * remaining / completed
    return projected_seconds * hourly_rate / 3600


def _project_stage_three(budget):
    # Frozen has exactly four times the development provider workload.
    return float(budget.snapshot()["measured_cost_inr"]) * 4


def _manager_cost(manager):
    receipt = manager.receipt()
    value = receipt.get("observed_account_spend_inr")
    if value is None:
        value = receipt.get("gpu_cost_inr")
    if not isinstance(value, (int, float)):
        raise QwenPipelineError("Jarvis cumulative cost is unavailable")
    return float(value)


def _validate_remote_gpu(expected: str, receipt: Mapping[str, object]) -> None:
    actual = str(receipt.get("gpu", "")).upper().replace(" ", "")
    if expected == "H100-80GB" and "H100" not in actual:
        raise QwenPipelineError("remote GPU does not match the approved H100 offer")
    if expected == "RTX-PRO6000-96GB" and not ("RTX" in actual and "6000" in actual):
        raise QwenPipelineError("remote GPU does not match the approved RTX-PRO6000 offer")


def _wait_for_server(
    endpoint, api_key, *, expected_alias, budget, server_alive, timeout_seconds=3600,
):
    deadline = time.monotonic() + timeout_seconds
    headers = {"Authorization": f"Bearer {api_key}"}
    last_code = "unavailable"
    while time.monotonic() < deadline:
        budget.assert_cleanup_safe()
        try:
            request = Request(f"{endpoint}/v1/models", headers=headers)
            with urlopen(request, timeout=30) as response:
                value = json.loads(response.read().decode("utf-8"))
            aliases = [str(row.get("id")) for row in value.get("data", []) if isinstance(row, Mapping)]
            if aliases == [expected_alias]:
                return {"exact_model_alias": True, "aliases": aliases}
            raise QwenPipelineError("vLLM returned an unexpected served model alias")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_code = type(error).__name__
            if not server_alive():
                raise QwenPipelineError("vLLM server exited during startup")
            time.sleep(15)
    raise QwenPipelineError(f"vLLM startup timed out: {last_code}")


def _local_preflight(root, output):
    verify_series_manifest(root)
    if _git(root, "branch", "--show-current") != "testing":
        raise QwenPipelineError("Qwen execution requires the testing branch")
    if _git(root, "status", "--short", "--untracked-files=no"):
        raise QwenPipelineError("Qwen execution requires a clean committed worktree")
    if output.exists() and any(output.iterdir()):
        allowed = {"series.json", "attempts", "lifecycle"}
        unexpected = {path.name for path in output.iterdir()} - allowed
        if unexpected:
            raise QwenPipelineError("Qwen v2 output root contains non-attempt run artifacts")


def _require_paid_execution_confirmation(value: str) -> None:
    if value != PAID_EXECUTION_CONFIRMATION:
        raise QwenPipelineError(
            "paid GPU execution is locked; fresh approval and the exact confirmation are required"
        )


def _historical_gpu_cost(output: Path) -> float:
    receipts = list((output / "lifecycle").glob("attempt-*/lifecycle.json"))
    legacy = output / "lifecycle/lifecycle.json"
    if legacy.is_file():
        receipts.append(legacy)
    machine_ids: set[int] = set()
    total = 0.0
    for path in sorted(receipts):
        value = json.loads(path.read_text(encoding="utf-8"))
        machine_id = value.get("machine_id")
        observed = value.get("observed_account_spend_inr")
        cost = observed if isinstance(observed, (int, float)) else value.get("gpu_cost_inr")
        if not value.get("destruction_verified"):
            raise QwenPipelineError(f"historical instance cleanup is unverified: {path}")
        if not isinstance(machine_id, int) or isinstance(machine_id, bool):
            raise QwenPipelineError(f"historical lifecycle receipt has no machine ID: {path}")
        if machine_id in machine_ids:
            raise QwenPipelineError(f"duplicate historical machine receipt: {machine_id}")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
            raise QwenPipelineError(f"historical lifecycle receipt has invalid cost: {path}")
        machine_ids.add(machine_id)
        total += float(cost)
    return round(total, 6)


def _next_attempt_id(output: Path) -> str:
    numbers = []
    for path in (output / "attempts").glob("attempt-*"):
        suffix = path.name.removeprefix("attempt-")
        if suffix.isdigit():
            numbers.append(int(suffix))
    for path in (output / "lifecycle").glob("attempt-*"):
        suffix = path.name.removeprefix("attempt-")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"attempt-{max(numbers, default=0) + 1:03d}"


def _docker(root, action):
    command = ["docker", "compose"]
    command += ["up", "-d", "--wait", "storage-db"] if action == "up" else ["down", "-v"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode != 0 and action == "up":
        raise QwenPipelineError("local PostgreSQL startup failed")


def _write_stage_receipt(path, stage, result, gate, budget, remote):
    path.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "qwen_stage_receipt_v2",
        "series_id": SERIES_ID,
        "stage": stage,
        "result_status": result.get("status"),
        "gate": None if gate is None else asdict(gate),
        "budget": budget.snapshot(),
        "remote_provenance": remote,
    }
    _write_json(path / "receipt.json", value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git(root, *args):
    completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--secret-file", default=SECRET_FILE.as_posix())
    parser.add_argument("--output", default=RESULT_ROOT.as_posix())
    parser.add_argument("--confirm-paid-gpu", required=True)
    args = parser.parse_args()
    result = run_pipeline(
        repo_root=Path(args.repo_root),
        secret_file=Path(args.secret_file),
        output_root=Path(args.output),
        paid_execution_confirmation=args.confirm_paid_gpu,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
