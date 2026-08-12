"""Run resumable Qwen extraction and B0-B7 answers without scorer inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

from extraction.atomic import validate_atomic_response
from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format

from .frozen_answer_contracts import TASK_BODY, validate_answer_output
from .qwen_compatibility import _answer_schema, _response_format
from .qwen_series import BudgetGate, append_checkpoint, stage_three_is_allowed, write_immutable_json
from .vllm_client import VLLMClient


RUNTIME = Path("data/scaled-v1/runtime")
PROMPTS = Path("configs/evaluation/frozen_prompts_v1.json")
REGISTRY = Path("configs/extraction/predicate_registry_v2.json")
BASELINES = tuple(f"B{index}" for index in range(8))
TASKS = ("qa", "summary", "interactive")
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 42}


ANSWER_SUFFICIENCY_POLICY = {
    "sufficiency_standard": (
        "A direct allowed citation is sufficient for an answered factual claim. "
        "Missing perfect evidence, duplicate confirmation, or complete source coverage "
        "is not a reason to abstain when the supplied citation supports the requested fact."
    ),
    "citation_policy": (
        "Copy citations from allowed_citations exactly. Cite the minimal sufficient set; "
        "do not cite every allowed item by default."
    ),
    "status_policy": (
        "Use answered when the core request is supported. Use partially_answered when "
        "some requested parts are supported and some are not. Use disputed when supplied "
        "evidence supports incompatible answers. Abstain only when no supplied citation "
        "supports the core requested fact, the evidence is about another person, the "
        "available evidence is stale for a current-state question, or the request asks "
        "for an unsupported broad inference."
    ),
}


def _answer_system_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Qwen answerability policy: exact citations are a validation requirement, "
        "not an answerability threshold. Do not abstain merely because evidence is "
        "incomplete relative to perfect coverage. If at least one supplied allowed "
        "citation directly supports the case user's requested fact at the as_of "
        "cutoff, answer with that citation. Use partial or disputed statuses for "
        "supported-but-incomplete or conflicting evidence instead of defaulting to "
        "abstention."
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_runtime(repo_root: Path, split: str) -> dict[str, list[dict[str, object]]]:
    if split not in {"development", "test"}:
        raise ValueError("split must be development or test")
    root = repo_root.resolve()
    users = [row for row in _jsonl(root / RUNTIME / "users.jsonl") if row["split"] == split]
    user_ids = {str(row["user_id"]) for row in users}
    selected = {
        "users": users,
        "sources": [row for row in _jsonl(root / RUNTIME / "sources.jsonl") if row["user_id"] in user_ids],
        "qa": [row for row in _jsonl(root / RUNTIME / "qa.jsonl") if row["user_id"] in user_ids],
        "summary": [row for row in _jsonl(root / RUNTIME / "summaries.jsonl") if row["user_id"] in user_ids],
        "interactive": [row for row in _jsonl(root / RUNTIME / "interactive.jsonl") if row["user_id"] in user_ids],
    }
    expected = 932 if split == "development" else 3728
    actual = len(selected["sources"]) + len(BASELINES) * sum(len(selected[task]) for task in TASKS)
    if actual != expected:
        raise ValueError(f"{split} Qwen request count changed: {actual}, expected {expected}")
    return selected


def _client(
    *, base_url: str, model: str, api_key: str | None,
    response_format: Mapping[str, object], max_output_tokens: int,
) -> VLLMClient:
    return VLLMClient(
        base_url=base_url, model=model, api_key=api_key,
        temperature=SAMPLING["temperature"], max_output_tokens=max_output_tokens,
        text_format=response_format,
        payload_options={
            "top_p": SAMPLING["top_p"], "top_k": SAMPLING["top_k"],
            "presence_penalty": SAMPLING["presence_penalty"], "seed": SAMPLING["seed"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def _render_answer_prompt(task: str, case: Mapping[str, object], records: Iterable[Mapping[str, object]]) -> str:
    fields = {
        "qa": ("case_id", "user_id", "as_of", "question"),
        "summary": ("case_id", "user_id", "as_of", "instruction"),
        "interactive": ("case_id", "user_id", "as_of", "scenario", "initial_user_message"),
    }[task]
    body = TASK_BODY[task]
    context_records = list(records)
    payload = {
        "runtime_case": {name: case[name] for name in fields},
        "context_records": context_records,
        "allowed_citations": _allowed_citations(context_records),
        "answer_policy": ANSWER_SUFFICIENCY_POLICY,
        "response_format": "JSON object",
        "output_contract": {
            "exact_fields": ["status", body, "confidence", "statements", "citations", "unresolved_parts", "abstention_reason"],
            "statuses": ["answered", "abstained", "disputed", "partially_answered"],
            "citation_fields": ["source_id", "message_id", "quote"],
            "rules": (
                "Apply answer_policy before abstaining. Citations must exactly match "
                "allowed_citations. Answer from sufficient cited evidence; do not "
                "refuse because evidence is not perfect. Abstain rather than infer "
                "unsupported core facts."
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _allowed_citations(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    citations: dict[tuple[str, str | None, str], dict[str, object]] = {}
    for record in records:
        values = record.get("citation_evidence")
        if not isinstance(values, list):
            values = record.get("evidence")
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            source_id = item.get("source_id")
            message_id = item.get("message_id")
            quote = item.get("quote")
            if not isinstance(source_id, str) or not isinstance(quote, str):
                continue
            if message_id is not None and not isinstance(message_id, str):
                continue
            key = (source_id, message_id, quote)
            citations.setdefault(key, {
                "source_id": source_id,
                "message_id": message_id,
                "quote": quote,
            })
    return list(citations.values())


def _existing_records(output_dir: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in output_dir.glob("batches/*/responses.jsonl") if output_dir.exists() else ():
        for row in _jsonl(path):
            request_id = str(row["request_id"])
            if request_id in records:
                raise ValueError(f"duplicate checkpoint request: {request_id}")
            records[request_id] = row
    return records


def _budget(
    *, stage: int, stage_cap_inr: float, cumulative_cap_inr: float,
    prior_cost_inr: float, hourly_rate_inr: float, run_started: float,
    setup_seconds: float, request_reserve_seconds: float,
) -> None:
    current = hourly_rate_inr * (setup_seconds + time.monotonic() - run_started) / 3600
    reserve = hourly_rate_inr * request_reserve_seconds / 3600
    BudgetGate(
        stage, stage_cap_inr, cumulative_cap_inr,
        current, prior_cost_inr + current,
    ).assert_can_spend(reserve)


def run_series_slice(
    *, repo_root: Path, split: str, output_dir: Path, base_url: str,
    model: str, api_key: str | None, hourly_rate_inr: float,
    stage_cap_inr: float, cumulative_cap_inr: float, prior_cost_inr: float,
    setup_seconds: float, request_reserve_seconds: float = 120,
    resume: bool = False,
) -> dict[str, object]:
    from .frozen_contexts import FrozenRuntime, build_context_records, context_evidence_index

    root = repo_root.resolve()
    output = output_dir if output_dir.is_absolute() else root / output_dir
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass resume to use checkpoints: {output}")
    output.mkdir(parents=True, exist_ok=True)
    existing = _existing_records(output)
    selected = select_runtime(root, split)
    stage = 2 if split == "development" else 3
    run_started = time.monotonic()
    registry = load_predicate_registry(root / REGISTRY)
    names = {str(row["user_id"]): str(row["display_name"]) for row in selected["users"]}
    extraction_system = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    raw_schema = atomic_extraction_text_format(registry)
    extraction_format = _response_format(str(raw_schema["name"]), raw_schema["schema"])
    claims: list[dict[str, object]] = []
    valid = completed = input_tokens = output_tokens = 0

    extraction_path = output / "batches/extraction/responses.jsonl"
    for position, source in enumerate(selected["sources"], 1):
        request_id = f"extraction:{source['source_id']}"
        prior = existing.get(request_id)
        if prior is not None:
            completed += 1
            valid += int(bool(prior["valid"]))
            claims.extend(prior.get("claims", []))
            input_tokens += int(prior.get("input_tokens", 0))
            output_tokens += int(prior.get("output_tokens", 0))
            continue
        _budget(
            stage=stage, stage_cap_inr=stage_cap_inr, cumulative_cap_inr=cumulative_cap_inr,
            prior_cost_inr=prior_cost_inr, hourly_rate_inr=hourly_rate_inr,
            run_started=run_started, setup_seconds=setup_seconds,
            request_reserve_seconds=request_reserve_seconds,
        )
        adapted = _adapt_source(source, (str(source["user_id"]), str(source["source_id"])), names)
        prompt = build_atomic_extraction_prompt(adapted, include_speaker_name=False)
        began = time.monotonic()
        raw, metadata = _client(
            base_url=base_url, model=model, api_key=api_key,
            response_format=extraction_format, max_output_tokens=1200,
        ).complete_with_metadata(system_prompt=extraction_system, user_prompt=prompt)
        try:
            result = validate_atomic_response(adapted, raw, metadata, registry=registry)
            extracted = [
                {**asdict(claim), "source_id": source["source_id"], "user_id": source["user_id"], "observed_at": source["created_at"]}
                for claim in result.claims
            ]
            is_valid, error = True, None
            claims.extend(extracted)
        except Exception as exc:  # validation failures are checkpointed and never retried
            extracted, is_valid, error = [], False, f"{type(exc).__name__}: {exc}"
        record = {
            "request_id": request_id, "position": position, "task": "extraction",
            "record_id": source["source_id"], "user_id": source["user_id"],
            "valid": is_valid, "error": error, "claims": extracted,
            "input_tokens": metadata.input_tokens, "output_tokens": metadata.output_tokens,
            "inference_seconds": round(time.monotonic() - began, 6),
        }
        append_checkpoint(extraction_path, record)
        existing[request_id] = record
        completed += 1
        valid += int(is_valid)
        input_tokens += metadata.input_tokens or 0
        output_tokens += metadata.output_tokens or 0
    if not (output / "batches/extraction/manifest.json").exists():
        write_immutable_json(output / "batches/extraction/manifest.json", {
            "status": "completed", "request_count": len(selected["sources"]),
            "valid_response_count": sum(bool(existing[f"extraction:{row['source_id']}"]["valid"]) for row in selected["sources"]),
        })

    runtime = FrozenRuntime(
        users=tuple(selected["users"]), sources=tuple(selected["sources"]),
        cases={task: tuple(selected[task]) for task in TASKS}, claims=tuple(claims),
    )
    prompt_rows = json.loads((root / PROMPTS).read_text(encoding="utf-8"))["tasks"]
    prompts = {str(row["task"]): str(row["system_instruction"]) for row in prompt_rows}
    for baseline in BASELINES:
        for task in TASKS:
            batch_id = f"{baseline}_{task}"
            checkpoint = output / f"batches/{batch_id}/responses.jsonl"
            for position, case in enumerate(selected[task], 1):
                request_id = f"answer:{baseline}:{task}:{case['case_id']}"
                prior = existing.get(request_id)
                if prior is not None:
                    completed += 1
                    valid += int(bool(prior["valid"]))
                    input_tokens += int(prior.get("input_tokens", 0))
                    output_tokens += int(prior.get("output_tokens", 0))
                    continue
                _budget(
                    stage=stage, stage_cap_inr=stage_cap_inr, cumulative_cap_inr=cumulative_cap_inr,
                    prior_cost_inr=prior_cost_inr, hourly_rate_inr=hourly_rate_inr,
                    run_started=run_started, setup_seconds=setup_seconds,
                    request_reserve_seconds=request_reserve_seconds,
                )
                records = build_context_records(runtime, baseline_id=baseline, task=task, case=case)
                prompt = _render_answer_prompt(task, case, records)
                began = time.monotonic()
                raw, metadata = _client(
                    base_url=base_url, model=model, api_key=api_key,
                    response_format=_response_format(f"qwen_{task}_v1", _answer_schema(task)),
                    max_output_tokens=1000,
                ).complete_with_metadata(system_prompt=prompts[task], user_prompt=prompt)
                try:
                    parsed = json.loads(raw)
                    canonical = validate_answer_output(parsed, task=task, evidence_index=context_evidence_index(records))
                    is_valid, error = True, None
                except Exception as exc:  # invalid output is recorded once and not retried
                    canonical, is_valid, error = None, False, f"{type(exc).__name__}: {exc}"
                record = {
                    "request_id": request_id, "position": position, "task": task,
                    "baseline_id": baseline, "case_id": case["case_id"], "user_id": case["user_id"],
                    "context_count": len(records), "valid": is_valid, "error": error,
                    "output": canonical, "raw_output": None if is_valid else raw,
                    "input_tokens": metadata.input_tokens, "output_tokens": metadata.output_tokens,
                    "inference_seconds": round(time.monotonic() - began, 6),
                }
                append_checkpoint(checkpoint, record)
                existing[request_id] = record
                completed += 1
                valid += int(is_valid)
                input_tokens += metadata.input_tokens or 0
                output_tokens += metadata.output_tokens or 0
            batch_manifest = output / f"batches/{batch_id}/manifest.json"
            if not batch_manifest.exists():
                batch_rows = [existing[f"answer:{baseline}:{task}:{case['case_id']}"] for case in selected[task]]
                write_immutable_json(batch_manifest, {
                    "status": "completed", "baseline_id": baseline, "task": task,
                    "request_count": len(batch_rows),
                    "valid_response_count": sum(bool(row["valid"]) for row in batch_rows),
                })
    elapsed = time.monotonic() - run_started
    run_cost = hourly_rate_inr * (setup_seconds + elapsed) / 3600
    manifest = {
        "schema_version": "qwen_benchmark_run_v1", "series_id": "qwen35-27b-fp8-v1",
        "split": split, "status": "completed", "request_count": completed,
        "valid_response_count": valid, "valid_response_rate": valid / completed,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "setup_seconds": setup_seconds, "inference_wall_seconds": round(elapsed, 6),
        "hourly_rate_inr": hourly_rate_inr, "gpu_cost_inr": round(run_cost, 4),
        "prior_cost_inr": prior_cost_inr, "automatic_retry_count": 0,
        "gold_opened": False, "oracle_opened": False, "review_opened": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed run manifest already exists: {manifest_path}")
    write_immutable_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen35-27b-fp8-v1")
    parser.add_argument("--api-key")
    parser.add_argument("--hourly-rate-inr", type=float, required=True)
    parser.add_argument("--stage-cap-inr", type=float, required=True)
    parser.add_argument("--cumulative-cap-inr", type=float, required=True)
    parser.add_argument("--prior-cost-inr", type=float, required=True)
    parser.add_argument("--setup-seconds", type=float, required=True)
    parser.add_argument("--request-reserve-seconds", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage2-valid-responses", type=int)
    parser.add_argument("--stage2-total-responses", type=int)
    parser.add_argument("--projected-cost-inr", type=float)
    args = parser.parse_args()
    if args.split == "test":
        gate_values = (args.stage2_valid_responses, args.stage2_total_responses, args.projected_cost_inr)
        if any(value is None for value in gate_values) or not stage_three_is_allowed(
            valid_responses=args.stage2_valid_responses,
            total_responses=args.stage2_total_responses,
            projected_run_cost_inr=args.projected_cost_inr,
            remaining_budget_inr=args.cumulative_cap_inr - args.prior_cost_inr,
        ):
            raise SystemExit("Stage 3 quality or reserved-budget gate did not pass")
    result = run_series_slice(
        repo_root=Path(args.repo_root), split=args.split, output_dir=Path(args.output),
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        hourly_rate_inr=args.hourly_rate_inr, stage_cap_inr=args.stage_cap_inr,
        cumulative_cap_inr=args.cumulative_cap_inr, prior_cost_inr=args.prior_cost_inr,
        setup_seconds=args.setup_seconds, request_reserve_seconds=args.request_reserve_seconds,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
