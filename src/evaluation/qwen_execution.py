"""Concurrent, resumable execution for OpenAI-compatible Qwen series."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from extraction.atomic import validate_atomic_response
from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format

from .frozen_answer_contracts import TASK_BODY, validate_answer_output
from .frozen_run_contracts import FrozenRunError
from .openai_client import OpenAIResponseMetadata
from .qwen_benchmark import PROMPTS, REGISTRY, SAMPLING, _render_answer_prompt, select_runtime
from .qwen_compatibility import _answer_schema, _response_format
from .qwen_materialization import ContextPackage, TASK_KEYS
from .qwen_v2_contract import SERIES_ID, load_qwen_v2_config
from .vllm_client import (
    VLLMClient,
    VLLMModelMismatchError,
    VLLMResponseError,
    VLLMTransportError,
)


MAX_WORKERS = 8
MAX_TRANSPORT_RETRIES = 25


class QwenExecutionError(RuntimeError):
    """Raised when a v2 request release cannot be executed or verified."""


Validator = Callable[[str, OpenAIResponseMetadata], Mapping[str, object]]
ClientFactory = Callable[["ExecutionJob"], object]
AttemptGate = Callable[["ExecutionJob", bool], None]
AttemptComplete = Callable[["ExecutionJob", bool, float], None]


@dataclass(frozen=True)
class ExecutionJob:
    request_id: str
    position: int
    split: str
    task: str
    record_id: str
    user_id: str
    baseline_id: str | None
    context_sha256: str | None
    context_count: int
    system_prompt: str
    user_prompt: str
    response_format: Mapping[str, object]
    max_output_tokens: int
    validator: Validator
    local_metadata: Mapping[str, object] | None = None
    series_id: str = SERIES_ID

    @property
    def request_sha256(self) -> str:
        return _sha({
            "series_id": self.series_id,
            "request_id": self.request_id,
            "position": self.position,
            "split": self.split,
            "task": self.task,
            "record_id": self.record_id,
            "user_id": self.user_id,
            "baseline_id": self.baseline_id,
            "context_sha256": self.context_sha256,
            "context_count": self.context_count,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response_format": self.response_format,
            "max_output_tokens": self.max_output_tokens,
            "local_metadata": self.local_metadata,
        })


class TransportRetryLedger:
    """Thread-safe global retry allowance shared by every batch in one run."""

    def __init__(self, maximum: int = MAX_TRANSPORT_RETRIES) -> None:
        if maximum < 0:
            raise ValueError("maximum retry count cannot be negative")
        self.maximum = maximum
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def claim(self) -> bool:
        with self._lock:
            if self._used >= self.maximum:
                return False
            self._used += 1
            return True


def build_extraction_jobs(
    repo_root: Path,
    split: str,
    *,
    series_id: str = SERIES_ID,
) -> tuple[ExecutionJob, ...]:
    root = repo_root.resolve()
    selected = select_runtime(root, split)
    registry = load_predicate_registry(root / REGISTRY)
    boolean_predicates = {
        definition.predicate
        for definition in registry.definitions
        if definition.object_shape == "boolean"
    }
    names = {str(row["user_id"]): str(row["display_name"]) for row in selected["users"]}
    system_prompt = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    raw_format = atomic_extraction_text_format(registry)
    response_format = _response_format(str(raw_format["name"]), raw_format["schema"])
    jobs = []
    for position, source in enumerate(selected["sources"], 1):
        adapted = _adapt_source(
            source,
            (str(source["user_id"]), str(source["source_id"])),
            names,
        )

        def validate(raw: str, metadata: OpenAIResponseMetadata, *, item=adapted) -> Mapping[str, object]:
            normalized = _normalize_qwen_extraction_response(raw, boolean_predicates)
            result = validate_atomic_response(item, normalized, metadata, registry=registry)
            return {"claims": [asdict(claim) for claim in result.claims]}

        jobs.append(ExecutionJob(
            request_id=f"extraction:{source['source_id']}",
            position=position,
            split=split,
            task="extraction",
            record_id=str(source["source_id"]),
            user_id=str(source["user_id"]),
            baseline_id=None,
            context_sha256=None,
            context_count=0,
            system_prompt=system_prompt,
            user_prompt=build_atomic_extraction_prompt(adapted, include_speaker_name=False),
            response_format=response_format,
            max_output_tokens=1200,
            validator=validate,
            series_id=series_id,
        ))
    return tuple(jobs)


def _normalize_qwen_extraction_response(
    raw: str,
    boolean_predicates: set[str],
) -> str:
    """Coerce Qwen's JSON-schema-valid string booleans for boolean predicates."""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
        return raw
    changed = False
    for record in parsed["claims"]:
        if not isinstance(record, dict):
            continue
        if record.get("predicate") not in boolean_predicates:
            continue
        value = record.get("object")
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            record["object"] = value.strip().lower() == "true"
            changed = True
    if not changed:
        return raw
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def build_answer_jobs(
    repo_root: Path,
    split: str,
    contexts: Sequence[ContextPackage],
    *,
    series_id: str = SERIES_ID,
) -> tuple[ExecutionJob, ...]:
    root = repo_root.resolve()
    selected = select_runtime(root, split)
    cases = {
        (task, str(case["case_id"])): case
        for task in TASK_KEYS
        for case in selected[task]
    }
    prompt_rows = json.loads((root / PROMPTS).read_text(encoding="utf-8"))["tasks"]
    prompts = {str(row["task"]): str(row["system_instruction"]) for row in prompt_rows}
    provider_contexts = sorted(
        (item for item in contexts if item.provider_call),
        key=lambda item: (int(item.baseline_id[1:]), TASK_KEYS.index(item.task), item.case_id),
    )
    expected = 798 if split == "development" else 3192
    if len(provider_contexts) != expected:
        raise QwenExecutionError(
            f"{split} provider context count changed: {len(provider_contexts)}, expected {expected}"
        )
    jobs = []
    for position, context in enumerate(provider_contexts, 1):
        case = cases.get((context.task, context.case_id))
        if case is None or str(case["user_id"]) != context.user_id:
            raise QwenExecutionError("materialized context does not match its runtime case")
        evidence_index = _evidence_index(context.context_records)

        def validate(
            raw: str,
            _metadata: OpenAIResponseMetadata,
            *, task=context.task,
            evidence=evidence_index,
        ) -> Mapping[str, object]:
            parsed = _normalize_qwen_answer_response(json.loads(raw), task, evidence)
            return validate_answer_output(parsed, task=task, evidence_index=evidence)

        jobs.append(ExecutionJob(
            request_id=f"answer:{context.baseline_id}:{context.task}:{context.case_id}",
            position=position,
            split=split,
            task=context.task,
            record_id=context.case_id,
            user_id=context.user_id,
            baseline_id=context.baseline_id,
            context_sha256=context.context_sha256,
            context_count=len(context.context_records),
            system_prompt=prompts[context.task],
            user_prompt=_render_answer_prompt(context.task, case, context.context_records),
            response_format=_response_format(
                f"qwen_v2_{context.task}", _answer_schema(context.task)
            ),
            max_output_tokens=_answer_max_output_tokens(context.task),
            validator=validate,
            series_id=series_id,
        ))
    return tuple(jobs)


def _answer_max_output_tokens(task: str) -> int:
    return 1600 if task == "summary" else 1000


def _normalize_qwen_answer_response(
    value: object,
    task: str,
    evidence_index: Mapping[tuple[str, str | None, str], object] | None = None,
) -> object:
    """Normalize Qwen answer phrasing into the frozen answer contract."""

    if not isinstance(value, dict):
        return value
    body_field = TASK_BODY.get(task)
    if body_field is None:
        return value
    if value.get("status") != "abstained":
        normalized = dict(value)
        if normalized.get("abstention_reason") == "":
            normalized["abstention_reason"] = None
        body = normalized.get(body_field)
        statements = normalized.get("statements")
        if isinstance(body, str) and body.strip() and isinstance(statements, list):
            normalized["statements"] = [
                item for item in statements
                if isinstance(item, str) and item.strip() and item in body
            ] or [body]
        if normalized.get("status") == "answered":
            normalized["unresolved_parts"] = []
        normalized["citations"] = _normalize_qwen_citations(
            normalized.get("citations"), evidence_index
        )
        return normalized
    normalized = dict(value)
    reason = normalized.get("abstention_reason")
    body = normalized.get(body_field)
    unresolved = normalized.get("unresolved_parts")
    if not isinstance(reason, str) or not reason.strip():
        if isinstance(body, str) and body.strip():
            reason = body
        elif isinstance(unresolved, list) and unresolved:
            reason = "; ".join(str(item) for item in unresolved if str(item).strip())
    if isinstance(reason, str) and reason.strip():
        normalized["abstention_reason"] = reason.strip()
        if not isinstance(body, str) or not body.strip():
            normalized[body_field] = reason.strip()
        normalized["confidence"] = 0
        normalized["statements"] = []
        normalized["citations"] = []
        normalized["unresolved_parts"] = []
    return normalized


def _normalize_qwen_citations(
    value: object,
    evidence_index: Mapping[tuple[str, str | None, str], object] | None,
) -> object:
    if evidence_index is None or not isinstance(value, list):
        return value
    normalized = []
    for citation in value:
        if not isinstance(citation, Mapping):
            normalized.append(citation)
            continue
        source_id = citation.get("source_id")
        message_id = citation.get("message_id")
        quote = citation.get("quote")
        if not isinstance(source_id, str) or not isinstance(quote, str):
            normalized.append(citation)
            continue
        normalized_message_id = (
            None if message_id == "" else message_id
            if isinstance(message_id, str) or message_id is None else None
        )
        key = (source_id, normalized_message_id, quote)
        if key in evidence_index:
            fixed = dict(citation)
            fixed["message_id"] = normalized_message_id
            normalized.append(fixed)
            continue
        candidates = [
            candidate
            for candidate in evidence_index
            if candidate[0] == source_id
            and _qwen_message_id_matches(candidate[1], key[1], source_id)
            and (quote in candidate[2] or candidate[2] in quote)
        ]
        if len(candidates) == 1:
            fixed = dict(citation)
            fixed["message_id"] = candidates[0][1]
            fixed["quote"] = candidates[0][2]
            normalized.append(fixed)
        else:
            normalized.append(citation)
    return normalized


def _qwen_message_id_matches(
    expected: str | None,
    actual: str | None,
    source_id: str,
) -> bool:
    if expected == actual:
        return True
    if expected is None:
        return actual is None or isinstance(actual, str)
    return False


def execute_jobs(
    jobs: Sequence[ExecutionJob],
    *,
    output_dir: Path,
    client_factory: ClientFactory,
    retry_ledger: TransportRetryLedger | None = None,
    workers: int = MAX_WORKERS,
    before_attempt: AttemptGate | None = None,
    after_attempt: AttemptComplete | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    retryable_http_statuses: Sequence[int] = (),
    retry_validation_failures: bool = False,
) -> Mapping[str, object]:
    """Execute missing jobs and seal an ordered batch after every job is terminal."""

    if workers <= 0:
        raise QwenExecutionError("worker count must be positive")
    if not jobs:
        raise QwenExecutionError("execution batch cannot be empty")
    ordered = tuple(sorted(jobs, key=lambda item: item.position))
    if [item.position for item in ordered] != list(range(1, len(ordered) + 1)):
        raise QwenExecutionError("job positions must be contiguous and one-based")
    if len({item.request_id for item in ordered}) != len(ordered):
        raise QwenExecutionError("request IDs must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_dir = output_dir / "requests"
    request_dir.mkdir(exist_ok=True)
    ledger = retry_ledger or TransportRetryLedger()
    retryable_statuses = frozenset(int(status) for status in retryable_http_statuses)
    prior = _load_terminal_records(request_dir, ordered)
    missing = [item for item in ordered if item.request_id not in prior]
    gate = before_attempt or (lambda _job, _retry: None)
    complete = after_attempt or (lambda _job, _retry, _seconds: None)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_one,
                job,
                request_dir,
                client_factory,
                ledger,
                gate,
                complete,
                monotonic,
                retryable_statuses,
                retry_validation_failures,
            ): job
            for job in missing
        }
        for future in as_completed(futures):
            future.result()
    records = _load_terminal_records(request_dir, ordered)
    if len(records) != len(ordered):
        raise QwenExecutionError("batch ended without a terminal record for every request")
    if len({item.series_id for item in ordered}) != 1:
        raise QwenExecutionError("batch cannot mix series IDs")
    rows = [records[item.request_id] for item in ordered]
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _write_or_verify(output_dir / "responses.jsonl", payload)
    successes = [row for row in rows if row["status"] == "succeeded"]
    failures = [row for row in rows if row["status"] == "failed"]
    failure_payload = b"".join(_canonical_bytes(row) + b"\n" for row in failures)
    _write_or_verify(output_dir / "failures.jsonl", failure_payload)
    manifest = {
        "schema_version": "qwen_execution_batch_v2",
        "series_id": ordered[0].series_id,
        "split": ordered[0].split,
        "task": ordered[0].task if len({item.task for item in ordered}) == 1 else "mixed",
        "status": "completed" if not failures else "completed_with_failures",
        "planned_request_count": len(rows),
        "provider_request_count": sum(int(row["provider_attempt_count"]) for row in rows),
        "successful_count": len(successes),
        "failure_count": len(failures),
        "transport_retry_count": sum(int(row["transport_retry_count"]) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in successes),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in successes),
        "worker_count": workers,
        "raw_invalid_output_persisted": False,
        "gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "responses_sha256": sha256(payload).hexdigest(),
        "failures_sha256": sha256(failure_payload).hexdigest(),
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_or_verify(output_dir / "manifest.json", manifest_bytes)
    return manifest


def derive_b7_records(
    contexts: Sequence[ContextPackage],
    b6_records: Sequence[Mapping[str, object]],
    *,
    series_id: str = SERIES_ID,
) -> tuple[dict[str, object], ...]:
    """Wrap each B6 terminal result with the deterministic answerability gate."""

    b7_contexts = sorted(
        (item for item in contexts if item.baseline_id == "B7"),
        key=lambda item: (TASK_KEYS.index(item.task), item.case_id),
    )
    by_key = {
        (str(row.get("task")), str(row.get("record_id"))): row
        for row in b6_records
        if row.get("baseline_id") == "B6"
    }
    derived = []
    for position, context in enumerate(b7_contexts, 1):
        underlying = by_key.get((context.task, context.case_id))
        base = {
            "schema_version": "qwen_execution_record_v2",
            "series_id": series_id,
            "request_id": f"derived:B7:{context.task}:{context.case_id}",
            "position": position,
            "split": context.split,
            "task": context.task,
            "record_id": context.case_id,
            "user_id": context.user_id,
            "baseline_id": "B7",
            "context_sha256": context.context_sha256,
            "context_count": len(context.context_records),
            "provider_attempt_count": 0,
            "transport_retry_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
        }
        if underlying is None or underlying.get("status") != "succeeded":
            derived.append({
                **base,
                "status": "failed",
                "generation_mode": "deterministic_answerability_gate",
                "failure_stage": "upstream",
                "failure_code": "b6_prediction_unavailable",
                "output": None,
                "underlying_answer_sha256": None,
                "gate_action": None,
            })
            continue
        if underlying.get("context_sha256") != context.context_sha256:
            raise QwenExecutionError("B6 and B7 context hashes differ")
        output = underlying.get("output")
        if not isinstance(output, Mapping):
            raise QwenExecutionError("successful B6 record has no output")
        underlying_hash = _sha(output)
        if underlying.get("underlying_answer_sha256") != underlying_hash:
            raise QwenExecutionError("B6 underlying answer hash changed")
        gated, action = _answerability_gate(context.task, context.context_records, output)
        derived.append({
            **base,
            "status": "succeeded",
            "generation_mode": "deterministic_answerability_gate",
            "failure_stage": None,
            "failure_code": None,
            "output": gated,
            "underlying_answer_sha256": underlying_hash,
            "gate_action": action,
        })
    return tuple(derived)


def extraction_rows_from_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Adapt sealed extraction terminals to the materializer's runtime-only input."""

    rows = []
    for record in sorted(records, key=lambda item: int(item["position"])):
        if record.get("task") != "extraction" or record.get("baseline_id") is not None:
            raise QwenExecutionError("non-extraction record entered extraction adaptation")
        succeeded = record.get("status") == "succeeded"
        output = record.get("output")
        claims = output.get("claims") if succeeded and isinstance(output, Mapping) else []
        if not isinstance(claims, list):
            raise QwenExecutionError("successful extraction record has no claims list")
        rows.append({
            "record_id": str(record["record_id"]),
            "user_id": str(record["user_id"]),
            "valid": succeeded,
            "claims": claims,
            "failure_code": None if succeeded else str(record.get("failure_code")),
            "request_sha256": str(record["request_sha256"]),
        })
    return tuple(rows)


def write_derived_b7(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    *,
    series_id: str = SERIES_ID,
) -> Mapping[str, object]:
    """Seal deterministic B7 logical predictions without provider accounting."""

    if not records:
        raise QwenExecutionError("B7 derivation cannot be empty")
    ordered = sorted(records, key=lambda item: int(item["position"]))
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in ordered)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_verify(output_dir / "responses.jsonl", payload)
    failures = [row for row in ordered if row.get("status") == "failed"]
    failure_payload = b"".join(_canonical_bytes(row) + b"\n" for row in failures)
    _write_or_verify(output_dir / "failures.jsonl", failure_payload)
    manifest = {
        "schema_version": "qwen_b7_derivation_v2",
        "series_id": series_id,
        "split": str(ordered[0]["split"]),
        "status": "completed" if not failures else "completed_with_failures",
        "logical_prediction_count": len(ordered),
        "successful_count": len(ordered) - len(failures),
        "failure_count": len(failures),
        "provider_request_count": 0,
        "b6_b7_underlying_answer_identity": all(
            row.get("underlying_answer_sha256") is not None
            for row in ordered
            if row.get("status") == "succeeded"
        ),
        "responses_sha256": sha256(payload).hexdigest(),
        "failures_sha256": sha256(failure_payload).hexdigest(),
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_or_verify(output_dir / "manifest.json", manifest_bytes)
    return manifest


def _execute_one(
    job: ExecutionJob,
    request_dir: Path,
    client_factory: ClientFactory,
    ledger: TransportRetryLedger,
    before_attempt: AttemptGate,
    after_attempt: AttemptComplete,
    monotonic: Callable[[], float],
    retryable_http_statuses: frozenset[int],
    retry_validation_failures: bool,
) -> None:
    attempts = 0
    retries = 0
    began = monotonic()
    while True:
        attempts += 1
        retrying = retries > 0
        before_attempt(job, retrying)
        attempt_began = monotonic()
        try:
            raw, metadata = client_factory(job).complete_with_metadata(
                system_prompt=job.system_prompt,
                user_prompt=job.user_prompt,
            )
            if not isinstance(metadata, OpenAIResponseMetadata):
                raise QwenExecutionError("vLLM provider metadata is missing")
            output = job.validator(raw, metadata)
            output_hash = _sha(output)
            record = _record_base(job, attempts, retries, monotonic() - began)
            record.update({
                "status": "succeeded",
                "generation_mode": "provider",
                "failure_stage": None,
                "failure_code": None,
                "output": output,
                "underlying_answer_sha256": output_hash if job.baseline_id == "B6" else None,
                "input_tokens": metadata.input_tokens,
                "output_tokens": metadata.output_tokens,
                "response_id": metadata.response_id,
                "returned_model": metadata.returned_model,
            })
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
            if error.status_code in retryable_http_statuses and ledger.claim():
                retries += 1
                continue
            code = f"http_{error.status_code}" if error.status_code is not None else "provider_response_error"
            record = _failure_record(job, attempts, retries, monotonic() - began, "provider", code)
            break
        except (json.JSONDecodeError, ValueError, TypeError, FrozenRunError) as error:
            if retry_validation_failures and ledger.claim():
                retries += 1
                continue
            record = _failure_record(job, attempts, retries, monotonic() - began, "validation", type(error).__name__)
            break
        except Exception as error:
            stage = "validation" if not isinstance(error, QwenExecutionError) else "execution"
            record = _failure_record(job, attempts, retries, monotonic() - began, stage, type(error).__name__)
            break
        finally:
            after_attempt(job, retrying, max(0.0, monotonic() - attempt_began))
    path = request_dir / f"{job.position:05d}-{sha256(job.request_id.encode()).hexdigest()[:16]}.json"
    _write_exclusive_atomic(path, json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _record_base(job: ExecutionJob, attempts: int, retries: int, seconds: float) -> dict[str, object]:
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
    }


def _failure_record(
    job: ExecutionJob,
    attempts: int,
    retries: int,
    seconds: float,
    stage: str,
    code: str,
) -> dict[str, object]:
    return {
        **_record_base(job, attempts, retries, seconds),
        "status": "failed",
        "generation_mode": "provider",
        "failure_stage": stage,
        "failure_code": code,
        "output": None,
        "underlying_answer_sha256": None,
        "input_tokens": None,
        "output_tokens": None,
        "response_id": None,
        "returned_model": None,
    }


def _answerability_gate(
    task: str,
    context_records: Sequence[Mapping[str, object]],
    output: Mapping[str, object],
) -> tuple[Mapping[str, object], str]:
    if output.get("status") == "abstained":
        return dict(output), "preserved_model_abstention"
    evidence = _evidence_index(context_records)
    promoted = any(
        set(record.get("lifecycle_statuses", ())).intersection(
            {"confirmed", "current", "historical", "disputed", "superseded"}
        )
        for record in context_records
    )
    if context_records and evidence and promoted:
        return dict(output), "allowed"
    body = TASK_BODY[task]
    return {
        "status": "abstained",
        body: "I don't have enough reliable memory to answer that.",
        "confidence": 0,
        "statements": [],
        "citations": [],
        "unresolved_parts": [],
        "abstention_reason": "no_promoted_claims" if not promoted else "incomplete_evidence",
    }, "forced_abstention"


def _evidence_index(
    records: Sequence[Mapping[str, object]],
) -> Mapping[tuple[str, str | None, str], Mapping[str, object]]:
    result = {}
    for record in records:
        evidence = record.get("evidence", [])
        if not isinstance(evidence, list):
            raise QwenExecutionError("context evidence is malformed")
        if not evidence and record.get("record_kind") == "source":
            source_id = record.get("source_id")
            content = record.get("content")
            if isinstance(source_id, str) and isinstance(content, str) and content.strip():
                result[(source_id, None, content)] = {
                    "source_id": source_id,
                    "message_id": None,
                    "quote": content,
                }
        for item in evidence:
            if not isinstance(item, Mapping):
                raise QwenExecutionError("context evidence item is malformed")
            source_id, message_id, quote = item.get("source_id"), item.get("message_id"), item.get("quote")
            if not isinstance(source_id, str) or not isinstance(quote, str):
                raise QwenExecutionError("context evidence identity is malformed")
            if message_id is not None and not isinstance(message_id, str):
                raise QwenExecutionError("context evidence message ID is malformed")
            result[(source_id, message_id, quote)] = item
    return result


def _load_terminal_records(
    request_dir: Path,
    jobs: Sequence[ExecutionJob],
) -> dict[str, dict[str, object]]:
    expected = {job.request_id: job for job in jobs}
    records = {}
    for path in sorted(request_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        request_id = str(row.get("request_id"))
        job = expected.get(request_id)
        if job is None:
            raise QwenExecutionError("checkpoint contains an unexpected request")
        if row.get("request_sha256") != job.request_sha256:
            raise QwenExecutionError("checkpoint request hash changed")
        if row.get("status") not in {"succeeded", "failed"} or request_id in records:
            raise QwenExecutionError("checkpoint terminal state is invalid")
        records[request_id] = row
    return records


def _client_factory(
    *, base_url: str, model: str, api_key: str | None, temperature: float | None = None,
) -> ClientFactory:
    def create(job: ExecutionJob) -> VLLMClient:
        judge = job.task == "judge"
        request_temperature = (
            float(temperature)
            if temperature is not None
            else (0.0 if judge else float(SAMPLING["temperature"]))
        )
        return VLLMClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=request_temperature,
            max_output_tokens=job.max_output_tokens,
            text_format=job.response_format,
            payload_options={
                **({} if judge else {
                "top_p": SAMPLING["top_p"],
                "top_k": SAMPLING["top_k"],
                "presence_penalty": SAMPLING["presence_penalty"],
                }),
                "seed": SAMPLING["seed"],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    return create


def _load_contexts(path: Path) -> tuple[ContextPackage, ...]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return tuple(ContextPackage(
        row["series_id"], row["split"], row["task"], row["case_id"], row["user_id"],
        row["as_of"], row["baseline_id"], row["provider_call"], row["snapshot"],
        tuple(row["context_records"]), row["context_sha256"],
    ) for row in rows)


def _write_exclusive_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise QwenExecutionError(f"checkpoint already exists: {path.name}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise QwenExecutionError(f"sealed artifact changed: {path.name}")
        return
    _write_exclusive_atomic(path, payload)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extraction", "answers"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--contexts")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    config = load_qwen_v2_config(root)
    model = str(config["model"]["model_alias"])
    api_key = os.environ.get("QWEN_VLLM_API_KEY")
    if args.mode == "extraction":
        jobs = build_extraction_jobs(root, args.split)
    else:
        if not args.contexts:
            parser.error("--contexts is required for answer execution")
        jobs = build_answer_jobs(root, args.split, _load_contexts(Path(args.contexts)))
    manifest = execute_jobs(
        jobs,
        output_dir=Path(args.output),
        client_factory=_client_factory(base_url=args.base_url, model=model, api_key=api_key),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
