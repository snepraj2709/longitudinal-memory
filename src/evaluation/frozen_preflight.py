"""Build and deeply verify the provider-disabled Step 10.2 preflight."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

import tiktoken

from extraction.atomic import validate_atomic_response  # imported to pin runtime dependency
from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.run_safety import count_step35_request_tokens
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format

from .comparison_freeze import verify_frozen_comparison
from .frozen_preflight_contracts import (
    BASELINE_ORDER,
    PREFLIGHT_VERSION,
    SCHEMA_VERSION,
    TASK_ORDER,
    BatchPlan,
    FrozenPreflightError,
    PreflightChecks,
    TokenEstimate,
    TransmissionRecord,
    batch_from_mapping,
    calculate_cost,
    canonical_json_bytes,
    checks_from_mapping,
    estimate_from_mapping,
    money,
    parse_json_bytes,
    stable_sha256,
    transmission_from_mapping,
)


STARTING_COMMIT = "b2ae263e1129758325a30db57c03620628c6355e"
GUIDANCE_VERSION = "step-10.2-guidance-v1"
GUIDANCE_SHA256 = "b91d4b44752e649c3419eb557c9e0148839820998e94400e48a0f13a00f83f05"
CONFIG_PATH = Path("configs/evaluation/frozen_preflight_v1.json")
CONFIG_SHA256 = "de6c779db44a853d499ec8cd9e1ed06f60072ae531424dbd8e13fd457ff80c21"
COMPARISON_MANIFEST_PATH = Path("results/evaluation/frozen-comparison-v1/manifest.json")
COMPARISON_MANIFEST_SHA256 = "14d14b701ac54005a8038335cd37be6cc2eb6446ef68be8a440f0159d2291159"
COMPARISON_PROMPT_PATH = Path("configs/evaluation/frozen_prompts_v1.json")
COMPARISON_PROMPT_SHA256 = "681935ec199e6f0a4f7cb37c971c0c28295293af9f792f1f41f6225ce5f468bf"
SCALED_MANIFEST_PATH = Path("data/scaled-v1/manifest.json")
SCALED_MANIFEST_SHA256 = "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3"
SCALED_DATASET_SHA256 = "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61"
REGISTRY_PATH = Path("configs/extraction/predicate_registry_v2.json")
REGISTRY_SHA256 = "15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1"
DATA_ROOT = Path("data/evaluation/frozen-preflight-v1")
RESULT_ROOT = Path("results/evaluation/frozen-preflight-v1")
RUNTIME_INPUTS = (
    Path("data/scaled-v1/runtime/users.jsonl"),
    Path("data/scaled-v1/runtime/sources.jsonl"),
    Path("data/scaled-v1/runtime/qa.jsonl"),
    Path("data/scaled-v1/runtime/summaries.jsonl"),
    Path("data/scaled-v1/runtime/interactive.jsonl"),
)
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/evaluation/frozen_preflight_contracts.py"),
    Path("src/evaluation/frozen_preflight.py"),
)
RESULT_ARTIFACTS = (
    "preflight.json", "batches.jsonl", "token-estimates.jsonl",
    "approval-request.md", "checks.json", "failures.jsonl", "run.json", "findings.md",
)
USER_FIELDS = {"user_id", "display_name", "timezone", "split", "profile_note"}
SOURCE_FIELDS = {
    "source_id", "source_type", "user_id", "created_at", "ingested_at",
    "participants", "content", "messages", "metadata",
}
CASE_FIELDS = {
    "qa": {"as_of", "benchmark_version", "capability", "case_id", "difficulty", "question", "split", "task", "user_id"},
    "summary": {"as_of", "benchmark_version", "capability", "case_id", "difficulty", "instruction", "split", "task", "user_id"},
    "interactive": {"allowed_turns", "as_of", "benchmark_version", "capability", "case_id", "difficulty", "initial_user_message", "scenario", "split", "task", "user_id"},
}
SOURCE_TRANSMITTED_FIELDS = tuple(sorted((
    "content", "created_at", "messages[].speaker_id", "messages[].text", "metadata.title",
    "participants", "source_id", "source_type", "user_id", "user_display_names",
)))
CASE_TRANSMITTED_FIELDS = {
    "qa": tuple(sorted(("as_of", "baseline_memory_context", "case_id", "question", "user_id"))),
    "summary": tuple(sorted(("as_of", "baseline_memory_context", "case_id", "instruction", "user_id"))),
    "interactive": tuple(sorted(("as_of", "baseline_memory_context", "case_id", "initial_user_message", "scenario", "user_id"))),
}


def build_frozen_preflight(
    *,
    repo_root: str | Path = ".",
    data_root: str | Path | None = None,
    result_root: str | Path | None = None,
    reader: Callable[[Path], bytes] | None = None,
) -> Mapping[str, str]:
    """Create the immutable no-call preflight from runtime-only inputs."""

    root = Path(repo_root).resolve()
    data = Path(data_root).resolve() if data_root is not None else root / DATA_ROOT
    result = Path(result_root).resolve() if result_root is not None else root / RESULT_ROOT
    read = reader or (lambda path: path.read_bytes())
    _empty_or_absent(data)
    _empty_or_absent(result)
    config, raw_inputs, transmissions, estimates, batches = _rebuild(root, read)

    runtime = data / "runtime"
    runtime.mkdir(parents=True)
    result.mkdir(parents=True)
    transmission_bytes = b"".join(canonical_json_bytes(item) for item in transmissions)
    (runtime / "transmission-plan.jsonl").write_bytes(transmission_bytes)
    runtime_manifest = {
        "preflight_version": PREFLIGHT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "runtime_inputs": {
            path.relative_to(root).as_posix(): _sha_bytes(raw_inputs[path])
            for path in RUNTIME_INPUTS_ABSOLUTE(root)
        },
        "runtime_input_count": 5,
        "runtime_row_count": 680,
        "transmission_record_count": len(transmissions),
        "transmission_plan_sha256": _sha_bytes(transmission_bytes),
        "local_runtime_only": True,
        "provider_request_count": 0,
        "gold_opened": False,
        "oracle_opened": False,
        "review_queue_opened": False,
    }
    (runtime / "manifest.json").write_bytes(canonical_json_bytes(runtime_manifest))
    dataset_manifest = {
        "dataset_version": "frozen-preflight-v1",
        "preflight_version": PREFLIGHT_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "scaled_manifest_sha256": SCALED_MANIFEST_SHA256,
        "scaled_dataset_sha256": SCALED_DATASET_SHA256,
        "runtime_manifest_sha256": _sha_file(runtime / "manifest.json"),
        "transmission_plan_sha256": _sha_file(runtime / "transmission-plan.jsonl"),
        "runtime_rows_opened_locally_for_preflight": True,
        "runtime_content_logged": False,
        "excluded_inputs": [
            "data/scaled-v1/gold", "data/scaled-v1/oracle", "data/scaled-v1/review_queues",
            "data/scaled-v1/predictions", ".env", "OPENAI_API_KEY value",
        ],
        "credential_reuse_approved": True,
        "data_transmission_approved": False,
        "paid_execution_approved": False,
    }
    (data / "manifest.json").write_bytes(canonical_json_bytes(dataset_manifest))

    batch_bytes = b"".join(canonical_json_bytes(item) for item in batches)
    estimate_bytes = b"".join(canonical_json_bytes(item) for item in estimates)
    (result / "batches.jsonl").write_bytes(batch_bytes)
    (result / "token-estimates.jsonl").write_bytes(estimate_bytes)
    totals = _totals(batches)
    preflight = _preflight_record(config, totals)
    checks = PreflightChecks(
        preflight_version=PREFLIGHT_VERSION,
        batch_count=25, extraction_batch_count=1, answer_batch_count=24,
        planned_request_count=4660, maximum_retry_requests=0,
        transmission_record_count=4660, source_count=100, case_count=570,
        development_case_count=114, frozen_test_case_count=456,
        provider_request_count=0, gold_opened=False, oracle_opened=False,
        review_queue_opened=False, credential_reuse_approved=True,
        data_transmission_approved=False, paid_execution_approved=False,
    )
    (result / "preflight.json").write_bytes(canonical_json_bytes(preflight))
    (result / "checks.json").write_bytes(canonical_json_bytes(checks))
    (result / "failures.jsonl").write_bytes(b"")
    (result / "approval-request.md").write_text(_approval_request(batches, totals), encoding="utf-8")
    (result / "findings.md").write_text(_findings(totals), encoding="utf-8")
    run = {
        "run_version": "frozen-preflight-run-v1",
        "status": "awaiting_transmission_and_spend_approval",
        "provider_request_count": 0,
        "planned_request_count": 4660,
        "maximum_retry_requests": 0,
        "batch_count": 25,
        "expected_incremental_cost_usd": totals["expected"],
        "maximum_incremental_cost_usd": totals["maximum"],
        "historical_openai_spend_usd": "0.2314404",
        "projected_expected_cumulative_spend_usd": totals["cumulative_expected"],
        "projected_maximum_cumulative_spend_usd": totals["cumulative_maximum"],
        "credential_reuse_approved": True,
        "data_transmission_approved": False,
        "paid_execution_approved": False,
        "step_10_3_started": False,
    }
    (result / "run.json").write_bytes(canonical_json_bytes(run))
    artifacts = {name: _sha_file(result / name) for name in RESULT_ARTIFACTS}
    result_manifest = {
        "result_version": "frozen-preflight-v1",
        "preflight_version": PREFLIGHT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "implementation": {
            path.as_posix(): _sha_file(root / path) for path in IMPLEMENTATION_PATHS
        },
        "inputs": {
            CONFIG_PATH.as_posix(): CONFIG_SHA256,
            COMPARISON_MANIFEST_PATH.as_posix(): COMPARISON_MANIFEST_SHA256,
            COMPARISON_PROMPT_PATH.as_posix(): COMPARISON_PROMPT_SHA256,
            SCALED_MANIFEST_PATH.as_posix(): SCALED_MANIFEST_SHA256,
            REGISTRY_PATH.as_posix(): REGISTRY_SHA256,
            DATA_ROOT.joinpath("manifest.json").as_posix(): _sha_file(data / "manifest.json"),
        },
        "runtime_inputs": {
            path.relative_to(root).as_posix(): _sha_bytes(raw_inputs[path])
            for path in RUNTIME_INPUTS_ABSOLUTE(root)
        },
        "artifacts": artifacts,
        "counts": asdict(checks),
        "cost": totals,
        "pricing_sources": config["pricing"],
        "approvals": config["approval_policy"],
        "provider_request_count": 0,
        "step_10_3_started": False,
        "limitations": [
            "costs_are_preflight_estimates_not_provider_usage",
            "derived_baseline_runtime_releases_do_not_exist_yet",
            "runtime_rows_were_read_locally_but_not_transmitted",
            "gold_oracle_and_review_rows_remained_unopened",
        ],
    }
    (result / "manifest.json").write_bytes(canonical_json_bytes(result_manifest))
    return _artifact_hashes(data, result)


def verify_frozen_preflight(
    *, repo_root: str | Path = ".",
    data_root: str | Path | None = None,
    result_root: str | Path | None = None,
    reader: Callable[[Path], bytes] | None = None,
) -> Mapping[str, str]:
    """Rebuild the preflight and reject any changed source or artifact byte."""

    root = Path(repo_root).resolve()
    data = Path(data_root).resolve() if data_root is not None else root / DATA_ROOT
    result = Path(result_root).resolve() if result_root is not None else root / RESULT_ROOT
    read = reader or (lambda path: path.read_bytes())
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        expected = build_frozen_preflight(
            repo_root=root, data_root=base / "data", result_root=base / "result", reader=read,
        )
        actual = _artifact_hashes(data, result)
        if expected != actual:
            raise FrozenPreflightError("checked preflight does not recompute")
    for line_number, line in enumerate((data / "runtime/transmission-plan.jsonl").read_bytes().splitlines(), 1):
        transmission_from_mapping(parse_json_bytes(line, location=f"transmission:{line_number}"))
    for line_number, line in enumerate((result / "token-estimates.jsonl").read_bytes().splitlines(), 1):
        estimate_from_mapping(parse_json_bytes(line, location=f"estimate:{line_number}"))
    for line_number, line in enumerate((result / "batches.jsonl").read_bytes().splitlines(), 1):
        batch_from_mapping(parse_json_bytes(line, location=f"batch:{line_number}"))
    checks_from_mapping(parse_json_bytes((result / "checks.json").read_bytes(), location="checks"))
    return actual


def _rebuild(root: Path, read: Callable[[Path], bytes]):
    config_raw = read(root / CONFIG_PATH)
    if _sha_bytes(config_raw) != CONFIG_SHA256:
        raise FrozenPreflightError("preflight config changed")
    config = parse_json_bytes(config_raw, location=CONFIG_PATH.as_posix())
    _validate_config(config)
    if _sha_bytes(read(root / COMPARISON_MANIFEST_PATH)) != COMPARISON_MANIFEST_SHA256:
        raise FrozenPreflightError("Step 10.1 manifest changed")
    if _sha_bytes(read(root / COMPARISON_PROMPT_PATH)) != COMPARISON_PROMPT_SHA256:
        raise FrozenPreflightError("comparison prompt changed")
    if _sha_bytes(read(root / SCALED_MANIFEST_PATH)) != SCALED_MANIFEST_SHA256:
        raise FrozenPreflightError("scaled manifest changed")
    if _sha_bytes(read(root / REGISTRY_PATH)) != REGISTRY_SHA256:
        raise FrozenPreflightError("predicate registry changed")
    verify_frozen_comparison(repo_root=root, reader=read)
    scaled_manifest = parse_json_bytes(read(root / SCALED_MANIFEST_PATH), location="scaled manifest")
    expected_runtime = _runtime_hashes(scaled_manifest)
    raw_inputs: dict[Path, bytes] = {}
    for relative in RUNTIME_INPUTS:
        absolute = root / relative
        raw = read(absolute)
        if _sha_bytes(raw) != expected_runtime[relative.as_posix()]:
            raise FrozenPreflightError(f"runtime input changed: {relative}")
        raw_inputs[absolute] = raw
    users = _jsonl(raw_inputs[root / RUNTIME_INPUTS[0]], "users")
    sources = _jsonl(raw_inputs[root / RUNTIME_INPUTS[1]], "sources")
    qa = _jsonl(raw_inputs[root / RUNTIME_INPUTS[2]], "qa")
    summaries = _jsonl(raw_inputs[root / RUNTIME_INPUTS[3]], "summaries")
    interactive = _jsonl(raw_inputs[root / RUNTIME_INPUTS[4]], "interactive")
    _validate_runtime(users, sources, qa, summaries, interactive)
    prompts = parse_json_bytes(read(root / COMPARISON_PROMPT_PATH), location="comparison prompts")
    transmissions, estimates, batches = _plans(root, config, users, sources, {
        "qa": qa, "summary": summaries, "interactive": interactive,
    }, prompts)
    return config, raw_inputs, transmissions, estimates, batches


def _plans(root: Path, config: Mapping[str, object], users, sources, cases, prompts):
    encoding = tiktoken.get_encoding("o200k_base")
    user_names = {str(row["user_id"]): str(row["display_name"]) for row in users}
    user_splits = {str(row["user_id"]): str(row["split"]) for row in users}
    registry = load_predicate_registry(root / REGISTRY_PATH)
    system_prompt = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    text_format = atomic_extraction_text_format(registry)
    extraction_model = _mapping(config["extraction_model"], "extraction model")
    answer_model = _mapping(config["answer_model"], "answer model")
    answer_policy = _mapping(config["answer_token_policy"], "answer token policy")
    prompt_by_task = {
        str(item["task"]): str(item["system_instruction"])
        for item in _list_of_mappings(prompts["tasks"], "prompt tasks")
    }
    transmissions: list[TransmissionRecord] = []
    estimates: list[TokenEstimate] = []
    batch_members: dict[str, list[TokenEstimate]] = {}
    batch_names = [str(value) for value in _list(config["batch_order"], "batch order")]

    extraction_batch = "batch_01_extraction_all_sources"
    batch_members[extraction_batch] = []
    for position, record in enumerate(sources, 1):
        user_id = str(record["user_id"])
        source_id = str(record["source_id"])
        adapted = _adapt_source(record, (user_id, source_id), user_names)
        prompt = build_atomic_extraction_prompt(adapted, include_speaker_name=False)
        exact = count_step35_request_tokens(
            model=str(extraction_model["requested_model"]),
            system_prompt=system_prompt,
            user_prompt=prompt,
            text_format=text_format,
            generation_settings={
                "temperature": extraction_model["temperature"],
                "max_output_tokens": extraction_model["max_output_tokens"],
                "store": extraction_model["store"],
            },
        )
        maximum = exact + int(extraction_model["input_reserve_tokens"])
        expected_output = int(extraction_model["expected_output_tokens"])
        maximum_output = int(extraction_model["max_output_tokens"])
        transmission_payload = {
            "kind": "extraction_source", "record_id": source_id, "user_id": user_id,
            "split": user_splits[user_id], "baseline_id": None, "task": None,
            "source_type": str(record["source_type"]),
            "fields_transmitted": SOURCE_TRANSMITTED_FIELDS,
            "synthetic_benchmark": True, "restricted_content": False,
        }
        transmission = TransmissionRecord(
            transmission_id=stable_sha256(transmission_payload), **transmission_payload,
        )
        transmissions.append(transmission)
        estimate_payload = {
            "batch_id": extraction_batch, "position": position, "record_id": source_id,
            "user_id": user_id, "split": user_splits[user_id], "task": "extraction",
            "model": str(extraction_model["requested_model"]),
            "exact_runtime_text_tokens": exact, "expected_memory_tokens": 0,
            "framing_reserve_tokens": int(extraction_model["input_reserve_tokens"]),
            "expected_input_tokens": exact, "maximum_input_tokens": maximum,
            "expected_output_tokens": expected_output, "maximum_output_tokens": maximum_output,
            "expected_cost_usd": calculate_cost(exact, expected_output, str(extraction_model["input_usd_per_million_tokens"]), str(extraction_model["output_usd_per_million_tokens"])),
            "maximum_cost_usd": calculate_cost(maximum, maximum_output, str(extraction_model["input_usd_per_million_tokens"]), str(extraction_model["output_usd_per_million_tokens"])),
        }
        estimate = TokenEstimate(estimate_id=stable_sha256(estimate_payload), **estimate_payload)
        estimates.append(estimate)
        batch_members[extraction_batch].append(estimate)

    expected_memory = _mapping(answer_policy["expected_memory_tokens"], "expected memory")
    maximum_memory = _mapping(answer_policy["maximum_memory_tokens"], "maximum memory")
    expected_output_by_task = _mapping(answer_policy["expected_output_tokens"], "expected output")
    reserve = int(answer_policy["framing_reserve_tokens"])
    batch_position = 1
    for baseline in BASELINE_ORDER:
        for task in TASK_ORDER:
            batch_position += 1
            batch_id = f"batch_{batch_position:02d}_{baseline}_{task}"
            batch_members[batch_id] = []
            for position, record in enumerate(cases[task], 1):
                user_id = str(record["user_id"])
                case_id = str(record["case_id"])
                split = str(record["split"])
                runtime_text = _case_runtime_text(task, record)
                exact_runtime = len(encoding.encode(prompt_by_task[task] + "\n" + runtime_text))
                expected_input = exact_runtime + int(expected_memory[baseline]) + reserve
                maximum_input = exact_runtime + int(maximum_memory[baseline]) + reserve
                expected_output = int(expected_output_by_task[task])
                maximum_output = int(answer_model["max_output_tokens"])
                transmission_payload = {
                    "kind": "answer_case", "record_id": case_id, "user_id": user_id,
                    "split": split, "baseline_id": baseline, "task": task, "source_type": None,
                    "fields_transmitted": CASE_TRANSMITTED_FIELDS[task],
                    "synthetic_benchmark": True, "restricted_content": False,
                }
                transmissions.append(TransmissionRecord(
                    transmission_id=stable_sha256(transmission_payload), **transmission_payload,
                ))
                estimate_payload = {
                    "batch_id": batch_id, "position": position, "record_id": case_id,
                    "user_id": user_id, "split": split, "task": task,
                    "model": str(answer_model["requested_model"]),
                    "exact_runtime_text_tokens": exact_runtime,
                    "expected_memory_tokens": int(expected_memory[baseline]),
                    "framing_reserve_tokens": reserve,
                    "expected_input_tokens": expected_input,
                    "maximum_input_tokens": maximum_input,
                    "expected_output_tokens": expected_output,
                    "maximum_output_tokens": maximum_output,
                    "expected_cost_usd": calculate_cost(expected_input, expected_output, str(answer_model["input_usd_per_million_tokens"]), str(answer_model["output_usd_per_million_tokens"])),
                    "maximum_cost_usd": calculate_cost(maximum_input, maximum_output, str(answer_model["input_usd_per_million_tokens"]), str(answer_model["output_usd_per_million_tokens"])),
                }
                estimate = TokenEstimate(estimate_id=stable_sha256(estimate_payload), **estimate_payload)
                estimates.append(estimate)
                batch_members[batch_id].append(estimate)
    if batch_names != ["extraction_all_sources", *[f"{baseline}:{task}" for baseline in BASELINE_ORDER for task in TASK_ORDER]]:
        raise FrozenPreflightError("batch order changed")
    batches = tuple(_batch_record(index, name, batch_members[name], config) for index, name in enumerate(batch_members, 1))
    if len(transmissions) != 4660 or len(estimates) != 4660 or len(batches) != 25:
        raise FrozenPreflightError("preflight plan counts changed")
    return tuple(transmissions), tuple(estimates), batches


def _batch_record(position: int, batch_id: str, items: list[TokenEstimate], config: Mapping[str, object]) -> BatchPlan:
    extraction = position == 1
    baseline = None if extraction else batch_id.split("_")[2]
    task = "extraction" if extraction else batch_id.rsplit("_", 1)[1]
    model_policy = _mapping(config["extraction_model" if extraction else "answer_model"], "model policy")
    expected = money(sum((Decimal(item.expected_cost_usd) for item in items), Decimal(0)))
    maximum = money(sum((Decimal(item.maximum_cost_usd) for item in items), Decimal(0)))
    return BatchPlan(
        batch_id=batch_id, position=position, stage="extraction" if extraction else "answer",
        baseline_id=baseline, task=task, model=str(model_policy["requested_model"]),
        request_count=len(items), maximum_retry_requests=0,
        input_token_estimate=sum(item.expected_input_tokens for item in items),
        input_token_maximum=sum(item.maximum_input_tokens for item in items),
        expected_output_tokens=sum(item.expected_output_tokens for item in items),
        maximum_output_tokens=sum(item.maximum_output_tokens for item in items),
        expected_cost_usd=expected, maximum_cost_usd=maximum,
        output_directory=f"results/evaluation/frozen-run-v1/batches/{batch_id}",
        checkpoint_policy="after_each_success_no_successful_replay",
        provider_execution_authorized=False,
    )


def _validate_config(config: Mapping[str, object]) -> None:
    required = {
        "approval_policy", "answer_model", "answer_token_policy", "batch_order",
        "comparison_config_path", "comparison_config_sha256", "comparison_manifest_path",
        "comparison_manifest_sha256", "comparison_prompt_path", "comparison_prompt_sha256",
        "extraction_model", "guidance_sha256", "guidance_version",
        "historical_openai_spend_usd", "maximum_retry_requests", "predicate_registry_path",
        "predicate_registry_sha256", "predicate_registry_version", "preflight_version", "pricing",
        "runtime_input_paths", "scaled_dataset_sha256", "scaled_manifest_path",
        "scaled_manifest_sha256", "schema_version", "starting_commit", "transmission_policy",
    }
    if set(config) != required:
        raise FrozenPreflightError("preflight config fields changed")
    if config["preflight_version"] != PREFLIGHT_VERSION or config["schema_version"] != SCHEMA_VERSION:
        raise FrozenPreflightError("preflight config version changed")
    if config["starting_commit"] != STARTING_COMMIT or config["guidance_sha256"] != GUIDANCE_SHA256:
        raise FrozenPreflightError("preflight authority changed")
    if config["maximum_retry_requests"] != 0 or config["historical_openai_spend_usd"] != "0.2314404":
        raise FrozenPreflightError("retry or historical spend changed")
    approval = _mapping(config["approval_policy"], "approval policy")
    if approval != {
        "credential_reuse_approved": True, "data_transmission_approved": False,
        "paid_execution_approved": False, "provider_execution_authorized": False,
    }:
        raise FrozenPreflightError("approval policy changed")
    transmission = _mapping(config["transmission_policy"], "transmission policy")
    if transmission.get("gold_opened") or transmission.get("oracle_opened") or transmission.get("review_queue_opened"):
        raise FrozenPreflightError("scorer-only input boundary changed")


def _validate_runtime(users, sources, qa, summaries, interactive) -> None:
    if (len(users), len(sources), len(qa), len(summaries), len(interactive)) != (10, 100, 500, 50, 20):
        raise FrozenPreflightError("runtime input counts changed")
    if any(set(row) != USER_FIELDS for row in users):
        raise FrozenPreflightError("runtime user fields changed")
    if any(set(row) != SOURCE_FIELDS for row in sources):
        raise FrozenPreflightError("runtime source fields changed")
    for task, runtime_task, rows in (("qa", "qa", qa), ("summary", "summarization", summaries), ("interactive", "interactive", interactive)):
        if any(set(row) != CASE_FIELDS[task] or row.get("task") != runtime_task for row in rows):
            raise FrozenPreflightError(f"runtime {task} fields changed")
    user_ids = [str(row["user_id"]) for row in users]
    if user_ids != [f"user_{number:03d}" for number in range(1, 11)]:
        raise FrozenPreflightError("runtime user order changed")
    if len({str(row["source_id"]) for row in sources}) != 100:
        raise FrozenPreflightError("runtime source IDs changed")
    all_cases = [*qa, *summaries, *interactive]
    if len({str(row["case_id"]) for row in all_cases}) != 570:
        raise FrozenPreflightError("runtime case IDs changed")
    if sum(row["split"] == "development" for row in all_cases) != 114:
        raise FrozenPreflightError("development case count changed")
    for row in sources:
        metadata = row["metadata"]
        if not isinstance(metadata, dict) or any(token in str(key).lower() for key in metadata for token in ("sensitive", "restricted", "secret")):
            raise FrozenPreflightError("runtime source sensitivity metadata changed")


def _runtime_hashes(manifest: Mapping[str, object]) -> Mapping[str, str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise FrozenPreflightError("scaled manifest files missing")
    result = {}
    for item in files:
        if isinstance(item, dict) and item.get("layer") == "runtime" and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
            result[item["path"]] = item["sha256"]
    if set(result) != {path.as_posix() for path in RUNTIME_INPUTS}:
        raise FrozenPreflightError("scaled runtime input set changed")
    return result


def _preflight_record(config: Mapping[str, object], totals: Mapping[str, str]) -> Mapping[str, object]:
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "status": "awaiting_transmission_and_spend_approval",
        "credential_reuse_approved": True,
        "data_transmission_approved": False,
        "paid_execution_approved": False,
        "provider_execution_authorized": False,
        "provider_request_count": 0,
        "batch_count": 25,
        "planned_request_count": 4660,
        "maximum_retry_requests": 0,
        "source_count": 100,
        "case_count": 570,
        "frozen_test_case_count": 456,
        "expected_incremental_cost_usd": totals["expected"],
        "maximum_incremental_cost_usd": totals["maximum"],
        "historical_openai_spend_usd": "0.2314404",
        "projected_expected_cumulative_spend_usd": totals["cumulative_expected"],
        "projected_maximum_cumulative_spend_usd": totals["cumulative_maximum"],
        "pricing": config["pricing"],
        "runtime_rows_opened_locally_for_preflight": True,
        "runtime_content_logged": False,
        "gold_opened": False,
        "oracle_opened": False,
        "review_queue_opened": False,
        "step_10_3_started": False,
    }


def _totals(batches: Iterable[BatchPlan]) -> Mapping[str, str]:
    batches = tuple(batches)
    expected = sum((Decimal(item.expected_cost_usd) for item in batches), Decimal(0))
    maximum = sum((Decimal(item.maximum_cost_usd) for item in batches), Decimal(0))
    historical = Decimal("0.2314404")
    return {
        "expected": money(expected), "maximum": money(maximum),
        "cumulative_expected": money(historical + expected),
        "cumulative_maximum": money(historical + maximum),
    }


def _approval_request(batches: Iterable[BatchPlan], totals: Mapping[str, str]) -> str:
    rows = [
        "| Batch | Requests | Model | Expected | Maximum |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    rows.extend(
        f"| {item.batch_id} | {item.request_count} | `{item.model}` | ${item.expected_cost_usd} | ${item.maximum_cost_usd} |"
        for item in batches
    )
    return "\n".join([
        "# Frozen-run approval request", "",
        "Step 10.2 is ready, but no model request has been sent.", "",
        "The run has 25 resumable batches: one extraction batch for 100 synthetic sources and 24 answer batches for B0-B7 across QA, summary, and interactive tasks. It plans 4,660 requests with no retries.", "",
        "The existing local `OPENAI_API_KEY` may be reused. Its value was not read into an artifact and will not be logged or committed.", "",
        "## Data that would leave this machine", "",
        "The extraction batch would send synthetic source text, timestamps, participant and speaker IDs, source metadata, and user display names. Answer batches would send the synthetic case prompt, `as_of`, user and case IDs, and only the memory context allowed for that baseline. Gold, oracle records, review queues, credentials, and scorer-only fields are excluded.", "",
        "Users are `user_001` through `user_010`. The exact 100 source IDs and 570 case IDs are frozen in `data/evaluation/frozen-preflight-v1/runtime/transmission-plan.jsonl`.", "",
        "## Models and settings", "",
        "Extraction uses `gpt-4.1-mini-2025-04-14` with structured JSON, temperature 0, `store=false`, and a 1,200-token output limit. Answers use `gpt-4.1-2025-04-14` with JSON output, temperature 0, `store=false`, and a 1,000-token output limit.", "",
        "The estimate uses standard uncached prices checked on 2026-08-10: GPT-4.1 input $2.00/M and output $8.00/M; GPT-4.1 mini input $0.40/M and output $1.60/M. Cached-input discounts are not assumed.", "",
        "## Batch costs", "", *rows, "",
        f"Expected incremental cost: ${totals['expected']}",
        f"Hard maximum incremental cost: ${totals['maximum']}",
        f"Historical spend: $0.2314404",
        f"Projected expected cumulative spend: ${totals['cumulative_expected']}",
        f"Projected hard maximum cumulative spend: ${totals['cumulative_maximum']}", "",
        "Each successful response is checkpointed immediately. A successful case is never replayed. Provider failures are preserved, and this plan allows no automatic retry.", "",
        "## Approval needed", "",
        "Please approve both remaining facts before Step 10.3 starts:", "",
        "1. Send the listed synthetic runtime fields to OpenAI.",
        f"2. Allow paid execution up to the ${totals['maximum']} incremental hard cap (${totals['cumulative_maximum']} cumulative including historical spend).", "",
        "Credential reuse is already approved. A provider call will not be made until the transmission and spend approvals are explicit.", "",
    ])


def _findings(totals: Mapping[str, str]) -> str:
    return "\n".join([
        "# Step 10.2 findings", "",
        "The deterministic preflight covers 100 synthetic sources and 570 cases across all eight baselines. It creates 25 resumable batches and plans 4,660 requests with no retries.", "",
        f"The expected incremental cost is ${totals['expected']}. The hard maximum is ${totals['maximum']}. These are estimates, not provider usage.", "",
        "The local runtime rows were read only to validate schemas, list opaque IDs, and count tokens. Their text was not logged. Gold, oracle, review queues, predictions, and credentials were not opened by the builder.", "",
        "The existing key can be reused, but data transmission and paid execution are still unapproved. Step 10.3 has not started.", "",
    ])


def _case_runtime_text(task: str, record: Mapping[str, object]) -> str:
    fields = {
        "qa": ("case_id", "user_id", "as_of", "question"),
        "summary": ("case_id", "user_id", "as_of", "instruction"),
        "interactive": ("case_id", "user_id", "as_of", "scenario", "initial_user_message"),
    }[task]
    return json.dumps({field: record[field] for field in fields}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsonl(raw: bytes, location: str) -> tuple[Mapping[str, object], ...]:
    return tuple(parse_json_bytes(line, location=f"{location}:{index}") for index, line in enumerate(raw.splitlines(), 1) if line)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise FrozenPreflightError(f"{label} must be an array")
    return value


def _list_of_mappings(value: object, label: str) -> list[Mapping[str, object]]:
    values = _list(value, label)
    if not all(isinstance(item, dict) for item in values):
        raise FrozenPreflightError(f"{label} must contain objects")
    return values  # type: ignore[return-value]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FrozenPreflightError(f"{label} must be an object")
    return value


def _empty_or_absent(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FrozenPreflightError(f"output directory is not empty: {path}")


def _artifact_hashes(data: Path, result: Path) -> Mapping[str, str]:
    paths = {
        "data_manifest_sha256": data / "manifest.json",
        "runtime_manifest_sha256": data / "runtime/manifest.json",
        "transmission_plan_sha256": data / "runtime/transmission-plan.jsonl",
        **{name.replace(".", "_") + "_sha256": result / name for name in (*RESULT_ARTIFACTS, "manifest.json")},
    }
    if any(not path.is_file() for path in paths.values()):
        raise FrozenPreflightError("preflight artifact is missing")
    return {name: _sha_file(path) for name, path in paths.items()}


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def RUNTIME_INPUTS_ABSOLUTE(root: Path) -> tuple[Path, ...]:
    return tuple(root / path for path in RUNTIME_INPUTS)
