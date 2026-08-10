"""Build and deeply verify the manifest-only Step 10.1 comparison freeze."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from .comparison_freeze_contracts import (
    BASELINE_ORDER,
    COMPARISON_VERSION,
    CONFIG_VERSION,
    DATASET_VERSION,
    RESULT_VERSION,
    SCHEMA_VERSION,
    TASK_ORDER,
    AuthorityBinding,
    BaselineDefinition,
    ComparisonFreezeChecks,
    ComparisonFreezeError,
    FrozenComparisonDefinition,
    ManifestFileBinding,
    PrerequisiteGap,
    PromptBinding,
    TaskDefinition,
    canonical_json_bytes,
    checks_from_mapping,
    definition_from_mapping,
    parse_json_bytes,
)


STARTING_COMMIT = "78ed4900fd9a7aecbd7ca8c70b5726b356a07ff4"
GUIDANCE_VERSION = "step-10.1-guidance-v1"
GUIDANCE_SHA256 = "fb8e38cf51393968183db1a40e1659bcffaac6cebe6733cfaf9bdbc3a0f1959a"
SCALED_MANIFEST_SHA256 = "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3"
SCALED_DATASET_SHA256 = "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61"
STEP9_4_MANIFEST_SHA256 = "e8aed9be4455a1dd8b33f390928bcec537e5c69e752a41a1a23265acb5e12fbd"
CONFIG_SHA256 = "6034a2cd6cd3a883af6ceb71dee7f92eba6eabff71d041bce5e82f9679c5c957"
PROMPT_SHA256 = "681935ec199e6f0a4f7cb37c971c0c28295293af9f792f1f41f6225ce5f468bf"
CONFIG_PATH = Path("configs/evaluation/frozen_comparison_v1.json")
PROMPT_PATH = Path("configs/evaluation/frozen_prompts_v1.json")
SCALED_MANIFEST_PATH = Path("data/scaled-v1/manifest.json")
DATA_ROOT = Path("data/evaluation/frozen-comparison-v1")
RESULT_ROOT = Path("results/evaluation/frozen-comparison-v1")
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    PROMPT_PATH,
    Path("src/evaluation/comparison_freeze_contracts.py"),
    Path("src/evaluation/comparison_freeze.py"),
)
RESULT_ARTIFACTS = ("checks.json", "run.json", "findings.md")
EXPECTED_CONFIG_FIELDS = {
    "comparison_version", "schema_version", "config_version", "dataset_version",
    "result_version", "starting_commit", "guidance_version", "guidance_sha256",
    "scaled_manifest_path", "scaled_manifest_sha256", "scaled_dataset_sha256",
    "baseline_order", "baselines", "task_order", "tasks", "development_users",
    "frozen_test_users", "model_policy", "prompt_contract_path", "prompt_bindings",
    "ordering_policy", "slice_order", "metric_groups", "null_reasons", "authorities",
    "prerequisite_gaps", "provider_request_count", "incremental_cost_usd",
    "historical_openai_spend_usd", "implementer_pilot_answer_reference_exposure",
    "implementer_pilot_answer_reference_used", "referenced_files_opened",
    "step_10_2_started",
}
EXPECTED_PROMPT_FIELDS = {"prompt_contract_version", "schema_version", "baseline_order", "tasks"}
EXPECTED_SCALED_FIELDS = {
    "manifest_version", "benchmark_version", "release_status", "dataset_sha256",
    "files", "counts", "capability_counts", "splits", "review",
}


Reader = Callable[[Path], bytes]


def build_frozen_comparison(
    *,
    repo_root: str | Path = ".",
    data_root: str | Path = DATA_ROOT,
    result_root: str | Path = RESULT_ROOT,
    reader: Reader | None = None,
) -> Mapping[str, str]:
    """Write one immutable definition release after reading the scaled manifest only."""
    root = Path(repo_root).resolve()
    read = reader or (lambda path: path.read_bytes())
    data_target = _output_path(root, data_root)
    result_target = _output_path(root, result_root)
    _require_empty(data_target)
    _require_empty(result_target)

    definition, input_hashes = _rebuild_definition(root, read)
    data_bytes = canonical_json_bytes(asdict(definition))
    checks = _checks(definition)
    checks_bytes = canonical_json_bytes(asdict(checks))
    run = {
        "comparison_version": COMPARISON_VERSION,
        "result_version": RESULT_VERSION,
        "mode": "manifest_only_definition_freeze",
        "scaled_manifest_only": True,
        "referenced_files_opened": False,
        "prediction_count": 0,
        "score_count": 0,
        "provider_request_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": "0.0000000",
        "step_10_2_started": False,
    }
    run_bytes = canonical_json_bytes(run)
    findings_bytes = _findings().encode()

    data_target.mkdir(parents=True, exist_ok=True)
    result_target.mkdir(parents=True, exist_ok=True)
    _exclusive_write(data_target / "manifest.json", data_bytes)
    outputs = {
        "checks.json": checks_bytes,
        "run.json": run_bytes,
        "findings.md": findings_bytes,
    }
    for name, raw in outputs.items():
        _exclusive_write(result_target / name, raw)

    implementation_hashes = {
        path.as_posix(): _sha(read(_resolve(root, path))) for path in IMPLEMENTATION_PATHS
    }
    artifact_hashes = {name: _sha(raw) for name, raw in outputs.items()}
    manifest = {
        "comparison_version": COMPARISON_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "dataset_version": DATASET_VERSION,
        "result_version": RESULT_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "inputs": input_hashes,
        "data_manifest_sha256": _sha(data_bytes),
        "implementation": implementation_hashes,
        "artifacts": artifact_hashes,
        "counts": asdict(checks),
        "baseline_order": list(BASELINE_ORDER),
        "task_order": list(TASK_ORDER),
        "referenced_files_opened": False,
        "provider_execution_authorized": False,
        "provider_request_count": 0,
        "incremental_cost_usd": "0.0000000",
        "historical_openai_spend_usd": "0.2314404",
        "prerequisite_status": "all_eight_scaled_runtime_and_prediction_releases_missing",
        "phase_9_complete": False,
        "full_step_9_3_roadmap_complete": False,
        "frozen_interactive_cases_deferred": 16,
        "step_10_2_started": False,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "limitations": [
            "definition_only_no_predictions_or_scores",
            "scaled_referenced_files_not_opened",
            "b0_b7_scaled_runtime_not_authorized",
            "no_provider_execution",
        ],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _exclusive_write(result_target / "manifest.json", manifest_bytes)
    verify_frozen_comparison(
        repo_root=root, data_root=data_target, result_root=result_target, reader=read,
    )
    return {
        "data_manifest_sha256": _sha(data_bytes),
        "checks_sha256": _sha(checks_bytes),
        "run_sha256": _sha(run_bytes),
        "findings_sha256": _sha(findings_bytes),
        "result_manifest_sha256": _sha(manifest_bytes),
    }


def verify_frozen_comparison(
    *,
    repo_root: str | Path = ".",
    data_root: str | Path = DATA_ROOT,
    result_root: str | Path = RESULT_ROOT,
    reader: Reader | None = None,
) -> Mapping[str, str]:
    """Deeply reconstruct every definition and result byte from explicit authorities."""
    root = Path(repo_root).resolve()
    read = reader or (lambda path: path.read_bytes())
    data_target = _output_path(root, data_root)
    result_target = _output_path(root, result_root)
    expected_definition, input_hashes = _rebuild_definition(root, read)
    expected_data = canonical_json_bytes(asdict(expected_definition))
    actual_data = read(data_target / "manifest.json")
    if actual_data != expected_data:
        raise ComparisonFreezeError("data manifest changed")
    definition_from_mapping(parse_json_bytes(actual_data, location="data manifest"))

    expected_checks = canonical_json_bytes(asdict(_checks(expected_definition)))
    actual_checks = read(result_target / "checks.json")
    if actual_checks != expected_checks:
        raise ComparisonFreezeError("checks changed")
    checks_from_mapping(parse_json_bytes(actual_checks, location="checks"))
    expected_run = canonical_json_bytes({
        "comparison_version": COMPARISON_VERSION,
        "result_version": RESULT_VERSION,
        "mode": "manifest_only_definition_freeze",
        "scaled_manifest_only": True,
        "referenced_files_opened": False,
        "prediction_count": 0,
        "score_count": 0,
        "provider_request_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": "0.0000000",
        "step_10_2_started": False,
    })
    actual_run = read(result_target / "run.json")
    if actual_run != expected_run:
        raise ComparisonFreezeError("run record changed")
    if read(result_target / "findings.md") != _findings().encode():
        raise ComparisonFreezeError("findings changed")

    actual_manifest_raw = read(result_target / "manifest.json")
    actual_manifest = parse_json_bytes(actual_manifest_raw, location="result manifest")
    implementation_hashes = {
        path.as_posix(): _sha(read(_resolve(root, path))) for path in IMPLEMENTATION_PATHS
    }
    expected_manifest = {
        "comparison_version": COMPARISON_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "dataset_version": DATASET_VERSION,
        "result_version": RESULT_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "inputs": input_hashes,
        "data_manifest_sha256": _sha(expected_data),
        "implementation": implementation_hashes,
        "artifacts": {
            "checks.json": _sha(expected_checks),
            "run.json": _sha(expected_run),
            "findings.md": _sha(_findings().encode()),
        },
        "counts": asdict(_checks(expected_definition)),
        "baseline_order": list(BASELINE_ORDER),
        "task_order": list(TASK_ORDER),
        "referenced_files_opened": False,
        "provider_execution_authorized": False,
        "provider_request_count": 0,
        "incremental_cost_usd": "0.0000000",
        "historical_openai_spend_usd": "0.2314404",
        "prerequisite_status": "all_eight_scaled_runtime_and_prediction_releases_missing",
        "phase_9_complete": False,
        "full_step_9_3_roadmap_complete": False,
        "frozen_interactive_cases_deferred": 16,
        "step_10_2_started": False,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "limitations": [
            "definition_only_no_predictions_or_scores",
            "scaled_referenced_files_not_opened",
            "b0_b7_scaled_runtime_not_authorized",
            "no_provider_execution",
        ],
    }
    if actual_manifest != expected_manifest or actual_manifest_raw != canonical_json_bytes(expected_manifest):
        raise ComparisonFreezeError("result manifest changed")
    return {
        "data_manifest_sha256": _sha(expected_data),
        "checks_sha256": _sha(expected_checks),
        "run_sha256": _sha(expected_run),
        "findings_sha256": _sha(_findings().encode()),
        "result_manifest_sha256": _sha(actual_manifest_raw),
    }


def _rebuild_definition(root: Path, read: Reader) -> tuple[FrozenComparisonDefinition, Mapping[str, str]]:
    config_raw = read(_resolve(root, CONFIG_PATH))
    if _sha(config_raw) != CONFIG_SHA256:
        raise ComparisonFreezeError("comparison config bytes changed")
    prompt_raw = read(_resolve(root, PROMPT_PATH))
    if _sha(prompt_raw) != PROMPT_SHA256:
        raise ComparisonFreezeError("prompt config bytes changed")
    scaled_raw = read(_resolve(root, SCALED_MANIFEST_PATH))
    config = parse_json_bytes(config_raw, location="comparison config")
    prompts = parse_json_bytes(prompt_raw, location="prompt config")
    scaled = parse_json_bytes(scaled_raw, location="scaled manifest")
    _validate_config(config)
    prompt_bindings = _validate_prompts(prompts, config)
    manifest_files, manifest_counts, capability_counts, development_users, frozen_users = _validate_scaled(scaled, scaled_raw)

    authorities = tuple(sorted(
        (AuthorityBinding(path=_string(item["path"]), sha256=_string(item["sha256"]))
         for item in _mapping_array(config["authorities"])),
        key=lambda item: item.path,
    ))
    for binding in authorities:
        if _sha(read(_resolve(root, binding.path))) != binding.sha256:
            raise ComparisonFreezeError(f"authority drift: {binding.path}")
    definition = FrozenComparisonDefinition(
        comparison_version=COMPARISON_VERSION,
        schema_version=SCHEMA_VERSION,
        config_version=CONFIG_VERSION,
        dataset_version=DATASET_VERSION,
        result_version=RESULT_VERSION,
        guidance_version=GUIDANCE_VERSION,
        guidance_sha256=GUIDANCE_SHA256,
        starting_commit=STARTING_COMMIT,
        scaled_manifest_path=SCALED_MANIFEST_PATH.as_posix(),
        scaled_manifest_sha256=SCALED_MANIFEST_SHA256,
        scaled_dataset_sha256=SCALED_DATASET_SHA256,
        referenced_files_opened=False,
        manifest_files=manifest_files,
        manifest_counts=manifest_counts,
        capability_counts=capability_counts,
        development_users=development_users,
        frozen_test_users=frozen_users,
        tasks=tuple(TaskDefinition(**item) for item in _mapping_array(config["tasks"])),
        baselines=tuple(BaselineDefinition(
            baseline_id=_string(item["baseline_id"]),
            memory_semantics=_string(item["memory_semantics"]),
            capabilities=_string_array(item["capabilities"]),
        ) for item in _mapping_array(config["baselines"])),
        prompt_bindings=prompt_bindings,
        model_policy=_mapping(config["model_policy"]),
        ordering_policy=_mapping(config["ordering_policy"]),
        slice_order=_string_array(config["slice_order"]),
        metric_groups=tuple(_mapping(item) for item in _array(config["metric_groups"])),
        null_reasons=_string_array(config["null_reasons"]),
        authorities=authorities,
        prerequisite_gaps=tuple(PrerequisiteGap(
            baseline_id=_string(item["baseline_id"]),
            status=_string(item["status"]),
            required_in=_string_array(item["required_in"]),
        ) for item in _mapping_array(config["prerequisite_gaps"])),
        provider_request_count=_integer(config["provider_request_count"]),
        incremental_cost_usd=_string(config["incremental_cost_usd"]),
        historical_openai_spend_usd=_string(config["historical_openai_spend_usd"]),
        implementer_pilot_answer_reference_exposure=_boolean(config["implementer_pilot_answer_reference_exposure"]),
        implementer_pilot_answer_reference_used=_boolean(config["implementer_pilot_answer_reference_used"]),
        step_10_2_started=_boolean(config["step_10_2_started"]),
    )
    return definition, {
        CONFIG_PATH.as_posix(): _sha(config_raw),
        PROMPT_PATH.as_posix(): _sha(prompt_raw),
        SCALED_MANIFEST_PATH.as_posix(): _sha(scaled_raw),
        "results/abstention/b7-evaluation-development-v1/manifest.json": STEP9_4_MANIFEST_SHA256,
    }


def _validate_config(config: Mapping[str, object]) -> None:
    if set(config) != EXPECTED_CONFIG_FIELDS:
        raise ComparisonFreezeError("comparison config fields changed")
    exact = {
        "comparison_version": COMPARISON_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "dataset_version": DATASET_VERSION,
        "result_version": RESULT_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "scaled_manifest_path": SCALED_MANIFEST_PATH.as_posix(),
        "scaled_manifest_sha256": SCALED_MANIFEST_SHA256,
        "scaled_dataset_sha256": SCALED_DATASET_SHA256,
        "baseline_order": list(BASELINE_ORDER),
        "task_order": list(TASK_ORDER),
        "development_users": ["user_001", "user_002"],
        "frozen_test_users": [f"user_{number:03d}" for number in range(3, 11)],
        "prompt_contract_path": PROMPT_PATH.as_posix(),
        "provider_request_count": 0,
        "incremental_cost_usd": "0.0000000",
        "historical_openai_spend_usd": "0.2314404",
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "referenced_files_opened": False,
        "step_10_2_started": False,
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ComparisonFreezeError(f"comparison config changed: {key}")
    model = _mapping(config["model_policy"])
    if model != {
        "provider": "OpenAI", "api": "responses",
        "requested_model": "gpt-4.1-2025-04-14",
        "resolved_model": "gpt-4.1-2025-04-14",
        "provider_returned_model": None, "temperature": 0.0,
        "max_output_tokens": 1000, "store": False,
        "text_format": "json_object", "provider_execution_authorized": False,
    }:
        raise ComparisonFreezeError("model policy changed")
    baselines = _mapping_array(config["baselines"])
    if tuple(_string(item["baseline_id"]) for item in baselines) != BASELINE_ORDER:
        raise ComparisonFreezeError("baseline order changed")
    gaps = _mapping_array(config["prerequisite_gaps"])
    if len(gaps) != 8 or any(_integer(config["provider_request_count"]) != 0 for _ in gaps):
        raise ComparisonFreezeError("prerequisite gaps are invalid")
    if not config["metric_groups"] or not config["null_reasons"]:
        raise ComparisonFreezeError("scorer policy is empty")


def _validate_prompts(prompts: Mapping[str, object], config: Mapping[str, object]) -> tuple[PromptBinding, ...]:
    if set(prompts) != EXPECTED_PROMPT_FIELDS:
        raise ComparisonFreezeError("prompt config fields changed")
    if prompts["prompt_contract_version"] != "frozen_comparison_prompts_v1" or prompts["schema_version"] != "frozen_prompt_schema_v1":
        raise ComparisonFreezeError("prompt config version changed")
    if prompts["baseline_order"] != list(BASELINE_ORDER):
        raise ComparisonFreezeError("prompt baseline order changed")
    tasks = _mapping_array(prompts["tasks"])
    config_bindings = {item["task"]: item["prompt_sha256"] for item in _mapping_array(config["prompt_bindings"])}
    bindings: list[PromptBinding] = []
    for expected_task, record in zip(TASK_ORDER, tasks, strict=True):
        expected_fields = {
            "task", "prompt_version", "prompt_sha256", "baseline_ids",
            "system_instruction", "input_fields", "output_fields",
        }
        if set(record) != expected_fields or record["task"] != expected_task:
            raise ComparisonFreezeError("prompt task fields or order changed")
        preimage = dict(record)
        recorded_hash = _string(preimage.pop("prompt_sha256"))
        computed_hash = _sha(canonical_json_bytes(preimage))
        if recorded_hash != computed_hash or config_bindings.get(expected_task) != computed_hash:
            raise ComparisonFreezeError("prompt hash changed")
        instruction = _string(record["system_instruction"])
        required = ("only the supplied context", "untrusted evidence", "as_of", "Abstain", "exact evidence IDs", "exact quote", "JSON only")
        if not all(fragment.lower() in instruction.lower() for fragment in required):
            raise ComparisonFreezeError("prompt safety contract changed")
        bindings.append(PromptBinding(
            task=expected_task,
            prompt_version=_string(record["prompt_version"]),
            prompt_sha256=computed_hash,
            baseline_ids=_string_array(record["baseline_ids"]),
        ))
    return tuple(bindings)


def _validate_scaled(
    scaled: Mapping[str, object], raw: bytes,
) -> tuple[tuple[ManifestFileBinding, ...], Mapping[str, object], Mapping[str, object], tuple[str, ...], tuple[str, ...]]:
    if _sha(raw) != SCALED_MANIFEST_SHA256 or set(scaled) != EXPECTED_SCALED_FIELDS:
        raise ComparisonFreezeError("scaled manifest changed")
    if (
        scaled["manifest_version"], scaled["benchmark_version"], scaled["release_status"],
        scaled["dataset_sha256"],
    ) != ("1", "scaled_v1", "frozen", SCALED_DATASET_SHA256):
        raise ComparisonFreezeError("scaled dataset identity changed")
    file_records = _mapping_array(scaled["files"])
    if len(file_records) != 27:
        raise ComparisonFreezeError("scaled manifest file accounting changed")
    files = tuple(ManifestFileBinding(
        path=_string(item["path"]), layer=_string(item["layer"]),
        sha256=_string(item["sha256"]), records=_integer(item["records"]),
    ) for item in file_records)
    counts = _mapping(scaled["counts"])
    if (counts.get("users"), counts.get("sources"), counts.get("qa"), counts.get("summaries"), counts.get("interactive_scenarios")) != (10, 100, 500, 50, 20):
        raise ComparisonFreezeError("scaled counts changed")
    splits = _mapping(scaled["splits"])
    development = _mapping(splits.get("development"))
    frozen = _mapping(splits.get("test"))
    if splits.get("strategy") != "whole_user" or frozen.get("frozen_for_tuning") is not True:
        raise ComparisonFreezeError("scaled split policy changed")
    development_users = _string_array(development["user_ids"])
    frozen_users = _string_array(frozen["user_ids"])
    if development_users != ("user_001", "user_002") or frozen_users != tuple(f"user_{number:03d}" for number in range(3, 11)):
        raise ComparisonFreezeError("whole-user split changed")
    if (development.get("qa"), development.get("summaries"), development.get("interactive_scenarios")) != (100, 10, 4):
        raise ComparisonFreezeError("development split counts changed")
    if (frozen.get("qa"), frozen.get("summaries"), frozen.get("interactive_scenarios")) != (400, 40, 16):
        raise ComparisonFreezeError("frozen split counts changed")
    return files, counts, _mapping(scaled["capability_counts"]), development_users, frozen_users


def _checks(definition: FrozenComparisonDefinition) -> ComparisonFreezeChecks:
    metric_count = sum(len(_array(group["metrics"])) for group in definition.metric_groups)
    return ComparisonFreezeChecks(
        comparison_version=COMPARISON_VERSION,
        manifest_only=True,
        referenced_files_opened=False,
        manifest_file_count=len(definition.manifest_files),
        baseline_count=len(definition.baselines),
        task_count=len(definition.tasks),
        metric_count=metric_count,
        prerequisite_gap_count=len(definition.prerequisite_gaps),
        prediction_count=0,
        score_count=0,
        provider_request_count=0,
        step_10_2_started=False,
    )


def _findings() -> str:
    return """# Step 10.1 comparison freeze\n\nThis release freezes the B0-B7 comparison definition. It does not contain predictions, scores, or model output.\n\nOnly `data/scaled-v1/manifest.json` was opened. None of its referenced runtime, gold, oracle, review, schema, or prediction files was opened or hashed. The prompt contract contains no case or source content.\n\nAll eight scaled runtime adapters and prediction releases are still prerequisites for Steps 10.2 and 10.3; their absence is not reported as a zero result. Step 10.2 has not started. Phase 9 and the literal 20-case Step 9.3 roadmap remain incomplete, with 16 frozen interactive cases deferred.\n\nNo provider request was made. Incremental cost is $0. Historical OpenAI spend remains $0.2314404. The implementer had prior pilot answer-reference exposure, and none of it was used for this definition.\n"""


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ComparisonFreezeError(f"output directory is not empty: {path}")


def _exclusive_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def _resolve(root: Path, path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ComparisonFreezeError("path escapes repository root")
    return target


def _output_path(root: Path, path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    return target.resolve()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ComparisonFreezeError("mapping required")
    return value


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ComparisonFreezeError("array required")
    return value


def _mapping_array(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _array(value))


def _string_array(value: object) -> tuple[str, ...]:
    values = _array(value)
    if not all(isinstance(item, str) for item in values):
        raise ComparisonFreezeError("string array required")
    return tuple(values)  # type: ignore[return-value]


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ComparisonFreezeError("string required")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ComparisonFreezeError("integer required")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ComparisonFreezeError("boolean required")
    return value
