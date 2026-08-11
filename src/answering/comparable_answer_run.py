"""Freeze the comparable Step 8.2 answers without calling a provider."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .answer_contracts import BASELINE_ORDER, MemoryAnswer
from .answer_evaluation import verify_memory_answer_release
from .answer_input import load_answer_package_views
from .answer_run_contracts import (
    AVAILABLE_BASELINES,
    FINAL_RELEASE,
    MODEL,
    PROMPT_SHA256,
    PROMPT_VERSION,
    RUN_VERSION,
    RUNTIME_RELEASE,
    SCORER_VERSION,
    AnswerRunError,
    AnswerRuntimeCheckpoint,
    ComparableAnswerRunConfig,
    NoCallPreflight,
)
from .contracts import canonical_json_bytes
from .evaluation import verify_evidence_package_release
from .memory_answer import memory_answer_from_mapping


GUIDANCE_VERSION = "step-8.3-guidance-v1"
GUIDANCE_SHA256 = "64138cb9675729b340c1a471dcf698302609a35c08d4d1f36fe3956e3f596737"
STARTING_COMMIT = "9f7625455abafb85b513b8d30a53d793580160ce"
CONFIG_PATH = Path("configs/answering/comparable_answer_run_v1.json")
RUNTIME_ROOT = Path("results/answering/memory-answer-quality-development-runtime-v1")
STEP81_ROOT = Path("results/answering/evidence-package-development-v1")
STEP82_ROOT = Path("results/answering/memory-answer-contract-development-v1")
STEP82_DATASET = Path("data/answering/memory-answer-contract-development-v1/manifest.json")
STEP81_MANIFEST_SHA256 = "8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213"
STEP81_PACKAGES_SHA256 = "bb57898bea51417b2ecad1252b451669ae2c748033c9cef88360f16a825f8186"
STEP82_DATASET_SHA256 = "015e0d5e16f9870e04728f3668ee1f21f8c1b1a9a383f274c3c6bcb23c6f455e"
STEP82_MANIFEST_SHA256 = "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841"
STEP82_ANSWERS_SHA256 = "d83fcac2eed3a4b2493c575cc49f593657669a690b9cfa61307afc8890940ba4"
STEP82_CHECKS_SHA256 = "3642da6ae21165b1c2ddf6c4e64773e423615aa3a85b18834c8682ba26252164"
STEP82_RUN_SHA256 = "bcbe9eef40c0b126e106c7b88c61e2a5c74db6df5fd2be54cd1b8183cb1e80d2"
STEP82_FINDINGS_SHA256 = "63aefda40e96341de9e62181db8397be70aaa152d45a78a70279389a5726bd55"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
B1_CONFIG_SHA256 = "9ac8881719153f09369e195e1c81eb55cc5f2f50e591f87b4fe2664c16bf6d76"
RUNTIME_IMPLEMENTATION_PATHS = (
    "configs/answering/comparable_answer_run_v1.json",
    "src/answering/answer_run_contracts.py",
    "src/answering/comparable_answer_run.py",
)


def load_comparable_answer_run_config(
    path: str | Path = CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> tuple[ComparableAnswerRunConfig, str]:
    root = Path(repo_root).resolve()
    target = _resolve(root, path)
    value = _read_object(target)
    expected = {
        "available_baselines", "configured_resolved_model", "deferred_baselines",
        "expected_baseline_case_counts", "expected_prediction_count", "final_release",
        "generation_settings", "hard_maximum_run_cost_usd",
        "historical_openai_spend_usd", "input_answer_version",
        "input_package_version", "metric_names", "null_reasons", "prompt_sha256",
        "prompt_version", "provider_returned_model", "requested_model", "run_version",
        "runtime_release", "scorer_version",
    }
    if set(value) != expected:
        raise AnswerRunError("comparable answer configuration fields changed")
    config = ComparableAnswerRunConfig(
        run_version=value["run_version"],
        runtime_release=value["runtime_release"],
        scorer_version=value["scorer_version"],
        final_release=value["final_release"],
        input_answer_version=value["input_answer_version"],
        input_package_version=value["input_package_version"],
        prompt_version=value["prompt_version"],
        prompt_sha256=value["prompt_sha256"],
        requested_model=value["requested_model"],
        configured_resolved_model=value["configured_resolved_model"],
        provider_returned_model=value["provider_returned_model"],
        generation_settings=_mapping(value["generation_settings"]),
        available_baselines=tuple(value["available_baselines"]),
        deferred_baselines=tuple(value["deferred_baselines"]),
        expected_prediction_count=value["expected_prediction_count"],
        expected_baseline_case_counts=_mapping(value["expected_baseline_case_counts"]),
        metric_names=tuple(value["metric_names"]),
        null_reasons=_mapping(value["null_reasons"]),
        hard_maximum_run_cost_usd=value["hard_maximum_run_cost_usd"],
        historical_openai_spend_usd=value["historical_openai_spend_usd"],
    )
    _verify_b1_settings(root, config)
    return config, _sha(target)


def build_no_call_preflight(
    *, package_count: int = 24, eligible_provider_case_count: int = 0
) -> NoCallPreflight:
    return NoCallPreflight(
        RUN_VERSION,
        (),
        (),
        (),
        (),
        0,
        0,
        0,
        0,
        0,
        "0",
        "0",
        "0.2314404",
        "0.2314404",
        None,
        False,
        eligible_provider_case_count,
        package_count,
    )


def freeze_answer_predictions(
    output_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> AnswerRuntimeCheckpoint:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    _, config_sha256 = load_comparable_answer_run_config(repo_root=root)
    views, answers, answer_bytes = _validated_inputs(root)
    eligible = sum(view.answer_allowed for view in views)
    preflight = build_no_call_preflight(
        package_count=len(views), eligible_provider_case_count=eligible
    )
    preflight_bytes = canonical_json_bytes(preflight)
    output.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output / "preflight.json", preflight_bytes)
    _write_exclusive(output / "predictions.jsonl", answer_bytes)
    _write_exclusive(output / "failures.jsonl", b"")
    checkpoint = _checkpoint(
        root,
        config_sha256,
        preflight_bytes,
        answer_bytes,
        len(views),
        answers,
    )
    _write_exclusive(output / "checkpoint_manifest.json", canonical_json_bytes(checkpoint))
    verify_answer_runtime_checkpoint(output, repo_root=root)
    return checkpoint


def verify_answer_runtime_checkpoint(
    output_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> AnswerRuntimeCheckpoint:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _, config_sha256 = load_comparable_answer_run_config(repo_root=root)
    views, answers, expected_predictions = _validated_inputs(root)
    preflight_bytes = (output / "preflight.json").read_bytes()
    predictions = (output / "predictions.jsonl").read_bytes()
    failures = (output / "failures.jsonl").read_bytes()
    if predictions != expected_predictions or failures:
        raise AnswerRunError("runtime predictions or failures changed")
    preflight = _preflight_from_mapping(json.loads(preflight_bytes))
    expected_preflight = build_no_call_preflight(
        package_count=len(views), eligible_provider_case_count=sum(
            view.answer_allowed for view in views
        )
    )
    if preflight != expected_preflight or preflight_bytes != canonical_json_bytes(expected_preflight):
        raise AnswerRunError("runtime preflight changed")
    checkpoint_bytes = (output / "checkpoint_manifest.json").read_bytes()
    checkpoint = _checkpoint_from_mapping(json.loads(checkpoint_bytes))
    expected_checkpoint = _checkpoint(
        root, config_sha256, preflight_bytes, predictions, len(views), answers
    )
    if checkpoint != expected_checkpoint or checkpoint_bytes != canonical_json_bytes(expected_checkpoint):
        raise AnswerRunError("runtime checkpoint changed")
    return checkpoint


def _validated_inputs(root: Path) -> tuple[tuple[object, ...], tuple[MemoryAnswer, ...], bytes]:
    verify_evidence_package_release(STEP81_ROOT, repo_root=root)
    verify_memory_answer_release(STEP82_ROOT, repo_root=root)
    expected_hashes = {
        STEP81_ROOT / "manifest.json": STEP81_MANIFEST_SHA256,
        STEP81_ROOT / "packages.jsonl": STEP81_PACKAGES_SHA256,
        STEP82_DATASET: STEP82_DATASET_SHA256,
        STEP82_ROOT / "manifest.json": STEP82_MANIFEST_SHA256,
        STEP82_ROOT / "answers.jsonl": STEP82_ANSWERS_SHA256,
        STEP82_ROOT / "checks.json": STEP82_CHECKS_SHA256,
        STEP82_ROOT / "run.json": STEP82_RUN_SHA256,
        STEP82_ROOT / "findings.md": STEP82_FINDINGS_SHA256,
        STEP82_ROOT / "failures.jsonl": EMPTY_SHA256,
    }
    if any(_sha(root / path) != digest for path, digest in expected_hashes.items()):
        raise AnswerRunError("input release hash changed")
    views = tuple(load_answer_package_views(root))
    answer_bytes = (root / STEP82_ROOT / "answers.jsonl").read_bytes()
    answers = tuple(
        memory_answer_from_mapping(value)
        for value in _read_jsonl_bytes(answer_bytes)
    )
    if len(views) != 24 or len(answers) != 24:
        raise AnswerRunError("input answer accounting changed")
    view_by_package = {view.package_id: view for view in views}
    if len(view_by_package) != len(views):
        raise AnswerRunError("input package identity is duplicated")
    for answer in answers:
        view = view_by_package.get(answer.package_id)
        if view is None or (
            answer.package_sha256,
            answer.user_id,
            answer.query_id,
            answer.baseline_id,
            answer.execution_id,
            answer.plan_id,
            answer.snapshot_run_id,
            answer.as_of,
            answer.requested_valid_time,
            answer.structural_blockers,
        ) != (
            view.package_sha256,
            view.user_id,
            view.query_id,
            view.baseline_id,
            view.execution_id,
            view.plan_id,
            view.snapshot_run_id,
            view.as_of,
            view.requested_valid_time,
            view.structural_blockers,
        ):
            raise AnswerRunError("answer and package identities changed")
        if (
            view.answer_allowed
            or answer.status != "abstained"
            or answer.structural_blockers != ("no_promoted_claims",)
            or answer.statements
            or answer.requested_model is not None
            or answer.resolved_model is not None
        ):
            raise AnswerRunError("provider eligibility is not zero")
    expected_order = tuple(sorted(
        answers, key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id])
    ))
    if answers != expected_order:
        raise AnswerRunError("input answers are not canonically ordered")
    counts = {baseline: sum(item.baseline_id == baseline for item in answers) for baseline in AVAILABLE_BASELINES}
    if counts != {"B2": 8, "B3": 8, "B4": 8}:
        raise AnswerRunError("input baseline accounting changed")
    return views, answers, answer_bytes


def _checkpoint(
    root: Path,
    config_sha256: str,
    preflight_bytes: bytes,
    answer_bytes: bytes,
    package_count: int,
    answers: Sequence[MemoryAnswer],
) -> AnswerRuntimeCheckpoint:
    return AnswerRuntimeCheckpoint(
        RUN_VERSION,
        RUNTIME_RELEASE,
        STARTING_COMMIT,
        GUIDANCE_VERSION,
        GUIDANCE_SHA256,
        config_sha256,
        {
            "step8_1_manifest": STEP81_MANIFEST_SHA256,
            "step8_1_packages": STEP81_PACKAGES_SHA256,
            "step8_2_dataset_manifest": STEP82_DATASET_SHA256,
            "step8_2_result_manifest": STEP82_MANIFEST_SHA256,
            "step8_2_answers": STEP82_ANSWERS_SHA256,
            "b1_configuration": B1_CONFIG_SHA256,
        },
        {path: _sha(root / path) for path in RUNTIME_IMPLEMENTATION_PATHS},
        hashlib.sha256(preflight_bytes).hexdigest(),
        hashlib.sha256(answer_bytes).hexdigest(),
        EMPTY_SHA256,
        package_count,
        len(answers),
        0,
        {baseline: sum(item.baseline_id == baseline for item in answers) for baseline in AVAILABLE_BASELINES},
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        False,
    )


def _verify_b1_settings(root: Path, config: ComparableAnswerRunConfig) -> None:
    path = root / "configs/full_history_baseline_v1.json"
    if _sha(path) != B1_CONFIG_SHA256:
        raise AnswerRunError("B1 comparison configuration changed")
    value = _read_object(path)
    expected = {
        "provider": value.get("provider"),
        "api": _mapping(value.get("generation_settings")).get("api"),
        "temperature": value.get("temperature"),
        "max_output_tokens": _mapping(value.get("generation_settings")).get("max_output_tokens"),
        "store": _mapping(value.get("generation_settings")).get("store"),
        "text_format": _mapping(value.get("generation_settings")).get("text_format"),
    }
    if expected != dict(config.generation_settings) or (
        value.get("requested_model") != MODEL or value.get("resolved_model") != MODEL
    ):
        raise AnswerRunError("B1 comparison settings do not match")


def _preflight_from_mapping(value: object) -> NoCallPreflight:
    mapping = _mapping(value)
    if set(mapping) != set(asdict(build_no_call_preflight())):
        raise AnswerRunError("preflight fields changed")
    return NoCallPreflight(
        mapping["run_version"],
        tuple(mapping["runtime_cases_transmitted"]),
        tuple(mapping["users_transmitted"]),
        tuple(mapping["source_ids_transmitted"]),
        tuple(mapping["fields_transmitted"]),
        mapping["planned_requests"],
        mapping["maximum_retry_requests"],
        mapping["estimated_input_tokens"],
        mapping["expected_output_tokens"],
        mapping["hard_output_token_ceiling"],
        mapping["expected_run_cost_usd"],
        mapping["hard_maximum_run_cost_usd"],
        mapping["spend_already_used_usd"],
        mapping["new_cumulative_ceiling_usd"],
        mapping["provider_returned_model"],
        mapping["oracle_and_gold_transmitted"],
        mapping["eligible_provider_case_count"],
        mapping["package_count"],
    )


def _checkpoint_from_mapping(value: object) -> AnswerRuntimeCheckpoint:
    mapping = _mapping(value)
    expected = set(asdict(_placeholder_checkpoint()))
    if set(mapping) != expected:
        raise AnswerRunError("checkpoint fields changed")
    return AnswerRuntimeCheckpoint(
        mapping["run_version"], mapping["runtime_release"], mapping["starting_commit"],
        mapping["guidance_version"], mapping["guidance_sha256"], mapping["config_sha256"],
        _mapping(mapping["input_authorities"]), _mapping(mapping["implementation_hashes"]),
        mapping["preflight_sha256"], mapping["predictions_sha256"],
        mapping["failures_sha256"], mapping["package_count"], mapping["prediction_count"],
        mapping["eligible_provider_case_count"], _mapping(mapping["baseline_case_counts"]),
        mapping["provider_request_count"], mapping["retry_count"],
        mapping["input_token_count"], mapping["output_token_count"],
        mapping["incremental_cost_usd"], mapping["failure_count"],
        mapping["scorer_opened"], mapping["answer_gold_opened"],
    )


def _placeholder_checkpoint() -> AnswerRuntimeCheckpoint:
    return AnswerRuntimeCheckpoint(
        RUN_VERSION, RUNTIME_RELEASE, "0" * 40, "guidance", "0" * 64, "0" * 64,
        {"input": "0" * 64}, {"implementation": "0" * 64}, "0" * 64,
        STEP82_ANSWERS_SHA256, EMPTY_SHA256, 24, 24, 0,
        {"B2": 8, "B3": 8, "B4": 8}, 0, 0, 0, 0, 0, 0, False, False,
    )


def _read_jsonl_bytes(raw: bytes) -> list[Mapping[str, object]]:
    try:
        values = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerRunError("answer JSONL is invalid") from error
    if any(not isinstance(value, dict) for value in values):
        raise AnswerRunError("answer JSONL record is invalid")
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerRunError("JSON object is invalid") from error
    return _mapping(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnswerRunError("mapping is invalid")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise AnswerRunError("output directory must be empty")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
