"""Frozen identity and zero-call manifest for the corrected Qwen v2 series."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .qwen_series import write_immutable_json


SERIES_ID = "qwen35-27b-fp8-v2"
SCHEMA_VERSION = "qwen_series_config_v2"
CONFIG_PATH = Path("configs/evaluation/qwen35_27b_fp8_v2.json")
SERIES_MANIFEST_PATH = Path("results/evaluation/qwen35-27b-fp8-v2/series.json")
BASELINES = tuple(f"B{index}" for index in range(8))
EXPECTED_PROVIDER_COUNTS = {
    "stage_1_compatibility": 12,
    "development_extraction": 20,
    "development_answers_b0_b6": 798,
    "development_judge": 112,
    "frozen_extraction": 80,
    "frozen_answers_b0_b6": 3192,
    "frozen_judge": 448,
    "planned_including_compatibility": 4662,
    "maximum_including_transport_retries": 4687,
}
EXPECTED_LOGICAL_COUNTS = {"development": 912, "frozen_test": 3648, "total": 4560}


class QwenV2ContractError(ValueError):
    """Raised when the frozen Qwen v2 definition changes or is inconsistent."""


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_qwen_v2_config(repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    path = root / CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QwenV2ContractError("Qwen v2 configuration must be an object")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("series_id") != SERIES_ID:
        raise QwenV2ContractError("Qwen v2 series identity changed")
    if value.get("status") != "frozen_not_run":
        raise QwenV2ContractError("Qwen v2 freeze must not claim execution")

    model = _mapping(value, "model")
    if model != {
        "context_length": 16384,
        "enable_thinking": False,
        "hugging_face_id": "Qwen/Qwen3.5-27B-FP8",
        "model_alias": SERIES_ID,
        "revision": "97f5941bf617e31c5e237364a8602ce3f03a551a",
        "task": "text",
    }:
        raise QwenV2ContractError("model contract changed")

    runtime = _mapping(value, "runtime")
    required_runtime = {
        "client_concurrency": 8,
        "generation_config": "vllm",
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 16,
        "storage_gb": 100,
        "template": "pytorch",
        "tensor_parallel_size": 1,
        "vllm_git_revision": "65b7662d3fcb773afaf751ab29ac6960a0cf011d",
    }
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected:
            raise QwenV2ContractError(f"runtime contract changed: {key}")
    resources = runtime.get("resources")
    if resources != [
        {"gpu": "H100-80GB", "maximum_spot_rate_inr_per_hour": 133.33, "priority": 1, "region": "IN2", "spot": True},
        {"gpu": "RTX-PRO6000-96GB", "maximum_spot_rate_inr_per_hour": 100.0, "priority": 2, "region": "IN1", "spot": True},
    ]:
        raise QwenV2ContractError("approved resource policy changed")

    baselines = value.get("baselines")
    if not isinstance(baselines, list) or tuple(row.get("baseline_id") for row in baselines if isinstance(row, dict)) != BASELINES:
        raise QwenV2ContractError("baseline order changed")
    provider_flags = {str(row["baseline_id"]): row.get("provider_call") for row in baselines}
    if provider_flags != {**{baseline: True for baseline in BASELINES[:-1]}, "B7": False}:
        raise QwenV2ContractError("B7 must reuse B6 without a provider call")
    b7 = baselines[-1]
    if b7.get("context") != "exact_b6_context_and_response_plus_deterministic_answerability":
        raise QwenV2ContractError("B7 identity contract changed")

    workload = _mapping(value, "workload")
    if workload.get("provider_requests") != EXPECTED_PROVIDER_COUNTS:
        raise QwenV2ContractError("provider request accounting changed")
    if workload.get("logical_predictions") != EXPECTED_LOGICAL_COUNTS:
        raise QwenV2ContractError("logical prediction accounting changed")
    _validate_budget(value)
    _validate_boundaries(value)
    _verify_authorities(root, value)
    return value


def build_series_manifest(repo_root: Path, output: Path | None = None) -> Path:
    root = repo_root.resolve()
    config = load_qwen_v2_config(root)
    destination = output or root / SERIES_MANIFEST_PATH
    if not destination.is_absolute():
        destination = root / destination
    authorities = {
        str(row["path"]): str(row["sha256"])
        for row in config["authority_bindings"]  # type: ignore[index]
    }
    write_immutable_json(destination, {
        "authority_bindings": authorities,
        "config_path": CONFIG_PATH.as_posix(),
        "config_sha256": _sha256(root / CONFIG_PATH),
        "historical_openai_series_status": "interrupted_not_scored",
        "logical_prediction_count": EXPECTED_LOGICAL_COUNTS["total"],
        "predecessor_series_id": "qwen35-27b-fp8-v1",
        "predecessor_status": "superseded_not_run",
        "provider_request_count": 0,
        "planned_provider_request_count": EXPECTED_PROVIDER_COUNTS["planned_including_compatibility"],
        "schema_version": "qwen_series_manifest_v2",
        "series_id": SERIES_ID,
        "status": "frozen_not_run",
    })
    return destination


def verify_series_manifest(repo_root: Path, path: Path | None = None) -> Mapping[str, object]:
    root = repo_root.resolve()
    config = load_qwen_v2_config(root)
    target = path or root / SERIES_MANIFEST_PATH
    if not target.is_absolute():
        target = root / target
    actual = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "authority_bindings": {str(row["path"]): str(row["sha256"]) for row in config["authority_bindings"]},  # type: ignore[index]
        "config_path": CONFIG_PATH.as_posix(),
        "config_sha256": _sha256(root / CONFIG_PATH),
        "historical_openai_series_status": "interrupted_not_scored",
        "logical_prediction_count": EXPECTED_LOGICAL_COUNTS["total"],
        "predecessor_series_id": "qwen35-27b-fp8-v1",
        "predecessor_status": "superseded_not_run",
        "provider_request_count": 0,
        "planned_provider_request_count": EXPECTED_PROVIDER_COUNTS["planned_including_compatibility"],
        "schema_version": "qwen_series_manifest_v2",
        "series_id": SERIES_ID,
        "status": "frozen_not_run",
    }
    if actual != expected:
        raise QwenV2ContractError("Qwen v2 series manifest changed")
    return actual


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise QwenV2ContractError(f"{key} must be an object")
    return nested


def _validate_budget(value: Mapping[str, object]) -> None:
    budget = _mapping(value, "budget_inr")
    if budget.get("stage_1") != 200.0 or budget.get("stage_2") != 300.0 or budget.get("stage_3") != 1000.0:
        raise QwenV2ContractError("stage budgets changed")
    if budget.get("total") != 1500.0:
        raise QwenV2ContractError("cumulative budget changed")
    if budget.get("stage_3_minimum_valid_response_count") != 884:
        raise QwenV2ContractError("Stage 2 structural validity gate changed")
    if budget.get("stage_3_minimum_extraction_valid_count") != 20:
        raise QwenV2ContractError("Stage 2 extraction gate changed")


def _validate_boundaries(value: Mapping[str, object]) -> None:
    boundary = _mapping(value, "data_access")
    required_false = ("cross_split_access", "cross_user_access", "openai_artifact_reuse", "qwen_v1_artifact_reuse")
    if any(boundary.get(key) is not False for key in required_false):
        raise QwenV2ContractError("runtime isolation boundary changed")
    if boundary.get("gold_after_prediction_seal_only") is not True:
        raise QwenV2ContractError("gold-open boundary changed")
    prohibited = boundary.get("prohibited_runtime_roots")
    if not isinstance(prohibited, list) or not {"data/scaled-v1/gold", "data/scaled-v1/oracle", "data/scaled-v1/review"}.issubset(prohibited):
        raise QwenV2ContractError("prohibited runtime roots changed")


def _verify_authorities(root: Path, value: Mapping[str, object]) -> None:
    bindings = value.get("authority_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise QwenV2ContractError("authority bindings are missing")
    for row in bindings:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise QwenV2ContractError("authority binding is invalid")
        path = root / str(row["path"])
        if _sha256(path) != row["sha256"]:
            raise QwenV2ContractError(f"authority changed: {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root)
    path = root.resolve() / SERIES_MANIFEST_PATH
    if args.verify:
        verify_series_manifest(root, path)
    else:
        build_series_manifest(root, path)
    print(path)


if __name__ == "__main__":
    main()
