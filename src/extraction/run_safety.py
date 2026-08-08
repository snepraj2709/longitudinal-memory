"""Frozen configuration and deterministic safety checks for atomic extraction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_UP
import json
from pathlib import Path
from typing import Mapping

from evaluation.run_config import canonical_sha256


DEFAULT_CONFIG_PATH = Path("configs/extraction/atomic_extraction_run_v2.json")


class AtomicRunConfigError(ValueError):
    """Raised when a frozen run configuration is malformed or has drifted."""


@dataclass(frozen=True)
class AtomicRunConfig:
    configuration_version: str
    run_format_version: str
    provider: str
    dataset_version: str
    dataset_split: str
    runtime_dataset_sha256: str
    requested_model: str
    resolved_model: str
    prompt_version: str
    prompt_sha256: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    temperature: float
    generation_settings: Mapping[str, object]
    case_order: tuple[tuple[str, str], ...]
    source_file_sha256: Mapping[str, str]
    gold_file_sha256: str
    planned_request_count: int
    maximum_retry_requests: int
    maximum_request_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    expected_output_tokens: int
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    hard_cost_cap_usd: Decimal
    configuration_sha256: str


_FIELDS = {
    "configuration_version", "run_format_version", "provider", "dataset_version",
    "dataset_split", "runtime_dataset_sha256",
    "requested_model", "resolved_model", "prompt_version", "prompt_sha256",
    "predicate_registry_version", "predicate_registry_sha256", "temperature",
    "generation_settings", "case_order", "source_file_sha256",
    "gold_file_sha256", "planned_request_count", "maximum_retry_requests",
    "maximum_request_attempts",
    "maximum_input_tokens", "maximum_output_tokens", "expected_output_tokens",
    "input_usd_per_million_tokens", "output_usd_per_million_tokens",
    "hard_cost_cap_usd",
}


def load_atomic_run_config(
    repo_root: str | Path = ".", config_path: str | Path = DEFAULT_CONFIG_PATH
) -> AtomicRunConfig:
    """Load the versioned configuration and reject incomplete or unsafe values."""

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(repo_root).resolve() / path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AtomicRunConfigError(f"could not load frozen run config: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise AtomicRunConfigError("frozen run config fields changed")

    strings = (
        "configuration_version", "run_format_version", "provider", "dataset_version",
        "dataset_split", "runtime_dataset_sha256",
        "requested_model", "resolved_model", "prompt_version", "prompt_sha256",
        "predicate_registry_version", "predicate_registry_sha256",
        "gold_file_sha256",
    )
    for name in strings:
        if not isinstance(raw[name], str) or not raw[name]:
            raise AtomicRunConfigError(f"{name} must be a non-empty string")
    for name in (
        "runtime_dataset_sha256", "prompt_sha256", "predicate_registry_sha256",
        "gold_file_sha256",
    ):
        _require_sha256(raw[name], name)
    if raw["provider"] != "openai":
        raise AtomicRunConfigError("provider must be openai")
    if raw["requested_model"] != raw["resolved_model"]:
        raise AtomicRunConfigError("requested and resolved model snapshots must match")
    if not isinstance(raw["temperature"], (int, float)) or isinstance(raw["temperature"], bool):
        raise AtomicRunConfigError("temperature must be numeric")

    generation = raw["generation_settings"]
    if not isinstance(generation, dict) or generation != {
        "api": "responses", "max_output_tokens": 4000,
        "store": False, "text_format": "json_object",
    }:
        raise AtomicRunConfigError("generation settings are incompatible")

    case_order_raw = raw["case_order"]
    if not isinstance(case_order_raw, list):
        raise AtomicRunConfigError("case_order must be a list")
    case_order: list[tuple[str, str]] = []
    for item in case_order_raw:
        if not isinstance(item, dict) or set(item) != {"case_id", "source_id"}:
            raise AtomicRunConfigError("case_order entry fields changed")
        if not all(isinstance(item[name], str) and item[name] for name in item):
            raise AtomicRunConfigError("case_order IDs must be non-empty strings")
        case_order.append((item["case_id"], item["source_id"]))
    if len(case_order) != len(set(case_order)):
        raise AtomicRunConfigError("case_order contains duplicates")

    source_hashes = raw["source_file_sha256"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise AtomicRunConfigError("source_file_sha256 must be a non-empty object")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise AtomicRunConfigError("source hash entries must be strings")
        _require_sha256(digest, f"source_file_sha256.{name}")

    integers = (
        "planned_request_count", "maximum_retry_requests", "maximum_request_attempts",
        "maximum_input_tokens",
        "maximum_output_tokens", "expected_output_tokens",
    )
    for name in integers:
        if not isinstance(raw[name], int) or isinstance(raw[name], bool) or raw[name] <= 0:
            raise AtomicRunConfigError(f"{name} must be a positive integer")
    if raw["planned_request_count"] != len(case_order):
        raise AtomicRunConfigError("planned request count does not match case order")
    if raw["maximum_request_attempts"] != (
        raw["planned_request_count"] + raw["maximum_retry_requests"]
    ):
        raise AtomicRunConfigError("maximum request attempts does not match plan plus retries")

    prices = {}
    for name in (
        "input_usd_per_million_tokens", "output_usd_per_million_tokens",
        "hard_cost_cap_usd",
    ):
        try:
            prices[name] = Decimal(raw[name])
        except (InvalidOperation, TypeError):
            raise AtomicRunConfigError(f"{name} must be a decimal string") from None
        if prices[name] <= 0:
            raise AtomicRunConfigError(f"{name} must be positive")

    return AtomicRunConfig(
        **{name: raw[name] for name in strings},
        temperature=float(raw["temperature"]),
        generation_settings=dict(generation),
        case_order=tuple(case_order),
        source_file_sha256=dict(source_hashes),
        planned_request_count=raw["planned_request_count"],
        maximum_retry_requests=raw["maximum_retry_requests"],
        maximum_request_attempts=raw["maximum_request_attempts"],
        maximum_input_tokens=raw["maximum_input_tokens"],
        maximum_output_tokens=raw["maximum_output_tokens"],
        expected_output_tokens=raw["expected_output_tokens"],
        input_usd_per_million_tokens=prices["input_usd_per_million_tokens"],
        output_usd_per_million_tokens=prices["output_usd_per_million_tokens"],
        hard_cost_cap_usd=prices["hard_cost_cap_usd"],
        configuration_sha256=canonical_sha256(raw),
    )


def token_cost(config: AtomicRunConfig, input_tokens: int, output_tokens: int) -> Decimal:
    million = Decimal(1_000_000)
    return (
        Decimal(input_tokens) * config.input_usd_per_million_tokens / million
        + Decimal(output_tokens) * config.output_usd_per_million_tokens / million
    )


def cost_text(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_UP))


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AtomicRunConfigError(f"{name} must be a lowercase SHA-256")
