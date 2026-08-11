"""Frozen configuration and deterministic safety checks for atomic extraction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_UP
import hashlib
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
    legacy_generation = {
        "api": "responses", "max_output_tokens": 4000,
        "store": False, "text_format": "json_object",
    }
    strict_generation_fields = {
        "api", "max_output_tokens", "store", "text_format",
        "text_schema_version", "text_schema_sha256",
    }
    strict_generation_field_sets = (
        strict_generation_fields,
        strict_generation_fields | {"normalization_version"},
    )
    strict_generation_valid = (
        isinstance(generation, dict)
        and set(generation) in strict_generation_field_sets
        and generation["api"] == "responses"
        and generation["max_output_tokens"] == 4000
        and generation["store"] is False
        and generation["text_format"] == "json_schema"
        and generation["text_schema_version"] == "atomic_extraction_v1"
        and generation.get("normalization_version") in (
            None,
            "unicode_punctuation_v1",
            "source_span_v1",
            "source_span_boolean_polarity_v2",
        )
        and isinstance(generation["text_schema_sha256"], str)
    )
    if not isinstance(generation, dict) or not (
        generation == legacy_generation or strict_generation_valid
    ):
        raise AtomicRunConfigError("generation settings are incompatible")
    if strict_generation_valid:
        _require_sha256(generation["text_schema_sha256"], "text_schema_sha256")

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


def additive_cost_text(value: Decimal) -> str:
    """Serialize new fallback-run costs without losing additive precision."""

    return format(value.quantize(Decimal("0.0000001")), ".7f")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AtomicRunConfigError(f"{name} must be a lowercase SHA-256")


STEP35_TOKEN_COUNTER_VERSION = "tiktoken_0_13_0_request_json_v1"
STEP35_TOKEN_ENCODING = "o200k_base"


@dataclass(frozen=True)
class Step35ModelPlan:
    label: str
    requested_model: str
    resolved_model: str
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    output_directory: str


@dataclass(frozen=True)
class Step35RunConfig:
    configuration_version: str
    stage: str
    dataset_version: str
    dataset_split: str
    dataset_sha256: str
    runtime_user_file_sha256: str
    runtime_source_file_sha256: str
    gold_claim_file_sha256: str
    predicate_registry_path: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    prompt_version: str
    prompt_sha256: str
    schema_version: str
    schema_sha256: str
    generation_settings: Mapping[str, object]
    token_counter_version: str
    token_encoding: str
    input_token_reserve_per_request: int
    expected_output_tokens_per_request: int
    case_order: tuple[tuple[str, str], ...]
    models: tuple[Step35ModelPlan, ...]
    maximum_retry_requests: int
    prior_spend_usd: Decimal
    cumulative_authorization_usd: Decimal
    predecessor: Mapping[str, object] | None
    configuration_sha256: str


_STEP35_FIELDS = {
    "configuration_version",
    "stage",
    "dataset_version",
    "dataset_split",
    "dataset_sha256",
    "runtime_user_file_sha256",
    "runtime_source_file_sha256",
    "gold_claim_file_sha256",
    "predicate_registry_path",
    "predicate_registry_version",
    "predicate_registry_sha256",
    "prompt_version",
    "prompt_sha256",
    "schema_version",
    "schema_sha256",
    "generation_settings",
    "token_counter_version",
    "token_encoding",
    "input_token_reserve_per_request",
    "expected_output_tokens_per_request",
    "case_order",
    "models",
    "maximum_retry_requests",
    "prior_spend_usd",
    "cumulative_authorization_usd",
}
_STEP35_V2_FIELDS = _STEP35_FIELDS | {"predecessor"}
_STEP35_MODEL_FIELDS = {
    "label",
    "requested_model",
    "resolved_model",
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "output_directory",
}


def load_step35_run_config(
    path: str | Path,
    repo_root: str | Path = ".",
) -> Step35RunConfig:
    """Load one frozen Step 3.5 stage, including zero-retry plans."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(repo_root).resolve() / config_path
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AtomicRunConfigError(f"could not load Step 3.5 config: {error}") from error
    if not isinstance(raw, dict) or frozenset(raw) not in {
        frozenset(_STEP35_FIELDS),
        frozenset(_STEP35_V2_FIELDS),
    }:
        raise AtomicRunConfigError("Step 3.5 config fields changed")
    for name in (
        "configuration_version",
        "dataset_version",
        "predicate_registry_path",
        "predicate_registry_version",
        "prompt_version",
        "schema_version",
    ):
        if not isinstance(raw[name], str) or not raw[name]:
            raise AtomicRunConfigError(f"{name} must be a non-empty string")
    if raw["stage"] not in {"qualification", "full"}:
        raise AtomicRunConfigError("Step 3.5 stage must be qualification or full")
    if raw["dataset_split"] != "development":
        raise AtomicRunConfigError("Step 3.5 may use only the development split")
    for name in (
        "dataset_sha256",
        "runtime_user_file_sha256",
        "runtime_source_file_sha256",
        "gold_claim_file_sha256",
        "predicate_registry_sha256",
        "prompt_sha256",
        "schema_sha256",
    ):
        if not isinstance(raw[name], str):
            raise AtomicRunConfigError(f"{name} must be a SHA-256 string")
        _require_sha256(raw[name], name)

    settings = raw["generation_settings"]
    if not isinstance(settings, dict) or settings != {
        "api": "responses",
        "max_output_tokens": 1200,
        "store": False,
        "temperature": 0.0,
        "text_format": "json_schema",
    }:
        raise AtomicRunConfigError("Step 3.5 generation settings changed")
    if raw["token_counter_version"] != STEP35_TOKEN_COUNTER_VERSION:
        raise AtomicRunConfigError("Step 3.5 token counter version changed")
    if raw["token_encoding"] != STEP35_TOKEN_ENCODING:
        raise AtomicRunConfigError("Step 3.5 token encoding changed")
    reserve = raw["input_token_reserve_per_request"]
    expected_output = raw["expected_output_tokens_per_request"]
    retries = raw["maximum_retry_requests"]
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        raise AtomicRunConfigError("input token reserve must be a non-negative integer")
    if not isinstance(expected_output, int) or isinstance(expected_output, bool) or not 0 < expected_output <= 1200:
        raise AtomicRunConfigError("expected output tokens must be between 1 and 1200")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise AtomicRunConfigError("maximum retry requests must be zero or greater")

    case_order_raw = raw["case_order"]
    if not isinstance(case_order_raw, list) or not case_order_raw:
        raise AtomicRunConfigError("Step 3.5 case_order must be non-empty")
    cases: list[tuple[str, str]] = []
    for item in case_order_raw:
        if not isinstance(item, dict) or set(item) != {"user_id", "source_id"}:
            raise AtomicRunConfigError("Step 3.5 case fields changed")
        if not all(isinstance(item[name], str) and item[name] for name in item):
            raise AtomicRunConfigError("Step 3.5 case IDs must be non-empty")
        cases.append((item["user_id"], item["source_id"]))
    if len(cases) != len(set(cases)):
        raise AtomicRunConfigError("Step 3.5 case_order contains duplicates")

    models_raw = raw["models"]
    if not isinstance(models_raw, list) or not models_raw:
        raise AtomicRunConfigError("Step 3.5 models must be non-empty")
    models: list[Step35ModelPlan] = []
    labels: set[str] = set()
    for item in models_raw:
        if not isinstance(item, dict) or set(item) != _STEP35_MODEL_FIELDS:
            raise AtomicRunConfigError("Step 3.5 model fields changed")
        for name in ("label", "requested_model", "resolved_model", "output_directory"):
            if not isinstance(item[name], str) or not item[name]:
                raise AtomicRunConfigError(f"model {name} must be non-empty")
        if item["label"] in labels:
            raise AtomicRunConfigError("Step 3.5 model labels must be unique")
        labels.add(item["label"])
        if item["requested_model"] != item["resolved_model"]:
            raise AtomicRunConfigError("Step 3.5 model snapshots must be pinned")
        prices: dict[str, Decimal] = {}
        for name in ("input_usd_per_million_tokens", "output_usd_per_million_tokens"):
            try:
                prices[name] = Decimal(item[name])
            except (InvalidOperation, TypeError):
                raise AtomicRunConfigError(f"model {name} must be decimal text") from None
            if prices[name] <= 0:
                raise AtomicRunConfigError(f"model {name} must be positive")
        models.append(
            Step35ModelPlan(
                label=item["label"],
                requested_model=item["requested_model"],
                resolved_model=item["resolved_model"],
                input_usd_per_million_tokens=prices["input_usd_per_million_tokens"],
                output_usd_per_million_tokens=prices["output_usd_per_million_tokens"],
                output_directory=item["output_directory"],
            )
        )
    decimals: dict[str, Decimal] = {}
    for name in ("prior_spend_usd", "cumulative_authorization_usd"):
        try:
            decimals[name] = Decimal(raw[name])
        except (InvalidOperation, TypeError):
            raise AtomicRunConfigError(f"{name} must be decimal text") from None
    if decimals["prior_spend_usd"] < 0 or decimals["cumulative_authorization_usd"] <= 0:
        raise AtomicRunConfigError("Step 3.5 spend values are invalid")
    predecessor = raw.get("predecessor")
    has_predecessor = raw["configuration_version"] in {
        "phase4-input-model-qualification-v2",
        "phase4-input-development-v2",
        "phase4-input-development-gpt41-fallback-v1",
    }
    if has_predecessor != (set(raw) == _STEP35_V2_FIELDS):
        raise AtomicRunConfigError(
            "Step 3.5 recovery configs require exactly one predecessor block"
        )
    if predecessor is not None and not isinstance(predecessor, dict):
        raise AtomicRunConfigError("Step 3.5 predecessor must be an object")
    return Step35RunConfig(
        configuration_version=raw["configuration_version"],
        stage=raw["stage"],
        dataset_version=raw["dataset_version"],
        dataset_split=raw["dataset_split"],
        dataset_sha256=raw["dataset_sha256"],
        runtime_user_file_sha256=raw["runtime_user_file_sha256"],
        runtime_source_file_sha256=raw["runtime_source_file_sha256"],
        gold_claim_file_sha256=raw["gold_claim_file_sha256"],
        predicate_registry_path=raw["predicate_registry_path"],
        predicate_registry_version=raw["predicate_registry_version"],
        predicate_registry_sha256=raw["predicate_registry_sha256"],
        prompt_version=raw["prompt_version"],
        prompt_sha256=raw["prompt_sha256"],
        schema_version=raw["schema_version"],
        schema_sha256=raw["schema_sha256"],
        generation_settings=dict(settings),
        token_counter_version=raw["token_counter_version"],
        token_encoding=raw["token_encoding"],
        input_token_reserve_per_request=reserve,
        expected_output_tokens_per_request=expected_output,
        case_order=tuple(cases),
        models=tuple(models),
        maximum_retry_requests=retries,
        prior_spend_usd=decimals["prior_spend_usd"],
        cumulative_authorization_usd=decimals["cumulative_authorization_usd"],
        predecessor=dict(predecessor) if predecessor is not None else None,
        configuration_sha256=canonical_sha256(raw),
    )


def count_step35_request_tokens(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    text_format: Mapping[str, object],
    generation_settings: Mapping[str, object],
) -> int:
    """Count the complete serialized request with the frozen model encoding."""

    try:
        import tiktoken
    except ImportError as error:
        raise AtomicRunConfigError(
            "Step 3.5 requires tiktoken==0.13.0; install requirements-step35.txt"
        ) from error
    if getattr(tiktoken, "__version__", None) != "0.13.0":
        raise AtomicRunConfigError("Step 3.5 requires tiktoken==0.13.0")
    try:
        encoding = tiktoken.encoding_for_model(model)
    except (KeyError, OSError, RuntimeError) as error:
        raise AtomicRunConfigError(f"could not load the Step 3.5 tokenizer: {error}") from error
    if encoding.name != STEP35_TOKEN_ENCODING:
        raise AtomicRunConfigError("Step 3.5 model encoding changed")
    serialized = _step35_request_json(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        text_format=text_format,
        generation_settings=generation_settings,
    )
    return len(encoding.encode(serialized))


def step35_request_sha256(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    text_format: Mapping[str, object],
    generation_settings: Mapping[str, object],
) -> str:
    """Hash the exact JSON request body counted for one frozen request."""

    serialized = _step35_request_json(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        text_format=text_format,
        generation_settings=generation_settings,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _step35_request_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    text_format: Mapping[str, object],
    generation_settings: Mapping[str, object],
) -> str:
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "temperature": float(generation_settings["temperature"]),
        "max_output_tokens": generation_settings["max_output_tokens"],
        "store": generation_settings["store"],
        "text": {"format": text_format},
    }
    # Match OpenAIResponsesClient._post_response byte-for-byte. The fixed
    # reserve covers provider framing that is not represented in this body.
    return json.dumps(payload, ensure_ascii=False)


def reserve_step35_input_tokens(count: int, reserve_per_request: int) -> int:
    """Add the frozen provider-framing reserve to an exact request count."""

    return count + reserve_per_request


def step35_cost(
    model: Step35ModelPlan,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    million = Decimal(1_000_000)
    return (
        Decimal(input_tokens) * model.input_usd_per_million_tokens / million
        + Decimal(output_tokens) * model.output_usd_per_million_tokens / million
    )
