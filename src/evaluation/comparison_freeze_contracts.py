"""Strict, immutable contracts for the Step 10.1 comparison freeze."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
import re
from typing import Mapping


COMPARISON_VERSION = "frozen_comparison_v1"
SCHEMA_VERSION = "frozen_comparison_schema_v1"
CONFIG_VERSION = "frozen_comparison_config_v1"
DATASET_VERSION = "frozen-comparison-v1"
RESULT_VERSION = "frozen-comparison-v1"
DEFINITION_SHA256 = "f701db8601a1f0241b6c553b86d3cd7432c4e4563db8a7a53eeaac7d4c195858"
BASELINE_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
TASK_ORDER = ("qa", "summary", "interactive")
SPLIT_ORDER = ("development", "frozen_test")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:/+-]{1,240}$")


class ComparisonFreezeError(ValueError):
    """Reject a changed, unsafe, or non-canonical comparison definition."""


@dataclass(frozen=True)
class ManifestFileBinding:
    path: str
    layer: str
    sha256: str
    records: int

    def __post_init__(self) -> None:
        _name(self.path, "manifest path")
        _name(self.layer, "manifest layer")
        _sha(self.sha256, "manifest file hash")
        if type(self.records) is not int or self.records < 0:
            raise ComparisonFreezeError("manifest record count is invalid")


@dataclass(frozen=True)
class BaselineDefinition:
    baseline_id: str
    memory_semantics: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.baseline_id not in BASELINE_ORDER:
            raise ComparisonFreezeError("baseline ID is invalid")
        _name(self.memory_semantics, "baseline semantics")
        _ordered_unique(self.capabilities, "baseline capabilities")


@dataclass(frozen=True)
class TaskDefinition:
    task: str
    total_count: int
    development_count: int
    frozen_test_count: int

    def __post_init__(self) -> None:
        if self.task not in TASK_ORDER:
            raise ComparisonFreezeError("task is invalid")
        if any(type(value) is not int or value < 0 for value in (
            self.total_count, self.development_count, self.frozen_test_count,
        )):
            raise ComparisonFreezeError("task count is invalid")
        if self.development_count + self.frozen_test_count != self.total_count:
            raise ComparisonFreezeError("task split does not equal task total")


@dataclass(frozen=True)
class PromptBinding:
    task: str
    prompt_version: str
    prompt_sha256: str
    baseline_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task not in TASK_ORDER or self.baseline_ids != BASELINE_ORDER:
            raise ComparisonFreezeError("prompt task or baseline coverage changed")
        _name(self.prompt_version, "prompt version")
        _sha(self.prompt_sha256, "prompt hash")


@dataclass(frozen=True)
class AuthorityBinding:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _name(self.path, "authority path")
        _sha(self.sha256, "authority hash")


@dataclass(frozen=True)
class PrerequisiteGap:
    baseline_id: str
    status: str
    required_in: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.baseline_id not in BASELINE_ORDER:
            raise ComparisonFreezeError("prerequisite baseline is invalid")
        if self.status != "missing_scaled_runtime_and_prediction_release":
            raise ComparisonFreezeError("prerequisite status changed")
        if self.required_in != ("step_10_2", "step_10_3"):
            raise ComparisonFreezeError("prerequisite ownership changed")


@dataclass(frozen=True)
class FrozenComparisonDefinition:
    comparison_version: str
    schema_version: str
    config_version: str
    dataset_version: str
    result_version: str
    guidance_version: str
    guidance_sha256: str
    starting_commit: str
    scaled_manifest_path: str
    scaled_manifest_sha256: str
    scaled_dataset_sha256: str
    referenced_files_opened: bool
    manifest_files: tuple[ManifestFileBinding, ...]
    manifest_counts: Mapping[str, object]
    capability_counts: Mapping[str, object]
    development_users: tuple[str, ...]
    frozen_test_users: tuple[str, ...]
    tasks: tuple[TaskDefinition, ...]
    baselines: tuple[BaselineDefinition, ...]
    prompt_bindings: tuple[PromptBinding, ...]
    model_policy: Mapping[str, object]
    ordering_policy: Mapping[str, object]
    slice_order: tuple[str, ...]
    metric_groups: tuple[Mapping[str, object], ...]
    null_reasons: tuple[str, ...]
    authorities: tuple[AuthorityBinding, ...]
    prerequisite_gaps: tuple[PrerequisiteGap, ...]
    provider_request_count: int
    incremental_cost_usd: str
    historical_openai_spend_usd: str
    implementer_pilot_answer_reference_exposure: bool
    implementer_pilot_answer_reference_used: bool
    step_10_2_started: bool

    def __post_init__(self) -> None:
        if (
            self.comparison_version, self.schema_version, self.config_version,
            self.dataset_version, self.result_version,
        ) != (
            COMPARISON_VERSION, SCHEMA_VERSION, CONFIG_VERSION,
            DATASET_VERSION, RESULT_VERSION,
        ):
            raise ComparisonFreezeError("definition version changed")
        if self.referenced_files_opened or self.provider_request_count != 0:
            raise ComparisonFreezeError("definition crossed the no-read/no-call boundary")
        if self.incremental_cost_usd != "0.0000000" or self.historical_openai_spend_usd != "0.2314404":
            raise ComparisonFreezeError("cost policy changed")
        if not self.implementer_pilot_answer_reference_exposure or self.implementer_pilot_answer_reference_used:
            raise ComparisonFreezeError("exposure disclosure changed")
        if self.step_10_2_started:
            raise ComparisonFreezeError("Step 10.2 has started")
        if tuple(item.baseline_id for item in self.baselines) != BASELINE_ORDER:
            raise ComparisonFreezeError("baseline order changed")
        if tuple(item.task for item in self.tasks) != TASK_ORDER:
            raise ComparisonFreezeError("task order changed")
        if tuple(item.task for item in self.prompt_bindings) != TASK_ORDER:
            raise ComparisonFreezeError("prompt order changed")
        if tuple(item.baseline_id for item in self.prerequisite_gaps) != BASELINE_ORDER:
            raise ComparisonFreezeError("prerequisite order changed")
        _ordered_unique(self.development_users, "development users")
        _ordered_unique(self.frozen_test_users, "frozen-test users")
        _ordered_unique(self.slice_order, "slice order")
        _ordered_unique(self.null_reasons, "null reasons")
        paths = tuple(item.path for item in self.manifest_files)
        if len(paths) != len(set(paths)):
            raise ComparisonFreezeError("duplicate manifest file path")
        authorities = tuple(item.path for item in self.authorities)
        if len(authorities) != len(set(authorities)) or authorities != tuple(sorted(authorities)):
            raise ComparisonFreezeError("authority paths are not sorted and unique")
        _json_safe(self)


@dataclass(frozen=True)
class ComparisonFreezeChecks:
    comparison_version: str
    manifest_only: bool
    referenced_files_opened: bool
    manifest_file_count: int
    baseline_count: int
    task_count: int
    metric_count: int
    prerequisite_gap_count: int
    prediction_count: int
    score_count: int
    provider_request_count: int
    step_10_2_started: bool

    def __post_init__(self) -> None:
        if self.comparison_version != COMPARISON_VERSION or not self.manifest_only:
            raise ComparisonFreezeError("checks version or mode changed")
        if self.referenced_files_opened or self.prediction_count or self.score_count or self.provider_request_count:
            raise ComparisonFreezeError("checks report prohibited work")
        if (self.manifest_file_count, self.baseline_count, self.task_count, self.prerequisite_gap_count) != (27, 8, 3, 8):
            raise ComparisonFreezeError("checks count changed")
        if self.metric_count <= 0 or self.step_10_2_started:
            raise ComparisonFreezeError("checks metric or phase boundary changed")


def canonical_json_bytes(value: object) -> bytes:
    _json_safe(value)
    if is_dataclass(value):
        value = asdict(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_json_bytes(raw: bytes, *, location: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw, parse_constant=lambda value: (_raise(f"non-finite JSON at {location}: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonFreezeError(f"unsafe JSON at {location}") from exc
    if not isinstance(value, dict):
        raise ComparisonFreezeError(f"JSON object required at {location}")
    _json_safe(value)
    return value


def manifest_file_from_mapping(value: Mapping[str, object]) -> ManifestFileBinding:
    _strict(value, ManifestFileBinding)
    return ManifestFileBinding(**value)  # type: ignore[arg-type]


def baseline_from_mapping(value: Mapping[str, object]) -> BaselineDefinition:
    _strict(value, BaselineDefinition)
    return BaselineDefinition(
        baseline_id=_string(value["baseline_id"]),
        memory_semantics=_string(value["memory_semantics"]),
        capabilities=_strings(value["capabilities"]),
    )


def task_from_mapping(value: Mapping[str, object]) -> TaskDefinition:
    _strict(value, TaskDefinition)
    return TaskDefinition(**value)  # type: ignore[arg-type]


def prompt_from_mapping(value: Mapping[str, object]) -> PromptBinding:
    _strict(value, PromptBinding)
    return PromptBinding(
        task=_string(value["task"]),
        prompt_version=_string(value["prompt_version"]),
        prompt_sha256=_string(value["prompt_sha256"]),
        baseline_ids=_strings(value["baseline_ids"]),
    )


def authority_from_mapping(value: Mapping[str, object]) -> AuthorityBinding:
    _strict(value, AuthorityBinding)
    return AuthorityBinding(**value)  # type: ignore[arg-type]


def gap_from_mapping(value: Mapping[str, object]) -> PrerequisiteGap:
    _strict(value, PrerequisiteGap)
    return PrerequisiteGap(
        baseline_id=_string(value["baseline_id"]),
        status=_string(value["status"]),
        required_in=_strings(value["required_in"]),
    )


def checks_from_mapping(value: Mapping[str, object]) -> ComparisonFreezeChecks:
    _strict(value, ComparisonFreezeChecks)
    return ComparisonFreezeChecks(**value)  # type: ignore[arg-type]


def definition_from_mapping(value: Mapping[str, object]) -> FrozenComparisonDefinition:
    expected = {field.name for field in fields(FrozenComparisonDefinition)}
    if set(value) != expected:
        raise ComparisonFreezeError("definition fields changed")
    definition = FrozenComparisonDefinition(
        comparison_version=_string(value["comparison_version"]),
        schema_version=_string(value["schema_version"]),
        config_version=_string(value["config_version"]),
        dataset_version=_string(value["dataset_version"]),
        result_version=_string(value["result_version"]),
        guidance_version=_string(value["guidance_version"]),
        guidance_sha256=_string(value["guidance_sha256"]),
        starting_commit=_string(value["starting_commit"]),
        scaled_manifest_path=_string(value["scaled_manifest_path"]),
        scaled_manifest_sha256=_string(value["scaled_manifest_sha256"]),
        scaled_dataset_sha256=_string(value["scaled_dataset_sha256"]),
        referenced_files_opened=_bool(value["referenced_files_opened"]),
        manifest_files=tuple(manifest_file_from_mapping(_mapping(item)) for item in _sequence(value["manifest_files"])),
        manifest_counts=_mapping(value["manifest_counts"]),
        capability_counts=_mapping(value["capability_counts"]),
        development_users=_strings(value["development_users"]),
        frozen_test_users=_strings(value["frozen_test_users"]),
        tasks=tuple(task_from_mapping(_mapping(item)) for item in _sequence(value["tasks"])),
        baselines=tuple(baseline_from_mapping(_mapping(item)) for item in _sequence(value["baselines"])),
        prompt_bindings=tuple(prompt_from_mapping(_mapping(item)) for item in _sequence(value["prompt_bindings"])),
        model_policy=_mapping(value["model_policy"]),
        ordering_policy=_mapping(value["ordering_policy"]),
        slice_order=_strings(value["slice_order"]),
        metric_groups=tuple(_mapping(item) for item in _sequence(value["metric_groups"])),
        null_reasons=_strings(value["null_reasons"]),
        authorities=tuple(authority_from_mapping(_mapping(item)) for item in _sequence(value["authorities"])),
        prerequisite_gaps=tuple(gap_from_mapping(_mapping(item)) for item in _sequence(value["prerequisite_gaps"])),
        provider_request_count=_int(value["provider_request_count"]),
        incremental_cost_usd=_string(value["incremental_cost_usd"]),
        historical_openai_spend_usd=_string(value["historical_openai_spend_usd"]),
        implementer_pilot_answer_reference_exposure=_bool(value["implementer_pilot_answer_reference_exposure"]),
        implementer_pilot_answer_reference_used=_bool(value["implementer_pilot_answer_reference_used"]),
        step_10_2_started=_bool(value["step_10_2_started"]),
    )
    if stable_sha256(definition) != DEFINITION_SHA256:
        raise ComparisonFreezeError("definition policy changed")
    return definition


def _json_safe(value: object) -> None:
    if is_dataclass(value):
        value = asdict(value)
    if value is None or isinstance(value, bool) or type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComparisonFreezeError("non-finite JSON number")
        return
    if isinstance(value, str):
        if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
            raise ComparisonFreezeError("unsafe JSON string")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ComparisonFreezeError("JSON object key is not a string")
            _json_safe(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_safe(item)
        return
    raise ComparisonFreezeError("value is not JSON-safe")


def _strict(value: Mapping[str, object], contract: type[object]) -> None:
    if set(value) != {field.name for field in fields(contract)}:
        raise ComparisonFreezeError(f"{contract.__name__} fields changed")


def _ordered_unique(values: tuple[str, ...], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ComparisonFreezeError(f"{label} are empty or duplicated")


def _sha(value: str, label: str) -> None:
    if SHA256.fullmatch(value) is None:
        raise ComparisonFreezeError(f"{label} is invalid")


def _name(value: str, label: str) -> None:
    if SAFE_NAME.fullmatch(value) is None:
        raise ComparisonFreezeError(f"{label} is invalid")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ComparisonFreezeError("mapping required")
    return value


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ComparisonFreezeError("array required")
    return tuple(value)


def _strings(value: object) -> tuple[str, ...]:
    values = _sequence(value)
    if not all(isinstance(item, str) for item in values):
        raise ComparisonFreezeError("string array required")
    return tuple(values)  # type: ignore[return-value]


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ComparisonFreezeError("string required")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise ComparisonFreezeError("boolean required")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise ComparisonFreezeError("integer required")
    return value


def _raise(message: str) -> object:
    raise ComparisonFreezeError(message)
