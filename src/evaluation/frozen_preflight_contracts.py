"""Strict contracts for the Step 10.2 frozen-run preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Mapping


PREFLIGHT_VERSION = "frozen_preflight_v1"
SCHEMA_VERSION = "frozen_preflight_schema_v1"
BASELINE_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
TASK_ORDER = ("qa", "summary", "interactive")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:+/\[\]-]{1,240}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FrozenPreflightError(ValueError):
    """Reject unsafe, changed, or non-canonical preflight data."""


@dataclass(frozen=True)
class TransmissionRecord:
    transmission_id: str
    kind: str
    record_id: str
    user_id: str
    split: str
    baseline_id: str | None
    task: str | None
    source_type: str | None
    fields_transmitted: tuple[str, ...]
    synthetic_benchmark: bool
    restricted_content: bool

    def __post_init__(self) -> None:
        _sha(self.transmission_id, "transmission ID")
        if self.kind not in {"extraction_source", "answer_case"}:
            raise FrozenPreflightError("transmission kind is invalid")
        _safe(self.record_id, "record ID")
        _safe(self.user_id, "user ID")
        if self.split not in {"development", "test"}:
            raise FrozenPreflightError("transmission split is invalid")
        if self.task is not None and self.task not in TASK_ORDER:
            raise FrozenPreflightError("transmission task is invalid")
        if self.kind == "extraction_source" and self.baseline_id is not None:
            raise FrozenPreflightError("extraction transmission cannot name a baseline")
        if self.kind == "answer_case" and self.baseline_id not in BASELINE_ORDER:
            raise FrozenPreflightError("answer transmission baseline is invalid")
        if self.source_type is not None and self.source_type not in {
            "conversation", "email", "chat", "calendar"
        }:
            raise FrozenPreflightError("source type is invalid")
        _ordered_unique(self.fields_transmitted, "transmitted fields")
        if not self.synthetic_benchmark or self.restricted_content:
            raise FrozenPreflightError("only unrestricted synthetic records are allowed")
        _json_safe(self)


@dataclass(frozen=True)
class TokenEstimate:
    estimate_id: str
    batch_id: str
    position: int
    record_id: str
    user_id: str
    split: str
    task: str
    model: str
    exact_runtime_text_tokens: int
    expected_memory_tokens: int
    framing_reserve_tokens: int
    expected_input_tokens: int
    maximum_input_tokens: int
    expected_output_tokens: int
    maximum_output_tokens: int
    expected_cost_usd: str
    maximum_cost_usd: str

    def __post_init__(self) -> None:
        _sha(self.estimate_id, "estimate ID")
        for value, label in (
            (self.batch_id, "batch ID"), (self.record_id, "record ID"),
            (self.user_id, "user ID"), (self.model, "model"),
        ):
            _safe(value, label)
        if self.position <= 0 or type(self.position) is not int:
            raise FrozenPreflightError("estimate position is invalid")
        if self.split not in {"development", "test"}:
            raise FrozenPreflightError("estimate split is invalid")
        if self.task not in {*TASK_ORDER, "extraction"}:
            raise FrozenPreflightError("estimate task is invalid")
        counts = (
            self.exact_runtime_text_tokens, self.expected_memory_tokens,
            self.framing_reserve_tokens, self.expected_input_tokens,
            self.maximum_input_tokens, self.expected_output_tokens,
            self.maximum_output_tokens,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise FrozenPreflightError("token estimate contains an invalid count")
        if self.expected_input_tokens > self.maximum_input_tokens:
            raise FrozenPreflightError("expected input exceeds maximum input")
        if self.expected_output_tokens > self.maximum_output_tokens:
            raise FrozenPreflightError("expected output exceeds maximum output")
        _money(self.expected_cost_usd, "expected cost")
        _money(self.maximum_cost_usd, "maximum cost")
        _json_safe(self)


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    position: int
    stage: str
    baseline_id: str | None
    task: str
    model: str
    request_count: int
    maximum_retry_requests: int
    input_token_estimate: int
    input_token_maximum: int
    expected_output_tokens: int
    maximum_output_tokens: int
    expected_cost_usd: str
    maximum_cost_usd: str
    output_directory: str
    checkpoint_policy: str
    provider_execution_authorized: bool

    def __post_init__(self) -> None:
        _safe(self.batch_id, "batch ID")
        _safe(self.model, "batch model")
        _safe(self.output_directory, "output directory")
        if self.position <= 0 or type(self.position) is not int:
            raise FrozenPreflightError("batch position is invalid")
        if self.stage not in {"extraction", "answer"}:
            raise FrozenPreflightError("batch stage is invalid")
        if self.stage == "extraction":
            if self.baseline_id is not None or self.task != "extraction":
                raise FrozenPreflightError("extraction batch identity is invalid")
        elif self.baseline_id not in BASELINE_ORDER or self.task not in TASK_ORDER:
            raise FrozenPreflightError("answer batch identity is invalid")
        counts = (
            self.request_count, self.maximum_retry_requests,
            self.input_token_estimate, self.input_token_maximum,
            self.expected_output_tokens, self.maximum_output_tokens,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise FrozenPreflightError("batch contains an invalid count")
        if not self.request_count or self.maximum_retry_requests:
            raise FrozenPreflightError("batch request or retry policy changed")
        if self.input_token_estimate > self.input_token_maximum:
            raise FrozenPreflightError("batch expected input exceeds maximum")
        if self.expected_output_tokens > self.maximum_output_tokens:
            raise FrozenPreflightError("batch expected output exceeds maximum")
        _money(self.expected_cost_usd, "batch expected cost")
        _money(self.maximum_cost_usd, "batch maximum cost")
        if self.checkpoint_policy != "after_each_success_no_successful_replay":
            raise FrozenPreflightError("checkpoint policy changed")
        if self.provider_execution_authorized:
            raise FrozenPreflightError("preflight cannot authorize provider execution")
        _json_safe(self)


@dataclass(frozen=True)
class PreflightChecks:
    preflight_version: str
    batch_count: int
    extraction_batch_count: int
    answer_batch_count: int
    planned_request_count: int
    maximum_retry_requests: int
    transmission_record_count: int
    source_count: int
    case_count: int
    development_case_count: int
    frozen_test_case_count: int
    provider_request_count: int
    gold_opened: bool
    oracle_opened: bool
    review_queue_opened: bool
    credential_reuse_approved: bool
    data_transmission_approved: bool
    paid_execution_approved: bool

    def __post_init__(self) -> None:
        if self.preflight_version != PREFLIGHT_VERSION:
            raise FrozenPreflightError("checks version changed")
        expected = (25, 1, 24, 4660, 0, 4660, 100, 570, 114, 456)
        actual = (
            self.batch_count, self.extraction_batch_count, self.answer_batch_count,
            self.planned_request_count, self.maximum_retry_requests,
            self.transmission_record_count, self.source_count, self.case_count,
            self.development_case_count, self.frozen_test_case_count,
        )
        if actual != expected:
            raise FrozenPreflightError("preflight checks counts changed")
        if self.provider_request_count:
            raise FrozenPreflightError("preflight made a provider request")
        if self.gold_opened or self.oracle_opened or self.review_queue_opened:
            raise FrozenPreflightError("preflight opened scorer-only data")
        if not self.credential_reuse_approved:
            raise FrozenPreflightError("credential reuse decision is missing")
        if self.data_transmission_approved or self.paid_execution_approved:
            raise FrozenPreflightError("preflight cannot self-approve transmission or spend")


def canonical_json_bytes(value: object) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    _json_safe(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_json_bytes(raw: bytes, *, location: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw, parse_constant=lambda item: _raise(f"non-finite JSON at {location}: {item}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenPreflightError(f"unsafe JSON at {location}") from error
    if not isinstance(value, dict):
        raise FrozenPreflightError(f"JSON object required at {location}")
    _json_safe(value)
    return value


def transmission_from_mapping(value: Mapping[str, object]) -> TransmissionRecord:
    _strict(value, TransmissionRecord)
    return TransmissionRecord(
        transmission_id=_string(value["transmission_id"]),
        kind=_string(value["kind"]), record_id=_string(value["record_id"]),
        user_id=_string(value["user_id"]), split=_string(value["split"]),
        baseline_id=_optional_string(value["baseline_id"]),
        task=_optional_string(value["task"]),
        source_type=_optional_string(value["source_type"]),
        fields_transmitted=_strings(value["fields_transmitted"]),
        synthetic_benchmark=_boolean(value["synthetic_benchmark"]),
        restricted_content=_boolean(value["restricted_content"]),
    )


def estimate_from_mapping(value: Mapping[str, object]) -> TokenEstimate:
    _strict(value, TokenEstimate)
    return TokenEstimate(**value)  # type: ignore[arg-type]


def batch_from_mapping(value: Mapping[str, object]) -> BatchPlan:
    _strict(value, BatchPlan)
    return BatchPlan(**value)  # type: ignore[arg-type]


def checks_from_mapping(value: Mapping[str, object]) -> PreflightChecks:
    _strict(value, PreflightChecks)
    return PreflightChecks(**value)  # type: ignore[arg-type]


def money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.0000001')):.7f}"


def calculate_cost(input_tokens: int, output_tokens: int, input_rate: str, output_rate: str) -> str:
    total = (Decimal(input_tokens) * Decimal(input_rate) + Decimal(output_tokens) * Decimal(output_rate)) / Decimal(1_000_000)
    return money(total)


def _strict(value: Mapping[str, object], cls: type[object]) -> None:
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise FrozenPreflightError(f"{cls.__name__} fields changed")


def _ordered_unique(values: tuple[str, ...], label: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise FrozenPreflightError(f"{label} must be sorted and unique")
    for value in values:
        _safe(value, label)


def _safe(value: str, label: str) -> None:
    if not SAFE_ID.fullmatch(value):
        raise FrozenPreflightError(f"{label} is unsafe")


def _sha(value: str, label: str) -> None:
    if not SHA256.fullmatch(value):
        raise FrozenPreflightError(f"{label} is invalid")


def _money(value: str, label: str) -> None:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise FrozenPreflightError(f"{label} is invalid") from error
    if parsed < 0 or value != money(parsed):
        raise FrozenPreflightError(f"{label} is not canonical")


def _json_safe(value: object) -> None:
    if is_dataclass(value):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FrozenPreflightError("non-finite JSON value")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise FrozenPreflightError("JSON object key is not a string")
        for item in value.values():
            _json_safe(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_safe(item)
        return
    raise FrozenPreflightError(f"unsafe JSON type: {type(value).__name__}")


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise FrozenPreflightError("string required")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FrozenPreflightError("string array required")
    return tuple(value)


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise FrozenPreflightError("boolean required")
    return value


def _raise(message: str) -> object:
    raise FrozenPreflightError(message)
