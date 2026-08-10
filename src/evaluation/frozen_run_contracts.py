"""Strict runtime records for the approved Step 10.3 frozen run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:+/-]{1,240}$")


class FrozenRunError(RuntimeError):
    """Reject an unsafe, changed, incomplete, or non-canonical runtime."""


@dataclass(frozen=True)
class ProviderRecord:
    response_id: str
    request_id: str | None
    requested_model: str
    returned_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    pacing_delay_seconds: str
    latency_ms: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.response_id, "response ID"),
            (self.requested_model, "requested model"),
            (self.returned_model, "returned model"),
        ):
            _safe_id(value, label)
        if self.request_id is not None:
            _safe_id(self.request_id, "request ID")
        for value, label in (
            (self.input_tokens, "input tokens"),
            (self.output_tokens, "output tokens"),
            (self.total_tokens, "total tokens"),
            (self.latency_ms, "latency"),
        ):
            if type(value) is not int or value < 0:
                raise FrozenRunError(f"{label} is invalid")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise FrozenRunError("provider token total is inconsistent")
        _nonnegative_decimal(self.pacing_delay_seconds, "pacing delay")


@dataclass(frozen=True)
class ExtractionPrediction:
    prediction_id: str
    batch_id: str
    position: int
    source_id: str
    user_id: str
    split: str
    source_sha256: str
    request_sha256: str
    claims: tuple[Mapping[str, object], ...]
    normalization_diagnostics: tuple[Mapping[str, str], ...]
    provider: ProviderRecord

    def __post_init__(self) -> None:
        _sha(self.prediction_id, "prediction ID")
        _sha(self.source_sha256, "source hash")
        _sha(self.request_sha256, "request hash")
        for value, label in (
            (self.batch_id, "batch ID"),
            (self.source_id, "source ID"),
            (self.user_id, "user ID"),
        ):
            _safe_id(value, label)
        if type(self.position) is not int or self.position <= 0:
            raise FrozenRunError("prediction position is invalid")
        if self.split not in {"development", "test"}:
            raise FrozenRunError("prediction split is invalid")
        if not all(isinstance(item, Mapping) for item in self.claims):
            raise FrozenRunError("prediction claims are invalid")
        if not all(
            isinstance(item, Mapping) and set(item) == {"code", "location"}
            and all(isinstance(value, str) and value for value in item.values())
            for item in self.normalization_diagnostics
        ):
            raise FrozenRunError("normalization diagnostics are invalid")
        _json_safe(self)
        expected = stable_sha256(prediction_payload(self))
        if self.prediction_id != expected:
            raise FrozenRunError("prediction ID does not recompute")


@dataclass(frozen=True)
class ExtractionFailure:
    failure_id: str
    batch_id: str
    position: int
    source_id: str
    user_id: str
    stage: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _sha(self.failure_id, "failure ID")
        for value, label in (
            (self.batch_id, "batch ID"),
            (self.source_id, "source ID"),
            (self.user_id, "user ID"),
            (self.code, "failure code"),
            (self.location, "failure location"),
        ):
            _safe_id(value, label)
        if type(self.position) is not int or self.position <= 0:
            raise FrozenRunError("failure position is invalid")
        if self.stage not in {
            "provider_pending", "provider", "model_mismatch", "validation", "cost_cap"
        }:
            raise FrozenRunError("failure stage is invalid")
        _json_safe(self)
        payload = asdict(self)
        identity = payload.pop("failure_id")
        if identity != stable_sha256(payload):
            raise FrozenRunError("failure ID does not recompute")


def prediction_payload(record: ExtractionPrediction) -> Mapping[str, object]:
    payload = asdict(record)
    payload.pop("prediction_id")
    return payload


def provider_from_mapping(value: object) -> ProviderRecord:
    mapping = _strict_mapping(value, ProviderRecord, "provider")
    return ProviderRecord(**mapping)


def prediction_from_mapping(value: object) -> ExtractionPrediction:
    mapping = _strict_mapping(value, ExtractionPrediction, "prediction")
    mapping["claims"] = tuple(_mapping(item, "claim") for item in _list(mapping["claims"], "claims"))
    mapping["normalization_diagnostics"] = tuple(
        _mapping(item, "normalization diagnostic")
        for item in _list(mapping["normalization_diagnostics"], "normalization diagnostics")
    )
    mapping["provider"] = provider_from_mapping(mapping["provider"])
    return ExtractionPrediction(**mapping)


def failure_from_mapping(value: object) -> ExtractionFailure:
    return ExtractionFailure(**_strict_mapping(value, ExtractionFailure, "failure"))


def canonical_json_bytes(value: object) -> bytes:
    _json_safe(value)
    prepared = asdict(value) if is_dataclass(value) else value
    return (
        json.dumps(
            prepared, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_json_bytes(raw: bytes, *, location: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: _reject_constant(item))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenRunError(f"{location} is not valid JSON") from error
    return _mapping(value, location)


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def money(value: Decimal | str) -> str:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise FrozenRunError("money value is invalid") from error
    if not parsed.is_finite() or parsed < 0:
        raise FrozenRunError("money value is invalid")
    return f"{parsed:.7f}"


def _strict_mapping(value: object, cls: type, label: str) -> dict[str, object]:
    mapping = dict(_mapping(value, label))
    expected = {field.name for field in fields(cls)}
    if set(mapping) != expected:
        raise FrozenRunError(f"{label} fields changed")
    return mapping


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrozenRunError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise FrozenRunError(f"{label} must be an array")
    return value


def _safe_id(value: object, label: str) -> None:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise FrozenRunError(f"{label} is invalid")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise FrozenRunError(f"{label} is invalid")


def _nonnegative_decimal(value: object, label: str) -> None:
    try:
        parsed = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError) as error:
        raise FrozenRunError(f"{label} is invalid") from error
    if not parsed.is_finite() or parsed < 0:
        raise FrozenRunError(f"{label} is invalid")


def _json_safe(value: object) -> None:
    prepared = asdict(value) if is_dataclass(value) else value
    try:
        json.dumps(prepared, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FrozenRunError("value is not safe finite JSON") from error
    _reject_nonfinite(prepared)


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FrozenRunError("value is not finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FrozenRunError("JSON object keys must be strings")
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)


def _reject_constant(value: str) -> object:
    raise FrozenRunError(f"non-finite JSON constant is forbidden: {value}")
