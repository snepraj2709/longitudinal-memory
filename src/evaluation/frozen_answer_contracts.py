"""Strict prediction records for the Step 10.3 frozen answer batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Mapping

from .frozen_run_contracts import (
    FrozenRunError,
    ProviderRecord,
    provider_from_mapping,
    stable_sha256,
)


STATUSES = {"answered", "abstained", "disputed", "partially_answered"}
TASK_BODY = {"qa": "answer", "summary": "summary", "interactive": "response"}


@dataclass(frozen=True)
class FrozenAnswerPrediction:
    prediction_id: str
    batch_id: str
    position: int
    baseline_id: str
    task: str
    case_id: str
    user_id: str
    split: str
    context_sha256: str
    context_count: int
    request_sha256: str
    generation_mode: str
    output: Mapping[str, object]
    provider: ProviderRecord | None

    def __post_init__(self) -> None:
        if self.position <= 0 or type(self.position) is not int:
            raise FrozenRunError("answer prediction position is invalid")
        if self.baseline_id not in {f"B{index}" for index in range(8)}:
            raise FrozenRunError("answer prediction baseline is invalid")
        if self.task not in TASK_BODY:
            raise FrozenRunError("answer prediction task is invalid")
        if self.split not in {"development", "test"}:
            raise FrozenRunError("answer prediction split is invalid")
        if self.context_count < 0 or type(self.context_count) is not int:
            raise FrozenRunError("answer prediction context count is invalid")
        if self.generation_mode not in {"provider", "deterministic_answerability_gate"}:
            raise FrozenRunError("answer generation mode is invalid")
        if (self.generation_mode == "provider") != (self.provider is not None):
            raise FrozenRunError("answer provider metadata is inconsistent")
        validate_answer_output(self.output, task=self.task, evidence_index=None)
        payload = asdict(self)
        identity = payload.pop("prediction_id")
        if identity != stable_sha256(payload):
            raise FrozenRunError("answer prediction ID does not recompute")


@dataclass(frozen=True)
class FrozenAnswerFailure:
    failure_id: str
    batch_id: str
    position: int
    baseline_id: str
    task: str
    case_id: str
    user_id: str
    stage: str
    code: str
    location: str

    def __post_init__(self) -> None:
        if self.stage not in {"provider", "model_mismatch", "validation", "cost_cap"}:
            raise FrozenRunError("answer failure stage is invalid")
        payload = asdict(self)
        identity = payload.pop("failure_id")
        if identity != stable_sha256(payload):
            raise FrozenRunError("answer failure ID does not recompute")


def answer_prediction_from_mapping(value: object) -> FrozenAnswerPrediction:
    mapping = _strict(value, FrozenAnswerPrediction, "answer prediction")
    provider = mapping["provider"]
    mapping["provider"] = None if provider is None else provider_from_mapping(provider)
    if not isinstance(mapping["output"], Mapping):
        raise FrozenRunError("answer output must be an object")
    return FrozenAnswerPrediction(**mapping)


def answer_failure_from_mapping(value: object) -> FrozenAnswerFailure:
    return FrozenAnswerFailure(**_strict(value, FrozenAnswerFailure, "answer failure"))


def validate_answer_output(
    value: object,
    *,
    task: str,
    evidence_index: Mapping[tuple[str, str | None, str], object] | None,
) -> Mapping[str, object]:
    """Validate model output and return a canonical JSON-safe mapping."""

    if task not in TASK_BODY or not isinstance(value, Mapping):
        raise FrozenRunError("answer output is invalid")
    body_field = TASK_BODY[task]
    expected = {
        "status", body_field, "confidence", "statements", "citations",
        "unresolved_parts", "abstention_reason",
    }
    if set(value) != expected:
        raise FrozenRunError("answer output fields changed")
    status = value["status"]
    body = value[body_field]
    confidence = value["confidence"]
    statements = value["statements"]
    citations = value["citations"]
    unresolved = value["unresolved_parts"]
    reason = value["abstention_reason"]
    if status not in STATUSES or not isinstance(body, str) or not body.strip():
        raise FrozenRunError("answer status or body is invalid")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise FrozenRunError("answer confidence is invalid")
    if not isinstance(statements, list) or not all(isinstance(item, str) and item.strip() for item in statements):
        raise FrozenRunError("answer statements are invalid")
    if len(statements) != len(set(statements)):
        raise FrozenRunError("answer statements are duplicated")
    if not isinstance(unresolved, list) or not all(isinstance(item, str) and item.strip() for item in unresolved):
        raise FrozenRunError("answer unresolved parts are invalid")
    if len(unresolved) != len(set(unresolved)):
        raise FrozenRunError("answer unresolved parts are duplicated")
    if not isinstance(citations, list):
        raise FrozenRunError("answer citations are invalid")
    canonical_citations = []
    seen = set()
    for citation in citations:
        if not isinstance(citation, Mapping) or set(citation) != {"source_id", "message_id", "quote"}:
            raise FrozenRunError("answer citation fields changed")
        source_id = citation["source_id"]
        message_id = citation["message_id"]
        quote = citation["quote"]
        if not isinstance(source_id, str) or not source_id or not isinstance(quote, str) or not quote:
            raise FrozenRunError("answer citation identity is invalid")
        if message_id is not None and (not isinstance(message_id, str) or not message_id):
            raise FrozenRunError("answer citation message ID is invalid")
        key = (source_id, message_id, quote)
        if key in seen:
            raise FrozenRunError("answer citation is duplicated")
        if evidence_index is not None and key not in evidence_index:
            raise FrozenRunError("answer citation is outside the supplied context")
        seen.add(key)
        canonical_citations.append({"source_id": source_id, "message_id": message_id, "quote": quote})
    if status == "abstained":
        if statements or citations or unresolved or confidence != 0 or not isinstance(reason, str) or not reason.strip():
            raise FrozenRunError("answer abstention is invalid")
    else:
        if not statements or not citations or reason is not None:
            raise FrozenRunError("grounded answer is incomplete")
        if any(statement not in body for statement in statements):
            raise FrozenRunError("answer contains untracked statement text")
        if status == "disputed" and len(citations) < 2:
            raise FrozenRunError("disputed answer lacks conflicting evidence")
        if status == "partially_answered" and not unresolved:
            raise FrozenRunError("partial answer lacks unresolved parts")
        if status == "answered" and unresolved:
            raise FrozenRunError("answered output has unresolved parts")
    return {
        "status": status,
        body_field: body,
        "confidence": confidence,
        "statements": list(statements),
        "citations": sorted(
            canonical_citations,
            key=lambda item: (item["source_id"], item["message_id"] or "", item["quote"]),
        ),
        "unresolved_parts": sorted(unresolved),
        "abstention_reason": reason,
    }


def _strict(value: object, cls: type, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FrozenRunError(f"{label} must be an object")
    result = dict(value)
    if set(result) != {item.name for item in fields(cls)}:
        raise FrozenRunError(f"{label} fields changed")
    return result
