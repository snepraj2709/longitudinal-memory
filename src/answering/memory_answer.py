"""Deterministic abstention and strict grounded-candidate validation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping

from retrieval.query_contracts import RequestedValidTime

from .answer_contracts import (
    ANSWER_VERSION,
    BLOCKERS,
    CONFIG_VERSION,
    DORMANT_MODEL,
    FIXED_ABSTENTION_TEXT,
    INPUT_RELEASE_VERSION,
    PROMPT_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    AnswerCitation,
    AnswerClaimReference,
    AnswerPackageView,
    AnswerStatement,
    CandidateStatement,
    MemoryAnswer,
    MemoryAnswerCandidate,
    MemoryAnswerError,
    answer_id_payload,
    statement_id,
)
from .answer_input import PROMPT_SHA256, render_answer_prompt


CONFIG_PATH = Path("configs/answering/memory_answer_v1.json")
ABSTENTION_TEXT = FIXED_ABSTENTION_TEXT
ABSTENTION_REASONS = {
    "no_retrieved_claims": "The available memory does not contain evidence for this question.",
    "incomplete_evidence": "The available memory does not contain a complete evidence trail for this question.",
    "no_promoted_claims": "The available memory contains only unconfirmed claims, so it cannot support an answer.",
    "clarification_required": "The question needs clarification before the available memory can support an answer.",
}
MODEL = DORMANT_MODEL


def load_memory_answer_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[Mapping[str, object], str]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryAnswerError("memory answer config is invalid") from error
    expected = {
        "answer_version", "schema_version", "config_version", "prompt_version",
        "runtime_version", "input_release_version", "output_release_version",
        "statuses", "blocker_priority", "abstention_text", "abstention_reasons",
        "provider_execution_enabled", "requested_model", "resolved_model",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise MemoryAnswerError("memory answer config fields changed")
    if (
        value["answer_version"] != ANSWER_VERSION
        or value["schema_version"] != SCHEMA_VERSION
        or value["config_version"] != CONFIG_VERSION
        or value["prompt_version"] != PROMPT_VERSION
        or value["runtime_version"] != RUNTIME_VERSION
        or value["input_release_version"] != INPUT_RELEASE_VERSION
        or value["output_release_version"] != "memory_answer_contract_development_v1"
        or tuple(value["statuses"]) != ("answered", "abstained", "disputed", "partially_answered")
        or tuple(value["blocker_priority"]) != BLOCKERS
        or value["abstention_text"] != ABSTENTION_TEXT
        or value["abstention_reasons"] != ABSTENTION_REASONS
        or value["provider_execution_enabled"] is not False
        or value["requested_model"] != MODEL
        or value["resolved_model"] != MODEL
    ):
        raise MemoryAnswerError("memory answer policy changed")
    return value, hashlib.sha256(raw).hexdigest()


def build_memory_answer(
    view: AnswerPackageView,
    candidate: MemoryAnswerCandidate | None = None,
    *,
    config_path: str | Path = CONFIG_PATH,
) -> MemoryAnswer:
    config, config_sha256 = load_memory_answer_config(config_path)
    if not view.answer_allowed:
        blocker = next((item for item in BLOCKERS if item in view.structural_blockers), None)
        if blocker is None:
            raise MemoryAnswerError("blocked package has no structural reason")
        return _answer(
            view, config_sha256, "deterministic_structural_abstention",
            None, None, "abstained", ABSTENTION_TEXT, 0.0, (), (),
            ABSTENTION_REASONS[blocker],
        )
    if candidate is None:
        raise MemoryAnswerError("allowed package requires a supplied candidate")
    render_answer_prompt(view)
    _validate_candidate(view, candidate)
    statements = tuple(AnswerStatement(
        statement_id(item), item.text, item.claim_references, item.citations
    ) for item in candidate.statements)
    return _answer(
        view, config_sha256, "validated_supplied_candidate",
        str(config["requested_model"]), str(config["resolved_model"]),
        candidate.status, candidate.answer, float(candidate.confidence), statements,
        candidate.unresolved_parts, candidate.abstention_reason,
    )


def _validate_candidate(view: AnswerPackageView, candidate: MemoryAnswerCandidate) -> None:
    claim_index = {
        (item.category, item.claim_id, item.claim_version_id): item for item in view.claims
    }
    span_index = {item.evidence_id: item for item in view.evidence_spans}
    for statement in candidate.statements:
        references = {
            (item.category, item.claim_id, item.claim_version_id)
            for item in statement.claim_references
        }
        citation_pairs = set()
        for reference in statement.claim_references:
            if (reference.category, reference.claim_id, reference.claim_version_id) not in claim_index:
                raise MemoryAnswerError("candidate references a claim outside the package")
        for citation in statement.citations:
            matching = {
                key for key in references if key[1:] == (citation.claim_id, citation.claim_version_id)
            }
            if len(matching) != 1:
                raise MemoryAnswerError("citation has no exact claim reference")
            span = span_index.get(citation.evidence_id)
            if span is None or (
                span.claim_id, span.claim_version_id, span.source_id, span.span_id,
                span.message_id, span.quote,
            ) != (
                citation.claim_id, citation.claim_version_id, citation.source_id,
                citation.span_id, citation.message_id, citation.quote,
            ):
                raise MemoryAnswerError("citation does not match package evidence")
            claim = claim_index[next(iter(matching))]
            if citation.evidence_id not in claim.evidence_ids:
                raise MemoryAnswerError("citation is not evidence for the claim")
            citation_pairs.add((citation.claim_id, citation.claim_version_id))
        if {(key[1], key[2]) for key in references} != citation_pairs:
            raise MemoryAnswerError("every claim reference needs an exact citation")
    if candidate.status == "abstained":
        if (
            candidate.statements or candidate.unresolved_parts or candidate.confidence != 0
            or not candidate.abstention_reason or candidate.answer != ABSTENTION_TEXT
        ):
            raise MemoryAnswerError("candidate abstention is invalid")
        return
    if not candidate.statements or candidate.abstention_reason is not None:
        raise MemoryAnswerError("candidate lacks grounded statements")
    if candidate.answer != "\n".join(item.text for item in candidate.statements):
        raise MemoryAnswerError("candidate answer contains untracked prose")
    categories = [
        reference.category
        for statement in candidate.statements
        for reference in statement.claim_references
    ]
    if candidate.status == "answered" and (
        candidate.unresolved_parts or "conflicting_claims" in categories
    ):
        raise MemoryAnswerError("answered candidate is inconsistent")
    if candidate.status == "disputed":
        versions = {
            (reference.claim_id, reference.claim_version_id)
            for statement in candidate.statements
            for reference in statement.claim_references
        }
        if any(item != "conflicting_claims" for item in categories) or len(versions) < 2:
            raise MemoryAnswerError("disputed candidate is incomplete")
    if candidate.status == "partially_answered" and not candidate.unresolved_parts:
        raise MemoryAnswerError("partial candidate has no unresolved part")


def _answer(
    view: AnswerPackageView,
    config_sha256: str,
    generation_mode: str,
    requested_model: str | None,
    resolved_model: str | None,
    status: str,
    answer: str,
    confidence: float,
    statements: tuple[AnswerStatement, ...],
    unresolved_parts: tuple[str, ...],
    abstention_reason: str | None,
) -> MemoryAnswer:
    payload = {
        "answer_version": ANSWER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_sha256": config_sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": view.input_release_version,
        "input_release_manifest_sha256": view.input_release_manifest_sha256,
        "input_packages_sha256": view.input_packages_sha256,
        "package_id": view.package_id,
        "package_sha256": view.package_sha256,
        "user_id": view.user_id,
        "query_id": view.query_id,
        "query_text": view.query_text,
        "baseline_id": view.baseline_id,
        "execution_id": view.execution_id,
        "plan_id": view.plan_id,
        "snapshot_run_id": view.snapshot_run_id,
        "as_of": view.as_of,
        "requested_valid_time": view.requested_valid_time,
        "generation_mode": generation_mode,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "statements": statements,
        "unresolved_parts": unresolved_parts,
        "abstention_reason": abstention_reason,
        "structural_blockers": view.structural_blockers,
    }
    return MemoryAnswer(answer_id_payload(payload), **payload)


def memory_answer_candidate_from_mapping(value: object) -> MemoryAnswerCandidate:
    row = _strict_mapping(value, {
        "status", "answer", "confidence", "statements", "unresolved_parts",
        "abstention_reason",
    }, "candidate")
    statements = row["statements"]
    unresolved = row["unresolved_parts"]
    if not isinstance(statements, list) or not isinstance(unresolved, list):
        raise MemoryAnswerError("candidate collections are invalid")
    return MemoryAnswerCandidate(
        _string(row["status"], "candidate status"),
        row["answer"] if isinstance(row["answer"], str) else _invalid("candidate answer"),
        row["confidence"],
        tuple(_candidate_statement(item) for item in statements),
        tuple(_string(item, "unresolved part") for item in unresolved),
        None if row["abstention_reason"] is None else _string(
            row["abstention_reason"], "abstention reason"
        ),
    )


def _candidate_statement(value: object) -> CandidateStatement:
    row = _strict_mapping(value, {"text", "claim_references", "citations"}, "statement")
    if not isinstance(row["claim_references"], list) or not isinstance(row["citations"], list):
        raise MemoryAnswerError("statement provenance collections are invalid")
    return CandidateStatement(
        _string(row["text"], "statement text"),
        tuple(_claim_reference(item) for item in row["claim_references"]),
        tuple(_citation(item) for item in row["citations"]),
    )


def _claim_reference(value: object) -> AnswerClaimReference:
    row = _strict_mapping(value, {"claim_id", "claim_version_id", "category"}, "claim reference")
    return AnswerClaimReference(*(_string(row[name], name) for name in (
        "claim_id", "claim_version_id", "category"
    )))


def _citation(value: object) -> AnswerCitation:
    fields = {
        "claim_id", "claim_version_id", "evidence_id", "source_id", "span_id",
        "message_id", "quote",
    }
    row = _strict_mapping(value, fields, "citation")
    return AnswerCitation(
        _string(row["claim_id"], "claim ID"),
        _string(row["claim_version_id"], "claim version ID"),
        _string(row["evidence_id"], "evidence ID"),
        _string(row["source_id"], "source ID"),
        _string(row["span_id"], "span ID"),
        None if row["message_id"] is None else _string(row["message_id"], "message ID"),
        _string(row["quote"], "quote"),
    )


def memory_answer_from_mapping(value: object) -> MemoryAnswer:
    fields = {
        "answer_id", "answer_version", "schema_version", "config_version",
        "config_sha256", "prompt_version", "prompt_sha256", "runtime_version",
        "input_release_version", "input_release_manifest_sha256", "input_packages_sha256",
        "package_id", "package_sha256", "user_id", "query_id", "query_text",
        "baseline_id", "execution_id", "plan_id", "snapshot_run_id", "as_of",
        "requested_valid_time", "generation_mode", "requested_model", "resolved_model",
        "status", "answer", "confidence", "statements", "unresolved_parts",
        "abstention_reason", "structural_blockers",
    }
    row = _strict_mapping(value, fields, "answer")
    statements = row["statements"]
    if not isinstance(statements, list):
        raise MemoryAnswerError("answer statements are invalid")
    parsed_statements = []
    for value_statement in statements:
        statement = _strict_mapping(
            value_statement, {"statement_id", "text", "claim_references", "citations"}, "answer statement"
        )
        candidate = _candidate_statement({
            "text": statement["text"],
            "claim_references": statement["claim_references"],
            "citations": statement["citations"],
        })
        parsed_statements.append(AnswerStatement(
            _string(statement["statement_id"], "statement ID"), candidate.text,
            candidate.claim_references, candidate.citations,
        ))
    requested = _requested_valid_time(row["requested_valid_time"])
    unresolved = row["unresolved_parts"]
    blockers = row["structural_blockers"]
    if not isinstance(unresolved, list) or not isinstance(blockers, list):
        raise MemoryAnswerError("answer collections are invalid")
    return MemoryAnswer(
        *(_string(row[name], name) for name in (
            "answer_id", "answer_version", "schema_version", "config_version",
            "config_sha256", "prompt_version", "prompt_sha256", "runtime_version",
            "input_release_version", "input_release_manifest_sha256", "input_packages_sha256",
            "package_id", "package_sha256", "user_id", "query_id", "query_text",
            "baseline_id", "execution_id", "plan_id", "snapshot_run_id",
        )),
        _datetime(row["as_of"]), requested,
        _string(row["generation_mode"], "generation mode"),
        None if row["requested_model"] is None else _string(row["requested_model"], "requested model"),
        None if row["resolved_model"] is None else _string(row["resolved_model"], "resolved model"),
        _string(row["status"], "status"),
        row["answer"] if isinstance(row["answer"], str) else _invalid("answer text"),
        row["confidence"], tuple(parsed_statements),
        tuple(_string(item, "unresolved part") for item in unresolved),
        None if row["abstention_reason"] is None else _string(row["abstention_reason"], "abstention reason"),
        tuple(_string(item, "structural blocker") for item in blockers),
    )


def _requested_valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    row = _strict_mapping(value, {
        "kind", "point_date", "point_timestamp", "range_start_date",
        "range_start_timestamp", "range_end_date", "range_end_timestamp",
    }, "requested valid time")
    return RequestedValidTime(
        kind=_string(row["kind"], "valid-time kind"),
        point_date=_date_or_none(row["point_date"]),
        point_timestamp=_datetime_or_none(row["point_timestamp"]),
        range_start_date=_date_or_none(row["range_start_date"]),
        range_end_date=_date_or_none(row["range_end_date"]),
        range_start_timestamp=_datetime_or_none(row["range_start_timestamp"]),
        range_end_timestamp=_datetime_or_none(row["range_end_timestamp"]),
    )


def _strict_mapping(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise MemoryAnswerError(f"{name} fields are invalid")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryAnswerError(f"{name} is invalid")
    return value


def _invalid(name: str):
    raise MemoryAnswerError(f"{name} is invalid")


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise MemoryAnswerError("datetime is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MemoryAnswerError("datetime is naive")
    return parsed


def _datetime_or_none(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _date_or_none(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MemoryAnswerError("date is invalid")
    return date.fromisoformat(value)
