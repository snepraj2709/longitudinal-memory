"""Deterministic B0-B7 runtime contexts for the frozen scaled comparison."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .frozen_run import DEFAULT_OUTPUT as EXTRACTION_OUTPUT, verify_extraction_batch
from .frozen_run_contracts import (
    FrozenRunError,
    parse_json_bytes,
    prediction_from_mapping,
    stable_sha256,
)


BASELINES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
TASKS = ("qa", "summary", "interactive")
RUNTIME_PATHS = {
    "users": (Path("data/scaled-v1/runtime/users.jsonl"), "13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef"),
    "sources": (Path("data/scaled-v1/runtime/sources.jsonl"), "a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de"),
    "qa": (Path("data/scaled-v1/runtime/qa.jsonl"), "e7112dbda8fb8868f425566c6ee8c6fb0c4f3e4c3c2f718a1c603caaecfee934"),
    "summary": (Path("data/scaled-v1/runtime/summaries.jsonl"), "080e622a3085b1c323dd7aee9565f8c02e0a2895fcd892b175cbbb5acccf1ee5"),
    "interactive": (Path("data/scaled-v1/runtime/interactive.jsonl"), "9fd5ca22a9c7af22f8e4dbbd54c80745d30149751a6b65cf9b41b7de5403d706"),
}
MAX_MEMORY_TOKENS = {
    "B0": 0, "B1": 7000, "B2": 5000, "B3": 4000,
    "B4": 7000, "B5": 7000, "B6": 7000, "B7": 7500,
}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FrozenRuntime:
    users: tuple[Mapping[str, object], ...]
    sources: tuple[Mapping[str, object], ...]
    cases: Mapping[str, tuple[Mapping[str, object], ...]]
    claims: tuple[Mapping[str, object], ...]


def load_frozen_runtime(repo_root: str | Path = ".") -> FrozenRuntime:
    """Load only runtime inputs and the already-frozen extraction predictions."""

    root = Path(repo_root).resolve()
    verify_extraction_batch(repo_root=root, output_dir=root / EXTRACTION_OUTPUT)
    loaded = {}
    for name, (relative, expected) in RUNTIME_PATHS.items():
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise FrozenRunError(f"scaled runtime {name} changed")
        loaded[name] = _jsonl(raw, name)
    predictions = tuple(
        prediction_from_mapping(item)
        for item in _jsonl(
            (root / EXTRACTION_OUTPUT / "predictions.jsonl").read_bytes(),
            "extraction prediction",
        )
    )
    if len(predictions) != 100:
        raise FrozenRunError("frozen extraction predictions are incomplete")
    sources_by_id = {str(item["source_id"]): item for item in loaded["sources"]}
    claims: list[Mapping[str, object]] = []
    for prediction in predictions:
        source = sources_by_id.get(prediction.source_id)
        if source is None or source.get("user_id") != prediction.user_id:
            raise FrozenRunError("extracted claim source ownership changed")
        for claim in prediction.claims:
            claims.append({
                **claim,
                "source_id": prediction.source_id,
                "user_id": prediction.user_id,
                "observed_at": source["created_at"],
            })
    return FrozenRuntime(
        users=loaded["users"], sources=loaded["sources"],
        cases={task: loaded[task] for task in TASKS}, claims=tuple(claims),
    )


def build_context_records(
    runtime: FrozenRuntime,
    *,
    baseline_id: str,
    task: str,
    case: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Build an ordered, same-user, as-of-safe context without scorer input."""

    if baseline_id not in BASELINES or task not in TASKS:
        raise FrozenRunError("unknown baseline or task")
    user_id = str(case["user_id"])
    as_of = _time(case["as_of"])
    query = _case_query(task, case)
    sources = tuple(
        item for item in runtime.sources
        if item.get("user_id") == user_id and _time(item.get("created_at")) <= as_of
    )
    if any(item.get("user_id") != user_id for item in sources):
        raise FrozenRunError("context contains another user")
    if baseline_id == "B0":
        return ()
    if baseline_id == "B1":
        return tuple(_source_records(sources))

    source_ids = {str(item["source_id"]) for item in sources}
    claims = tuple(
        item for item in runtime.claims
        if item.get("user_id") == user_id and item.get("source_id") in source_ids
    )
    lifecycle = _lifecycle(claims, resolve_conflicts=baseline_id in {"B6", "B7"})
    atomic = tuple(
        _atomic_record(item, lifecycle.get(_claim_key(item), "atomic")) for item in claims
    )
    sessions = tuple(_session_record(source, claims) for source in sources)
    if baseline_id == "B2":
        return _rank(query, atomic, 18)
    if baseline_id == "B3":
        return _rank(query, sessions, 8)
    if baseline_id == "B4":
        return _merge_ranked(_rank(query, atomic, 14), _rank(query, sessions, 6))
    temporal_atomic = tuple(
        item for item in atomic if item["lifecycle_status"] in {"current", "historical", "disputed"}
    )
    return _merge_ranked(_rank(query, temporal_atomic, 16), _rank(query, sessions, 5))


def context_evidence_index(
    records: Sequence[Mapping[str, object]],
) -> Mapping[tuple[str, str | None, str], Mapping[str, object]]:
    index = {}
    for record in records:
        evidence = record.get("evidence")
        if not isinstance(evidence, list):
            raise FrozenRunError("context evidence is malformed")
        for item in evidence:
            if not isinstance(item, Mapping) or set(item) != {"source_id", "message_id", "quote"}:
                raise FrozenRunError("context evidence fields changed")
            key = (str(item["source_id"]), item["message_id"], str(item["quote"]))
            index[key] = item
    return index


def _source_records(sources: Sequence[Mapping[str, object]]):
    records = []
    for source in sorted(sources, key=lambda item: (str(item["created_at"]), str(item["source_id"]))):
        source_id = str(source["source_id"])
        messages = source["messages"]
        if messages:
            for message in messages:
                evidence = [{
                    "source_id": source_id,
                    "message_id": message["message_id"],
                    "quote": message["text"],
                }]
                payload = {
                    "record_kind": "source", "source_id": source_id,
                    "message_id": message["message_id"], "observed_at": source["created_at"],
                    "speaker_id": message["speaker_id"], "text": message["text"],
                    "evidence": evidence,
                }
                records.append({"context_id": stable_sha256(payload), **payload})
        else:
            evidence = [{"source_id": source_id, "message_id": None, "quote": source["content"]}]
            payload = {
                "record_kind": "source", "source_id": source_id, "message_id": None,
                "observed_at": source["created_at"], "speaker_id": source["user_id"],
                "text": source["content"], "evidence": evidence,
            }
            records.append({"context_id": stable_sha256(payload), **payload})
    return records


def _atomic_record(claim: Mapping[str, object], lifecycle: str) -> Mapping[str, object]:
    evidence = [dict(item) for item in claim["evidence"]]
    payload = {
        "record_kind": "atomic", "claim_id": claim["claim_id"],
        "source_id": claim["source_id"], "observed_at": claim["observed_at"],
        "subject_id": claim["subject_id"], "speaker_id": claim["speaker_id"],
        "predicate": claim["predicate"], "object": claim["object"],
        "polarity": claim["polarity"], "epistemic_status": claim["epistemic_status"],
        "valid_from": claim["valid_from"], "valid_to": claim["valid_to"],
        "lifecycle_status": lifecycle, "evidence": evidence,
    }
    return {"context_id": stable_sha256(payload), **payload}


def _session_record(
    source: Mapping[str, object], claims: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    source_id = str(source["source_id"])
    selected = sorted(
        (item for item in claims if item["source_id"] == source_id),
        key=lambda item: str(item["claim_id"]),
    )
    statements = [
        f"{item['subject_id']} {item['predicate']} {json.dumps(item['object'], ensure_ascii=False, sort_keys=True)} ({item['polarity']}, {item['epistemic_status']})"
        for item in selected
    ]
    evidence = []
    seen = set()
    for item in selected:
        for span in item["evidence"]:
            key = (span["source_id"], span["message_id"], span["quote"])
            if key not in seen:
                seen.add(key)
                evidence.append(dict(span))
    if not statements:
        statements = [str(source["content"])]
        if source["messages"]:
            evidence = [
                {"source_id": source_id, "message_id": item["message_id"], "quote": item["text"]}
                for item in source["messages"]
            ]
        else:
            evidence = [{"source_id": source_id, "message_id": None, "quote": source["content"]}]
    payload = {
        "record_kind": "session", "session_id": f"session_{source_id}",
        "source_id": source_id, "observed_at": source["created_at"],
        "summary": "\n".join(statements), "evidence": evidence,
    }
    return {"context_id": stable_sha256(payload), **payload}


def _lifecycle(
    claims: Sequence[Mapping[str, object]], *, resolve_conflicts: bool,
) -> Mapping[tuple[str, str, str], str]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for claim in claims:
        grouped.setdefault((str(claim["subject_id"]), str(claim["predicate"])), []).append(claim)
    result = {}
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (str(item["observed_at"]), str(item["source_id"]), str(item["claim_id"])),
        )
        latest_time = str(ordered[-1]["observed_at"])
        latest = [item for item in ordered if str(item["observed_at"]) == latest_time]
        variants = {
            (json.dumps(item["object"], ensure_ascii=False, sort_keys=True), str(item["polarity"]))
            for item in latest
        }
        for item in ordered:
            if str(item["observed_at"]) != latest_time:
                status = "historical"
            elif resolve_conflicts and len(variants) > 1:
                status = "disputed"
            else:
                status = "current"
            result[_claim_key(item)] = status
    return result


def _claim_key(item: Mapping[str, object]) -> tuple[str, str, str]:
    return str(item["source_id"]), str(item["claim_id"]), str(item["user_id"])


def _rank(
    query: str, records: Sequence[Mapping[str, object]], limit: int,
) -> tuple[Mapping[str, object], ...]:
    query_tokens = set(_TOKEN.findall(query.lower()))
    ranked = sorted(
        records,
        key=lambda item: (
            -len(query_tokens.intersection(_TOKEN.findall(_record_text(item).lower()))),
            -_time(item["observed_at"]).timestamp(),
            str(item["context_id"]),
        ),
    )
    return tuple(ranked[:limit])


def _merge_ranked(*groups: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    result = []
    seen = set()
    width = max((len(group) for group in groups), default=0)
    for index in range(width):
        for group in groups:
            if index < len(group) and group[index]["context_id"] not in seen:
                seen.add(group[index]["context_id"])
                result.append(group[index])
    return tuple(result)


def _record_text(record: Mapping[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _case_query(task: str, case: Mapping[str, object]) -> str:
    if task == "qa":
        return str(case["question"])
    if task == "summary":
        return str(case["instruction"])
    return f"{case['scenario']}\n{case['initial_user_message']}"


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise FrozenRunError("runtime timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise FrozenRunError("runtime timestamp is invalid") from error
    if parsed.utcoffset() is None:
        raise FrozenRunError("runtime timestamp must be timezone-aware")
    return parsed


def _jsonl(raw: bytes, location: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        parse_json_bytes(line, location=f"{location}:{index}")
        for index, line in enumerate(raw.splitlines(), 1) if line
    )
