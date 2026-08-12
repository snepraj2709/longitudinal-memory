"""Build manual review packets for the scaled-v1 data foundation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


SCALED_ROOT = Path("data/scaled-v1")
DEFAULT_OUTPUT_ROOT = SCALED_ROOT / "review_packets"
PACKET_SCHEMA_VERSION = "scaled_review_packet_v1"
GOLD_INPUTS = (
    ("gold_claims", SCALED_ROOT / "gold/claims.jsonl"),
    ("gold_qa", SCALED_ROOT / "gold/qa.jsonl"),
    ("gold_summaries", SCALED_ROOT / "gold/summaries.jsonl"),
    ("gold_interactive", SCALED_ROOT / "gold/interactive.jsonl"),
)
ORACLE_INPUTS = (("oracle_events", SCALED_ROOT / "oracle/events.jsonl"),)
REVIEW_QUEUE_NAMES = (
    "abstentions",
    "conflicts",
    "corrections",
    "evidence",
    "interactive_expectations",
    "summaries",
    "timelines",
)
REVIEW_INPUTS = tuple((f"review_{name}", SCALED_ROOT / f"review_queues/{name}.jsonl") for name in REVIEW_QUEUE_NAMES)
PACKET_INPUTS = GOLD_INPUTS + ORACLE_INPUTS + REVIEW_INPUTS


def build_review_packets(repo_root: str | Path) -> Mapping[str, list[dict[str, Any]]]:
    """Return review packets grouped by output name without changing files."""

    root = Path(repo_root).resolve()
    runtime_sources = _jsonl(root / SCALED_ROOT / "runtime/sources.jsonl")
    source_index = _source_index(runtime_sources)
    rows = {name: _jsonl(root / path) for name, path in PACKET_INPUTS}
    target_index = _target_index(rows)
    packets: dict[str, list[dict[str, Any]]] = {}
    for name, relative_path in GOLD_INPUTS:
        packets[name] = [
            _gold_packet(name, relative_path, row, index, source_index, target_index)
            for index, row in enumerate(rows[name], 1)
        ]
    for name, relative_path in ORACLE_INPUTS:
        packets[name] = [
            _oracle_packet(relative_path, row, index)
            for index, row in enumerate(rows[name], 1)
        ]
    for name, relative_path in REVIEW_INPUTS:
        queue = name.removeprefix("review_")
        packets[name] = [
            _review_queue_packet(queue, relative_path, row, index, source_index, target_index)
            for index, row in enumerate(rows[name], 1)
        ]
    return packets


def write_review_packets(
    repo_root: str | Path,
    output_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Write packet JSONL files and a compact index."""

    root = Path(repo_root).resolve()
    destination = root / (output_root or DEFAULT_OUTPUT_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    packets = build_review_packets(root)
    files: list[dict[str, Any]] = []
    for name in sorted(packets):
        path = destination / f"{name}.jsonl"
        _write_jsonl(path, packets[name])
        files.append({
            "name": name,
            "path": _display_path(path, root),
            "records": len(packets[name]),
            "pending_review": sum(1 for packet in packets[name] if packet["current_status"] == "pending_human_review"),
        })
    index = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "pending_manual_review",
        "source_dataset": str(SCALED_ROOT),
        "output_root": _display_path(destination, root),
        "total_packets": sum(item["records"] for item in files),
        "files": files,
        "review_instruction": "Review each packet against the cited source text, then patch the source data and statuses explicitly. These packets do not approve any row.",
    }
    (destination / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def _gold_packet(
    name: str,
    source_path: Path,
    row: Mapping[str, Any],
    line_number: int,
    source_index: Mapping[tuple[str, str | None], Mapping[str, Any]],
    target_index: Mapping[str, Any],
) -> dict[str, Any]:
    target_id = str(row.get("claim_id") or row.get("case_id"))
    packet = _base_packet(source_path, line_number, name, target_id, row.get("user_id"), row.get("review_status"))
    packet.update({
        "review_action": "verify_expected_value_evidence_and_status_before_approval",
        "risk_tags": _gold_risk_tags(row),
        "expected": _summarise_target(row),
        "source_evidence": _hydrate_evidence(row.get("evidence"), source_index),
        "linked_claims": [_summarise_target(target_index["claims"].get(claim_id)) for claim_id in _list(row.get("required_claim_ids"))],
        "linked_events": [_summarise_target(target_index["events"].get(event_id)) for event_id in _list(row.get("gold_event_ids"))],
    })
    return packet


def _oracle_packet(source_path: Path, row: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    event_id = str(row.get("event_id"))
    packet = _base_packet(source_path, line_number, "oracle_event", event_id, row.get("user_id"), "pending_human_review")
    packet.update({
        "review_action": "verify_event_facts_time_links_and_disclosure_status",
        "risk_tags": _oracle_risk_tags(row),
        "expected": _summarise_target(row),
        "source_evidence": [],
        "linked_claims": [],
        "linked_events": [],
    })
    return packet


def _review_queue_packet(
    queue: str,
    source_path: Path,
    row: Mapping[str, Any],
    line_number: int,
    source_index: Mapping[tuple[str, str | None], Mapping[str, Any]],
    target_index: Mapping[str, Any],
) -> dict[str, Any]:
    target_records = _review_targets(row, target_index)
    packet = _base_packet(source_path, line_number, f"review_queue_{queue}", row.get("target_id"), row.get("user_id"), row.get("status"))
    packet.update({
        "review_action": "resolve_review_queue_item_then_patch_source_row",
        "risk_tags": [queue, str(row.get("target_type"))] + [str(check) for check in _list(row.get("checks"))],
        "expected": {
            "review_id": row.get("review_id"),
            "queue": queue,
            "target_type": row.get("target_type"),
            "target_id": row.get("target_id"),
            "checks": row.get("checks"),
            "notes": row.get("notes"),
        },
        "source_evidence": _target_evidence(target_records, source_index),
        "target_records": [_summarise_target(record) for record in target_records],
        "linked_claims": [],
        "linked_events": [],
    })
    return packet


def _base_packet(
    source_path: Path,
    line_number: int,
    packet_type: str,
    target_id: object,
    user_id: object,
    current_status: object,
) -> dict[str, Any]:
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": f"{packet_type}:{target_id}",
        "source_path": source_path.as_posix(),
        "source_line": line_number,
        "packet_type": packet_type,
        "target_id": target_id,
        "user_id": user_id,
        "current_status": current_status,
        "allowed_statuses_after_review": ["approved", "resolved", "requires_patch"],
    }


def _source_index(sources: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str | None], dict[str, Any]]:
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("source_id"))
        source_common = {
            "source_id": source_id,
            "source_type": source.get("source_type"),
            "user_id": source.get("user_id"),
            "created_at": source.get("created_at"),
        }
        messages = source.get("messages")
        if isinstance(messages, list) and messages:
            for message in messages:
                message_id = str(message.get("message_id"))
                index[(source_id, message_id)] = {
                    **source_common,
                    "message_id": message_id,
                    "speaker_id": message.get("speaker_id"),
                    "source_text": message.get("text"),
                }
        else:
            index[(source_id, None)] = {
                **source_common,
                "message_id": None,
                "speaker_id": None,
                "source_text": source.get("content"),
            }
    return index


def _target_index(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
    gold_cases = {
        str(row.get("case_id")): row
        for name in ("gold_qa", "gold_summaries", "gold_interactive")
        for row in rows[name]
    }
    events = {str(row.get("event_id")): row for row in rows["oracle_events"]}
    return {
        "claims": {str(row.get("claim_id")): row for row in rows["gold_claims"]},
        "cases": gold_cases,
        "events": events,
        "events_by_user": _group_by_user(rows["oracle_events"]),
    }


def _review_targets(row: Mapping[str, Any], target_index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    target_id = row.get("target_id")
    target_type = row.get("target_type")
    if target_type == "claim":
        return _maybe_record(target_index["claims"].get(target_id))
    if target_type in {"qa_case", "summary_case", "interactive_case"}:
        return _maybe_record(target_index["cases"].get(target_id))
    if target_type == "claim_pair" and isinstance(target_id, str):
        return [
            target_index["claims"][claim_id]
            for claim_id in target_id.split("__")
            if claim_id in target_index["claims"]
        ]
    if target_type == "user_timeline":
        return list(target_index["events_by_user"].get(str(target_id), ()))
    return []


def _target_evidence(
    records: Sequence[Mapping[str, Any]],
    source_index: Mapping[tuple[str, str | None], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for record in records:
        for item in _hydrate_evidence(record.get("evidence"), source_index):
            key = (item.get("source_id"), item.get("message_id"), item.get("quote"))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    return evidence


def _hydrate_evidence(
    evidence: object,
    source_index: Mapping[tuple[str, str | None], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    hydrated = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id"))
        raw_message_id = item.get("message_id")
        message_id = str(raw_message_id) if raw_message_id is not None else None
        source = source_index.get((source_id, message_id)) or source_index.get((source_id, None), {})
        quote = item.get("quote")
        source_text = source.get("source_text")
        hydrated.append({
            "source_id": source_id,
            "message_id": message_id,
            "speaker_id": source.get("speaker_id"),
            "source_type": source.get("source_type"),
            "created_at": source.get("created_at"),
            "quote": quote,
            "source_text": source_text,
            "exact_quote_match": isinstance(quote, str) and isinstance(source_text, str) and _normalise(quote) in _normalise(source_text),
        })
    return hydrated


def _summarise_target(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not row:
        return {}
    fields = (
        "claim_id",
        "case_id",
        "event_id",
        "user_id",
        "subject_id",
        "speaker_id",
        "predicate",
        "object",
        "polarity",
        "epistemic_status",
        "memory_kind",
        "status",
        "valid_from",
        "valid_to",
        "question",
        "reference_answer",
        "acceptable_answers",
        "should_abstain",
        "abstention_reason",
        "instruction",
        "reference_summary",
        "initial_user_message",
        "scenario",
        "expected_behaviours",
        "event_type",
        "facts",
        "caused_by",
        "superseded_by",
        "disclosure_status",
        "required_claim_ids",
        "gold_event_ids",
        "failure_tags",
        "review_status",
    )
    return {field: row[field] for field in fields if field in row}


def _gold_risk_tags(row: Mapping[str, Any]) -> list[str]:
    tags = [str(row.get("task") or "claim"), str(row.get("capability") or row.get("predicate"))]
    tags.extend(str(tag) for tag in _list(row.get("failure_tags")))
    if row.get("should_abstain") is True:
        tags.append("abstention")
    if row.get("review_status") == "pending_human_review":
        tags.append("pending_review")
    return [tag for tag in tags if tag and tag != "None"]


def _oracle_risk_tags(row: Mapping[str, Any]) -> list[str]:
    tags = [str(row.get("event_type")), str(row.get("disclosure_status"))]
    if row.get("caused_by"):
        tags.append("causal_link")
    if row.get("superseded_by"):
        tags.append("superseded")
    return [tag for tag in tags if tag and tag != "None"]


def _group_by_user(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("user_id")), []).append(row)
    return {user_id: tuple(records) for user_id, records in grouped.items()}


def _maybe_record(value: object) -> list[Mapping[str, Any]]:
    return [value] if isinstance(value, Mapping) else []


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalise(value: str) -> str:
    return " ".join(value.split())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build scaled-v1 manual review packets.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or __import__("sys").stdout
    args = _build_parser().parse_args(argv)
    index = write_review_packets(args.repo_root, args.output_root)
    print(json.dumps({
        "status": "written",
        "output_root": index["output_root"],
        "total_packets": index["total_packets"],
        "files": {item["name"]: item["records"] for item in index["files"]},
    }, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
