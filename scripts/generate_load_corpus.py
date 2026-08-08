#!/usr/bin/env python3
"""Generate the deterministic R3 load-test corpus without model calls."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable


CORPUS_VERSION = "load_test_v1"
DATASET_ROLE = "load_test"
GENERATOR_NAME = "deterministic_load_corpus"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260808
DEFAULT_EVENT_COUNT = 500_000
DEFAULT_USER_COUNT = 2_000
DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_GENERATION_DATE = "2026-08-08"
START_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
SOURCE_TYPES = ("conversation", "email", "calendar", "chat")
QUALITY_RELEASE_HASHES = {
    "pilot_v0_user_sha256": "ec951239b54197c76c3fd048564b18494c2d97a3d632a63b985e3c2c9c247140",
    "benchmark_v1_dataset_sha256": "1a0c6db251dc43d793662a919be18b0c7d32f01aad2fb52a086c2ba8fb1f18de",
    "scaled_v1_dataset_sha256": "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61",
}

CONTENT_TEMPLATES = {
    "conversation": (
        "I reviewed {topic} today. The next check-in is slot {slot}.",
        "We talked through {topic}. I will revisit it in slot {slot}.",
        "I finished the first pass on {topic}. The follow-up is slot {slot}.",
    ),
    "email": (
        "Please use revision {revision} for {topic}. I will send the next note in slot {slot}.",
        "The latest email update for {topic} is revision {revision}.",
        "I recorded the {topic} handoff under revision {revision}.",
    ),
    "calendar": (
        "Working session for {topic} in slot {slot}. This is a planning hold, not a deadline.",
        "Review {topic} in slot {slot}. Attendance is optional.",
        "Hold time for {topic} in slot {slot}; details may change.",
    ),
    "chat": (
        "I moved the {topic} check-in to slot {slot}.",
        "Quick update: {topic} is on revision {revision}.",
        "I am still checking {topic}. I should know more by slot {slot}.",
    ),
}


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _operation(user_sequence: int) -> str:
    if user_sequence > 0 and user_sequence % 50 == 0:
        return "correction"
    if user_sequence > 0 and user_sequence % 25 == 0:
        return "repeated_observation"
    if user_sequence % 40 == 0:
        return "retrieval_distractor"
    if user_sequence % 100 == 10:
        return "deletion_candidate"
    if user_sequence % 100 == 20:
        return "reprocessing_candidate"
    return "standard"


def _event(index: int, seed: int, user_count: int) -> dict[str, Any]:
    user_number = index % user_count
    user_sequence = index // user_count
    user_id = f"load_user_{user_number:04d}"
    source_type = SOURCE_TYPES[index % len(SOURCE_TYPES)]
    operation = _operation(user_sequence)
    variant = ((index * 1_664_525 + seed) & 0xFFFFFFFF) % len(CONTENT_TEMPLATES[source_type])
    topic_number = (index * 2_654_435_761 + seed) % 10_000
    topic = f"topic {topic_number:04d}"
    slot = (user_sequence * 7 + seed) % 365
    revision = 1 + (user_sequence % 12)
    content = CONTENT_TEMPLATES[source_type][variant].format(
        topic=topic,
        slot=slot,
        revision=revision,
    )
    reference_index: int | None = None
    if operation in {"correction", "repeated_observation"}:
        reference_index = index - user_count
        prefix = (
            "Correction to the earlier record: "
            if operation == "correction"
            else "Following up on the earlier record: "
        )
        content = prefix + content
    created_at = START_TIME + timedelta(minutes=index)
    ingested_at = created_at + timedelta(minutes=index % 7)
    source_id = f"load_v1_source_{index:09d}"
    return {
        "source_id": source_id,
        "corpus_version": CORPUS_VERSION,
        "dataset_role": DATASET_ROLE,
        "user_id": user_id,
        "idempotency_key": f"load_v1_idempotency_{index:09d}",
        "source_type": source_type,
        "session_id": f"{user_id}_session_{user_sequence // 5:05d}",
        "created_at": _iso(created_at),
        "ingested_at": _iso(ingested_at),
        "participants": [user_id, f"load_contact_{topic_number % 500:03d}"],
        "content": content,
        "metadata": {
            "operation_class": operation,
            "template_id": f"{source_type}_{variant + 1:02d}",
            "topic_key": f"topic_{topic_number:04d}",
            "sequence_number": user_sequence,
            "source_reference": (
                f"load_v1_source_{reference_index:09d}"
                if reference_index is not None
                else None
            ),
            "replay_batch": index // 1_000,
        },
    }


def _user(number: int) -> dict[str, Any]:
    return {
        "user_id": f"load_user_{number:04d}",
        "corpus_version": CORPUS_VERSION,
        "dataset_role": DATASET_ROLE,
        "timezone": "UTC",
        "cohort": f"load_cohort_{number % 20:02d}",
    }


def _schemas() -> dict[str, dict[str, Any]]:
    source_fields = {
        "source_id": {"type": "string"},
        "corpus_version": {"const": CORPUS_VERSION},
        "dataset_role": {"const": DATASET_ROLE},
        "user_id": {"type": "string"},
        "idempotency_key": {"type": "string"},
        "source_type": {"enum": list(SOURCE_TYPES)},
        "session_id": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "ingested_at": {"type": "string", "format": "date-time"},
        "participants": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "content": {"type": "string", "minLength": 1},
        "metadata": {
            "type": "object",
            "required": ["operation_class", "template_id", "topic_key", "sequence_number", "source_reference", "replay_batch"],
            "properties": {
                "operation_class": {"enum": ["standard", "repeated_observation", "correction", "retrieval_distractor", "deletion_candidate", "reprocessing_candidate"]},
                "template_id": {"type": "string"},
                "topic_key": {"type": "string"},
                "sequence_number": {"type": "integer", "minimum": 0},
                "source_reference": {"type": ["string", "null"]},
                "replay_batch": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    }
    user_fields = {
        "user_id": {"type": "string"},
        "corpus_version": {"const": CORPUS_VERSION},
        "dataset_role": {"const": DATASET_ROLE},
        "timezone": {"type": "string"},
        "cohort": {"type": "string"},
    }
    manifest_fields = {
        "manifest_version": {"const": "1"},
        "corpus_version": {"const": CORPUS_VERSION},
        "dataset_role": {"const": DATASET_ROLE},
        "release_status": {"type": "string"},
        "quality_denominator_eligible": {"const": False},
        "contains_evaluation_records": {"const": False},
        "generator": {"type": "object"},
        "generation_date": {"type": "string", "format": "date"},
        "counts": {"type": "object"},
        "time_range": {"type": "object"},
        "files": {"type": "array"},
        "corpus_sha256": {"type": "string"},
        "protected_quality_releases": {"type": "object"},
        "review": {"type": "object"},
    }

    def schema(name: str, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://longitudinal-memory.local/load-test-v1/{name}.schema.json",
            "type": "object",
            "required": list(fields),
            "properties": fields,
            "additionalProperties": False,
        }

    return {
        "source_event": schema("source_event", source_fields),
        "user": schema("user", user_fields),
        "manifest": schema("manifest", manifest_fields),
    }


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_users(path: Path, user_count: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as handle:
        for number in range(user_count):
            payload = _json_bytes(_user(number))
            handle.write(payload)
            digest.update(payload)
            size += len(payload)
    return digest.hexdigest(), size


def _write_events(
    data_root: Path,
    event_count: int,
    user_count: int,
    chunk_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    files: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    total_size = 0
    events_root = data_root / "events"
    events_root.mkdir(parents=True, exist_ok=False)
    for chunk_number, start in enumerate(range(0, event_count, chunk_size)):
        end = min(start + chunk_size, event_count)
        filename = f"events-{chunk_number:05d}.jsonl"
        path = events_root / filename
        digest = hashlib.sha256()
        size = 0
        with path.open("wb") as handle:
            for index in range(start, end):
                event = _event(index, seed, user_count)
                payload = _json_bytes(event)
                handle.write(payload)
                digest.update(payload)
                size += len(payload)
                source_counts[event["source_type"]] += 1
                operation_counts[event["metadata"]["operation_class"]] += 1
        total_size += size
        files.append({
            "path": f"data/load-test-v1/events/{filename}",
            "layer": "load_test_source",
            "sha256": digest.hexdigest(),
            "records": end - start,
            "bytes": size,
        })
    return files, source_counts, operation_counts, total_size


def _write_json(path: Path, value: dict[str, Any]) -> tuple[str, int]:
    payload = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _stream_hash(root_map: dict[str, Path], files: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        relative = item["path"]
        if relative.startswith("data/load-test-v1/"):
            physical = root_map["data"] / relative.removeprefix("data/load-test-v1/")
        else:
            physical = root_map["schema"] / relative.removeprefix("schemas/load-test-v1/")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(physical.stat().st_size.to_bytes(8, "big"))
        with physical.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(value / (1024 * 1024), 2)
    return round(value / 1024, 2)


def generate(
    data_root: Path,
    schema_root: Path,
    seed: int,
    event_count: int,
    user_count: int,
    chunk_size: int,
    generation_date: str,
) -> dict[str, Any]:
    if event_count <= 0 or user_count <= 0 or chunk_size <= 0:
        raise ValueError("event_count, user_count, and chunk_size must be positive")
    if event_count % user_count:
        raise ValueError("event_count must divide evenly across users")
    if data_root.exists() or schema_root.exists():
        raise FileExistsError("output roots must not already exist")
    datetime.fromisoformat(generation_date)
    started = time.perf_counter()
    data_root.mkdir(parents=True)
    schema_root.mkdir(parents=True)

    users_path = data_root / "users.jsonl"
    users_hash, users_size = _write_users(users_path, user_count)
    files = [{
        "path": "data/load-test-v1/users.jsonl",
        "layer": "load_test_user",
        "sha256": users_hash,
        "records": user_count,
        "bytes": users_size,
    }]
    event_files, source_counts, operation_counts, event_bytes = _write_events(
        data_root, event_count, user_count, chunk_size, seed
    )
    files.extend(event_files)
    for name, value in _schemas().items():
        schema_path = schema_root / f"{name}.schema.json"
        sha256, size = _write_json(schema_path, value)
        files.append({
            "path": f"schemas/load-test-v1/{name}.schema.json",
            "layer": "load_test_schema",
            "sha256": sha256,
            "records": 1,
            "bytes": size,
        })

    last_created = START_TIME + timedelta(minutes=event_count - 1)
    manifest = {
        "manifest_version": "1",
        "corpus_version": CORPUS_VERSION,
        "dataset_role": DATASET_ROLE,
        "release_status": "candidate",
        "quality_denominator_eligible": False,
        "contains_evaluation_records": False,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "seed": seed,
            "event_count": event_count,
            "user_count": user_count,
            "chunk_size": chunk_size,
            "source_ordering_rule": "created_at_then_source_id",
            "method": "streamed_deterministic_templates",
            "llm_calls": 0,
        },
        "generation_date": generation_date,
        "counts": {
            "users": user_count,
            "source_events": event_count,
            "sessions": user_count * ((event_count // user_count) // 5),
            "event_chunks": len(event_files),
            "source_types": {name: source_counts[name] for name in SOURCE_TYPES},
            "operation_classes": dict(sorted(operation_counts.items())),
            "quality_cases": 0,
            "quality_gold_records": 0,
        },
        "time_range": {
            "created_at_start": _iso(START_TIME),
            "created_at_end": _iso(last_created),
        },
        "files": files,
        "corpus_sha256": _stream_hash({"data": data_root, "schema": schema_root}, files),
        "protected_quality_releases": QUALITY_RELEASE_HASHES,
        "review": {
            "automated_validation_status": "passed",
            "human_review_status": "pending_sneha_review",
        },
    }
    _write_json(data_root / "manifest.json", manifest)
    return {
        "status": "generated",
        "corpus_sha256": manifest["corpus_sha256"],
        "source_events": event_count,
        "users": user_count,
        "event_chunks": len(event_files),
        "event_bytes": event_bytes,
        "total_declared_bytes": sum(item["bytes"] for item in files),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mib": _peak_rss_mib(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the R3 load-test corpus.")
    parser.add_argument("--data-root", type=Path, default=Path("data/load-test-v1"))
    parser.add_argument("--schema-root", type=Path, default=Path("schemas/load-test-v1"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--event-count", type=int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument("--user-count", type=int, default=DEFAULT_USER_COUNT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--generation-date", default=DEFAULT_GENERATION_DATE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = generate(
            args.data_root,
            args.schema_root,
            args.seed,
            args.event_count,
            args.user_count,
            args.chunk_size,
            args.generation_date,
        )
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
