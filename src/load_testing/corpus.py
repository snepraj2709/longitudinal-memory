"""Stream and validate the isolated R3 load-test corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterator, Mapping, Sequence, TextIO


CORPUS_VERSION = "load_test_v1"
DATASET_ROLE = "load_test"
DATA_ROOT = Path("data/load-test-v1")
SCHEMA_ROOT = Path("schemas/load-test-v1")
MANIFEST_PATH = DATA_ROOT / "manifest.json"
EVENT_COUNT = 500_000
USER_COUNT = 2_000
CHUNK_SIZE = 5_000
CHUNK_COUNT = 100
SEED = 20260808
START_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
SOURCE_TYPES = ("conversation", "email", "calendar", "chat")
SCHEMA_NAMES = ("manifest", "source_event", "user")
QUALITY_RELEASE_HASHES = {
    "pilot_v0_user_sha256": "ec951239b54197c76c3fd048564b18494c2d97a3d632a63b985e3c2c9c247140",
    "benchmark_v1_dataset_sha256": "1a0c6db251dc43d793662a919be18b0c7d32f01aad2fb52a086c2ba8fb1f18de",
    "scaled_v1_dataset_sha256": "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61",
}
USER_FIELDS = frozenset({"user_id", "corpus_version", "dataset_role", "timezone", "cohort"})
EVENT_FIELDS = frozenset({"source_id", "corpus_version", "dataset_role", "user_id", "idempotency_key", "source_type", "session_id", "created_at", "ingested_at", "participants", "content", "metadata"})
METADATA_FIELDS = frozenset({"operation_class", "template_id", "topic_key", "sequence_number", "source_reference", "replay_batch"})
MANIFEST_FIELDS = frozenset({"manifest_version", "corpus_version", "dataset_role", "release_status", "quality_denominator_eligible", "contains_evaluation_records", "generator", "generation_date", "counts", "time_range", "files", "corpus_sha256", "protected_quality_releases", "review"})
FORBIDDEN_QUALITY_KEYS = frozenset({"case_id", "capability", "question", "reference_answer", "acceptable_answers", "should_abstain", "abstention_reason", "evidence", "gold_event_ids", "required_claim_ids", "reference_summary", "expected_behaviours", "failure_tags", "split"})


class LoadCorpusError(ValueError):
    """Aggregate load-corpus contract failures."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class LoadCorpusReport:
    """Deterministic validation evidence for the R3 corpus."""

    corpus_version: str
    corpus_sha256: str
    users: int
    source_events: int
    sessions: int
    event_chunks: int
    source_type_counts: Mapping[str, int]
    operation_counts: Mapping[str, int]
    time_start: str
    time_end: str
    declared_bytes: int
    human_review_status: str


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


def reject_quality_use(manifest: Mapping[str, object]) -> None:
    """Reject a load-test manifest at any quality-evaluation boundary."""

    if manifest.get("dataset_role") == DATASET_ROLE:
        raise LoadCorpusError(("load_test data cannot enter quality evaluation or scoring",))


def iter_load_events(
    repo_root: str | Path,
    *,
    data_root: Path = DATA_ROOT,
) -> Iterator[dict[str, Any]]:
    """Yield load events one at a time from manifest-declared chunk files."""

    root = Path(repo_root).resolve()
    physical_data_root = _resolve_under(root, data_root)
    manifest = _read_json(physical_data_root / "manifest.json")
    _require_load_manifest(manifest)
    for item in manifest["files"]:
        if item.get("layer") != "load_test_source":
            continue
        logical = item.get("path")
        if not isinstance(logical, str):
            raise LoadCorpusError(("manifest source path must be a string",))
        try:
            suffix = PurePosixPath(logical).relative_to("data/load-test-v1")
        except ValueError as error:
            raise LoadCorpusError((f"manifest source path is outside the load corpus: {logical}",)) from error
        path = _resolve_under(physical_data_root, Path(*suffix.parts))
        yield from _iter_jsonl(path)


def validate_load_corpus(
    repo_root: str | Path,
    *,
    data_root: Path = DATA_ROOT,
    schema_root: Path = SCHEMA_ROOT,
) -> LoadCorpusReport:
    """Validate counts, identities, references, ordering, hashes, and separation."""

    root = Path(repo_root).resolve()
    physical_data_root = _resolve_under(root, data_root)
    physical_schema_root = _resolve_under(root, schema_root)
    errors: list[str] = []
    try:
        manifest = _read_json(physical_data_root / "manifest.json")
    except LoadCorpusError as error:
        raise error
    _validate_manifest_shape(manifest, errors)

    declared_files = _declared_files(manifest, errors)
    expected_paths = {"data/load-test-v1/users.jsonl"}
    expected_paths.update(f"data/load-test-v1/events/events-{number:05d}.jsonl" for number in range(CHUNK_COUNT))
    expected_paths.update(f"schemas/load-test-v1/{name}.schema.json" for name in SCHEMA_NAMES)
    if set(declared_files) != expected_paths:
        errors.append("manifest file list does not match the fixed load-corpus file set")

    users_path = physical_data_root / "users.jsonl"
    _validate_users(users_path, declared_files.get("data/load-test-v1/users.jsonl"), errors)

    source_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    total_events = 0
    last_created: datetime | None = None
    for chunk_number in range(CHUNK_COUNT):
        logical = f"data/load-test-v1/events/events-{chunk_number:05d}.jsonl"
        path = physical_data_root / "events" / f"events-{chunk_number:05d}.jsonl"
        count, last_created = _validate_event_chunk(
            path,
            chunk_number,
            total_events,
            last_created,
            source_counts,
            operation_counts,
            errors,
        )
        total_events += count
        _validate_file_record(path, declared_files.get(logical), count, logical, errors)
    if total_events != EVENT_COUNT:
        errors.append(f"source event count must be {EVENT_COUNT}, got {total_events}")

    for name in SCHEMA_NAMES:
        logical = f"schemas/load-test-v1/{name}.schema.json"
        path = physical_schema_root / f"{name}.schema.json"
        try:
            schema = _read_json(path)
        except LoadCorpusError as error:
            errors.extend(error.errors)
            continue
        _validate_schema(name, schema, errors)
        _validate_file_record(path, declared_files.get(logical), 1, logical, errors)

    _validate_counts(manifest, source_counts, operation_counts, errors)
    roots = {"data": physical_data_root, "schema": physical_schema_root}
    try:
        corpus_hash = _stream_hash(roots, manifest.get("files", []))
    except (LoadCorpusError, OSError) as error:
        errors.extend(error.errors if isinstance(error, LoadCorpusError) else (str(error),))
        corpus_hash = ""
    if manifest.get("corpus_sha256") != corpus_hash:
        errors.append("manifest corpus_sha256 does not match declared files")
    if errors:
        raise LoadCorpusError(errors)

    counts = manifest["counts"]
    return LoadCorpusReport(
        corpus_version=CORPUS_VERSION,
        corpus_sha256=corpus_hash,
        users=counts["users"],
        source_events=counts["source_events"],
        sessions=counts["sessions"],
        event_chunks=counts["event_chunks"],
        source_type_counts=dict(source_counts),
        operation_counts=dict(operation_counts),
        time_start=manifest["time_range"]["created_at_start"],
        time_end=manifest["time_range"]["created_at_end"],
        declared_bytes=sum(item["bytes"] for item in manifest["files"]),
        human_review_status=manifest["review"]["human_review_status"],
    )


def _validate_users(path: Path, declared: Mapping[str, object] | None, errors: list[str]) -> None:
    count = 0
    try:
        for index, user in enumerate(_iter_jsonl(path)):
            location = f"users line {index + 1}"
            _exact(user, USER_FIELDS, location, errors)
            _snake_keys(user, location, errors)
            _no_quality_fields(user, location, errors)
            expected = f"load_user_{index:04d}"
            if user.get("user_id") != expected:
                errors.append(f"{location}.user_id must be {expected}")
            if user.get("corpus_version") != CORPUS_VERSION or user.get("dataset_role") != DATASET_ROLE:
                errors.append(f"{location} has invalid load-test labels")
            count += 1
    except LoadCorpusError as error:
        errors.extend(error.errors)
    if count != USER_COUNT:
        errors.append(f"user count must be {USER_COUNT}, got {count}")
    _validate_file_record(path, declared, count, "data/load-test-v1/users.jsonl", errors)


def _validate_event_chunk(
    path: Path,
    chunk_number: int,
    start_index: int,
    last_created: datetime | None,
    source_counts: Counter[str],
    operation_counts: Counter[str],
    errors: list[str],
) -> tuple[int, datetime | None]:
    count = 0
    try:
        for offset, event in enumerate(_iter_jsonl(path)):
            index = start_index + offset
            location = f"{path.name} line {offset + 1}"
            _exact(event, EVENT_FIELDS, location, errors)
            _snake_keys(event, location, errors)
            _no_quality_fields(event, location, errors)
            user_number = index % USER_COUNT
            user_sequence = index // USER_COUNT
            user_id = f"load_user_{user_number:04d}"
            expected_source = f"load_v1_source_{index:09d}"
            expected_idempotency = f"load_v1_idempotency_{index:09d}"
            expected_type = SOURCE_TYPES[index % len(SOURCE_TYPES)]
            if event.get("source_id") != expected_source:
                errors.append(f"{location}.source_id must be {expected_source}")
            if event.get("idempotency_key") != expected_idempotency:
                errors.append(f"{location}.idempotency_key must be {expected_idempotency}")
            if event.get("user_id") != user_id:
                errors.append(f"{location}.user_id must be {user_id}")
            if event.get("corpus_version") != CORPUS_VERSION or event.get("dataset_role") != DATASET_ROLE:
                errors.append(f"{location} has invalid load-test labels")
            if event.get("source_type") != expected_type:
                errors.append(f"{location}.source_type must be {expected_type}")
            expected_session = f"{user_id}_session_{user_sequence // 5:05d}"
            if event.get("session_id") != expected_session:
                errors.append(f"{location}.session_id must be {expected_session}")
            created = _timestamp(event.get("created_at"), f"{location}.created_at", errors)
            ingested = _timestamp(event.get("ingested_at"), f"{location}.ingested_at", errors)
            expected_created = START_TIME + timedelta(minutes=index)
            if created != expected_created:
                errors.append(f"{location}.created_at is outside the fixed chronology")
            expected_ingested = expected_created + timedelta(minutes=index % 7)
            if ingested != expected_ingested:
                errors.append(f"{location}.ingested_at is outside the fixed ingestion chronology")
            if last_created and created and created <= last_created:
                errors.append(f"{location}.created_at breaks chronological ordering")
            if created:
                last_created = created
            participants = event.get("participants")
            topic_number = (index * 2_654_435_761 + SEED) % 10_000
            expected_participants = [user_id, f"load_contact_{topic_number % 500:03d}"]
            if participants != expected_participants:
                errors.append(f"{location}.participants do not match the fixed user scope")
            if not isinstance(event.get("content"), str) or not event["content"].strip():
                errors.append(f"{location}.content must be a non-empty string")
            metadata = event.get("metadata")
            _exact(metadata, METADATA_FIELDS, f"{location}.metadata", errors)
            if isinstance(metadata, dict):
                expected_operation = _operation(user_sequence)
                if metadata.get("operation_class") != expected_operation:
                    errors.append(f"{location}.metadata.operation_class must be {expected_operation}")
                if metadata.get("sequence_number") != user_sequence:
                    errors.append(f"{location}.metadata.sequence_number is invalid")
                if metadata.get("replay_batch") != index // 1_000:
                    errors.append(f"{location}.metadata.replay_batch is invalid")
                variant = ((index * 1_664_525 + SEED) & 0xFFFFFFFF) % 3
                if metadata.get("template_id") != f"{expected_type}_{variant + 1:02d}":
                    errors.append(f"{location}.metadata.template_id is invalid")
                if metadata.get("topic_key") != f"topic_{topic_number:04d}":
                    errors.append(f"{location}.metadata.topic_key is invalid")
                reference = metadata.get("source_reference")
                if expected_operation in {"correction", "repeated_observation"}:
                    expected_reference = f"load_v1_source_{index - USER_COUNT:09d}"
                    if reference != expected_reference:
                        errors.append(f"{location}.metadata.source_reference must be {expected_reference}")
                    elif (index - USER_COUNT) % USER_COUNT != user_number:
                        errors.append(f"{location}.metadata.source_reference crosses users")
                elif reference is not None:
                    errors.append(f"{location}.metadata.source_reference must be null")
                operation_counts[str(metadata.get("operation_class"))] += 1
            source_counts[str(event.get("source_type"))] += 1
            count += 1
    except LoadCorpusError as error:
        errors.extend(error.errors)
    expected_count = CHUNK_SIZE
    if count != expected_count:
        errors.append(f"chunk {chunk_number:05d} must contain {expected_count} events, got {count}")
    return count, last_created


def _validate_manifest_shape(manifest: dict[str, Any], errors: list[str]) -> None:
    _exact(manifest, MANIFEST_FIELDS, "manifest", errors)
    _snake_keys(manifest, "manifest", errors)
    _no_quality_fields(manifest, "manifest", errors, allow_manifest_counts=True)
    _require_load_manifest(manifest, errors)
    if manifest.get("manifest_version") != "1":
        errors.append("manifest_version must be 1")
    generator = manifest.get("generator")
    expected_generator = {
        "name": "deterministic_load_corpus",
        "version": "1.0.0",
        "seed": SEED,
        "event_count": EVENT_COUNT,
        "user_count": USER_COUNT,
        "chunk_size": CHUNK_SIZE,
        "source_ordering_rule": "created_at_then_source_id",
        "method": "streamed_deterministic_templates",
        "llm_calls": 0,
    }
    if generator != expected_generator:
        errors.append("manifest generator configuration is invalid")
    if manifest.get("generation_date") != "2026-08-08":
        errors.append("manifest generation_date is invalid")
    if manifest.get("protected_quality_releases") != QUALITY_RELEASE_HASHES:
        errors.append("manifest protected quality hashes are invalid")
    expected_time = {
        "created_at_start": _iso(START_TIME),
        "created_at_end": _iso(START_TIME + timedelta(minutes=EVENT_COUNT - 1)),
    }
    if manifest.get("time_range") != expected_time:
        errors.append("manifest time range is invalid")
    candidate_review = {
        "automated_validation_status": "passed",
        "human_review_status": "pending_sneha_review",
    }
    approved_review = {
        "approved_on": "2026-08-08",
        "automated_validation_status": "passed",
        "human_review_status": "approved",
        "implementation_review_status": "complete",
        "reviewed_by": "Sneha",
        "reviewed_corpus_sha256": "c271b987c53262176f9a1692140ba5caa039cc2533404caf692bd7199b8b957b",
    }
    release_status = manifest.get("release_status")
    review = manifest.get("review")
    if not (
        (release_status == "candidate" and review == candidate_review)
        or (release_status == "frozen" and review == approved_review)
    ):
        errors.append("manifest review state is invalid")


def _require_load_manifest(manifest: Mapping[str, object], errors: list[str] | None = None) -> None:
    found: list[str] = []
    if manifest.get("corpus_version") != CORPUS_VERSION:
        found.append("corpus_version must be load_test_v1")
    if manifest.get("dataset_role") != DATASET_ROLE:
        found.append("dataset_role must be load_test")
    if manifest.get("quality_denominator_eligible") is not False:
        found.append("quality_denominator_eligible must be false")
    if manifest.get("contains_evaluation_records") is not False:
        found.append("contains_evaluation_records must be false")
    if errors is not None:
        errors.extend(found)
    elif found:
        raise LoadCorpusError(found)


def _validate_counts(manifest: Mapping[str, Any], source_counts: Counter[str], operation_counts: Counter[str], errors: list[str]) -> None:
    expected_source_counts = {name: EVENT_COUNT // len(SOURCE_TYPES) for name in SOURCE_TYPES}
    expected_operation_counts = Counter(_operation(sequence) for sequence in range(EVENT_COUNT // USER_COUNT))
    expected_operation_counts = {name: count * USER_COUNT for name, count in sorted(expected_operation_counts.items())}
    expected = {
        "users": USER_COUNT,
        "source_events": EVENT_COUNT,
        "sessions": 100_000,
        "event_chunks": CHUNK_COUNT,
        "source_types": expected_source_counts,
        "operation_classes": expected_operation_counts,
        "quality_cases": 0,
        "quality_gold_records": 0,
    }
    if manifest.get("counts") != expected:
        errors.append("manifest counts do not match the fixed load corpus")
    if dict(source_counts) != expected_source_counts:
        errors.append("observed source-type counts are invalid")
    if dict(sorted(operation_counts.items())) != expected_operation_counts:
        errors.append("observed operation-class counts are invalid")


def _declared_files(manifest: Mapping[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        errors.append("manifest.files must be a list")
        return {}
    declared: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != {"path", "layer", "sha256", "records", "bytes"}:
            errors.append(f"manifest.files[{index}] has invalid fields")
            continue
        path = item.get("path")
        if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            errors.append(f"manifest.files[{index}].path is unsafe")
            continue
        if path in declared:
            errors.append(f"manifest repeats file {path}")
        declared[path] = item
    return declared


def _validate_file_record(path: Path, declared: Mapping[str, object] | None, records: int, logical: str, errors: list[str]) -> None:
    if declared is None:
        return
    try:
        size = path.stat().st_size
        digest = _file_sha256(path)
    except OSError as error:
        errors.append(f"could not hash {logical}: {error}")
        return
    expected_layer = "load_test_source" if "/events/" in logical else "load_test_user" if logical.endswith("users.jsonl") else "load_test_schema"
    if declared.get("layer") != expected_layer:
        errors.append(f"manifest layer does not match {logical}")
    if declared.get("sha256") != digest:
        errors.append(f"manifest hash does not match {logical}")
    if declared.get("records") != records:
        errors.append(f"manifest record count does not match {logical}")
    if declared.get("bytes") != size:
        errors.append(f"manifest byte count does not match {logical}")


def _validate_schema(name: str, schema: Mapping[str, Any], errors: list[str]) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append(f"schema {name} must use JSON Schema draft 2020-12")
    if schema.get("$id") != f"https://longitudinal-memory.local/load-test-v1/{name}.schema.json":
        errors.append(f"schema {name} has an invalid versioned $id")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        errors.append(f"schema {name} must be a strict object")
    if not isinstance(schema.get("required"), list) or not isinstance(schema.get("properties"), dict):
        errors.append(f"schema {name} must declare required fields and properties")


def _no_quality_fields(value: object, location: str, errors: list[str], allow_manifest_counts: bool = False) -> None:
    found = _find_keys(value) & FORBIDDEN_QUALITY_KEYS
    if allow_manifest_counts:
        found.discard("split")
    if found:
        errors.append(f"{location} contains quality-evaluation fields: {', '.join(sorted(found))}")
    forbidden_values = _find_string_values(value) & {"development", "test"}
    if forbidden_values:
        errors.append(f"{location} contains a quality split label")


def _find_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            found.update(_find_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_keys(item))
    return found


def _find_string_values(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_find_string_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_string_values(item))
    return found


def _snake_keys(value: object, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                errors.append(f"{location} contains non-snake_case key {key!r}")
            _snake_keys(item, location, errors)
    elif isinstance(value, list):
        for item in value:
            _snake_keys(item, location, errors)


def _timestamp(value: object, location: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str):
        errors.append(f"{location} must be an ISO 8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{location} must be an ISO 8601 timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{location} must include a UTC offset")
        return None
    return parsed


def _exact(value: object, fields: frozenset[str], location: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{location} must be an object")
        return
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        errors.append(f"{location} missing fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"{location} unknown fields: {', '.join(unknown)}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise LoadCorpusError((f"{path}:{line_number}: blank JSONL line",))
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LoadCorpusError((f"{path}:{line_number}: record must be an object",))
                yield value
    except (OSError, json.JSONDecodeError) as error:
        raise LoadCorpusError((f"could not read {path}: {error}",)) from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LoadCorpusError((f"could not read {path}: {error}",)) from error
    if not isinstance(value, dict):
        raise LoadCorpusError((f"{path} must contain one JSON object",))
    return value


def _resolve_under(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve() if not relative.is_absolute() else relative.resolve()
    if path != root and root not in path.parents:
        raise LoadCorpusError((f"path escapes repository root: {relative}",))
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_hash(roots: Mapping[str, Path], files: object) -> str:
    if not isinstance(files, list):
        raise LoadCorpusError(("manifest.files must be a list",))
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.get("path", "") if isinstance(value, dict) else ""):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise LoadCorpusError(("manifest contains an invalid file record",))
        logical = item["path"]
        if logical.startswith("data/load-test-v1/"):
            path = roots["data"] / logical.removeprefix("data/load-test-v1/")
        elif logical.startswith("schemas/load-test-v1/"):
            path = roots["schema"] / logical.removeprefix("schemas/load-test-v1/")
        else:
            raise LoadCorpusError((f"manifest path is outside load-test roots: {logical}",))
        encoded = logical.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the R3 load-test corpus.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--schema-root", type=Path, default=SCHEMA_ROOT)
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or __import__("sys").stdout
    args = _parser().parse_args(argv)
    try:
        report = validate_load_corpus(
            args.repo_root,
            data_root=args.data_root,
            schema_root=args.schema_root,
        )
    except LoadCorpusError as error:
        print(json.dumps({"status": "failed", "errors": list(error.errors)}), file=stdout)
        return 1
    print(json.dumps({
        "status": "passed",
        "corpus_version": report.corpus_version,
        "corpus_sha256": report.corpus_sha256,
        "counts": {
            "users": report.users,
            "source_events": report.source_events,
            "sessions": report.sessions,
            "event_chunks": report.event_chunks,
            "source_types": report.source_type_counts,
            "operation_classes": report.operation_counts,
        },
        "time_range": {"start": report.time_start, "end": report.time_end},
        "declared_bytes": report.declared_bytes,
        "human_review_status": report.human_review_status,
    }, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
