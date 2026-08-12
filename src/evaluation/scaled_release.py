"""Runtime loader and independent deterministic validator for Step R2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .run_config import dataset_sha256


BENCHMARK_VERSION = "scaled_v1"
RELEASE_ROOT = Path("data/scaled-v1")
MANIFEST_PATH = RELEASE_ROOT / "manifest.json"
RUNTIME_PATHS = (
    RELEASE_ROOT / "runtime/users.jsonl",
    RELEASE_ROOT / "runtime/sources.jsonl",
    RELEASE_ROOT / "runtime/qa.jsonl",
    RELEASE_ROOT / "runtime/summaries.jsonl",
    RELEASE_ROOT / "runtime/interactive.jsonl",
)
ORACLE_PATHS = (RELEASE_ROOT / "oracle/events.jsonl",)
GOLD_PATHS = (
    RELEASE_ROOT / "gold/claims.jsonl",
    RELEASE_ROOT / "gold/qa.jsonl",
    RELEASE_ROOT / "gold/summaries.jsonl",
    RELEASE_ROOT / "gold/interactive.jsonl",
)
PREDICTION_PATHS = (RELEASE_ROOT / "predictions/predictions.jsonl",)
REVIEW_NAMES = (
    "corrections",
    "conflicts",
    "abstentions",
    "timelines",
    "summaries",
    "interactive_expectations",
    "evidence",
)
REVIEW_PATHS = tuple(RELEASE_ROOT / f"review_queues/{name}.jsonl" for name in REVIEW_NAMES)
SCHEMA_NAMES = (
    "user",
    "source",
    "runtime_case",
    "oracle_event",
    "gold_claim",
    "gold_case",
    "prediction",
    "review_queue",
    "manifest",
)
SCHEMA_PATHS = tuple(Path(f"schemas/scaled-v1/{name}.schema.json") for name in SCHEMA_NAMES)
RELEASE_DATA_PATHS = RUNTIME_PATHS + ORACLE_PATHS + GOLD_PATHS + PREDICTION_PATHS + REVIEW_PATHS + SCHEMA_PATHS

CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "user_modeling",
    "abstention",
)
USER_IDS = tuple(f"user_{number:03d}" for number in range(1, 11))
DEV_USERS = USER_IDS[:2]
TEST_USERS = USER_IDS[2:]
SOURCE_TYPES = ("conversation", "email", "calendar", "chat")
LEAKAGE_KEYS = frozenset(
    {
        "acceptable_answers",
        "abstention_reason",
        "evidence",
        "expected_behaviours",
        "failure_tags",
        "facts",
        "gold_event_ids",
        "oracle_event_ids",
        "reference_answer",
        "reference_summary",
        "required_claim_ids",
        "review_status",
        "should_abstain",
    }
)

USER_FIELDS = frozenset({"user_id", "display_name", "timezone", "split", "profile_note"})
SOURCE_FIELDS = frozenset({"source_id", "source_type", "user_id", "created_at", "ingested_at", "participants", "content", "messages", "metadata"})
MESSAGE_FIELDS = frozenset({"message_id", "speaker_id", "text"})
COMMON_CASE_FIELDS = frozenset({"case_id", "benchmark_version", "split", "user_id", "task", "capability", "as_of", "difficulty"})
RUNTIME_FIELDS = {
    "qa": COMMON_CASE_FIELDS | {"question"},
    "summarization": COMMON_CASE_FIELDS | {"instruction"},
    "interactive": COMMON_CASE_FIELDS | {"scenario", "initial_user_message", "allowed_turns"},
}
GOLD_FIELDS = {
    "qa": RUNTIME_FIELDS["qa"] | {"reference_answer", "acceptable_answers", "should_abstain", "abstention_reason", "evidence", "required_claim_ids", "gold_event_ids", "failure_tags", "review_status"},
    "summarization": RUNTIME_FIELDS["summarization"] | {"reference_summary", "should_abstain", "evidence", "gold_event_ids", "required_claim_ids", "failure_tags", "review_status"},
    "interactive": RUNTIME_FIELDS["interactive"] | {"should_abstain", "evidence", "required_claim_ids", "expected_behaviours", "failure_tags", "review_status"},
}
CLAIM_FIELDS = frozenset({"claim_id", "benchmark_version", "user_id", "subject_id", "speaker_id", "predicate", "object", "polarity", "epistemic_status", "memory_kind", "status", "valid_from", "valid_to", "time_precision", "evidence", "review_status"})
EVENT_FIELDS = frozenset({"event_id", "benchmark_version", "user_id", "event_type", "valid_from", "valid_to", "facts", "caused_by", "superseded_by", "disclosure_status"})
FACT_FIELDS = frozenset({"fact_id", "subject_id", "predicate", "object"})
REVIEW_FIELDS = frozenset({"review_id", "benchmark_version", "queue", "user_id", "target_type", "target_id", "status", "checks", "notes"})
APPROVED_GOLD_REVIEW_STATUSES = frozenset({"approved", "implementation_reviewed"})
RESOLVED_REVIEW_QUEUE_STATUSES = frozenset({"approved", "resolved"})
PENDING_REVIEW_STATUS = "pending_human_review"


class ScaledReleaseError(ValueError):
    """Aggregate deterministic release validation failures."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class ScaledRuntime:
    """Runtime-visible records for exactly one synthetic user."""

    user: Mapping[str, object]
    sources: tuple[Mapping[str, object], ...]
    qa: tuple[Mapping[str, object], ...]
    summaries: tuple[Mapping[str, object], ...]
    interactive: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ScaledValidationReport:
    """Evidence from a complete scaled-release validation pass."""

    benchmark_version: str
    dataset_sha256: str
    users: int
    sources: int
    qa: int
    summaries: int
    interactive_scenarios: int
    gold_claims: int
    qa_capability_counts: Mapping[str, int]
    summary_type_counts: Mapping[str, int]
    interactive_capability_counts: Mapping[str, int]
    split_counts: Mapping[str, Mapping[str, int]]
    review_queue_counts: Mapping[str, int]
    human_review_status: str


def load_scaled_runtime(repo_root: str | Path, user_id: str) -> ScaledRuntime:
    """Load allowlisted runtime files and return records for one user only."""

    if user_id not in USER_IDS:
        raise ScaledReleaseError((f"unknown scaled benchmark user_id {user_id!r}",))
    root = Path(repo_root).resolve()
    users = _read_jsonl(root / RUNTIME_PATHS[0])
    sources = _read_jsonl(root / RUNTIME_PATHS[1])
    qa = _read_jsonl(root / RUNTIME_PATHS[2])
    summaries = _read_jsonl(root / RUNTIME_PATHS[3])
    interactive = _read_jsonl(root / RUNTIME_PATHS[4])
    errors: list[str] = []
    _validate_runtime(users, sources, qa, summaries, interactive, errors)
    if errors:
        raise ScaledReleaseError(errors)
    user = next(item for item in users if item["user_id"] == user_id)
    return ScaledRuntime(
        user=user,
        sources=tuple(item for item in sources if item["user_id"] == user_id),
        qa=tuple(item for item in qa if item["user_id"] == user_id),
        summaries=tuple(item for item in summaries if item["user_id"] == user_id),
        interactive=tuple(item for item in interactive if item["user_id"] == user_id),
    )


def validate_scaled_release(repo_root: str | Path) -> ScaledValidationReport:
    """Validate R2 counts, IDs, references, time, evidence, review, and hashes."""

    root = Path(repo_root).resolve()
    errors: list[str] = []
    users = _checked_jsonl(root / RUNTIME_PATHS[0], errors)
    sources = _checked_jsonl(root / RUNTIME_PATHS[1], errors)
    runtime_qa = _checked_jsonl(root / RUNTIME_PATHS[2], errors)
    runtime_summaries = _checked_jsonl(root / RUNTIME_PATHS[3], errors)
    runtime_interactive = _checked_jsonl(root / RUNTIME_PATHS[4], errors)
    events = _checked_jsonl(root / ORACLE_PATHS[0], errors)
    claims = _checked_jsonl(root / GOLD_PATHS[0], errors)
    gold_qa = _checked_jsonl(root / GOLD_PATHS[1], errors)
    gold_summaries = _checked_jsonl(root / GOLD_PATHS[2], errors)
    gold_interactive = _checked_jsonl(root / GOLD_PATHS[3], errors)
    predictions = _checked_jsonl(root / PREDICTION_PATHS[0], errors)
    reviews = {name: _checked_jsonl(root / path, errors) for name, path in zip(REVIEW_NAMES, REVIEW_PATHS)}
    schemas = {name: _checked_json(root / path, errors) for name, path in zip(SCHEMA_NAMES, SCHEMA_PATHS)}
    manifest = _checked_json(root / MANIFEST_PATH, errors)
    review = manifest.get("review")
    review_state = review if isinstance(review, dict) else {}
    require_reviewed_gold = review_state.get("human_review_status") == "approved"
    require_resolved_review_queues = review_state.get("review_queue_status") == "approved"

    _validate_runtime(users, sources, runtime_qa, runtime_summaries, runtime_interactive, errors)
    _validate_oracle(events, errors)
    _validate_claims(claims, errors, require_approved_review=require_reviewed_gold)
    runtime_by_task = {"qa": runtime_qa, "summarization": runtime_summaries, "interactive": runtime_interactive}
    gold_by_task = {"qa": gold_qa, "summarization": gold_summaries, "interactive": gold_interactive}
    _validate_gold(gold_by_task, runtime_by_task, claims, events, sources, errors, require_approved_review=require_reviewed_gold)
    _validate_reviews(reviews, gold_by_task, claims, errors, require_resolved_status=require_resolved_review_queues)
    _validate_phenomena(gold_qa, sources, errors)
    _validate_schemas(schemas, errors)
    for path, records in (
        tuple(zip(RUNTIME_PATHS, (users, sources, runtime_qa, runtime_summaries, runtime_interactive)))
        + ((ORACLE_PATHS[0], events),)
        + tuple(zip(GOLD_PATHS, (claims, gold_qa, gold_summaries, gold_interactive)))
        + tuple(zip(REVIEW_PATHS, (reviews[name] for name in REVIEW_NAMES)))
    ):
        _snake_keys(records, str(path), errors)
    if predictions:
        errors.append("candidate prediction file must be empty before any runtime execution")
    _validate_manifest(root, manifest, users, sources, events, claims, runtime_by_task, reviews, errors)
    if errors:
        raise ScaledReleaseError(errors)

    qa_counts = Counter(item["capability"] for item in runtime_qa)
    summary_counts = Counter(item["capability"] for item in runtime_summaries)
    interactive_counts = Counter(item["capability"] for item in runtime_interactive)
    split_counts = {
        split: {
            "qa": sum(item["split"] == split for item in runtime_qa),
            "summaries": sum(item["split"] == split for item in runtime_summaries),
            "interactive_scenarios": sum(item["split"] == split for item in runtime_interactive),
        }
        for split in ("development", "test")
    }
    return ScaledValidationReport(
        benchmark_version=BENCHMARK_VERSION,
        dataset_sha256=dataset_sha256(root, tuple(path.as_posix() for path in RELEASE_DATA_PATHS)),
        users=len(users),
        sources=len(sources),
        qa=len(runtime_qa),
        summaries=len(runtime_summaries),
        interactive_scenarios=len(runtime_interactive),
        gold_claims=len(claims),
        qa_capability_counts={name: qa_counts[name] for name in CAPABILITIES},
        summary_type_counts={"temporal": summary_counts["temporal_reasoning"], "user_model": summary_counts["user_modeling"]},
        interactive_capability_counts={name: interactive_counts[name] for name in CAPABILITIES},
        split_counts=split_counts,
        review_queue_counts={name: len(records) for name, records in reviews.items()},
        human_review_status=manifest["review"]["human_review_status"],
    )


def _validate_runtime(
    users: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    interactive: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if len(users) != 10:
        errors.append(f"runtime users count must be 10, got {len(users)}")
    if [item.get("user_id") for item in users] != list(USER_IDS):
        errors.append("runtime user IDs must be user_001 through user_010 in order")
    for index, user in enumerate(users, 1):
        location = f"runtime user line {index}"
        _exact(user, USER_FIELDS, location, errors)
        expected_split = "development" if user.get("user_id") in DEV_USERS else "test"
        if user.get("split") != expected_split:
            errors.append(f"{location}.split must be {expected_split}")
        _leakage(user, location, errors)

    if len(sources) != 100:
        errors.append(f"runtime source count must be 100, got {len(sources)}")
    source_ids: list[str] = []
    message_ids: list[str] = []
    per_user_types: dict[str, Counter[str]] = defaultdict(Counter)
    for index, source in enumerate(sources, 1):
        location = f"runtime source line {index}"
        _exact(source, SOURCE_FIELDS, location, errors)
        _leakage(source, location, errors)
        source_id = source.get("source_id")
        user_id = source.get("user_id")
        source_type = source.get("source_type")
        if isinstance(source_id, str):
            source_ids.append(source_id)
            if not re.fullmatch(r"scaled_user_\d{3}_(conversation|email|calendar|chat)_\d{3}", source_id):
                errors.append(f"{location}.source_id has an invalid stable-ID shape")
        if user_id not in USER_IDS:
            errors.append(f"{location}.user_id is invalid")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{location}.source_type is invalid")
        elif isinstance(user_id, str):
            per_user_types[user_id][source_type] += 1
        created = _timestamp(source.get("created_at"), f"{location}.created_at", errors)
        ingested = _timestamp(source.get("ingested_at"), f"{location}.ingested_at", errors)
        if created and ingested and ingested < created:
            errors.append(f"{location}.ingested_at precedes created_at")
        if not _strings(source.get("participants"), allow_empty=False):
            errors.append(f"{location}.participants must be a non-empty string list")
        if not isinstance(source.get("content"), str) or not source["content"].strip():
            errors.append(f"{location}.content must be a non-empty string")
        messages = source.get("messages")
        if not isinstance(messages, list):
            errors.append(f"{location}.messages must be a list")
            continue
        if source_type == "calendar" and messages:
            errors.append(f"{location} calendar messages must be empty")
        if source_type != "calendar" and not messages:
            errors.append(f"{location} non-calendar source needs messages")
        for message_index, message in enumerate(messages, 1):
            _exact(message, MESSAGE_FIELDS, f"{location}.messages[{message_index}]", errors)
            message_id = message.get("message_id")
            if isinstance(message_id, str):
                message_ids.append(message_id)
                if not message_id.startswith(f"{source_id}_message_"):
                    errors.append(f"{location}.messages[{message_index}].message_id must be scoped to its source")
    _duplicates(source_ids, "source_id", errors)
    _duplicates(message_ids, "message_id", errors)
    expected_source_ids = {
        f"scaled_{user_id}_{source_type}_{number:03d}"
        for user_id in USER_IDS
        for source_type, maximum in (("conversation", 4), ("email", 2), ("calendar", 2), ("chat", 2))
        for number in range(1, maximum + 1)
    }
    if set(source_ids) != expected_source_ids:
        errors.append("source stable IDs must use each user's fixed type-local sequence")
    expected_source_types = Counter({"conversation": 4, "email": 2, "calendar": 2, "chat": 2})
    for user_id in USER_IDS:
        if per_user_types[user_id] != expected_source_types:
            errors.append(f"{user_id} source balance must be {dict(expected_source_types)}")

    task_sets = (("qa", qa, 500), ("summarization", summaries, 50), ("interactive", interactive, 20))
    all_case_ids: list[str] = []
    for task, records, expected_count in task_sets:
        if len(records) != expected_count:
            errors.append(f"runtime {task} count must be {expected_count}, got {len(records)}")
        per_user = Counter(item.get("user_id") for item in records)
        expected_per_user = {"qa": 50, "summarization": 5, "interactive": 2}[task]
        if any(per_user[user_id] != expected_per_user for user_id in USER_IDS):
            errors.append(f"runtime {task} must contain {expected_per_user} cases per user")
        for index, case in enumerate(records, 1):
            location = f"runtime {task} line {index}"
            _exact(case, RUNTIME_FIELDS[task], location, errors)
            _leakage(case, location, errors)
            _case_common(case, task, location, errors)
            case_id = case.get("case_id")
            if isinstance(case_id, str):
                all_case_ids.append(case_id)
                pattern_task = "qa" if task == "qa" else "summary" if task == "summarization" else "interactive"
                if not re.fullmatch(rf"scaled_user_\d{{3}}_{pattern_task}_[a-z_]+_\d{{3}}", case_id):
                    errors.append(f"{location}.case_id has an invalid stable-ID shape")
            field = "question" if task == "qa" else "instruction" if task == "summarization" else "initial_user_message"
            if not isinstance(case.get(field), str) or not case[field].strip():
                errors.append(f"{location}.{field} must be a non-empty string")
            if task == "interactive" and (not isinstance(case.get("allowed_turns"), int) or case["allowed_turns"] < 1):
                errors.append(f"{location}.allowed_turns must be a positive integer")
    _duplicates(all_case_ids, "runtime case_id", errors)
    expected_qa_ids = {
        f"scaled_{user_id}_qa_{capability}_{number:03d}"
        for user_id in USER_IDS
        for capability in CAPABILITIES
        for number in range(1, 11)
    }
    if {item.get("case_id") for item in qa} != expected_qa_ids:
        errors.append("QA stable IDs must use each user's fixed capability-local sequence")
    expected_summary_ids: set[str] = set()
    for user_number, user_id in enumerate(USER_IDS, 1):
        temporal_total, model_total = (3, 2) if user_number % 2 else (2, 3)
        expected_summary_ids.update(f"scaled_{user_id}_summary_temporal_reasoning_{number:03d}" for number in range(1, temporal_total + 1))
        expected_summary_ids.update(f"scaled_{user_id}_summary_user_modeling_{number:03d}" for number in range(1, model_total + 1))
    if {item.get("case_id") for item in summaries} != expected_summary_ids:
        errors.append("summary stable IDs do not match the frozen per-user sequence")
    pair_cycle = (
        ("extraction", "temporal_reasoning"),
        ("conflict_detection", "abstention"),
        ("user_modeling", "extraction"),
        ("temporal_reasoning", "conflict_detection"),
        ("abstention", "user_modeling"),
    )
    expected_interactive_ids = {
        f"scaled_{user_id}_interactive_{capability}_{position:03d}"
        for user_number, user_id in enumerate(USER_IDS, 1)
        for position, capability in enumerate(pair_cycle[(user_number - 1) % 5], 1)
    }
    if {item.get("case_id") for item in interactive} != expected_interactive_ids:
        errors.append("interactive stable IDs do not match the frozen per-user sequence")

    qa_counts = Counter(item.get("capability") for item in qa)
    if qa_counts != Counter({name: 100 for name in CAPABILITIES}):
        errors.append("QA capability balance must be exactly 100 per capability")
    for user_id in USER_IDS:
        user_counts = Counter(item.get("capability") for item in qa if item.get("user_id") == user_id)
        if user_counts != Counter({name: 10 for name in CAPABILITIES}):
            errors.append(f"{user_id} must have 10 QA cases per capability")
    summary_counts = Counter(item.get("capability") for item in summaries)
    if summary_counts != Counter({"temporal_reasoning": 25, "user_modeling": 25}):
        errors.append("summaries must contain 25 temporal and 25 user-model cases")
    interactive_counts = Counter(item.get("capability") for item in interactive)
    if interactive_counts != Counter({name: 4 for name in CAPABILITIES}):
        errors.append("interactive cases must contain four cases per capability")
    _validate_split_records(qa, summaries, interactive, errors)


def _validate_split_records(qa: list[dict[str, Any]], summaries: list[dict[str, Any]], interactive: list[dict[str, Any]], errors: list[str]) -> None:
    for task, records, dev_count, test_count in (
        ("qa", qa, 100, 400),
        ("summaries", summaries, 10, 40),
        ("interactive", interactive, 4, 16),
    ):
        counts = Counter(item.get("split") for item in records)
        if counts != Counter({"development": dev_count, "test": test_count}):
            errors.append(f"{task} split counts must be development={dev_count}, test={test_count}")
        for item in records:
            expected = "development" if item.get("user_id") in DEV_USERS else "test"
            if item.get("split") != expected:
                errors.append(f"{item.get('case_id')} splits one user's records across datasets")


def _validate_oracle(events: list[dict[str, Any]], errors: list[str]) -> None:
    if len(events) != 120:
        errors.append(f"oracle event count must be 120, got {len(events)}")
    ids: list[str] = []
    fact_ids: list[str] = []
    for index, event in enumerate(events, 1):
        location = f"oracle event line {index}"
        _exact(event, EVENT_FIELDS, location, errors)
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            ids.append(event_id)
            if not re.fullmatch(r"scaled_user_\d{3}_event_\d{3}", event_id):
                errors.append(f"{location}.event_id has an invalid stable-ID shape")
        _date_or_timestamp(event.get("valid_from"), f"{location}.valid_from", errors)
        _date_or_timestamp(event.get("valid_to"), f"{location}.valid_to", errors, allow_none=True)
        facts = event.get("facts")
        if not isinstance(facts, list) or not facts:
            errors.append(f"{location}.facts must be a non-empty list")
            continue
        for fact in facts:
            _exact(fact, FACT_FIELDS, f"{location}.fact", errors)
            if isinstance(fact.get("fact_id"), str):
                fact_ids.append(fact["fact_id"])
            _snake(fact.get("predicate"), f"{location}.fact.predicate", errors)
    _duplicates(ids, "oracle event_id", errors)
    _duplicates(fact_ids, "oracle fact_id", errors)
    expected_ids = {f"scaled_{user_id}_event_{number:03d}" for user_id in USER_IDS for number in range(1, 13)}
    if set(ids) != expected_ids:
        errors.append("oracle stable IDs must use 001 through 012 for every user")
    event_map = {item.get("event_id"): item for item in events}
    for event in events:
        for reference in _list(event.get("caused_by")) + ([event.get("superseded_by")] if event.get("superseded_by") else []):
            target = event_map.get(reference)
            if target is None:
                errors.append(f"oracle event {event.get('event_id')} references unknown event {reference!r}")
            elif target.get("user_id") != event.get("user_id"):
                errors.append(f"oracle event {event.get('event_id')} crosses the user boundary")


def _validate_claims(claims: list[dict[str, Any]], errors: list[str], *, require_approved_review: bool = False) -> None:
    if len(claims) != 160:
        errors.append(f"gold claim count must be 160, got {len(claims)}")
    ids: list[str] = []
    bad_review_statuses: Counter[str] = Counter()
    bad_review_examples: list[str] = []
    per_user = Counter(item.get("user_id") for item in claims)
    if any(per_user[user_id] != 16 for user_id in USER_IDS):
        errors.append("gold claims must contain 16 claims per user")
    for index, claim in enumerate(claims, 1):
        location = f"gold claim line {index}"
        _exact(claim, CLAIM_FIELDS, location, errors)
        claim_id = claim.get("claim_id")
        if isinstance(claim_id, str):
            ids.append(claim_id)
            if not re.fullmatch(r"scaled_user_\d{3}_claim_\d{3}", claim_id):
                errors.append(f"{location}.claim_id has an invalid stable-ID shape")
        if claim.get("benchmark_version") != BENCHMARK_VERSION:
            errors.append(f"{location}.benchmark_version is invalid")
        if claim.get("user_id") not in USER_IDS:
            errors.append(f"{location}.user_id is invalid")
        _snake(claim.get("predicate"), f"{location}.predicate", errors)
        if claim.get("polarity") not in {"positive", "negative"}:
            errors.append(f"{location}.polarity is invalid")
        if claim.get("epistemic_status") not in {"asserted", "reported_by_other", "hypothetical", "uncertain", "denied", "corrected", "inferred"}:
            errors.append(f"{location}.epistemic_status is invalid")
        if claim.get("memory_kind") not in {"episodic", "durative"}:
            errors.append(f"{location}.memory_kind is invalid")
        if claim.get("status") not in {"candidate", "confirmed", "current", "historical", "disputed", "superseded", "excluded"}:
            errors.append(f"{location}.status is invalid")
        _track_review_status(
            claim.get("review_status"),
            location,
            require_approved_review,
            APPROVED_GOLD_REVIEW_STATUSES,
            bad_review_statuses,
            bad_review_examples,
        )
        valid_from = _timestamp(claim.get("valid_from"), f"{location}.valid_from", errors)
        valid_to = _timestamp(claim.get("valid_to"), f"{location}.valid_to", errors, allow_none=True)
        if valid_from and valid_to and valid_from > valid_to:
            errors.append(f"{location}.valid_from must not be after valid_to")
        if not isinstance(claim.get("evidence"), list) or not claim["evidence"]:
            errors.append(f"{location}.evidence must be a non-empty list")
    _duplicates(ids, "claim_id", errors)
    expected_ids = {f"scaled_{user_id}_claim_{number:03d}" for user_id in USER_IDS for number in range(1, 17)}
    if set(ids) != expected_ids:
        errors.append("claim stable IDs must use 001 through 016 for every user")
    _append_review_status_error(
        errors,
        "gold claims",
        "review_status",
        require_approved_review,
        APPROVED_GOLD_REVIEW_STATUSES,
        bad_review_statuses,
        bad_review_examples,
    )


def _validate_gold(
    gold_by_task: Mapping[str, list[dict[str, Any]]],
    runtime_by_task: Mapping[str, list[dict[str, Any]]],
    claims: list[dict[str, Any]],
    events: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    errors: list[str],
    *,
    require_approved_review: bool = False,
) -> None:
    claim_map = {item.get("claim_id"): item for item in claims}
    event_map = {item.get("event_id"): item for item in events}
    source_map = {item.get("source_id"): item for item in sources}
    for task, expected_count in (("qa", 500), ("summarization", 50), ("interactive", 20)):
        records = gold_by_task[task]
        runtime = runtime_by_task[task]
        if len(records) != expected_count:
            errors.append(f"gold {task} count must be {expected_count}, got {len(records)}")
        if [item.get("case_id") for item in records] != [item.get("case_id") for item in runtime]:
            errors.append(f"gold {task} order and IDs must match runtime")
        runtime_map = {item.get("case_id"): item for item in runtime}
        bad_review_statuses: Counter[str] = Counter()
        bad_review_examples: list[str] = []
        for index, record in enumerate(records, 1):
            location = f"gold {task} line {index}"
            _exact(record, GOLD_FIELDS[task], location, errors)
            _track_review_status(
                record.get("review_status"),
                location,
                require_approved_review,
                APPROVED_GOLD_REVIEW_STATUSES,
                bad_review_statuses,
                bad_review_examples,
            )
            runtime_record = runtime_map.get(record.get("case_id"))
            if runtime_record:
                for field in RUNTIME_FIELDS[task]:
                    if record.get(field) != runtime_record.get(field):
                        errors.append(f"{location}.{field} differs from runtime")
            if not isinstance(record.get("should_abstain"), bool):
                errors.append(f"{location}.should_abstain must be boolean")
            if not _strings(record.get("failure_tags"), allow_empty=False):
                errors.append(f"{location}.failure_tags must be a non-empty string list")
            if task == "qa":
                if not isinstance(record.get("reference_answer"), str) or not record["reference_answer"].strip():
                    errors.append(f"{location}.reference_answer must be non-empty")
                if not _strings(record.get("acceptable_answers"), allow_empty=False):
                    errors.append(f"{location}.acceptable_answers must be non-empty")
                if record.get("should_abstain") and not isinstance(record.get("abstention_reason"), str):
                    errors.append(f"{location} abstention requires a reason")
                if not record.get("should_abstain") and record.get("abstention_reason") is not None:
                    errors.append(f"{location} answerable case cannot have abstention_reason")
            elif task == "summarization" and (not isinstance(record.get("reference_summary"), str) or not record["reference_summary"].strip()):
                errors.append(f"{location}.reference_summary must be non-empty")
            elif task == "interactive" and not _strings(record.get("expected_behaviours"), allow_empty=False):
                errors.append(f"{location}.expected_behaviours must be non-empty")
            user_id = record.get("user_id")
            cutoff = _timestamp(record.get("as_of"), f"{location}.as_of", errors)
            _validate_evidence(record.get("evidence"), location, user_id, cutoff, source_map, errors)
            for claim_id in _list(record.get("required_claim_ids")):
                claim = claim_map.get(claim_id)
                if claim is None:
                    errors.append(f"{location} references unknown claim {claim_id!r}")
                elif claim.get("user_id") != user_id:
                    errors.append(f"{location} references another user's claim")
            for event_id in _list(record.get("gold_event_ids")):
                event = event_map.get(event_id)
                if event is None:
                    errors.append(f"{location} references unknown event {event_id!r}")
                elif event.get("user_id") != user_id:
                    errors.append(f"{location} references another user's oracle event")
        _append_review_status_error(
            errors,
            f"gold {task}",
            "review_status",
            require_approved_review,
            APPROVED_GOLD_REVIEW_STATUSES,
            bad_review_statuses,
            bad_review_examples,
        )
    for claim in claims:
        _validate_evidence(claim.get("evidence"), f"claim {claim.get('claim_id')}", claim.get("user_id"), None, source_map, errors)


def _validate_evidence(evidence: object, location: str, user_id: object, cutoff: datetime | None, source_map: Mapping[object, dict[str, Any]], errors: list[str]) -> None:
    if not isinstance(evidence, list):
        errors.append(f"{location}.evidence must be a list")
        return
    seen: set[tuple[object, object, object]] = set()
    for index, item in enumerate(evidence):
        item_location = f"{location}.evidence[{index}]"
        if not isinstance(item, dict) or set(item) != {"source_id", "message_id", "quote"}:
            errors.append(f"{item_location} has invalid fields")
            continue
        source = source_map.get(item.get("source_id"))
        if source is None:
            errors.append(f"{item_location} references an unknown source")
            continue
        if source.get("user_id") != user_id:
            errors.append(f"{item_location} crosses the user boundary")
        created = _timestamp(source.get("created_at"), f"{item_location}.created_at", errors)
        if cutoff and created and created > cutoff:
            errors.append(f"{item_location} cites evidence after as_of")
        message_id = item.get("message_id")
        if message_id is None:
            haystack = source.get("content") if source.get("source_type") == "calendar" else None
        else:
            message = next((value for value in source.get("messages", []) if value.get("message_id") == message_id), None)
            haystack = message.get("text") if message else None
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip() or not isinstance(haystack, str) or _normalise(quote) not in _normalise(haystack):
            errors.append(f"{item_location}.quote is not an exact normalized source substring")
        key = (item.get("source_id"), message_id, quote)
        if key in seen:
            errors.append(f"{item_location} duplicates evidence")
        seen.add(key)


def _track_review_status(
    value: object,
    location: str,
    require_reviewed: bool,
    reviewed_statuses: frozenset[str],
    bad_statuses: Counter[str],
    examples: list[str],
) -> None:
    status = value if isinstance(value, str) else "<missing_or_invalid>"
    valid = status in reviewed_statuses if require_reviewed else status == PENDING_REVIEW_STATUS
    if valid:
        return
    bad_statuses[status] += 1
    if len(examples) < 3:
        examples.append(location)


def _append_review_status_error(
    errors: list[str],
    label: str,
    field: str,
    require_reviewed: bool,
    reviewed_statuses: frozenset[str],
    bad_statuses: Counter[str],
    examples: Sequence[str],
) -> None:
    if not bad_statuses:
        return
    counts = ", ".join(f"{status}={count}" for status, count in sorted(bad_statuses.items()))
    example_text = ", ".join(examples)
    if require_reviewed:
        allowed = "/".join(sorted(reviewed_statuses))
        errors.append(
            f"{label}.{field} must be {allowed} after manifest review approval; "
            f"invalid statuses: {counts}; examples: {example_text}"
        )
    else:
        errors.append(
            f"{label}.{field} must remain {PENDING_REVIEW_STATUS} before manifest review approval; "
            f"invalid statuses: {counts}; examples: {example_text}"
        )


def _validate_reviews(
    reviews: Mapping[str, list[dict[str, Any]]],
    gold_by_task: Mapping[str, list[dict[str, Any]]],
    claims: list[dict[str, Any]],
    errors: list[str],
    *,
    require_resolved_status: bool = False,
) -> None:
    expected_targets: dict[str, set[str]] = {
        "corrections": {f"scaled_{user_id}_claim_004__scaled_{user_id}_claim_005" for user_id in USER_IDS},
        "conflicts": {f"scaled_{user_id}_claim_010__scaled_{user_id}_claim_011" for user_id in USER_IDS},
        "abstentions": {item["case_id"] for item in gold_by_task["qa"] if item.get("should_abstain")},
        "timelines": set(USER_IDS),
        "summaries": {item["case_id"] for item in gold_by_task["summarization"]},
        "interactive_expectations": {item["case_id"] for item in gold_by_task["interactive"]},
        "evidence": {
            str(item.get("claim_id", item.get("case_id")))
            for item in claims + gold_by_task["qa"] + gold_by_task["summarization"] + gold_by_task["interactive"]
        },
    }
    review_ids: list[str] = []
    for queue, records in reviews.items():
        targets: set[str] = set()
        bad_review_statuses: Counter[str] = Counter()
        bad_review_examples: list[str] = []
        for index, record in enumerate(records, 1):
            location = f"review queue {queue} line {index}"
            _exact(record, REVIEW_FIELDS, location, errors)
            if record.get("queue") != queue:
                errors.append(f"{location}.queue must be {queue}")
            _track_review_status(
                record.get("status"),
                location,
                require_resolved_status,
                RESOLVED_REVIEW_QUEUE_STATUSES,
                bad_review_statuses,
                bad_review_examples,
            )
            if record.get("benchmark_version") != BENCHMARK_VERSION:
                errors.append(f"{location}.benchmark_version is invalid")
            if record.get("user_id") not in USER_IDS:
                errors.append(f"{location}.user_id is invalid")
            if not _strings(record.get("checks"), allow_empty=False):
                errors.append(f"{location}.checks must be non-empty")
            if isinstance(record.get("review_id"), str):
                review_ids.append(record["review_id"])
            if isinstance(record.get("target_id"), str):
                targets.add(record["target_id"])
        if targets != expected_targets[queue]:
            errors.append(f"review queue {queue} does not cover its exact required targets")
        _append_review_status_error(
            errors,
            f"review queue {queue}",
            "status",
            require_resolved_status,
            RESOLVED_REVIEW_QUEUE_STATUSES,
            bad_review_statuses,
            bad_review_examples,
        )
    _duplicates(review_ids, "review_id", errors)


def _validate_phenomena(gold_qa: list[dict[str, Any]], sources: list[dict[str, Any]], errors: list[str]) -> None:
    tags = {tag for item in gold_qa for tag in _list(item.get("failure_tags"))}
    required = {"multi_session", "explicit_correction", "temporal_change", "source_disagreement", "uncertainty", "wrong_person_distractor", "hypothetical", "missing_fact"}
    missing = sorted(required - tags)
    if missing:
        errors.append(f"scaled phenomena are missing tags: {', '.join(missing)}")
    if {item.get("source_type") for item in sources} != set(SOURCE_TYPES):
        errors.append("runtime histories must include conversation, email, calendar, and chat")
    if len({item.get("user_id") for item in sources}) != 10:
        errors.append("shared runtime corpus must contain cross-user distractors")


def _validate_schemas(schemas: Mapping[str, dict[str, Any]], errors: list[str]) -> None:
    schema_ids: list[str] = []
    for name, schema in schemas.items():
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"schema {name} must use JSON Schema draft 2020-12")
        if schema.get("type") != "object":
            errors.append(f"schema {name} must describe an object")
        if schema.get("additionalProperties") is not False:
            errors.append(f"schema {name} must reject unknown fields")
        required = schema.get("required")
        properties = schema.get("properties")
        if not _strings(required, allow_empty=False) or not isinstance(properties, dict) or not set(required).issubset(properties):
            errors.append(f"schema {name} must define every required property")
        schema_id = schema.get("$id")
        if isinstance(schema_id, str):
            schema_ids.append(schema_id)
        if schema_id != f"https://longitudinal-memory.local/scaled-v1/{name}.schema.json":
            errors.append(f"schema {name} has an invalid versioned $id")
    _duplicates(schema_ids, "schema $id", errors)


def _validate_manifest(root: Path, manifest: dict[str, Any], users: list[dict[str, Any]], sources: list[dict[str, Any]], events: list[dict[str, Any]], claims: list[dict[str, Any]], runtime_by_task: Mapping[str, list[dict[str, Any]]], reviews: Mapping[str, list[dict[str, Any]]], errors: list[str]) -> None:
    fields = {"manifest_version", "benchmark_version", "release_status", "dataset_sha256", "files", "counts", "capability_counts", "splits", "review"}
    _exact(manifest, fields, "manifest", errors)
    if manifest.get("manifest_version") != "1" or manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append("manifest version fields are invalid")
    if manifest.get("release_status") != "frozen":
        errors.append("scaled release must be frozen after Sneha's approval")
    try:
        current_hash = dataset_sha256(root, tuple(path.as_posix() for path in RELEASE_DATA_PATHS))
    except Exception as error:
        errors.append(f"could not reproduce dataset hash: {error}")
        current_hash = ""
    if manifest.get("dataset_sha256") != current_hash:
        errors.append("manifest dataset_sha256 does not match release files")
    declared: dict[str, dict[str, Any]] = {}
    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest.files must be a list")
    else:
        for index, item in enumerate(files):
            if not isinstance(item, dict) or set(item) != {"path", "layer", "sha256", "records"}:
                errors.append(f"manifest.files[{index}] has invalid fields")
                continue
            relative = item.get("path")
            if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                errors.append(f"manifest.files[{index}].path is unsafe")
                continue
            declared[relative] = item
    expected_paths = {path.as_posix() for path in RELEASE_DATA_PATHS}
    if set(declared) != expected_paths:
        errors.append("manifest file list does not match the frozen scaled release")
    for relative in RELEASE_DATA_PATHS:
        item = declared.get(relative.as_posix())
        path = root / relative
        if item is None or not path.is_file():
            continue
        if item.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f"manifest hash does not match {relative}")
        records = 1 if path.suffix == ".json" else len(_checked_jsonl(path, errors))
        if item.get("records") != records:
            errors.append(f"manifest record count does not match {relative}")
        expected_layer = (
            "runtime" if relative in RUNTIME_PATHS else
            "oracle" if relative in ORACLE_PATHS else
            "gold" if relative in GOLD_PATHS else
            "prediction" if relative in PREDICTION_PATHS else
            "review" if relative in REVIEW_PATHS else
            "schema"
        )
        if item.get("layer") != expected_layer:
            errors.append(f"manifest layer does not match {relative}")
    expected_counts = {
        "users": len(users), "sources": len(sources), "oracle_events": len(events), "gold_claims": len(claims),
        "qa": len(runtime_by_task["qa"]), "summaries": len(runtime_by_task["summarization"]), "interactive_scenarios": len(runtime_by_task["interactive"]),
        "predictions": 0, "review_queue_items": {name: len(value) for name, value in reviews.items()},
    }
    if manifest.get("counts") != expected_counts:
        errors.append("manifest counts do not match release records")
    expected_capabilities = {
        "qa": {name: 100 for name in CAPABILITIES},
        "summaries": {"temporal_reasoning": 25, "user_modeling": 25},
        "interactive": {name: 4 for name in CAPABILITIES},
    }
    if manifest.get("capability_counts") != expected_capabilities:
        errors.append("manifest capability counts are invalid")
    expected_splits = {
        "strategy": "whole_user",
        "development": {"user_ids": list(DEV_USERS), "qa": 100, "summaries": 10, "interactive_scenarios": 4},
        "test": {"user_ids": list(TEST_USERS), "qa": 400, "summaries": 40, "interactive_scenarios": 16, "frozen_for_tuning": True},
    }
    if manifest.get("splits") != expected_splits:
        errors.append("manifest split information is invalid")
    _validate_manifest_review_state(manifest.get("review"), current_hash, errors)


def _validate_manifest_review_state(review: object, current_hash: str, errors: list[str]) -> None:
    fields = {
        "automated_validation_status",
        "implementation_review_status",
        "human_review_status",
        "review_queue_status",
        "reviewed_by",
        "reviewed_dataset_sha256",
        "approved_on",
    }
    _exact(review, fields, "manifest.review", errors)
    if not isinstance(review, dict):
        return
    if review.get("automated_validation_status") != "passed":
        errors.append("manifest.review.automated_validation_status must be passed")
    if review.get("implementation_review_status") not in {"complete", "pending", "in_progress"}:
        errors.append("manifest.review.implementation_review_status is invalid")
    if review.get("human_review_status") not in {"approved", "pending_sneha_review", "pending_human_review"}:
        errors.append("manifest.review.human_review_status is invalid")
    if review.get("review_queue_status") not in {"approved", "pending_sneha_review", "pending_human_review"}:
        errors.append("manifest.review.review_queue_status is invalid")

    human_approved = review.get("human_review_status") == "approved"
    queue_approved = review.get("review_queue_status") == "approved"
    if human_approved:
        if not isinstance(review.get("reviewed_by"), str) or not review["reviewed_by"].strip():
            errors.append("manifest.review.reviewed_by is required after human review approval")
        if review.get("reviewed_dataset_sha256") != current_hash:
            errors.append("manifest.review.reviewed_dataset_sha256 must match current reviewed dataset hash")
        _date_or_timestamp(review.get("approved_on"), "manifest.review.approved_on", errors)
    if queue_approved and not human_approved:
        errors.append("manifest.review.review_queue_status cannot be approved before human_review_status")


def _case_common(case: Mapping[str, Any], task: str, location: str, errors: list[str]) -> None:
    if case.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(f"{location}.benchmark_version is invalid")
    user_id = case.get("user_id")
    if user_id not in USER_IDS:
        errors.append(f"{location}.user_id is invalid")
    expected_split = "development" if user_id in DEV_USERS else "test"
    if case.get("split") != expected_split:
        errors.append(f"{location}.split must be {expected_split}")
    if case.get("task") != task:
        errors.append(f"{location}.task must be {task}")
    if case.get("capability") not in CAPABILITIES:
        errors.append(f"{location}.capability is invalid")
    _timestamp(case.get("as_of"), f"{location}.as_of", errors)


def _leakage(record: object, location: str, errors: list[str]) -> None:
    leaked = sorted(_find_keys(record) & LEAKAGE_KEYS)
    if leaked:
        errors.append(f"{location} leaks scorer fields: {', '.join(leaked)}")


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


def _snake_keys(value: object, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                errors.append(f"{location} contains non-snake_case key {key!r}")
            _snake_keys(item, location, errors)
    elif isinstance(value, list):
        for item in value:
            _snake_keys(item, location, errors)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ScaledReleaseError((f"{path}:{line_number}: blank JSONL line",))
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ScaledReleaseError((f"{path}:{line_number}: record must be an object",))
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ScaledReleaseError((f"could not read {path}: {error}",)) from error
    return records


def _checked_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except ScaledReleaseError as error:
        errors.extend(error.errors)
        return []


def _checked_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"could not read {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain one JSON object")
        return {}
    return value


def _exact(record: object, fields: Iterable[str], location: str, errors: list[str]) -> None:
    if not isinstance(record, dict):
        errors.append(f"{location} must be an object")
        return
    expected = set(fields)
    missing = sorted(expected - set(record))
    unknown = sorted(set(record) - expected)
    if missing:
        errors.append(f"{location} missing fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"{location} unknown fields: {', '.join(unknown)}")


def _timestamp(value: object, location: str, errors: list[str], allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
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


def _date_or_timestamp(value: object, location: str, errors: list[str], allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str):
        errors.append(f"{location} must be an ISO date or timestamp")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{location} must be an ISO date or timestamp")


def _snake(value: object, location: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        errors.append(f"{location} must be lowercase snake_case")


def _strings(value: object, allow_empty: bool) -> bool:
    return isinstance(value, list) and (allow_empty or bool(value)) and all(isinstance(item, str) and bool(item.strip()) for item in value)


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _duplicates(values: Sequence[str], label: str, errors: list[str]) -> None:
    duplicate = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicate:
        errors.append(f"duplicate {label}: {', '.join(duplicate)}")


def _normalise(value: str) -> str:
    return " ".join(value.split())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Step R2 scaled benchmark release.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or __import__("sys").stdout
    args = _build_parser().parse_args(argv)
    try:
        report = validate_scaled_release(args.repo_root)
    except ScaledReleaseError as error:
        print(json.dumps({"status": "failed", "errors": list(error.errors)}), file=stdout)
        return 1
    print(json.dumps({
        "status": "passed",
        "benchmark_version": report.benchmark_version,
        "dataset_sha256": report.dataset_sha256,
        "counts": {"users": report.users, "sources": report.sources, "qa": report.qa, "summaries": report.summaries, "interactive_scenarios": report.interactive_scenarios, "gold_claims": report.gold_claims},
        "qa_capability_counts": report.qa_capability_counts,
        "summary_type_counts": report.summary_type_counts,
        "interactive_capability_counts": report.interactive_capability_counts,
        "split_counts": report.split_counts,
        "review_queue_counts": report.review_queue_counts,
        "human_review_status": report.human_review_status,
    }, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
