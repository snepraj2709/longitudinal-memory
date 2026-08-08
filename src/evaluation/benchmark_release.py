"""Load and validate the frozen Benchmark v1 release without model calls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Mapping, Sequence, TextIO

from .history import HistoryObservation, load_history_observations
from .run_config import dataset_sha256


BENCHMARK_VERSION = "benchmark_v1"
RELEASE_ROOT = Path("data/benchmark-v1")
MANIFEST_PATH = RELEASE_ROOT / "manifest.json"
RUNTIME_CASE_PATHS = (
    RELEASE_ROOT / "runtime/qa.jsonl",
    RELEASE_ROOT / "runtime/summaries.jsonl",
    RELEASE_ROOT / "runtime/interactive.jsonl",
)
RUNTIME_SUPPORT_PATHS = (
    Path("data/pilot/user.jsonl"),
    Path("data/pilot/sources/calendar.jsonl"),
    Path("data/pilot/sources/conversations.jsonl"),
    Path("data/pilot/sources/emails.jsonl"),
)
ORACLE_PATHS = (Path("data/pilot/oracle-event.jsonl"),)
GOLD_PATHS = (
    RELEASE_ROOT / "gold/claims.jsonl",
    RELEASE_ROOT / "gold/qa.jsonl",
    RELEASE_ROOT / "gold/summaries.jsonl",
    RELEASE_ROOT / "gold/interactive.jsonl",
)
RELEASE_DATA_PATHS = RUNTIME_SUPPORT_PATHS + RUNTIME_CASE_PATHS + ORACLE_PATHS + GOLD_PATHS

CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "user_modeling",
    "abstention",
)
TASK_COUNTS = {"qa": 50, "summarization": 5, "interactive": 2}
LEAKAGE_KEYS = frozenset(
    {
        "acceptable_answers",
        "abstention_reason",
        "evidence",
        "expected_behaviours",
        "failure_tags",
        "gold_event_ids",
        "oracle_fact_ids",
        "reference_answer",
        "reference_summary",
        "required_claim_ids",
        "review_status",
        "should_abstain",
    }
)
_ID_PATTERNS = {
    "qa": re.compile(
        r"^(?:extraction|temporal|conflict|user_modeling|abstention)_\d{3}$"
    ),
    "summarization": re.compile(r"^summary_\d{3}$"),
    "interactive": re.compile(r"^interactive_\d{3}$"),
    "claim": re.compile(r"^claim_\d{3}$"),
}
_RUNTIME_FIELDS = {
    "qa": frozenset(
        {
            "case_id",
            "benchmark_version",
            "split",
            "user_id",
            "task",
            "capability",
            "as_of",
            "difficulty",
            "question",
        }
    ),
    "summarization": frozenset(
        {
            "case_id",
            "benchmark_version",
            "split",
            "user_id",
            "task",
            "capability",
            "as_of",
            "difficulty",
            "instruction",
        }
    ),
    "interactive": frozenset(
        {
            "case_id",
            "benchmark_version",
            "split",
            "user_id",
            "task",
            "capability",
            "as_of",
            "difficulty",
            "scenario",
            "initial_user_message",
            "allowed_turns",
        }
    ),
}
_GOLD_FIELDS = {
    "qa": _RUNTIME_FIELDS["qa"]
    | {
        "failure_tags",
        "should_abstain",
        "evidence",
        "reference_answer",
        "acceptable_answers",
        "abstention_reason",
        "required_claim_ids",
        "oracle_fact_ids",
        "review_status",
    },
    "summarization": _RUNTIME_FIELDS["summarization"]
    | {
        "failure_tags",
        "should_abstain",
        "evidence",
        "reference_summary",
        "gold_event_ids",
        "required_claim_ids",
        "review_status",
    },
    "interactive": _RUNTIME_FIELDS["interactive"]
    | {
        "failure_tags",
        "should_abstain",
        "evidence",
        "expected_behaviours",
        "required_claim_ids",
        "review_status",
    },
}
_CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "user_id",
        "subject_id",
        "speaker_id",
        "predicate",
        "object",
        "polarity",
        "epistemic_status",
        "valid_from",
        "valid_to",
        "time_precision",
        "evidence",
        "review_status",
    }
)


class BenchmarkReleaseError(ValueError):
    """Report every deterministic release error in one exception."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class RuntimeCase:
    """A case projection containing only fields allowed at runtime."""

    case_id: str
    benchmark_version: str
    split: str
    user_id: str
    task: str
    capability: str
    as_of: datetime
    difficulty: str
    prompt: str
    scenario: str | None = None
    allowed_turns: int | None = None


@dataclass(frozen=True)
class BenchmarkRuntime:
    """The complete Benchmark v1 input visible to a runtime system."""

    users: tuple[Mapping[str, object], ...]
    observations: tuple[HistoryObservation, ...]
    qa: tuple[RuntimeCase, ...]
    summaries: tuple[RuntimeCase, ...]
    interactive: tuple[RuntimeCase, ...]


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic evidence returned after a complete release check."""

    benchmark_version: str
    dataset_sha256: str
    qa_count: int
    summary_count: int
    interactive_count: int
    claim_count: int
    capability_counts: Mapping[str, int]
    human_review_status: str


def load_benchmark_v1_runtime(repo_root: str | Path) -> BenchmarkRuntime:
    """Load only allowlisted runtime inputs, never oracle or scorer gold."""

    root = Path(repo_root).resolve()
    users = tuple(_read_jsonl(root / RUNTIME_SUPPORT_PATHS[0]))
    observations = load_history_observations(root / "data/pilot/sources")
    raw_cases = {path.stem: _read_jsonl(root / path) for path in RUNTIME_CASE_PATHS}
    records_by_task = {
        "qa": raw_cases["qa"],
        "summarization": raw_cases["summaries"],
        "interactive": raw_cases["interactive"],
    }
    errors: list[str] = []
    _validate_runtime_records(records_by_task, errors)
    if errors:
        raise BenchmarkReleaseError(errors)
    cases = {
        stem: tuple(
            _runtime_case(record, stem, root / path)
            for record in raw_cases[stem]
        )
        for stem, path in (
            ("qa", RUNTIME_CASE_PATHS[0]),
            ("summaries", RUNTIME_CASE_PATHS[1]),
            ("interactive", RUNTIME_CASE_PATHS[2]),
        )
    }
    return BenchmarkRuntime(
        users=users,
        observations=observations,
        qa=cases["qa"],
        summaries=cases["summaries"],
        interactive=cases["interactive"],
    )


def validate_benchmark_v1(repo_root: str | Path) -> ValidationReport:
    """Validate release shape, references, evidence, hashes, and leakage."""

    root = Path(repo_root).resolve()
    errors: list[str] = []
    runtime_records = {
        path.stem: _read_jsonl_checked(root / path, errors) for path in RUNTIME_CASE_PATHS
    }
    gold_records = {
        path.stem: _read_jsonl_checked(root / path, errors)
        for path in GOLD_PATHS
        if path.stem != "claims"
    }
    claims = _read_jsonl_checked(root / GOLD_PATHS[0], errors)
    oracle = _read_jsonl_checked(root / ORACLE_PATHS[0], errors)
    pilot_questions = _read_jsonl_checked(
        root / "data/pilot/evaluation/eval_questions.jsonl", errors
    )
    pilot_answers = _read_jsonl_checked(
        root / "data/pilot/evaluation/eval_answer.jsonl", errors
    )
    manifest = _read_json_checked(root / MANIFEST_PATH, errors)

    runtime_by_task = {
        "qa": runtime_records.get("qa", []),
        "summarization": runtime_records.get("summaries", []),
        "interactive": runtime_records.get("interactive", []),
    }
    gold_by_task = {
        "qa": gold_records.get("qa", []),
        "summarization": gold_records.get("summaries", []),
        "interactive": gold_records.get("interactive", []),
    }

    _validate_runtime_records(runtime_by_task, errors)
    for path in RUNTIME_SUPPORT_PATHS:
        for index, record in enumerate(_read_jsonl_checked(root / path, errors), 1):
            leaked = sorted(_find_leakage_keys(record))
            if leaked:
                errors.append(
                    f"{path} line {index} leaks scorer fields: {', '.join(leaked)}"
                )
    _validate_gold_records(gold_by_task, runtime_by_task, errors)
    _validate_stable_ids(runtime_by_task, pilot_questions, pilot_answers, errors)
    _validate_claims(claims, errors)

    observations: tuple[HistoryObservation, ...] = ()
    try:
        observations = load_history_observations(root / "data/pilot/sources")
    except Exception as error:  # deterministic aggregation boundary
        errors.append(f"could not load source evidence: {error}")
    _validate_references(gold_by_task, claims, oracle, observations, errors)
    _validate_manifest(root, manifest, runtime_by_task, claims, oracle, errors)

    if errors:
        raise BenchmarkReleaseError(errors)

    capability_counts = Counter(record["capability"] for record in runtime_by_task["qa"])
    return ValidationReport(
        benchmark_version=BENCHMARK_VERSION,
        dataset_sha256=dataset_sha256(root, tuple(path.as_posix() for path in RELEASE_DATA_PATHS)),
        qa_count=len(runtime_by_task["qa"]),
        summary_count=len(runtime_by_task["summarization"]),
        interactive_count=len(runtime_by_task["interactive"]),
        claim_count=len(claims),
        capability_counts={name: capability_counts[name] for name in CAPABILITIES},
        human_review_status=manifest["review"]["human_review_status"],
    )


def _runtime_case(record: Mapping[str, object], stem: str, path: Path) -> RuntimeCase:
    task = {"qa": "qa", "summaries": "summarization", "interactive": "interactive"}[stem]
    expected = _RUNTIME_FIELDS[task]
    if set(record) != expected:
        raise BenchmarkReleaseError(
            (f"{path}: runtime fields must be {sorted(expected)}",)
        )
    prompt_field = {
        "qa": "question",
        "summarization": "instruction",
        "interactive": "initial_user_message",
    }[task]
    return RuntimeCase(
        case_id=_string(record, "case_id", str(path)),
        benchmark_version=_string(record, "benchmark_version", str(path)),
        split=_string(record, "split", str(path)),
        user_id=_string(record, "user_id", str(path)),
        task=_string(record, "task", str(path)),
        capability=_string(record, "capability", str(path)),
        as_of=_timestamp(record.get("as_of"), f"{path}.as_of"),
        difficulty=_string(record, "difficulty", str(path)),
        prompt=_string(record, prompt_field, str(path)),
        scenario=record.get("scenario") if task == "interactive" else None,
        allowed_turns=record.get("allowed_turns") if task == "interactive" else None,
    )


def _validate_runtime_records(
    records_by_task: Mapping[str, list[dict[str, object]]], errors: list[str]
) -> None:
    all_ids: list[str] = []
    for task, expected_count in TASK_COUNTS.items():
        records = records_by_task[task]
        if len(records) != expected_count:
            errors.append(f"runtime {task} count must be {expected_count}, got {len(records)}")
        for index, record in enumerate(records, 1):
            location = f"runtime {task} line {index}"
            _exact_fields(record, _RUNTIME_FIELDS[task], location, errors)
            _common_case_fields(record, task, location, errors)
            leaked = sorted(_find_leakage_keys(record))
            if leaked:
                errors.append(f"{location} leaks scorer fields: {', '.join(leaked)}")
            case_id = record.get("case_id")
            if isinstance(case_id, str):
                all_ids.append(case_id)
                if not _ID_PATTERNS[task].fullmatch(case_id):
                    errors.append(f"{location}.case_id has an invalid stable-ID shape")
            if task == "qa":
                _expect_string(record.get("question"), f"{location}.question", errors)
            elif task == "summarization":
                _expect_string(
                    record.get("instruction"), f"{location}.instruction", errors
                )
            else:
                _expect_string(record.get("scenario"), f"{location}.scenario", errors)
                _expect_string(
                    record.get("initial_user_message"),
                    f"{location}.initial_user_message",
                    errors,
                )
                turns = record.get("allowed_turns")
                if not isinstance(turns, int) or isinstance(turns, bool) or turns < 1:
                    errors.append(f"{location}.allowed_turns must be a positive integer")
    _duplicates(all_ids, "runtime case_id", errors)
    counts = Counter(record.get("capability") for record in records_by_task["qa"])
    expected = {capability: 10 for capability in CAPABILITIES}
    if dict(counts) != expected:
        errors.append(f"QA capability counts must be {expected}, got {dict(counts)}")


def _validate_gold_records(
    records_by_task: Mapping[str, list[dict[str, object]]],
    runtime_by_task: Mapping[str, list[dict[str, object]]],
    errors: list[str],
) -> None:
    for task, expected_count in TASK_COUNTS.items():
        records = records_by_task[task]
        if len(records) != expected_count:
            errors.append(f"gold {task} count must be {expected_count}, got {len(records)}")
        ids: list[str] = []
        for index, record in enumerate(records, 1):
            location = f"gold {task} line {index}"
            _exact_fields(record, _GOLD_FIELDS[task], location, errors)
            _common_case_fields(record, task, location, errors)
            case_id = record.get("case_id")
            if isinstance(case_id, str):
                ids.append(case_id)
            if record.get("review_status") != "implementation_reviewed":
                errors.append(f"{location}.review_status must be implementation_reviewed")
            failure_tags = record.get("failure_tags")
            if not _non_empty_string_list(failure_tags):
                errors.append(f"{location}.failure_tags must be a non-empty string list")
            should_abstain = record.get("should_abstain")
            if not isinstance(should_abstain, bool):
                errors.append(f"{location}.should_abstain must be boolean")
            evidence = record.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"{location}.evidence must be a list")
            if should_abstain is True and evidence:
                errors.append(f"{location} abstention evidence must be empty")
            if should_abstain is False and not evidence:
                errors.append(f"{location} answerable case needs evidence")
            if task == "qa":
                reason = record.get("abstention_reason")
                if should_abstain is True and not isinstance(reason, str):
                    errors.append(f"{location} abstention needs a reason")
                if should_abstain is False and reason is not None:
                    errors.append(f"{location} answerable case cannot have abstention_reason")
                _expect_string(
                    record.get("reference_answer"),
                    f"{location}.reference_answer",
                    errors,
                )
                if not _non_empty_string_list(record.get("acceptable_answers")):
                    errors.append(
                        f"{location}.acceptable_answers must be a non-empty string list"
                    )
                _string_list(record.get("required_claim_ids"), f"{location}.required_claim_ids", errors)
                _string_list(record.get("oracle_fact_ids"), f"{location}.oracle_fact_ids", errors)
            elif task == "summarization":
                _expect_string(
                    record.get("reference_summary"),
                    f"{location}.reference_summary",
                    errors,
                )
                if not _non_empty_string_list(record.get("gold_event_ids")):
                    errors.append(
                        f"{location}.gold_event_ids must be a non-empty string list"
                    )
                if not _non_empty_string_list(record.get("required_claim_ids")):
                    errors.append(
                        f"{location}.required_claim_ids must be a non-empty string list"
                    )
            else:
                if not _non_empty_string_list(record.get("expected_behaviours")):
                    errors.append(
                        f"{location}.expected_behaviours must be a non-empty string list"
                    )
                if not _non_empty_string_list(record.get("required_claim_ids")):
                    errors.append(
                        f"{location}.required_claim_ids must be a non-empty string list"
                    )
        _duplicates(ids, f"gold {task} case_id", errors)
        runtime = runtime_by_task[task]
        runtime_ids = [item.get("case_id") for item in runtime]
        if ids != runtime_ids:
            errors.append(f"gold {task} case order does not match runtime")
        runtime_by_id = {item.get("case_id"): item for item in runtime}
        for gold in records:
            case_id = gold.get("case_id")
            runtime_record = runtime_by_id.get(case_id)
            if runtime_record is None:
                continue
            for field in _RUNTIME_FIELDS[task]:
                if gold.get(field) != runtime_record.get(field):
                    errors.append(f"gold {case_id}.{field} does not match runtime")


def _validate_stable_ids(
    runtime_by_task: Mapping[str, list[dict[str, object]]],
    pilot_questions: list[dict[str, object]],
    pilot_answers: list[dict[str, object]],
    errors: list[str],
) -> None:
    qa = runtime_by_task["qa"]
    runtime_by_id = {item.get("case_id"): item for item in qa}
    pilot_answer_ids = [item.get("case_id") for item in pilot_answers]
    pilot_question_ids = [item.get("case_id") for item in pilot_questions]
    if pilot_question_ids != pilot_answer_ids:
        errors.append("Pilot v0 question and answer IDs no longer align")
    for pilot in pilot_questions:
        case_id = pilot.get("case_id")
        current = runtime_by_id.get(case_id)
        if current is None:
            errors.append(f"Benchmark v1 does not preserve Pilot case_id {case_id!r}")
            continue
        for field in ("case_id", "capability", "question", "as_of", "difficulty"):
            if current.get(field) != pilot.get(field):
                errors.append(f"Benchmark v1 changed Pilot {case_id}.{field}")
        if current.get("failure_tags") is not None:
            errors.append(f"runtime {case_id} must not include failure_tags")
    expected_ids = []
    for prefix in ("extraction", "temporal", "conflict", "user_modeling", "abstention"):
        expected_ids.extend(f"{prefix}_{index:03d}" for index in range(1, 11))
    if [item.get("case_id") for item in qa] != expected_ids:
        errors.append("QA stable IDs must use the fixed capability order and 001-010 sequence")
    expected_capability_by_prefix = {
        "extraction": "extraction",
        "temporal": "temporal_reasoning",
        "conflict": "conflict_detection",
        "user_modeling": "user_modeling",
        "abstention": "abstention",
    }
    for record in qa:
        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            continue
        prefix = case_id.rsplit("_", 1)[0]
        expected_capability = expected_capability_by_prefix.get(prefix)
        if expected_capability is not None and record.get("capability") != expected_capability:
            errors.append(f"{case_id}.capability must be {expected_capability}")
    if [item.get("case_id") for item in runtime_by_task["summarization"]] != [
        f"summary_{index:03d}" for index in range(1, 6)
    ]:
        errors.append("summary stable IDs must be summary_001 through summary_005")
    if [item.get("case_id") for item in runtime_by_task["interactive"]] != [
        "interactive_001",
        "interactive_002",
    ]:
        errors.append("interactive stable IDs must be interactive_001 and interactive_002")


def _validate_claims(claims: list[dict[str, object]], errors: list[str]) -> None:
    ids: list[str] = []
    for index, claim in enumerate(claims, 1):
        location = f"gold claim line {index}"
        _exact_fields(claim, _CLAIM_FIELDS, location, errors)
        claim_id = claim.get("claim_id")
        if isinstance(claim_id, str):
            ids.append(claim_id)
            if not _ID_PATTERNS["claim"].fullmatch(claim_id):
                errors.append(f"{location}.claim_id has an invalid stable-ID shape")
        for field in ("user_id", "subject_id", "speaker_id", "predicate"):
            _expect_string(claim.get(field), f"{location}.{field}", errors)
        predicate = claim.get("predicate")
        if isinstance(predicate, str) and (
            predicate.lower() != predicate or not re.fullmatch(r"[a-z][a-z0-9_]*", predicate)
        ):
            errors.append(f"{location}.predicate must be lowercase snake_case")
        if claim.get("polarity") not in {"positive", "negative"}:
            errors.append(f"{location}.polarity is invalid")
        if claim.get("epistemic_status") not in {
            "asserted",
            "reported_by_other",
            "hypothetical",
            "uncertain",
            "denied",
            "corrected",
        }:
            errors.append(f"{location}.epistemic_status is invalid")
        if claim.get("time_precision") not in {
            "timestamp",
            "day",
            "month",
            "year",
            "approximate",
            "unknown",
        }:
            errors.append(f"{location}.time_precision is invalid")
        if claim.get("object") is None:
            errors.append(f"{location}.object must not be null")
        start = _optional_timestamp(claim.get("valid_from"), f"{location}.valid_from", errors)
        end = _optional_timestamp(claim.get("valid_to"), f"{location}.valid_to", errors)
        if start is not None and end is not None and start > end:
            errors.append(f"{location} valid_from must not be after valid_to")
        if claim.get("review_status") != "implementation_reviewed":
            errors.append(f"{location}.review_status must be implementation_reviewed")
        if not isinstance(claim.get("evidence"), list) or not claim["evidence"]:
            errors.append(f"{location}.evidence must be a non-empty list")
    _duplicates(ids, "claim_id", errors)
    expected = [f"claim_{index:03d}" for index in range(1, len(claims) + 1)]
    if ids != expected:
        errors.append("claim stable IDs must be contiguous and ordered")


def _validate_references(
    gold_by_task: Mapping[str, list[dict[str, object]]],
    claims: list[dict[str, object]],
    oracle: list[dict[str, object]],
    observations: tuple[HistoryObservation, ...],
    errors: list[str],
) -> None:
    claim_ids = {claim.get("claim_id") for claim in claims}
    oracle_event_ids = {event.get("event_id") for event in oracle}
    oracle_fact_ids = {
        fact.get("fact_id")
        for event in oracle
        for fact in event.get("facts", [])
        if isinstance(fact, dict)
    }
    observation_map = {
        (item.source_id, item.message_id): item for item in observations
    }
    source_ids = {item.source_id for item in observations}
    for claim in claims:
        _validate_evidence(
            claim.get("evidence"),
            f"claim {claim.get('claim_id')}",
            None,
            observation_map,
            source_ids,
            errors,
        )
    for task, records in gold_by_task.items():
        for record in records:
            location = f"{task} {record.get('case_id')}"
            cutoff = _optional_timestamp(record.get("as_of"), f"{location}.as_of", errors)
            _validate_evidence(
                record.get("evidence"),
                location,
                cutoff,
                observation_map,
                source_ids,
                errors,
            )
            for claim_id in _list(record.get("required_claim_ids")):
                if claim_id not in claim_ids:
                    errors.append(f"{location} references unknown claim {claim_id!r}")
            for event_id in _list(record.get("gold_event_ids")):
                if event_id not in oracle_event_ids:
                    errors.append(f"{location} references unknown oracle event {event_id!r}")
            for fact_id in _list(record.get("oracle_fact_ids")):
                if fact_id not in oracle_fact_ids:
                    errors.append(f"{location} references unknown oracle fact {fact_id!r}")


def _validate_evidence(
    evidence: object,
    location: str,
    cutoff: datetime | None,
    observation_map: Mapping[tuple[str, str | None], HistoryObservation],
    source_ids: set[str],
    errors: list[str],
) -> None:
    if not isinstance(evidence, list):
        return
    seen: set[tuple[str, str | None, str]] = set()
    for index, item in enumerate(evidence):
        item_location = f"{location}.evidence[{index}]"
        if not isinstance(item, dict) or set(item) != {"source_id", "message_id", "quote"}:
            errors.append(
                f"{item_location} must contain source_id, message_id, and quote"
            )
            continue
        source_id = item.get("source_id")
        message_id = item.get("message_id")
        quote = item.get("quote")
        if not isinstance(source_id, str) or source_id not in source_ids:
            errors.append(f"{item_location} references an unknown source")
            continue
        if message_id is not None and not isinstance(message_id, str):
            errors.append(f"{item_location}.message_id must be string or null")
            continue
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"{item_location}.quote must be a non-empty string")
            continue
        observation = observation_map.get((source_id, message_id))
        if observation is None:
            errors.append(f"{item_location} references an unknown source/message pair")
            continue
        if _normalise(quote) not in _normalise(observation.text):
            errors.append(f"{item_location}.quote is not an exact normalised source substring")
        if cutoff is not None and observation.observed_at > cutoff:
            errors.append(f"{item_location} cites evidence after as_of")
        ref = (source_id, message_id, quote)
        if ref in seen:
            errors.append(f"{item_location} duplicates evidence")
        seen.add(ref)


def _validate_manifest(
    root: Path,
    manifest: dict[str, object],
    runtime_by_task: Mapping[str, list[dict[str, object]]],
    claims: list[dict[str, object]],
    oracle: list[dict[str, object]],
    errors: list[str],
) -> None:
    expected_fields = {
        "manifest_version",
        "benchmark_version",
        "release_status",
        "dataset_sha256",
        "files",
        "counts",
        "capability_counts",
        "splits",
        "review",
    }
    _exact_fields(manifest, expected_fields, "manifest", errors)
    if manifest.get("manifest_version") != "1":
        errors.append("manifest_version must be 1")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(f"manifest benchmark_version must be {BENCHMARK_VERSION}")
    if manifest.get("release_status") != "candidate":
        errors.append("release_status must remain candidate until Sneha reviews it")
    expected_hash = ""
    try:
        expected_hash = dataset_sha256(
            root, tuple(path.as_posix() for path in RELEASE_DATA_PATHS)
        )
    except Exception as error:
        errors.append(f"could not calculate dataset hash: {error}")
    if manifest.get("dataset_sha256") != expected_hash:
        errors.append("manifest dataset_sha256 does not match release files")
    files = manifest.get("files")
    manifest_files: dict[str, dict[str, object]] = {}
    if not isinstance(files, list):
        errors.append("manifest.files must be a list")
    else:
        for index, item in enumerate(files):
            if not isinstance(item, dict) or set(item) != {"path", "layer", "sha256", "records"}:
                errors.append(f"manifest.files[{index}] has invalid fields")
                continue
            path = item.get("path")
            if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                errors.append(f"manifest.files[{index}].path must be repository relative")
                continue
            manifest_files[path] = item
    expected_paths = {path.as_posix() for path in RELEASE_DATA_PATHS}
    if set(manifest_files) != expected_paths:
        errors.append("manifest file list does not match the frozen release file set")
    for relative in RELEASE_DATA_PATHS:
        path = root / relative
        item = manifest_files.get(relative.as_posix())
        if item is None or not path.is_file():
            continue
        if item.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f"manifest hash does not match {relative}")
        try:
            record_count = len(_read_jsonl(path))
        except BenchmarkReleaseError:
            continue
        if item.get("records") != record_count:
            errors.append(f"manifest record count does not match {relative}")
        expected_layer = (
            "runtime"
            if relative in RUNTIME_SUPPORT_PATHS + RUNTIME_CASE_PATHS
            else "oracle"
            if relative in ORACLE_PATHS
            else "gold"
        )
        if item.get("layer") != expected_layer:
            errors.append(f"manifest layer does not match {relative}")
    counts = manifest.get("counts")
    expected_counts = {
        "users": 1,
        "oracle_events": len(oracle),
        "qa": len(runtime_by_task["qa"]),
        "summaries": len(runtime_by_task["summarization"]),
        "interactive_scenarios": len(runtime_by_task["interactive"]),
        "gold_claims": len(claims),
    }
    if counts != expected_counts:
        errors.append(f"manifest counts must be {expected_counts}")
    expected_capabilities = {capability: 10 for capability in CAPABILITIES}
    if manifest.get("capability_counts") != {"qa": expected_capabilities}:
        errors.append("manifest capability_counts must record ten QA cases per capability")
    expected_splits = {
        "strategy": "whole_user",
        "development": {
            "user_ids": ["i_am_maya"],
            "qa": 50,
            "summaries": 5,
            "interactive_scenarios": 2,
        },
        "test": {
            "user_ids": [],
            "qa": 0,
            "summaries": 0,
            "interactive_scenarios": 0,
        },
    }
    if manifest.get("splits") != expected_splits:
        errors.append("manifest split information does not match the one-user release")
    review = manifest.get("review")
    expected_review = {
        "implementation_review_status": "complete",
        "automated_validation_status": "passed",
        "human_review_status": "pending_sneha_review",
    }
    if review != expected_review:
        errors.append(f"manifest review state must be {expected_review}")


def _common_case_fields(
    record: Mapping[str, object], task: str, location: str, errors: list[str]
) -> None:
    for field in ("case_id", "benchmark_version", "split", "user_id", "task", "capability", "difficulty"):
        _expect_string(record.get(field), f"{location}.{field}", errors)
    if record.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(f"{location}.benchmark_version must be {BENCHMARK_VERSION}")
    if record.get("split") != "development":
        errors.append(f"{location}.split must be development")
    if record.get("user_id") != "i_am_maya":
        errors.append(f"{location}.user_id must be i_am_maya")
    if record.get("task") != task:
        errors.append(f"{location}.task must be {task}")
    if record.get("capability") not in CAPABILITIES:
        errors.append(f"{location}.capability is invalid")
    _optional_timestamp(record.get("as_of"), f"{location}.as_of", errors)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise BenchmarkReleaseError((f"{path}:{line_number}: blank JSONL line",))
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BenchmarkReleaseError((f"{path}:{line_number}: record must be an object",))
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkReleaseError((f"could not read {path}: {error}",)) from error
    return records


def _read_jsonl_checked(path: Path, errors: list[str]) -> list[dict[str, object]]:
    try:
        return _read_jsonl(path)
    except BenchmarkReleaseError as error:
        errors.extend(error.errors)
        return []


def _read_json_checked(path: Path, errors: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"could not read {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain one JSON object")
        return {}
    return value


def _exact_fields(
    record: Mapping[str, object], expected: Iterable[str], location: str, errors: list[str]
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(record))
    unknown = sorted(set(record) - expected_set)
    if missing:
        errors.append(f"{location} missing fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"{location} unknown fields: {', '.join(unknown)}")


def _string(record: Mapping[str, object], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkReleaseError((f"{location}.{field} must be a non-empty string",))
    return value


def _expect_string(value: object, location: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location} must be a non-empty string")


def _timestamp(value: object, location: str) -> datetime:
    if not isinstance(value, str):
        raise BenchmarkReleaseError((f"{location} must be a timestamp string",))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BenchmarkReleaseError((f"{location} must be ISO 8601",)) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BenchmarkReleaseError((f"{location} must include a UTC offset",))
    return parsed


def _optional_timestamp(
    value: object, location: str, errors: list[str]
) -> datetime | None:
    if value is None:
        return None
    try:
        return _timestamp(value, location)
    except BenchmarkReleaseError as error:
        errors.extend(error.errors)
        return None


def _non_empty_string_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _string_list(value: object, location: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and bool(item.strip()) for item in value
    ):
        errors.append(f"{location} must be a string list")


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _duplicates(values: Sequence[str], label: str, errors: list[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate {label}: {', '.join(duplicates)}")


def _normalise(value: str) -> str:
    return " ".join(value.split())


def _find_leakage_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LEAKAGE_KEYS:
                found.add(key)
            found.update(_find_leakage_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_leakage_keys(item))
    return found


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the frozen Benchmark v1 release.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or __import__("sys").stdout
    args = _build_parser().parse_args(argv)
    try:
        report = validate_benchmark_v1(args.repo_root)
    except BenchmarkReleaseError as error:
        print(json.dumps({"status": "failed", "errors": list(error.errors)}), file=stdout)
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "benchmark_version": report.benchmark_version,
                "dataset_sha256": report.dataset_sha256,
                "counts": {
                    "qa": report.qa_count,
                    "summaries": report.summary_count,
                    "interactive_scenarios": report.interactive_count,
                    "gold_claims": report.claim_count,
                },
                "capability_counts": report.capability_counts,
                "human_review_status": report.human_review_status,
            },
            sort_keys=True,
        ),
        file=stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
