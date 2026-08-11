"""Strict records and deterministic scoring for the Step 4.4 temporal evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from extraction.scaled_source import (
    DEVELOPMENT_SOURCE_REFS,
    DEVELOPMENT_USER_IDS,
    load_scaled_development_sources,
)
from ingestion.contracts import stable_id
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.repository import StorageRepository
from temporal.contracts import CorrectionRequest, TemporalQuery, TransitionRequest
from temporal.service import TemporalError, TemporalService


DATASET_VERSION = "temporal_development_v1"
ALLOWED_USERS = frozenset(DEVELOPMENT_USER_IDS)
RELATIONS = frozenset(
    {"before", "after", "equal", "contains", "contained_by", "overlaps", "unknown"}
)
RUNTIME_FIELDS = frozenset(
    {"case_id", "dataset_version", "split", "user_id", "tags", "sources", "spans", "claims", "commands", "query", "interval_pairs"}
)
GOLD_FIELDS = frozenset(
    {"case_id", "dataset_version", "split", "user_id", "review_status", "expected_ordered_source_ids", "expected_normalized_claims", "expected_intervals", "expected_current_claim_ids", "expected_historical_claim_ids", "expected_visible_versions"}
)
PREDICTION_FIELDS = frozenset(
    {"case_id", "user_id", "ordered_source_ids", "normalized_claims", "intervals", "current_claim_ids", "historical_claim_ids", "visible_versions"}
)
FAILURE_FIELDS = frozenset({"failure_id", "case_id", "user_id", "code", "location"})
METRIC_NAMES = (
    "event_ordering_accuracy",
    "date_normalization_accuracy",
    "interval_relation_accuracy",
    "mean_interval_iou",
    "current_state_accuracy",
    "historical_state_accuracy",
    "correction_visibility_accuracy",
)
SANITIZED_TOKEN = re.compile(r"^[a-z0-9_:-]+$")


class TemporalEvaluationError(ValueError):
    """Reject invalid evaluation data without exposing source values."""


class _RollbackCase(Exception):
    """Carry a completed prediction out of its isolated rollback transaction."""

    def __init__(self, prediction: "TemporalPrediction") -> None:
        self.prediction = prediction


@dataclass(frozen=True)
class TemporalRuntimeCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    tags: tuple[str, ...]
    sources: tuple[Mapping[str, object], ...]
    spans: tuple[Mapping[str, object], ...]
    claims: tuple[Mapping[str, object], ...]
    commands: tuple[Mapping[str, object], ...]
    query: Mapping[str, object]
    interval_pairs: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class TemporalGoldCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    review_status: str
    expected_ordered_source_ids: tuple[str, ...] | None
    expected_normalized_claims: tuple[Mapping[str, object], ...] | None
    expected_intervals: tuple[Mapping[str, object], ...] | None
    expected_current_claim_ids: tuple[str, ...] | None
    expected_historical_claim_ids: tuple[str, ...] | None
    expected_visible_versions: tuple[Mapping[str, object], ...] | None


@dataclass(frozen=True)
class TemporalPrediction:
    case_id: str
    user_id: str
    ordered_source_ids: tuple[str, ...]
    normalized_claims: tuple[Mapping[str, object], ...]
    intervals: tuple[Mapping[str, object], ...]
    current_claim_ids: tuple[str, ...]
    historical_claim_ids: tuple[str, ...]
    visible_versions: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class TemporalFailure:
    failure_id: str
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "case_id", "user_id", "code", "location"):
            value = getattr(self, name)
            if not isinstance(value, str) or not SANITIZED_TOKEN.fullmatch(value):
                raise TemporalEvaluationError(f"{name} must be a sanitized token")


@dataclass(frozen=True)
class TemporalScorecard:
    dataset_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    event_ordering_accuracy: Mapping[str, object]
    date_normalization_accuracy: Mapping[str, object]
    interval_relation_accuracy: Mapping[str, object]
    mean_interval_iou: Mapping[str, object]
    current_state_accuracy: Mapping[str, object]
    historical_state_accuracy: Mapping[str, object]
    correction_visibility_accuracy: Mapping[str, object]


def load_temporal_runtime(path: str | Path) -> tuple[TemporalRuntimeCase, ...]:
    records = _read_jsonl(path)
    cases: list[TemporalRuntimeCase] = []
    for position, record in enumerate(records):
        _exact_fields(record, RUNTIME_FIELDS, f"runtime[{position}]")
        case_id, user_id = _common_case(record, position)
        tags = _string_tuple(record["tags"], f"runtime[{position}].tags")
        if len(tags) != len(set(tags)):
            raise TemporalEvaluationError("runtime tags must be unique")
        sources = _object_tuple(record["sources"], "sources")
        spans = _object_tuple(record["spans"], "spans")
        claims = _object_tuple(record["claims"], "claims")
        commands = _object_tuple(record["commands"], "commands")
        interval_pairs = _object_tuple(record["interval_pairs"], "interval_pairs")
        query = record["query"]
        if not isinstance(query, dict):
            raise TemporalEvaluationError("runtime query must be an object")
        _require_unique(sources, "source_id", "runtime source IDs")
        _require_unique(spans, "span_id", "runtime span IDs")
        _require_unique(claims, "claim_id", "runtime claim IDs")
        cases.append(
            TemporalRuntimeCase(
                case_id, DATASET_VERSION, "development", user_id, tags,
                sources, spans, claims, commands, dict(query), interval_pairs,
            )
        )
    _validate_release_case_set(cases)
    return tuple(cases)


def load_temporal_gold(path: str | Path) -> tuple[TemporalGoldCase, ...]:
    records = _read_jsonl(path)
    cases: list[TemporalGoldCase] = []
    for position, record in enumerate(records):
        _exact_fields(record, GOLD_FIELDS, f"gold[{position}]")
        case_id, user_id = _common_case(record, position)
        if record["review_status"] != "approved":
            raise TemporalEvaluationError("temporal gold must be approved")
        values: dict[str, object] = {}
        for name in (
            "expected_ordered_source_ids", "expected_current_claim_ids",
            "expected_historical_claim_ids",
        ):
            value = record[name]
            values[name] = None if value is None else _string_tuple(value, name)
        for name in (
            "expected_normalized_claims", "expected_intervals", "expected_visible_versions"
        ):
            value = record[name]
            values[name] = None if value is None else _object_tuple(value, name)
        expected_intervals = values["expected_intervals"]
        if expected_intervals is not None:
            for item in expected_intervals:
                _exact_fields(
                    item,
                    frozenset(
                        {"left_claim_id", "right_claim_id", "relation", "iou", "iou_reason"}
                    ),
                    "gold interval",
                )
                if item["relation"] not in RELATIONS:
                    raise TemporalEvaluationError("gold interval relation is invalid")
                iou = item["iou"]
                if iou is None:
                    if item["iou_reason"] != "unknown_or_open_interval":
                        raise TemporalEvaluationError("null interval IoU requires a reason")
                elif (
                    isinstance(iou, bool)
                    or not isinstance(iou, (int, float))
                    or not math.isfinite(float(iou))
                    or not 0 <= float(iou) <= 1
                    or item["iou_reason"] is not None
                ):
                    raise TemporalEvaluationError("gold interval IoU is invalid")
        cases.append(
            TemporalGoldCase(
                case_id, DATASET_VERSION, "development", user_id, "approved",
                values["expected_ordered_source_ids"],
                values["expected_normalized_claims"],
                values["expected_intervals"],
                values["expected_current_claim_ids"],
                values["expected_historical_claim_ids"],
                values["expected_visible_versions"],
            )
        )
    _validate_release_case_set(cases)
    return tuple(cases)


def run_temporal_cases(
    connection: object,
    cases: Sequence[TemporalRuntimeCase],
    repo_root: str | Path = ".",
) -> tuple[tuple[TemporalPrediction, ...], tuple[TemporalFailure, ...]]:
    _validate_case_set(cases)
    source_catalog = {
        (item.user_id, item.source_id): item
        for item in load_scaled_development_sources(repo_root)
    }
    if tuple(source_catalog) != DEVELOPMENT_SOURCE_REFS:
        raise TemporalEvaluationError("scaled development source order changed")
    repository = StorageRepository(connection)
    service = TemporalService(connection)
    with connection.transaction():
        for user_id in DEVELOPMENT_USER_IDS:
            repository.insert_user(
                MemoryUser(user_id, datetime(2026, 1, 1, tzinfo=timezone.utc))
            )
        repository.insert_extraction_version(
            ExtractionVersionRecord(
                "temporal_eval_v1", "deterministic", "none", "0" * 64,
                "temporal_runtime_v1", "0" * 64, "predicate_registry_v2",
                "0" * 64, "0" * 64, datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
    predictions: list[TemporalPrediction] = []
    failures: list[TemporalFailure] = []
    for case in cases:
        try:
            with connection.transaction():
                prediction = _run_case(repository, service, case, source_catalog)
                raise _RollbackCase(prediction)
        except _RollbackCase as completed:
            predictions.append(completed.prediction)
        except Exception as error:
            code = error.code if isinstance(error, TemporalError) else type(error).__name__.lower()
            location = error.location if isinstance(error, TemporalError) else "case"
            failure_id = stable_id("temporal_failure", case.case_id, code, location)
            failures.append(TemporalFailure(failure_id, case.case_id, case.user_id, code, location))
    return tuple(predictions), tuple(failures)


def score_temporal(
    predictions: Sequence[TemporalPrediction],
    failures: Sequence[TemporalFailure],
    gold: Sequence[TemporalGoldCase],
) -> TemporalScorecard:
    _validate_case_set(gold)
    prediction_by_id = {item.case_id: item for item in predictions}
    failure_by_id = {item.case_id: item for item in failures}
    if len(prediction_by_id) != len(predictions) or len(failure_by_id) != len(failures):
        raise TemporalEvaluationError("prediction and failure case IDs must be unique")
    if set(prediction_by_id) & set(failure_by_id):
        raise TemporalEvaluationError("a case cannot have both prediction and failure")
    if set(prediction_by_id) | set(failure_by_id) != {item.case_id for item in gold}:
        raise TemporalEvaluationError("predictions and failures must cover every gold case")

    exact_specs = (
        ("event_ordering_accuracy", "expected_ordered_source_ids", "ordered_source_ids", False),
        ("date_normalization_accuracy", "expected_normalized_claims", "normalized_claims", False),
        ("current_state_accuracy", "expected_current_claim_ids", "current_claim_ids", True),
        ("historical_state_accuracy", "expected_historical_claim_ids", "historical_claim_ids", True),
        ("correction_visibility_accuracy", "expected_visible_versions", "visible_versions", True),
    )
    metrics: dict[str, Mapping[str, object]] = {}
    for metric, expected_name, actual_name, compare_as_set in exact_specs:
        numerator = 0
        denominator = 0
        for expected in gold:
            value = getattr(expected, expected_name)
            if value is None:
                continue
            denominator += 1
            actual = prediction_by_id.get(expected.case_id)
            if actual is not None and _exact_match(
                getattr(actual, actual_name), value, compare_as_set
            ):
                numerator += 1
        metrics[metric] = _metric(numerator, denominator)

    relation_numerator = 0
    relation_denominator = 0
    iou_numerator = 0.0
    iou_denominator = 0
    for expected in gold:
        if expected.expected_intervals is None:
            continue
        actual = prediction_by_id.get(expected.case_id)
        actual_by_pair = {
            (item["left_claim_id"], item["right_claim_id"]): item
            for item in (() if actual is None else actual.intervals)
        }
        for wanted in expected.expected_intervals:
            pair = (wanted["left_claim_id"], wanted["right_claim_id"])
            got = actual_by_pair.get(pair)
            relation_denominator += 1
            if got is not None and got.get("relation") == wanted.get("relation"):
                relation_numerator += 1
            if wanted.get("iou") is not None:
                iou_denominator += 1
                if got is not None and got.get("iou") is not None:
                    iou_numerator += float(got["iou"])
            elif wanted.get("iou_reason") != "unknown_or_open_interval":
                raise TemporalEvaluationError("null interval IoU requires a reason")
    metrics["interval_relation_accuracy"] = _metric(relation_numerator, relation_denominator)
    metrics["mean_interval_iou"] = _metric(iou_numerator, iou_denominator)
    return TemporalScorecard(
        DATASET_VERSION, len(gold), len(predictions), len(failures),
        *(metrics[name] for name in METRIC_NAMES),
    )


def interval_relation_and_iou(left: Mapping[str, object], right: Mapping[str, object]) -> tuple[str, float | None, str | None]:
    if left["time_precision"] == "unknown" or right["time_precision"] == "unknown":
        return "unknown", None, "unknown_or_open_interval"
    kind = "timestamp" if left["time_precision"] == "timestamp" else "date"
    if (right["time_precision"] == "timestamp") != (kind == "timestamp"):
        return "unknown", None, "unknown_or_open_interval"
    left_start = _boundary(left["valid_from"], kind)
    left_end = _boundary(left["valid_to"], kind)
    right_start = _boundary(right["valid_from"], kind)
    right_end = _boundary(right["valid_to"], kind)
    if None in (left_start, left_end, right_start, right_end):
        return "unknown", None, "unknown_or_open_interval"
    assert left_start is not None and left_end is not None
    assert right_start is not None and right_end is not None
    if left_end < right_start:
        relation = "before"
    elif left_start > right_end:
        relation = "after"
    elif left_start == right_start and left_end == right_end:
        relation = "equal"
    elif left_end == right_start or left_start == right_end:
        relation = "overlaps"
    elif left_start <= right_start and left_end >= right_end:
        relation = "contains"
    elif right_start <= left_start and right_end >= left_end:
        relation = "contained_by"
    else:
        relation = "overlaps"
    if kind == "date":
        intersection = (min(left_end, right_end) - max(left_start, right_start)).days + 1
        union = (max(left_end, right_end) - min(left_start, right_start)).days + 1
        iou = max(0, intersection) / union
    else:
        if left_start == left_end or right_start == right_end:
            iou = 1.0 if left_start == left_end == right_start == right_end else 0.0
        else:
            intersection = (min(left_end, right_end) - max(left_start, right_start)).total_seconds()
            union = (max(left_end, right_end) - min(left_start, right_start)).total_seconds()
            iou = max(0.0, intersection) / union
    return relation, round(iou, 6), None


def record(value: object) -> dict[str, object]:
    return _json_safe(asdict(value))


def serialize_jsonl(values: Sequence[object]) -> bytes:
    """Return the release's canonical, byte-stable JSONL representation."""

    return b"".join(
        json.dumps(
            record(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for value in values
    )


def _run_case(repository: StorageRepository, service: TemporalService, case: TemporalRuntimeCase, catalog: Mapping[tuple[str, str], object]) -> TemporalPrediction:
    source_records: dict[str, SourceEventRecord] = {}
    for item in case.sources:
        _exact_fields(item, frozenset({"source_id", "source_ref", "ingested_at"}), "source")
        source_ref = _string(item["source_ref"], "source_ref")
        adapted = catalog.get((case.user_id, source_ref))
        if adapted is None:
            raise TemporalEvaluationError("runtime source reference is not development-owned")
        observations = adapted.observations
        raw_content = "\n".join(observation.text for observation in observations)
        produced_at = min(observation.observed_at for observation in observations)
        source = SourceEventRecord(
            _string(item["source_id"], "source_id"), case.user_id, adapted.source_type,
            None, f"temporal:{case.case_id}:{item['source_id']}", produced_at,
            _timestamp(item["ingested_at"], "source.ingested_at"), raw_content,
            [entity.entity_id for entity in adapted.known_entities], {},
            hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        )
        repository.insert_source_event(source)
        source_records[source.source_id] = source
    spans: dict[str, SourceSpanRecord] = {}
    for item in case.spans:
        _exact_fields(item, frozenset({"span_id", "source_id", "message_id", "speaker_id", "quote"}), "span")
        source = source_records.get(item["source_id"])
        if source is None or not isinstance(item["quote"], str) or item["quote"] not in source.raw_content:
            raise TemporalEvaluationError("runtime span provenance is invalid")
        span = SourceSpanRecord(
            _string(item["span_id"], "span_id"), case.user_id,
            _string(item["source_id"], "source_id"), item["message_id"],
            _string(item["speaker_id"], "speaker_id"), item["quote"],
        )
        repository.insert_source_span(span)
        spans[span.span_id] = span
    claims: dict[str, ClaimRecord] = {}
    for item in case.claims:
        _exact_fields(item, frozenset({"claim_id", "version_id", "span_ids", "subject_id", "speaker_id", "predicate", "object_json", "polarity", "epistemic_status", "valid_from", "valid_to", "time_precision", "memory_kind", "sensitivity", "transaction_from"}), "claim")
        precision = _string(item["time_precision"], "time_precision")
        valid_from_date, valid_from_timestamp = _valid_boundary(item["valid_from"], precision)
        valid_to_date, valid_to_timestamp = _valid_boundary(item["valid_to"], precision)
        claim = ClaimRecord(
            _string(item["claim_id"], "claim_id"), case.user_id,
            _string(item["subject_id"], "subject_id"), _string(item["speaker_id"], "speaker_id"),
            _string(item["predicate"], "predicate"), "predicate_registry_v2",
            item["object_json"], item["polarity"], item["epistemic_status"],
            valid_from_date, valid_from_timestamp, valid_to_date, valid_to_timestamp,
            precision, 1.0, item["memory_kind"], item["sensitivity"], "temporal_eval_v1",
        )
        repository.insert_claim(claim)
        transaction_from = _timestamp(item["transaction_from"], "transaction_from")
        repository.insert_claim_version(
            ClaimVersionRecord(
                _string(item["version_id"], "version_id"), case.user_id,
                claim.claim_id, "candidate", transaction_from,
                valid_from_date=valid_from_date, valid_from_timestamp=valid_from_timestamp,
                valid_to_date=valid_to_date, valid_to_timestamp=valid_to_timestamp,
                time_precision=precision,
            )
        )
        for span_id in _string_tuple(item["span_ids"], "span_ids"):
            if span_id not in spans:
                raise TemporalEvaluationError("claim span reference is invalid")
            repository.insert_evidence_link(
                EvidenceLinkRecord(case.user_id, claim.claim_id, span_id, "supports", 1.0)
            )
        claims[claim.claim_id] = claim
    for command in case.commands:
        kind = command.get("type")
        if kind == "transition":
            _exact_fields(command, frozenset({"type", "claim_id", "idempotency_key", "target_status", "reason", "transitioned_at", "belief_confidence"}), "transition")
            service.transition(
                TransitionRequest(
                    case.user_id, command["claim_id"], command["idempotency_key"],
                    command["target_status"], command["reason"],
                    _timestamp(command["transitioned_at"], "transitioned_at"),
                    command["belief_confidence"],
                )
            )
        elif kind == "correction":
            _exact_fields(command, frozenset({"type", "replaced_claim_id", "replacement_claim_id", "idempotency_key", "replacement_target_status", "reason", "transitioned_at", "replaced_belief_confidence", "replacement_belief_confidence"}), "correction")
            service.correct(
                CorrectionRequest(
                    case.user_id, command["replaced_claim_id"], command["replacement_claim_id"],
                    command["idempotency_key"], command["replacement_target_status"], command["reason"],
                    _timestamp(command["transitioned_at"], "transitioned_at"),
                    command["replaced_belief_confidence"], command["replacement_belief_confidence"],
                )
            )
        else:
            raise TemporalEvaluationError("runtime command type is invalid")
    query = case.query
    _exact_fields(query, frozenset({"transaction_as_of", "valid_at", "valid_at_type", "statuses"}), "query")
    visible = service.query(
        TemporalQuery(
            case.user_id, _timestamp(query["transaction_as_of"], "transaction_as_of"),
            _query_valid_at(query["valid_at"], query["valid_at_type"]),
            frozenset(_string_tuple(query["statuses"], "statuses")),
        )
    )
    normalized_claims = tuple(
        {
            "claim_id": claim.claim_id,
            "valid_from": _format_boundary(claim.valid_from_date, claim.valid_from_timestamp),
            "valid_to": _format_boundary(claim.valid_to_date, claim.valid_to_timestamp),
            "time_precision": claim.time_precision,
        }
        for claim in sorted(claims.values(), key=lambda value: value.claim_id)
    )
    intervals = []
    normalized_by_id = {item["claim_id"]: item for item in normalized_claims}
    for pair in case.interval_pairs:
        _exact_fields(pair, frozenset({"left_claim_id", "right_claim_id"}), "interval_pair")
        left_id, right_id = pair["left_claim_id"], pair["right_claim_id"]
        relation, iou, reason = interval_relation_and_iou(normalized_by_id[left_id], normalized_by_id[right_id])
        intervals.append({"left_claim_id": left_id, "right_claim_id": right_id, "relation": relation, "iou": iou, "iou_reason": reason})
    ordered_sources = tuple(
        item.source_id for item in sorted(source_records.values(), key=lambda value: (value.produced_at, value.source_id))
    )
    visible_versions = tuple(
        {"claim_id": item.claim.claim_id, "version_id": item.version.version_id, "status": item.version.lifecycle_status}
        for item in visible
    )
    return TemporalPrediction(
        case.case_id, case.user_id, ordered_sources, normalized_claims, tuple(intervals),
        tuple(item["claim_id"] for item in visible_versions if item["status"] == "current"),
        tuple(item["claim_id"] for item in visible_versions if item["status"] == "historical"),
        visible_versions,
    )


def _metric(numerator: int | float, denominator: int) -> Mapping[str, object]:
    if denominator == 0:
        return {"value": None, "numerator": numerator, "denominator": 0, "null_reason": "zero_denominator"}
    return {"value": round(float(numerator) / denominator, 6), "numerator": numerator, "denominator": denominator, "null_reason": None}


def _exact_match(actual: object, expected: object, compare_as_set: bool) -> bool:
    if not compare_as_set:
        return actual == expected
    if not isinstance(actual, tuple) or not isinstance(expected, tuple):
        return False
    return {
        json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for item in actual
    } == {
        json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for item in expected
    }


def _valid_boundary(value: object, precision: str) -> tuple[date | None, datetime | None]:
    if value is None:
        return None, None
    if precision == "timestamp":
        return None, _timestamp(value, "valid boundary")
    if precision == "unknown":
        raise TemporalEvaluationError("unknown precision cannot have a boundary")
    try:
        return date.fromisoformat(_string(value, "valid boundary")), None
    except ValueError:
        raise TemporalEvaluationError("valid date boundary is invalid") from None


def _query_valid_at(value: object, kind: object) -> date | datetime | None:
    if kind == "none" and value is None:
        return None
    if kind == "date":
        try:
            return date.fromisoformat(_string(value, "valid_at"))
        except ValueError:
            raise TemporalEvaluationError("query valid_at date is invalid") from None
    if kind == "timestamp":
        return _timestamp(value, "valid_at")
    raise TemporalEvaluationError("query valid_at type is invalid")


def _boundary(value: object, kind: str) -> date | datetime | None:
    if value is None:
        return None
    return _timestamp(value, "interval boundary") if kind == "timestamp" else date.fromisoformat(str(value))


def _format_boundary(date_value: date | None, timestamp_value: datetime | None) -> str | None:
    if timestamp_value is not None:
        return timestamp_value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return None if date_value is None else date_value.isoformat()


def _timestamp(value: object, location: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(value, location).replace("Z", "+00:00"))
    except ValueError:
        raise TemporalEvaluationError(f"{location} must be an ISO timestamp") from None
    if parsed.utcoffset() is None:
        raise TemporalEvaluationError(f"{location} must be timezone-aware")
    return parsed


def _common_case(record: Mapping[str, object], position: int) -> tuple[str, str]:
    case_id = _string(record["case_id"], f"case[{position}].case_id")
    user_id = _string(record["user_id"], f"case[{position}].user_id")
    if record["dataset_version"] != DATASET_VERSION or record["split"] != "development":
        raise TemporalEvaluationError("temporal case release identity is invalid")
    if user_id not in ALLOWED_USERS:
        raise TemporalEvaluationError("temporal cases must use development users")
    return case_id, user_id


def _validate_case_set(cases: Sequence[object]) -> None:
    ids = [getattr(item, "case_id") for item in cases]
    if len(ids) != len(set(ids)):
        raise TemporalEvaluationError("temporal case IDs must be unique")


def _validate_release_case_set(cases: Sequence[object]) -> None:
    _validate_case_set(cases)
    counts = {
        user: sum(getattr(item, "user_id") == user for item in cases)
        for user in DEVELOPMENT_USER_IDS
    }
    if len(cases) != 12 or counts != {"user_001": 6, "user_002": 6}:
        raise TemporalEvaluationError(
            "temporal release must contain exactly six cases per development user"
        )


def _read_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    records = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TemporalEvaluationError(f"line {line_number} is invalid JSON: {error.msg}") from None
                if not isinstance(value, dict):
                    raise TemporalEvaluationError(f"line {line_number} must be an object")
                records.append(value)
    except OSError as error:
        raise TemporalEvaluationError(f"could not read temporal data: {error}") from error
    return tuple(records)


def _exact_fields(value: Mapping[str, object], fields: frozenset[str], location: str) -> None:
    if set(value) != fields:
        raise TemporalEvaluationError(f"{location} fields are invalid")


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TemporalEvaluationError(f"{location} must be non-empty text")
    return value


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TemporalEvaluationError(f"{location} must be a list")
    return tuple(_string(item, location) for item in value)


def _object_tuple(value: object, location: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TemporalEvaluationError(f"{location} must be a list of objects")
    return tuple(dict(item) for item in value)


def _require_unique(records: Sequence[Mapping[str, object]], key: str, location: str) -> None:
    values = [record.get(key) for record in records]
    if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
        raise TemporalEvaluationError(f"{location} must be unique non-empty strings")


def _json_safe(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
