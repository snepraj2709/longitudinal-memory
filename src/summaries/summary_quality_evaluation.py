"""Strict deterministic scoring for the frozen summary-quality checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .summary_quality_runtime import (
    BASELINE_VERSION,
    EXPECTED_CASE_IDS,
    RUNTIME_VERSION,
    ClaimSnapshot,
    ExactEvidence,
    PredictedStatement,
    SummaryQualityRuntimeError,
    TemporalFailure,
    TemporalPrediction,
    VisibleSessionSummary,
)


SCORER_VERSION = "summary_quality_scorer_v2"
CHECKPOINT_VERSION = "summary_quality_development_runtime_checkpoint_v2"
CHECKPOINT_MANIFEST_SHA256 = (
    "6b0a474e42962ac792516c0ed024fa78d9d52842974cf8e8f221f0297e04500e"
)
GUIDANCE_VERSION = "step-6.4-guidance-v2"
GUIDANCE_SHA256 = (
    "3b5673d169a1ad58cffda7ff45bd83ffe0a541da9ed38e870be99503842955b9"
)
RUNTIME_MODULE_SHA256 = (
    "5b8b991d7fe5ea6a723fe7ca18929351ade8559b6e3082aa38c43afc228ea093"
)
CASE_COUNT = 10
DEVELOPMENT_USERS = ("user_001", "user_002")
EVIDENCE_FIELDS = ("source_id", "message_id", "quote")
CLAIM_MATCH_FIELDS = (
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
)
GOLD_CASE_FIELDS = frozenset(
    {
        "as_of",
        "benchmark_version",
        "capability",
        "case_id",
        "difficulty",
        "evidence",
        "failure_tags",
        "gold_event_ids",
        "instruction",
        "reference_summary",
        "required_claim_ids",
        "review_status",
        "should_abstain",
        "split",
        "task",
        "user_id",
    }
)
GOLD_CLAIM_FIELDS = frozenset(
    {
        "benchmark_version",
        "claim_id",
        "epistemic_status",
        "evidence",
        "memory_kind",
        "object",
        "polarity",
        "predicate",
        "review_status",
        "speaker_id",
        "status",
        "subject_id",
        "time_precision",
        "user_id",
        "valid_from",
        "valid_to",
    }
)
MAP_FIELDS = frozenset({"case_id", "events", "review_status", "user_id"})
EVENT_FIELDS = frozenset(
    {
        "correction_role",
        "event_id",
        "evidence",
        "expected_state",
        "required_claim_ids",
        "uncertainty_expected",
    }
)


class SummaryQualityEvaluationError(ValueError):
    """Reject incomplete scorer inputs without leaking source content."""


@dataclass(frozen=True, order=True)
class GoldEvidence:
    source_id: str
    message_id: str | None
    quote: str

    def __post_init__(self) -> None:
        _text(self.source_id, "source_id")
        if self.message_id is not None:
            _text(self.message_id, "message_id")
        _text(self.quote, "quote")


@dataclass(frozen=True)
class SummaryGoldCase:
    case_id: str
    user_id: str
    as_of: datetime
    gold_event_ids: tuple[str, ...]
    required_claim_ids: tuple[str, ...]
    evidence: tuple[GoldEvidence, ...]
    reference_summary: str

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _user(self.user_id)
        _aware(self.as_of, "as_of")
        _ordered_unique(self.gold_event_ids, "gold_event_ids")
        _ordered_unique(self.required_claim_ids, "required_claim_ids")
        if not self.evidence or len(self.evidence) != len(set(self.evidence)):
            raise SummaryQualityEvaluationError("case evidence is empty or duplicated")
        _text(self.reference_summary, "reference_summary")


@dataclass(frozen=True)
class GoldClaim:
    claim_id: str
    user_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object: object
    polarity: str
    epistemic_status: str
    valid_from: str | None
    valid_to: str | None
    time_precision: str
    status: str
    evidence: tuple[GoldEvidence, ...]

    def __post_init__(self) -> None:
        for name in (
            "claim_id", "subject_id", "speaker_id", "predicate", "polarity",
            "epistemic_status", "time_precision", "status",
        ):
            _text(getattr(self, name), name)
        _user(self.user_id)
        object.__setattr__(self, "object", _json_value(self.object))
        if not self.evidence or len(self.evidence) != len(set(self.evidence)):
            raise SummaryQualityEvaluationError("Claim evidence is empty or duplicated")


@dataclass(frozen=True)
class MappedGoldEvent:
    event_id: str
    required_claim_ids: tuple[str, ...]
    evidence: tuple[GoldEvidence, ...]
    expected_state: str
    correction_role: str
    uncertainty_expected: bool

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        _ordered_unique(self.required_claim_ids, "event required_claim_ids")
        if not self.evidence or len(self.evidence) != len(set(self.evidence)):
            raise SummaryQualityEvaluationError("event evidence is empty or duplicated")
        if self.expected_state not in {"current", "historical", "none"}:
            raise SummaryQualityEvaluationError("event expected_state is invalid")
        if self.correction_role not in {"correcting", "corrected", "none"}:
            raise SummaryQualityEvaluationError("event correction_role is invalid")
        if not isinstance(self.uncertainty_expected, bool):
            raise SummaryQualityEvaluationError("event uncertainty flag is invalid")


@dataclass(frozen=True)
class EventMapCase:
    case_id: str
    user_id: str
    events: tuple[MappedGoldEvent, ...]
    review_status: str

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _user(self.user_id)
        if not self.events:
            raise SummaryQualityEvaluationError("event map case is empty")
        event_ids = tuple(item.event_id for item in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise SummaryQualityEvaluationError("event map contains duplicate events")
        if self.review_status != "implementation_reviewed":
            raise SummaryQualityEvaluationError("event map is not implementation reviewed")


@dataclass(frozen=True)
class EventMatch:
    case_id: str
    statement_id: str
    event_id: str


@dataclass(frozen=True)
class Metric:
    numerator: int | float
    denominator: int
    value: float | None
    null_reason: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or not isinstance(self.numerator, (int, float))
            or self.numerator < 0
            or self.denominator < 0
            or self.numerator > self.denominator
        ):
            raise SummaryQualityEvaluationError("metric counts are invalid")
        if self.denominator == 0:
            if self.value is not None or not self.null_reason:
                raise SummaryQualityEvaluationError("null metric is missing its reason")
        elif (
            self.value is None
            or abs(self.value - self.numerator / self.denominator) > 1e-12
            or self.null_reason is not None
        ):
            raise SummaryQualityEvaluationError("evaluated metric value is invalid")


@dataclass(frozen=True)
class SummaryQualityScorecard:
    scorer_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    gold_event_micro_precision: Metric
    gold_event_micro_recall: Metric
    gold_event_micro_f1: Metric
    gold_event_macro_precision: Metric
    gold_event_macro_recall: Metric
    gold_event_macro_f1: Metric
    supporting_evidence_micro_precision: Metric
    supporting_evidence_micro_recall: Metric
    supporting_evidence_micro_f1: Metric
    current_historical_accuracy: Metric
    correction_preservation: Metric
    uncertainty_preservation: Metric
    case_accounting: Metric
    provenance_coverage: Metric
    cross_user_prediction_count: int
    stale_reference_count: int
    unsupported_statement_count: int
    runtime_failure_count: int

    def __post_init__(self) -> None:
        if self.scorer_version != SCORER_VERSION:
            raise SummaryQualityEvaluationError("scorer version changed")
        for name in (
            "case_count", "prediction_count", "failure_count",
            "cross_user_prediction_count", "stale_reference_count",
            "unsupported_statement_count", "runtime_failure_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SummaryQualityEvaluationError("scorecard count is invalid")


def load_summary_gold_prefix(path: str | Path) -> tuple[SummaryGoldCase, ...]:
    """Read only the authorized ten-record scorer prefix."""

    rows = _read_prefix(path, CASE_COUNT, "summary case")
    cases: list[SummaryGoldCase] = []
    for index, row in enumerate(rows):
        if frozenset(row) != GOLD_CASE_FIELDS:
            raise SummaryQualityEvaluationError("summary case fields changed")
        if row.get("case_id") != EXPECTED_CASE_IDS[index]:
            raise SummaryQualityEvaluationError("summary case order or identity changed")
        expected_user = "user_001" if index < 5 else "user_002"
        expected = {
            "benchmark_version": "scaled_v1",
            "difficulty": "longitudinal",
            "review_status": "pending_human_review",
            "should_abstain": False,
            "split": "development",
            "task": "summarization",
            "user_id": expected_user,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise SummaryQualityEvaluationError("summary case contract changed")
        cases.append(
            SummaryGoldCase(
                case_id=_string(row.get("case_id"), "case_id"),
                user_id=_string(row.get("user_id"), "user_id"),
                as_of=_datetime(row.get("as_of"), "as_of"),
                gold_event_ids=_strings(row.get("gold_event_ids"), "gold_event_ids"),
                required_claim_ids=_strings(row.get("required_claim_ids"), "required_claim_ids"),
                evidence=_evidence(row.get("evidence")),
                reference_summary=_string(row.get("reference_summary"), "reference_summary"),
            )
        )
    return tuple(cases)


def load_required_claims(path: str | Path) -> tuple[GoldClaim, ...]:
    rows = _read_all(path, "required Claim")
    claims: list[GoldClaim] = []
    seen: set[str] = set()
    for row in rows:
        if frozenset(row) != GOLD_CLAIM_FIELDS:
            raise SummaryQualityEvaluationError("required Claim fields changed")
        claim_id = _string(row.get("claim_id"), "claim_id")
        if claim_id in seen:
            raise SummaryQualityEvaluationError("required Claim is duplicated")
        seen.add(claim_id)
        if row.get("benchmark_version") != "scaled_v1" or row.get("review_status") != "pending_human_review":
            raise SummaryQualityEvaluationError("required Claim contract changed")
        claims.append(
            GoldClaim(
                claim_id=claim_id,
                user_id=_string(row.get("user_id"), "user_id"),
                subject_id=_string(row.get("subject_id"), "subject_id"),
                speaker_id=_string(row.get("speaker_id"), "speaker_id"),
                predicate=_string(row.get("predicate"), "predicate"),
                object=row.get("object"),
                polarity=_string(row.get("polarity"), "polarity"),
                epistemic_status=_string(row.get("epistemic_status"), "epistemic_status"),
                valid_from=_optional_string(row.get("valid_from"), "valid_from"),
                valid_to=_optional_string(row.get("valid_to"), "valid_to"),
                time_precision=_string(row.get("time_precision"), "time_precision"),
                status=_string(row.get("status"), "status"),
                evidence=_evidence(row.get("evidence")),
            )
        )
    return tuple(claims)


def load_event_map(path: str | Path) -> tuple[EventMapCase, ...]:
    rows = _read_all(path, "event map")
    if len(rows) != CASE_COUNT:
        raise SummaryQualityEvaluationError("event map case count changed")
    mapped: list[EventMapCase] = []
    for index, row in enumerate(rows):
        if frozenset(row) != MAP_FIELDS or row.get("case_id") != EXPECTED_CASE_IDS[index]:
            raise SummaryQualityEvaluationError("event map case fields or order changed")
        raw_events = row.get("events")
        if not isinstance(raw_events, list):
            raise SummaryQualityEvaluationError("event map events are invalid")
        events: list[MappedGoldEvent] = []
        for raw in raw_events:
            if not isinstance(raw, dict) or frozenset(raw) != EVENT_FIELDS:
                raise SummaryQualityEvaluationError("event map event fields changed")
            events.append(
                MappedGoldEvent(
                    event_id=_string(raw.get("event_id"), "event_id"),
                    required_claim_ids=_strings(raw.get("required_claim_ids"), "required_claim_ids"),
                    evidence=_evidence(raw.get("evidence")),
                    expected_state=_string(raw.get("expected_state"), "expected_state"),
                    correction_role=_string(raw.get("correction_role"), "correction_role"),
                    uncertainty_expected=raw.get("uncertainty_expected"),
                )
            )
        mapped.append(
            EventMapCase(
                case_id=_string(row.get("case_id"), "case_id"),
                user_id=_string(row.get("user_id"), "user_id"),
                events=tuple(events),
                review_status=_string(row.get("review_status"), "review_status"),
            )
        )
    return tuple(mapped)


def load_scorer_inputs(
    checkpoint_root: str | Path,
    case_path: str | Path,
    claim_path: str | Path,
    event_map_path: str | Path,
) -> tuple[
    tuple[TemporalPrediction, ...],
    tuple[TemporalFailure, ...],
    tuple[SummaryGoldCase, ...],
    tuple[GoldClaim, ...],
    tuple[EventMapCase, ...],
]:
    """Finish checkpoint validation before opening any scorer-only input."""

    predictions, failures = load_runtime_checkpoint(checkpoint_root)
    cases = load_summary_gold_prefix(case_path)
    claims = load_required_claims(claim_path)
    event_map = load_event_map(event_map_path)
    validate_scorer_inputs(predictions, failures, cases, claims, event_map)
    return predictions, failures, cases, claims, event_map


def load_runtime_checkpoint(
    root: str | Path,
) -> tuple[tuple[TemporalPrediction, ...], tuple[TemporalFailure, ...]]:
    directory = Path(root)
    runtime_names = {"predictions.jsonl", "failures.jsonl", "run.json", "manifest.json"}
    expected_names = {
        *runtime_names,
        "checkpoint_preflight.json",
        "checkpoint_manifest.json",
    }
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != expected_names:
        raise SummaryQualityEvaluationError("runtime checkpoint artifacts changed")
    if _sha(directory / "checkpoint_manifest.json") != CHECKPOINT_MANIFEST_SHA256:
        raise SummaryQualityEvaluationError("runtime checkpoint manifest hash mismatch")
    checkpoint_manifest = _read_object(
        directory / "checkpoint_manifest.json", "checkpoint manifest"
    )
    if (
        frozenset(checkpoint_manifest)
        != {
            "artifact_version",
            "artifacts",
            "base_commit",
            "case_count",
            "failure_count",
            "guidance",
            "immutable",
            "prediction_count",
            "runtime_module_sha256",
        }
        or checkpoint_manifest.get("artifact_version") != CHECKPOINT_VERSION
        or checkpoint_manifest.get("base_commit")
        != "d52b4a2a9a1178ab37354fc65fd1d202519185cc"
        or checkpoint_manifest.get("immutable") is not True
        or checkpoint_manifest.get("case_count") != CASE_COUNT
        or checkpoint_manifest.get("prediction_count") != CASE_COUNT
        or checkpoint_manifest.get("failure_count") != 0
        or checkpoint_manifest.get("runtime_module_sha256") != RUNTIME_MODULE_SHA256
        or checkpoint_manifest.get("guidance")
        != {"sha256": GUIDANCE_SHA256, "version": GUIDANCE_VERSION}
    ):
        raise SummaryQualityEvaluationError("runtime checkpoint manifest fields changed")
    checkpoint_artifacts = checkpoint_manifest.get("artifacts")
    expected_checkpoint_artifacts = runtime_names | {"checkpoint_preflight.json"}
    if (
        not isinstance(checkpoint_artifacts, dict)
        or set(checkpoint_artifacts) != expected_checkpoint_artifacts
    ):
        raise SummaryQualityEvaluationError("runtime checkpoint artifact map changed")
    for name, expected_hash in checkpoint_artifacts.items():
        if not isinstance(expected_hash, str) or _sha(directory / name) != expected_hash:
            raise SummaryQualityEvaluationError("runtime checkpoint artifact hash mismatch")
    preflight = _read_object(
        directory / "checkpoint_preflight.json", "checkpoint preflight"
    )
    absence = preflight.get("absence_attestation")
    model_usage = preflight.get("model_usage")
    if (
        preflight.get("artifact_version")
        != "summary_quality_development_runtime_checkpoint_preflight_v2"
        or preflight.get("base_commit")
        != "d52b4a2a9a1178ab37354fc65fd1d202519185cc"
        or preflight.get("guidance")
        != {"sha256": GUIDANCE_SHA256, "version": GUIDANCE_VERSION}
        or preflight.get("runtime_module")
        != {
            "path": "src/summaries/summary_quality_runtime.py",
            "sha256": RUNTIME_MODULE_SHA256,
        }
        or not isinstance(absence, dict)
        or absence.get("verified_before_runtime_generation") is not True
        or absence.get("gold_or_scorer_restored") is not False
        or preflight.get("prior_authorized_v1_development_gold_exposure") is not True
        or tuple(preflight.get("case_ids", ())) != EXPECTED_CASE_IDS
        or preflight.get("user_case_counts") != {"user_001": 5, "user_002": 5}
        or preflight.get("runtime_trials")
        != {"byte_identical": True, "count": 2}
        or preflight.get("outputs")
        != {name: checkpoint_artifacts[name] for name in runtime_names}
        or model_usage
        != {
            "cost_usd": "0",
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
            "retries": 0,
        }
    ):
        raise SummaryQualityEvaluationError("runtime checkpoint preflight changed")
    manifest = _read_object(directory / "manifest.json", "runtime manifest")
    if (
        frozenset(manifest) != {"artifacts", "runtime_version"}
        or manifest.get("runtime_version") != RUNTIME_VERSION
    ):
        raise SummaryQualityEvaluationError("runtime manifest fields changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != runtime_names - {"manifest.json"}:
        raise SummaryQualityEvaluationError("runtime manifest artifact map changed")
    for name, expected_hash in artifacts.items():
        if not isinstance(expected_hash, str) or _sha(directory / name) != expected_hash:
            raise SummaryQualityEvaluationError("runtime checkpoint hash mismatch")
    run = _read_object(directory / "run.json", "runtime run")
    prediction_rows = _read_all(directory / "predictions.jsonl", "runtime prediction")
    failure_rows = _read_all(directory / "failures.jsonl", "runtime failure")
    predictions = tuple(_prediction(row) for row in prediction_rows)
    failures = tuple(_failure(row) for row in failure_rows)
    if run.get("case_count") != len(predictions) + len(failures):
        raise SummaryQualityEvaluationError("runtime case accounting differs")
    if run.get("prediction_count") != len(predictions) or run.get("failure_count") != len(failures):
        raise SummaryQualityEvaluationError("runtime output accounting differs")
    if run.get("model_calls") != 0 or run.get("baseline_version") != BASELINE_VERSION:
        raise SummaryQualityEvaluationError("runtime execution contract changed")
    ids = [item.case_id for item in predictions] + [item.case_id for item in failures]
    if len(ids) != len(set(ids)) or set(ids) != set(EXPECTED_CASE_IDS):
        raise SummaryQualityEvaluationError("runtime cases are duplicated or incomplete")
    return predictions, failures


def validate_scorer_inputs(
    predictions: Sequence[TemporalPrediction],
    failures: Sequence[TemporalFailure],
    cases: Sequence[SummaryGoldCase],
    claims: Sequence[GoldClaim],
    event_map: Sequence[EventMapCase],
) -> None:
    if tuple(item.case_id for item in cases) != EXPECTED_CASE_IDS:
        raise SummaryQualityEvaluationError("scorer cases changed")
    if tuple(item.case_id for item in event_map) != EXPECTED_CASE_IDS:
        raise SummaryQualityEvaluationError("event map cases changed")
    output_ids = [item.case_id for item in predictions] + [item.case_id for item in failures]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(EXPECTED_CASE_IDS):
        raise SummaryQualityEvaluationError("prediction and failure accounting changed")
    claim_by_id = {item.claim_id: item for item in claims}
    if len(claim_by_id) != len(claims):
        raise SummaryQualityEvaluationError("required Claims are duplicated")
    required_all = {claim_id for case in cases for claim_id in case.required_claim_ids}
    if set(claim_by_id) != required_all:
        raise SummaryQualityEvaluationError("required Claim set is incomplete or extra")
    for case, mapped in zip(cases, event_map, strict=True):
        if mapped.case_id != case.case_id or mapped.user_id != case.user_id:
            raise SummaryQualityEvaluationError("event map case ownership differs")
        if tuple(item.event_id for item in mapped.events) != case.gold_event_ids:
            raise SummaryQualityEvaluationError("event map event set or order differs")
        mapped_claims = {claim_id for event in mapped.events for claim_id in event.required_claim_ids}
        if mapped_claims != set(case.required_claim_ids):
            raise SummaryQualityEvaluationError("event map Claim union differs")
        mapped_evidence = {item for event in mapped.events for item in event.evidence}
        if mapped_evidence != set(case.evidence):
            raise SummaryQualityEvaluationError("event map evidence union differs")
        for event in mapped.events:
            for claim_id in event.required_claim_ids:
                claim = claim_by_id.get(claim_id)
                if claim is None or claim.user_id != case.user_id:
                    raise SummaryQualityEvaluationError("event map Claim ownership differs")
            if any(item not in case.evidence for item in event.evidence):
                raise SummaryQualityEvaluationError("event map evidence is outside its case")
            if event.correction_role != "none" and not any(
                claim_by_id[claim_id].epistemic_status == "corrected"
                for claim_id in event.required_claim_ids
            ):
                raise SummaryQualityEvaluationError("correction event lacks a corrected Claim")
            if event.expected_state != "none" and not any(
                claim_by_id[claim_id].status == event.expected_state
                for claim_id in event.required_claim_ids
            ):
                raise SummaryQualityEvaluationError("event expected state lacks support")


def match_case_events(
    prediction: TemporalPrediction,
    mapped: EventMapCase,
    claims: Mapping[str, GoldClaim],
) -> tuple[EventMatch, ...]:
    statements = tuple(
        sorted(
            (
                statement
                for session in prediction.session_summaries
                for statement in session.observed_events
            ),
            key=lambda item: item.statement_id,
        )
    )
    events = tuple(sorted(mapped.events, key=lambda item: item.event_id))
    edges = {
        statement.statement_id: tuple(
            event.event_id
            for event in events
            if _event_edge(statement, event, claims)
        )
        for statement in statements
    }
    event_index = {item.event_id: index for index, item in enumerate(events)}

    @lru_cache(maxsize=None)
    def choose(index: int, used: int) -> tuple[tuple[str, str], ...]:
        if index == len(statements):
            return ()
        statement_id = statements[index].statement_id
        candidates = [choose(index + 1, used)]
        for event_id in edges[statement_id]:
            bit = 1 << event_index[event_id]
            if used & bit:
                continue
            candidates.append(
                ((statement_id, event_id),) + choose(index + 1, used | bit)
            )
        return min(candidates, key=lambda item: (-len(item), item))

    selected = choose(0, 0)
    return tuple(
        EventMatch(case_id=prediction.case_id, statement_id=statement_id, event_id=event_id)
        for statement_id, event_id in selected
    )


def score_summary_quality(
    predictions: Sequence[TemporalPrediction],
    failures: Sequence[TemporalFailure],
    cases: Sequence[SummaryGoldCase],
    gold_claims: Sequence[GoldClaim],
    event_map: Sequence[EventMapCase],
) -> SummaryQualityScorecard:
    validate_scorer_inputs(predictions, failures, cases, gold_claims, event_map)
    prediction_by_case = {item.case_id: item for item in predictions}
    map_by_case = {item.case_id: item for item in event_map}
    claim_by_id = {item.claim_id: item for item in gold_claims}
    matches_by_case: dict[str, tuple[EventMatch, ...]] = {}
    predicted_events = 0
    gold_events = 0
    event_true_positive = 0
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    predicted_evidence = 0
    gold_evidence = 0
    evidence_true_positive = 0
    state_num = state_den = 0
    correction_num = correction_den = 0
    uncertainty_num = uncertainty_den = 0
    provenance_num = provenance_den = 0
    cross_user = stale = unsupported = 0
    for case in cases:
        mapped = map_by_case[case.case_id]
        prediction = prediction_by_case.get(case.case_id)
        statements = () if prediction is None else tuple(
            statement
            for session in prediction.session_summaries
            for statement in session.observed_events
        )
        questions = () if prediction is None else tuple(
            statement
            for session in prediction.session_summaries
            for statement in session.unresolved_questions
        )
        matches = () if prediction is None else match_case_events(prediction, mapped, claim_by_id)
        matches_by_case[case.case_id] = matches
        p_count = len(statements)
        g_count = len(mapped.events)
        tp_count = len(matches)
        predicted_events += p_count
        gold_events += g_count
        event_true_positive += tp_count
        precision = tp_count / p_count if p_count else 0.0
        recall = tp_count / g_count if g_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1
        statements_by_id = {item.statement_id: item for item in statements}
        events_by_id = {item.event_id: item for item in mapped.events}
        predicted_evidence += sum(len(item.evidence) for item in statements)
        gold_evidence += sum(len(item.evidence) for item in mapped.events)
        for match in matches:
            statement = statements_by_id[match.statement_id]
            event = events_by_id[match.event_id]
            evidence_true_positive += len(_statement_evidence(statement) & set(event.evidence))
            if event.expected_state in {"current", "historical"}:
                state_den += 1
                if any(item.lifecycle_status == event.expected_state for item in statement.claims):
                    state_num += 1
        for event in mapped.events:
            if event.correction_role != "none":
                correction_den += 1
                if prediction is not None and _preserves_all_required(
                    statements, event, claim_by_id
                ):
                    correction_num += 1
            if event.uncertainty_expected:
                uncertainty_den += 1
                matched_statement = next(
                    (
                        statements_by_id[item.statement_id]
                        for item in matches
                        if item.event_id == event.event_id
                    ),
                    None,
                )
                if matched_statement is not None and _preserves_uncertainty(
                    matched_statement, questions, event
                ):
                    uncertainty_num += 1
        for statement in (*statements, *questions):
            provenance_den += 1
            if _lineage_valid(statement, case.user_id):
                provenance_num += 1
            else:
                stale += 1
            cross_user += sum(item.user_id != case.user_id for item in statement.claims)
            if not statement.evidence:
                unsupported += 1
    micro_precision = _metric(event_true_positive, predicted_events, "no_predicted_events")
    micro_recall = _metric(event_true_positive, gold_events, "no_gold_events")
    evidence_precision = _metric(evidence_true_positive, predicted_evidence, "no_predicted_evidence")
    evidence_recall = _metric(evidence_true_positive, gold_evidence, "no_gold_evidence")
    case_den = len(cases)
    return SummaryQualityScorecard(
        scorer_version=SCORER_VERSION,
        case_count=case_den,
        prediction_count=len(predictions),
        failure_count=len(failures),
        gold_event_micro_precision=micro_precision,
        gold_event_micro_recall=micro_recall,
        gold_event_micro_f1=_f1_metric(micro_precision, micro_recall, "no_event_precision_or_recall"),
        gold_event_macro_precision=_float_metric(macro_precision_sum, case_den, "no_cases"),
        gold_event_macro_recall=_float_metric(macro_recall_sum, case_den, "no_cases"),
        gold_event_macro_f1=_float_metric(macro_f1_sum, case_den, "no_cases"),
        supporting_evidence_micro_precision=evidence_precision,
        supporting_evidence_micro_recall=evidence_recall,
        supporting_evidence_micro_f1=_f1_metric(evidence_precision, evidence_recall, "no_evidence_precision_or_recall"),
        current_historical_accuracy=_metric(state_num, state_den, "no_matched_reviewed_state_events"),
        correction_preservation=_metric(correction_num, correction_den, "no_reviewed_correction_events"),
        uncertainty_preservation=_metric(uncertainty_num, uncertainty_den, "no_reviewed_uncertainty_events"),
        case_accounting=_metric(len(predictions) + len(failures), case_den, "no_cases"),
        provenance_coverage=_metric(provenance_num, provenance_den, "no_predicted_statements"),
        cross_user_prediction_count=cross_user,
        stale_reference_count=stale,
        unsupported_statement_count=unsupported,
        runtime_failure_count=len(failures),
    )


def canonical_json(value: object) -> bytes:
    return (json.dumps(_record(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _event_edge(
    statement: PredictedStatement,
    event: MappedGoldEvent,
    claims: Mapping[str, GoldClaim],
) -> bool:
    claim_match = any(
        _claim_equal(predicted, claims[claim_id])
        for predicted in statement.claims
        for claim_id in event.required_claim_ids
    )
    return claim_match and bool(_statement_evidence(statement) & set(event.evidence))


def _claim_equal(predicted: ClaimSnapshot, expected: GoldClaim) -> bool:
    return all(
        _canonical(getattr(predicted, field)) == _canonical(getattr(expected, field))
        for field in CLAIM_MATCH_FIELDS
    )


def _statement_evidence(statement: PredictedStatement) -> set[GoldEvidence]:
    return {
        GoldEvidence(item.source_id, item.message_id, item.quote)
        for item in statement.evidence
    }


def _preserves_all_required(
    statements: Sequence[PredictedStatement],
    event: MappedGoldEvent,
    claims: Mapping[str, GoldClaim],
) -> bool:
    for claim_id in event.required_claim_ids:
        expected = claims[claim_id]
        if not any(
            _claim_equal(predicted, expected)
            and bool(_statement_evidence(statement) & set(expected.evidence))
            for statement in statements
            for predicted in statement.claims
        ):
            return False
    return True


def _preserves_uncertainty(
    statement: PredictedStatement,
    questions: Sequence[PredictedStatement],
    event: MappedGoldEvent,
) -> bool:
    if any(
        item.lifecycle_status == "disputed"
        or item.epistemic_status in {"uncertain", "reported_by_other"}
        for item in statement.claims
    ):
        return True
    required = set(event.required_claim_ids)
    event_evidence = set(event.evidence)
    return any(
        required & {item.claim_id for item in question.claims}
        and event_evidence & _statement_evidence(question)
        for question in questions
    )


def _lineage_valid(statement: PredictedStatement, user_id: str) -> bool:
    claims = {(item.claim_id, item.claim_version_id) for item in statement.claims}
    evidence = {(item.claim_id, item.claim_version_id) for item in statement.evidence}
    return claims == evidence and all(item.user_id == user_id for item in statement.claims)


def _metric(numerator: int, denominator: int, reason: str) -> Metric:
    if denominator == 0:
        return Metric(0, 0, None, reason)
    return Metric(numerator, denominator, numerator / denominator, None)


def _float_metric(total: float, denominator: int, reason: str) -> Metric:
    if denominator == 0:
        return Metric(0, 0, None, reason)
    return Metric(total, denominator, total / denominator, None)


def _f1_metric(precision: Metric, recall: Metric, reason: str) -> Metric:
    if precision.value is None or recall.value is None:
        return Metric(0, 0, None, reason)
    value = (
        2 * precision.value * recall.value / (precision.value + recall.value)
        if precision.value + recall.value
        else 0.0
    )
    return Metric(value, 1, value, None)


def _prediction(row: Mapping[str, object]) -> TemporalPrediction:
    sessions_raw = row.get("session_summaries")
    if not isinstance(sessions_raw, list):
        raise SummaryQualityEvaluationError("runtime sessions are invalid")
    sessions = tuple(_session(item) for item in sessions_raw if isinstance(item, dict))
    if len(sessions) != len(sessions_raw):
        raise SummaryQualityEvaluationError("runtime session row is invalid")
    try:
        return TemporalPrediction(
            case_id=_string(row.get("case_id"), "case_id"),
            user_id=_string(row.get("user_id"), "user_id"),
            capability=_string(row.get("capability"), "capability"),
            instruction=_string(row.get("instruction"), "instruction"),
            as_of=_datetime(row.get("as_of"), "as_of"),
            baseline_version=_string(row.get("baseline_version"), "baseline_version"),
            session_summaries=sessions,
            durative_claim_ids=_strings_allow_empty(row.get("durative_claim_ids"), "durative_claim_ids"),
        )
    except SummaryQualityRuntimeError as error:
        raise SummaryQualityEvaluationError("runtime prediction is invalid") from error


def _session(row: Mapping[str, object]) -> VisibleSessionSummary:
    observed = row.get("observed_events")
    questions = row.get("unresolved_questions")
    if not isinstance(observed, list) or not isinstance(questions, list):
        raise SummaryQualityEvaluationError("runtime statements are invalid")
    try:
        return VisibleSessionSummary(
            summary_id=_string(row.get("summary_id"), "summary_id"),
            session_id=_string(row.get("session_id"), "session_id"),
            start_at=_datetime(row.get("start_at"), "start_at"),
            end_at=_datetime(row.get("end_at"), "end_at"),
            source_ids=_strings(row.get("source_ids"), "source_ids"),
            summary_text=_string(row.get("summary_text"), "summary_text"),
            observed_events=tuple(_statement(item) for item in observed if isinstance(item, dict)),
            unresolved_questions=tuple(_statement(item) for item in questions if isinstance(item, dict)),
        )
    except SummaryQualityRuntimeError as error:
        raise SummaryQualityEvaluationError("runtime session is invalid") from error


def _statement(row: Mapping[str, object]) -> PredictedStatement:
    raw_claims = row.get("claims")
    raw_evidence = row.get("evidence")
    if not isinstance(raw_claims, list) or not isinstance(raw_evidence, list):
        raise SummaryQualityEvaluationError("runtime statement lineage is invalid")
    try:
        claims = tuple(
            ClaimSnapshot(**item) for item in raw_claims if isinstance(item, dict)
        )
        evidence = tuple(
            ExactEvidence(**item) for item in raw_evidence if isinstance(item, dict)
        )
        if len(claims) != len(raw_claims) or len(evidence) != len(raw_evidence):
            raise SummaryQualityEvaluationError("runtime statement lineage row is invalid")
        return PredictedStatement(
            statement_id=_string(row.get("statement_id"), "statement_id"),
            session_id=_string(row.get("session_id"), "session_id"),
            text=_string(row.get("text"), "text"),
            lifecycle_view=_string(row.get("lifecycle_view"), "lifecycle_view"),
            claims=claims,
            evidence=evidence,
        )
    except (SummaryQualityRuntimeError, TypeError) as error:
        raise SummaryQualityEvaluationError("runtime statement is invalid") from error


def _failure(row: Mapping[str, object]) -> TemporalFailure:
    try:
        return TemporalFailure(
            case_id=_string(row.get("case_id"), "case_id"),
            user_id=_string(row.get("user_id"), "user_id"),
            code=_string(row.get("code"), "code"),
            location=_string(row.get("location"), "location"),
        )
    except SummaryQualityRuntimeError as error:
        raise SummaryQualityEvaluationError("runtime failure is invalid") from error


def _evidence(value: object) -> tuple[GoldEvidence, ...]:
    if not isinstance(value, list) or not value:
        raise SummaryQualityEvaluationError("evidence list is invalid")
    result = []
    for item in value:
        if not isinstance(item, dict) or frozenset(item) != frozenset(EVIDENCE_FIELDS):
            raise SummaryQualityEvaluationError("evidence fields changed")
        result.append(
            GoldEvidence(
                source_id=_string(item.get("source_id"), "source_id"),
                message_id=_optional_string(item.get("message_id"), "message_id"),
                quote=_string(item.get("quote"), "quote"),
            )
        )
    return tuple(result)


def _read_prefix(path: str | Path, count: int, label: str) -> tuple[Mapping[str, object], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for index in range(count):
            line = stream.readline()
            if not line:
                raise SummaryQualityEvaluationError(f"{label} prefix is incomplete")
            rows.append(_decode(line, f"{label} {index + 1}"))
    return tuple(rows)


def _read_all(path: str | Path, label: str) -> tuple[Mapping[str, object], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                raise SummaryQualityEvaluationError(f"{label} contains a blank row")
            rows.append(_decode(line, f"{label} {index}"))
    return tuple(rows)


def _read_object(path: Path, label: str) -> Mapping[str, object]:
    return _decode(path.read_text(encoding="utf-8"), label)


def _decode(text: str, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise SummaryQualityEvaluationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise SummaryQualityEvaluationError(f"{label} must be an object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _record(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return _record(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> object:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise SummaryQualityEvaluationError("Claim object is not JSON-safe") from error


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SummaryQualityEvaluationError(f"{label} is empty")
    return value


def _text(value: object, label: str) -> None:
    _string(value, label)


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _strings(value: object, label: str) -> tuple[str, ...]:
    result = _strings_allow_empty(value, label)
    if not result:
        raise SummaryQualityEvaluationError(f"{label} is empty")
    return result


def _strings_allow_empty(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SummaryQualityEvaluationError(f"{label} is invalid")
    result = tuple(_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise SummaryQualityEvaluationError(f"{label} contains duplicates")
    return result


def _ordered_unique(value: tuple[str, ...], label: str) -> None:
    if not value or len(value) != len(set(value)):
        raise SummaryQualityEvaluationError(f"{label} is empty or duplicated")


def _datetime(value: object, label: str) -> datetime:
    text = _string(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SummaryQualityEvaluationError(f"{label} is invalid") from error
    _aware(result, label)
    return result


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SummaryQualityEvaluationError(f"{label} must be timezone-aware")


def _user(value: str) -> None:
    if value not in DEVELOPMENT_USERS:
        raise SummaryQualityEvaluationError("user_id is outside development")
