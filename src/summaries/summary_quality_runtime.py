"""Deterministic runtime checkpoint for development summary-quality cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


BASELINE_VERSION = "all_visible_session_summaries_v1"
RUNTIME_VERSION = "summary_quality_development_runtime_v1"
DEVELOPMENT_USERS = ("user_001", "user_002")
PREFIX_COUNT = 10
EXPECTED_CASE_IDS = (
    "scaled_user_001_summary_temporal_reasoning_001",
    "scaled_user_001_summary_temporal_reasoning_002",
    "scaled_user_001_summary_temporal_reasoning_003",
    "scaled_user_001_summary_user_modeling_001",
    "scaled_user_001_summary_user_modeling_002",
    "scaled_user_002_summary_temporal_reasoning_001",
    "scaled_user_002_summary_temporal_reasoning_002",
    "scaled_user_002_summary_user_modeling_001",
    "scaled_user_002_summary_user_modeling_002",
    "scaled_user_002_summary_user_modeling_003",
)
CASE_FIELDS = frozenset(
    {
        "as_of",
        "benchmark_version",
        "capability",
        "case_id",
        "difficulty",
        "instruction",
        "split",
        "task",
        "user_id",
    }
)
CLAIM_FIELDS = (
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
)
TOKEN = re.compile(r"^[a-z0-9_:-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SummaryQualityRuntimeError(ValueError):
    """Reject incomplete, cross-user, stale, or non-deterministic runtime state."""


@dataclass(frozen=True)
class TemporalRuntimeCase:
    case_id: str
    user_id: str
    capability: str
    instruction: str
    as_of: datetime

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _user(self.user_id)
        if self.capability not in {"temporal_reasoning", "user_modeling"}:
            raise SummaryQualityRuntimeError("case capability is invalid")
        _text(self.instruction, "instruction")
        _aware(self.as_of, "as_of")


@dataclass(frozen=True)
class ClaimSnapshot:
    claim_id: str
    claim_version_id: str
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
    lifecycle_status: str

    def __post_init__(self) -> None:
        for name in (
            "claim_id",
            "claim_version_id",
            "subject_id",
            "speaker_id",
            "predicate",
            "polarity",
            "epistemic_status",
            "time_precision",
            "lifecycle_status",
        ):
            _text(getattr(self, name), name)
        _user(self.user_id)
        object.__setattr__(self, "object", _safe_json(self.object, "claim object"))


@dataclass(frozen=True)
class ExactEvidence:
    claim_id: str
    claim_version_id: str
    source_id: str
    span_id: str
    message_id: str | None
    quote: str
    support_type: str

    def __post_init__(self) -> None:
        for name in (
            "claim_id", "claim_version_id", "source_id", "span_id", "quote"
        ):
            _text(getattr(self, name), name)
        if self.message_id is not None:
            _text(self.message_id, "message_id")
        if self.support_type not in {"supports", "contradicts", "corrects"}:
            raise SummaryQualityRuntimeError("support_type is invalid")


@dataclass(frozen=True)
class PredictedStatement:
    statement_id: str
    session_id: str
    text: str
    lifecycle_view: str
    claims: tuple[ClaimSnapshot, ...]
    evidence: tuple[ExactEvidence, ...]

    def __post_init__(self) -> None:
        _sha(self.statement_id, "statement_id")
        _sha(self.session_id, "session_id")
        _text(self.text, "statement text")
        if self.lifecycle_view not in {"accepted", "candidate", "historical", "disputed"}:
            raise SummaryQualityRuntimeError("lifecycle_view is invalid")
        if not self.claims or not self.evidence:
            raise SummaryQualityRuntimeError("statement lineage is incomplete")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if claim_ids != tuple(sorted(set(claim_ids))):
            raise SummaryQualityRuntimeError("statement claims are not stably ordered")
        evidence_keys = tuple(
            (item.claim_id, item.claim_version_id, item.source_id, item.span_id)
            for item in self.evidence
        )
        if evidence_keys != tuple(sorted(set(evidence_keys))):
            raise SummaryQualityRuntimeError("statement evidence is not stably ordered")
        if set(claim_ids) != {item.claim_id for item in self.evidence}:
            raise SummaryQualityRuntimeError("statement Claim and evidence lineage differs")
        versions = {item.claim_version_id for item in self.claims}
        if versions != {item.claim_version_id for item in self.evidence}:
            raise SummaryQualityRuntimeError("statement version and evidence lineage differs")


@dataclass(frozen=True)
class VisibleSessionSummary:
    summary_id: str
    session_id: str
    start_at: datetime
    end_at: datetime
    source_ids: tuple[str, ...]
    summary_text: str
    observed_events: tuple[PredictedStatement, ...]
    unresolved_questions: tuple[PredictedStatement, ...]

    def __post_init__(self) -> None:
        _sha(self.summary_id, "summary_id")
        _sha(self.session_id, "session_id")
        _aware(self.start_at, "session start_at")
        _aware(self.end_at, "session end_at")
        if self.end_at < self.start_at:
            raise SummaryQualityRuntimeError("session time is invalid")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise SummaryQualityRuntimeError("session sources are invalid")
        _text(self.summary_text, "summary_text")
        if not self.observed_events and not self.unresolved_questions:
            raise SummaryQualityRuntimeError("visible summary is empty")
        source_ids = set(self.source_ids)
        for statement in (*self.observed_events, *self.unresolved_questions):
            if statement.session_id != self.session_id:
                raise SummaryQualityRuntimeError("statement session lineage differs")
            if any(item.source_id not in source_ids for item in statement.evidence):
                raise SummaryQualityRuntimeError("statement source is outside its session")


@dataclass(frozen=True)
class TemporalPrediction:
    case_id: str
    user_id: str
    capability: str
    instruction: str
    as_of: datetime
    baseline_version: str
    session_summaries: tuple[VisibleSessionSummary, ...]
    durative_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _user(self.user_id)
        _text(self.capability, "capability")
        _text(self.instruction, "instruction")
        _aware(self.as_of, "as_of")
        if self.baseline_version != BASELINE_VERSION:
            raise SummaryQualityRuntimeError("baseline version changed")
        order = tuple((item.start_at, item.session_id) for item in self.session_summaries)
        if order != tuple(sorted(order)):
            raise SummaryQualityRuntimeError("session summaries are not stably ordered")
        if tuple(sorted(set(self.durative_claim_ids))) != self.durative_claim_ids:
            raise SummaryQualityRuntimeError("durative Claim IDs are not stably ordered")
        for summary in self.session_summaries:
            if summary.end_at > self.as_of:
                raise SummaryQualityRuntimeError("prediction contains a future session")
            for statement in (*summary.observed_events, *summary.unresolved_questions):
                if any(claim.user_id != self.user_id for claim in statement.claims):
                    raise SummaryQualityRuntimeError("prediction contains a cross-user Claim")


@dataclass(frozen=True)
class TemporalFailure:
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _user(self.user_id)
        if TOKEN.fullmatch(self.code) is None or TOKEN.fullmatch(self.location) is None:
            raise SummaryQualityRuntimeError("failure details are not sanitized")


def load_temporal_runtime(path: str | Path) -> tuple[TemporalRuntimeCase, ...]:
    """Read and validate only the approved ten-record development prefix."""

    records: list[Mapping[str, object]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for index in range(PREFIX_COUNT):
            line = stream.readline()
            if not line:
                raise SummaryQualityRuntimeError("runtime prefix is incomplete")
            records.append(_object(line, f"runtime case {index + 1}"))
    cases: list[TemporalRuntimeCase] = []
    for index, record in enumerate(records):
        if frozenset(record) != CASE_FIELDS:
            raise SummaryQualityRuntimeError("runtime case fields changed")
        if record["case_id"] != EXPECTED_CASE_IDS[index]:
            raise SummaryQualityRuntimeError("runtime case order or identity changed")
        expected_user = "user_001" if index < 5 else "user_002"
        expected = {
            "benchmark_version": "scaled_v1",
            "difficulty": "longitudinal",
            "split": "development",
            "task": "summarization",
            "user_id": expected_user,
        }
        if any(record[name] != value for name, value in expected.items()):
            raise SummaryQualityRuntimeError("runtime case contract changed")
        cases.append(
            TemporalRuntimeCase(
                case_id=_string(record["case_id"], "case_id"),
                user_id=_string(record["user_id"], "user_id"),
                capability=_string(record["capability"], "capability"),
                instruction=_string(record["instruction"], "instruction"),
                as_of=_datetime(record["as_of"], "as_of"),
            )
        )
    return tuple(cases)


def load_jsonl_records(path: str | Path) -> tuple[Mapping[str, object], ...]:
    """Load a frozen development-only payload used by the runtime bundle."""

    records: list[Mapping[str, object]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                raise SummaryQualityRuntimeError("development payload contains a blank row")
            records.append(_object(line, f"development row {index}"))
    return tuple(records)


def run_temporal_cases(
    cases: Sequence[TemporalRuntimeCase],
    summary_rows: Sequence[Mapping[str, object]],
    session_rows: Sequence[Mapping[str, object]],
    claim_rows: Sequence[Mapping[str, object]],
    durative_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[TemporalPrediction, ...]:
    """Bundle every user-visible frozen summary without instruction filtering."""

    if tuple(item.case_id for item in cases) != EXPECTED_CASE_IDS:
        raise SummaryQualityRuntimeError("runtime cases are incomplete or reordered")
    sessions = _unique(session_rows, "definition_id", "session definition")
    claims = _unique(claim_rows, "claim_id", "Claim")
    summaries = _unique(summary_rows, "summary_id", "summary")
    duratives = _unique(durative_rows, "claim_id", "durative Claim")
    predictions: list[TemporalPrediction] = []
    for case in cases:
        user_summaries = tuple(
            row for row in summaries.values() if row.get("user_id") == case.user_id
        )
        visible: list[VisibleSessionSummary] = []
        for row in user_summaries:
            session_id = _string(row.get("session_definition_id"), "session_definition_id")
            session = sessions.get(session_id)
            if session is None:
                raise SummaryQualityRuntimeError("summary session is missing")
            if session.get("user_id") != case.user_id:
                raise SummaryQualityRuntimeError("summary session ownership differs")
            session_cutoff = _datetime(session.get("transaction_as_of"), "transaction_as_of")
            end_at = _datetime(session.get("end_at"), "end_at")
            if session_cutoff > case.as_of or end_at > case.as_of:
                continue
            visible.append(_build_summary(row, session, claims, case.user_id))
        visible.sort(key=lambda item: (item.start_at, item.session_id))
        durative_ids = tuple(
            sorted(
                claim_id
                for claim_id, row in duratives.items()
                if row.get("user_id") == case.user_id
            )
        )
        predictions.append(
            TemporalPrediction(
                case_id=case.case_id,
                user_id=case.user_id,
                capability=case.capability,
                instruction=case.instruction,
                as_of=case.as_of,
                baseline_version=BASELINE_VERSION,
                session_summaries=tuple(visible),
                durative_claim_ids=durative_ids,
            )
        )
    return tuple(predictions)


def canonical_jsonl(records: Sequence[object]) -> bytes:
    return b"".join(
        (json.dumps(_record(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in records
    )


def write_runtime_checkpoint(
    output_dir: str | Path,
    predictions: Sequence[TemporalPrediction],
    failures: Sequence[TemporalFailure],
) -> Mapping[str, str]:
    """Write a new immutable runtime checkpoint, refusing a nonempty directory."""

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise SummaryQualityRuntimeError("runtime checkpoint directory is not empty")
    root.mkdir(parents=True, exist_ok=True)
    prediction_bytes = canonical_jsonl(predictions)
    failure_bytes = canonical_jsonl(failures)
    run = {
        "baseline_version": BASELINE_VERSION,
        "case_count": len(predictions) + len(failures),
        "failure_count": len(failures),
        "model_calls": 0,
        "prediction_count": len(predictions),
        "runtime_version": RUNTIME_VERSION,
    }
    run_bytes = _canonical_object(run)
    hashes = {
        "failures.jsonl": _digest(failure_bytes),
        "predictions.jsonl": _digest(prediction_bytes),
        "run.json": _digest(run_bytes),
    }
    manifest_bytes = _canonical_object(
        {
            "artifacts": hashes,
            "runtime_version": RUNTIME_VERSION,
        }
    )
    payloads = {
        "predictions.jsonl": prediction_bytes,
        "failures.jsonl": failure_bytes,
        "run.json": run_bytes,
        "manifest.json": manifest_bytes,
    }
    for name, payload in payloads.items():
        with (root / name).open("xb") as stream:
            stream.write(payload)
    return {**hashes, "manifest.json": _digest(manifest_bytes)}


def _build_summary(
    row: Mapping[str, object],
    session: Mapping[str, object],
    claims: Mapping[str, Mapping[str, object]],
    user_id: str,
) -> VisibleSessionSummary:
    session_id = _string(row.get("session_definition_id"), "session_definition_id")
    source_ids = _string_tuple(session.get("source_ids"), "source_ids", sorted_values=False)
    raw_statements = row.get("statements")
    if not isinstance(raw_statements, list) or not raw_statements:
        raise SummaryQualityRuntimeError("summary statements are missing")
    observed: list[PredictedStatement] = []
    questions: list[PredictedStatement] = []
    for value in raw_statements:
        if not isinstance(value, dict):
            raise SummaryQualityRuntimeError("summary statement is invalid")
        statement = _build_statement(value, session_id, source_ids, claims, user_id)
        kind = value.get("statement_kind")
        if kind == "observed_fact":
            observed.append(statement)
        elif kind == "unresolved_question":
            questions.append(statement)
        else:
            raise SummaryQualityRuntimeError("statement kind is invalid")
    observed.sort(key=lambda item: item.statement_id)
    questions.sort(key=lambda item: item.statement_id)
    return VisibleSessionSummary(
        summary_id=_string(row.get("summary_id"), "summary_id"),
        session_id=session_id,
        start_at=_datetime(session.get("start_at"), "start_at"),
        end_at=_datetime(session.get("end_at"), "end_at"),
        source_ids=source_ids,
        summary_text=_string(row.get("summary_text"), "summary_text"),
        observed_events=tuple(observed),
        unresolved_questions=tuple(questions),
    )


def _build_statement(
    row: Mapping[str, object],
    session_id: str,
    source_ids: tuple[str, ...],
    claims: Mapping[str, Mapping[str, object]],
    user_id: str,
) -> PredictedStatement:
    raw_evidence = row.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise SummaryQualityRuntimeError("statement evidence is missing")
    evidence: list[ExactEvidence] = []
    snapshots: dict[str, ClaimSnapshot] = {}
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise SummaryQualityRuntimeError("statement evidence is invalid")
        claim_id = _string(item.get("claim_id"), "claim_id")
        claim = claims.get(claim_id)
        if claim is None:
            raise SummaryQualityRuntimeError("statement Claim is missing")
        if claim.get("user_id") != user_id:
            raise SummaryQualityRuntimeError("statement Claim ownership differs")
        source_id = _string(item.get("source_id"), "source_id")
        if source_id not in source_ids:
            raise SummaryQualityRuntimeError("statement source is outside its session")
        raw_claim_evidence = claim.get("evidence")
        if not isinstance(raw_claim_evidence, list):
            raise SummaryQualityRuntimeError("Claim evidence is invalid")
        exact = [value for value in raw_claim_evidence if isinstance(value, dict) and value.get("source_id") == source_id]
        if len(exact) != 1:
            raise SummaryQualityRuntimeError("exact Claim evidence is missing or ambiguous")
        claim_version_id = _string(item.get("claim_version_id"), "claim_version_id")
        lifecycle_view = _string(row.get("lifecycle_view"), "lifecycle_view")
        lifecycle_status = claim.get("lifecycle_status")
        if lifecycle_status is None:
            if lifecycle_view not in {"candidate", "historical", "disputed"}:
                raise SummaryQualityRuntimeError("exact lifecycle status is unavailable")
            lifecycle_status = lifecycle_view
        snapshots[claim_id] = ClaimSnapshot(
            claim_id=claim_id,
            claim_version_id=claim_version_id,
            user_id=user_id,
            subject_id=_string(claim.get("subject_id"), "subject_id"),
            speaker_id=_string(claim.get("speaker_id"), "speaker_id"),
            predicate=_string(claim.get("predicate"), "predicate"),
            object=claim.get("object"),
            polarity=_string(claim.get("polarity"), "polarity"),
            epistemic_status=_string(claim.get("epistemic_status"), "epistemic_status"),
            valid_from=_optional_string(claim.get("valid_from"), "valid_from"),
            valid_to=_optional_string(claim.get("valid_to"), "valid_to"),
            time_precision=_string(claim.get("time_precision"), "time_precision"),
            lifecycle_status=_string(lifecycle_status, "lifecycle_status"),
        )
        exact_item = exact[0]
        evidence.append(
            ExactEvidence(
                claim_id=claim_id,
                claim_version_id=claim_version_id,
                source_id=source_id,
                span_id=_string(item.get("span_id"), "span_id"),
                message_id=_optional_string(exact_item.get("message_id"), "message_id"),
                quote=_string(exact_item.get("quote"), "quote"),
                support_type=_string(item.get("support_type"), "support_type"),
            )
        )
    declared_claims = _string_tuple(row.get("claim_ids"), "claim_ids", sorted_values=True)
    if declared_claims != tuple(sorted(snapshots)):
        raise SummaryQualityRuntimeError("statement declared Claim IDs differ")
    return PredictedStatement(
        statement_id=_string(row.get("statement_id"), "statement_id"),
        session_id=session_id,
        text=_string(row.get("text"), "text"),
        lifecycle_view=_string(row.get("lifecycle_view"), "lifecycle_view"),
        claims=tuple(snapshots[key] for key in sorted(snapshots)),
        evidence=tuple(sorted(evidence, key=lambda item: (item.claim_id, item.claim_version_id, item.source_id, item.span_id))),
    )


def _unique(
    rows: Sequence[Mapping[str, object]], key: str, label: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        value = _string(row.get(key), key)
        if value in result:
            raise SummaryQualityRuntimeError(f"duplicate {label}")
        result[value] = row
    return result


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


def _canonical_object(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(line: str, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SummaryQualityRuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise SummaryQualityRuntimeError(f"{label} must be an object")
    return value


def _safe_json(value: object, label: str) -> object:
    def reject(item: object) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SummaryQualityRuntimeError(f"{label} is not JSON-safe")
            return
        if isinstance(item, list):
            for child in item:
                reject(child)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise SummaryQualityRuntimeError(f"{label} is not JSON-safe")
            for child in item.values():
                reject(child)
            return
        raise SummaryQualityRuntimeError(f"{label} is not JSON-safe")

    reject(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SummaryQualityRuntimeError(f"{label} is empty")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _string_tuple(value: object, label: str, *, sorted_values: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SummaryQualityRuntimeError(f"{label} is invalid")
    result = tuple(_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise SummaryQualityRuntimeError(f"{label} contains duplicates")
    if sorted_values and result != tuple(sorted(result)):
        raise SummaryQualityRuntimeError(f"{label} is not sorted")
    return result


def _datetime(value: object, label: str) -> datetime:
    text = _string(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SummaryQualityRuntimeError(f"{label} is invalid") from error
    _aware(result, label)
    return result


def _text(value: object, label: str) -> None:
    _string(value, label)


def _user(value: str) -> None:
    if value not in DEVELOPMENT_USERS:
        raise SummaryQualityRuntimeError("user_id is outside the development split")


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SummaryQualityRuntimeError(f"{label} must be timezone-aware")


def _sha(value: str, label: str) -> None:
    if SHA256.fullmatch(value) is None:
        raise SummaryQualityRuntimeError(f"{label} must be a lowercase SHA-256")
