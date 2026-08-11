"""Typed contracts for deterministic durative-claim inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from storage.contracts import StorageValidationError, safe_json


RULES_VERSION = "durative_claim_rules_v1"
REGISTRY_VERSION = "predicate_registry_v2"
REGISTRY_PATH = "configs/extraction/predicate_registry_v2.json"
REGISTRY_SHA256 = "5cb9ba2f2b7a81aa2ce61d52d9425fbf1f2d39964a3a66286d5679ebb2e5175d"
ALLOWED_FAMILIES = ("belief", "goal", "preference", "relationship", "role", "state")
ELIGIBLE_LIFECYCLE = ("confirmed", "current", "historical")
ELIGIBLE_EPISTEMIC = ("asserted", "corrected")
COUNTER_RULES = (
    "explicit_contradicts_or_corrects_exact_proposition",
    "ineligible_exact_assertion",
    "persisted_unresolved_correction_or_conflict",
    "incompatible_value_when_time_overlaps_or_unknown",
)
REJECTION_PRECEDENCE = (
    "predicate_not_durative",
    "no_exact_support",
    "restricted_evidence",
    "recursive_durative",
    "memory_kind_ineligible",
    "not_visible",
    "lifecycle_ineligible",
    "epistemic_ineligible",
    "counterevidence",
    "insufficient_repetition",
)
ACCEPT_REASONS = frozenset(
    {"accepted_repeated_episodes", "accepted_explicit_closed_interval"}
)
REJECT_REASONS = frozenset(REJECTION_PRECEDENCE)
POLARITIES = frozenset({"positive", "negative"})
EPISTEMIC_STATUSES = frozenset(
    {"asserted", "inferred", "reported_by_other", "hypothetical", "uncertain", "denied", "corrected"}
)
LIFECYCLE_STATUSES = frozenset(
    {"candidate", "confirmed", "current", "historical", "disputed", "superseded", "excluded"}
)
TIME_PRECISIONS = frozenset(
    {"timestamp", "day", "month", "year", "approximate", "unknown"}
)
SENSITIVITIES = frozenset({"standard", "sensitive", "restricted"})
SUPPORT_TYPES = frozenset({"supports", "contradicts", "corrects"})
CONFIG_FIELDS = frozenset(
    {
        "rules_version", "predicate_registry_path", "predicate_registry_version",
        "predicate_registry_sha256", "allowed_predicate_families",
        "required_temporal_behavior", "eligible_memory_kind",
        "eligible_lifecycle_statuses", "eligible_epistemic_statuses",
        "minimum_distinct_sessions", "minimum_distinct_sources",
        "minimum_distinct_episode_times", "explicit_closed_interval_minimum",
        "counterevidence_rules", "rejection_precedence", "derived_memory_kind",
        "derived_epistemic_status", "derived_lifecycle_status",
        "derived_speaker_id", "derived_belief_confidence",
        "multi_episode_time_precision", "unknown_time_precision", "review_status",
    }
)


class DurativeClaimError(ValueError):
    """Reject unsafe or inconsistent durative inference inputs."""


@dataclass(frozen=True)
class DurativeRulesConfig:
    rules_version: str
    predicate_registry_path: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    allowed_predicate_families: tuple[str, ...]
    required_temporal_behavior: str
    eligible_memory_kind: str
    eligible_lifecycle_statuses: tuple[str, ...]
    eligible_epistemic_statuses: tuple[str, ...]
    minimum_distinct_sessions: int
    minimum_distinct_sources: int
    minimum_distinct_episode_times: int
    explicit_closed_interval_minimum: int
    counterevidence_rules: tuple[str, ...]
    rejection_precedence: tuple[str, ...]
    derived_memory_kind: str
    derived_epistemic_status: str
    derived_lifecycle_status: str
    derived_speaker_id: str
    derived_belief_confidence: str
    multi_episode_time_precision: str
    unknown_time_precision: str
    review_status: str

    def __post_init__(self) -> None:
        actual = (
            self.rules_version, self.predicate_registry_path,
            self.predicate_registry_version, self.predicate_registry_sha256,
            self.allowed_predicate_families, self.required_temporal_behavior,
            self.eligible_memory_kind, self.eligible_lifecycle_statuses,
            self.eligible_epistemic_statuses, self.minimum_distinct_sessions,
            self.minimum_distinct_sources, self.minimum_distinct_episode_times,
            self.explicit_closed_interval_minimum, self.counterevidence_rules,
            self.rejection_precedence, self.derived_memory_kind,
            self.derived_epistemic_status, self.derived_lifecycle_status,
            self.derived_speaker_id, self.derived_belief_confidence,
            self.multi_episode_time_precision, self.unknown_time_precision,
            self.review_status,
        )
        expected = (
            RULES_VERSION, REGISTRY_PATH, REGISTRY_VERSION, REGISTRY_SHA256,
            ALLOWED_FAMILIES, "interval", "episodic", ELIGIBLE_LIFECYCLE,
            ELIGIBLE_EPISTEMIC, 2, 2, 2, 1, COUNTER_RULES,
            REJECTION_PRECEDENCE, "durative", "inferred", "candidate",
            "memory_system", "null_only", "approximate", "unknown",
            "implementation_reviewed",
        )
        if actual != expected:
            raise DurativeClaimError("durative rules configuration changed")


@dataclass(frozen=True)
class DurativeInferenceRequest:
    user_id: str
    transaction_as_of: datetime
    idempotency_key: str
    rule_version: str = RULES_VERSION

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        _text(self.idempotency_key, "idempotency_key")
        _aware(self.transaction_as_of, "transaction_as_of")
        if self.rule_version != RULES_VERSION:
            raise DurativeClaimError("rules version is unsupported")


@dataclass(frozen=True)
class DurativePropositionRequest:
    """Internal request for planning one exact proposition."""

    user_id: str
    subject_id: str
    predicate: str
    predicate_registry_version: str
    object_json: object
    polarity: str
    transaction_as_of: datetime
    idempotency_key: str
    rules_version: str = RULES_VERSION

    def __post_init__(self) -> None:
        for name in ("user_id", "subject_id", "predicate", "idempotency_key"):
            _text(getattr(self, name), name)
        if self.predicate_registry_version != REGISTRY_VERSION:
            raise DurativeClaimError("predicate registry version is unsupported")
        if self.polarity not in POLARITIES:
            raise DurativeClaimError("polarity is invalid")
        if self.rules_version != RULES_VERSION:
            raise DurativeClaimError("rules version is unsupported")
        _aware(self.transaction_as_of, "transaction_as_of")
        object.__setattr__(self, "object_json", _safe_json(self.object_json, "object_json"))


@dataclass(frozen=True)
class DurativeEpisode:
    episode_id: str
    user_id: str
    claim_id: str
    claim_version_id: str
    session_definition_id: str
    source_id: str
    span_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    predicate_registry_version: str
    object_json: object
    polarity: str
    epistemic_status: str
    lifecycle_status: str
    memory_kind: str | None
    sensitivity: str | None
    extraction_confidence: float
    belief_confidence: float | None
    support_type: str
    time_precision: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    episode_at: date | datetime | None
    transaction_from: datetime
    transaction_to: datetime | None = None
    conflict_labels: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _sha(self.episode_id, "episode_id")
        _sha(self.session_definition_id, "session_definition_id")
        for name in (
            "user_id", "claim_id", "claim_version_id", "source_id", "span_id",
            "subject_id", "speaker_id", "predicate",
        ):
            _text(getattr(self, name), name)
        if self.predicate_registry_version != REGISTRY_VERSION:
            raise DurativeClaimError("episode registry version is unsupported")
        if self.polarity not in POLARITIES:
            raise DurativeClaimError("episode polarity is invalid")
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise DurativeClaimError("episode epistemic status is invalid")
        if self.lifecycle_status not in LIFECYCLE_STATUSES:
            raise DurativeClaimError("episode lifecycle status is invalid")
        if self.memory_kind not in {None, "episodic", "durative"}:
            raise DurativeClaimError("episode memory kind is invalid")
        if self.sensitivity not in {None, *SENSITIVITIES}:
            raise DurativeClaimError("episode sensitivity is invalid")
        if self.support_type not in SUPPORT_TYPES:
            raise DurativeClaimError("episode support type is invalid")
        allowed_conflicts = {
            "hard_contradiction", "temporal_change", "explicit_correction",
            "refinement", "source_disagreement", "retraction",
            "unresolved_ambiguity", "unrelated",
        }
        allowed_relations = {
            "supports", "contradicts", "corrects", "supersedes", "refines",
            "same_event_as", "caused_by", "hindered_by", "same_topic_as",
        }
        if tuple(sorted(set(self.conflict_labels))) != self.conflict_labels or not set(self.conflict_labels) <= allowed_conflicts:
            raise DurativeClaimError("episode conflict labels are invalid")
        if tuple(sorted(set(self.relation_types))) != self.relation_types or not set(self.relation_types) <= allowed_relations:
            raise DurativeClaimError("episode relation types are invalid")
        if self.time_precision not in TIME_PRECISIONS:
            raise DurativeClaimError("episode time precision is invalid")
        object.__setattr__(
            self, "extraction_confidence",
            _confidence(self.extraction_confidence, "extraction_confidence", False),
        )
        object.__setattr__(
            self, "belief_confidence",
            _confidence(self.belief_confidence, "belief_confidence", True),
        )
        _aware(self.transaction_from, "transaction_from")
        if self.transaction_to is not None:
            _aware(self.transaction_to, "transaction_to")
            if self.transaction_to <= self.transaction_from:
                raise DurativeClaimError("episode transaction interval is invalid")
        if self.episode_at is not None:
            if isinstance(self.episode_at, datetime):
                _aware(self.episode_at, "episode_at")
            elif type(self.episode_at) is not date:
                raise DurativeClaimError("episode_at must be a date or aware timestamp")
        _valid_time(self)
        object.__setattr__(self, "object_json", _safe_json(self.object_json, "object_json"))


@dataclass(frozen=True)
class DurativeEvidenceRef:
    episode_id: str
    claim_id: str
    claim_version_id: str
    session_definition_id: str
    source_id: str
    span_id: str
    support_type: str
    episode_at: date | datetime | None

    def __post_init__(self) -> None:
        _sha(self.episode_id, "episode_id")
        _sha(self.session_definition_id, "session_definition_id")
        for name in ("claim_id", "claim_version_id", "source_id", "span_id"):
            _text(getattr(self, name), name)
        if self.support_type not in SUPPORT_TYPES:
            raise DurativeClaimError("evidence support type is invalid")
        if isinstance(self.episode_at, datetime):
            _aware(self.episode_at, "episode_at")
        elif self.episode_at is not None and type(self.episode_at) is not date:
            raise DurativeClaimError("evidence episode time is invalid")


@dataclass(frozen=True)
class DurativeExtractionMetadata:
    extraction_version_id: str
    extractor_kind: str
    rules_version: str
    rules_sha256: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    input_snapshot_sha256: str

    def __post_init__(self) -> None:
        for name in ("extraction_version_id", "rules_sha256", "predicate_registry_sha256", "input_snapshot_sha256"):
            _sha(getattr(self, name), name)
        if (
            self.extractor_kind != "deterministic_rules"
            or self.rules_version != RULES_VERSION
            or self.predicate_registry_version != REGISTRY_VERSION
            or self.predicate_registry_sha256 != REGISTRY_SHA256
        ):
            raise DurativeClaimError("durative extraction metadata changed")


@dataclass(frozen=True)
class DurativeClaimPlan:
    claim_id: str
    semantic_sha256: str
    user_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    predicate_registry_version: str
    object_json: object
    polarity: str
    epistemic_status: str
    lifecycle_status: str
    memory_kind: str
    sensitivity: str
    extraction_confidence: float
    belief_confidence: None
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    time_precision: str
    evidence: tuple[DurativeEvidenceRef, ...]
    extraction: DurativeExtractionMetadata

    def __post_init__(self) -> None:
        _sha(self.claim_id, "claim_id")
        _sha(self.semantic_sha256, "semantic_sha256")
        if self.claim_id != self.semantic_sha256:
            raise DurativeClaimError("durative claim ID is not semantic")
        for name in ("user_id", "subject_id", "predicate"):
            _text(getattr(self, name), name)
        if (
            self.speaker_id, self.predicate_registry_version,
            self.epistemic_status, self.lifecycle_status, self.memory_kind,
        ) != ("memory_system", REGISTRY_VERSION, "inferred", "candidate", "durative"):
            raise DurativeClaimError("derived claim constants changed")
        if self.polarity not in POLARITIES or self.sensitivity not in {"standard", "sensitive"}:
            raise DurativeClaimError("derived claim field is invalid")
        object.__setattr__(
            self, "extraction_confidence",
            _confidence(self.extraction_confidence, "extraction_confidence", False),
        )
        if self.belief_confidence is not None:
            raise DurativeClaimError("derived belief confidence must be null")
        if not self.evidence or len({item.episode_id for item in self.evidence}) != len(self.evidence):
            raise DurativeClaimError("derived evidence must be nonempty and unique")
        _plan_time(self)
        object.__setattr__(self, "object_json", _safe_json(self.object_json, "object_json"))


@dataclass(frozen=True)
class DurativeDecision:
    decision_id: str
    user_id: str
    rules_version: str
    status: str
    reason: str
    input_snapshot_sha256: str
    input_episode_ids: tuple[str, ...]
    support_episode_ids: tuple[str, ...]
    counter_episode_ids: tuple[str, ...]
    ignored_episode_ids: tuple[str, ...]
    claim_id: str | None

    def __post_init__(self) -> None:
        _sha(self.decision_id, "decision_id")
        _sha(self.input_snapshot_sha256, "input_snapshot_sha256")
        _text(self.user_id, "user_id")
        if self.rules_version != RULES_VERSION or self.status not in {"accepted", "rejected"}:
            raise DurativeClaimError("durative decision identity changed")
        allowed = ACCEPT_REASONS if self.status == "accepted" else REJECT_REASONS
        if self.reason not in allowed:
            raise DurativeClaimError("durative decision reason is invalid")
        groups = (self.support_episode_ids, self.counter_episode_ids, self.ignored_episode_ids)
        if any(tuple(sorted(set(group))) != group for group in groups):
            raise DurativeClaimError("decision episode groups must be sorted and unique")
        combined = tuple(sorted(item for group in groups for item in group))
        if tuple(sorted(set(self.input_episode_ids))) != self.input_episode_ids or combined != self.input_episode_ids:
            raise DurativeClaimError("decision does not account for every episode")
        if self.status == "accepted":
            _sha(self.claim_id, "claim_id")
        elif self.claim_id is not None:
            raise DurativeClaimError("rejected decision cannot name a claim")


@dataclass(frozen=True)
class DurativeInferenceResult:
    run_id: str
    user_id: str
    rule_version: str
    input_snapshot_sha256: str
    decisions: tuple[DurativeDecision, ...]
    created_count: int
    replayed_count: int

    def __post_init__(self) -> None:
        _sha(self.run_id, "run_id")
        _sha(self.input_snapshot_sha256, "input_snapshot_sha256")
        _text(self.user_id, "user_id")
        if self.rule_version != RULES_VERSION:
            raise DurativeClaimError("rules version is unsupported")
        if tuple(sorted(self.decisions, key=lambda item: item.decision_id)) != self.decisions:
            raise DurativeClaimError("durative decisions must be ordered")
        if len({item.decision_id for item in self.decisions}) != len(self.decisions):
            raise DurativeClaimError("durative decisions must be unique")
        if any(item.user_id != self.user_id for item in self.decisions):
            raise DurativeClaimError("durative result leaves its user boundary")
        for name in ("created_count", "replayed_count"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= len(self.decisions):
                raise DurativeClaimError(f"{name} is invalid")
        if self.created_count and self.replayed_count:
            raise DurativeClaimError("a durative run cannot be both created and replayed")
        if self.created_count + self.replayed_count != len(self.decisions):
            raise DurativeClaimError("durative result accounting is incomplete")

    @property
    def proposition_count(self) -> int:
        return len(self.decisions)

    @property
    def accepted_count(self) -> int:
        return sum(item.status == "accepted" for item in self.decisions)

    @property
    def rejected_count(self) -> int:
        return sum(item.status == "rejected" for item in self.decisions)


@dataclass(frozen=True)
class DurativePropositionPlan:
    """Internal, side-effect-free plan for one exact proposition."""

    plan_id: str
    input_snapshot_sha256: str
    request: DurativePropositionRequest
    decision: DurativeDecision
    claim: DurativeClaimPlan | None

    def __post_init__(self) -> None:
        _sha(self.plan_id, "plan_id")
        _sha(self.input_snapshot_sha256, "input_snapshot_sha256")
        if (
            self.request.user_id != self.decision.user_id
            or self.decision.input_snapshot_sha256 != self.input_snapshot_sha256
            or (self.claim is None) != (self.decision.status == "rejected")
        ):
            raise DurativeClaimError("durative plan leaves its request boundary")
        if self.claim is not None and self.claim.claim_id != self.decision.claim_id:
            raise DurativeClaimError("durative decision and claim differ")


def load_durative_rules_config(path: str | Path) -> DurativeRulesConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DurativeClaimError("durative rules configuration is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise DurativeClaimError("durative rules configuration fields changed")
    for name in (
        "allowed_predicate_families", "eligible_lifecycle_statuses",
        "eligible_epistemic_statuses", "counterevidence_rules", "rejection_precedence",
    ):
        value[name] = tuple(value[name])
    return DurativeRulesConfig(**value)


def canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )


def stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(canonical_json([namespace, *values]).encode("utf-8")).hexdigest()


def _safe_json(value: object, name: str) -> object:
    try:
        return safe_json(value, name)
    except StorageValidationError as error:
        raise DurativeClaimError(f"{name} is not JSON-safe") from error


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        _aware(value, "canonical timestamp")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DurativeClaimError("canonical JSON keys must be strings")
            result[key] = _json_value(item)
        return result
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise DurativeClaimError("value is not canonical JSON")


def _valid_time(value: DurativeEpisode) -> None:
    dates = (value.valid_from_date, value.valid_to_date)
    timestamps = (value.valid_from_timestamp, value.valid_to_timestamp)
    if any(item is not None and type(item) is not date for item in dates):
        raise DurativeClaimError("valid date boundary is invalid")
    for item in timestamps:
        if item is not None:
            _aware(item, "valid timestamp boundary")
    if any(item is not None for item in dates) and any(item is not None for item in timestamps):
        raise DurativeClaimError("valid-time representations cannot mix")
    if value.time_precision == "unknown":
        if any(item is not None for item in (*dates, *timestamps)):
            raise DurativeClaimError("unknown valid time must be empty")
    elif value.time_precision == "timestamp":
        if not any(item is not None for item in timestamps) or any(item is not None for item in dates):
            raise DurativeClaimError("timestamp precision requires timestamps")
    elif value.time_precision == "approximate":
        if not any(item is not None for item in (*dates, *timestamps)):
            raise DurativeClaimError("approximate valid time requires a boundary")
    elif not any(item is not None for item in dates) or any(item is not None for item in timestamps):
        raise DurativeClaimError("date precision requires dates")
    if dates[0] is not None and dates[1] is not None and dates[1] < dates[0]:
        raise DurativeClaimError("valid date interval is reversed")
    if timestamps[0] is not None and timestamps[1] is not None and timestamps[1] < timestamps[0]:
        raise DurativeClaimError("valid timestamp interval is reversed")


def _plan_time(value: DurativeClaimPlan) -> None:
    dates = (value.valid_from_date, value.valid_to_date)
    timestamps = (value.valid_from_timestamp, value.valid_to_timestamp)
    if value.time_precision == "unknown":
        if any(item is not None for item in (*dates, *timestamps)):
            raise DurativeClaimError("unknown derived time must be empty")
    elif value.time_precision == "timestamp":
        if any(item is not None for item in dates) or any(item is None for item in timestamps):
            raise DurativeClaimError("timestamp derived time is incomplete")
        _aware(timestamps[0], "derived valid_from")
        _aware(timestamps[1], "derived valid_to")
    elif value.time_precision in {"day", "month", "year"}:
        if any(item is not None for item in timestamps) or any(item is None for item in dates):
            raise DurativeClaimError("date derived time is incomplete")
    elif value.time_precision == "approximate":
        date_complete = all(item is not None for item in dates) and all(item is None for item in timestamps)
        timestamp_complete = all(item is not None for item in timestamps) and all(item is None for item in dates)
        if not (date_complete or timestamp_complete):
            raise DurativeClaimError("approximate derived time is incomplete")
        if timestamp_complete:
            _aware(timestamps[0], "derived valid_from")
            _aware(timestamps[1], "derived valid_to")
    else:
        raise DurativeClaimError("derived time precision is invalid")


def _confidence(value: object, name: str, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DurativeClaimError(f"{name} is invalid")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized > 1:
        raise DurativeClaimError(f"{name} is invalid")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DurativeClaimError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise DurativeClaimError(f"{name} must be timezone-aware")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DurativeClaimError(f"{name} must be a lowercase SHA-256")
    return value
