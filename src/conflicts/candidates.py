"""User-scoped deterministic candidate linking for Step 5.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import unicodedata

from extraction.predicate_registry import PredicateRegistry, load_predicate_registry
from storage.contracts import ClaimVersionRecord
from temporal.contracts import TemporalClaim, TemporalQuery
from temporal.service import TemporalService


DEFAULT_CONFIG_PATH = Path("configs/conflicts/candidate_linker_v1.json")
ELIGIBLE_STATUSES = (
    "candidate",
    "confirmed",
    "current",
    "historical",
    "disputed",
    "superseded",
)
TEMPORAL_RELATIONS = frozenset({"overlap", "within_gap", "distant", "unknown"})
CONFIG_FIELDS = frozenset(
    {
        "linker_version",
        "rule_version",
        "predicate_registry_path",
        "predicate_registry_version",
        "predicate_registry_sha256",
        "eligible_lifecycle_statuses",
        "maximum_temporal_gap_days",
        "lexical_jaccard_threshold",
        "unicode_normalization",
        "case_normalization",
        "tokenization",
        "review_status",
    }
)


class ConflictCandidateError(ValueError):
    """Reject invalid or unsafe candidate-linking inputs."""


@dataclass(frozen=True)
class CandidateConfig:
    linker_version: str
    rule_version: str
    predicate_registry_path: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    eligible_lifecycle_statuses: tuple[str, ...]
    maximum_temporal_gap_days: int
    lexical_jaccard_threshold: float
    unicode_normalization: str
    case_normalization: str
    tokenization: str
    review_status: str

    def __post_init__(self) -> None:
        for name in (
            "linker_version",
            "rule_version",
            "predicate_registry_path",
            "predicate_registry_version",
        ):
            _text(getattr(self, name), name)
        if (
            self.linker_version,
            self.rule_version,
            self.predicate_registry_path,
            self.predicate_registry_version,
        ) != (
            "candidate_linker_v1",
            "candidate_rules_v1",
            "configs/extraction/predicate_registry_v2.json",
            "predicate_registry_v2",
        ):
            raise ConflictCandidateError("candidate linker identity changed")
        if not _sha256(self.predicate_registry_sha256):
            raise ConflictCandidateError("predicate_registry_sha256 is invalid")
        if self.eligible_lifecycle_statuses != ELIGIBLE_STATUSES:
            raise ConflictCandidateError("eligible lifecycle statuses changed")
        if self.maximum_temporal_gap_days != 90:
            raise ConflictCandidateError("maximum temporal gap changed")
        if self.lexical_jaccard_threshold != 0.5:
            raise ConflictCandidateError("lexical Jaccard threshold changed")
        if (
            self.unicode_normalization,
            self.case_normalization,
            self.tokenization,
            self.review_status,
        ) != ("NFKC", "casefold", "alphanumeric", "implementation_reviewed"):
            raise ConflictCandidateError("candidate linker normalization contract changed")


@dataclass(frozen=True)
class CandidateRequest:
    user_id: str
    transaction_as_of: datetime
    incoming_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.user_id, "user_id")
        if not isinstance(self.transaction_as_of, datetime) or self.transaction_as_of.utcoffset() is None:
            raise ConflictCandidateError("transaction_as_of must be timezone-aware")
        if (
            not isinstance(self.incoming_claim_ids, tuple)
            or not self.incoming_claim_ids
            or any(not isinstance(item, str) or not item.strip() for item in self.incoming_claim_ids)
        ):
            raise ConflictCandidateError("incoming_claim_ids must be a non-empty tuple of IDs")
        if len(self.incoming_claim_ids) != len(set(self.incoming_claim_ids)):
            raise ConflictCandidateError("incoming_claim_ids must be unique")


@dataclass(frozen=True)
class CandidateSignals:
    same_subject: bool
    same_predicate_family: bool
    shared_entities: tuple[str, ...]
    temporal_relation: str
    temporal_gap_days: float | None
    approximate_time: bool
    lexical_jaccard: float

    def __post_init__(self) -> None:
        if not isinstance(self.same_subject, bool) or not isinstance(self.same_predicate_family, bool):
            raise ConflictCandidateError("candidate identity signals must be booleans")
        if (
            not isinstance(self.shared_entities, tuple)
            or tuple(sorted(set(self.shared_entities))) != self.shared_entities
        ):
            raise ConflictCandidateError("shared_entities must be sorted and unique")
        if self.temporal_relation not in TEMPORAL_RELATIONS:
            raise ConflictCandidateError("temporal_relation is invalid")
        if self.temporal_relation == "unknown":
            if self.temporal_gap_days is not None:
                raise ConflictCandidateError("unknown temporal relation cannot have a gap")
        elif (
            isinstance(self.temporal_gap_days, bool)
            or not isinstance(self.temporal_gap_days, (int, float))
            or not math.isfinite(float(self.temporal_gap_days))
            or self.temporal_gap_days < 0
        ):
            raise ConflictCandidateError("known temporal relation requires a finite gap")
        if not isinstance(self.approximate_time, bool):
            raise ConflictCandidateError("approximate_time must be a boolean")
        if (
            isinstance(self.lexical_jaccard, bool)
            or not isinstance(self.lexical_jaccard, (int, float))
            or not math.isfinite(float(self.lexical_jaccard))
            or not 0 <= self.lexical_jaccard <= 1
        ):
            raise ConflictCandidateError("lexical_jaccard must be between zero and one")


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    linker_version: str
    user_id: str
    left_claim_id: str
    right_claim_id: str
    signals: CandidateSignals
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("linker_version", "user_id", "left_claim_id", "right_claim_id"):
            _text(getattr(self, name), name)
        if self.left_claim_id >= self.right_claim_id:
            raise ConflictCandidateError("candidate claim IDs must be lexically ordered")
        if self.pair_id != _pair_id(
            self.linker_version, self.user_id, self.left_claim_id, self.right_claim_id
        ):
            raise ConflictCandidateError("candidate pair_id is not canonical")
        if not isinstance(self.signals, CandidateSignals):
            raise ConflictCandidateError("signals must be CandidateSignals")
        if (
            not isinstance(self.source_ids, tuple)
            or not self.source_ids
            or tuple(sorted(set(self.source_ids))) != self.source_ids
        ):
            raise ConflictCandidateError("source_ids must be sorted unique visible IDs")


def load_candidate_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> CandidateConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConflictCandidateError("candidate config is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise ConflictCandidateError("candidate config fields changed")
    statuses = value["eligible_lifecycle_statuses"]
    if not isinstance(statuses, list):
        raise ConflictCandidateError("eligible_lifecycle_statuses must be a list")
    return CandidateConfig(
        linker_version=value["linker_version"],
        rule_version=value["rule_version"],
        predicate_registry_path=value["predicate_registry_path"],
        predicate_registry_version=value["predicate_registry_version"],
        predicate_registry_sha256=value["predicate_registry_sha256"],
        eligible_lifecycle_statuses=tuple(statuses),
        maximum_temporal_gap_days=value["maximum_temporal_gap_days"],
        lexical_jaccard_threshold=value["lexical_jaccard_threshold"],
        unicode_normalization=value["unicode_normalization"],
        case_normalization=value["case_normalization"],
        tokenization=value["tokenization"],
        review_status=value["review_status"],
    )


def generate_candidate_pairs(
    config: CandidateConfig,
    request: CandidateRequest,
    claims: Sequence[TemporalClaim],
    registry: PredicateRegistry,
) -> tuple[CandidatePair, ...]:
    """Generate every qualifying visible pair involving an incoming claim."""

    if (
        registry.registry_version != config.predicate_registry_version
        or registry.content_sha256 != config.predicate_registry_sha256
    ):
        raise ConflictCandidateError("predicate registry binding changed")
    if any(
        item.claim.user_id != request.user_id
        or item.version.user_id != request.user_id
        for item in claims
    ):
        raise ConflictCandidateError("cross-user candidate input")
    eligible = frozenset(config.eligible_lifecycle_statuses)
    visible = tuple(
        item
        for item in claims
        if item.version.lifecycle_status in eligible
        and item.version.transaction_contains(request.transaction_as_of)
        and bool(item.evidence)
    )
    by_id: dict[str, TemporalClaim] = {}
    for item in visible:
        if item.claim.claim_id != item.version.claim_id:
            raise ConflictCandidateError("claim and version identity differ")
        if item.claim.claim_id in by_id:
            raise ConflictCandidateError("visible claim IDs must be unique")
        if item.claim.predicate_registry_version != registry.registry_version:
            raise ConflictCandidateError("claim predicate registry version changed")
        if item.claim.predicate not in registry.by_predicate:
            raise ConflictCandidateError("visible claim predicate is unknown")
        by_id[item.claim.claim_id] = item
    missing = set(request.incoming_claim_ids) - set(by_id)
    if missing:
        raise ConflictCandidateError("incoming claim is not visible and supported")

    pairs: dict[tuple[str, str], CandidatePair] = {}
    for incoming_id in request.incoming_claim_ids:
        for other_id in sorted(by_id):
            if incoming_id == other_id:
                continue
            left_id, right_id = sorted((incoming_id, other_id))
            key = (left_id, right_id)
            if key in pairs:
                continue
            left, right = by_id[left_id], by_id[right_id]
            signals = _candidate_signals(config, left, right, registry)
            if not _qualifies(config, signals):
                continue
            source_ids = tuple(
                sorted(
                    {
                        evidence.source_id
                        for item in (left, right)
                        for evidence in item.evidence
                    }
                )
            )
            pairs[key] = CandidatePair(
                pair_id=_pair_id(config.linker_version, request.user_id, left_id, right_id),
                linker_version=config.linker_version,
                user_id=request.user_id,
                left_claim_id=left_id,
                right_claim_id=right_id,
                signals=signals,
                source_ids=source_ids,
            )
    return tuple(pairs[key] for key in sorted(pairs))


class ConflictCandidateService:
    def __init__(
        self,
        connection: object,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        repo_root: str | Path = ".",
    ) -> None:
        self._root = Path(repo_root).resolve()
        self.config = load_candidate_config(config_path, repo_root=self._root)
        registry_path = (self._root / self.config.predicate_registry_path).resolve()
        try:
            registry_path.relative_to(self._root)
        except ValueError:
            raise ConflictCandidateError("predicate registry path leaves repository") from None
        self._registry = load_predicate_registry(registry_path)
        if (
            self._registry.registry_version != self.config.predicate_registry_version
            or self._registry.content_sha256 != self.config.predicate_registry_sha256
        ):
            raise ConflictCandidateError("predicate registry binding changed")
        self._temporal = TemporalService(connection)

    def generate(self, request: CandidateRequest) -> tuple[CandidatePair, ...]:
        visible = self._temporal.query(
            TemporalQuery(
                user_id=request.user_id,
                transaction_as_of=request.transaction_as_of,
                statuses=frozenset(self.config.eligible_lifecycle_statuses),
            )
        )
        return generate_candidate_pairs(self.config, request, visible, self._registry)


def _candidate_signals(
    config: CandidateConfig,
    left: TemporalClaim,
    right: TemporalClaim,
    registry: PredicateRegistry,
) -> CandidateSignals:
    left_family = registry.by_predicate[left.claim.predicate].family
    right_family = registry.by_predicate[right.claim.predicate].family
    relation, gap = _temporal_signal(
        left.version, right.version, config.maximum_temporal_gap_days
    )
    left_entities = _entities(left.claim.subject_id, left.claim.object_json)
    right_entities = _entities(right.claim.subject_id, right.claim.object_json)
    return CandidateSignals(
        same_subject=left.claim.subject_id == right.claim.subject_id,
        same_predicate_family=left_family == right_family,
        shared_entities=tuple(sorted(left_entities & right_entities)),
        temporal_relation=relation,
        temporal_gap_days=gap,
        approximate_time=(
            left.version.time_precision == "approximate"
            or right.version.time_precision == "approximate"
        ),
        lexical_jaccard=_jaccard(
            _lexical_tokens(left.claim.predicate, left_family, left.claim.object_json),
            _lexical_tokens(right.claim.predicate, right_family, right.claim.object_json),
        ),
    )


def _qualifies(config: CandidateConfig, signals: CandidateSignals) -> bool:
    shared_entity = bool(signals.shared_entities)
    temporally_close = signals.temporal_relation in {"overlap", "within_gap"}
    return (
        (signals.same_subject and signals.same_predicate_family)
        or (shared_entity and temporally_close)
        or (
            signals.lexical_jaccard >= config.lexical_jaccard_threshold
            and (signals.same_subject or signals.same_predicate_family or shared_entity)
        )
    )


def _temporal_signal(
    left: ClaimVersionRecord,
    right: ClaimVersionRecord,
    maximum_gap_days: int,
) -> tuple[str, float | None]:
    left_interval = _finite_interval(left)
    right_interval = _finite_interval(right)
    if left_interval is None or right_interval is None or left_interval[0] != right_interval[0]:
        return "unknown", None
    _, left_start, left_end = left_interval
    _, right_start, right_end = right_interval
    if left_start <= right_end and right_start <= left_end:
        return "overlap", 0.0
    if left_end < right_start:
        difference = right_start - left_end
    else:
        difference = left_start - right_end
    gap = difference.days if type(left_start) is date else difference.total_seconds() / 86400
    gap = round(float(gap), 6)
    return ("within_gap" if gap <= maximum_gap_days else "distant"), gap


def _finite_interval(
    version: ClaimVersionRecord,
) -> tuple[str, date | datetime, date | datetime] | None:
    if version.time_precision == "unknown":
        return None
    if version.time_precision == "timestamp":
        if version.valid_from_timestamp is None or version.valid_to_timestamp is None:
            return None
        return "timestamp", version.valid_from_timestamp, version.valid_to_timestamp
    if version.valid_from_date is None or version.valid_to_date is None:
        return None
    return "date", version.valid_from_date, version.valid_to_date


def _entities(subject_id: str, value: object) -> frozenset[str]:
    entities = {_normalize(subject_id)}
    entities.update(
        normalized
        for leaf in _string_leaves(value)
        if (normalized := _normalize(leaf))
    )
    entities.discard("")
    return frozenset(entities)


def _string_leaves(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(leaf for item in value for leaf in _string_leaves(item))
    if isinstance(value, Mapping):
        return tuple(
            leaf
            for key in sorted(value)
            for leaf in _string_leaves(value[key])
        )
    return ()


def _lexical_tokens(predicate: str, family: str, value: object) -> frozenset[str]:
    canonical_object = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return frozenset(_normalize(f"{predicate} {family} {canonical_object}").split())


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else round(len(left & right) / len(union), 6)


def _pair_id(linker_version: str, user_id: str, left_id: str, right_id: str) -> str:
    payload = json.dumps(
        [linker_version, user_id, left_id, right_id],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConflictCandidateError(f"{name} must be non-empty text")
    return value


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
