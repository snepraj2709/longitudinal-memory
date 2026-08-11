"""Deterministic Step 5.2 conflict classification and checked relations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from extraction.predicate_registry import (
    PredicateRegistry,
    load_predicate_registry,
    validate_object_shape,
)
from temporal.contracts import TemporalClaim

from .candidates import CandidatePair, ELIGIBLE_STATUSES


DEFAULT_CLASSIFIER_CONFIG_PATH = Path(
    "configs/conflicts/relation_classifier_v1.json"
)
CLASSIFIER_VERSION = "relation_classifier_v1"
RULE_VERSION = "relation_rules_v1"
CONFLICT_LABELS = (
    "hard_contradiction",
    "temporal_change",
    "explicit_correction",
    "refinement",
    "source_disagreement",
    "retraction",
    "unresolved_ambiguity",
    "unrelated",
)
PRECEDENCE = (
    "retraction",
    "explicit_correction",
    "temporal_change",
    "source_disagreement",
    "hard_contradiction",
    "refinement",
    "unresolved_ambiguity",
    "unrelated",
)
ONTOLOGY_RELATIONS = (
    "supports",
    "contradicts",
    "corrects",
    "supersedes",
    "refines",
    "same_event_as",
    "caused_by",
    "hindered_by",
    "same_topic_as",
)
EMITTED_RELATIONS = (
    "contradicts",
    "corrects",
    "refines",
    "same_topic_as",
)
SYMMETRIC_RELATIONS = ("contradicts", "same_topic_as")
DIRECTED_RELATIONS = ("corrects", "refines")
CONFIG_FIELDS = frozenset(
    {
        "classifier_version",
        "rule_version",
        "predicate_registry_path",
        "predicate_registry_version",
        "predicate_registry_sha256",
        "eligible_lifecycle_statuses",
        "labels",
        "precedence",
        "accepted_relation_vocabulary",
        "emitted_relation_types",
        "symmetric_relation_types",
        "directed_relation_types",
        "matched_rule_confidence",
        "review_status",
    }
)
LABEL_RELATION = {
    "hard_contradiction": "contradicts",
    "temporal_change": "same_topic_as",
    "explicit_correction": "corrects",
    "refinement": "refines",
    "source_disagreement": "contradicts",
    "retraction": "corrects",
    "unresolved_ambiguity": None,
    "unrelated": None,
}


class ConflictClassifierError(ValueError):
    """Reject invalid, invisible, or ambiguous classifier inputs."""


@dataclass(frozen=True)
class ClassifierConfig:
    classifier_version: str
    rule_version: str
    predicate_registry_path: str
    predicate_registry_version: str
    predicate_registry_sha256: str
    eligible_lifecycle_statuses: tuple[str, ...]
    labels: tuple[str, ...]
    precedence: tuple[str, ...]
    accepted_relation_vocabulary: tuple[str, ...]
    emitted_relation_types: tuple[str, ...]
    symmetric_relation_types: tuple[str, ...]
    directed_relation_types: tuple[str, ...]
    matched_rule_confidence: float
    review_status: str

    def __post_init__(self) -> None:
        if (
            self.classifier_version,
            self.rule_version,
            self.predicate_registry_path,
            self.predicate_registry_version,
        ) != (
            CLASSIFIER_VERSION,
            RULE_VERSION,
            "configs/extraction/predicate_registry_v2.json",
            "predicate_registry_v2",
        ):
            raise ConflictClassifierError("classifier identity changed")
        if not _sha256_text(self.predicate_registry_sha256):
            raise ConflictClassifierError("predicate registry hash is invalid")
        if self.eligible_lifecycle_statuses != ELIGIBLE_STATUSES:
            raise ConflictClassifierError("eligible lifecycle statuses changed")
        if (
            self.labels != CONFLICT_LABELS
            or self.precedence != PRECEDENCE
            or self.accepted_relation_vocabulary != ONTOLOGY_RELATIONS
            or self.emitted_relation_types != EMITTED_RELATIONS
            or self.symmetric_relation_types != SYMMETRIC_RELATIONS
            or self.directed_relation_types != DIRECTED_RELATIONS
        ):
            raise ConflictClassifierError("classifier vocabulary changed")
        if self.matched_rule_confidence != 1.0:
            raise ConflictClassifierError("matched rule confidence changed")
        if self.review_status != "implementation_reviewed":
            raise ConflictClassifierError("classifier config is not reviewed")


@dataclass(frozen=True)
class ClassificationRequest:
    classifier_version: str
    user_id: str
    transaction_as_of: datetime
    pair: CandidatePair
    left: TemporalClaim
    right: TemporalClaim
    source_ingested_at: tuple[tuple[str, datetime], ...]
    explicit_target_claim_id: str | None = None

    def __post_init__(self) -> None:
        if self.classifier_version != CLASSIFIER_VERSION:
            raise ConflictClassifierError("request classifier version changed")
        _text(self.user_id, "user_id")
        _aware(self.transaction_as_of, "transaction_as_of")
        if not isinstance(self.pair, CandidatePair):
            raise ConflictClassifierError("pair must be a canonical CandidatePair")
        if not isinstance(self.left, TemporalClaim) or not isinstance(
            self.right, TemporalClaim
        ):
            raise ConflictClassifierError("left and right must be TemporalClaim snapshots")
        if (
            self.left.claim.claim_id != self.pair.left_claim_id
            or self.right.claim.claim_id != self.pair.right_claim_id
            or self.left.claim.claim_id >= self.right.claim.claim_id
        ):
            raise ConflictClassifierError("claim snapshots must follow canonical pair order")
        if not isinstance(self.source_ingested_at, tuple) or not self.source_ingested_at:
            raise ConflictClassifierError("source ingestion snapshots are required")
        source_ids: list[str] = []
        for item in self.source_ingested_at:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ConflictClassifierError("source ingestion snapshot is invalid")
            source_id, ingested_at = item
            source_ids.append(_text(source_id, "source_id"))
            _aware(ingested_at, "source ingested_at")
        if source_ids != sorted(set(source_ids)):
            raise ConflictClassifierError("source ingestion snapshots must be sorted and unique")
        if self.explicit_target_claim_id is not None:
            _text(self.explicit_target_claim_id, "explicit_target_claim_id")
            if self.explicit_target_claim_id not in {
                self.pair.left_claim_id,
                self.pair.right_claim_id,
            }:
                raise ConflictClassifierError("explicit target must name one paired claim")


@dataclass(frozen=True)
class CheckedRelation:
    relation_id: str
    classifier_version: str
    user_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    confidence: float
    input_snapshot_sha256: str

    def __post_init__(self) -> None:
        if self.classifier_version != CLASSIFIER_VERSION:
            raise ConflictClassifierError("relation classifier version changed")
        for name in ("user_id", "source_claim_id", "target_claim_id"):
            _text(getattr(self, name), name)
        if self.source_claim_id == self.target_claim_id:
            raise ConflictClassifierError("relation claims must differ")
        if self.relation_type not in EMITTED_RELATIONS:
            raise ConflictClassifierError("relation type is not emitted by v1")
        if self.relation_type in SYMMETRIC_RELATIONS and (
            self.source_claim_id >= self.target_claim_id
        ):
            raise ConflictClassifierError("symmetric relation claims must be canonical")
        if self.confidence != 1.0:
            raise ConflictClassifierError("checked relation confidence must be one")
        if not _sha256_text(self.input_snapshot_sha256):
            raise ConflictClassifierError("relation snapshot hash is invalid")
        expected = _stable_id(
            "checked_relation",
            self.classifier_version,
            self.user_id,
            self.source_claim_id,
            self.target_claim_id,
            self.relation_type,
            self.input_snapshot_sha256,
        )
        if self.relation_id != expected:
            raise ConflictClassifierError("relation ID is not canonical")


@dataclass(frozen=True)
class ConflictDecision:
    decision_id: str
    classifier_version: str
    user_id: str
    pair_id: str
    label: str
    confidence: float
    input_snapshot_sha256: str
    relations: tuple[CheckedRelation, ...]

    def __post_init__(self) -> None:
        if self.classifier_version != CLASSIFIER_VERSION:
            raise ConflictClassifierError("decision classifier version changed")
        _text(self.user_id, "user_id")
        _text(self.pair_id, "pair_id")
        if self.label not in CONFLICT_LABELS:
            raise ConflictClassifierError("conflict label is invalid")
        if self.confidence != 1.0:
            raise ConflictClassifierError("decision confidence must be one")
        if not _sha256_text(self.input_snapshot_sha256):
            raise ConflictClassifierError("decision snapshot hash is invalid")
        if not isinstance(self.relations, tuple):
            raise ConflictClassifierError("relations must be a tuple")
        expected_type = LABEL_RELATION[self.label]
        if (
            (expected_type is None and self.relations)
            or (expected_type is not None and len(self.relations) != 1)
            or any(relation.relation_type != expected_type for relation in self.relations)
        ):
            raise ConflictClassifierError("decision relation does not match its label")
        expected = _stable_id(
            "conflict_decision",
            self.classifier_version,
            self.user_id,
            self.pair_id,
            self.label,
            self.input_snapshot_sha256,
        )
        if self.decision_id != expected:
            raise ConflictClassifierError("decision ID is not canonical")


def load_classifier_config(
    path: str | Path = DEFAULT_CLASSIFIER_CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> ClassifierConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConflictClassifierError("classifier config is unreadable") from error
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise ConflictClassifierError("classifier config fields changed")
    tuple_fields = {
        "eligible_lifecycle_statuses",
        "labels",
        "precedence",
        "accepted_relation_vocabulary",
        "emitted_relation_types",
        "symmetric_relation_types",
        "directed_relation_types",
    }
    if any(not isinstance(value[name], list) for name in tuple_fields):
        raise ConflictClassifierError("classifier config vocabularies must be lists")
    return ClassifierConfig(
        classifier_version=value["classifier_version"],
        rule_version=value["rule_version"],
        predicate_registry_path=value["predicate_registry_path"],
        predicate_registry_version=value["predicate_registry_version"],
        predicate_registry_sha256=value["predicate_registry_sha256"],
        eligible_lifecycle_statuses=tuple(value["eligible_lifecycle_statuses"]),
        labels=tuple(value["labels"]),
        precedence=tuple(value["precedence"]),
        accepted_relation_vocabulary=tuple(value["accepted_relation_vocabulary"]),
        emitted_relation_types=tuple(value["emitted_relation_types"]),
        symmetric_relation_types=tuple(value["symmetric_relation_types"]),
        directed_relation_types=tuple(value["directed_relation_types"]),
        matched_rule_confidence=value["matched_rule_confidence"],
        review_status=value["review_status"],
    )


class ConflictClassifier:
    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CLASSIFIER_CONFIG_PATH,
        repo_root: str | Path = ".",
    ) -> None:
        root = Path(repo_root).resolve()
        self.config = load_classifier_config(config_path, repo_root=root)
        registry_path = (root / self.config.predicate_registry_path).resolve()
        try:
            registry_path.relative_to(root)
        except ValueError:
            raise ConflictClassifierError("predicate registry path leaves repository") from None
        self.registry = load_predicate_registry(registry_path)
        if (
            self.registry.registry_version != self.config.predicate_registry_version
            or self.registry.content_sha256 != self.config.predicate_registry_sha256
        ):
            raise ConflictClassifierError("predicate registry binding changed")

    def classify(self, request: ClassificationRequest) -> ConflictDecision:
        return classify_conflict(self.config, request, self.registry)


def classify_conflict(
    config: ClassifierConfig,
    request: ClassificationRequest,
    registry: PredicateRegistry,
) -> ConflictDecision:
    _validate_request(config, request, registry)
    snapshot = _input_snapshot(request)
    left, right = request.left, request.right
    left_value = _semantic_value(left)
    right_value = _semantic_value(right)
    values_changed = left_value != right_value

    targeted = _targeted_actor(request)
    if targeted is not None:
        actor, target = targeted
        if actor.claim.epistemic_status == "denied":
            return _decision(request, snapshot, "retraction", actor, target)
        if actor.claim.epistemic_status == "corrected" and values_changed:
            return _decision(request, snapshot, "explicit_correction", actor, target)
        return _decision(request, snapshot, "unresolved_ambiguity")
    if any(
        item.claim.epistemic_status in {"denied", "corrected"}
        for item in (left, right)
    ):
        return _decision(request, snapshot, "unresolved_ambiguity")
    if _prose_only_targeting(left, right):
        return _decision(request, snapshot, "unresolved_ambiguity")

    same_predicate = left.claim.predicate == right.claim.predicate
    left_definition = registry.by_predicate[left.claim.predicate]
    right_definition = registry.by_predicate[right.claim.predicate]
    time_relation = _time_relation(left, right)
    single_value = (
        same_predicate
        and left_definition.conflict_compatibility == "single_value"
        and right_definition.conflict_compatibility == "single_value"
    )
    if single_value and values_changed and time_relation == "nonoverlap":
        return _decision(request, snapshot, "temporal_change", left, right)
    if single_value and values_changed and time_relation == "overlap":
        label = (
            "hard_contradiction"
            if left.claim.speaker_id == right.claim.speaker_id
            else "source_disagreement"
        )
        return _decision(request, snapshot, label, left, right)

    refinement = _refinement(left, right) if same_predicate else None
    if refinement is not None:
        specific, broad = refinement
        return _decision(request, snapshot, "refinement", specific, broad)

    if (
        same_predicate
        and values_changed
        and (
            time_relation in {"unknown", "mixed", "approximate"}
            or any(
                item.claim.epistemic_status in {"uncertain", "hypothetical", "inferred"}
                for item in (left, right)
            )
        )
    ):
        return _decision(request, snapshot, "unresolved_ambiguity")
    return _decision(request, snapshot, "unrelated")


def _validate_request(
    config: ClassifierConfig,
    request: ClassificationRequest,
    registry: PredicateRegistry,
) -> None:
    if request.classifier_version != config.classifier_version:
        raise ConflictClassifierError("request and config versions differ")
    if (
        registry.registry_version != config.predicate_registry_version
        or registry.content_sha256 != config.predicate_registry_sha256
    ):
        raise ConflictClassifierError("predicate registry binding changed")
    pair = request.pair
    left, right = request.left, request.right
    if pair.linker_version != "candidate_linker_v1":
        raise ConflictClassifierError("candidate linker version changed")
    if pair.user_id != request.user_id:
        raise ConflictClassifierError("pair user differs from request user")
    for side, item in (("left", left), ("right", right)):
        claim, version = item.claim, item.version
        if (
            claim.user_id != request.user_id
            or version.user_id != request.user_id
            or version.claim_id != claim.claim_id
        ):
            raise ConflictClassifierError(f"{side} claim ownership changed")
        if version.lifecycle_status not in config.eligible_lifecycle_statuses:
            raise ConflictClassifierError(f"{side} claim is not classifier-visible")
        if not version.transaction_contains(request.transaction_as_of):
            raise ConflictClassifierError(f"{side} version is not visible as of cutoff")
        if claim.predicate_registry_version != registry.registry_version:
            raise ConflictClassifierError(f"{side} claim registry version changed")
        if claim.predicate not in registry.by_predicate:
            raise ConflictClassifierError(f"{side} predicate is unknown")
        definition = registry.by_predicate[claim.predicate]
        if validate_object_shape(claim.object_json, definition.object_shape):
            raise ConflictClassifierError(f"{side} claim object shape is invalid")
        if not item.evidence:
            raise ConflictClassifierError(f"{side} claim has no visible evidence")
        evidence_keys: list[tuple[str, str]] = []
        for evidence in item.evidence:
            for name in ("source_id", "span_id", "speaker_id", "quote"):
                _text(getattr(evidence, name), f"{side} evidence {name}")
            if evidence.speaker_id != claim.speaker_id:
                raise ConflictClassifierError(f"{side} evidence speaker differs from claim")
            evidence_keys.append((evidence.source_id, evidence.span_id))
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ConflictClassifierError(f"{side} evidence repeats")
    evidence_sources = tuple(
        sorted(
            {
                evidence.source_id
                for item in (left, right)
                for evidence in item.evidence
            }
        )
    )
    if evidence_sources != pair.source_ids:
        raise ConflictClassifierError("pair source provenance changed")
    ingested = dict(request.source_ingested_at)
    if tuple(ingested) != pair.source_ids:
        raise ConflictClassifierError("source ingestion snapshot does not match pair")
    if any(value > request.transaction_as_of for value in ingested.values()):
        raise ConflictClassifierError("future evidence is not classifier-visible")


def _targeted_actor(
    request: ClassificationRequest,
) -> tuple[TemporalClaim, TemporalClaim] | None:
    target_id = request.explicit_target_claim_id
    if target_id is None:
        return None
    if request.left.claim.claim_id == target_id:
        return request.right, request.left
    return request.left, request.right


def _prose_only_targeting(left: TemporalClaim, right: TemporalClaim) -> bool:
    markers = (
        "actually",
        "correction",
        "correct that",
        "ignore what i said",
        "retract",
        "withdraw",
    )
    return any(
        marker in evidence.quote.casefold()
        for item in (left, right)
        for evidence in item.evidence
        for marker in markers
    )


def _decision(
    request: ClassificationRequest,
    snapshot: str,
    label: str,
    source: TemporalClaim | None = None,
    target: TemporalClaim | None = None,
) -> ConflictDecision:
    relation_type = LABEL_RELATION[label]
    relations: tuple[CheckedRelation, ...] = ()
    if relation_type is not None:
        if source is None or target is None:
            raise ConflictClassifierError("matched relation rule requires oriented claims")
        source_id, target_id = source.claim.claim_id, target.claim.claim_id
        if relation_type in SYMMETRIC_RELATIONS:
            source_id, target_id = sorted((source_id, target_id))
        relation = CheckedRelation(
            relation_id=_stable_id(
                "checked_relation",
                request.classifier_version,
                request.user_id,
                source_id,
                target_id,
                relation_type,
                snapshot,
            ),
            classifier_version=request.classifier_version,
            user_id=request.user_id,
            source_claim_id=source_id,
            target_claim_id=target_id,
            relation_type=relation_type,
            confidence=1.0,
            input_snapshot_sha256=snapshot,
        )
        relations = (relation,)
    return ConflictDecision(
        decision_id=_stable_id(
            "conflict_decision",
            request.classifier_version,
            request.user_id,
            request.pair.pair_id,
            label,
            snapshot,
        ),
        classifier_version=request.classifier_version,
        user_id=request.user_id,
        pair_id=request.pair.pair_id,
        label=label,
        confidence=1.0,
        input_snapshot_sha256=snapshot,
        relations=relations,
    )


def _time_relation(left: TemporalClaim, right: TemporalClaim) -> str:
    if "approximate" in {left.version.time_precision, right.version.time_precision}:
        return "approximate"
    left_interval = _finite_interval(left)
    right_interval = _finite_interval(right)
    if left_interval is None or right_interval is None:
        return "unknown"
    if left_interval[0] != right_interval[0]:
        return "mixed"
    _, left_start, left_end = left_interval
    _, right_start, right_end = right_interval
    return (
        "overlap"
        if left_start <= right_end and right_start <= left_end
        else "nonoverlap"
    )


def _finite_interval(
    item: TemporalClaim,
) -> tuple[str, date | datetime, date | datetime] | None:
    version = item.version
    if version.time_precision == "unknown":
        return None
    if version.time_precision == "timestamp":
        if version.valid_from_timestamp is None or version.valid_to_timestamp is None:
            return None
        return "timestamp", version.valid_from_timestamp, version.valid_to_timestamp
    if version.valid_from_date is None or version.valid_to_date is None:
        return None
    return "date", version.valid_from_date, version.valid_to_date


def _semantic_value(item: TemporalClaim) -> str:
    return _canonical_json([item.claim.polarity, item.claim.object_json])


def _refinement(
    left: TemporalClaim, right: TemporalClaim
) -> tuple[TemporalClaim, TemporalClaim] | None:
    left_value = left.claim.object_json
    right_value = right.claim.object_json
    if _strict_detail(left_value, right_value):
        return left, right
    if _strict_detail(right_value, left_value):
        return right, left
    return None


def _strict_detail(specific: object, broad: object) -> bool:
    if isinstance(specific, dict) and isinstance(broad, dict):
        if not set(broad) <= set(specific):
            return False
        compatible = all(
            specific[key] == broad[key]
            or _strict_detail(specific[key], broad[key])
            for key in broad
        )
        return compatible and specific != broad
    if isinstance(specific, list) and isinstance(broad, list):
        specific_items = {_canonical_json(item) for item in specific}
        broad_items = {_canonical_json(item) for item in broad}
        return broad_items < specific_items
    return False


def _input_snapshot(request: ClassificationRequest) -> str:
    payload = {
        "classifier_version": request.classifier_version,
        "user_id": request.user_id,
        "transaction_as_of": request.transaction_as_of,
        "pair": asdict(request.pair),
        "left": asdict(request.left),
        "right": asdict(request.right),
        "source_ingested_at": request.source_ingested_at,
        "explicit_target_claim_id": request.explicit_target_claim_id,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: object) -> object:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConflictClassifierError("snapshot JSON keys must be strings")
            result[key] = _json_value(item)
        return result
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConflictClassifierError("snapshot JSON must be finite")
        return value
    raise ConflictClassifierError("snapshot input is not JSON-safe")


def _stable_id(namespace: str, *values: object) -> str:
    return hashlib.sha256(_canonical_json([namespace, *values]).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConflictClassifierError(f"{name} must be nonempty text")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ConflictClassifierError(f"{name} must be timezone-aware")
    return value


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
