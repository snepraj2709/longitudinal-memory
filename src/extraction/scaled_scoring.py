"""Development-only extraction gold and Step 3.5 score slices."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .atomic import AtomicExtractionValidationError, _validate_claim_records
from .contracts import AtomicClaimV1
from .gold import AtomicGoldCase
from .phase4_input import Phase4InputClaim
from .predicate_registry import PREDICATE_FAMILIES, PredicateRegistry
from .scoring import score_atomic_extraction
from .source import ExtractionSource


SCALED_GOLD_CLAIMS_PATH = Path("data/scaled-v1/gold/claims.jsonl")
DEVELOPMENT_GOLD_CLAIM_COUNT = 32
_GOLD_FIELDS = {
    "claim_id",
    "benchmark_version",
    "user_id",
    "subject_id",
    "speaker_id",
    "predicate",
    "object",
    "polarity",
    "epistemic_status",
    "memory_kind",
    "status",
    "valid_from",
    "valid_to",
    "time_precision",
    "evidence",
    "review_status",
}


class ScaledAtomicScoringError(ValueError):
    """Report development-gold or prediction inconsistencies."""


def load_scaled_development_gold(
    repo_root: str | Path,
    sources: Sequence[ExtractionSource],
    registry: PredicateRegistry,
    *,
    selected_source_ids: set[str] | None = None,
) -> tuple[AtomicGoldCase, ...]:
    """Load only the first two users' reviewed claim labels."""

    source_map = {source.source_id: source for source in sources}
    grouped: dict[str, list[object]] = defaultdict(list)
    path = Path(repo_root).resolve() / SCALED_GOLD_CLAIMS_PATH
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number in range(1, DEVELOPMENT_GOLD_CLAIM_COUNT + 1):
                line = handle.readline()
                if not line:
                    raise ScaledAtomicScoringError(
                        f"scaled gold ended before development line {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ScaledAtomicScoringError(
                        f"scaled gold line {line_number} is invalid: {error.msg}"
                    ) from None
                if not isinstance(record, dict) or set(record) != _GOLD_FIELDS:
                    raise ScaledAtomicScoringError("scaled development gold fields changed")
                user_id = record.get("user_id")
                if user_id not in {"user_001", "user_002"}:
                    raise ScaledAtomicScoringError(
                        "scaled development gold crossed the split boundary"
                    )
                evidence = record.get("evidence")
                if not isinstance(evidence, list) or not evidence:
                    raise ScaledAtomicScoringError("scaled gold evidence is missing")
                source_ids = {
                    item.get("source_id")
                    for item in evidence
                    if isinstance(item, dict)
                }
                if len(source_ids) != 1:
                    raise ScaledAtomicScoringError("scaled gold evidence is not source-local")
                source_id = next(iter(source_ids))
                source = source_map.get(source_id) if isinstance(source_id, str) else None
                if source is None or source.user_id != user_id:
                    raise ScaledAtomicScoringError("scaled gold evidence has wrong ownership")
                grouped[source_id].append(
                    {
                        "claim_id": record["claim_id"],
                        "subject_id": record["subject_id"],
                        "speaker_id": record["speaker_id"],
                        "predicate": record["predicate"],
                        "object": record["object"],
                        "polarity": record["polarity"],
                        "epistemic_status": record["epistemic_status"],
                        "valid_from": record["valid_from"],
                        "valid_to": record["valid_to"],
                        "confidence": 1.0,
                        "evidence": evidence,
                    }
                )
    except OSError as error:
        raise ScaledAtomicScoringError(f"could not load scaled gold: {error}") from error

    if sum(len(items) for items in grouped.values()) != DEVELOPMENT_GOLD_CLAIM_COUNT:
        raise ScaledAtomicScoringError("scaled development gold count changed")
    selected = selected_source_ids or set(source_map)
    if not selected <= set(source_map):
        raise ScaledAtomicScoringError("selected gold sources are unknown")
    cases: list[AtomicGoldCase] = []
    for source in sources:
        if source.source_id not in selected:
            continue
        try:
            claims = _validate_claim_records(
                grouped.get(source.source_id, []), source, registry=registry
            )
        except AtomicExtractionValidationError as error:
            raise ScaledAtomicScoringError(str(error)) from None
        cases.append(
            AtomicGoldCase(
                case_id=source.source_id,
                source_id=source.source_id,
                expected_claims=claims,
            )
        )
    return tuple(cases)


def score_scaled_development(
    gold_cases: Sequence[AtomicGoldCase],
    predictions_by_source: Mapping[str, Sequence[Phase4InputClaim]],
    sources: Sequence[ExtractionSource],
    registry: PredicateRegistry,
) -> dict[str, object]:
    """Report overall metrics and stable user, source, and family slices."""

    source_map = {source.source_id: source for source in sources}
    atomic_predictions = {
        source_id: tuple(_atomic_claim(claim) for claim in claims)
        for source_id, claims in predictions_by_source.items()
    }
    overall = score_atomic_extraction(gold_cases, atomic_predictions)
    by_user = {
        user_id: _score_subset(
            gold_cases,
            atomic_predictions,
            lambda case, wanted=user_id: source_map[case.source_id].user_id == wanted,
        )
        for user_id in ("user_001", "user_002")
    }
    by_source_type = {
        source_type: _score_subset(
            gold_cases,
            atomic_predictions,
            lambda case, wanted=source_type: source_map[case.source_id].source_type == wanted,
        )
        for source_type in ("conversation", "email", "chat", "calendar")
    }
    by_family = {
        family: _score_family(
            gold_cases,
            atomic_predictions,
            family,
            registry,
        )
        for family in sorted(PREDICATE_FAMILIES)
    }
    return {
        **overall,
        "slices": {
            "user": by_user,
            "source_type": by_source_type,
            "predicate_family": by_family,
        },
    }


def qualification_gate(
    full_scores: Mapping[str, object],
    mini_scores: Mapping[str, object],
    *,
    full_failures: Mapping[str, int],
    mini_failures: Mapping[str, int],
) -> dict[str, object]:
    """Apply the frozen mini-model quality and safety gate."""

    reasons: list[str] = []
    for label, failures in (("full", full_failures), ("mini", mini_failures)):
        for name in (
            "execution",
            "persistence",
            "cross_user",
            "provenance",
        ):
            if failures.get(name, 0):
                reasons.append(f"{label} {name} failures must be zero")
    if full_scores.get("total_unsupported_claims") or mini_scores.get(
        "total_unsupported_claims"
    ):
        reasons.append("unsupported claims must be zero")
    full_tp = sum(item["true_positives"] for item in full_scores["case_results"])
    mini_tp = sum(item["true_positives"] for item in mini_scores["case_results"])
    if mini_tp < full_tp - 1:
        reasons.append("mini exact true positives are more than one below full")
    full_spans = full_scores["provenance_span_precision"]["numerator"]
    mini_spans = mini_scores["provenance_span_precision"]["numerator"]
    if mini_spans < full_spans - 1:
        reasons.append("mini matched provenance is more than one below full")
    for metric in ("subject_accuracy", "speaker_accuracy"):
        full_value = full_scores[metric]["value"]
        mini_value = mini_scores[metric]["value"]
        if full_value is not None and (mini_value is None or mini_value < full_value):
            reasons.append(f"mini {metric} is worse than full")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "full_exact_true_positives": full_tp,
        "mini_exact_true_positives": mini_tp,
        "full_matched_provenance_spans": full_spans,
        "mini_matched_provenance_spans": mini_spans,
    }


def _score_subset(
    cases: Sequence[AtomicGoldCase],
    predictions: Mapping[str, Sequence[AtomicClaimV1]],
    include: Callable[[AtomicGoldCase], bool],
) -> dict[str, object]:
    selected = tuple(case for case in cases if include(case))
    return score_atomic_extraction(
        selected,
        {case.case_id: predictions[case.case_id] for case in selected},
    )


def _score_family(
    cases: Sequence[AtomicGoldCase],
    predictions: Mapping[str, Sequence[AtomicClaimV1]],
    family: str,
    registry: PredicateRegistry,
) -> dict[str, object]:
    family_cases = tuple(
        AtomicGoldCase(
            case_id=case.case_id,
            source_id=case.source_id,
            expected_claims=tuple(
                claim
                for claim in case.expected_claims
                if registry.by_predicate[claim.predicate].family == family
            ),
        )
        for case in cases
    )
    family_predictions = {
        case.case_id: tuple(
            claim
            for claim in predictions[case.case_id]
            if registry.by_predicate[claim.predicate].family == family
        )
        for case in cases
    }
    return score_atomic_extraction(family_cases, family_predictions)


def _atomic_claim(claim: Phase4InputClaim) -> AtomicClaimV1:
    return AtomicClaimV1(
        claim_id=claim.claim_id,
        subject_id=claim.subject_id,
        speaker_id=claim.speaker_id,
        predicate=claim.predicate,
        object=claim.object,
        polarity=claim.polarity,
        epistemic_status=claim.epistemic_status,
        valid_from=claim.valid_from,
        valid_to=claim.valid_to,
        confidence=claim.confidence,
        evidence=claim.evidence,
    )
