"""Deterministic threshold overlay for frozen Step 9.1 decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .calibration_contracts import (
    PROFILE_ORDER,
    SELECTED_PROFILE,
    THRESHOLD_VERSION,
    AnswerabilityThresholdProfile,
    CalibrationError,
    CalibrationFixture,
    ThresholdApplication,
    ThresholdConfig,
    ThresholdPrediction,
    canonical_json_bytes,
    load_config_mapping,
)
from .contracts import AnswerabilityDecision, canonical_json_bytes as answerability_json


CONFIG_PATH = Path("configs/abstention/answerability_thresholds_v1.json")
CONFIDENCE = {
    "value": None,
    "calibration_status": "not_calibrated",
    "null_reason": "no_authorized_answerability_gold",
}


def distinct_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize repeated evidence lineage before threshold counting."""

    result = tuple(sorted(set(values)))
    try:
        valid = all(isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0 for value in result)
    except ValueError:
        valid = False
    if not valid:
        raise CalibrationError("evidence identity is invalid")
    return result


def load_threshold_config(path: str | Path = CONFIG_PATH, *, repo_root: str | Path = ".") -> ThresholdConfig:
    target = _resolve(Path(repo_root).resolve(), path)
    value = json.loads(target.read_text(encoding="utf-8"))
    return load_config_mapping(value)


def predict_fixture(
    fixture: CalibrationFixture,
    profile: AnswerabilityThresholdProfile,
) -> ThresholdPrediction:
    """Apply one frozen profile without changing the input fixture decision."""

    if fixture.input_decision != "answerable":
        output_decision = fixture.input_decision
        output_reasons = fixture.input_reasons
        withheld = False
        override_used = False
    else:
        passes, override_used = _fixture_passes(fixture, profile)
        output_decision = "answerable" if passes else "abstain"
        output_reasons = () if passes else ("withheld_by_threshold",)
        withheld = not passes
    fields = {
        "threshold_version": THRESHOLD_VERSION,
        "fixture_id": fixture.fixture_id,
        "profile_id": profile.profile_id,
        "input_decision": fixture.input_decision,
        "output_decision": output_decision,
        "input_reasons": fixture.input_reasons,
        "output_reasons": output_reasons,
        "withheld_by_threshold": withheld,
        "distinct_authoritative_source_count": len(fixture.authoritative_source_ids),
        "exact_evidence_path_count": len(fixture.exact_evidence_path_ids),
        "distinct_session_count": len(fixture.session_ids),
        "distinct_episode_time_count": len(fixture.episode_times),
        "closed_durative_interval_override_used": override_used,
    }
    return ThresholdPrediction(prediction_id=_id_for(fields), **fields)


def predict_all_fixtures(
    fixtures: Iterable[CalibrationFixture],
    config: ThresholdConfig,
) -> tuple[ThresholdPrediction, ...]:
    predictions = tuple(
        predict_fixture(fixture, profile)
        for fixture in sorted(fixtures, key=lambda item: item.fixture_id)
        for profile in config.profiles
    )
    if len(predictions) != 144 or len({item.prediction_id for item in predictions}) != 144:
        raise CalibrationError("runtime prediction accounting changed")
    return predictions


def apply_selected_thresholds(
    decisions: Iterable[AnswerabilityDecision],
    config: ThresholdConfig,
) -> tuple[ThresholdApplication, ...]:
    profile = next(item for item in config.profiles if item.profile_id == SELECTED_PROFILE)
    applications = tuple(
        _apply_decision(decision, profile)
        for decision in sorted(
            decisions,
            key=lambda item: (item.query_id, ("B2", "B3", "B4").index(item.baseline_id)),
        )
    )
    if len(applications) != 24 or len({item.application_id for item in applications}) != 24:
        raise CalibrationError("development application accounting changed")
    return applications


def _apply_decision(
    decision: AnswerabilityDecision,
    profile: AnswerabilityThresholdProfile,
) -> ThresholdApplication:
    accepted_bytes = canonical_json_bytes(decision.accepted_evidence)
    rejected_bytes = canonical_json_bytes(decision.rejected_evidence)
    input_bytes = answerability_json(decision)
    output_decision = decision.decision
    withheld = False
    generation_allowed = decision.generation_allowed
    if decision.decision == "answerable" and not _decision_meets_threshold(decision, profile):
        output_decision = "abstain"
        generation_allowed = False
        withheld = True
    fields = {
        "threshold_version": THRESHOLD_VERSION,
        "profile_id": profile.profile_id,
        "input_decision_id": decision.decision_id,
        "input_decision_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "package_id": decision.package_id,
        "user_id": decision.user_id,
        "query_id": decision.query_id,
        "baseline_id": decision.baseline_id,
        "input_decision": decision.decision,
        "output_decision": output_decision,
        "primary_reason": decision.primary_reason,
        "reasons": decision.reasons,
        "generation_allowed": generation_allowed,
        "accepted_evidence_count": len(decision.accepted_evidence),
        "accepted_evidence_sha256": hashlib.sha256(accepted_bytes).hexdigest(),
        "rejected_evidence_count": len(decision.rejected_evidence),
        "rejected_evidence_sha256": hashlib.sha256(rejected_bytes).hexdigest(),
        "withheld_by_threshold": withheld,
        "confidence": CONFIDENCE,
    }
    return ThresholdApplication(application_id=_id_for(fields), **fields)


def _fixture_passes(
    fixture: CalibrationFixture,
    profile: AnswerabilityThresholdProfile,
) -> tuple[bool, bool]:
    if (
        not fixture.semantic_guards_passed
        or fixture.promoted_claim_count < profile.minimum_promoted_claims
        or fixture.required_part_count < 1
        or fixture.supported_part_count != fixture.required_part_count
        or len(fixture.exact_evidence_path_ids) < profile.minimum_exact_evidence_paths
        or fixture.unresolved_conflict_count > profile.maximum_unresolved_conflicts
        or fixture.stale_evidence_count > profile.maximum_stale_evidence
        or (fixture.information_kind == "causal" and not fixture.causal_relation_verified)
    ):
        return False, False
    if fixture.information_kind == "stable_trait":
        if fixture.closed_durative_interval and profile.explicit_closed_durative_interval_override:
            return True, True
        return (
            len(fixture.authoritative_source_ids) >= profile.stable_trait_minimum_distinct_sources
            and len(fixture.session_ids) >= profile.stable_trait_minimum_distinct_sessions
            and len(fixture.episode_times) >= profile.stable_trait_minimum_distinct_episode_times,
            False,
        )
    if fixture.information_kind == "causal":
        return True, False
    return (
        len(fixture.authoritative_source_ids)
        >= profile.ordinary_minimum_distinct_authoritative_sources,
        False,
    )


def _decision_meets_threshold(
    decision: AnswerabilityDecision,
    profile: AnswerabilityThresholdProfile,
) -> bool:
    assessment = decision.assessment
    source_ids = {source_id for item in decision.accepted_evidence for source_id in item.source_ids}
    evidence_ids = {evidence_id for item in decision.accepted_evidence for evidence_id in item.evidence_ids}
    if (
        not assessment.structural_coverage_complete
        or assessment.promoted_claim_count < profile.minimum_promoted_claims
        or assessment.required_part_count < 1
        or assessment.supported_part_count != assessment.required_part_count
        or len(evidence_ids) < profile.minimum_exact_evidence_paths
        or assessment.unresolved_conflict_count > profile.maximum_unresolved_conflicts
        or assessment.stale_evidence_count > profile.maximum_stale_evidence
        or len(source_ids) < profile.ordinary_minimum_distinct_authoritative_sources
    ):
        return False
    if decision.query_type == "stable_trait":
        # Step 9.1 references do not serialize session or episode counts. Fail closed.
        return False
    return True


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _id_for(fields: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(fields)).hexdigest()
