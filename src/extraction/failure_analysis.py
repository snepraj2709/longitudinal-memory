"""Deterministic failure analysis for the frozen Phase 3 atomic v2 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Mapping, Sequence

from .atomic import _validate_claim_records
from .contracts import ALLOWED_PREDICATES, AtomicClaimV1
from .gold import ATOMIC_GOLD_PATH, AtomicGoldCase, load_atomic_gold
from .prompt import (
    ATOMIC_EXTRACTION_PROMPT_VERSION,
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
)
from .scoring import (
    ATOMIC_SCORING_VERSION,
    _evidence_counts,
    _field_matches,
    _match_claims,
    score_atomic_extraction,
)
from .source import ExtractionSource, load_pilot_sources


ANALYSIS_VERSION = "atomic-extraction-failure-analysis-v1"
DEFAULT_INPUT_DIR = Path("results/phase3/atomic-extraction-v2")
DEFAULT_OUTPUT_DIR = Path(
    "results/phase3/atomic-extraction-v2-failure-analysis-v1"
)
GOLD_REVIEW_PATH = Path("data/phase3/atomic_extraction_gold_review.json")

STABLE_CATEGORIES = (
    "missed_gold_claim",
    "extra_source_backed_claim",
    "unsupported_claim",
    "subject_mismatch",
    "speaker_mismatch",
    "predicate_mismatch",
    "object_mismatch",
    "polarity_mismatch",
    "epistemic_status_mismatch",
    "missing_valid_time",
    "wrong_valid_time",
    "over_broad_valid_time",
    "evidence_span_mismatch",
)

PREDICATE_FAMILIES = {
    "accepted_offer": "event",
    "assigned_task": "task",
    "believes_aryans_move_signals_relationship_change": "belief",
    "believes_marketing_is_right_fit": "belief",
    "campaign_launch_date": "schedule",
    "can_continue_marketing_and_product_work_this_way": "state",
    "explains_work_supportively": "relationship",
    "feels_exhausted": "state",
    "feels_lonely_in_bengaluru": "state",
    "finds_returning_to_empty_flat_difficult": "state",
    "found_onboarding_useful": "assessment",
    "has_difficulty_focusing_at_work": "state",
    "has_handoff_cover": "task",
    "has_scheduled_event": "schedule",
    "hoped_aryans_move_signaled_relationship_change": "belief",
    "job_location": "role",
    "job_start_date": "schedule",
    "leave_approved": "commitment",
    "leave_denial_reason": "commitment",
    "likes": "preference",
    "misses": "state",
    "needs_to_finish_college": "goal",
    "offered_help": "commitment",
    "plans_to_decide_marketing_fit_after_work_experience": "goal",
    "plans_to_finish_campaign_draft_before_leave": "task",
    "plans_to_share_kids_spark_notes_with": "task",
    "plans_to_try_marketing_work": "goal",
    "rated_first_work_week": "assessment",
    "relationship_would_change_if_living_same_city": "relationship",
    "requested_leave": "commitment",
    "wants_to_restart_relationship_with": "relationship",
    "wants_to_talk_to": "relationship",
    "will_have_job_role": "role",
    "will_run_marketing_team": "role",
    "will_work_for": "role",
    "works_on_product_backlog_after_marketing_work": "task",
}

FROZEN_INPUT_SHA256 = {
    "data/phase3/atomic_extraction_gold.jsonl": "e802835dcb3e4278ec68e45474d094896dcad9a780c4115a01e0f0bc44f07f63",
    "data/phase3/atomic_extraction_gold_review.json": "9b98d6dcb093e804a46afab2d36c30b61d5be254a2730aacc72805ad7eeae9f7",
    "data/pilot/sources/calendar.jsonl": "8f5d3dfef2ab3115b2099e79b4151415925649b892c9c3443cfe89eee89752b9",
    "data/pilot/sources/conversations.jsonl": "0f4900d8b13ee55040bcfde4d1b7c6943786b11d1723c6c7aaf2ce0eb6480c9e",
    "data/pilot/sources/emails.jsonl": "ba94d6c80b8d307862e2692ad5ee37db0114599a3190811145f647976e816bed",
    "results/phase3/atomic-extraction-v2/run.json": "7a22b863e8e6da796a308ea1553818a957189159807d8087917626cdd9524a6d",
    "results/phase3/atomic-extraction-v2/predictions.jsonl": "f0dc940505d0d99874b055ddc1f0c8d0be3ac9d74d975018d88829405e6c65b5",
    "results/phase3/atomic-extraction-v2/scores.json": "d81cdedd2323f92c2c2ff6f94fc6c87e4d14c80fa3fb0e059f9fdcd70bc78b23",
    "results/phase3/atomic-extraction-v2/case_scores.jsonl": "4883eaec9d0de7c864c5b162c0bec6c7acb7cecacc8406a811e68886fa16f0a3",
    "results/phase3/atomic-extraction-v2/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}

PROTECTED_B1_SHA256 = {
    "configs/full_history_baseline_v1.json": "9ac8881719153f09369e195e1c81eb55cc5f2f50e591f87b4fe2664c16bf6d76",
    "results/pilot/b1-full-history/run.json": "b8f7b38fac1d28b70de219a96a567e98052a86764513da01d692019b87926507",
    "results/pilot/b1-full-history/predictions.jsonl": "63e5b1061c98edbec8a2c77c2260bd364bb3362d876fe22a02e3a10a2f5187b0",
    "results/pilot/b1-full-history/scores.json": "3187cedbabb60fbec2d6aca1dfd517d09cd37a404f7cd00e04d9065155b91f3d",
    "results/pilot/b1-full-history/case_scores.jsonl": "07a0db890fef5b1fa203edcf642b1cbaeef8602d147aac22c6c660f377da4ab8",
    "results/pilot/b1-full-history/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/pilot/b1-full-history/manual_review.jsonl": "9e9a74331a9af4f608b780a23fce5e936e0964f06edac874424d127188fbe2c0",
    "results/pilot/b1-full-history/failure_analysis.jsonl": "8332084a3316e326074c767d45bf7fd5824a45f62e0bcae9e3f165b9cf053bf1",
    "results/pilot/b1-full-history/failure_summary.json": "3c309527c10c9fc4344b7acb5994795e20e1f174557087e769d76a6b2b96bd0f",
    "results/pilot/b1-full-history/baseline_findings.md": "62c8f0d1d64972b2fe031dd5aadd7efa7d1cb9f0d17ec5de9019879dcc2151f8",
}

DENOMINATOR_RULES = {
    "claim_errors": "False positives and false negatives use exact claim-content matching from atomic-scoring-v2.",
    "field_errors": "Field errors use the scorer's deterministic evidence alignment; each aligned pair is one field denominator.",
    "evidence_errors": "Evidence precision uses predicted exact spans and evidence recall uses gold exact spans. A span is source_id, message_id, and exact quote.",
    "issue_counts": "Category counts count classified issue records, not distinct claims. One representation pair can have several field or evidence issues.",
    "execution_failures": "Execution failures are reported separately and excluded from semantic failure categories.",
}


class AtomicFailureAnalysisError(RuntimeError):
    """Raised when frozen inputs or deterministic reconciliation fail."""


def run_failure_analysis(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Analyse the frozen v2 result without changing any protected artifact."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    _require_empty_output_directory(output_path)
    input_hashes = _verify_hashes(root, FROZEN_INPUT_SHA256, "Phase 3 v2 input")
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")
    _validate_predicate_families()

    sources = load_pilot_sources() if root == Path.cwd().resolve() else _load_sources(root)
    sources_by_id = {source.source_id: source for source in sources}
    gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, sources)
    predictions = _load_predictions(
        root / DEFAULT_INPUT_DIR / "predictions.jsonl", sources_by_id
    )
    stored_scores = _read_json(root / DEFAULT_INPUT_DIR / "scores.json")
    recomputed_scores = score_atomic_extraction(gold_cases, predictions)
    if recomputed_scores != stored_scores:
        raise AtomicFailureAnalysisError(
            "frozen scores do not reconcile with atomic-scoring-v2"
        )

    failures = _read_jsonl(root / DEFAULT_INPUT_DIR / "failures.jsonl")
    failures_by_case = {record["case_id"]: record for record in failures}
    records = [
        _analyse_case(
            gold_case,
            predictions[gold_case.case_id],
            sources_by_id[gold_case.source_id],
            failures_by_case.get(gold_case.case_id),
        )
        for gold_case in gold_cases
    ]
    summary = _build_summary(records, stored_scores)
    findings = _render_findings(summary)

    output_path.mkdir(parents=True)
    _write_jsonl(output_path / "case_failures.jsonl", records)
    _write_json(output_path / "summary.json", summary)
    (output_path / "findings.md").write_text(findings, encoding="utf-8")

    output_hashes = {
        name: _sha256(output_path / name)
        for name in ("case_failures.jsonl", "summary.json", "findings.md")
    }
    prompt_hash = _canonical_sha256(
        {
            "prompt_version": ATOMIC_EXTRACTION_PROMPT_VERSION,
            "system_prompt": ATOMIC_EXTRACTION_SYSTEM_PROMPT,
            "user_prompts": [
                build_atomic_extraction_prompt(sources_by_id[case.source_id])
                for case in gold_cases
            ],
        }
    )
    clock = now or (lambda: datetime.now(timezone.utc))
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "source_prompt_version": ATOMIC_EXTRACTION_PROMPT_VERSION,
        "source_scoring_version": ATOMIC_SCORING_VERSION,
        "release_status": "approved",
        "human_review": {
            "status": "approved",
            "reviewer": "Sneha",
            "reviewed_manifest_sha256": "d44a71d6da47d4fc970473db97c10432edf68a8282df9ecac7f4d4c40ef835a5",
            "approved_on": "2026-08-08",
            "note": "Sneha approved the Step 3.1 candidate against its manifest SHA-256.",
        },
        "generated_at_utc": _utc_text(clock()),
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "repository_worktree_state": (
            "clean" if not _git(root, "status", "--porcelain") else "dirty"
        ),
        "analysis_scope": {
            "split": "phase3_atomic_extraction_pilot",
            "case_count": len(records),
            "source_types": sorted({record["source_type"] for record in records}),
            "no_api_calls": True,
        },
        "input_file_sha256": input_hashes,
        "protected_b1_file_sha256": protected_hashes,
        "prepared_prompt_sha256": prompt_hash,
        "counts": summary["counts"],
        "denominator_rules": DENOMINATOR_RULES,
        "output_file_sha256": output_hashes,
        "repeatability": {
            "isolated_nondeterministic_field": "generated_at_utc",
            "substantive_outputs_expected_byte_identical": True,
        },
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _analyse_case(
    gold_case: AtomicGoldCase,
    predictions: Sequence[AtomicClaimV1],
    source: ExtractionSource,
    execution_failure: Mapping[str, object] | None,
) -> dict[str, object]:
    gold = tuple(gold_case.expected_claims)
    predicted = tuple(predictions)
    exact = _match_claims(gold, predicted, "exact")
    aligned = _match_claims(gold, predicted, "aligned")
    supported = set(_match_claims(gold, predicted, "supported"))
    supported_predictions = {prediction_index for _, prediction_index in supported}
    exact_set = set(exact)
    aligned_set = set(aligned)
    issues: list[dict[str, object]] = []

    def add(
        category: str,
        representation: str,
        gold_claim: AtomicClaimV1 | None,
        predicted_claim: AtomicClaimV1 | None,
        explanation: str,
        *,
        accounts_for: Mapping[str, int],
        gold_value: object = None,
        predicted_value: object = None,
        missing_spans: Sequence[tuple[str, str | None, str]] = (),
        extra_spans: Sequence[tuple[str, str | None, str]] = (),
    ) -> None:
        predicate = (
            gold_claim.predicate if gold_claim is not None else predicted_claim.predicate
        )
        issue = {
            "issue_id": f"{gold_case.case_id}_issue_{len(issues) + 1:03d}",
            "category": category,
            "representation": representation,
            "predicate": predicate,
            "predicate_family": PREDICATE_FAMILIES[predicate],
            "gold_claim_id": gold_claim.claim_id if gold_claim else None,
            "predicted_claim_id": predicted_claim.claim_id if predicted_claim else None,
            "accounts_for": dict(accounts_for),
            "explanation": explanation,
            "gold_value": gold_value,
            "predicted_value": predicted_value,
            "source_references": _source_references(gold_claim, predicted_claim),
            "missing_gold_evidence_spans": [_span_record(span) for span in missing_spans],
            "extra_predicted_evidence_spans": [_span_record(span) for span in extra_spans],
        }
        if not issue["source_references"]:
            raise AtomicFailureAnalysisError(
                f"classified issue {issue['issue_id']} has no source reference"
            )
        issues.append(issue)

    for gold_index, prediction_index in aligned:
        gold_claim = gold[gold_index]
        prediction = predicted[prediction_index]
        pair = (gold_index, prediction_index)
        if pair not in exact_set:
            if prediction_index in supported_predictions:
                add(
                    "extra_source_backed_claim",
                    "source_backed_representation_difference",
                    gold_claim,
                    prediction,
                    "The prediction cites the same source record as the gold claim and passes the frozen scorer's subject, predicate, and polarity support check, but its claim content is not exact.",
                    accounts_for={"false_positive": 1, "false_negative": 1},
                    gold_value=_claim_record(gold_claim),
                    predicted_value=_claim_record(prediction),
                )
            else:
                add(
                    "missed_gold_claim",
                    "missed_extraction",
                    gold_claim,
                    None,
                    "No source-supported exact prediction represents this gold claim.",
                    accounts_for={"false_positive": 0, "false_negative": 1},
                    gold_value=_claim_record(gold_claim),
                    missing_spans=[
                        (span.source_id, span.message_id, span.quote)
                        for span in gold_claim.evidence
                    ],
                )
                add(
                    "unsupported_claim",
                    "unsupported_model_output",
                    None,
                    prediction,
                    "The prediction does not pass the frozen scorer's source-supported alignment check against any gold claim.",
                    accounts_for={"false_positive": 1, "false_negative": 0},
                    predicted_value=_claim_record(prediction),
                    extra_spans=[
                        (span.source_id, span.message_id, span.quote)
                        for span in prediction.evidence
                    ],
                )

        matches = _field_matches(gold_claim, prediction)
        field_categories = {
            "subject": "subject_mismatch",
            "speaker": "speaker_mismatch",
            "predicate": "predicate_mismatch",
            "object": "object_mismatch",
            "polarity": "polarity_mismatch",
            "epistemic_status": "epistemic_status_mismatch",
        }
        for field, category in field_categories.items():
            if not matches[field]:
                gold_value, prediction_value = _field_values(
                    field, gold_claim, prediction
                )
                add(
                    category,
                    "source_backed_representation_difference",
                    gold_claim,
                    prediction,
                    f"The aligned source-backed claims differ in {field}.",
                    accounts_for={"field_error": 1},
                    gold_value=gold_value,
                    predicted_value=prediction_value,
                )
        if not matches["valid_time"]:
            category = classify_valid_time_mismatch(gold_claim, prediction)
            add(
                category,
                "source_backed_representation_difference",
                gold_claim,
                prediction,
                "The aligned source-backed claims use different valid-time boundaries.",
                accounts_for={"field_error": 1},
                gold_value={
                    "valid_from": gold_claim.valid_from,
                    "valid_to": gold_claim.valid_to,
                },
                predicted_value={
                    "valid_from": prediction.valid_from,
                    "valid_to": prediction.valid_to,
                },
            )

        missing, extra = _pair_evidence_difference(gold_claim, prediction)
        if missing or extra:
            add(
                "evidence_span_mismatch",
                "evidence_only_mismatch",
                gold_claim,
                prediction,
                "The aligned claims do not use the same exact source_id, message_id, and quote spans.",
                accounts_for={"evidence_mismatch": 1},
                missing_spans=missing,
                extra_spans=extra,
            )

    aligned_gold = {gold_index for gold_index, _ in aligned_set}
    aligned_predictions = {prediction_index for _, prediction_index in aligned_set}
    for gold_index, gold_claim in enumerate(gold):
        if gold_index not in aligned_gold:
            add(
                "missed_gold_claim",
                "missed_extraction",
                gold_claim,
                None,
                "No prediction aligned to this source-backed gold claim.",
                accounts_for={"false_positive": 0, "false_negative": 1},
                gold_value=_claim_record(gold_claim),
                missing_spans=[
                    (span.source_id, span.message_id, span.quote)
                    for span in gold_claim.evidence
                ],
            )
    for prediction_index, prediction in enumerate(predicted):
        if prediction_index not in aligned_predictions:
            category = (
                "extra_source_backed_claim"
                if prediction_index in supported_predictions
                else "unsupported_claim"
            )
            add(
                category,
                (
                    "source_backed_representation_difference"
                    if category == "extra_source_backed_claim"
                    else "unsupported_model_output"
                ),
                None,
                prediction,
                (
                    "The source-backed prediction has no exact gold claim."
                    if category == "extra_source_backed_claim"
                    else "The prediction has no frozen-scorer-supported gold alignment."
                ),
                accounts_for={"false_positive": 1, "false_negative": 0},
                predicted_value=_claim_record(prediction),
                extra_spans=[
                    (span.source_id, span.message_id, span.quote)
                    for span in prediction.evidence
                ],
            )

    gold_spans = _evidence_counts(gold)
    predicted_spans = _evidence_counts(predicted)
    matched_span_count = sum((gold_spans & predicted_spans).values())
    score = score_atomic_extraction([gold_case], {gold_case.case_id: predicted})[
        "case_results"
    ][0]
    accounted_fp = sum(issue["accounts_for"].get("false_positive", 0) for issue in issues)
    accounted_fn = sum(issue["accounts_for"].get("false_negative", 0) for issue in issues)
    if accounted_fp != score["false_positives"] or accounted_fn != score["false_negatives"]:
        raise AtomicFailureAnalysisError(
            f"failure accounting does not reconcile for {gold_case.case_id}"
        )

    category_counts = Counter(issue["category"] for issue in issues)
    field_error_categories = {
        "subject_accuracy": ("subject_mismatch",),
        "predicate_accuracy": ("predicate_mismatch",),
        "object_accuracy": ("object_mismatch",),
        "polarity_accuracy": ("polarity_mismatch",),
        "speaker_accuracy": ("speaker_mismatch",),
        "epistemic_status_accuracy": ("epistemic_status_mismatch",),
        "valid_time_accuracy": (
            "missing_valid_time",
            "wrong_valid_time",
            "over_broad_valid_time",
        ),
    }
    for metric, categories in field_error_categories.items():
        expected_errors = score[metric]["denominator"] - score[metric]["numerator"]
        classified_errors = sum(category_counts[category] for category in categories)
        if classified_errors != expected_errors:
            raise AtomicFailureAnalysisError(
                f"{metric} failure accounting does not reconcile for {gold_case.case_id}"
            )
    classified_missing_spans = sum(
        len(issue["missing_gold_evidence_spans"]) for issue in issues
    )
    classified_extra_spans = sum(
        len(issue["extra_predicted_evidence_spans"]) for issue in issues
    )
    if classified_missing_spans != sum(gold_spans.values()) - matched_span_count:
        raise AtomicFailureAnalysisError(
            f"missing evidence accounting does not reconcile for {gold_case.case_id}"
        )
    if classified_extra_spans != sum(predicted_spans.values()) - matched_span_count:
        raise AtomicFailureAnalysisError(
            f"extra evidence accounting does not reconcile for {gold_case.case_id}"
        )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": gold_case.case_id,
        "source_id": gold_case.source_id,
        "source_type": source.source_type,
        "execution_failure": dict(execution_failure) if execution_failure else None,
        "counts": {
            "gold_claims": len(gold),
            "predicted_claims": len(predicted),
            "exact_matches": len(exact),
            "aligned_claims": len(aligned),
            "supported_aligned_claims": len(supported),
            "false_positives": score["false_positives"],
            "false_negatives": score["false_negatives"],
            "issues": len(issues),
        },
        "evidence_span_counts": {
            "gold": sum(gold_spans.values()),
            "predicted": sum(predicted_spans.values()),
            "exact_matches": matched_span_count,
            "false_positives": sum(predicted_spans.values()) - matched_span_count,
            "false_negatives": sum(gold_spans.values()) - matched_span_count,
        },
        "category_counts": {
            category: category_counts.get(category, 0) for category in STABLE_CATEGORIES
        },
        "issues": issues,
    }


def classify_valid_time_mismatch(
    gold: AtomicClaimV1, prediction: AtomicClaimV1
) -> str:
    """Assign one stable valid-time mismatch category."""

    gold_bounds = (gold.valid_from, gold.valid_to)
    predicted_bounds = (prediction.valid_from, prediction.valid_to)
    if gold_bounds == predicted_bounds:
        raise AtomicFailureAnalysisError("valid-time values are equal")
    if any(gold_value is not None and predicted_value is None for gold_value, predicted_value in zip(gold_bounds, predicted_bounds)):
        return "missing_valid_time"
    if _contains_gold_interval(prediction, gold):
        return "over_broad_valid_time"
    return "wrong_valid_time"


def _build_summary(
    records: Sequence[Mapping[str, object]], stored_scores: Mapping[str, object]
) -> dict[str, object]:
    dimensions: dict[str, dict[str, dict[str, object]]] = {
        "by_case": {},
        "by_source_type": {},
        "by_predicate_family": {},
        "by_category": {},
    }
    aggregate_categories: Counter[str] = Counter()
    total_evidence = Counter()
    total_counts = Counter()

    for record in records:
        counts = record["counts"]
        evidence = record["evidence_span_counts"]
        total_counts.update(counts)
        total_evidence.update(evidence)
        case_categories = Counter(record["category_counts"])
        aggregate_categories.update(case_categories)
        dimensions["by_case"][record["case_id"]] = {
            "source_id": record["source_id"],
            "source_type": record["source_type"],
            "false_positives": counts["false_positives"],
            "false_negatives": counts["false_negatives"],
            "issue_count": counts["issues"],
            "category_counts": dict(case_categories),
        }
        _update_dimension(
            dimensions["by_source_type"], record["source_type"], record["issues"]
        )
        for issue in record["issues"]:
            _update_dimension(
                dimensions["by_predicate_family"],
                issue["predicate_family"],
                [issue],
            )
            _update_dimension(dimensions["by_category"], issue["category"], [issue])

    counts = {
        "cases": len(records),
        "execution_failures": sum(record["execution_failure"] is not None for record in records),
        "gold_claims": total_counts["gold_claims"],
        "predicted_claims": total_counts["predicted_claims"],
        "exact_matches": total_counts["exact_matches"],
        "aligned_claims": total_counts["aligned_claims"],
        "false_positives": total_counts["false_positives"],
        "false_negatives": total_counts["false_negatives"],
        "issues": total_counts["issues"],
        "unsupported_claims": aggregate_categories["unsupported_claim"],
        "source_backed_representation_differences": aggregate_categories[
            "extra_source_backed_claim"
        ],
        "evidence_gold_spans": total_evidence["gold"],
        "evidence_predicted_spans": total_evidence["predicted"],
        "evidence_exact_matches": total_evidence["exact_matches"],
        "evidence_false_positives": total_evidence["false_positives"],
        "evidence_false_negatives": total_evidence["false_negatives"],
    }
    if (
        counts["gold_claims"] != stored_scores["total_gold_claims"]
        or counts["predicted_claims"] != stored_scores["total_predicted_claims"]
        or counts["unsupported_claims"] != stored_scores["total_unsupported_claims"]
    ):
        raise AtomicFailureAnalysisError("aggregate analysis does not reconcile")

    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_prompt_version": ATOMIC_EXTRACTION_PROMPT_VERSION,
        "source_scoring_version": ATOMIC_SCORING_VERSION,
        "counts": counts,
        "published_metrics": {
            key: stored_scores[key]
            for key in (
                "micro_claim_precision",
                "micro_claim_recall",
                "micro_claim_f1",
                "subject_accuracy",
                "predicate_accuracy",
                "object_accuracy",
                "polarity_accuracy",
                "speaker_accuracy",
                "epistemic_status_accuracy",
                "valid_time_accuracy",
                "provenance_span_precision",
                "provenance_span_recall",
                "unsupported_memory_rate",
            )
        },
        "category_counts": {
            category: aggregate_categories.get(category, 0)
            for category in STABLE_CATEGORIES
        },
        **{
            dimension: {
                key: value for key, value in sorted(entries.items())
            }
            for dimension, entries in dimensions.items()
        },
        "denominator_rules": DENOMINATOR_RULES,
    }


def _update_dimension(
    dimension: dict[str, dict[str, object]],
    key: str,
    issues: Sequence[Mapping[str, object]],
) -> None:
    entry = dimension.setdefault(
        key,
        {
            "issue_count": 0,
            "false_positives_accounted": 0,
            "false_negatives_accounted": 0,
            "category_counts": {},
        },
    )
    categories = Counter(entry["category_counts"])
    for issue in issues:
        entry["issue_count"] += 1
        entry["false_positives_accounted"] += issue["accounts_for"].get(
            "false_positive", 0
        )
        entry["false_negatives_accounted"] += issue["accounts_for"].get(
            "false_negative", 0
        )
        categories[issue["category"]] += 1
    entry["category_counts"] = dict(sorted(categories.items()))


def _render_findings(summary: Mapping[str, object]) -> str:
    counts = summary["counts"]
    metrics = summary["published_metrics"]
    active_categories = [
        (category, count)
        for category, count in summary["category_counts"].items()
        if count
    ]
    active_categories.sort(key=lambda item: (-item[1], item[0]))
    affected_cases = [
        (case_id, data)
        for case_id, data in summary["by_case"].items()
        if data["false_positives"] or data["false_negatives"]
    ]
    affected_cases.sort(
        key=lambda item: (
            -(item[1]["false_positives"] + item[1]["false_negatives"]),
            item[0],
        )
    )
    category_text = ", ".join(
        f"`{category}` ({count})" for category, count in active_categories
    )
    case_text = ", ".join(
        f"`{case_id}` ({data['false_positives']} FP, {data['false_negatives']} FN)"
        for case_id, data in affected_cases
    )
    return f"""# Phase 3 atomic extraction v2: failure analysis

This report analyses the frozen ten-case v2 pilot. It does not rerun extraction or change the published scores.

## What the run got right

All {counts['cases']} cases completed. The run produced {counts['predicted_claims']} claims against {counts['gold_claims']} gold claims, with {counts['exact_matches']} exact matches. Precision was {metrics['micro_claim_precision']['value']:.6f}, recall was {metrics['micro_claim_recall']['value']:.6f}, and F1 was {metrics['micro_claim_f1']['value']:.6f}. No prediction was classified as unsupported by the frozen scorer.

## Where it failed

The exact-match totals contain {counts['false_positives']} false positives and {counts['false_negatives']} false negatives. The affected cases were {case_text}.

The classified issues were {category_text}. Source-backed representation differences are kept separate from unsupported output: there were {counts['source_backed_representation_differences']} representation pairs and {counts['unsupported_claims']} unsupported claims.

Valid-time accuracy was {metrics['valid_time_accuracy']['value']:.6f}. All {summary['category_counts']['missing_valid_time']} observed time errors were missing boundaries. The taxonomy can also record wrong or over-broad ranges when they occur.

Evidence was the weakest part of the run. Only {counts['evidence_exact_matches']} of {counts['evidence_predicted_spans']} predicted spans exactly matched a gold span, while the gold set contained {counts['evidence_gold_spans']} spans. The case records show the missing and extra exact spans for every aligned evidence mismatch.

## Reading the artifacts

`case_failures.jsonl` is the claim-level audit trail. `summary.json` groups the same issues by case, source type, predicate family, and category. Execution failures have their own field and are not mixed into semantic errors. The manifest records the frozen inputs, protected B1 hashes, output hashes, denominator rules, and review state.
"""


def _load_predictions(
    path: Path, sources_by_id: Mapping[str, ExtractionSource]
) -> dict[str, tuple[AtomicClaimV1, ...]]:
    predictions: dict[str, tuple[AtomicClaimV1, ...]] = {}
    for line_number, record in enumerate(_read_jsonl(path), start=1):
        if set(record) != {"case_id", "source_id", "claims"}:
            raise AtomicFailureAnalysisError(
                f"{path}:{line_number}: invalid prediction record fields"
            )
        case_id = record["case_id"]
        source_id = record["source_id"]
        if case_id in predictions:
            raise AtomicFailureAnalysisError(f"duplicate prediction case {case_id}")
        source = sources_by_id.get(source_id)
        if source is None:
            raise AtomicFailureAnalysisError(f"unknown prediction source {source_id}")
        predictions[case_id] = _validate_claim_records(record["claims"], source)
    return predictions


def _load_sources(root: Path) -> tuple[ExtractionSource, ...]:
    from evaluation.history import load_history_observations
    from .source import group_source_observations

    return group_source_observations(
        load_history_observations(root / "data/pilot/sources")
    )


def _source_references(
    gold: AtomicClaimV1 | None, prediction: AtomicClaimV1 | None
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    for role, claim in (("gold", gold), ("prediction", prediction)):
        if claim is None:
            continue
        references.extend(
            {
                "role": role,
                "source_id": evidence.source_id,
                "message_id": evidence.message_id,
                "quote": evidence.quote,
            }
            for evidence in claim.evidence
        )
    return references


def _claim_record(claim: AtomicClaimV1) -> dict[str, object]:
    record = asdict(claim)
    record.pop("claim_id")
    record.pop("confidence")
    record.pop("evidence")
    return record


def _field_values(
    field: str, gold: AtomicClaimV1, prediction: AtomicClaimV1
) -> tuple[object, object]:
    attribute = {
        "subject": "subject_id",
        "speaker": "speaker_id",
        "predicate": "predicate",
        "object": "object",
        "polarity": "polarity",
        "epistemic_status": "epistemic_status",
    }[field]
    return getattr(gold, attribute), getattr(prediction, attribute)


def _pair_evidence_difference(
    gold: AtomicClaimV1, prediction: AtomicClaimV1
) -> tuple[list[tuple[str, str | None, str]], list[tuple[str, str | None, str]]]:
    gold_counts = _evidence_counts([gold])
    prediction_counts = _evidence_counts([prediction])
    return (
        sorted((gold_counts - prediction_counts).elements(), key=_span_sort_key),
        sorted((prediction_counts - gold_counts).elements(), key=_span_sort_key),
    )


def _span_sort_key(span: tuple[str, str | None, str]) -> tuple[str, str, str]:
    return span[0], span[1] or "", span[2]


def _span_record(span: tuple[str, str | None, str]) -> dict[str, object]:
    return {"source_id": span[0], "message_id": span[1], "quote": span[2]}


def _contains_gold_interval(
    prediction: AtomicClaimV1, gold: AtomicClaimV1
) -> bool:
    if gold.valid_from is None:
        start_contains = prediction.valid_from is None
        start_strict = False
    else:
        start_contains = (
            prediction.valid_from is not None
            and _compare_time(prediction.valid_from, gold.valid_from) <= 0
        )
        start_strict = (
            prediction.valid_from is not None
            and _compare_time(prediction.valid_from, gold.valid_from) < 0
        )
    if gold.valid_to is None:
        end_contains = prediction.valid_to is None
        end_strict = False
    else:
        end_contains = (
            prediction.valid_to is not None
            and _compare_time(prediction.valid_to, gold.valid_to) >= 0
        )
        end_strict = (
            prediction.valid_to is not None
            and _compare_time(prediction.valid_to, gold.valid_to) > 0
        )
    return start_contains and end_contains and (start_strict or end_strict)


def _parse_time(value: str) -> date | datetime:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value)


def _compare_time(left: str, right: str) -> int:
    left_value = _parse_time(left)
    right_value = _parse_time(right)
    if isinstance(left_value, datetime) and isinstance(right_value, datetime):
        left_value = left_value.astimezone(timezone.utc)
        right_value = right_value.astimezone(timezone.utc)
    else:
        left_value = left_value.date() if isinstance(left_value, datetime) else left_value
        right_value = right_value.date() if isinstance(right_value, datetime) else right_value
    return (left_value > right_value) - (left_value < right_value)


def _validate_predicate_families() -> None:
    mapped = set(PREDICATE_FAMILIES)
    if mapped != set(ALLOWED_PREDICATES):
        missing = sorted(set(ALLOWED_PREDICATES) - mapped)
        extra = sorted(mapped - set(ALLOWED_PREDICATES))
        raise AtomicFailureAnalysisError(
            f"predicate-family map mismatch; missing={missing}; extra={extra}"
        )


def _verify_hashes(
    root: Path, expected: Mapping[str, str], label: str
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative_path, expected_hash in expected.items():
        path = root / relative_path
        if not path.is_file():
            raise AtomicFailureAnalysisError(f"missing {label}: {relative_path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise AtomicFailureAnalysisError(
                f"{label} hash mismatch for {relative_path}: {actual_hash}"
            )
        actual[relative_path] = actual_hash
    return actual


def _require_empty_output_directory(output_path: Path) -> None:
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise AtomicFailureAnalysisError(
            f"refusing to overwrite non-empty output: {output_path}"
        )


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AtomicFailureAnalysisError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from None
            if not isinstance(record, dict):
                raise AtomicFailureAnalysisError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    return records


def _write_json(path: Path, record: object) -> None:
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _utc_text(value: datetime) -> str:
    if value.utcoffset() is None:
        raise AtomicFailureAnalysisError("generation time must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    arguments = parser.parse_args(argv)
    manifest = run_failure_analysis(
        repo_root=arguments.repo_root, output_dir=arguments.output_dir
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
