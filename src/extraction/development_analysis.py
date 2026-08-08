"""Deterministic analysis for the Step 3.4 development candidate."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Mapping

from .failure_analysis import (
    AtomicFailureAnalysisError,
    FROZEN_INPUT_SHA256,
    PREDICATE_FAMILIES,
    PROTECTED_B1_SHA256,
    STABLE_CATEGORIES,
    _git,
    _load_predictions,
    _load_sources,
    _read_json,
    _read_jsonl,
    _require_empty_output_directory,
    _sha256,
    _verify_hashes,
    _write_json,
    _write_jsonl,
    classify_valid_time_mismatch,
)
from .gold import ATOMIC_GOLD_PATH, load_atomic_gold
from .scoring import (
    ATOMIC_SCORING_VERSION,
    _field_matches,
    _match_claims,
    score_atomic_case,
    score_atomic_extraction,
)


ANALYSIS_VERSION = "atomic-extraction-v4-development-analysis-v1"
BASELINE_DIR = Path("results/phase3/atomic-extraction-v2")
SMOKE_DIR = Path("results/phase3/atomic-extraction-v4-smoke-v1")
FULL_RUN_DIR = Path("results/phase3/atomic-extraction-v4-development-v1")
DEFAULT_OUTPUT_DIR = Path(
    "results/phase3/atomic-extraction-v4-development-analysis-v1"
)
METRICS = (
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


class DevelopmentAnalysisError(RuntimeError):
    """Raised when the Step 3.4 artifacts do not reconcile."""


def run_development_analysis(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Compare the valid smoke with v2 and document the blocked full run."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    try:
        _require_empty_output_directory(output_path)
    except AtomicFailureAnalysisError as error:
        raise DevelopmentAnalysisError(str(error)) from error
    frozen_hashes = _verify_hashes(root, FROZEN_INPUT_SHA256, "Phase 3 v2 input")
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")

    smoke_run = _require_run(root / SMOKE_DIR / "run.json", "completed")
    full_run = _require_run(
        root / FULL_RUN_DIR / "run.json", "failed_validation"
    )
    if smoke_run["prompt_version"] != "atomic-extraction-v4":
        raise DevelopmentAnalysisError("smoke prompt version is not v4")
    if full_run["prompt_version"] != "atomic-extraction-v4":
        raise DevelopmentAnalysisError("full-run prompt version is not v4")
    if full_run["failure_stage"] != "validation":
        raise DevelopmentAnalysisError("full run did not stop at validation")

    sources = _load_sources(root)
    sources_by_id = {source.source_id: source for source in sources}
    gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, sources)
    gold_by_id = {case.case_id: case for case in gold_cases}
    smoke_case_ids = tuple(smoke_run["case_ids"])
    smoke_gold = tuple(gold_by_id[case_id] for case_id in smoke_case_ids)

    smoke_predictions = _load_predictions(
        root / SMOKE_DIR / "predictions.jsonl", sources_by_id
    )
    smoke_scores = score_atomic_extraction(smoke_gold, smoke_predictions)
    stored_smoke_scores = _read_json(root / SMOKE_DIR / "scores.json")
    if smoke_scores != stored_smoke_scores:
        raise DevelopmentAnalysisError("smoke scores do not reconcile")

    baseline_predictions = _load_predictions(
        root / BASELINE_DIR / "predictions.jsonl", sources_by_id
    )
    baseline_subset = {
        case_id: baseline_predictions[case_id] for case_id in smoke_case_ids
    }
    baseline_smoke_scores = score_atomic_extraction(smoke_gold, baseline_subset)
    baseline_full_scores = _read_json(root / BASELINE_DIR / "scores.json")

    case_records = [
        _analyse_smoke_case(
            gold_case,
            smoke_predictions[gold_case.case_id],
            sources_by_id[gold_case.source_id],
        )
        for gold_case in smoke_gold
    ]
    smoke_failure_summary = _build_smoke_failure_summary(case_records)
    if (
        smoke_failure_summary["category_counts"]["unsupported_claim"]
        != smoke_scores["total_unsupported_claims"]
    ):
        raise DevelopmentAnalysisError(
            "smoke unsupported-claim analysis does not reconcile"
        )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "candidate_prompt_version": "atomic-extraction-v4",
        "scoring_version": ATOMIC_SCORING_VERSION,
        "candidate_status": "blocked_failed_validation",
        "intervention": (
            "One prompt paragraph asks for clause-level coverage, minimal exact "
            "evidence spans, boolean/polarity separation, explicit denial status, "
            "and ISO valid-time boundaries."
        ),
        "full_run": {
            **full_run,
            "metrics": None,
            "metrics_unavailable_reason": (
                "The frozen scorer rejects incomplete executions. The run stopped "
                "on a validation failure after two successful cases."
            ),
        },
        "smoke_comparison": {
            "case_ids": list(smoke_case_ids),
            "baseline_prompt_version": "atomic-extraction-v2",
            "candidate_metrics": _published_metrics(smoke_scores),
            "baseline_metrics": _published_metrics(baseline_smoke_scores),
            "metric_value_delta_candidate_minus_baseline": _metric_deltas(
                smoke_scores, baseline_smoke_scores
            ),
            "candidate_total_gold_claims": smoke_scores["total_gold_claims"],
            "candidate_total_predicted_claims": smoke_scores[
                "total_predicted_claims"
            ],
            "candidate_total_unsupported_claims": smoke_scores[
                "total_unsupported_claims"
            ],
        },
        "baseline_full_metrics": _published_metrics(baseline_full_scores),
        "smoke_failure_analysis": smoke_failure_summary,
        "limitations": [
            "The candidate has no full-suite metrics because the full run is incomplete.",
            "The three-case smoke is useful for structural checks, not release promotion.",
            "The sanitized validation record does not retain invalid model output, so the exact rejected field cannot be classified.",
        ],
    }
    findings = _render_findings(summary)

    output_path.mkdir(parents=True)
    _write_jsonl(output_path / "case_failures.jsonl", case_records)
    _write_json(output_path / "summary.json", summary)
    (output_path / "findings.md").write_text(findings, encoding="utf-8")
    output_hashes = {
        name: _sha256(output_path / name)
        for name in ("case_failures.jsonl", "summary.json", "findings.md")
    }
    clock = now or (lambda: datetime.now(timezone.utc))
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "release_status": "blocked_failed_validation",
        "review_status": "not_ready_for_review",
        "generated_at_utc": _utc_text(clock()),
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "repository_worktree_state": (
            "clean" if not _git(root, "status", "--porcelain") else "dirty"
        ),
        "scope": {
            "development_user": "Maya",
            "smoke_case_count": len(smoke_case_ids),
            "full_planned_case_count": full_run["planned_case_count"],
            "full_successful_case_count": full_run["successful_case_count"],
            "full_failed_case_count": full_run["failed_case_count"],
            "frozen_test_user_records_accessed": False,
            "api_calls_made_by_analysis": False,
        },
        "input_file_sha256": {
            **frozen_hashes,
            **_artifact_hashes(root, SMOKE_DIR),
            **_artifact_hashes(root, FULL_RUN_DIR),
        },
        "protected_b1_file_sha256": protected_hashes,
        "output_file_sha256": output_hashes,
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _require_run(path: Path, expected_status: str) -> dict[str, object]:
    record = _read_json(path)
    if not isinstance(record, dict) or record.get("run_status") != expected_status:
        raise DevelopmentAnalysisError(f"unexpected run status in {path}")
    failures = _read_jsonl(path.parent / "failures.jsonl")
    attempts = record.get("attempts")
    case_order = record.get("case_order")
    if not isinstance(attempts, list) or not isinstance(case_order, list):
        raise DevelopmentAnalysisError(f"invalid run structure in {path}")
    failure_stage = failures[0]["failure_stage"] if failures else None
    return {
        "run_status": record["run_status"],
        "prompt_version": record["prompt_version"],
        "prompt_sha256": record["prompt_sha256"],
        "requested_model": record["requested_model"],
        "resolved_model": record["resolved_model"],
        "provider_requests_attempted": record["provider_requests_attempted"],
        "planned_case_count": record["planned_request_count"],
        "successful_case_count": record["successful_cases"],
        "failed_case_count": record["failed_cases"],
        "remaining_case_count": record["remaining_cases"],
        "resume_count": record["resume_count"],
        "budget_input_tokens": record["budget_input_tokens"],
        "budget_output_tokens": record["budget_output_tokens"],
        "estimated_cost_usd": record["estimated_cost_usd"],
        "hard_cost_cap_usd": record["hard_cost_cap_usd"],
        "case_ids": [item["case_id"] for item in case_order],
        "source_ids": [item["source_id"] for item in case_order],
        "failure_stage": failure_stage,
        "failure_case_id": failures[0]["case_id"] if failures else None,
        "failure_source_id": failures[0]["source_id"] if failures else None,
        "failure_error": failures[0]["error"] if failures else None,
    }


def _published_metrics(scores: Mapping[str, object]) -> dict[str, object]:
    return {metric: scores[metric] for metric in METRICS}


def _metric_deltas(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for metric in METRICS:
        candidate_value = candidate[metric]["value"]
        baseline_value = baseline[metric]["value"]
        deltas[metric] = (
            None
            if candidate_value is None or baseline_value is None
            else round(candidate_value - baseline_value, 6)
        )
    return deltas


def _analyse_smoke_case(gold_case, predictions, source) -> dict[str, object]:
    gold = tuple(gold_case.expected_claims)
    predicted = tuple(predictions)
    exact = set(_match_claims(gold, predicted, "exact"))
    aligned = tuple(_match_claims(gold, predicted, "aligned"))
    supported_prediction_indexes = {
        prediction_index
        for _, prediction_index in _match_claims(gold, predicted, "supported")
    }
    issues: list[dict[str, object]] = []

    def add(category, gold_claim=None, predicted_claim=None, detail=None):
        claim = gold_claim or predicted_claim
        issues.append(
            {
                "issue_id": f"{gold_case.case_id}_issue_{len(issues) + 1:03d}",
                "category": category,
                "predicate": claim.predicate,
                "predicate_family": PREDICATE_FAMILIES[claim.predicate],
                "gold_claim_id": gold_claim.claim_id if gold_claim else None,
                "predicted_claim_id": (
                    predicted_claim.claim_id if predicted_claim else None
                ),
                "detail": detail,
            }
        )

    aligned_gold = set()
    aligned_predictions = set()
    for gold_index, prediction_index in aligned:
        aligned_gold.add(gold_index)
        aligned_predictions.add(prediction_index)
        gold_claim = gold[gold_index]
        prediction = predicted[prediction_index]
        if (gold_index, prediction_index) not in exact:
            if prediction_index in supported_prediction_indexes:
                add(
                    "extra_source_backed_claim",
                    gold_claim,
                    prediction,
                    "The aligned source-backed claim is not an exact match.",
                )
            else:
                add(
                    "missed_gold_claim",
                    gold_claim=gold_claim,
                    detail="No supported exact prediction represents this claim.",
                )
                add(
                    "unsupported_claim",
                    predicted_claim=prediction,
                    detail="The prediction has no scorer-supported gold alignment.",
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
                add(category, gold_claim, prediction, f"{field} differs.")
        if not matches["valid_time"]:
            add(
                classify_valid_time_mismatch(gold_claim, prediction),
                gold_claim,
                prediction,
                "Valid-time boundaries differ.",
            )
        gold_evidence = {
            (item.source_id, item.message_id, item.quote)
            for item in gold_claim.evidence
        }
        predicted_evidence = {
            (item.source_id, item.message_id, item.quote)
            for item in prediction.evidence
        }
        if gold_evidence != predicted_evidence:
            add(
                "evidence_span_mismatch",
                gold_claim,
                prediction,
                "Exact evidence spans differ.",
            )

    for index, claim in enumerate(gold):
        if index not in aligned_gold:
            add("missed_gold_claim", gold_claim=claim)
    for index, claim in enumerate(predicted):
        if index not in aligned_predictions:
            category = (
                "extra_source_backed_claim"
                if index in supported_prediction_indexes
                else "unsupported_claim"
            )
            add(category, predicted_claim=claim)

    score = score_atomic_case(gold_case, predicted)
    category_counts = Counter(issue["category"] for issue in issues)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": gold_case.case_id,
        "source_id": gold_case.source_id,
        "source_type": source.source_type,
        "score": score,
        "category_counts": {
            category: category_counts.get(category, 0)
            for category in STABLE_CATEGORIES
        },
        "issues": issues,
    }


def _build_smoke_failure_summary(records) -> dict[str, object]:
    category_counts = Counter()
    by_source_type: dict[str, Counter] = {}
    by_predicate_family: dict[str, Counter] = {}
    by_category: dict[str, Counter] = {}
    for record in records:
        for issue in record["issues"]:
            category = issue["category"]
            family = issue["predicate_family"]
            category_counts[category] += 1
            by_source_type.setdefault(record["source_type"], Counter())[category] += 1
            by_predicate_family.setdefault(family, Counter())[category] += 1
            by_category.setdefault(category, Counter())[family] += 1

    def render(entries):
        return {
            key: {
                "issue_count": sum(counts.values()),
                "breakdown": dict(sorted(counts.items())),
            }
            for key, counts in sorted(entries.items())
        }

    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_prompt_version": "atomic-extraction-v4",
        "case_count": len(records),
        "category_counts": {
            category: category_counts.get(category, 0)
            for category in STABLE_CATEGORIES
        },
        "by_source_type": render(by_source_type),
        "by_predicate_family": render(by_predicate_family),
        "by_category": render(by_category),
    }


def _artifact_hashes(root: Path, directory: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted((root / directory).iterdir())
        if path.is_file()
    }


def _render_findings(summary: Mapping[str, object]) -> str:
    full = summary["full_run"]
    smoke = summary["smoke_comparison"]
    candidate = smoke["candidate_metrics"]
    baseline = smoke["baseline_metrics"]
    failures = summary["smoke_failure_analysis"]
    source_types = ", ".join(
        f"`{name}` ({_issue_count_text(data['issue_count'])})"
        for name, data in failures["by_source_type"].items()
    )
    families = ", ".join(
        f"`{name}` ({_issue_count_text(data['issue_count'])})"
        for name, data in failures["by_predicate_family"].items()
    )
    categories = ", ".join(
        f"`{name}` ({count})"
        for name, count in failures["category_counts"].items()
        if count
    )
    return f"""# Step 3.4 extraction development result

The v4 candidate is blocked. Its full Maya run stopped on `{full['failure_case_id']}` because the model response failed deterministic validation. Two of ten cases completed before the stop. The protocol does not allow a resume or a score for this failure type, so the accepted v3 prompt remains the runtime default.

## What changed

The candidate adds one prompt paragraph. It asks the model to cover independent clauses, use shorter exact evidence spans, keep boolean objects separate from polarity, mark explicit denials, and fill valid-time boundaries when the source gives a date.

## Smoke result

The three-case smoke completed and produced {smoke['candidate_total_predicted_claims']} claims for {smoke['candidate_total_gold_claims']} gold claims. Against the same three v2 cases, claim F1 moved from {baseline['micro_claim_f1']['value']:.6f} to {candidate['micro_claim_f1']['value']:.6f}. Evidence precision moved from {baseline['provenance_span_precision']['value']:.6f} to {candidate['provenance_span_precision']['value']:.6f}, and evidence recall moved from {baseline['provenance_span_recall']['value']:.6f} to {candidate['provenance_span_recall']['value']:.6f}. Unsupported-memory rate moved from {baseline['unsupported_memory_rate']['value']:.6f} to {candidate['unsupported_memory_rate']['value']:.6f}.

This smoke is not a release comparison. It covers only three cases and cannot replace the incomplete full run.

## Failure breakdown

For the valid smoke, issues by source type were {source_types}. Predicate-family issues were {families}. The active failure categories were {categories}.

The full-run failure record is intentionally sanitized. It identifies the case and validation stage but does not keep the invalid response, so this analysis cannot name the rejected field. `summary.json` contains every smoke metric, the v2 deltas, and the source-type, predicate-family, and category breakdowns. `case_failures.jsonl` contains the claim-level smoke audit trail.

## Decision

Do not promote v4 and do not start Step 3.5. A future, separately authorized development iteration would need a new prompt version and a fresh bounded run; this failed full run must remain unchanged.
"""


def _issue_count_text(count: int) -> str:
    return f"{count} issue" if count == 1 else f"{count} issues"


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    manifest = run_development_analysis()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
