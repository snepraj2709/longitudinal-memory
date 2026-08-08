"""Deterministic comparison and decision record for the v5 development run."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Callable, Mapping

from .development_analysis import (
    DevelopmentAnalysisError,
    METRICS,
    _artifact_hashes,
    _metric_deltas,
    _published_metrics,
    _require_run,
)
from .failure_analysis import (
    FROZEN_INPUT_SHA256,
    PROTECTED_B1_SHA256,
    AtomicFailureAnalysisError,
    _git,
    _load_predictions,
    _load_sources,
    _read_json,
    _require_empty_output_directory,
    _sha256,
    _verify_hashes,
    _write_json,
    _write_jsonl,
)
from .gold import ATOMIC_GOLD_PATH, load_atomic_gold
from .scoring import ATOMIC_SCORING_VERSION, score_atomic_extraction


ANALYSIS_VERSION = "atomic-extraction-v5-development-analysis-v1"
V2_DIR = Path("results/phase3/atomic-extraction-v2")
V4_SMOKE_DIR = Path("results/phase3/atomic-extraction-v4-smoke-v1")
V4_FULL_DIR = Path("results/phase3/atomic-extraction-v4-development-v1")
V5_DIAGNOSTIC_DIRS = (
    Path("results/phase3/atomic-extraction-v5-diagnostic-v1"),
    Path("results/phase3/atomic-extraction-v5-diagnostic-v2"),
    Path("results/phase3/atomic-extraction-v5-diagnostic-v3"),
)
V5_SMOKE_DIR = Path("results/phase3/atomic-extraction-v5-smoke-v1")
V5_FULL_DIR = Path("results/phase3/atomic-extraction-v5-development-v1")
DEFAULT_OUTPUT_DIR = Path(
    "results/phase3/atomic-extraction-v5-development-analysis-v1"
)


def run_v5_development_analysis(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Reconcile v5, compare every metric, and record the promotion decision."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    try:
        _require_empty_output_directory(output_path)
    except AtomicFailureAnalysisError as error:
        raise DevelopmentAnalysisError(str(error)) from error

    frozen_hashes = _verify_hashes(root, FROZEN_INPUT_SHA256, "Phase 3 v2 input")
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")
    v4_smoke_run = _require_run(root / V4_SMOKE_DIR / "run.json", "completed")
    v4_full_run = _require_run(
        root / V4_FULL_DIR / "run.json", "failed_validation"
    )
    v5_smoke_run = _require_run(root / V5_SMOKE_DIR / "run.json", "completed")
    v5_full_run = _require_run(root / V5_FULL_DIR / "run.json", "completed")

    sources = _load_sources(root)
    sources_by_id = {source.source_id: source for source in sources}
    gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, sources)
    gold_by_id = {case.case_id: case for case in gold_cases}

    v2_predictions = _load_predictions(
        root / V2_DIR / "predictions.jsonl", sources_by_id
    )
    v4_smoke_predictions = _load_predictions(
        root / V4_SMOKE_DIR / "predictions.jsonl", sources_by_id
    )
    v5_smoke_predictions = _load_predictions(
        root / V5_SMOKE_DIR / "predictions.jsonl", sources_by_id
    )
    v5_full_predictions = _load_predictions(
        root / V5_FULL_DIR / "predictions.jsonl", sources_by_id
    )

    v2_scores = _reconcile_scores(
        root / V2_DIR / "scores.json", gold_cases, v2_predictions
    )
    v5_full_scores = _reconcile_scores(
        root / V5_FULL_DIR / "scores.json", gold_cases, v5_full_predictions
    )
    v4_case_ids = tuple(v4_smoke_run["case_ids"])
    v5_smoke_case_ids = tuple(v5_smoke_run["case_ids"])
    v4_smoke_scores = _reconcile_scores(
        root / V4_SMOKE_DIR / "scores.json",
        _gold_subset(gold_by_id, v4_case_ids),
        v4_smoke_predictions,
    )
    v5_smoke_scores = _reconcile_scores(
        root / V5_SMOKE_DIR / "scores.json",
        _gold_subset(gold_by_id, v5_smoke_case_ids),
        v5_smoke_predictions,
    )
    v5_on_v4_scores = score_atomic_extraction(
        _gold_subset(gold_by_id, v4_case_ids),
        _prediction_subset(v5_full_predictions, v4_case_ids),
    )
    v2_on_v4_scores = score_atomic_extraction(
        _gold_subset(gold_by_id, v4_case_ids),
        _prediction_subset(v2_predictions, v4_case_ids),
    )
    v2_on_v5_smoke_scores = score_atomic_extraction(
        _gold_subset(gold_by_id, v5_smoke_case_ids),
        _prediction_subset(v2_predictions, v5_smoke_case_ids),
    )

    metric_records = _metric_records(v2_scores, v5_full_scores)
    regression_metrics = [
        record["metric"]
        for record in metric_records
        if record["classification"] == "regression"
    ]
    improvement_metrics = [
        record["metric"]
        for record in metric_records
        if record["classification"] == "improvement"
    ]
    unchanged_metrics = [
        record["metric"]
        for record in metric_records
        if record["classification"] == "unchanged"
    ]
    acceptance_checks = {
        "full_run_structurally_complete": True,
        "claim_f1_not_below_v2": (
            v5_full_scores["micro_claim_f1"]["value"]
            >= v2_scores["micro_claim_f1"]["value"]
        ),
        "unsupported_memory_rate_not_above_v2": (
            v5_full_scores["unsupported_memory_rate"]["value"]
            <= v2_scores["unsupported_memory_rate"]["value"]
        ),
    }
    decision = (
        "accept_v5"
        if all(acceptance_checks.values())
        else "tune_another_candidate"
    )
    run_costs, total_cost = _run_costs(root)
    v5_raw_run = _read_json(root / V5_FULL_DIR / "run.json")
    normalization_count = sum(
        len(item.get("normalization_diagnostics", []))
        for item in v5_raw_run["attempts"]
    )

    summary: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "scoring_version": ATOMIC_SCORING_VERSION,
        "candidate_prompt_version": "atomic-extraction-v5",
        "candidate_status": "rejected_quality_regression",
        "decision": decision,
        "acceptance_checks": acceptance_checks,
        "full_comparison": {
            "case_ids": list(v5_full_run["case_ids"]),
            "v2_metrics": _published_metrics(v2_scores),
            "v5_metrics": _published_metrics(v5_full_scores),
            "metric_value_delta_v5_minus_v2": _metric_deltas(
                v5_full_scores, v2_scores
            ),
            "regression_metrics": regression_metrics,
            "improvement_metrics": improvement_metrics,
            "unchanged_metrics": unchanged_metrics,
            "v2_total_predicted_claims": v2_scores["total_predicted_claims"],
            "v5_total_predicted_claims": v5_full_scores["total_predicted_claims"],
            "v2_total_unsupported_claims": v2_scores[
                "total_unsupported_claims"
            ],
            "v5_total_unsupported_claims": v5_full_scores[
                "total_unsupported_claims"
            ],
        },
        "v4_comparable_smoke": {
            "case_ids": list(v4_case_ids),
            "v2_metrics": _published_metrics(v2_on_v4_scores),
            "v4_metrics": _published_metrics(v4_smoke_scores),
            "v5_metrics": _published_metrics(v5_on_v4_scores),
            "metric_value_delta_v5_minus_v4": _metric_deltas(
                v5_on_v4_scores, v4_smoke_scores
            ),
        },
        "v5_smoke_comparison": {
            "case_ids": list(v5_smoke_case_ids),
            "v2_metrics": _published_metrics(v2_on_v5_smoke_scores),
            "v5_metrics": _published_metrics(v5_smoke_scores),
            "metric_value_delta_v5_minus_v2": _metric_deltas(
                v5_smoke_scores, v2_on_v5_smoke_scores
            ),
        },
        "structural_comparison": {
            "v4_smoke": v4_smoke_run,
            "v4_full": v4_full_run,
            "v5_smoke": v5_smoke_run,
            "v5_full": v5_full_run,
            "v5_normalized_quote_count": normalization_count,
        },
        "costs": {
            "runs": run_costs,
            "total_usd": _decimal_text(total_cost),
        },
        "limitations": [
            "The v5 evaluation covers Maya only; no frozen test user was inspected.",
            "The source-span fallback can replace a generated quote with the full cited observation, which favors validity over span brevity.",
            "The v4 full run has no metrics because it stopped on validation after two successful cases.",
            "One completed run per candidate does not measure provider-run variance.",
        ],
    }
    findings = _render_findings(summary)

    output_path.mkdir(parents=True)
    _write_jsonl(output_path / "metric_comparison.jsonl", metric_records)
    _write_json(output_path / "summary.json", summary)
    (output_path / "findings.md").write_text(findings, encoding="utf-8")
    output_hashes = {
        name: _sha256(output_path / name)
        for name in ("metric_comparison.jsonl", "summary.json", "findings.md")
    }
    input_directories = (
        V2_DIR,
        V4_SMOKE_DIR,
        V4_FULL_DIR,
        *V5_DIAGNOSTIC_DIRS,
        V5_SMOKE_DIR,
        V5_FULL_DIR,
    )
    input_hashes: dict[str, str] = dict(frozen_hashes)
    for directory in input_directories:
        input_hashes.update(_artifact_hashes(root, directory))
    clock = now or (lambda: datetime.now(timezone.utc))
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "release_status": "development_analysis_complete",
        "review_status": "awaiting_sneha_review",
        "decision": decision,
        "generated_at_utc": _utc_text(clock()),
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "repository_worktree_state": (
            "clean" if not _git(root, "status", "--porcelain") else "dirty"
        ),
        "scope": {
            "development_user": "Maya",
            "full_case_count": len(v5_full_run["case_ids"]),
            "frozen_test_user_records_accessed": False,
            "api_calls_made_by_analysis": False,
        },
        "input_file_sha256": input_hashes,
        "protected_b1_file_sha256": protected_hashes,
        "output_file_sha256": output_hashes,
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _reconcile_scores(path, gold_cases, predictions):
    stored = _read_json(path)
    recalculated = score_atomic_extraction(gold_cases, predictions)
    if stored != recalculated:
        raise DevelopmentAnalysisError(f"scores do not reconcile: {path}")
    return stored


def _gold_subset(gold_by_id, case_ids):
    return tuple(gold_by_id[case_id] for case_id in case_ids)


def _prediction_subset(predictions, case_ids):
    return {case_id: predictions[case_id] for case_id in case_ids}


def _metric_records(v2_scores, v5_scores):
    records = []
    for metric in METRICS:
        v2_value = v2_scores[metric]["value"]
        v5_value = v5_scores[metric]["value"]
        delta = round(v5_value - v2_value, 6)
        lower_is_better = metric == "unsupported_memory_rate"
        if delta == 0:
            classification = "unchanged"
        elif (delta < 0) == lower_is_better:
            classification = "improvement"
        else:
            classification = "regression"
        records.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "metric": metric,
                "preferred_direction": "lower" if lower_is_better else "higher",
                "v2_value": v2_value,
                "v5_value": v5_value,
                "delta_v5_minus_v2": delta,
                "classification": classification,
            }
        )
    return records


def _run_costs(root: Path):
    directories = (
        V4_SMOKE_DIR,
        V4_FULL_DIR,
        *V5_DIAGNOSTIC_DIRS,
        V5_SMOKE_DIR,
        V5_FULL_DIR,
    )
    records = []
    total = Decimal("0")
    for directory in directories:
        run = _read_json(root / directory / "run.json")
        cost = Decimal(run["estimated_cost_usd"])
        total += cost
        records.append(
            {
                "result_directory": directory.as_posix(),
                "run_status": run["run_status"],
                "provider_requests_attempted": run[
                    "provider_requests_attempted"
                ],
                "cost_usd": _decimal_text(cost),
            }
        )
    return records, total


def _decimal_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def _render_findings(summary: Mapping[str, object]) -> str:
    full = summary["full_comparison"]
    v2 = full["v2_metrics"]
    v5 = full["v5_metrics"]
    return f"""# Step 3.4 v5 development result

V5 completed all ten Maya cases without a structural failure, fixing the blocker that stopped v4. It is not ready to replace v3: compared with v2, it found fewer gold claims and introduced unsupported claims.

## Full comparison

The main trade-off is clear. Claim F1 moved from {v2['micro_claim_f1']['value']:.6f} to {v5['micro_claim_f1']['value']:.6f}; precision moved from {v2['micro_claim_precision']['value']:.6f} to {v5['micro_claim_precision']['value']:.6f}, and recall from {v2['micro_claim_recall']['value']:.6f} to {v5['micro_claim_recall']['value']:.6f}. The unsupported-memory rate moved from {v2['unsupported_memory_rate']['value']:.6f} to {v5['unsupported_memory_rate']['value']:.6f}.

Evidence handling improved: span precision moved from {v2['provenance_span_precision']['value']:.6f} to {v5['provenance_span_precision']['value']:.6f}, and span recall from {v2['provenance_span_recall']['value']:.6f} to {v5['provenance_span_recall']['value']:.6f}. Valid-time accuracy improved as well. `metric_comparison.jsonl` records all thirteen metrics and classifies each change.

## Structural result

The v4 full run stopped on its third case. V5 completed both its three-case smoke and the full ten-case run. One evidence quote needed the source-span fallback; the run metadata records that repair and its field location.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not start Step 3.5 yet. The next iteration should keep strict structured output and source-backed evidence repair, while recovering claim coverage and preventing unsupported claims.
"""


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    manifest = run_v5_development_analysis()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
