"""Deterministic comparison and decision record for the v6 development run."""

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
from .v5_development_analysis import (
    V2_DIR,
    V4_FULL_DIR,
    V4_SMOKE_DIR,
    V5_DIAGNOSTIC_DIRS,
    V5_FULL_DIR,
    V5_SMOKE_DIR,
    _decimal_text,
    _gold_subset,
    _prediction_subset,
    _reconcile_scores,
    _run_costs,
)


ANALYSIS_VERSION = "atomic-extraction-v6-development-analysis-v1"
V6_SMOKE_DIR = Path("results/phase3/atomic-extraction-v6-smoke-v1")
V6_FULL_DIR = Path("results/phase3/atomic-extraction-v6-development-v1")
DEFAULT_OUTPUT_DIR = Path(
    "results/phase3/atomic-extraction-v6-development-analysis-v1"
)


def run_v6_development_analysis(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Reconcile v6 and compare every metric with the frozen candidates."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    try:
        _require_empty_output_directory(output_path)
    except AtomicFailureAnalysisError as error:
        raise DevelopmentAnalysisError(str(error)) from error

    frozen_hashes = _verify_hashes(root, FROZEN_INPUT_SHA256, "Phase 3 v2 input")
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")
    v4_smoke_run = _require_run(root / V4_SMOKE_DIR / "run.json", "completed")
    _require_run(root / V4_FULL_DIR / "run.json", "failed_validation")
    v5_smoke_run = _require_run(root / V5_SMOKE_DIR / "run.json", "completed")
    v5_full_run = _require_run(root / V5_FULL_DIR / "run.json", "completed")
    v6_smoke_run = _require_run(root / V6_SMOKE_DIR / "run.json", "completed")
    v6_full_run = _require_run(root / V6_FULL_DIR / "run.json", "completed")
    for run, version in (
        (v5_full_run, "atomic-extraction-v5"),
        (v6_smoke_run, "atomic-extraction-v6"),
        (v6_full_run, "atomic-extraction-v6"),
    ):
        if run["prompt_version"] != version:
            raise DevelopmentAnalysisError(f"unexpected prompt version: {version}")

    sources = _load_sources(root)
    sources_by_id = {source.source_id: source for source in sources}
    gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, sources)
    gold_by_id = {case.case_id: case for case in gold_cases}

    predictions = {
        "v2": _load_predictions(root / V2_DIR / "predictions.jsonl", sources_by_id),
        "v4_smoke": _load_predictions(
            root / V4_SMOKE_DIR / "predictions.jsonl", sources_by_id
        ),
        "v5": _load_predictions(
            root / V5_FULL_DIR / "predictions.jsonl", sources_by_id
        ),
        "v6_smoke": _load_predictions(
            root / V6_SMOKE_DIR / "predictions.jsonl", sources_by_id
        ),
        "v6": _load_predictions(
            root / V6_FULL_DIR / "predictions.jsonl", sources_by_id
        ),
    }
    scores = {
        "v2": _reconcile_scores(
            root / V2_DIR / "scores.json", gold_cases, predictions["v2"]
        ),
        "v5": _reconcile_scores(
            root / V5_FULL_DIR / "scores.json", gold_cases, predictions["v5"]
        ),
        "v6": _reconcile_scores(
            root / V6_FULL_DIR / "scores.json", gold_cases, predictions["v6"]
        ),
    }
    v4_case_ids = tuple(v4_smoke_run["case_ids"])
    v4_gold = _gold_subset(gold_by_id, v4_case_ids)
    v4_scores = _reconcile_scores(
        root / V4_SMOKE_DIR / "scores.json",
        v4_gold,
        predictions["v4_smoke"],
    )
    v6_on_v4_scores = score_atomic_extraction(
        v4_gold, _prediction_subset(predictions["v6"], v4_case_ids)
    )
    v6_smoke_case_ids = tuple(v6_smoke_run["case_ids"])
    if v6_smoke_case_ids != tuple(v5_smoke_run["case_ids"]):
        raise DevelopmentAnalysisError("v5 and v6 smoke case order differs")
    v6_smoke_scores = _reconcile_scores(
        root / V6_SMOKE_DIR / "scores.json",
        _gold_subset(gold_by_id, v6_smoke_case_ids),
        predictions["v6_smoke"],
    )

    comparisons = [
        *_metric_records("full", "v2", scores["v2"], scores["v6"]),
        *_metric_records("v4_smoke_subset", "v4", v4_scores, v6_on_v4_scores),
        *_metric_records("full", "v5", scores["v5"], scores["v6"]),
    ]
    acceptance_checks = {
        "full_run_structurally_complete": True,
        "claim_f1_not_below_v2": (
            scores["v6"]["micro_claim_f1"]["value"]
            >= scores["v2"]["micro_claim_f1"]["value"]
        ),
        "unsupported_memory_rate_not_above_v2": (
            scores["v6"]["unsupported_memory_rate"]["value"]
            <= scores["v2"]["unsupported_memory_rate"]["value"]
        ),
    }
    decision = "accept_v6" if all(acceptance_checks.values()) else "tune_another_candidate"

    prior_costs, prior_total = _run_costs(root)
    v6_costs = [_cost_record(root, path) for path in (V6_SMOKE_DIR, V6_FULL_DIR)]
    total_cost = prior_total + sum(
        (Decimal(item["cost_usd"]) for item in v6_costs), Decimal("0")
    )
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "scoring_version": ATOMIC_SCORING_VERSION,
        "candidate_prompt_version": "atomic-extraction-v6",
        "candidate_status": (
            "accepted" if decision == "accept_v6" else "rejected_quality_regression"
        ),
        "decision": decision,
        "acceptance_checks": acceptance_checks,
        "full_comparison": {
            "case_ids": list(v6_full_run["case_ids"]),
            "v2_metrics": _published_metrics(scores["v2"]),
            "v5_metrics": _published_metrics(scores["v5"]),
            "v6_metrics": _published_metrics(scores["v6"]),
            "v2_total_predicted_claims": scores["v2"]["total_predicted_claims"],
            "v5_total_predicted_claims": scores["v5"]["total_predicted_claims"],
            "v6_total_predicted_claims": scores["v6"]["total_predicted_claims"],
            "v2_total_unsupported_claims": scores["v2"]["total_unsupported_claims"],
            "v5_total_unsupported_claims": scores["v5"]["total_unsupported_claims"],
            "v6_total_unsupported_claims": scores["v6"]["total_unsupported_claims"],
        },
        "v4_comparable_smoke": {
            "case_ids": list(v4_case_ids),
            "v4_metrics": _published_metrics(v4_scores),
            "v6_metrics": _published_metrics(v6_on_v4_scores),
        },
        "v6_smoke": {
            "case_ids": list(v6_smoke_case_ids),
            "metrics": _published_metrics(v6_smoke_scores),
            "run": v6_smoke_run,
        },
        "v6_full_run": v6_full_run,
        "costs": {"runs": [*prior_costs, *v6_costs], "total_usd": _decimal_text(total_cost)},
        "limitations": [
            "The comparison covers Maya development cases only; no frozen test user was inspected.",
            "V4 has comparable metrics only for its completed three-case smoke because its full run stopped on validation.",
            "One completed run per prompt version does not measure provider-run variance.",
            "Exact-match scoring treats source-backed wording differences as claim errors.",
        ],
    }
    findings = _render_findings(summary)

    output_path.mkdir(parents=True)
    _write_jsonl(output_path / "metric_comparison.jsonl", comparisons)
    _write_json(output_path / "summary.json", summary)
    (output_path / "findings.md").write_text(findings, encoding="utf-8")
    output_hashes = {
        name: _sha256(output_path / name)
        for name in ("metric_comparison.jsonl", "summary.json", "findings.md")
    }
    input_dirs = (
        V2_DIR,
        V4_SMOKE_DIR,
        V4_FULL_DIR,
        *V5_DIAGNOSTIC_DIRS,
        V5_SMOKE_DIR,
        V5_FULL_DIR,
        V6_SMOKE_DIR,
        V6_FULL_DIR,
    )
    input_hashes = dict(frozen_hashes)
    for directory in input_dirs:
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
            "full_case_count": len(v6_full_run["case_ids"]),
            "frozen_test_user_records_accessed": False,
            "api_calls_made_by_analysis": False,
        },
        "input_file_sha256": input_hashes,
        "protected_b1_file_sha256": protected_hashes,
        "output_file_sha256": output_hashes,
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _metric_records(scope, baseline_name, baseline, candidate):
    records = []
    for metric in METRICS:
        baseline_value = baseline[metric]["value"]
        candidate_value = candidate[metric]["value"]
        delta = round(candidate_value - baseline_value, 6)
        lower_is_better = metric == "unsupported_memory_rate"
        if delta == 0:
            classification = "unchanged"
        elif (delta < 0) == lower_is_better:
            classification = "improvement"
        else:
            classification = "regression"
        records.append({
            "analysis_version": ANALYSIS_VERSION,
            "comparison_scope": scope,
            "baseline_prompt_version": baseline_name,
            "candidate_prompt_version": "v6",
            "metric": metric,
            "preferred_direction": "lower" if lower_is_better else "higher",
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "delta_candidate_minus_baseline": delta,
            "classification": classification,
        })
    return records


def _cost_record(root: Path, directory: Path) -> dict[str, object]:
    run = _read_json(root / directory / "run.json")
    return {
        "result_directory": directory.as_posix(),
        "run_status": run["run_status"],
        "provider_requests_attempted": run["provider_requests_attempted"],
        "cost_usd": run["estimated_cost_usd"],
    }


def _render_findings(summary: Mapping[str, object]) -> str:
    full = summary["full_comparison"]
    v2 = full["v2_metrics"]
    v5 = full["v5_metrics"]
    v6 = full["v6_metrics"]
    return f"""# Step 3.4 v6 development result

V6 completed all ten Maya cases. It improved several v5 metrics but still fails the promotion gates against v2.

## Full comparison

Claim F1 increased from {v5['micro_claim_f1']['value']:.6f} in v5 to {v6['micro_claim_f1']['value']:.6f} in v6. V2 remains higher at {v2['micro_claim_f1']['value']:.6f}. V6 precision was {v6['micro_claim_precision']['value']:.6f}, recall was {v6['micro_claim_recall']['value']:.6f}, and its unsupported-memory rate was {v6['unsupported_memory_rate']['value']:.6f}. V2 had no unsupported claims.

V6 had better valid-time accuracy than both earlier full runs. Its exact evidence scores remained above v2 but fell below v5. `metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v5.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not begin Step 3.5. The next prompt should retain v6's recovered precision and field accuracy while removing unsupported claims and recovering the remaining missed claims.

## Limits

This result covers Maya development data only. The v4 full run is incomplete, so v4 comparisons use its completed three-case smoke. A single run per prompt version does not measure provider variance.
"""


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    print(json.dumps(run_v6_development_analysis(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
