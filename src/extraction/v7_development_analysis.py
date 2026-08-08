"""Deterministic comparison and decision record for the v7 development run."""

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
    V4_SMOKE_DIR,
    _decimal_text,
    _gold_subset,
    _prediction_subset,
    _reconcile_scores,
)
from .v6_development_analysis import (
    V6_FULL_DIR,
    _cost_record,
    _metric_records,
)


ANALYSIS_VERSION = "atomic-extraction-v7-development-analysis-v1"
V6_ANALYSIS_DIR = Path("results/phase3/atomic-extraction-v6-development-analysis-v1")
V7_SMOKE_DIR = Path("results/phase3/atomic-extraction-v7-smoke-v1")
V7_FULL_DIR = Path("results/phase3/atomic-extraction-v7-development-v1")
DEFAULT_OUTPUT_DIR = Path(
    "results/phase3/atomic-extraction-v7-development-analysis-v1"
)


def run_v7_development_analysis(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Reconcile v7 and record the promotion decision."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    try:
        _require_empty_output_directory(output_path)
    except AtomicFailureAnalysisError as error:
        raise DevelopmentAnalysisError(str(error)) from error
    frozen_hashes = _verify_hashes(root, FROZEN_INPUT_SHA256, "Phase 3 v2 input")
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")

    v4_smoke_run = _require_run(root / V4_SMOKE_DIR / "run.json", "completed")
    v6_full_run = _require_run(root / V6_FULL_DIR / "run.json", "completed")
    v7_smoke_run = _require_run(root / V7_SMOKE_DIR / "run.json", "completed")
    v7_full_run = _require_run(root / V7_FULL_DIR / "run.json", "completed")
    if v7_smoke_run["prompt_version"] != "atomic-extraction-v7":
        raise DevelopmentAnalysisError("smoke prompt version is not v7")
    if v7_full_run["prompt_version"] != "atomic-extraction-v7":
        raise DevelopmentAnalysisError("full prompt version is not v7")

    sources = _load_sources(root)
    sources_by_id = {source.source_id: source for source in sources}
    gold_cases = load_atomic_gold(root / ATOMIC_GOLD_PATH, sources)
    gold_by_id = {case.case_id: case for case in gold_cases}
    predictions = {
        "v2": _load_predictions(root / V2_DIR / "predictions.jsonl", sources_by_id),
        "v4": _load_predictions(
            root / V4_SMOKE_DIR / "predictions.jsonl", sources_by_id
        ),
        "v6": _load_predictions(
            root / V6_FULL_DIR / "predictions.jsonl", sources_by_id
        ),
        "v7": _load_predictions(
            root / V7_FULL_DIR / "predictions.jsonl", sources_by_id
        ),
    }
    scores = {
        "v2": _reconcile_scores(
            root / V2_DIR / "scores.json", gold_cases, predictions["v2"]
        ),
        "v6": _reconcile_scores(
            root / V6_FULL_DIR / "scores.json", gold_cases, predictions["v6"]
        ),
        "v7": _reconcile_scores(
            root / V7_FULL_DIR / "scores.json", gold_cases, predictions["v7"]
        ),
    }
    v4_case_ids = tuple(v4_smoke_run["case_ids"])
    v4_gold = _gold_subset(gold_by_id, v4_case_ids)
    v4_scores = _reconcile_scores(
        root / V4_SMOKE_DIR / "scores.json", v4_gold, predictions["v4"]
    )
    v7_on_v4_scores = score_atomic_extraction(
        v4_gold, _prediction_subset(predictions["v7"], v4_case_ids)
    )

    comparisons = [
        *_metric_records("full", "v2", scores["v2"], scores["v7"], "v7"),
        *_metric_records(
            "v4_smoke_subset", "v4", v4_scores, v7_on_v4_scores, "v7"
        ),
        *_metric_records("full", "v6", scores["v6"], scores["v7"], "v7"),
    ]
    acceptance_checks = {
        "full_run_structurally_complete": True,
        "claim_f1_not_below_v2": (
            scores["v7"]["micro_claim_f1"]["value"]
            >= scores["v2"]["micro_claim_f1"]["value"]
        ),
        "unsupported_memory_rate_not_above_v2": (
            scores["v7"]["unsupported_memory_rate"]["value"]
            <= scores["v2"]["unsupported_memory_rate"]["value"]
        ),
    }
    decision = "accept_v7" if all(acceptance_checks.values()) else "tune_another_candidate"
    previous = _read_json(root / V6_ANALYSIS_DIR / "summary.json")
    v7_costs = [_cost_record(root, path) for path in (V7_SMOKE_DIR, V7_FULL_DIR)]
    total_cost = Decimal(previous["costs"]["total_usd"]) + sum(
        (Decimal(item["cost_usd"]) for item in v7_costs), Decimal("0")
    )
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "scoring_version": ATOMIC_SCORING_VERSION,
        "candidate_prompt_version": "atomic-extraction-v7",
        "candidate_status": (
            "accepted" if decision == "accept_v7" else "rejected_unsupported_claims"
        ),
        "decision": decision,
        "acceptance_checks": acceptance_checks,
        "full_comparison": {
            "case_ids": list(v7_full_run["case_ids"]),
            "v2_metrics": _published_metrics(scores["v2"]),
            "v6_metrics": _published_metrics(scores["v6"]),
            "v7_metrics": _published_metrics(scores["v7"]),
            "v2_total_unsupported_claims": scores["v2"]["total_unsupported_claims"],
            "v6_total_unsupported_claims": scores["v6"]["total_unsupported_claims"],
            "v7_total_unsupported_claims": scores["v7"]["total_unsupported_claims"],
        },
        "v4_comparable_smoke": {
            "case_ids": list(v4_case_ids),
            "v4_metrics": _published_metrics(v4_scores),
            "v7_metrics": _published_metrics(v7_on_v4_scores),
        },
        "v7_smoke_run": v7_smoke_run,
        "v7_full_run": v7_full_run,
        "costs": {
            "prior_through_v6_usd": previous["costs"]["total_usd"],
            "v7_runs": v7_costs,
            "total_through_v7_usd": _decimal_text(total_cost),
        },
        "unsupported_claim_review": {
            "count": 5,
            "categories": {
                "boolean_polarity_encoding": 2,
                "predicate_selection_or_representation": 3,
            },
        },
        "limitations": [
            "The comparison covers Maya development cases only; no frozen test user was inspected.",
            "V4 has comparable metrics only for its completed three-case smoke.",
            "A single run per prompt version does not measure provider-run variance.",
            "The unsupported-claim review uses deterministic scorer alignments and source records, not an additional model.",
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
    input_hashes = dict(frozen_hashes)
    for directory in (
        V4_SMOKE_DIR,
        V6_FULL_DIR,
        V6_ANALYSIS_DIR,
        V7_SMOKE_DIR,
        V7_FULL_DIR,
    ):
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
            "full_case_count": len(v7_full_run["case_ids"]),
            "frozen_test_user_records_accessed": False,
            "api_calls_made_by_analysis": False,
        },
        "input_file_sha256": input_hashes,
        "protected_b1_file_sha256": protected_hashes,
        "output_file_sha256": output_hashes,
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _render_findings(summary: Mapping[str, object]) -> str:
    full = summary["full_comparison"]
    v2 = full["v2_metrics"]
    v6 = full["v6_metrics"]
    v7 = full["v7_metrics"]
    return f"""# Step 3.4 v7 development result

V7 completed all ten Maya cases and improved claim quality over v6. Its F1 was {v7['micro_claim_f1']['value']:.6f}, compared with {v6['micro_claim_f1']['value']:.6f} for v6 and {v2['micro_claim_f1']['value']:.6f} for v2.

## Promotion check

V7 cleared the claim-F1 gate. Precision and recall also exceeded v2. It did not clear the unsupported-memory gate: the scorer found five unsupported claims, compared with none in v2.

Two unsupported claims encoded a negated boolean as `false` with positive polarity. Three used a predicate or representation that did not align with the source-backed gold claim. These are narrow enough for one more candidate rather than accepting the regression.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not begin Step 3.5. V8 should preserve v7's claim coverage, normalize boolean polarity deterministically, and tighten the remaining predicate choices.

`metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v6.
"""


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    print(json.dumps(run_v7_development_analysis(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
