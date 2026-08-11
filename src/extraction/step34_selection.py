"""Freeze the reviewed Step 3.4 extractor selection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from .development_analysis import _published_metrics
from .failure_analysis import (
    PROTECTED_B1_SHA256,
    _git,
    _read_json,
    _require_empty_output_directory,
    _sha256,
    _verify_hashes,
    _write_json,
)


SELECTION_VERSION = "atomic-extraction-step34-selection-v1"
DEFAULT_OUTPUT_DIR = Path("results/phase3/atomic-extraction-step34-selection-v1")
V2_DIR = Path("results/phase3/atomic-extraction-v2")
V8_DIR = Path("results/phase3/atomic-extraction-v8-development-v1")
V9_DIR = Path("results/phase3/atomic-extraction-v9-development-v1")
ANALYSIS_DIRS = tuple(
    Path(f"results/phase3/atomic-extraction-v{version}-development-analysis-v1")
    for version in range(4, 10)
)


def freeze_step34_selection(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Write the approved selection and a hash-complete manifest."""

    root = Path(repo_root).resolve()
    output_path = _resolve(root, output_dir)
    _require_empty_output_directory(output_path)
    protected_hashes = _verify_hashes(root, PROTECTED_B1_SHA256, "B1 artifact")
    v2_scores = _read_json(root / V2_DIR / "scores.json")
    v8_scores = _read_json(root / V8_DIR / "scores.json")
    v9_scores = _read_json(root / V9_DIR / "scores.json")
    analyses = {
        directory.name: _read_json(root / directory / "summary.json")
        for directory in ANALYSIS_DIRS
    }
    if analyses[ANALYSIS_DIRS[-1].name]["decision"] != "tune_another_candidate":
        raise ValueError("v9 decision is not the reviewed rejection")

    decision = {
        "selection_version": SELECTION_VERSION,
        "step": "3.4",
        "step_status": "complete",
        "review_status": "approved_by_sneha",
        "reviewed_on": "2026-08-09",
        "selected_runtime_prompt_version": "atomic-extraction-v3",
        "quality_evidence_prompt_version": "atomic-extraction-v2",
        "selected_normalization_version": None,
        "selection": "retain_safe_runtime_default",
        "candidate_versions_reviewed": [
            "atomic-extraction-v4",
            "atomic-extraction-v5",
            "atomic-extraction-v6",
            "atomic-extraction-v7",
            "atomic-extraction-v8",
            "atomic-extraction-v9",
        ],
        "measured_results": {
            "v2_safety_baseline": {
                "metrics": _published_metrics(v2_scores),
                "total_unsupported_claims": v2_scores["total_unsupported_claims"],
            },
            "v8_best_f1_candidate": {
                "metrics": _published_metrics(v8_scores),
                "total_unsupported_claims": v8_scores["total_unsupported_claims"],
                "disposition": "rejected_known_unsupported_claims",
            },
            "v9_final_candidate": {
                "metrics": _published_metrics(v9_scores),
                "total_unsupported_claims": v9_scores["total_unsupported_claims"],
                "disposition": "rejected_regression",
            },
        },
        "selection_reason": (
            "V8 improved F1 but produced four known unsupported claims. V9 regressed "
            "on F1 and produced six. Sneha approved retaining the runtime default "
            "and the zero-unsupported v2 result as its measured safety evidence."
        ),
        "known_limitations": [
            "The v2 safety result missed 21 of 48 gold claims.",
            "The selected v3 runtime prompt has deterministic contract coverage but no separate paid full-development result.",
            "The candidate comparison covers Maya development cases only.",
            "One run per candidate does not measure provider-run variance.",
        ],
        "phase_3_5": {
            "ready_to_begin": True,
            "started": False,
            "requires_separate_paid_run_preflight": True,
        },
    }
    findings = _render_findings(decision)
    output_path.mkdir(parents=True)
    _write_json(output_path / "decision.json", decision)
    (output_path / "findings.md").write_text(findings, encoding="utf-8")

    input_paths = [
        V2_DIR / "run.json",
        V2_DIR / "scores.json",
        V8_DIR / "run.json",
        V8_DIR / "scores.json",
        V9_DIR / "run.json",
        V9_DIR / "scores.json",
        *(directory / "manifest.json" for directory in ANALYSIS_DIRS),
        *(directory / "summary.json" for directory in ANALYSIS_DIRS),
    ]
    input_hashes = {str(path): _sha256(root / path) for path in input_paths}
    output_hashes = {
        name: _sha256(output_path / name) for name in ("decision.json", "findings.md")
    }
    clock = now or (lambda: datetime.now(timezone.utc))
    manifest = {
        "selection_version": SELECTION_VERSION,
        "release_status": "step_3_4_complete",
        "review_status": "approved_by_sneha",
        "generated_at_utc": _utc_text(clock()),
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "repository_worktree_state": (
            "clean" if not _git(root, "status", "--porcelain") else "dirty"
        ),
        "scope": {
            "development_user": "Maya",
            "development_case_count": 10,
            "candidate_prompt_count": 6,
            "frozen_test_user_records_accessed": False,
            "api_calls_made_by_selection": False,
            "phase_3_5_started": False,
        },
        "input_file_sha256": input_hashes,
        "protected_b1_file_sha256": protected_hashes,
        "output_file_sha256": output_hashes,
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _render_findings(decision: dict[str, object]) -> str:
    measured = decision["measured_results"]
    v2 = measured["v2_safety_baseline"]
    v8 = measured["v8_best_f1_candidate"]
    v9 = measured["v9_final_candidate"]
    return f"""# Step 3.4 selection

Sneha approved keeping `atomic-extraction-v3` as the runtime prompt. The measured safety evidence remains the v2 run, which scored {v2['metrics']['micro_claim_f1']['value']:.6f} F1 with no unsupported claims.

V8 reached {v8['metrics']['micro_claim_f1']['value']:.6f} F1 but produced four unsupported claims. V9 fell to {v9['metrics']['micro_claim_f1']['value']:.6f} F1 and produced six. Neither candidate is promoted.

Step 3.4 is complete. Step 3.5 may begin after its own paid-run preflight and approval. It has not started yet.

## Known limits

- The v2 result missed 21 of 48 gold claims.
- V3 has contract-test coverage but no separate paid full-development result.
- Candidate tuning used Maya development cases only. Frozen test users were not inspected.
- Each candidate was run once, so the results do not measure provider variance.
"""


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    print(json.dumps(freeze_step34_selection(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
