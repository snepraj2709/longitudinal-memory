"""Zero-provider pre-run gates for Qwen B0-B7 execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .data_foundation_readiness import build_readiness_audit
from .qwen_extraction_quality_gate import CONFIG_PATH as EXTRACTION_GATE_CONFIG_PATH
from .scaled_release import ScaledReleaseError, validate_scaled_release


DEFAULT_EXTRACTION_GATE_ROOT = Path("results/evaluation/qwen3-8b-vllm-extraction-gate-v1")
DEFAULT_CONTEXT_AUDIT_ROOT = Path("results/evaluation/qwen3-8b-vllm-dev-v1/context-evidence-audit")
DEFAULT_CONTEXTS_PATH = Path("results/evaluation/qwen3-8b-vllm-dev-v1/contexts/contexts.jsonl")
DEFAULT_MATERIALIZATION_EXCLUSIONS = Path("results/evaluation/qwen3-8b-vllm-dev-v1/contexts/exclusions.json")
GATE_NAMES = ("primary_gate", "holdout_gate")


class QwenPreRunGateError(RuntimeError):
    """Raised when a provider run is blocked by deterministic pre-run gates."""


@dataclass(frozen=True)
class GateCheck:
    id: str
    status: str
    severity: str
    message: str
    evidence: Mapping[str, object]


def build_pre_run_gate_report(
    repo_root: str | Path,
    *,
    extraction_gate_root: str | Path = DEFAULT_EXTRACTION_GATE_ROOT,
    context_audit_root: str | Path = DEFAULT_CONTEXT_AUDIT_ROOT,
    contexts_path: str | Path = DEFAULT_CONTEXTS_PATH,
    materialization_exclusions: str | Path = DEFAULT_MATERIALIZATION_EXCLUSIONS,
) -> Mapping[str, object]:
    """Return a provider-independent readiness report for Qwen B0-B7 execution."""

    root = Path(repo_root).resolve()
    checks = [
        _scaled_validation_check(root),
        _scaled_review_check(root),
        _scaled_evidence_check(root),
        *_extraction_gate_checks(root, Path(extraction_gate_root)),
        _context_audit_check(root, Path(context_audit_root)),
        _materialization_check(root, Path(contexts_path), Path(materialization_exclusions)),
    ]
    blockers = [check for check in checks if check.severity == "blocker" and check.status != "passed"]
    return {
        "schema_version": "qwen_pre_run_gates_v1",
        "status": "passed" if not blockers else "blocked",
        "provider_request_count": 0,
        "checks": [asdict(check) for check in checks],
        "blocker_count": len(blockers),
        "next_action": (
            "provider_run_allowed"
            if not blockers
            else "run_missing_or_failed_zero_cost_gates_before_provider_execution"
        ),
    }


def assert_qwen_pre_run_gates(repo_root: str | Path, **kwargs: object) -> Mapping[str, object]:
    """Raise if any pre-run blocker remains."""

    report = build_pre_run_gate_report(repo_root, **kwargs)
    if report["status"] != "passed":
        blockers = [
            f"{check['id']}: {check['message']}"
            for check in report["checks"]  # type: ignore[index]
            if check["severity"] == "blocker" and check["status"] != "passed"
        ]
        raise QwenPreRunGateError("; ".join(blockers))
    return report


def _scaled_validation_check(root: Path) -> GateCheck:
    try:
        report = validate_scaled_release(root)
    except ScaledReleaseError as error:
        return _check(
            "scaled_validator",
            False,
            "scaled-v1 validator must pass before any provider run",
            {"errors": list(error.errors)},
        )
    return _check(
        "scaled_validator",
        True,
        "scaled-v1 validator passed",
        {"dataset_sha256": report.dataset_sha256, "human_review_status": report.human_review_status},
    )


def _scaled_review_check(root: Path) -> GateCheck:
    audit = build_readiness_audit(root)
    scaled = audit["scaled_v1"]  # type: ignore[index]
    pending_gold = int(scaled["pending_gold_rows"])  # type: ignore[index]
    pending_review = int(scaled["pending_review_queue_rows"])  # type: ignore[index]
    mismatch = bool(scaled["manifest_row_review_mismatch"])  # type: ignore[index]
    return _check(
        "scaled_review_status",
        pending_gold == 0 and pending_review == 0 and not mismatch,
        "scaled-v1 row review statuses must match manifest approval",
        {
            "pending_gold_rows": pending_gold,
            "pending_review_queue_rows": pending_review,
            "manifest_row_review_mismatch": mismatch,
        },
    )


def _scaled_evidence_check(root: Path) -> GateCheck:
    audit = build_readiness_audit(root)
    scaled = audit["scaled_v1"]  # type: ignore[index]
    evidence = scaled["evidence_integrity"]["totals"]  # type: ignore[index]
    references = scaled["reference_integrity"]["totals"]  # type: ignore[index]
    failures = {
        key: evidence.get(key, 0)
        for key in ("missing_source_message_refs", "bad_exact_quotes", "cross_user_evidence", "empty_or_malformed")
    }
    failures.update({
        key: references.get(key, 0)
        for key in ("missing_claim_refs", "missing_event_refs", "cross_user_claim_refs", "cross_user_event_refs")
    })
    return _check(
        "scaled_evidence_integrity",
        all(value == 0 for value in failures.values()),
        "scaled-v1 evidence, claim, event, and user-scope refs must be clean",
        failures,
    )


def _extraction_gate_checks(root: Path, extraction_gate_root: Path) -> list[GateCheck]:
    config = _json(root / EXTRACTION_GATE_CONFIG_PATH)
    checks: list[GateCheck] = []
    for gate_name in GATE_NAMES:
        gate = config.get(gate_name)
        expected = len(gate.get("source_ids", [])) if isinstance(gate, Mapping) else 0
        manifest_path = root / extraction_gate_root / gate_name / "run-manifest.json"
        if not manifest_path.is_file():
            checks.append(_check(
                f"qwen_extraction_{gate_name}",
                False,
                f"{gate_name} extraction gate must be run and pass",
                {"missing": str(manifest_path), "expected_request_count": expected},
            ))
            continue
        manifest = _json(manifest_path)
        extraction = manifest.get("extraction") if isinstance(manifest.get("extraction"), Mapping) else {}
        scores = manifest.get("scores") if isinstance(manifest.get("scores"), Mapping) else {}
        passed = (
            manifest.get("schema_version") == "qwen_extraction_gate_run_v1"
            and manifest.get("gate_name") == gate_name
            and scores.get("status") == "passed"
            and int(extraction.get("successful_count", -1)) == expected
        )
        checks.append(_check(
            f"qwen_extraction_{gate_name}",
            passed,
            f"{gate_name} extraction gate must be run and pass",
            {
                "path": str(manifest_path),
                "score_status": scores.get("status"),
                "successful_count": extraction.get("successful_count"),
                "expected_request_count": expected,
            },
        ))
    return checks


def _context_audit_check(root: Path, context_audit_root: Path) -> GateCheck:
    audit_path = root / context_audit_root / "audit.json"
    metrics_path = root / context_audit_root / "metrics.jsonl"
    if not audit_path.is_file() or not metrics_path.is_file():
        return _check(
            "qwen_context_evidence_audit",
            False,
            "context evidence audit must exist before answer-model execution",
            {"missing": [str(path) for path in (audit_path, metrics_path) if not path.is_file()]},
        )
    audit = _json(audit_path)
    metrics = _jsonl(metrics_path)
    cross_user_rows = [
        row for row in metrics
        if row.get("metric_group") == "context_evidence"
        and row.get("metric") == "cross_user_citation_evidence"
    ]
    cross_user_count = sum(float(row.get("value") or 0) for row in cross_user_rows)
    passed = (
        audit.get("schema_version") == "qwen_context_evidence_audit_v1"
        and audit.get("provider_request_count") == 0
        and audit.get("gold_opened_after_context_seal") is True
        and bool(cross_user_rows)
        and cross_user_count == 0
    )
    return _check(
        "qwen_context_evidence_audit",
        passed,
        "context evidence audit must pass with zero cross-user citation evidence",
        {
            "audit_path": str(audit_path),
            "metric_rows": len(metrics),
            "cross_user_metric_rows": len(cross_user_rows),
            "cross_user_citation_evidence": cross_user_count,
            "provider_request_count": audit.get("provider_request_count"),
        },
    )


def _materialization_check(root: Path, contexts_path: Path, exclusions_path: Path) -> GateCheck:
    context_path = root / contexts_path
    manifest_path = context_path.parent / "manifest.json"
    failures_path = context_path.parent / "failures.jsonl"
    if not context_path.is_file() or not manifest_path.is_file() or not failures_path.is_file():
        return _check(
            "qwen_materialization_clean",
            False,
            "materialization dry run must have a sealed context release",
            {"missing": [str(path) for path in (context_path, manifest_path, failures_path) if not path.is_file()]},
        )
    manifest = _json(manifest_path)
    failures = _jsonl(failures_path)
    excluded = _materialization_exclusions(root / exclusions_path)
    unresolved = [
        row for row in failures
        if str(row.get("record_id")) not in excluded
    ]
    passed = (
        manifest.get("schema_version") == "qwen_context_materialization_v2"
        and manifest.get("b6_b7_context_identity") is True
        and not unresolved
    )
    return _check(
        "qwen_materialization_clean",
        passed,
        "materialization must have zero unresolved failures or explicit exclusions",
        {
            "contexts_path": str(context_path),
            "manifest_status": manifest.get("status"),
            "failure_count": len(failures),
            "excluded_failure_count": len(excluded),
            "unresolved_failure_count": len(unresolved),
            "unresolved_stages": sorted({str(row.get("stage")) for row in unresolved}),
            "unresolved_codes": sorted({str(row.get("code")) for row in unresolved}),
        },
    )


def _materialization_exclusions(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    value = _json(path)
    if value.get("schema_version") != "qwen_materialization_failure_exclusions_v1":
        return set()
    ids = value.get("excluded_failure_record_ids")
    return {str(item) for item in ids} if isinstance(ids, list) else set()


def _check(id_: str, passed: bool, message: str, evidence: Mapping[str, object]) -> GateCheck:
    return GateCheck(
        id=id_,
        status="passed" if passed else "blocked",
        severity="blocker",
        message=message,
        evidence=evidence,
    )


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run zero-provider Qwen pre-run gates.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or __import__("sys").stdout
    args = _build_parser().parse_args(argv)
    report = build_pre_run_gate_report(args.repo_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else args.repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, file=stdout, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
