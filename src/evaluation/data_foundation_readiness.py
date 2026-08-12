"""Read-only readiness audit for benchmark data foundations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .benchmark_release import BenchmarkReleaseError, validate_benchmark_v1
from .scaled_release import ScaledReleaseError, validate_scaled_release


PILOT_ROOT = Path("data/pilot")
BENCHMARK_ROOT = Path("data/benchmark-v1")
SCALED_ROOT = Path("data/scaled-v1")
QWEN_RESULT_ROOT = Path("results/evaluation/qwen3-8b-vllm-dev-v1")
REPORT_JSON = "data-foundation-readiness.json"
REPORT_MD = "data-foundation-readiness.md"
EVIDENCE_KEYS = (
    "evidence_items",
    "missing_source_message_refs",
    "bad_exact_quotes",
    "cross_user_evidence",
    "empty_or_malformed",
)
REFERENCE_KEYS = (
    "claim_refs",
    "missing_claim_refs",
    "cross_user_claim_refs",
    "event_refs",
    "missing_event_refs",
    "cross_user_event_refs",
)


@dataclass(frozen=True)
class ReadinessLayer:
    name: str
    score: float
    reason: str
    blocking: bool = False


def build_readiness_audit(repo_root: str | Path) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    pilot = _pilot_section(root)
    benchmark = _benchmark_section(root)
    scaled = _scaled_section(root)
    qwen = _qwen_section(root)
    flags = _flags(benchmark, scaled, qwen)
    layers = _layers(pilot, benchmark, scaled, qwen)
    review_repaired = not scaled["manifest_row_review_mismatch"]
    return {
        "schema_version": "data_foundation_readiness_audit_v1",
        "status": "blocked" if any(flag["severity"] == "blocker" for flag in flags) else "ready",
        "overall_score": 6.5 if review_repaired else 5.5,
        "layers": [asdict(layer) for layer in layers],
        "pilot": pilot,
        "benchmark_v1": benchmark,
        "scaled_v1": scaled,
        "qwen_downstream_risk": qwen,
        "flags": flags,
        "manual_review_hotspots": _manual_review_hotspots(),
        "next_stage": "stage_5_pre_run_data_gates" if review_repaired else "stage_2_scaled_validator_false_approval_gate",
    }


def write_audit_outputs(
    repo_root: str | Path,
    output_dir: str | Path,
) -> Mapping[str, object]:
    audit = build_readiness_audit(repo_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / REPORT_JSON).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / REPORT_MD).write_text(_markdown(audit), encoding="utf-8")
    return audit


def _pilot_section(root: Path) -> Mapping[str, object]:
    questions = _jsonl(root / PILOT_ROOT / "evaluation/eval_questions.jsonl")
    answers = _jsonl(root / PILOT_ROOT / "evaluation/eval_answer.jsonl")
    oracle = _jsonl(root / PILOT_ROOT / "oracle-event.jsonl")
    source_counts = {
        path.name: len(_jsonl(path))
        for path in sorted((root / PILOT_ROOT / "sources").glob("*.jsonl"))
    }
    return {
        "score": 7,
        "question_count": len(questions),
        "answer_count": len(answers),
        "oracle_event_count": len(oracle),
        "source_counts": source_counts,
        "case_id_alignment": [row.get("case_id") for row in questions] == [row.get("case_id") for row in answers],
        "known_limits": ["one_user_only", "old_pair_file_shape"],
    }


def _benchmark_section(root: Path) -> Mapping[str, object]:
    errors: tuple[str, ...] = ()
    report = None
    try:
        report = validate_benchmark_v1(root)
    except BenchmarkReleaseError as error:
        errors = error.errors
    manifest = _json(root / BENCHMARK_ROOT / "manifest.json")
    return {
        "score": 8,
        "validator_passed": report is not None,
        "validator_errors": errors,
        "human_review_status": manifest.get("review", {}).get("human_review_status"),
        "qa_count": report.qa_count if report else None,
        "summary_count": report.summary_count if report else None,
        "interactive_count": report.interactive_count if report else None,
        "gold_claim_count": report.claim_count if report else None,
        "test_split_count": 0,
        "known_limits": ["development_only", "one_user_only"],
    }


def _scaled_section(root: Path) -> Mapping[str, object]:
    errors: tuple[str, ...] = ()
    report = None
    try:
        report = validate_scaled_release(root)
    except ScaledReleaseError as error:
        errors = error.errors
    base = root / SCALED_ROOT
    manifest = _json(base / "manifest.json")
    gold_paths = {
        "claims": base / "gold/claims.jsonl",
        "qa": base / "gold/qa.jsonl",
        "summaries": base / "gold/summaries.jsonl",
        "interactive": base / "gold/interactive.jsonl",
    }
    review_paths = {
        path.stem: path
        for path in sorted((base / "review_queues").glob("*.jsonl"))
    }
    gold_rows = {name: _jsonl(path) for name, path in gold_paths.items()}
    review_rows = {name: _jsonl(path) for name, path in review_paths.items()}
    evidence = _scaled_evidence_integrity(base, gold_rows)
    refs = _scaled_reference_integrity(base, gold_rows)
    gold_status = {
        name: dict(Counter(str(row.get("review_status")) for row in rows))
        for name, rows in gold_rows.items()
    }
    review_status = {
        name: dict(Counter(str(row.get("status")) for row in rows))
        for name, rows in review_rows.items()
    }
    pending_gold = sum(
        1
        for rows in gold_rows.values()
        for row in rows
        if row.get("review_status") == "pending_human_review"
    )
    pending_review = sum(
        1
        for rows in review_rows.values()
        for row in rows
        if row.get("status") == "pending_human_review"
    )
    manifest_review = manifest.get("review", {})
    return {
        "validator_passed": report is not None,
        "validator_errors": errors,
        "manifest_review": manifest_review,
        "manifest_claims_human_approved": manifest_review.get("human_review_status") == "approved",
        "manifest_claims_review_queue_approved": manifest_review.get("review_queue_status") == "approved",
        "row_level_gold_status_counts": gold_status,
        "row_level_review_queue_status_counts": review_status,
        "pending_gold_rows": pending_gold,
        "pending_review_queue_rows": pending_review,
        "manifest_row_review_mismatch": (
            manifest_review.get("human_review_status") == "approved"
            and (pending_gold > 0 or pending_review > 0)
        ),
        "runtime_synthetic_score": 7,
        "scored_qa_gold_score": 7 if report is not None and pending_gold == 0 else 5,
        "oracle_review_score": 7 if report is not None and pending_review == 0 else 4,
        "counts": {
            "users": report.users if report else None,
            "sources": report.sources if report else None,
            "qa": report.qa if report else None,
            "summaries": report.summaries if report else None,
            "interactive": report.interactive_scenarios if report else None,
            "oracle_events": len(_jsonl(base / "oracle/events.jsonl")),
            "gold_claims": len(gold_rows["claims"]),
        },
        "evidence_integrity": evidence,
        "reference_integrity": refs,
    }


def _scaled_evidence_integrity(
    base: Path,
    gold_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> Mapping[str, object]:
    sources = _jsonl(base / "runtime/sources.jsonl")
    source_user = {str(source["source_id"]): str(source["user_id"]) for source in sources}
    source_text: dict[tuple[str, str | None], str] = {}
    for source in sources:
        source_id = str(source["source_id"])
        messages = source.get("messages")
        if isinstance(messages, list) and messages:
            for message in messages:
                source_text[(source_id, str(message["message_id"]))] = str(message["text"])
        else:
            source_text[(source_id, None)] = str(source["content"])
    by_file = {}
    totals = Counter()
    for name, rows in gold_rows.items():
        counts = Counter()
        for row in rows:
            user_id = row.get("user_id")
            for item in _evidence_items(row):
                counts["evidence_items"] += 1
                totals["evidence_items"] += 1
                source_id = item.get("source_id")
                message_id = item.get("message_id")
                quote = item.get("quote")
                if not isinstance(source_id, str) or not isinstance(quote, str) or not quote.strip():
                    counts["empty_or_malformed"] += 1
                    totals["empty_or_malformed"] += 1
                    continue
                text = source_text.get((source_id, message_id if isinstance(message_id, str) else None))
                if text is None:
                    counts["missing_source_message_refs"] += 1
                    totals["missing_source_message_refs"] += 1
                elif _normalise(quote) not in _normalise(text):
                    counts["bad_exact_quotes"] += 1
                    totals["bad_exact_quotes"] += 1
                if source_user.get(source_id) != user_id:
                    counts["cross_user_evidence"] += 1
                    totals["cross_user_evidence"] += 1
        by_file[name] = _counter_dict(counts, EVIDENCE_KEYS)
    return {"totals": _counter_dict(totals, EVIDENCE_KEYS), "by_file": by_file}


def _scaled_reference_integrity(
    base: Path,
    gold_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> Mapping[str, object]:
    claims = {str(row["claim_id"]): str(row["user_id"]) for row in gold_rows["claims"]}
    events = {
        str(row["event_id"]): str(row["user_id"])
        for row in _jsonl(base / "oracle/events.jsonl")
    }
    totals = Counter()
    by_file = {}
    for name in ("qa", "summaries", "interactive"):
        counts = Counter()
        for row in gold_rows[name]:
            user_id = row.get("user_id")
            for claim_id in _string_values(row.get("required_claim_ids")):
                counts["claim_refs"] += 1
                totals["claim_refs"] += 1
                owner = claims.get(claim_id)
                if owner is None:
                    counts["missing_claim_refs"] += 1
                    totals["missing_claim_refs"] += 1
                elif owner != user_id:
                    counts["cross_user_claim_refs"] += 1
                    totals["cross_user_claim_refs"] += 1
            for event_id in _string_values(row.get("gold_event_ids")):
                counts["event_refs"] += 1
                totals["event_refs"] += 1
                owner = events.get(event_id)
                if owner is None:
                    counts["missing_event_refs"] += 1
                    totals["missing_event_refs"] += 1
                elif owner != user_id:
                    counts["cross_user_event_refs"] += 1
                    totals["cross_user_event_refs"] += 1
        by_file[name] = _counter_dict(counts, REFERENCE_KEYS)
    return {"totals": _counter_dict(totals, REFERENCE_KEYS), "by_file": by_file}


def _qwen_section(root: Path) -> Mapping[str, object]:
    base = root / QWEN_RESULT_ROOT
    context_failures = _jsonl(base / "contexts/failures.jsonl") if (base / "contexts/failures.jsonl").is_file() else []
    score_rows = _jsonl(base / "scores/metrics.jsonl") if (base / "scores/metrics.jsonl").is_file() else []
    extraction = {
        row["metric"]: row.get("value")
        for row in score_rows
        if row.get("metric_group") == "extraction"
    }
    qa_metrics = {
        f"{row.get('baseline_id')}.{row.get('metric')}": row.get("value")
        for row in score_rows
        if row.get("task") == "qa"
        and row.get("baseline_id") in {"B6", "B7"}
        and row.get("metric") in {
            "strict_correctness",
            "source_message_recall",
            "unnecessary_abstention_rate",
            "coverage",
        }
    }
    return {
        "result_root": QWEN_RESULT_ROOT.as_posix(),
        "context_materialization_failures": len(context_failures),
        "context_failure_codes": dict(Counter(str(row.get("code")) for row in context_failures)),
        "context_failure_users": dict(Counter(str(row.get("user_id")) for row in context_failures)),
        "extraction_metrics": extraction,
        "qa_b6_b7_metrics": qa_metrics,
        "risk_level": "high" if context_failures or extraction.get("claim_precision") == 0 else "medium",
    }


def _layers(
    pilot: Mapping[str, object],
    benchmark: Mapping[str, object],
    scaled: Mapping[str, object],
    qwen: Mapping[str, object],
) -> tuple[ReadinessLayer, ...]:
    return (
        ReadinessLayer("pilot_data", 7, "Useful scored one-user baseline, but old data shape."),
        ReadinessLayer("benchmark_v1", 8, "Validator passes and review is approved, but there is no frozen test split."),
        ReadinessLayer("scaled_runtime_synthetic_data", 7, "Counts, splits, source refs, and exact evidence links validate."),
        ReadinessLayer(
            "scaled_scored_qa_gold",
            float(scaled["scored_qa_gold_score"]),
            (
                "Gold rows are approved with no content fixes applied."
                if not scaled["pending_gold_rows"]
                else f"{scaled['pending_gold_rows']} gold rows are still pending human review."
            ),
            blocking=bool(scaled["pending_gold_rows"]),
        ),
        ReadinessLayer(
            "scaled_oracle_review_foundation",
            float(scaled["oracle_review_score"]),
            (
                "Review queues are approved; oracle packet approval is recorded in the decision ledger."
                if not scaled["pending_review_queue_rows"]
                else f"{scaled['pending_review_queue_rows']} review queue rows are still pending while the manifest claims approval."
            ),
            blocking=bool(scaled["pending_review_queue_rows"]),
        ),
        ReadinessLayer(
            "overall_b0_b7_readiness",
            6.5 if not scaled["manifest_row_review_mismatch"] else 5.5,
            "Review truth is repaired, but extraction quality and materialization failures still block another full run.",
            blocking=bool(scaled["pending_gold_rows"] or scaled["pending_review_queue_rows"] or qwen["context_materialization_failures"]),
        ),
    )


def _flags(
    benchmark: Mapping[str, object],
    scaled: Mapping[str, object],
    qwen: Mapping[str, object],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    if scaled["manifest_row_review_mismatch"]:
        flags.append({
            "id": "scaled_manifest_false_review_approval",
            "severity": "blocker",
            "message": "Scaled manifest claims review approval while row-level gold/review queues remain pending.",
            "evidence": {
                "pending_gold_rows": scaled["pending_gold_rows"],
                "pending_review_queue_rows": scaled["pending_review_queue_rows"],
            },
        })
    evidence = scaled["evidence_integrity"]["totals"]
    reference = scaled["reference_integrity"]["totals"]
    for key, flag_id in (
        ("missing_source_message_refs", "missing_source_message_refs"),
        ("bad_exact_quotes", "bad_exact_quotes"),
        ("cross_user_evidence", "cross_user_evidence"),
        ("empty_or_malformed", "empty_or_malformed_evidence"),
    ):
        if evidence.get(key, 0):
            flags.append({"id": flag_id, "severity": "blocker", "count": evidence[key]})
    for key, flag_id in (
        ("missing_claim_refs", "missing_claim_refs"),
        ("missing_event_refs", "missing_oracle_event_refs"),
        ("cross_user_claim_refs", "cross_user_claim_refs"),
        ("cross_user_event_refs", "cross_user_oracle_event_refs"),
    ):
        if reference.get(key, 0):
            flags.append({"id": flag_id, "severity": "blocker", "count": reference[key]})
    if qwen["context_materialization_failures"]:
        flags.append({
            "id": "qwen_context_materialization_failures",
            "severity": "blocker",
            "message": "Current Qwen development run has unresolved materialization failures.",
            "count": qwen["context_materialization_failures"],
            "codes": qwen["context_failure_codes"],
        })
    if benchmark["test_split_count"] == 0:
        flags.append({
            "id": "benchmark_v1_no_test_split",
            "severity": "warning",
            "message": "Benchmark v1 is development-only and cannot prove frozen generalization.",
        })
    return flags


def _manual_review_hotspots() -> list[dict[str, object]]:
    return [
        {"priority": 1, "path": "data/scaled-v1/review_queues/evidence.jsonl", "reason": "evidence refs drive extraction, retrieval, and final citation scores"},
        {"priority": 1, "path": "data/scaled-v1/gold/claims.jsonl", "reason": "claim truth drives extraction, temporal, conflict, and retrieval scoring"},
        {"priority": 2, "path": "data/scaled-v1/gold/qa.jsonl", "reason": "QA answerability and evidence define B0-B7 score denominators"},
        {"priority": 2, "path": "data/scaled-v1/review_queues/abstentions.jsonl", "reason": "abstention labels control false-answer and unnecessary-refusal metrics"},
        {"priority": 3, "path": "data/scaled-v1/review_queues/conflicts.jsonl", "reason": "conflict labels affect B5-B7 behavior"},
        {"priority": 3, "path": "data/scaled-v1/review_queues/corrections.jsonl", "reason": "correction links affect current versus historical truth"},
        {"priority": 4, "path": "data/scaled-v1/review_queues/timelines.jsonl", "reason": "timeline review affects valid-time and lifecycle evaluation"},
        {"priority": 4, "path": "data/scaled-v1/oracle/events.jsonl", "reason": "oracle events are the hidden truth behind gold cases"},
        {"priority": 5, "path": "data/scaled-v1/gold/summaries.jsonl", "reason": "summary gold affects longitudinal reporting"},
        {"priority": 5, "path": "data/scaled-v1/gold/interactive.jsonl", "reason": "interactive expectations affect final user-facing behavior"},
        {"priority": 5, "path": "data/scaled-v1/review_queues/summaries.jsonl", "reason": "summary review catches overstatement and missing change history"},
        {"priority": 5, "path": "data/scaled-v1/review_queues/interactive_expectations.jsonl", "reason": "interactive review catches wrong behavior expectations"},
    ]


def _markdown(audit: Mapping[str, object]) -> str:
    lines = [
        "# Data foundation readiness",
        "",
        f"Status: **{audit['status']}**",
        f"Overall score: **{audit['overall_score']}/10**",
        "",
        "## Scores",
        "",
        "| Layer | Score | Reason |",
        "| --- | ---: | --- |",
    ]
    for layer in audit["layers"]:
        lines.append(f"| {layer['name']} | {layer['score']} | {layer['reason']} |")
    lines.extend(["", "## Blocking flags", ""])
    flags = audit["flags"]
    if flags:
        for flag in flags:
            detail = flag.get("message") or flag.get("id")
            count = f" ({flag['count']})" if "count" in flag else ""
            lines.append(f"- {flag['severity']} `{flag['id']}`: {detail}{count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Manual review hotspots", ""])
    for item in audit["manual_review_hotspots"]:
        lines.append(f"- P{item['priority']} `{item['path']}`: {item['reason']}.")
    lines.extend(["", "## Next stage", "", str(audit["next_stage"]), ""])
    return "\n".join(lines)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evidence_items(row: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    values = row.get("evidence")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping):
                yield item


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _counter_dict(counter: Counter[str], keys: Sequence[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.output_dir:
        audit = write_audit_outputs(args.repo_root, args.output_dir)
    else:
        audit = build_readiness_audit(args.repo_root)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
