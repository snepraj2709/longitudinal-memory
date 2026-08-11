"""Deterministic Qwen v2 scorecards and blinded semantic-judge diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .qwen_execution import (
    AttemptComplete,
    AttemptGate,
    ExecutionJob,
    _client_factory,
    _load_contexts,
    _response_format,
    _write_or_verify,
    execute_jobs,
    TransportRetryLedger,
)
from .qwen_materialization import ContextPackage, TASK_KEYS
from .qwen_v2_contract import SERIES_ID, load_qwen_v2_config
from .scoring import deterministic_answer_match, normalize_text


EXPECTED_LOGICAL = {"development": 912, "test": 3648}
EXPECTED_JUDGE = {"development": 112, "test": 448}
BASELINES = tuple(f"B{index}" for index in range(8))


class QwenScoringError(RuntimeError):
    """Raised when sealed predictions and references cannot be scored exactly."""


@dataclass(frozen=True)
class MetricRow:
    series_id: str
    split: str
    task: str
    baseline_id: str
    metric_group: str
    metric: str
    numerator: int | float
    denominator: int
    value: float | None
    null_reason: str | None
    authoritative: bool


def seal_logical_predictions(
    provider_dir: Path,
    b7_dir: Path,
    output_dir: Path,
    *,
    split: str,
    series_id: str = SERIES_ID,
) -> Mapping[str, object]:
    """Combine sealed B0-B6 provider terminals and zero-call B7 terminals."""

    provider = _verify_execution_release(provider_dir)
    b7 = _verify_b7_release(b7_dir)
    if provider["split"] != split or b7["split"] != split:
        raise QwenScoringError("logical prediction split changed")
    rows = _jsonl(provider_dir / "responses.jsonl") + _jsonl(b7_dir / "responses.jsonl")
    expected = EXPECTED_LOGICAL[split]
    if len(rows) != expected:
        raise QwenScoringError(f"logical prediction count changed: {len(rows)}, expected {expected}")
    ordered = sorted(
        rows,
        key=lambda row: (
            int(str(row["baseline_id"])[1:]),
            TASK_KEYS.index(str(row["task"])),
            str(row["record_id"]),
        ),
    )
    keys = {(row["baseline_id"], row["task"], row["record_id"]) for row in ordered}
    if len(keys) != expected:
        raise QwenScoringError("logical prediction identity is duplicated")
    _verify_b6_b7_identity(ordered)
    payload = b"".join(_canonical(row) + b"\n" for row in ordered)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions.jsonl").write_bytes(payload)
    failures = [row for row in ordered if row["status"] == "failed"]
    failure_payload = b"".join(_canonical(row) + b"\n" for row in failures)
    (output_dir / "failures.jsonl").write_bytes(failure_payload)
    manifest = {
        "schema_version": "qwen_logical_prediction_seal_v2",
        "series_id": series_id,
        "split": split,
        "status": "sealed" if not failures else "sealed_with_failures",
        "logical_prediction_count": expected,
        "successful_count": expected - len(failures),
        "failure_count": len(failures),
        "provider_request_count": int(provider["provider_request_count"]),
        "b7_provider_request_count": 0,
        "b6_b7_context_identity": True,
        "b6_b7_underlying_answer_identity": True,
        "gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "predictions_sha256": sha256(payload).hexdigest(),
        "failures_sha256": sha256(failure_payload).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def score_sealed_release(
    *,
    repo_root: Path,
    prediction_dir: Path,
    contexts_path: Path,
    extraction_dir: Path,
    reference_loader: Callable[[], Mapping[str, Sequence[Mapping[str, object]]]],
    execution_metadata: Mapping[str, object],
    output_dir: Path,
    series_id: str = SERIES_ID,
    config_path: Path | None = None,
) -> Mapping[str, object]:
    """Verify the prediction checkpoint, then and only then open scorer references."""

    root = repo_root.resolve()
    if config_path is None:
        load_qwen_v2_config(root)
    manifest, predictions = verify_prediction_seal(prediction_dir, series_id=series_id)
    split = str(manifest["split"])
    _verify_context_release(contexts_path)
    contexts = _load_contexts(contexts_path)
    if len(contexts) != EXPECTED_LOGICAL[split]:
        raise QwenScoringError("materialized context count changed")
    extraction_manifest = _verify_execution_release(extraction_dir)
    if extraction_manifest["split"] != split:
        raise QwenScoringError("extraction and prediction splits differ")
    extraction = _jsonl(extraction_dir / "responses.jsonl")
    references = reference_loader()
    _validate_references(split, references)
    rows = score_records(
        split=split,
        predictions=predictions,
        contexts=contexts,
        extraction_records=extraction,
        references=references,
        execution_metadata=execution_metadata,
        series_id=series_id,
    )
    payload = b"".join(_canonical(asdict(row)) + b"\n" for row in rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.jsonl").write_bytes(payload)
    scorecard = {
        "schema_version": "qwen_scorecard_v2",
        "series_id": series_id,
        "split": split,
        "status": "completed" if manifest["failure_count"] == 0 else "completed_with_execution_failures",
        "composite_score": None,
        "composite_score_null_reason": "benchmark_contract_prohibits_composite_score",
        "metric_count": len(rows),
        "authoritative_metric_count": sum(row.authoritative for row in rows),
        "execution_failure_count": int(manifest["failure_count"]),
        "judge_status": "separate_uncalibrated_diagnostic",
        "prediction_manifest_sha256": _file_sha(prediction_dir / "manifest.json"),
        "metrics_sha256": sha256(payload).hexdigest(),
    }
    (output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return scorecard


def verify_prediction_seal(
    prediction_dir: Path,
    *,
    series_id: str = SERIES_ID,
) -> tuple[Mapping[str, object], tuple[dict[str, object], ...]]:
    expected_files = {"predictions.jsonl", "failures.jsonl", "manifest.json"}
    if not prediction_dir.is_dir() or {path.name for path in prediction_dir.iterdir()} != expected_files:
        raise QwenScoringError("prediction seal file set changed")
    manifest = json.loads((prediction_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen_logical_prediction_seal_v2":
        raise QwenScoringError("prediction seal schema changed")
    if manifest.get("series_id") != series_id or manifest.get("status") not in {"sealed", "sealed_with_failures"}:
        raise QwenScoringError("prediction seal identity or status changed")
    split = manifest.get("split")
    if split not in EXPECTED_LOGICAL:
        raise QwenScoringError("prediction seal split changed")
    prediction_bytes = (prediction_dir / "predictions.jsonl").read_bytes()
    failure_bytes = (prediction_dir / "failures.jsonl").read_bytes()
    if manifest.get("predictions_sha256") != sha256(prediction_bytes).hexdigest():
        raise QwenScoringError("sealed predictions changed")
    if manifest.get("failures_sha256") != sha256(failure_bytes).hexdigest():
        raise QwenScoringError("sealed failures changed")
    predictions = tuple(_jsonl_bytes(prediction_bytes))
    failures = tuple(row for row in predictions if row.get("status") == "failed")
    if len(predictions) != EXPECTED_LOGICAL[split] or len(failures) != manifest.get("failure_count"):
        raise QwenScoringError("prediction seal accounting changed")
    _verify_b6_b7_identity(predictions)
    return manifest, predictions


def score_records(
    *,
    split: str,
    predictions: Sequence[Mapping[str, object]],
    contexts: Sequence[ContextPackage],
    extraction_records: Sequence[Mapping[str, object]],
    references: Mapping[str, Sequence[Mapping[str, object]]],
    execution_metadata: Mapping[str, object],
    series_id: str = SERIES_ID,
) -> tuple[MetricRow, ...]:
    """Compute deterministic metrics without collapsing failures into score zeros."""

    required = {"claims", "qa", "summary", "interactive"}
    if not required.issubset(references):
        raise QwenScoringError("scoring references are incomplete")
    task_gold = {
        task: {str(row["case_id"]): row for row in references[task]}
        for task in TASK_KEYS
    }
    rows: list[MetricRow] = []
    rows.extend(_extraction_metrics(split, extraction_records, references["claims"], series_id))
    context_index = {(item.baseline_id, item.task, item.case_id): item for item in contexts}
    prediction_index = {
        (str(row["baseline_id"]), str(row["task"]), str(row["record_id"])): row
        for row in predictions
    }
    for baseline in BASELINES:
        for task in TASK_KEYS:
            gold = task_gold[task]
            cases = [
                (case_id, prediction_index.get((baseline, task, case_id)), context_index.get((baseline, task, case_id)), value)
                for case_id, value in sorted(gold.items())
            ]
            rows.extend(_execution_metrics(split, baseline, task, cases, series_id))
            rows.extend(_answer_metrics(split, baseline, task, cases, series_id))
            rows.extend(_retrieval_metrics(split, baseline, task, cases, series_id))
            rows.extend(_lifecycle_metrics(split, baseline, task, cases, references["claims"], series_id))
    rows.extend(_run_metrics(split, execution_metadata, series_id))
    rows.extend(_conflict_metrics(split, contexts, references.get("conflicts", ()), series_id))
    return tuple(sorted(rows, key=lambda row: (
        row.metric_group, row.metric, row.baseline_id, row.task
    )))


def build_judge_jobs(
    prediction_dir: Path,
    references: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    series_id: str = SERIES_ID,
) -> tuple[ExecutionJob, ...]:
    """Create blinded same-user, same-task, same-baseline batches of at most ten."""

    manifest, predictions = verify_prediction_seal(prediction_dir, series_id=series_id)
    split = str(manifest["split"])
    _validate_references(split, references)
    gold = {
        task: {str(row["case_id"]): row for row in references[task]}
        for task in TASK_KEYS
    }
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[(str(row["user_id"]), str(row["task"]), str(row["baseline_id"]))].append(row)
    jobs = []
    position = 0
    for (user_id, task, _baseline), records in sorted(groups.items()):
        records.sort(key=lambda row: str(row["record_id"]))
        for offset in range(0, len(records), 10):
            position += 1
            chunk = records[offset:offset + 10]
            candidates = []
            candidate_ids = []
            for record in chunk:
                case_id = str(record["record_id"])
                reference = gold[task].get(case_id)
                if reference is None:
                    raise QwenScoringError("judge reference does not match sealed prediction")
                candidate_id = sha256(f"{series_id}:{record['baseline_id']}:{case_id}".encode()).hexdigest()[:20]
                candidate_ids.append(candidate_id)
                candidates.append({
                    "candidate_id": candidate_id,
                    "task_input": _judge_task_input(task, reference),
                    "reference": _judge_reference(task, reference),
                    "response": record.get("output"),
                })
            schema = _judge_schema(candidate_ids)

            def validate(raw: str, _metadata, *, ids=tuple(candidate_ids)) -> Mapping[str, object]:
                return _validate_judge_output(json.loads(raw), ids)

            prompt = json.dumps({
                "task": task,
                "candidates": candidates,
                "rules": _judge_rules(task),
                "identity_policy": "Candidate and system identities are intentionally hidden.",
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            jobs.append(ExecutionJob(
                request_id=f"judge:{position:04d}",
                position=position,
                split=split,
                task="judge",
                record_id=f"judge_batch_{position:04d}",
                user_id=user_id,
                baseline_id=None,
                context_sha256=None,
                context_count=len(chunk),
                system_prompt=(
                    "Evaluate only the supplied response against the supplied reference. "
                    "Return the exact structured labels. Do not infer system identity."
                ),
                user_prompt=prompt,
                response_format=_response_format(f"qwen_judge_{position:04d}", schema),
                max_output_tokens=2000,
                validator=validate,
                series_id=series_id,
                local_metadata={
                    "candidates": {
                        candidate_id: {
                            "task": task,
                            "case_id": str(record["record_id"]),
                            "baseline_id": str(record["baseline_id"]),
                            "user_id": str(record["user_id"]),
                        }
                        for candidate_id, record in zip(candidate_ids, chunk, strict=True)
                    }
                },
            ))
    if len(jobs) != EXPECTED_JUDGE[split]:
        raise QwenScoringError(
            f"judge request count changed: {len(jobs)}, expected {EXPECTED_JUDGE[split]}"
        )
    return tuple(jobs)


def run_judge(
    *,
    repo_root: Path,
    prediction_dir: Path,
    reference_loader: Callable[[], Mapping[str, Sequence[Mapping[str, object]]]],
    output_dir: Path,
    base_url: str,
    api_key: str | None,
    retry_ledger: TransportRetryLedger | None = None,
    before_attempt: AttemptGate | None = None,
    after_attempt: AttemptComplete | None = None,
    series_id: str = SERIES_ID,
    model: str | None = None,
    temperature: float | None = None,
    workers: int = 8,
) -> Mapping[str, object]:
    """Verify prediction sealing before opening references and invoking the judge."""

    verify_prediction_seal(prediction_dir, series_id=series_id)
    references = reference_loader()
    jobs = build_judge_jobs(prediction_dir, references, series_id=series_id)
    model_alias = model
    if model_alias is None:
        config = load_qwen_v2_config(repo_root.resolve())
        model_alias = str(config["model"]["model_alias"])
    result = execute_jobs(
        jobs,
        output_dir=output_dir,
        client_factory=_client_factory(
            base_url=base_url,
            model=model_alias,
            api_key=api_key,
            temperature=temperature,
        ),
        retry_ledger=retry_ledger,
        before_attempt=before_attempt,
        after_attempt=after_attempt,
        workers=workers,
    )
    diagnostic = {
        **result,
        "diagnostic_only": True,
        "calibration_status": "not_calibrated",
        "published_judge_score": None,
        "published_judge_score_null_reason": "manual_20_percent_calibration_not_completed",
        "baseline_identity_shown_to_judge": False,
        "model_identity_shown_to_judge": False,
    }
    per_case = _compose_judge_per_case(jobs, output_dir)
    diagnostic["logical_diagnostic_count"] = len(per_case)
    diagnostic["label_counts"] = {
        label: sum(row.get("label") == label for row in per_case)
        for label in ("correct", "partially_correct", "incorrect", "unscorable")
    }
    per_case_payload = b"".join(_canonical(row) + b"\n" for row in per_case)
    _write_or_verify(output_dir / "per-case.jsonl", per_case_payload)
    diagnostic["per_case_sha256"] = sha256(per_case_payload).hexdigest()
    path = output_dir / "diagnostic.json"
    _write_or_verify(
        path,
        json.dumps(diagnostic, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return diagnostic


def _compose_judge_per_case(
    jobs: Sequence[ExecutionJob],
    output_dir: Path,
) -> tuple[dict[str, object], ...]:
    terminals = {int(row["position"]): row for row in _jsonl(output_dir / "responses.jsonl")}
    rows = []
    for job in jobs:
        terminal = terminals.get(job.position)
        if terminal is None:
            raise QwenScoringError("judge terminal record is missing")
        candidates = json.loads(job.user_prompt)["candidates"]
        if terminal.get("status") == "succeeded":
            judgments = {
                item["candidate_id"]: item
                for item in terminal["output"]["judgments"]
            }
        else:
            judgments = {}
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            judgment = judgments.get(candidate_id)
            metadata = job.local_metadata
            if not isinstance(metadata, Mapping) or not isinstance(metadata.get("candidates"), Mapping):
                raise QwenScoringError("local judge identity map is missing")
            identity = metadata["candidates"].get(candidate_id)
            if not isinstance(identity, Mapping):
                raise QwenScoringError("local judge candidate identity is missing")
            rows.append({
                "candidate_id": candidate_id,
                "judge_batch_id": job.record_id,
                "judge_batch_position": job.position,
                "task": identity["task"],
                "case_id": identity["case_id"],
                "baseline_id": identity["baseline_id"],
                "user_id": identity["user_id"],
                "status": "scored" if judgment is not None else "judge_execution_failed",
                "label": None if judgment is None else judgment["label"],
                "reason_codes": [] if judgment is None else judgment["reason_codes"],
                "failure_code": None if judgment is not None else terminal.get("failure_code"),
            })
    return tuple(sorted(rows, key=lambda row: (
        str(row["baseline_id"]), str(row["task"]), str(row["case_id"])
    )))


def _extraction_metrics(split, records, gold_claims, series_id):
    predicted = []
    valid_count = 0
    for row in records:
        if row.get("status") == "succeeded" and isinstance(row.get("output"), Mapping):
            valid_count += 1
            claims = row["output"].get("claims", [])
            if isinstance(claims, list):
                predicted.extend({**claim, "user_id": row.get("user_id")} for claim in claims)
    pred_keys = {_claim_key(item) for item in predicted}
    gold_keys = {_claim_key(item) for item in gold_claims}
    matched = len(pred_keys & gold_keys)
    rows = [
        _metric(split, "all", "all", "extraction", "structurally_valid_sources", valid_count, len(records), series_id=series_id),
        _metric(split, "all", "all", "extraction", "claim_precision", matched, len(pred_keys), series_id=series_id),
        _metric(split, "all", "all", "extraction", "claim_recall", matched, len(gold_keys), series_id=series_id),
    ]
    precision = matched / len(pred_keys) if pred_keys else 0
    recall = matched / len(gold_keys) if gold_keys else 0
    denominator = 1 if precision + recall else 0
    rows.append(_metric(
        split, "all", "all", "extraction", "claim_f1",
        2 * precision * recall / (precision + recall) if denominator else 0,
        denominator,
        "no_matched_or_predicted_claims",
        series_id=series_id,
    ))
    return rows


def _execution_metrics(split, baseline, task, cases, series_id):
    planned = len(cases)
    success = sum(record is not None and record.get("status") == "succeeded" for _, record, _, _ in cases)
    failed = sum(record is not None and record.get("status") == "failed" for _, record, _, _ in cases)
    return [
        _metric(split, baseline, task, "execution", "valid_response_rate", success, planned, series_id=series_id),
        _count_metric(split, baseline, task, "execution", "execution_failures", failed, series_id=series_id),
    ]


def _answer_metrics(split, baseline, task, cases, series_id):
    successful = [(record, gold) for _, record, _, gold in cases if record and record.get("status") == "succeeded"]
    if not successful:
        return [_metric(split, baseline, task, "answer", "answer_correctness", 0, 0, "no_valid_predictions", series_id=series_id)]
    expected_abstain = [bool(gold.get("should_abstain")) for record, gold in successful]
    predicted_abstain = [record["output"].get("status") == "abstained" for record, gold in successful]
    rows = [
        _metric(split, baseline, task, "abstention", "abstention_accuracy", sum(a == b for a, b in zip(expected_abstain, predicted_abstain)), len(successful), series_id=series_id),
        _metric(split, baseline, task, "abstention", "coverage", sum(not item for item in predicted_abstain), len(successful), series_id=series_id),
        _metric(split, baseline, task, "abstention", "false_answer_rate", sum(expected and not predicted for expected, predicted in zip(expected_abstain, predicted_abstain)), sum(expected_abstain), "no_unanswerable_cases", series_id=series_id),
        _metric(split, baseline, task, "abstention", "unnecessary_abstention_rate", sum(not expected and predicted for expected, predicted in zip(expected_abstain, predicted_abstain)), sum(not item for item in expected_abstain), "no_answerable_cases", series_id=series_id),
    ]
    evidence_scores = []
    quote_scores = []
    for record, gold in successful:
        output = record["output"]
        predicted_evidence = {
            (item["source_id"], item.get("message_id"))
            for item in output.get("citations", [])
        }
        gold_evidence = {
            (item["source_id"], item.get("message_id"))
            for item in gold.get("evidence", [])
        }
        evidence_scores.append((len(predicted_evidence & gold_evidence), len(predicted_evidence), len(gold_evidence)))
        exact = {(item["source_id"], item.get("message_id"), item["quote"]) for item in gold.get("evidence", [])}
        predicted_exact = {(item["source_id"], item.get("message_id"), item["quote"]) for item in output.get("citations", [])}
        quote_scores.append((len(exact & predicted_exact), len(predicted_exact)))
    tp = sum(item[0] for item in evidence_scores)
    pred_n = sum(item[1] for item in evidence_scores)
    gold_n = sum(item[2] for item in evidence_scores)
    rows.extend([
        _metric(split, baseline, task, "evidence", "source_message_precision", tp, pred_n, "no_predicted_evidence", series_id=series_id),
        _metric(split, baseline, task, "evidence", "source_message_recall", tp, gold_n, "no_gold_evidence", series_id=series_id),
        _metric(split, baseline, task, "evidence", "exact_quote_correctness", sum(item[0] for item in quote_scores), sum(item[1] for item in quote_scores), "no_predicted_citations", series_id=series_id),
    ])
    if task == "qa":
        strict = lenient = eligible = 0
        for record, gold in successful:
            output = record["output"]
            if bool(gold.get("should_abstain")):
                eligible += 1
                matched = output.get("status") == "abstained"
                strict += matched
                lenient += matched
            else:
                eligible += 1
                if output.get("status") != "abstained":
                    answer = str(output["answer"])
                    reference = str(gold["reference_answer"])
                    strict += normalize_text(answer) == normalize_text(reference)
                    acceptable = [str(item) for item in gold.get("acceptable_answers", [])]
                    lenient += deterministic_answer_match(answer, reference, acceptable)
        rows.extend([
            _metric(split, baseline, task, "answer", "strict_correctness", strict, eligible, "no_scorable_qa_predictions", series_id=series_id),
            _metric(split, baseline, task, "answer", "lenient_correctness", lenient, eligible, "no_scorable_qa_predictions", series_id=series_id),
        ])
    else:
        metric = "gold_event_f1" if task == "summary" else "behaviour_coverage"
        rows.append(_metric(split, baseline, task, task, metric, 0, 0, "semantic_judge_diagnostic_only", series_id=series_id))
    return rows


def _retrieval_metrics(split, baseline, task, cases, series_id):
    if baseline in {"B0", "B1"}:
        return [_metric(split, baseline, task, "retrieval", "recall_at_10", 0, 0, "retrieval_not_applicable", series_id=series_id)]
    recalls5 = []
    recalls10 = []
    reciprocal = []
    ndcgs = []
    for _, prediction, context, gold in cases:
        if not prediction or prediction.get("status") != "succeeded" or context is None:
            continue
        relevant = {(item["source_id"], item.get("message_id")) for item in gold.get("evidence", [])}
        if not relevant:
            continue
        ranked = []
        ranked_refs = []
        for record in context.context_records:
            refs = {(item["source_id"], item.get("message_id")) for item in record.get("evidence", [])}
            ranked_refs.append(refs & relevant)
            ranked.append(bool(refs & relevant))
        retrieved5 = set().union(*ranked_refs[:5]) if ranked_refs[:5] else set()
        retrieved10 = set().union(*ranked_refs[:10]) if ranked_refs[:10] else set()
        recalls5.append(len(retrieved5) / len(relevant))
        recalls10.append(len(retrieved10) / len(relevant))
        first = next((index + 1 for index, value in enumerate(ranked) if value), None)
        reciprocal.append(0.0 if first is None else 1 / first)
        dcg = sum((1 / math.log2(index + 2)) for index, value in enumerate(ranked[:10]) if value)
        ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), 10)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return [
        _average_metric(split, baseline, task, "retrieval", "recall_at_5", recalls5, series_id=series_id),
        _average_metric(split, baseline, task, "retrieval", "recall_at_10", recalls10, series_id=series_id),
        _average_metric(split, baseline, task, "retrieval", "mean_reciprocal_rank", reciprocal, series_id=series_id),
        _average_metric(split, baseline, task, "retrieval", "ndcg_at_10", ndcgs, series_id=series_id),
    ]


def _lifecycle_metrics(split, baseline, task, cases, gold_claims, series_id):
    if baseline not in {"B5", "B6", "B7"}:
        return [_metric(split, baseline, task, "temporal", "lifecycle_accuracy", 0, 0, "lifecycle_not_available", series_id=series_id)]
    claims = {str(item["claim_id"]): item for item in gold_claims}
    matched = total = 0
    for _, prediction, context, gold in cases:
        if not prediction or prediction.get("status") != "succeeded" or context is None:
            continue
        for claim_id in gold.get("required_claim_ids", []):
            claim = claims.get(str(claim_id))
            if claim is None:
                continue
            refs = {(item["source_id"], item.get("message_id")) for item in claim.get("evidence", [])}
            statuses = set()
            for record in context.context_records:
                evidence = {(item["source_id"], item.get("message_id")) for item in record.get("evidence", [])}
                if evidence & refs:
                    statuses.update(record.get("lifecycle_statuses", []))
            total += 1
            matched += str(claim.get("status")) in statuses
    return [_metric(split, baseline, task, "temporal", "lifecycle_accuracy", matched, total, "no_required_claim_lifecycle_matches", series_id=series_id)]


def _conflict_metrics(split, contexts, references, series_id):
    if not references:
        return [_metric(split, "B6", "all", "conflict", "relation_accuracy", 0, 0, "no_conflict_relation_reference", series_id=series_id)]
    return [_metric(split, "B6", "all", "conflict", "relation_accuracy", 0, 0, "conflict_reference_adapter_not_available", series_id=series_id)]


def _run_metrics(split, metadata, series_id):
    rows = []
    for name in ("input_tokens", "output_tokens", "provider_request_count", "transport_retry_count"):
        rows.append(_count_metric(split, "all", "all", "usage", name, int(metadata.get(name, 0)), series_id=series_id))
    for name in ("gpu_cost_inr", "inference_wall_seconds", "output_tokens_per_second"):
        value = metadata.get(name)
        rows.append(_metric(
            split, "all", "all", "performance", name,
            float(value) if isinstance(value, (int, float)) else 0,
            1 if isinstance(value, (int, float)) else 0,
            "measurement_not_available",
            series_id=series_id,
        ))
    return rows


def _metric(
    split,
    baseline,
    task,
    group,
    name,
    numerator,
    denominator,
    null_reason="zero_denominator",
    *,
    series_id=SERIES_ID,
):
    value = float(numerator) / denominator if denominator else None
    return MetricRow(series_id, split, task, baseline, group, name, numerator, denominator, value, None if denominator else null_reason, True)


def _count_metric(split, baseline, task, group, name, value, *, series_id=SERIES_ID):
    return MetricRow(series_id, split, task, baseline, group, name, value, 1, float(value), None, True)


def _average_metric(split, baseline, task, group, name, values, *, series_id=SERIES_ID):
    return _metric(
        split, baseline, task, group, name, sum(values), len(values),
        "no_scorable_retrieval_cases", series_id=series_id,
    )


def _claim_key(claim):
    precision = claim.get("time_precision")
    if precision is None:
        boundaries = (claim.get("valid_from"), claim.get("valid_to"))
        precision = "timestamp" if any(isinstance(value, str) and "T" in value for value in boundaries) else (
            "date" if any(value is not None for value in boundaries) else "unknown"
        )
    return (
        str(claim.get("user_id", "")), str(claim.get("subject_id", "")),
        str(claim.get("speaker_id", "")), str(claim.get("predicate", "")),
        json.dumps(claim.get("object"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        str(claim.get("polarity", "")), str(claim.get("epistemic_status", "")),
        claim.get("valid_from"), claim.get("valid_to"), str(precision),
    )


def _validate_references(
    split: str,
    references: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    required = {"claims", "qa", "summary", "interactive"}
    if not required.issubset(references):
        raise QwenScoringError("scoring references are incomplete")
    user_ids = set()
    for name in required:
        for row in references[name]:
            row_split = row.get("split")
            if row_split is not None and row_split != split:
                raise QwenScoringError("reference crosses the split boundary")
            user_id = row.get("user_id")
            if isinstance(user_id, str):
                user_ids.add(user_id)
    expected = {"user_001", "user_002"} if split == "development" else {
        f"user_{index:03d}" for index in range(3, 11)
    }
    if user_ids and not user_ids.issubset(expected):
        raise QwenScoringError("reference crosses the user boundary")


def _verify_b6_b7_identity(rows):
    by_key = {(str(row["baseline_id"]), str(row["task"]), str(row["record_id"])): row for row in rows}
    for (baseline, task, case_id), b7 in by_key.items():
        if baseline != "B7":
            continue
        b6 = by_key.get(("B6", task, case_id))
        if b6 is None or b6.get("context_sha256") != b7.get("context_sha256"):
            raise QwenScoringError("B6/B7 context identity changed")
        if b6.get("status") == "succeeded" and (
            b6.get("underlying_answer_sha256") != b7.get("underlying_answer_sha256")
        ):
            raise QwenScoringError("B6/B7 underlying answer identity changed")


def _verify_execution_release(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    response_bytes = (path / "responses.jsonl").read_bytes()
    failure_bytes = (path / "failures.jsonl").read_bytes()
    if manifest.get("schema_version") != "qwen_execution_batch_v2":
        raise QwenScoringError("execution manifest schema changed")
    if manifest.get("responses_sha256") != sha256(response_bytes).hexdigest():
        raise QwenScoringError("execution responses changed")
    if manifest.get("failures_sha256") != sha256(failure_bytes).hexdigest():
        raise QwenScoringError("execution failures changed")
    return manifest


def _verify_context_release(path: Path) -> Mapping[str, object]:
    manifest_path = path.parent / "manifest.json"
    failure_path = path.parent / "failures.jsonl"
    if path.name != "contexts.jsonl" or not manifest_path.is_file() or not failure_path.is_file():
        raise QwenScoringError("context release file set is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen_context_materialization_v2":
        raise QwenScoringError("context release schema changed")
    if manifest.get("contexts_sha256") != _file_sha(path):
        raise QwenScoringError("materialized contexts changed")
    if manifest.get("failures_sha256") != _file_sha(failure_path):
        raise QwenScoringError("materialization failures changed")
    if manifest.get("b6_b7_context_identity") is not True:
        raise QwenScoringError("materialized B6/B7 identity is not proven")
    return manifest


def _verify_b7_release(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen_b7_derivation_v2":
        raise QwenScoringError("B7 manifest schema changed")
    if manifest.get("responses_sha256") != _file_sha(path / "responses.jsonl"):
        raise QwenScoringError("B7 responses changed")
    if manifest.get("failures_sha256") != _file_sha(path / "failures.jsonl"):
        raise QwenScoringError("B7 failures changed")
    return manifest


def _judge_schema(candidate_ids):
    return {
        "type": "object",
        "properties": {"judgments": {"type": "array", "minItems": len(candidate_ids), "maxItems": len(candidate_ids), "items": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "enum": list(candidate_ids)},
                "label": {"type": "string", "enum": ["correct", "partially_correct", "incorrect", "unscorable"]},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_id", "label", "reason_codes"],
            "additionalProperties": False,
        }}},
        "required": ["judgments"],
        "additionalProperties": False,
    }


def _validate_judge_output(value, candidate_ids):
    if not isinstance(value, Mapping) or set(value) != {"judgments"} or not isinstance(value["judgments"], list):
        raise QwenScoringError("judge output fields changed")
    judgments = value["judgments"]
    ids = [item.get("candidate_id") for item in judgments if isinstance(item, Mapping)]
    if sorted(ids) != sorted(candidate_ids) or len(ids) != len(set(ids)):
        raise QwenScoringError("judge candidate coverage changed")
    allowed = {"correct", "partially_correct", "incorrect", "unscorable"}
    for item in judgments:
        if set(item) != {"candidate_id", "label", "reason_codes"} or item["label"] not in allowed:
            raise QwenScoringError("judge label changed")
        if not isinstance(item["reason_codes"], list) or not all(isinstance(code, str) for code in item["reason_codes"]):
            raise QwenScoringError("judge reason codes changed")
    return {"judgments": sorted((dict(item) for item in judgments), key=lambda item: item["candidate_id"])}


def _judge_task_input(task, gold):
    names = {"qa": ("question",), "summary": ("instruction",), "interactive": ("scenario", "initial_user_message")}[task]
    return {name: gold[name] for name in names}


def _judge_reference(task, gold):
    names = {
        "qa": ("reference_answer", "acceptable_answers", "should_abstain"),
        "summary": ("reference_summary", "gold_event_ids", "should_abstain"),
        "interactive": ("expected_behaviours", "should_abstain"),
    }[task]
    return {name: gold[name] for name in names}


def _judge_rules(task):
    return {
        "qa": "Judge factual correctness or justified abstention; evidence correctness is scored elsewhere.",
        "summary": "Judge faithfulness to required events, correction, time, and uncertainty.",
        "interactive": "Judge coverage of expected behaviours without requiring exact wording.",
    }[task]


def _jsonl(path):
    return _jsonl_bytes(path.read_bytes())


def _jsonl_bytes(raw):
    return [json.loads(line) for line in raw.splitlines()]


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--predictions", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--provider", required=True)
    seal.add_argument("--b7", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--split", choices=("development", "test"), required=True)
    score = subparsers.add_parser("score")
    _reference_arguments(score, include_claims=True)
    score.add_argument("--repo-root", default=".")
    score.add_argument("--predictions", required=True)
    score.add_argument("--contexts", required=True)
    score.add_argument("--extraction", required=True)
    score.add_argument("--execution-metadata", required=True)
    score.add_argument("--output", required=True)
    judge = subparsers.add_parser("judge")
    _reference_arguments(judge, include_claims=False)
    judge.add_argument("--repo-root", default=".")
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--base-url", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result, _ = verify_prediction_seal(Path(args.predictions))
    elif args.command == "seal":
        result = seal_logical_predictions(
            Path(args.provider), Path(args.b7), Path(args.output), split=args.split
        )
    elif args.command == "score":
        result = score_sealed_release(
            repo_root=Path(args.repo_root),
            prediction_dir=Path(args.predictions),
            contexts_path=Path(args.contexts),
            extraction_dir=Path(args.extraction),
            reference_loader=lambda: _load_reference_args(args, include_claims=True),
            execution_metadata=json.loads(Path(args.execution_metadata).read_text(encoding="utf-8")),
            output_dir=Path(args.output),
        )
    else:
        import os

        result = run_judge(
            repo_root=Path(args.repo_root),
            prediction_dir=Path(args.predictions),
            reference_loader=lambda: _load_reference_args(args, include_claims=False),
            output_dir=Path(args.output),
            base_url=args.base_url,
            api_key=os.environ.get("QWEN_VLLM_API_KEY"),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _reference_arguments(parser: argparse.ArgumentParser, *, include_claims: bool) -> None:
    if include_claims:
        parser.add_argument("--gold-claims", required=True)
    parser.add_argument("--gold-qa", required=True)
    parser.add_argument("--gold-summary", required=True)
    parser.add_argument("--gold-interactive", required=True)


def _load_reference_args(args: argparse.Namespace, *, include_claims: bool):
    # This helper is invoked only by the post-seal loader callback.
    values = {
        "qa": _jsonl(Path(args.gold_qa)),
        "summary": _jsonl(Path(args.gold_summary)),
        "interactive": _jsonl(Path(args.gold_interactive)),
    }
    values["claims"] = _jsonl(Path(args.gold_claims)) if include_claims else []
    return values


if __name__ == "__main__":
    main()
