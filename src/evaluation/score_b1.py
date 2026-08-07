"""Score a completed B1 full-history run without making model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, TextIO

from .history import load_evaluation_questions, load_history_observations
from .prediction import BaselinePrediction, PredictionValidationError, validate_prediction
from .run_config import FrozenBaselineConfig, load_frozen_config, verify_frozen_content
from .scoring import (
    CAPABILITIES,
    PARTIAL_CREDIT_VALUE,
    ScoringDataError,
    abstention_metrics,
    aggregate_reference_metrics,
    answer_accuracy,
    answer_result,
    acceptable_answers,
    build_manual_review_record,
    capability_metrics,
    case_claim_counts,
    compare_references,
    deterministic_answer_match,
    gold_evidence_references,
    outdated_fact_error,
    rate,
    required_bool,
    required_string,
)
from .smoke import EvidenceValidationError, validate_prediction_evidence


SCORER_VERSION = "b1-scorer-v1"
DEFAULT_RESULTS_DIR = Path("results/pilot/b1-full-history")
DEFAULT_QUESTIONS_PATH = Path("data/pilot/evaluation/eval_questions.jsonl")
DEFAULT_GOLD_PATH = Path("data/pilot/evaluation/eval_answer.jsonl")
DEFAULT_SOURCES_DIR = Path("data/pilot/sources")
DEFAULT_CONFIG_PATH = Path("configs/full_history_baseline_v1.json")


def score_completed_b1(
    *,
    repo_root: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    sources_dir: str | Path = DEFAULT_SOURCES_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Validate, score, and atomically write one completed B1 result set."""

    root = Path(repo_root).resolve()
    results = _under_root(root, results_dir)
    questions_file = _under_root(root, questions_path)
    gold_file = _under_root(root, gold_path)
    sources = _under_root(root, sources_dir)
    config_file = _under_root(root, config_path)
    predictions_file = results / "predictions.jsonl"
    run_file = results / "run.json"
    failures_file = results / "failures.jsonl"
    scores_file = results / "scores.json"
    case_scores_file = results / "case_scores.jsonl"
    manual_file = results / "manual_review.jsonl"

    run = _load_json_object(run_file, "run")
    if run.get("run_status") not in {"completed", "completed_with_failures"}:
        raise ScoringDataError("Step 5 run is not complete; gold data was not read")

    predictions_hash_before = file_sha256(predictions_file)
    gold_hash_before = file_sha256(gold_file)
    config = load_frozen_config(config_file)
    verify_frozen_content(config, root)
    questions = _load_jsonl(questions_file, "questions")
    gold = _load_jsonl(gold_file, "gold answers")
    prediction_records = _load_jsonl(predictions_file, "predictions")
    failures = _load_jsonl(failures_file, "failures")
    observations = load_history_observations(sources)
    question_contracts = load_evaluation_questions(questions_file)

    _validate_run_preconditions(
        run=run,
        config=config,
        gold_hash=gold_hash_before,
        questions=questions,
        gold=gold,
        predictions=prediction_records,
        failures=failures,
    )
    predictions_by_case = _validated_predictions(
        prediction_records, questions, question_contracts, observations
    )
    failures_by_case = _execution_failures(run, questions, predictions_by_case)
    existing_manual = {
        item["case_id"]: item
        for item in (_load_jsonl(manual_file, "manual reviews") if manual_file.exists() else [])
    }

    gold_by_case = {item["case_id"]: item for item in gold}
    case_scores: list[dict[str, object]] = []
    manual_records: list[dict[str, object]] = []
    source_comparisons: list[dict[str, object]] = []
    message_comparisons: list[dict[str, object]] = []

    for question in questions:
        case_id = question["case_id"]
        capability = required_string(question, "capability", case_id)
        if capability not in CAPABILITIES:
            raise ScoringDataError(f"{case_id} has unknown capability {capability}")
        gold_record = gold_by_case[case_id]
        prediction = predictions_by_case.get(case_id)
        failure = failures_by_case.get(case_id)
        manual = None
        if prediction is not None and _may_need_manual(capability, gold_record, prediction):
            manual = build_manual_review_record(
                question, gold_record, prediction, existing_manual.get(case_id)
            )
            manual_records.append(manual)

        correctness, method, note = answer_result(
            capability=capability,
            question=question,
            gold=gold_record,
            prediction=prediction,
            manual_review=manual,
            execution_failure=failure,
        )
        predicted_messages = (
            {(item.source_id, item.message_id) for item in prediction.evidence}
            if prediction is not None
            else set()
        )
        gold_messages = gold_evidence_references(gold_record, case_id)
        predicted_sources = {item[0] for item in predicted_messages}
        gold_sources = {item[0] for item in gold_messages}
        source = compare_references(predicted_sources, gold_sources)
        message = compare_references(predicted_messages, gold_messages)
        source_comparisons.append(source)
        message_comparisons.append(message)

        total_claims, unsupported_count, unsupported_findings = case_claim_counts(
            correctness=correctness,
            prediction=prediction,
            gold_references=gold_messages,
            manual_review=manual,
        )
        outdated_error, outdated_findings, outdated_applicable = _outdated_result(
            question=question,
            gold=gold_record,
            prediction=prediction,
            manual=manual,
        )
        expected_abstention = required_bool(gold_record, "should_abstain", case_id)
        predicted_abstention = prediction is not None and prediction.status == "abstained"
        case_scores.append(
            {
                "case_id": case_id,
                "capability": capability,
                "prediction_status": prediction.status if prediction else None,
                "expected_answer_status": required_string(
                    gold_record, "answer_status", case_id
                ),
                "execution_failure": failure,
                "answer_correctness": correctness,
                "scoring_method": method,
                "predicted_source_ids": sorted(predicted_sources),
                "gold_source_ids": sorted(gold_sources),
                "source_true_positives": sorted(source["true_positives"]),
                "source_false_positives": sorted(source["false_positives"]),
                "source_false_negatives": sorted(source["false_negatives"]),
                "source_precision": source["precision"],
                "source_recall": source["recall"],
                "predicted_source_message_references": _reference_records(predicted_messages),
                "gold_source_message_references": _reference_records(gold_messages),
                "message_true_positives": _reference_records(message["true_positives"]),
                "message_false_positives": _reference_records(message["false_positives"]),
                "message_false_negatives": _reference_records(message["false_negatives"]),
                "message_precision": message["precision"],
                "message_recall": message["recall"],
                "expected_abstention": expected_abstention,
                "predicted_abstention": predicted_abstention,
                "unsupported_claim_count": unsupported_count,
                "total_claim_count": total_claims,
                "unsupported_claims": unsupported_findings,
                "current_vs_outdated_fact_applicable": outdated_applicable,
                "current_vs_outdated_fact_error": outdated_error,
                "outdated_fact_findings": outdated_findings,
                "manual_review_status": (
                    "not_required"
                    if manual is None
                    else "completed"
                    if manual.get("reviewer_judgment") is not None
                    else "pending"
                ),
                "short_scoring_note": note,
            }
        )

    unknown_manual = sorted(set(existing_manual) - {item["case_id"] for item in manual_records})
    if unknown_manual:
        raise ScoringDataError("manual review has unexpected cases: " + ", ".join(unknown_manual))

    unresolved = [
        item["case_id"] for item in case_scores if item["answer_correctness"] == "unresolved"
    ]
    execution_failure_ids = [
        item["case_id"] for item in case_scores if item["execution_failure"] is not None
    ]
    unsupported_cases = [
        item["case_id"] for item in case_scores if item["unsupported_claim_count"]
    ]
    outdated_cases = [
        item["case_id"] for item in case_scores if item["current_vs_outdated_fact_error"] is True
    ]
    total_claims = sum(item["total_claim_count"] for item in case_scores)
    unsupported_claims = sum(item["unsupported_claim_count"] for item in case_scores)
    applicable_outdated = sum(
        item["current_vs_outdated_fact_applicable"] for item in case_scores
    )
    outdated_errors = len(outdated_cases)
    scored_at = (now or (lambda: datetime.now(timezone.utc)))()
    if scored_at.tzinfo is None or scored_at.utcoffset() is None:
        raise ScoringDataError("scoring timestamp must include a UTC offset")

    predictions_hash_after = file_sha256(predictions_file)
    gold_hash_after = file_sha256(gold_file)
    if predictions_hash_before != predictions_hash_after:
        raise ScoringDataError("predictions.jsonl changed during scoring")
    if gold_hash_before != gold_hash_after:
        raise ScoringDataError("eval_answer.jsonl changed during scoring")

    scores = {
        "scorer_version": SCORER_VERSION,
        "baseline_id": run["baseline_id"],
        "run_configuration_hash": config.configuration_sha256,
        "dataset_hash": config.dataset_sha256,
        "predictions_sha256": predictions_hash_before,
        "predictions_sha256_after": predictions_hash_after,
        "predictions_unchanged": predictions_hash_before == predictions_hash_after,
        "gold_answer_sha256": gold_hash_before,
        "gold_answer_sha256_after": gold_hash_after,
        "gold_answer_unchanged": gold_hash_before == gold_hash_after,
        "scored_at_utc": scored_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result_status": (
            "incomplete"
            if execution_failure_ids
            else "final"
            if not unresolved
            else "provisional"
        ),
        "answer_correctness": answer_accuracy(case_scores),
        "evidence_source": aggregate_reference_metrics(source_comparisons),
        "evidence_message": aggregate_reference_metrics(message_comparisons),
        "abstention": abstention_metrics(
            [item["expected_abstention"] for item in case_scores],
            [item["predicted_abstention"] for item in case_scores],
        ),
        "unsupported_claims": {
            "total_claims": total_claims,
            "unsupported_claims": unsupported_claims,
            "rate": rate(unsupported_claims, total_claims, "no factual claims were made"),
            "affected_case_ids": unsupported_cases,
            "findings": [
                {"case_id": item["case_id"], **finding}
                for item in case_scores
                for finding in item["unsupported_claims"]
            ],
        },
        "current_vs_outdated_facts": {
            "applicable_cases": applicable_outdated,
            "errors": outdated_errors,
            "error_rate": rate(
                outdated_errors,
                applicable_outdated,
                "no questions ask for a current or corrected fact",
            ),
            "affected_case_ids": outdated_cases,
            "findings": [
                {"case_id": item["case_id"], **finding}
                for item in case_scores
                for finding in item["outdated_fact_findings"]
            ],
        },
        "capability_accuracy": capability_metrics(case_scores),
        "unresolved_manual_review_count": len(unresolved),
        "unresolved_case_ids": unresolved,
        "execution_failure_count": len(execution_failure_ids),
        "execution_failure_case_ids": execution_failure_ids,
        "metric_formulas": {
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "abstention_accuracy": "(TP + TN) / all evaluation cases",
            "strict_answer_accuracy": "fully correct / all evaluation cases",
            "reviewed_answer_accuracy": "fully correct / cases with completed judgments",
            "lenient_answer_accuracy": (
                f"(fully correct + {PARTIAL_CREDIT_VALUE} * partially correct) / all evaluation cases"
            ),
            "unsupported_claim_rate": "unsupported factual claims / total factual claims",
            "outdated_fact_error_rate": "outdated fact errors / applicable cases",
        },
        "denominator_rules": (
            "A zero denominator returns null and a reason. Macro evidence metrics average "
            "only cases whose per-case metric is defined. Execution failures count as "
            "incorrect answers and contribute missing gold evidence as false negatives."
        ),
    }

    _write_jsonl_atomic(case_scores_file, case_scores)
    _write_jsonl_atomic(manual_file, manual_records)
    _write_json_atomic(scores_file, scores)
    return scores


def _validate_run_preconditions(
    *,
    run: Mapping[str, object],
    config: FrozenBaselineConfig,
    gold_hash: str,
    questions: Sequence[Mapping[str, object]],
    gold: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> None:
    if run.get("run_status") not in {"completed", "completed_with_failures"}:
        raise ScoringDataError("Step 5 run is not complete")
    expected_run_config = {
        "baseline_id": "b1-full-history",
        "frozen_configuration_version": config.configuration_version,
        "frozen_configuration_hash": config.configuration_sha256,
        "provider": config.provider,
        "requested_model_version": config.requested_model,
        "resolved_model_version": config.resolved_model,
        "temperature": config.temperature,
        "generation_settings": dict(config.generation_settings),
        "prompt_version": config.prompt_version,
        "prompt_fingerprint": config.prompt_sha256,
        "dataset_hash": config.dataset_sha256,
        "source_ordering_rule": config.source_ordering_rule,
    }
    mismatches = [
        field
        for field, expected in expected_run_config.items()
        if run.get(field) != expected
    ]
    if mismatches:
        raise ScoringDataError(
            "Step 5 does not match the frozen Step 4 configuration: "
            + ", ".join(mismatches)
        )
    if run.get("gold_answer_hash_before") != gold_hash or run.get("gold_answer_hash_after") != gold_hash:
        raise ScoringDataError("gold-answer hash does not match the completed run")
    if run.get("gold_answer_hash_unchanged") is not True:
        raise ScoringDataError("completed run did not preserve the gold-answer file")
    question_ids = _ordered_ids(questions, "questions")
    gold_ids = _ordered_ids(gold, "gold answers")
    if len(question_ids) != 25 or question_ids != gold_ids:
        raise ScoringDataError("questions and gold answers must contain the same 25 cases in order")
    attempts = run.get("attempts")
    if not isinstance(attempts, list):
        raise ScoringDataError("run attempts must be a list")
    successful_records = [
        item.get("prediction")
        for item in attempts
        if isinstance(item, dict) and item.get("passed") is True
    ]
    if successful_records != list(predictions):
        raise ScoringDataError("predictions.jsonl does not belong to this run")
    failed_attempt_ids = [
        item.get("case_id")
        for item in attempts
        if isinstance(item, dict) and item.get("passed") is not True
    ]
    if [item.get("case_id") for item in failures] != failed_attempt_ids:
        raise ScoringDataError("failures.jsonl does not match this run")
    expected_counts = {
        "total_questions": 25,
        "recorded_cases": len(attempts),
        "successful_predictions": len(predictions),
        "remaining_questions": 25 - len(attempts),
    }
    count_mismatches = [
        field for field, expected in expected_counts.items() if run.get(field) != expected
    ]
    if count_mismatches:
        raise ScoringDataError(
            "Step 5 run counts are inconsistent: " + ", ".join(count_mismatches)
        )


def _validated_predictions(
    records: Sequence[Mapping[str, object]],
    questions: Sequence[Mapping[str, object]],
    question_contracts: Sequence[object],
    observations: Sequence[object],
) -> dict[str, BaselinePrediction]:
    question_ids = _ordered_ids(questions, "questions")
    question_by_id = {item.case_id: item for item in question_contracts}
    predictions: dict[str, BaselinePrediction] = {}
    for record in records:
        try:
            prediction = validate_prediction(record)
        except PredictionValidationError as error:
            raise ScoringDataError(str(error)) from error
        if prediction.case_id not in question_ids or prediction.case_id in predictions:
            raise ScoringDataError(f"invalid prediction case ID {prediction.case_id}")
        question = question_by_id[prediction.case_id]
        eligible = [item for item in observations if item.observed_at <= question.as_of]
        try:
            validate_prediction_evidence(prediction, eligible)
        except EvidenceValidationError as error:
            raise ScoringDataError(
                f"prediction {prediction.case_id} has invalid evidence: {error}"
            ) from error
        predictions[prediction.case_id] = prediction
    return predictions


def _execution_failures(
    run: Mapping[str, object],
    questions: Sequence[Mapping[str, object]],
    predictions: Mapping[str, BaselinePrediction],
) -> dict[str, dict[str, object]]:
    attempts = run["attempts"]
    by_case = {
        item.get("case_id"): item for item in attempts if isinstance(item, dict)
    }
    failures: dict[str, dict[str, object]] = {}
    for question in questions:
        case_id = question["case_id"]
        if case_id in predictions:
            continue
        attempt = by_case.get(case_id)
        failures[case_id] = {
            "stage": attempt.get("failure_stage") if attempt else "missing_prediction",
            "error": attempt.get("error") if attempt else "No run attempt was recorded.",
        }
    return failures


def _may_need_manual(
    capability: str,
    gold: Mapping[str, object],
    prediction: BaselinePrediction,
) -> bool:
    if prediction.status == "abstained" or required_bool(gold, "should_abstain", prediction.case_id):
        return False
    if deterministic_answer_match(
        prediction.answer,
        required_string(gold, "reference_answer", prediction.case_id),
        acceptable_answers(gold, prediction.case_id),
    ):
        return False
    return capability in {"conflict_detection", "user_modeling"}


def _outdated_result(
    *,
    question: Mapping[str, object],
    gold: Mapping[str, object],
    prediction: BaselinePrediction | None,
    manual: Mapping[str, object] | None,
) -> tuple[bool | None, list[dict[str, str]], bool]:
    if manual is not None and manual.get("reviewer_judgment") is not None:
        applicable = bool(manual["outdated_fact_applicable"])
        findings = list(manual["outdated_facts_found"])
        return (bool(findings) if applicable else None), findings, applicable
    if prediction is None:
        return None, [], False
    result = outdated_fact_error(
        question=required_string(question, "question", prediction.case_id),
        predicted_answer=prediction.answer,
        reference_answer=required_string(gold, "reference_answer", prediction.case_id),
        acceptable_answers=acceptable_answers(gold, prediction.case_id),
    )
    return result, [], result is not None


def _reference_records(references: Sequence[object] | set[object]) -> list[dict[str, object]]:
    return [
        {"source_id": source_id, "message_id": message_id}
        for source_id, message_id in sorted(references, key=lambda item: (item[0], item[1] or ""))
    ]


def _ordered_ids(records: Sequence[Mapping[str, object]], description: str) -> list[str]:
    ids = [required_string(item, "case_id", description) for item in records]
    if len(ids) != len(set(ids)):
        raise ScoringDataError(f"{description} contain duplicate case IDs")
    return ids


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ScoringDataError(f"could not hash {path}: {error}") from error


def _load_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScoringDataError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScoringDataError(f"{description} must be an object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScoringDataError(f"could not read {description} {path}: {error}") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScoringDataError(
                f"{description} {path}:{line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ScoringDataError(f"{description} {path}:{line_number} must be an object")
        records.append(value)
    return records


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    _write_text_atomic(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in records]
    _write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _under_root(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ScoringDataError(f"path is outside the repository: {path}") from error
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a completed B1 full-history run.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    args = _build_parser().parse_args(argv)
    try:
        scores = score_completed_b1(repo_root=Path.cwd(), results_dir=args.results_dir)
    except Exception as error:
        print(f"error={error}", file=sys.stderr)
        return 1
    print(f"result_status={scores['result_status']}", file=stdout)
    print(
        f"strict_accuracy={scores['answer_correctness']['strict_accuracy']['value']}",
        file=stdout,
    )
    print(f"scores={args.results_dir / 'scores.json'}", file=stdout)
    print(f"case_scores={args.results_dir / 'case_scores.jsonl'}", file=stdout)
    print(f"manual_review={args.results_dir / 'manual_review.jsonl'}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
