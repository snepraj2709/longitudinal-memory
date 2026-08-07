"""Inspect and summarize failures from a completed B1 scoring run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, TextIO

from .history import HistoryObservation, load_history_observations
from .prediction import PredictionValidationError, validate_prediction
from .run_config import FrozenBaselineConfig, load_frozen_config, verify_frozen_content


FAILURE_TAXONOMY = (
    "missed_evidence",
    "wrong_date",
    "used_stale_information",
    "failed_to_apply_correction",
    "treated_assumption_as_fact",
    "unsupported_user_inference",
    "should_have_abstained",
    "abstained_despite_sufficient_evidence",
    "invalid_citation",
)
CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "user_modeling",
    "abstention",
)
REVIEWER_STATUSES = frozenset({"completed", "unresolved"})
ANALYZER_VERSION = "b1-failure-analysis-v1"

DEFAULT_RESULTS_DIR = Path("results/pilot/b1-full-history")
DEFAULT_QUESTIONS_PATH = Path("data/pilot/evaluation/eval_questions.jsonl")
DEFAULT_GOLD_PATH = Path("data/pilot/evaluation/eval_answer.jsonl")
DEFAULT_SOURCES_DIR = Path("data/pilot/sources")
DEFAULT_CONFIG_PATH = Path("configs/full_history_baseline_v1.json")

_RECORD_FIELDS = frozenset(
    {
        "case_id",
        "capability",
        "question",
        "as_of",
        "result",
        "predicted_status",
        "predicted_answer",
        "gold_status",
        "reference_answer",
        "primary_classification",
        "contributing_classifications",
        "execution_failure",
        "predicted_evidence",
        "expected_evidence",
        "relevant_evidence_inspected",
        "missed_evidence",
        "wrong_date",
        "stale_evidence",
        "correcting_evidence",
        "assumptions_treated_as_fact",
        "unsupported_claims",
        "abstention_error",
        "invalid_citations",
        "predicted_abstention",
        "expected_abstention",
        "concise_explanation",
        "reviewer_status",
        "reviewer_identifier",
        "reviewed_at",
    }
)
_CATEGORY_DETAIL_FIELDS = {
    "missed_evidence": "missed_evidence",
    "wrong_date": "wrong_date",
    "used_stale_information": "stale_evidence",
    "failed_to_apply_correction": "correcting_evidence",
    "treated_assumption_as_fact": "assumptions_treated_as_fact",
    "unsupported_user_inference": "unsupported_claims",
    "should_have_abstained": "abstention_error",
    "abstained_despite_sufficient_evidence": "abstention_error",
    "invalid_citation": "invalid_citations",
}


class FailureAnalysisError(ValueError):
    """Raised when a failure report cannot be trusted."""


def validate_failure_record(record: object) -> dict[str, object]:
    """Validate one manual failure record and its taxonomy details."""

    if not isinstance(record, dict):
        raise FailureAnalysisError("failure record must be an object")
    unknown = sorted(set(record) - _RECORD_FIELDS)
    missing = sorted(_RECORD_FIELDS - set(record))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise FailureAnalysisError("; ".join(details))

    case_id = _required_string(record, "case_id")
    for field in (
        "capability",
        "question",
        "as_of",
        "result",
        "predicted_status",
        "predicted_answer",
        "gold_status",
        "reference_answer",
        "concise_explanation",
        "reviewer_status",
    ):
        _required_string(record, field)
    _aware_timestamp(record["as_of"], f"{case_id}.as_of")

    capability = record["capability"]
    if capability not in CAPABILITIES:
        raise FailureAnalysisError(f"{case_id} has unknown capability {capability!r}")
    reviewer_status = record["reviewer_status"]
    if reviewer_status not in REVIEWER_STATUSES:
        raise FailureAnalysisError(f"{case_id} has invalid reviewer_status")

    primary = record["primary_classification"]
    contributing = record["contributing_classifications"]
    execution_failure = record["execution_failure"]
    if not isinstance(contributing, list):
        raise FailureAnalysisError(f"{case_id}.contributing_classifications must be a list")
    classifications = [primary, *contributing]
    unknown_categories = [
        item for item in classifications if item is not None and item not in FAILURE_TAXONOMY
    ]
    if unknown_categories:
        raise FailureAnalysisError(
            f"{case_id} has unknown failure classification {unknown_categories[0]!r}"
        )
    if len([item for item in classifications if item is not None]) != len(
        set(item for item in classifications if item is not None)
    ):
        raise FailureAnalysisError(f"{case_id} repeats a failure classification")

    if execution_failure is None:
        if primary is None:
            raise FailureAnalysisError(
                f"{case_id} reasoning failure needs one primary classification"
            )
    else:
        if not isinstance(execution_failure, dict):
            raise FailureAnalysisError(f"{case_id}.execution_failure must be an object")
        if primary is not None or contributing:
            raise FailureAnalysisError(
                f"{case_id} execution failure must remain separate from the taxonomy"
            )

    for field in (
        "predicted_evidence",
        "expected_evidence",
        "relevant_evidence_inspected",
        "missed_evidence",
        "stale_evidence",
        "correcting_evidence",
        "assumptions_treated_as_fact",
        "unsupported_claims",
        "invalid_citations",
    ):
        if not isinstance(record[field], list):
            raise FailureAnalysisError(f"{case_id}.{field} must be a list")
    for field in ("predicted_abstention", "expected_abstention"):
        if not isinstance(record[field], bool):
            raise FailureAnalysisError(f"{case_id}.{field} must be a boolean")

    for classification in (item for item in classifications if item is not None):
        detail = record[_CATEGORY_DETAIL_FIELDS[classification]]
        if detail is None or detail == []:
            raise FailureAnalysisError(
                f"{case_id} classification {classification!r} needs concrete details"
            )

    reviewer_identifier = record["reviewer_identifier"]
    reviewed_at = record["reviewed_at"]
    if reviewer_status == "completed":
        if not isinstance(reviewer_identifier, str) or not reviewer_identifier.strip():
            raise FailureAnalysisError(f"{case_id} completed review needs a reviewer")
        _aware_timestamp(reviewed_at, f"{case_id}.reviewed_at")
    elif reviewer_identifier is not None or reviewed_at is not None:
        raise FailureAnalysisError(
            f"{case_id} unresolved review cannot claim a reviewer or review time"
        )
    return record


def structural_citation_failure(attempt: Mapping[str, object] | None) -> bool:
    """Return true only for structural citation-integrity failures."""

    if attempt is None:
        return False
    return (
        attempt.get("valid_json") is True
        and attempt.get("valid_contract") is True
        and attempt.get("case_id_matches") is True
        and attempt.get("exact_evidence") is False
    )


def attempt_has_execution_failure(attempt: Mapping[str, object] | None) -> bool:
    """Return true when an attempt failed before producing a valid prediction."""

    if attempt is None:
        return True
    return (
        any(
            attempt.get(field) is False
            for field in ("valid_json", "valid_contract", "case_id_matches")
        )
        or attempt.get("failure_stage") is not None
    )


def expected_failure_case_ids(
    case_scores: Sequence[Mapping[str, object]],
    attempts: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Return every Step 5 or Step 6 failure in evaluation order."""

    failed: list[str] = []
    for item in case_scores:
        case_id = _required_string(item, "case_id")
        if (
            item.get("answer_correctness") != "correct"
            or item.get("execution_failure") is not None
            or bool(item.get("message_false_negatives"))
            or bool(item.get("unsupported_claim_count"))
            or item.get("current_vs_outdated_fact_error") is True
            or item.get("expected_abstention") != item.get("predicted_abstention")
            or attempt_has_execution_failure(attempts.get(case_id))
            or structural_citation_failure(attempts.get(case_id))
        ):
            failed.append(case_id)
    return failed


def validate_failure_records(
    *,
    records: Sequence[Mapping[str, object]],
    questions: Sequence[Mapping[str, object]],
    gold: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    case_scores: Sequence[Mapping[str, object]],
    attempts: Sequence[Mapping[str, object]],
    observations: Sequence[HistoryObservation],
) -> tuple[dict[str, object], ...]:
    """Cross-check manual records against all observable Step 5 and Step 6 data."""

    question_ids = _ordered_ids(questions, "questions")
    if len(question_ids) != 25:
        raise FailureAnalysisError("questions must contain 25 cases")
    if _ordered_ids(gold, "gold answers") != question_ids:
        raise FailureAnalysisError("gold answers do not match question order")
    if _ordered_ids(case_scores, "case scores") != question_ids:
        raise FailureAnalysisError("case scores do not match question order")

    questions_by_id = {item["case_id"]: item for item in questions}
    gold_by_id = {item["case_id"]: item for item in gold}
    predictions_by_id = {item["case_id"]: item for item in predictions}
    scores_by_id = {item["case_id"]: item for item in case_scores}
    attempts_by_id = {
        item["case_id"]: item
        for item in attempts
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    expected_ids = expected_failure_case_ids(case_scores, attempts_by_id)
    record_ids = [_required_string(item, "case_id") for item in records]
    if record_ids != expected_ids:
        raise FailureAnalysisError(
            "failure records must contain each failed case once in evaluation order; "
            f"expected {expected_ids}, got {record_ids}"
        )

    observations_by_ref = {
        (item.source_id, item.message_id): item for item in observations
    }
    validated: list[dict[str, object]] = []
    for raw in records:
        record = validate_failure_record(raw)
        case_id = record["case_id"]
        question = questions_by_id[case_id]
        gold_record = gold_by_id[case_id]
        prediction = predictions_by_id.get(case_id)
        case_score = scores_by_id[case_id]

        expected_values = {
            "capability": question.get("capability"),
            "question": question.get("question"),
            "as_of": question.get("as_of"),
            "result": case_score.get("answer_correctness"),
            "predicted_status": prediction.get("status") if prediction else "missing",
            "predicted_answer": prediction.get("answer") if prediction else "No prediction.",
            "gold_status": gold_record.get("answer_status"),
            "reference_answer": gold_record.get("reference_answer"),
            "predicted_abstention": (
                prediction is not None and prediction.get("status") == "abstained"
            ),
            "expected_abstention": gold_record.get("should_abstain"),
        }
        mismatches = [
            field
            for field, expected in expected_values.items()
            if record.get(field) != expected
        ]
        if mismatches:
            raise FailureAnalysisError(
                f"{case_id} does not match evaluation data: " + ", ".join(mismatches)
            )

        predicted_evidence = prediction.get("evidence", []) if prediction else []
        if record["predicted_evidence"] != predicted_evidence:
            raise FailureAnalysisError(f"{case_id}.predicted_evidence does not match")
        expected_evidence = _flatten_gold_evidence(gold_record, case_id)
        if record["expected_evidence"] != expected_evidence:
            raise FailureAnalysisError(f"{case_id}.expected_evidence does not match")

        missed_refs = {
            (item.get("source_id"), item.get("message_id"))
            for item in case_score.get("message_false_negatives", [])
        }
        recorded_missed_refs = {
            (item.get("source_id"), item.get("message_id"))
            for item in record["missed_evidence"]
            if isinstance(item, dict)
        }
        if recorded_missed_refs != missed_refs:
            raise FailureAnalysisError(f"{case_id}.missed_evidence does not match Step 6")

        as_of = _parse_timestamp(record["as_of"], f"{case_id}.as_of")
        inspected_refs: set[tuple[object, object]] = set()
        for evidence in record["relevant_evidence_inspected"]:
            if not isinstance(evidence, dict):
                raise FailureAnalysisError(
                    f"{case_id}.relevant_evidence_inspected must contain objects"
                )
            source_id = evidence.get("source_id")
            message_id = evidence.get("message_id")
            quote = evidence.get("quote")
            observation = observations_by_ref.get((source_id, message_id))
            if observation is None:
                raise FailureAnalysisError(
                    f"{case_id} inspected unknown evidence {(source_id, message_id)!r}"
                )
            if not isinstance(quote, str) or quote not in observation.text:
                raise FailureAnalysisError(
                    f"{case_id} inspected quote does not match {(source_id, message_id)!r}"
                )
            if observation.observed_at > as_of:
                raise FailureAnalysisError(f"{case_id} inspected evidence after as_of")
            inspected_refs.add((source_id, message_id))
        classified_refs = {
            (item.get("source_id"), item.get("message_id"))
            for field in ("predicted_evidence", "missed_evidence")
            for item in record[field]
            if isinstance(item, dict)
        }
        if not classified_refs.issubset(inspected_refs):
            raise FailureAnalysisError(
                f"{case_id} classification cites evidence that was not inspected"
            )
        validated.append(record)
    return tuple(validated)


def summarize_failures(
    *,
    records: Sequence[Mapping[str, object]],
    total_questions: int,
    run: Mapping[str, object],
    scores: Mapping[str, object],
    predictions_hash: str,
    scores_hash: str,
    protected_hashes: Mapping[str, str],
    reviewed_non_failures: Sequence[str],
    analyzed_at: datetime,
) -> dict[str, object]:
    """Build exact category and capability totals from reviewed records."""

    if analyzed_at.tzinfo is None or analyzed_at.utcoffset() is None:
        raise FailureAnalysisError("analysis timestamp must include a UTC offset")
    category_ids = {category: [] for category in FAILURE_TAXONOMY}
    primary_ids = {category: [] for category in FAILURE_TAXONOMY}
    contributing_ids = {category: [] for category in FAILURE_TAXONOMY}
    capability_ids = {capability: [] for capability in CAPABILITIES}
    execution_ids: list[str] = []
    unresolved_ids: list[str] = []
    multi_category: list[str] = []

    for record in records:
        case_id = record["case_id"]
        capability_ids[record["capability"]].append(case_id)
        if record["execution_failure"] is not None:
            execution_ids.append(case_id)
        if record["reviewer_status"] != "completed":
            unresolved_ids.append(case_id)
        classifications = [
            item
            for item in (
                record["primary_classification"],
                *record["contributing_classifications"],
            )
            if item is not None
        ]
        if len(classifications) > 1:
            multi_category.append(case_id)
        for category in classifications:
            category_ids[category].append(case_id)
        primary = record["primary_classification"]
        if primary is not None:
            primary_ids[primary].append(case_id)
        for category in record["contributing_classifications"]:
            contributing_ids[category].append(case_id)

    answer = scores.get("answer_correctness", {})
    failed_count = len(records)
    return {
        "analyzer_version": ANALYZER_VERSION,
        "baseline_id": run.get("baseline_id"),
        "run_configuration_hash": run.get("frozen_configuration_hash"),
        "dataset_hash": run.get("dataset_hash"),
        "predictions_hash": predictions_hash,
        "scores_hash": scores_hash,
        "analysis_date_utc": analyzed_at.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "report_status": "provisional" if unresolved_ids else "final",
        "total_questions": total_questions,
        "successful_cases": total_questions - failed_count,
        "failed_cases": failed_count,
        "execution_failures": len(execution_ids),
        "execution_failure_case_ids": execution_ids,
        "reasoning_failures": failed_count - len(execution_ids),
        "category_counts": {
            category: len(category_ids[category]) for category in FAILURE_TAXONOMY
        },
        "category_case_ids": category_ids,
        "failures_by_capability": {
            capability: {
                "count": len(capability_ids[capability]),
                "case_ids": capability_ids[capability],
            }
            for capability in CAPABILITIES
        },
        "primary_category_counts": {
            category: len(primary_ids[category]) for category in FAILURE_TAXONOMY
        },
        "contributing_category_counts": {
            category: len(contributing_ids[category]) for category in FAILURE_TAXONOMY
        },
        "multi_category_cases": multi_category,
        "unresolved_reviews": len(unresolved_ids),
        "unresolved_case_ids": unresolved_ids,
        "reviewed_non_failure_evidence_case_ids": list(reviewed_non_failures),
        "step6_answer_results": {
            "correct": answer.get("fully_correct"),
            "partially_correct": answer.get("partially_correct"),
            "incorrect": answer.get("incorrect"),
            "unresolved": answer.get("unresolved"),
        },
        "protected_artifact_hashes": dict(protected_hashes),
        "protected_artifacts_unchanged": True,
    }


def render_findings(
    summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    run: Mapping[str, object],
    prediction_statuses: Sequence[str],
) -> str:
    """Render a concise, evidence-based report without proposing fixes."""

    answer = summary["step6_answer_results"]
    abstained = sum(status == "abstained" for status in prediction_statuses)
    lines = [
        "# B1 full-history baseline findings",
        "",
        "## Run identity",
        "",
        f"- Provider: {run['provider']}",
        f"- Model: {run['resolved_model_version']}",
        f"- Prompt: {run['prompt_version']}",
        f"- Dataset hash: `{run['dataset_hash']}`",
        f"- Configuration hash: `{run['frozen_configuration_hash']}`",
        "",
        "## Baseline result",
        "",
        f"- Total questions: {summary['total_questions']}",
        f"- Correct answers: {answer['correct']}",
        f"- Partially correct answers: {answer['partially_correct']}",
        f"- Incorrect answers: {answer['incorrect']}",
        f"- Abstained predictions: {abstained}",
        f"- Execution failures: {summary['execution_failures']}",
        f"- Unresolved cases: {summary['unresolved_reviews']}",
        "",
        "## Failure distribution",
        "",
        "| Category | Cases |",
        "| --- | ---: |",
    ]
    for category in FAILURE_TAXONOMY:
        lines.append(f"| `{category}` | {summary['category_counts'][category]} |")
    lines.extend(
        [
            "",
            "| Capability | Failed cases |",
            "| --- | ---: |",
        ]
    )
    for capability in CAPABILITIES:
        lines.append(
            f"| `{capability}` | "
            f"{summary['failures_by_capability'][capability]['count']} |"
        )
    multi = summary["multi_category_cases"]
    lines.extend(
        [
            "",
            "Overlapping categories: " + (", ".join(multi) if multi else "none") + ".",
            "",
            "## Concrete findings",
            "",
            (
                f"- Evidence coverage was the main failure. {summary['category_counts']['missed_evidence']} "
                "cases omitted at least one gold-designated message. Conversation evidence was missed more "
                "often than email or calendar evidence. Some omitted messages were contextual rather than "
                "independent support."
            ),
            (
                '- In `temporal_004`, the omitted conversation says "yesterday" on June 27, which points to '
                "June 26. The exact email says June 25. Its failure count therefore reflects exact gold "
                "evidence recall, not stronger date support."
            ),
            (
                f"- One answer used April for Maya's marketing start even though its cited offer states May 4. "
                f"The `wrong_date` count is {summary['category_counts']['wrong_date']}."
            ),
            (
                "- The baseline applied the explicit Aryan and Kids Spark corrections. "
                f"Stale-information and failed-correction counts are "
                f"{summary['category_counts']['used_stale_information']} and "
                f"{summary['category_counts']['failed_to_apply_correction']}."
            ),
            (
                "- It did not turn Maya's stated suspicions into confirmed facts, and it made no unsupported "
                f"stable-trait inference. Those counts are "
                f"{summary['category_counts']['treated_assumption_as_fact']} and "
                f"{summary['category_counts']['unsupported_user_inference']}."
            ),
            (
                "- Abstention matched all five abstention cases. The two abstention-error counts are "
                f"{summary['category_counts']['should_have_abstained']} and "
                f"{summary['category_counts']['abstained_despite_sufficient_evidence']}."
            ),
            (
                "- Every citation passed the structural and verbatim-quote checks. "
                f"The invalid-citation count is {summary['category_counts']['invalid_citation']}."
            ),
            (
                "- Three correct answers cited additional non-gold evidence. Manual inspection found that "
                "the extra citations were valid and supported the answers, so they are not listed as failures."
            ),
            "",
            "## Failed cases",
            "",
            "| Case | Capability | Result | Primary failure | Contributing failures | Explanation |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        contributing = ", ".join(record["contributing_classifications"]) or "none"
        primary = record["primary_classification"] or "execution failure"
        explanation = str(record["concise_explanation"]).replace("|", "\\|")
        lines.append(
            f"| `{record['case_id']}` | `{record['capability']}` | {record['result']} | "
            f"`{primary}` | {contributing} | {explanation} |"
        )
    lines.extend(
        [
            "",
            "## Baseline interpretation",
            "",
            (
                "This first B1 run is the starting measurement. It does not establish a target or say whether "
                "the score is good or bad. Later memory designs can be compared by whether they reduce the "
                f"{summary['category_counts']['missed_evidence']} missed-evidence cases and the "
                f"{summary['category_counts']['wrong_date']} wrong-date case without introducing failures in "
                "the zero-count categories."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_failure_analysis(
    *,
    repo_root: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    sources_dir: str | Path = DEFAULT_SOURCES_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Validate manual reviews, preserve prior artifacts, and write Step 7 outputs."""

    root = Path(repo_root).resolve()
    results = _under_root(root, results_dir)
    config_file = _under_root(root, config_path)
    config = load_frozen_config(config_file)
    protected = _protected_paths(root, results, config_file, config)
    hashes_before = _hash_paths(protected, root)

    run = _load_json_object(results / "run.json", "run")
    if run.get("run_status") not in {"completed", "completed_with_failures"}:
        raise FailureAnalysisError("Step 5 run is not complete")
    verify_frozen_content(config, root)
    _verify_run_config(run, config)

    scores = _load_json_object(results / "scores.json", "scores")
    questions = _load_jsonl(_under_root(root, questions_path), "questions")
    gold = _load_jsonl(_under_root(root, gold_path), "gold answers")
    predictions = _load_jsonl(results / "predictions.jsonl", "predictions")
    case_scores = _load_jsonl(results / "case_scores.jsonl", "case scores")
    manual_reviews = _load_jsonl(results / "manual_review.jsonl", "manual reviews")
    failures = _load_jsonl(results / "failures.jsonl", "execution failures")
    analysis_records = _load_jsonl(results / "failure_analysis.jsonl", "failure analysis")
    observations = load_history_observations(_under_root(root, sources_dir))
    attempts = run.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 25:
        raise FailureAnalysisError("Step 5 must contain 25 attempts")
    _verify_step6(
        scores=scores,
        case_scores=case_scores,
        manual_reviews=manual_reviews,
        predictions_path=results / "predictions.jsonl",
        gold_path=_under_root(root, gold_path),
    )
    if len(failures) != scores.get("execution_failure_count"):
        raise FailureAnalysisError("Step 5 and Step 6 execution failures do not match")
    for record in predictions:
        try:
            validate_prediction(record)
        except PredictionValidationError as error:
            raise FailureAnalysisError(str(error)) from error

    validated = validate_failure_records(
        records=analysis_records,
        questions=questions,
        gold=gold,
        predictions=predictions,
        case_scores=case_scores,
        attempts=attempts,
        observations=observations,
    )
    reviewed_non_failures = [
        item["case_id"]
        for item in case_scores
        if item.get("message_false_positives")
        and item["case_id"] not in {record["case_id"] for record in validated}
    ]
    analysis_time = (now or (lambda: datetime.now(timezone.utc)))()
    hashes_after_review = _hash_paths(protected, root)
    if hashes_before != hashes_after_review:
        raise FailureAnalysisError("a protected artifact changed during failure analysis")
    summary = summarize_failures(
        records=validated,
        total_questions=len(questions),
        run=run,
        scores=scores,
        predictions_hash=hashes_before[_relative(results / "predictions.jsonl", root)],
        scores_hash=hashes_before[_relative(results / "scores.json", root)],
        protected_hashes=hashes_before,
        reviewed_non_failures=reviewed_non_failures,
        analyzed_at=analysis_time,
    )
    findings = render_findings(
        summary,
        validated,
        run,
        [record["status"] for record in predictions],
    )
    _write_json_atomic(results / "failure_summary.json", summary)
    _write_text_atomic(results / "baseline_findings.md", findings)
    hashes_after_write = _hash_paths(protected, root)
    if hashes_before != hashes_after_write:
        raise FailureAnalysisError("a protected artifact changed while writing outputs")
    return summary


def _verify_run_config(run: Mapping[str, object], config: FrozenBaselineConfig) -> None:
    expected = {
        "frozen_configuration_hash": config.configuration_sha256,
        "dataset_hash": config.dataset_sha256,
        "provider": config.provider,
        "requested_model_version": config.requested_model,
        "resolved_model_version": config.resolved_model,
        "temperature": config.temperature,
        "generation_settings": dict(config.generation_settings),
        "prompt_version": config.prompt_version,
        "prompt_fingerprint": config.prompt_sha256,
    }
    mismatches = [field for field, value in expected.items() if run.get(field) != value]
    if mismatches:
        raise FailureAnalysisError(
            "Step 5 does not match the frozen Step 4 configuration: "
            + ", ".join(mismatches)
        )
    if any(run.get(field) != 25 for field in ("total_questions", "recorded_cases")):
        raise FailureAnalysisError("Step 5 did not account for all 25 questions")


def _verify_step6(
    *,
    scores: Mapping[str, object],
    case_scores: Sequence[Mapping[str, object]],
    manual_reviews: Sequence[Mapping[str, object]],
    predictions_path: Path,
    gold_path: Path,
) -> None:
    if len(case_scores) != 25:
        raise FailureAnalysisError("Step 6 must contain 25 per-case scores")
    unresolved = scores.get("unresolved_manual_review_count")
    if not isinstance(unresolved, int):
        raise FailureAnalysisError("Step 6 unresolved review count is invalid")
    completed_manual = sum(item.get("reviewer_judgment") is not None for item in manual_reviews)
    if completed_manual + unresolved != len(manual_reviews):
        raise FailureAnalysisError("Step 6 manual review state is inconsistent")
    current_predictions_hash = _file_sha256(predictions_path)
    current_gold_hash = _file_sha256(gold_path)
    if (
        scores.get("predictions_sha256") != current_predictions_hash
        or scores.get("predictions_sha256_after") != current_predictions_hash
        or scores.get("predictions_unchanged") is not True
    ):
        raise FailureAnalysisError("predictions.jsonl does not match Step 6 hashes")
    if (
        scores.get("gold_answer_sha256") != current_gold_hash
        or scores.get("gold_answer_sha256_after") != current_gold_hash
        or scores.get("gold_answer_unchanged") is not True
    ):
        raise FailureAnalysisError("eval_answer.jsonl does not match Step 6 hashes")


def _flatten_gold_evidence(
    gold: Mapping[str, object], case_id: str
) -> list[dict[str, object]]:
    evidence = gold.get("evidence")
    if not isinstance(evidence, list):
        raise FailureAnalysisError(f"{case_id} gold evidence must be a list")
    flattened: list[dict[str, object]] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise FailureAnalysisError(f"{case_id} gold evidence item must be an object")
        source_id = item.get("source_id")
        message_ids = item.get("message_ids")
        quotes = item.get("quotes")
        if not isinstance(source_id, str) or not isinstance(message_ids, list) or not isinstance(quotes, list):
            raise FailureAnalysisError(f"{case_id} gold evidence item is invalid")
        if not message_ids:
            if len(quotes) != 1:
                raise FailureAnalysisError(f"{case_id} calendar evidence needs one quote")
            flattened.append(
                {"source_id": source_id, "message_id": None, "quote": quotes[0]}
            )
            continue
        if len(message_ids) == 1:
            flattened.extend(
                {"source_id": source_id, "message_id": message_ids[0], "quote": quote}
                for quote in quotes
            )
            continue
        if len(message_ids) != len(quotes):
            raise FailureAnalysisError(f"{case_id} gold message IDs and quotes do not align")
        flattened.extend(
            {"source_id": source_id, "message_id": message_id, "quote": quote}
            for message_id, quote in zip(message_ids, quotes)
        )
    return flattened


def _protected_paths(
    root: Path,
    results: Path,
    config_file: Path,
    config: FrozenBaselineConfig,
) -> tuple[Path, ...]:
    paths = [
        config_file,
        *(root / item for item in config.dataset_files),
        *(results / name for name in (
            "run.json",
            "predictions.jsonl",
            "failures.jsonl",
            "scores.json",
            "case_scores.jsonl",
            "manual_review.jsonl",
        )),
    ]
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _hash_paths(paths: Sequence[Path], root: Path) -> dict[str, str]:
    return {_relative(path, root): _file_sha256(path) for path in paths}


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise FailureAnalysisError(f"could not hash {path}: {error}") from error


def _required_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FailureAnalysisError(f"{field} must be a non-empty string")
    return value


def _aware_timestamp(value: object, location: str) -> None:
    _parse_timestamp(value, location)


def _parse_timestamp(value: object, location: str) -> datetime:
    if not isinstance(value, str):
        raise FailureAnalysisError(f"{location} must be an ISO timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise FailureAnalysisError(f"{location} must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FailureAnalysisError(f"{location} must include a UTC offset")
    return parsed


def _ordered_ids(records: Sequence[Mapping[str, object]], description: str) -> list[str]:
    values = [_required_string(item, "case_id") for item in records]
    if len(values) != len(set(values)):
        raise FailureAnalysisError(f"{description} contain duplicate case IDs")
    return values


def _load_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailureAnalysisError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FailureAnalysisError(f"{description} must be an object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FailureAnalysisError(f"could not read {description} {path}: {error}") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise FailureAnalysisError(
                f"{description} {path}:{line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise FailureAnalysisError(f"{description} {path}:{line_number} must be an object")
        records.append(value)
    return records


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    _write_text_atomic(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise FailureAnalysisError(f"path is outside the repository: {path}") from error


def _under_root(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    _relative(resolved, root)
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect failures from the completed B1 full-history baseline."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    args = _build_parser().parse_args(argv)
    try:
        summary = run_failure_analysis(
            repo_root=Path.cwd(), results_dir=args.results_dir
        )
    except Exception as error:
        print(f"error={error}", file=sys.stderr)
        return 1
    print(f"report_status={summary['report_status']}", file=stdout)
    print(f"failed_cases={summary['failed_cases']}", file=stdout)
    print(f"execution_failures={summary['execution_failures']}", file=stdout)
    print(f"reasoning_failures={summary['reasoning_failures']}", file=stdout)
    print(f"failure_analysis={args.results_dir / 'failure_analysis.jsonl'}", file=stdout)
    print(f"failure_summary={args.results_dir / 'failure_summary.json'}", file=stdout)
    print(f"baseline_findings={args.results_dir / 'baseline_findings.md'}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
