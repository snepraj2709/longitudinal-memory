"""Run the fixed five-case smoke test for the full-history baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Protocol, Sequence, TextIO

from .history import (
    EvaluationQuestion,
    HistoryObservation,
    HistoryPrompt,
    build_history_prompt,
    load_evaluation_questions,
    load_history_observations,
)
from .prediction import (
    BaselinePrediction,
    PredictionValidationError,
    validate_prediction,
)


SMOKE_CASE_IDS = (
    "extraction_001",
    "temporal_003",
    "conflict_004",
    "user_modeling_001",
    "abstention_001",
)

_CAPABILITY_CRITERIA = {
    "extraction_001": "identifies the supported memories, role and cites its source",
    "temporal_003": "uses the later explicit correction instead of the older date",
    "conflict_004": "uses the supported reason and does not state Maya's assumption as fact",
    "user_modeling_001": "describes a career change over time, not a permanent preference",
    "abstention_001": "abstains with empty evidence and a non-empty reason",
}


class SmokeTestSelectionError(ValueError):
    """Raised when the fixed smoke cases cannot be selected exactly once."""


class EvidenceValidationError(ValueError):
    """Raised when prediction evidence does not match the supplied history."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class AnswerModelClient(Protocol):
    """Small provider-independent interface used by the smoke-test runner."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw response for one prompt."""


@dataclass(frozen=True)
class ModelRunConfig:
    """Non-secret model metadata recorded with a smoke-test run."""

    provider: str
    model: str
    settings: Mapping[str, object]


@dataclass(frozen=True)
class SmokeCaseResult:
    """Deterministic checks and diagnostics for one requested case."""

    case_id: str
    raw_response: str | None
    valid_json: bool
    valid_contract: bool
    case_id_matches: bool
    exact_evidence: bool
    prediction: BaselinePrediction | None
    validation_error: str | None

    @property
    def passed(self) -> bool:
        return (
            self.valid_json
            and self.valid_contract
            and self.case_id_matches
            and self.exact_evidence
            and self.prediction is not None
        )


@dataclass(frozen=True)
class SmokeTestRun:
    """The ordered results and model metadata for one five-case run."""

    model_config: ModelRunConfig
    results: tuple[SmokeCaseResult, ...]

    @property
    def exit_code(self) -> int:
        return 0 if all(result.passed for result in self.results) else 1


@dataclass(frozen=True)
class SmokeTestArtifacts:
    """Paths written for predictions, diagnostics, and manual review."""

    predictions_path: Path
    diagnostics_path: Path
    report_path: Path


def select_smoke_questions(
    questions: Sequence[EvaluationQuestion],
) -> tuple[EvaluationQuestion, ...]:
    """Select the fixed smoke cases once each and in benchmark order."""

    by_case_id: dict[str, list[EvaluationQuestion]] = {}
    for question in questions:
        by_case_id.setdefault(question.case_id, []).append(question)

    errors: list[str] = []
    selected: list[EvaluationQuestion] = []
    for case_id in SMOKE_CASE_IDS:
        matches = by_case_id.get(case_id, [])
        if not matches:
            errors.append(f"missing smoke-test case {case_id!r}")
        elif len(matches) > 1:
            errors.append(
                f"smoke-test case {case_id!r} appears {len(matches)} times"
            )
        else:
            selected.append(matches[0])

    if errors:
        raise SmokeTestSelectionError("; ".join(errors))
    return tuple(selected)


def build_smoke_prompts(
    questions: Sequence[EvaluationQuestion],
    observations: tuple[HistoryObservation, ...],
) -> tuple[HistoryPrompt, ...]:
    """Build prompts for only the fixed smoke cases."""

    return tuple(
        build_history_prompt(question, observations)
        for question in select_smoke_questions(questions)
    )


def validate_prediction_evidence(
    prediction: BaselinePrediction,
    observations: Sequence[HistoryObservation],
) -> None:
    """Require every citation and quote to match one supplied observation."""

    by_reference = {
        (observation.source_id, observation.message_id): observation
        for observation in observations
    }
    errors: list[str] = []

    for index, evidence in enumerate(prediction.evidence):
        reference = (evidence.source_id, evidence.message_id)
        observation = by_reference.get(reference)
        if observation is None:
            errors.append(
                f"evidence[{index}] references unknown source/message pair "
                f"{reference!r}"
            )
            continue
        if observation.source_type == "calendar" and evidence.message_id is not None:
            errors.append(f"evidence[{index}] calendar evidence must use message_id null")
        if observation.source_type != "calendar" and evidence.message_id is None:
            errors.append(
                f"evidence[{index}] non-calendar evidence must use a message_id"
            )
        if evidence.quote not in observation.text:
            errors.append(
                f"evidence[{index}].quote is not an exact substring of the cited text"
            )

    if errors:
        raise EvidenceValidationError(errors)


def run_smoke_test(
    questions: Sequence[EvaluationQuestion],
    observations: tuple[HistoryObservation, ...],
    client: AnswerModelClient,
    model_config: ModelRunConfig,
) -> SmokeTestRun:
    """Call the answer model and validate all five responses in fixed order."""

    selected = select_smoke_questions(questions)
    results: list[SmokeCaseResult] = []

    for question in selected:
        prompt = build_history_prompt(question, observations)
        eligible_observations = tuple(
            observation
            for observation in observations
            if observation.observed_at <= question.as_of
        )
        raw_response: str | None = None
        try:
            raw_response = client.complete(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
            )
        except Exception as error:  # Provider adapters decide their exception types.
            results.append(
                SmokeCaseResult(
                    case_id=question.case_id,
                    raw_response=None,
                    valid_json=False,
                    valid_contract=False,
                    case_id_matches=False,
                    exact_evidence=False,
                    prediction=None,
                    validation_error=f"model call failed: {error}",
                )
            )
            continue

        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as error:
            results.append(
                SmokeCaseResult(
                    case_id=question.case_id,
                    raw_response=raw_response,
                    valid_json=False,
                    valid_contract=False,
                    case_id_matches=False,
                    exact_evidence=False,
                    prediction=None,
                    validation_error=f"invalid JSON: {error}",
                )
            )
            continue

        try:
            prediction = validate_prediction(parsed)
        except PredictionValidationError as error:
            results.append(
                SmokeCaseResult(
                    case_id=question.case_id,
                    raw_response=raw_response,
                    valid_json=True,
                    valid_contract=False,
                    case_id_matches=False,
                    exact_evidence=False,
                    prediction=None,
                    validation_error=f"prediction contract: {error}",
                )
            )
            continue

        case_id_matches = prediction.case_id == question.case_id
        evidence_error: str | None = None
        try:
            validate_prediction_evidence(prediction, eligible_observations)
            exact_evidence = True
        except EvidenceValidationError as error:
            exact_evidence = False
            evidence_error = f"evidence validation: {error}"

        errors: list[str] = []
        if not case_id_matches:
            errors.append(
                f"returned case_id {prediction.case_id!r} does not match "
                f"requested case {question.case_id!r}"
            )
        if evidence_error:
            errors.append(evidence_error)

        results.append(
            SmokeCaseResult(
                case_id=question.case_id,
                raw_response=raw_response,
                valid_json=True,
                valid_contract=True,
                case_id_matches=case_id_matches,
                exact_evidence=exact_evidence,
                prediction=prediction,
                validation_error="; ".join(errors) or None,
            )
        )

    return SmokeTestRun(model_config=model_config, results=tuple(results))


def write_smoke_test_artifacts(
    run: SmokeTestRun,
    output_dir: str | Path,
    capability_reviews: Mapping[str, str] | None = None,
) -> SmokeTestArtifacts:
    """Write validated predictions, raw diagnostics, and a review table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    diagnostics_path = output_dir / "diagnostics.jsonl"
    report_path = output_dir / "report.md"

    prediction_lines = [
        json.dumps(
            _prediction_record(result.prediction),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for result in run.results
        if result.passed and result.prediction is not None
    ]
    predictions_path.write_text(
        "\n".join(prediction_lines) + ("\n" if prediction_lines else ""),
        encoding="utf-8",
    )

    diagnostic_lines = [
        json.dumps(
            {
                "case_id": result.case_id,
                "raw_response": result.raw_response,
                "valid_json": result.valid_json,
                "valid_contract": result.valid_contract,
                "case_id_matches": result.case_id_matches,
                "exact_evidence": result.exact_evidence,
                "validation_error": result.validation_error,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for result in run.results
    ]
    diagnostics_path.write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8")

    report_path.write_text(
        _render_report(run, capability_reviews or {}),
        encoding="utf-8",
    )
    return SmokeTestArtifacts(
        predictions_path=predictions_path,
        diagnostics_path=diagnostics_path,
        report_path=report_path,
    )


def _prediction_record(prediction: BaselinePrediction) -> dict[str, object]:
    return {
        "case_id": prediction.case_id,
        "status": prediction.status,
        "answer": prediction.answer,
        "confidence": prediction.confidence,
        "evidence": [
            {
                "source_id": evidence.source_id,
                "message_id": evidence.message_id,
                "quote": evidence.quote,
            }
            for evidence in prediction.evidence
        ],
        "abstention_reason": prediction.abstention_reason,
    }


def _render_report(
    run: SmokeTestRun,
    capability_reviews: Mapping[str, str],
) -> str:
    settings = json.dumps(
        dict(run.model_config.settings), ensure_ascii=False, sort_keys=True
    )
    lines = [
        "# Full-history smoke-test report",
        "",
        f"Provider: {run.model_config.provider}",
        f"Model: {run.model_config.model}",
        f"Settings: `{settings}`",
        "",
        "| Case | Valid JSON | Valid contract | Exact evidence | Capability behavior | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in run.results:
        capability = capability_reviews.get(
            result.case_id,
            f"Not reviewed. Check whether it {_CAPABILITY_CRITERIA[result.case_id]}.",
        )
        note = result.validation_error or "Ready for manual capability review."
        lines.append(
            "| "
            + " | ".join(
                (
                    _table_text(result.case_id),
                    _yes_no(result.valid_json),
                    _yes_no(result.valid_contract),
                    _yes_no(result.exact_evidence),
                    _table_text(capability),
                    _table_text(note),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fixed five full-history smoke-test prompts."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/pilot/sources"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/pilot/evaluation/eval_questions.jsonl"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and print case metadata without calling a model.",
    )
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Run the CLI without selecting or configuring a model provider."""

    stdout = stdout or sys.stdout
    args = _build_parser().parse_args(argv)
    observations = load_history_observations(args.source_dir)
    questions = load_evaluation_questions(args.questions)
    prompts = build_smoke_prompts(questions, observations)

    if not args.dry_run:
        print(
            "No answer provider/model is configured. Use --dry-run or inject an "
            "approved AnswerModelClient through run_smoke_test().",
            file=stdout,
        )
        return 2

    for prompt in prompts:
        print(
            json.dumps(
                {
                    "case_id": prompt.case_id,
                    "observation_count": prompt.observation_count,
                },
                separators=(",", ":"),
            ),
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
