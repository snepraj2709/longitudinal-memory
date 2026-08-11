"""Deterministic structural report for the frozen no-call answer run."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from retrieval.query_contracts import RequestedValidTime

from .answer_contracts import BASELINE_ORDER, MemoryAnswer
from .answer_run_contracts import (
    AVAILABLE_BASELINES,
    BASELINES,
    FINAL_RELEASE,
    METRICS,
    NULL_REASONS,
    RUN_VERSION,
    SCORER_VERSION,
    AnswerQualityChecks,
    AnswerQualityFailure,
    AnswerQualityScorecard,
    AnswerRunError,
    MetricValue,
    PerAnswerQuality,
)
from .comparable_answer_run import (
    B1_CONFIG_SHA256,
    CONFIG_PATH,
    EMPTY_SHA256,
    GUIDANCE_SHA256,
    GUIDANCE_VERSION,
    MODEL,
    PROMPT_SHA256,
    RUNTIME_ROOT,
    STARTING_COMMIT,
    STEP81_MANIFEST_SHA256,
    STEP81_PACKAGES_SHA256,
    STEP82_ANSWERS_SHA256,
    STEP82_DATASET_SHA256,
    STEP82_MANIFEST_SHA256,
    freeze_answer_predictions,
    load_comparable_answer_run_config,
    verify_answer_runtime_checkpoint,
)
from .contracts import canonical_json_bytes
from .memory_answer import memory_answer_from_mapping


DATASET_MANIFEST = Path("data/answering/memory-answer-quality-development-v1/manifest.json")
RESULT_ROOT = Path("results/answering/memory-answer-quality-development-v1")
ARTIFACTS = (
    "per-case.jsonl",
    "scorecard.json",
    "checks.json",
    "failures.jsonl",
    "run.json",
    "findings.md",
)
IMPLEMENTATION_PATHS = (
    "configs/answering/comparable_answer_run_v1.json",
    "src/answering/answer_run_contracts.py",
    "src/answering/comparable_answer_run.py",
    "src/answering/answer_quality_evaluation.py",
)


def score_frozen_answers(
    output_dir: str | Path = RESULT_ROOT,
    *,
    runtime_dir: str | Path = RUNTIME_ROOT,
    repo_root: str | Path = ".",
) -> AnswerQualityChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    runtime = _resolve(root, runtime_dir)
    _require_empty(output)
    checkpoint = verify_answer_runtime_checkpoint(runtime, repo_root=root)
    config, config_sha256 = load_comparable_answer_run_config(repo_root=root)
    dataset = _dataset(root, config_sha256)
    answers = _runtime_answers(runtime)
    per_case = tuple(_per_case(answer) for answer in answers)
    scorecard = _scorecard()
    failures: tuple[AnswerQualityFailure, ...] = ()
    checks = _checks(per_case, scorecard, failures)
    payloads = _artifact_payloads(checks, per_case, scorecard, failures, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        _write_exclusive(output / name, payloads[name])
    _write_exclusive(
        output / "manifest.json",
        canonical_json_bytes(_manifest(root, dataset, config_sha256, checks, payloads, runtime)),
    )
    verify_answer_quality_release(output, runtime_dir=runtime, repo_root=root)
    return checks


def verify_answer_quality_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    runtime_dir: str | Path = RUNTIME_ROOT,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    runtime = _resolve(root, runtime_dir)
    checkpoint = verify_answer_runtime_checkpoint(runtime, repo_root=root)
    _, config_sha256 = load_comparable_answer_run_config(repo_root=root)
    dataset = _dataset(root, config_sha256)
    answers = _runtime_answers(runtime)
    expected_per_case = tuple(_per_case(answer) for answer in answers)
    per_case = tuple(_per_case_from_mapping(value) for value in _read_jsonl(output / "per-case.jsonl"))
    if per_case != expected_per_case:
        raise AnswerRunError("per-case report changed")
    scorecard = _scorecard_from_mapping(_read_object(output / "scorecard.json"))
    expected_scorecard = _scorecard()
    if scorecard != expected_scorecard:
        raise AnswerRunError("scorecard changed")
    failures = tuple(_failure_from_mapping(value) for value in _read_jsonl(output / "failures.jsonl"))
    checks = _checks(per_case, scorecard, failures)
    if _read_object(output / "checks.json") != asdict(checks):
        raise AnswerRunError("quality checks do not recompute")
    payloads = _artifact_payloads(checks, per_case, scorecard, failures, checkpoint)
    for name in ARTIFACTS:
        if (output / name).read_bytes() != payloads[name]:
            raise AnswerRunError("quality artifact does not recompute")
    manifest = _read_object(output / "manifest.json")
    expected_manifest = _manifest(root, dataset, config_sha256, checks, payloads, runtime)
    if manifest != expected_manifest:
        raise AnswerRunError("quality manifest does not recompute")
    for name in ARTIFACTS:
        if manifest["artifacts"].get(name) != _sha(output / name):
            raise AnswerRunError("quality artifact hash changed")


def execute_answer_quality_evaluation(
    runtime_dir: str | Path = RUNTIME_ROOT,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> AnswerQualityChecks:
    root = Path(repo_root).resolve()
    runtime = _resolve(root, runtime_dir)
    freeze_answer_predictions(runtime, repo_root=root)
    return score_frozen_answers(output_dir, runtime_dir=runtime, repo_root=root)


def _runtime_answers(runtime: Path) -> tuple[MemoryAnswer, ...]:
    raw = (runtime / "predictions.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != STEP82_ANSWERS_SHA256:
        raise AnswerRunError("runtime predictions changed")
    answers = tuple(memory_answer_from_mapping(value) for value in _read_jsonl_bytes(raw))
    expected = tuple(sorted(
        answers, key=lambda item: (item.query_id, BASELINE_ORDER[item.baseline_id])
    ))
    if answers != expected or len(answers) != 24:
        raise AnswerRunError("runtime prediction ordering changed")
    if any(
        answer.status != "abstained"
        or answer.structural_blockers != ("no_promoted_claims",)
        or answer.statements
        or answer.requested_model is not None
        or answer.resolved_model is not None
        for answer in answers
    ):
        raise AnswerRunError("runtime prediction is not a structural abstention")
    return answers


def _metric(metric: str, baseline: str) -> MetricValue:
    reason = NULL_REASONS[metric] if baseline in AVAILABLE_BASELINES else "baseline_not_available"
    return MetricValue(metric, 0, 0, None, reason)


def _per_case(answer: MemoryAnswer) -> PerAnswerQuality:
    return PerAnswerQuality(
        answer.query_id,
        answer.user_id,
        answer.baseline_id,
        answer.answer_id,
        answer.package_id,
        answer.package_sha256,
        answer.execution_id,
        answer.plan_id,
        answer.snapshot_run_id,
        answer.as_of,
        answer.requested_valid_time,
        answer.status,
        answer.structural_blockers,
        len(answer.statements),
        sum(len(statement.citations) for statement in answer.statements),
        tuple(_metric(metric, answer.baseline_id) for metric in METRICS),
    )


def _scorecard() -> AnswerQualityScorecard:
    rows = []
    for baseline in BASELINES:
        count = 8 if baseline in AVAILABLE_BASELINES else 0
        for metric in METRICS:
            rows.append({
                "baseline_id": baseline,
                "metric": metric,
                "case_count": count,
                "abstained_count": count,
                "value": _metric(metric, baseline),
            })
    return AnswerQualityScorecard(SCORER_VERSION, tuple(rows))


def _checks(
    per_case: Sequence[PerAnswerQuality],
    scorecard: AnswerQualityScorecard,
    failures: Sequence[AnswerQualityFailure],
) -> AnswerQualityChecks:
    values = AnswerQualityChecks(
        FINAL_RELEASE,
        len(per_case),
        len(scorecard.rows),
        sum(item.baseline_id == "B2" for item in per_case),
        sum(item.baseline_id == "B3" for item in per_case),
        sum(item.baseline_id == "B4" for item in per_case),
        0,
        0,
        sum(item.status == "abstained" for item in per_case),
        sum(item.status != "abstained" for item in per_case),
        sum(item.statement_count for item in per_case),
        sum(item.citation_count for item in per_case),
        sum(
            metric.value is not None
            for item in per_case for metric in item.metrics
        ) + sum(
            row["value"].value is not None for row in scorecard.rows
        ),
        len(failures),
        0,
    )
    if len({(item.answer_id, item.package_id) for item in per_case}) != len(per_case):
        raise AnswerRunError("quality identities are duplicated")
    return values


def _artifact_payloads(checks, per_case, scorecard, failures, checkpoint) -> dict[str, bytes]:
    return {
        "per-case.jsonl": b"".join(canonical_json_bytes(item) for item in per_case),
        "scorecard.json": canonical_json_bytes(scorecard),
        "checks.json": canonical_json_bytes(checks),
        "failures.jsonl": b"".join(canonical_json_bytes(item) for item in failures),
        "run.json": canonical_json_bytes({
            "run_version": RUN_VERSION,
            "scorer_version": SCORER_VERSION,
            "final_release": FINAL_RELEASE,
            "runtime_checkpoint_sha256": hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest(),
            "available_baselines": list(AVAILABLE_BASELINES),
            "deferred_baselines": ["B5", "B6"],
            "prediction_count": 24,
            "provider_eligible_case_count": 0,
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
            "requested_model": MODEL,
            "configured_resolved_model": MODEL,
            "provider_returned_model": None,
            "answer_gold_available": False,
        }),
        "findings.md": _findings().encode("utf-8"),
    }


def _manifest(root, dataset, config_sha256, checks, payloads, runtime) -> Mapping[str, object]:
    return {
        "final_release": FINAL_RELEASE,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "dataset": dataset,
        "dataset_manifest_sha256": _sha(root / DATASET_MANIFEST),
        "run_configuration_sha256": config_sha256,
        "b1_comparison_configuration_sha256": B1_CONFIG_SHA256,
        "input_authorities": {
            "step8_1_result_manifest": STEP81_MANIFEST_SHA256,
            "step8_1_packages": STEP81_PACKAGES_SHA256,
            "step8_2_dataset_manifest": STEP82_DATASET_SHA256,
            "step8_2_result_manifest": STEP82_MANIFEST_SHA256,
            "step8_2_answers": STEP82_ANSWERS_SHA256,
            "runtime_preflight": _sha(runtime / "preflight.json"),
            "runtime_predictions": _sha(runtime / "predictions.jsonl"),
            "runtime_failures": _sha(runtime / "failures.jsonl"),
            "runtime_checkpoint": _sha(runtime / "checkpoint_manifest.json"),
        },
        "implementation_hashes": {path: _sha(root / path) for path in IMPLEMENTATION_PATHS},
        "artifacts": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        "checks": asdict(checks),
        "null_reason_policy": {
            "available_baselines": NULL_REASONS,
            "unavailable_baselines": "baseline_not_available",
        },
        "model_usage": {
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
            "provider_returned_model": None,
        },
        "excluded_inputs": dataset["excluded_inputs"],
        "limitations": [
            "The release contains no factual answers and no authorized answer-quality reference.",
            "B5 and B6 have no frozen input release and are unavailable.",
            "The report proves reproducibility and structural safety only.",
        ],
        "predecessor_drift": [],
    }


def _dataset(root: Path, config_sha256: str) -> Mapping[str, object]:
    value = _read_object(root / DATASET_MANIFEST)
    expected = {
        "dataset_version": FINAL_RELEASE,
        "split": "development",
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "run_configuration_sha256": config_sha256,
        "input_release": {
            "step8_1_result_manifest_sha256": STEP81_MANIFEST_SHA256,
            "step8_1_packages_sha256": STEP81_PACKAGES_SHA256,
            "step8_2_dataset_manifest_sha256": STEP82_DATASET_SHA256,
            "step8_2_result_manifest_sha256": STEP82_MANIFEST_SHA256,
            "step8_2_answers_sha256": STEP82_ANSWERS_SHA256,
        },
        "available_baselines": ["B2", "B3", "B4"],
        "deferred_baselines": ["B5", "B6"],
        "expected_prediction_count": 24,
        "expected_per_case_count": 24,
        "expected_scorecard_row_count": 35,
        "answer_gold_available": False,
        "provider_execution": "forbidden_zero_call",
        "excluded_inputs": [
            "answer gold", "retrieval relevance annotations", "oracle data",
            "review queues", "frozen test users", ".env", "provider credentials",
            "network services",
        ],
    }
    if value != expected:
        raise AnswerRunError("quality dataset manifest changed")
    return value


def _findings() -> str:
    return (
        "# Comparable answer run findings\n\n"
        "The frozen B2, B3, and B4 runs each contain eight structural abstentions. "
        "No package could be sent to a provider because every package contains only "
        "unconfirmed claims. B5 and B6 are unavailable because the repository has no "
        "frozen input release for them.\n\n"
        "All answer-quality metrics are null. There are no factual predictions, citations, "
        "or authorized answer reference data to score. This release checks reproducibility "
        "and structural safety. It does not measure answer correctness, compare B5 or B6, "
        "validate a generated answer, or report Phase 9 answerability performance.\n"
    )


def _per_case_from_mapping(value: object) -> PerAnswerQuality:
    mapping = _mapping(value)
    expected = {
        "query_id", "user_id", "baseline_id", "answer_id", "package_id",
        "package_sha256", "execution_id", "plan_id", "snapshot_run_id", "as_of",
        "requested_valid_time", "status", "structural_blockers", "statement_count",
        "citation_count", "metrics",
    }
    if set(mapping) != expected or not isinstance(mapping["metrics"], list):
        raise AnswerRunError("per-case fields changed")
    return PerAnswerQuality(
        mapping["query_id"], mapping["user_id"], mapping["baseline_id"],
        mapping["answer_id"], mapping["package_id"], mapping["package_sha256"],
        mapping["execution_id"], mapping["plan_id"], mapping["snapshot_run_id"],
        _datetime(mapping["as_of"]), _requested_valid_time(mapping["requested_valid_time"]),
        mapping["status"], tuple(mapping["structural_blockers"]),
        mapping["statement_count"], mapping["citation_count"],
        tuple(_metric_from_mapping(item) for item in mapping["metrics"]),
    )


def _scorecard_from_mapping(value: object) -> AnswerQualityScorecard:
    mapping = _mapping(value)
    if set(mapping) != {"scorer_version", "rows"} or not isinstance(mapping["rows"], list):
        raise AnswerRunError("scorecard fields changed")
    rows = []
    for item in mapping["rows"]:
        row = _mapping(item)
        if set(row) != {"baseline_id", "metric", "case_count", "abstained_count", "value"}:
            raise AnswerRunError("scorecard row fields changed")
        rows.append({**row, "value": _metric_from_mapping(row["value"])})
    return AnswerQualityScorecard(mapping["scorer_version"], tuple(rows))


def _metric_from_mapping(value: object) -> MetricValue:
    mapping = _mapping(value)
    if set(mapping) != {"metric", "numerator", "denominator", "value", "null_reason"}:
        raise AnswerRunError("metric fields changed")
    return MetricValue(**mapping)


def _failure_from_mapping(value: object) -> AnswerQualityFailure:
    mapping = _mapping(value)
    expected = {
        "failure_id", "answer_id", "package_id", "query_id", "baseline_id",
        "code", "location",
    }
    if set(mapping) != expected:
        raise AnswerRunError("failure fields changed")
    return AnswerQualityFailure(**mapping)


def _requested_valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    mapping = _mapping(value)
    expected = {
        "kind", "point_date", "point_timestamp", "range_start_date", "range_end_date",
        "range_start_timestamp", "range_end_timestamp",
    }
    if set(mapping) != expected:
        raise AnswerRunError("requested valid-time fields changed")
    return RequestedValidTime(
        mapping["kind"],
        _date(mapping["point_date"]),
        _optional_datetime(mapping["point_timestamp"]),
        _date(mapping["range_start_date"]),
        _date(mapping["range_end_date"]),
        _optional_datetime(mapping["range_start_timestamp"]),
        _optional_datetime(mapping["range_end_timestamp"]),
    )


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise AnswerRunError("datetime is invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnswerRunError("datetime is invalid") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise AnswerRunError("datetime is naive")
    return result


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnswerRunError("date is invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise AnswerRunError("date is invalid") from error


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    return _read_jsonl_bytes(path.read_bytes())


def _read_jsonl_bytes(raw: bytes) -> list[Mapping[str, object]]:
    try:
        values = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerRunError("JSONL is invalid") from error
    if any(not isinstance(value, dict) for value in values):
        raise AnswerRunError("JSONL record is invalid")
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswerRunError("JSON object is invalid") from error
    return _mapping(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnswerRunError("mapping is invalid")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise AnswerRunError("output directory must be empty")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
