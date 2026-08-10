"""Runtime freeze and offline scoring for answerability threshold calibration."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import apply_selected_thresholds, load_threshold_config, predict_all_fixtures
from .calibration_contracts import (
    FINAL_RELEASE_VERSION,
    PROFILE_ORDER,
    RUNTIME_RELEASE_VERSION,
    SELECTED_PROFILE,
    THRESHOLD_VERSION,
    AnswerabilityCalibrationChecks,
    AnswerabilityCalibrationScorecard,
    CalibrationError,
    CalibrationFixture,
    CalibrationMetric,
    CalibrationReference,
    FrozenThresholdSelection,
    ThresholdApplication,
    ThresholdPrediction,
    ThresholdSweepRow,
    application_from_mapping,
    canonical_json_bytes,
    fixture_from_mapping,
    metric_from_mapping,
    prediction_from_mapping,
    reference_from_mapping,
    stable_sha256,
)
from .contracts import AnswerabilityDecision, answerability_decision_from_mapping
from .evaluation import verify_answerability_release


GUIDANCE_VERSION = "step-9.2-guidance-v1"
GUIDANCE_SHA256 = "c815ed3785101c08ce563c42c802c4513ab1bb00f5213e21e77098369d751628"
START_COMMIT = "35d0c64431f19d4243b72af712dadeb8d522128f"
CONFIG_PATH = Path("configs/abstention/answerability_thresholds_v1.json")
RUNTIME_DATASET_MANIFEST = Path("data/abstention/answerability-threshold-development-v1/runtime/manifest.json")
FIXTURES_PATH = Path("data/abstention/answerability-threshold-development-v1/runtime/fixtures.jsonl")
REFERENCE_PATH = Path("data/abstention/answerability-threshold-development-v1/reference/expected.jsonl")
DATASET_MANIFEST = Path("data/abstention/answerability-threshold-development-v1/manifest.json")
RUNTIME_ROOT = Path("results/abstention/answerability-threshold-development-runtime-v1")
FINAL_ROOT = Path("results/abstention/answerability-threshold-development-v1")
STEP91_RESULT = Path("results/abstention/answerability-development-v1")
IMPLEMENTATION_PATHS = (
    Path("src/abstention/calibration_contracts.py"),
    Path("src/abstention/calibration.py"),
    Path("src/abstention/calibration_evaluation.py"),
)
RUNTIME_ARTIFACTS = ("preflight.json", "predictions.jsonl", "failures.jsonl")
FINAL_ARTIFACTS = (
    "development-decisions.jsonl", "fixture-results.jsonl", "threshold-sweep.jsonl",
    "thresholds.json", "scorecard.json", "checks.json", "failures.jsonl",
    "run.json", "findings.md",
)
STEP91_HASHES = {
    "config": "b4e5dbd3fcb41c7a2b61cd2c208b7cca8cb2a0696d1d907215174eabeae4da9c",
    "dataset_manifest": "bbcf1a703b2cbb3d6facf700882f8e354e0db906f5dbf6fee4cfb53ceb9ae9bf",
    "decisions": "06a57523ed2262dbccdc412e11d214b5d7f00e4ead81b974fee87e8c6323dbe0",
    "checks": "1160a8861f2b86496e72aac25fac99a24fbd35c7ad52d4c5ab0236d44734911a",
    "run": "651648133df3c97435808525226dd2c5b357126aa189fca16956700578f50015",
    "findings": "d67bbc73f85bb54a18a1c05066277dd7f47605091ed62bc9f57981fbb7ab2ebc",
    "failures": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "result_manifest": "15667096c215de19fcf2ac6d897237791b62eef30e37d505adac2e5de4523de3",
}
STEP91_PATHS = {
    "config": Path("configs/abstention/answerability_v1.json"),
    "dataset_manifest": Path("data/abstention/answerability-development-v1/manifest.json"),
    "decisions": STEP91_RESULT / "decisions.jsonl",
    "checks": STEP91_RESULT / "checks.json",
    "run": STEP91_RESULT / "run.json",
    "findings": STEP91_RESULT / "findings.md",
    "failures": STEP91_RESULT / "failures.jsonl",
    "result_manifest": STEP91_RESULT / "manifest.json",
}
EXPECTED_SWEEP = {
    "ordinary_1_trait_2": (8, 8, 8, "0.333333", "1.000000"),
    "ordinary_1_trait_3": (7, 9, 8, "0.291667", "0.875000"),
    "ordinary_2_trait_2": (6, 10, 8, "0.250000", "0.750000"),
    "ordinary_2_trait_3": (5, 11, 8, "0.208333", "0.625000"),
    "ordinary_3_trait_2": (4, 12, 8, "0.166667", "0.500000"),
    "ordinary_3_trait_3": (3, 13, 8, "0.125000", "0.375000"),
}


class CalibrationEvaluationError(RuntimeError):
    """Fail a calibration release that does not recompute exactly."""


def freeze_threshold_runtime(
    output_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    if any((root / path).exists() for path in (REFERENCE_PATH, DATASET_MANIFEST, FINAL_ROOT)):
        raise CalibrationEvaluationError("reference or final scorer path existed before runtime freeze")
    config = load_threshold_config(repo_root=root)
    runtime_dataset = _load_runtime_dataset(root)
    _verify_step91(root)
    fixtures = _load_fixtures(root)
    predictions = predict_all_fixtures(fixtures, config)
    preflight = {
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": START_COMMIT,
        "fixture_count": 24,
        "profile_count": 6,
        "prediction_count": 144,
        "failure_count": 0,
        "reference_present": False,
        "reference_opened": False,
        "final_result_present": False,
        "benchmark_gold_opened": False,
        "answer_gold_opened": False,
        "relevance_gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "test_user_opened": False,
        "provider_environment_opened": False,
        "prompt_opened": False,
        "database_opened": False,
        "provider_request_count": 0,
        "retry_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": 0,
    }
    payloads = {
        "preflight.json": canonical_json_bytes(preflight),
        "predictions.jsonl": b"".join(canonical_json_bytes(item) for item in predictions),
        "failures.jsonl": b"",
    }
    checkpoint = _runtime_checkpoint(root, runtime_dataset, payloads)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    _write_exclusive(output / "checkpoint_manifest.json", canonical_json_bytes(checkpoint))
    verify_threshold_runtime_checkpoint(output, repo_root=root)
    return checkpoint


def verify_threshold_runtime_checkpoint(
    runtime_dir: str | Path = RUNTIME_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    runtime = _resolve(root, runtime_dir)
    if not runtime.is_dir() or {path.name for path in runtime.iterdir()} != {*RUNTIME_ARTIFACTS, "checkpoint_manifest.json"}:
        raise CalibrationEvaluationError("runtime artifact set changed")
    config = load_threshold_config(repo_root=root)
    runtime_dataset = _load_runtime_dataset(root)
    _verify_step91(root)
    fixtures = _load_fixtures(root)
    predictions = tuple(prediction_from_mapping(item) for item in _read_jsonl(runtime / "predictions.jsonl"))
    expected = predict_all_fixtures(fixtures, config)
    if predictions != expected:
        raise CalibrationEvaluationError("runtime predictions do not recompute")
    preflight = _read_object(runtime / "preflight.json")
    if preflight != {
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": START_COMMIT,
        "fixture_count": 24,
        "profile_count": 6,
        "prediction_count": 144,
        "failure_count": 0,
        "reference_present": False,
        "reference_opened": False,
        "final_result_present": False,
        "benchmark_gold_opened": False,
        "answer_gold_opened": False,
        "relevance_gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "test_user_opened": False,
        "provider_environment_opened": False,
        "prompt_opened": False,
        "database_opened": False,
        "provider_request_count": 0,
        "retry_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": 0,
    }:
        raise CalibrationEvaluationError("runtime preflight changed")
    if (runtime / "failures.jsonl").read_bytes() != b"":
        raise CalibrationEvaluationError("runtime failures changed")
    payloads = {name: (runtime / name).read_bytes() for name in RUNTIME_ARTIFACTS}
    checkpoint = _runtime_checkpoint(root, runtime_dataset, payloads)
    if _read_object(runtime / "checkpoint_manifest.json") != checkpoint:
        raise CalibrationEvaluationError("runtime checkpoint does not recompute")


def score_threshold_profiles(
    output_dir: str | Path = FINAL_ROOT,
    *,
    runtime_dir: str | Path = RUNTIME_ROOT,
    repo_root: str | Path = ".",
) -> AnswerabilityCalibrationChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    runtime = _resolve(root, runtime_dir)
    _require_empty(output)
    verify_threshold_runtime_checkpoint(runtime, repo_root=root)
    config = load_threshold_config(repo_root=root)
    references = _load_references(root)
    predictions = tuple(prediction_from_mapping(item) for item in _read_jsonl(runtime / "predictions.jsonl"))
    fixture_results, sweep = _score_fixtures(predictions, references)
    decisions = _load_step91_decisions(root)
    applications = apply_selected_thresholds(decisions, config)
    selection = FrozenThresholdSelection(
        THRESHOLD_VERSION,
        SELECTED_PROFILE,
        "preselected_by_clean_room_contract_before_reference",
        config.profiles[0],
        config.confidence_policy,
    )
    scorecard = _scorecard(decisions, applications, predictions, references)
    checks = _checks(applications, fixture_results, sweep)
    payloads = _final_payloads(applications, fixture_results, sweep, selection, scorecard, checks)
    dataset = _load_final_dataset(root)
    manifest = _final_manifest(root, runtime, dataset, checks, payloads)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    _write_exclusive(output / "manifest.json", canonical_json_bytes(manifest))
    verify_answerability_threshold_release(output, runtime_dir=runtime, repo_root=root)
    return checks


def verify_answerability_threshold_release(
    output_dir: str | Path = FINAL_ROOT,
    *,
    runtime_dir: str | Path = RUNTIME_ROOT,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    runtime = _resolve(root, runtime_dir)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {*FINAL_ARTIFACTS, "manifest.json"}:
        raise CalibrationEvaluationError("final artifact set changed")
    verify_threshold_runtime_checkpoint(runtime, repo_root=root)
    config = load_threshold_config(repo_root=root)
    references = _load_references(root)
    predictions = tuple(prediction_from_mapping(item) for item in _read_jsonl(runtime / "predictions.jsonl"))
    fixture_results, sweep = _score_fixtures(predictions, references)
    decisions = _load_step91_decisions(root)
    applications = apply_selected_thresholds(decisions, config)
    checked_applications = tuple(application_from_mapping(item) for item in _read_jsonl(output / "development-decisions.jsonl"))
    if checked_applications != applications:
        raise CalibrationEvaluationError("development applications do not recompute")
    selection = FrozenThresholdSelection(
        THRESHOLD_VERSION, SELECTED_PROFILE,
        "preselected_by_clean_room_contract_before_reference",
        config.profiles[0], config.confidence_policy,
    )
    scorecard = _scorecard(decisions, applications, predictions, references)
    checks = _checks(applications, fixture_results, sweep)
    payloads = _final_payloads(applications, fixture_results, sweep, selection, scorecard, checks)
    for name, payload in payloads.items():
        if (output / name).read_bytes() != payload:
            raise CalibrationEvaluationError(f"{name} does not recompute")
    dataset = _load_final_dataset(root)
    if _read_object(output / "manifest.json") != _final_manifest(root, runtime, dataset, checks, payloads):
        raise CalibrationEvaluationError("final manifest does not recompute")


def _verify_step91(root: Path) -> None:
    verify_answerability_release(repo_root=root)
    actual = {name: _sha(root / path) for name, path in STEP91_PATHS.items()}
    if actual != STEP91_HASHES:
        raise CalibrationEvaluationError("Step 9.1 authority changed")


def _load_step91_decisions(root: Path) -> tuple[AnswerabilityDecision, ...]:
    _verify_step91(root)
    decisions = tuple(answerability_decision_from_mapping(item) for item in _read_jsonl(root / STEP91_PATHS["decisions"]))
    if (
        len(decisions) != 24
        or sum(item.decision == "abstain" for item in decisions) != 24
        or sum(len(item.rejected_evidence) for item in decisions) != 491
        or any(item.accepted_evidence or item.primary_reason != "no_promoted_claims" for item in decisions)
    ):
        raise CalibrationEvaluationError("Step 9.1 decision handoff changed")
    return decisions


def _load_runtime_dataset(root: Path) -> Mapping[str, object]:
    value = _read_object(root / RUNTIME_DATASET_MANIFEST)
    if (
        value.get("dataset_version") != "answerability-threshold-development-runtime-v1"
        or value.get("guidance_sha256") != GUIDANCE_SHA256
        or value.get("fixture_count") != 24
        or value.get("profile_count") != 6
        or value.get("reference_included") is not False
        or value.get("config_sha256") != _sha(root / CONFIG_PATH)
        or value.get("fixtures_sha256") != _sha(root / FIXTURES_PATH)
        or value.get("implementation") != {path.as_posix(): _sha(root / path) for path in IMPLEMENTATION_PATHS}
    ):
        raise CalibrationEvaluationError("runtime dataset manifest changed")
    return value


def _load_final_dataset(root: Path) -> Mapping[str, object]:
    value = _read_object(root / DATASET_MANIFEST)
    if (
        value.get("dataset_version") != "answerability-threshold-development-v1"
        or value.get("guidance_sha256") != GUIDANCE_SHA256
        or value.get("runtime_manifest_sha256") != _sha(root / RUNTIME_DATASET_MANIFEST)
        or value.get("fixtures_sha256") != _sha(root / FIXTURES_PATH)
        or value.get("reference_sha256") != _sha(root / REFERENCE_PATH)
        or value.get("config_sha256") != _sha(root / CONFIG_PATH)
        or value.get("step_9_1_result_manifest_sha256") != STEP91_HASHES["result_manifest"]
        or value.get("reference_opened_after_checkpoint") is not True
    ):
        raise CalibrationEvaluationError("final dataset manifest changed")
    return value


def _load_fixtures(root: Path) -> tuple[CalibrationFixture, ...]:
    fixtures = tuple(fixture_from_mapping(item) for item in _read_jsonl(root / FIXTURES_PATH))
    if len(fixtures) != 24 or fixtures != tuple(sorted(fixtures, key=lambda item: item.fixture_id)):
        raise CalibrationEvaluationError("fixture coverage or order changed")
    reasons = {reason for item in fixtures for reason in item.input_reasons}
    if reasons != {
        "no_retrieved_claims", "incomplete_evidence", "no_promoted_claims",
        "clarification_required", "wrong_person_risk", "requested_information_absent",
        "requested_time_not_covered", "stale_evidence", "insufficient_speaker_authority",
        "insufficient_source_authority", "unresolved_conflict",
        "stable_trait_support_insufficient", "causal_support_insufficient",
    }:
        raise CalibrationEvaluationError("fixture reason coverage changed")
    return fixtures


def _load_references(root: Path) -> tuple[CalibrationReference, ...]:
    references = tuple(reference_from_mapping(item) for item in _read_jsonl(root / REFERENCE_PATH))
    if (
        len(references) != 24
        or references != tuple(sorted(references, key=lambda item: item.fixture_id))
        or [item.expected_decision for item in references].count("answerable") != 8
        or [item.expected_decision for item in references].count("abstain") != 8
        or [item.expected_decision for item in references].count("clarify") != 8
    ):
        raise CalibrationEvaluationError("controlled reference coverage changed")
    if {item.fixture_id for item in references} != {item.fixture_id for item in _load_fixtures(root)}:
        raise CalibrationEvaluationError("reference fixture IDs changed")
    return references


def _score_fixtures(
    predictions: Sequence[ThresholdPrediction],
    references: Sequence[CalibrationReference],
) -> tuple[tuple[Mapping[str, object], ...], tuple[ThresholdSweepRow, ...]]:
    reference_by_id = {item.fixture_id: item for item in references}
    selected = [item for item in predictions if item.profile_id == SELECTED_PROFILE]
    fixture_results = tuple(sorted(({
        "fixture_id": item.fixture_id,
        "prediction_id": item.prediction_id,
        "profile_id": item.profile_id,
        "expected_decision": reference_by_id[item.fixture_id].expected_decision,
        "actual_decision": item.output_decision,
        "expected_reasons": list(reference_by_id[item.fixture_id].expected_reasons),
        "actual_reasons": list(item.output_reasons),
        "matched": item.output_decision == reference_by_id[item.fixture_id].expected_decision
        and item.output_reasons == reference_by_id[item.fixture_id].expected_reasons,
        "scope": "controlled_fixture_only",
    } for item in selected), key=lambda item: item["fixture_id"]))
    if len(fixture_results) != 24 or sum(item["matched"] for item in fixture_results) != 24:
        raise CalibrationEvaluationError("selected fixture predictions changed")
    rows: list[ThresholdSweepRow] = []
    for profile_id in PROFILE_ORDER:
        values = [item for item in predictions if item.profile_id == profile_id]
        expected_answerable, expected_abstain, expected_clarify, total, answerable_coverage = EXPECTED_SWEEP[profile_id]
        row = ThresholdSweepRow(
            profile_id, 24,
            sum(item.output_decision == "answerable" for item in values),
            sum(item.output_decision == "abstain" for item in values),
            sum(item.output_decision == "clarify" for item in values),
            _ratio(sum(item.output_decision == "answerable" for item in values), 24),
            _ratio(sum(item.output_decision == "answerable" for item in values), 8),
            "0.000000",
        )
        if asdict(row) != {
            "profile_id": profile_id, "fixture_count": 24,
            "answerable_count": expected_answerable, "abstain_count": expected_abstain,
            "clarify_count": expected_clarify, "total_coverage": total,
            "answerable_case_coverage": answerable_coverage, "selective_risk": "0.000000",
        }:
            raise CalibrationEvaluationError("threshold sweep changed")
        rows.append(row)
    return fixture_results, tuple(rows)


def _checks(applications, fixture_results, sweep) -> AnswerabilityCalibrationChecks:
    checks = AnswerabilityCalibrationChecks(
        len(applications), sum(item.output_decision == "abstain" for item in applications),
        sum(item.output_decision == "answerable" for item in applications),
        sum(item.output_decision == "clarify" for item in applications),
        sum(item.generation_allowed for item in applications),
        sum(item.confidence.get("value") is None for item in applications),
        sum(item.rejected_evidence_count for item in applications),
        sum(item.accepted_evidence_count for item in applications),
        len(fixture_results), len(sweep), SELECTED_PROFILE, 0, 0,
    )
    if asdict(checks) != {
        "development_decision_count": 24, "development_abstain_count": 24,
        "development_answerable_count": 0, "development_clarify_count": 0,
        "development_generation_allowed_count": 0, "development_null_confidence_count": 24,
        "input_rejection_count": 491, "accepted_evidence_count": 0,
        "fixture_result_count": 24, "threshold_sweep_count": 6,
        "selected_profile": SELECTED_PROFILE, "failure_count": 0,
        "provider_request_count": 0,
    }:
        raise CalibrationEvaluationError("final calibration checks changed")
    return checks


def _scorecard(decisions, applications, predictions, references) -> AnswerabilityCalibrationScorecard:
    development = [
        _metric("real_development", None, "overall", "all", "coverage", 0, 24),
        _metric("real_development", None, "overall", "all", "answer_accuracy_on_answered_cases", 0, 0, "no_answered_cases_and_no_authorized_answerability_gold"),
        _metric("real_development", None, "overall", "all", "selective_risk", 0, 0, "no_answered_cases"),
        _metric("real_development", None, "overall", "all", "abstention_precision", 0, 0, "no_authorized_answerability_gold"),
        _metric("real_development", None, "overall", "all", "abstention_recall", 0, 0, "no_authorized_answerability_gold"),
        _metric("real_development", None, "overall", "all", "false_answer_rate_on_unanswerable_cases", 0, 0, "no_authorized_answerability_gold"),
        _metric("real_development", None, "overall", "all", "unnecessary_abstention_rate_on_answerable_cases", 0, 0, "no_authorized_answerability_gold"),
        _metric("real_development", None, "overall", "all", "confidence_calibration_error", 0, 0, "no_authorized_answerability_gold"),
    ]
    for baseline_id in ("B2", "B3", "B4"):
        denominator = sum(item.baseline_id == baseline_id for item in applications)
        development.append(_metric("real_development", None, "baseline", baseline_id, "coverage", 0, denominator))
    for query_type in sorted({item.query_type for item in decisions}):
        denominator = sum(item.query_type == query_type for item in decisions)
        development.append(_metric("real_development", None, "query_type", query_type, "coverage", 0, denominator))
    selected = [item for item in predictions if item.profile_id == SELECTED_PROFILE]
    reference_by_id = {item.fixture_id: item for item in references}
    answerable = sum(item.output_decision == "answerable" for item in selected)
    correct = sum(item.output_decision == reference_by_id[item.fixture_id].expected_decision for item in selected)
    controlled = [
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "decision_accuracy", correct, 24),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "total_coverage", answerable, 24),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "answerable_case_coverage", answerable, 8),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "selective_risk", 0, answerable),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "false_answer_rate", 0, 16),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "unnecessary_abstention_rate", 0, 8),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "clarification_recall", 8, 8),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "abstention_precision", 8, 8),
        _metric("controlled_fixture_only", SELECTED_PROFILE, "overall", "all", "abstention_recall", 8, 8),
    ]
    return AnswerabilityCalibrationScorecard(
        tuple(sorted(development, key=lambda item: item.metric_id)),
        tuple(sorted(controlled, key=lambda item: item.metric_id)),
        None,
    )


def _metric(scope, profile_id, slice_name, slice_value, name, numerator, denominator, null_reason=None):
    fields = {
        "scope": scope, "profile_id": profile_id, "slice_name": slice_name,
        "slice_value": slice_value, "metric": name, "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else _ratio(numerator, denominator),
        "null_reason": null_reason,
    }
    return CalibrationMetric(metric_id=stable_sha256(fields), **fields)


def _runtime_checkpoint(root, runtime_dataset, payloads) -> Mapping[str, object]:
    return {
        "runtime_release_version": RUNTIME_RELEASE_VERSION,
        "guidance": {"version": GUIDANCE_VERSION, "sha256": GUIDANCE_SHA256, "starting_commit": START_COMMIT},
        "configuration": {"path": CONFIG_PATH.as_posix(), "sha256": _sha(root / CONFIG_PATH)},
        "runtime_dataset": {"path": RUNTIME_DATASET_MANIFEST.as_posix(), "sha256": _sha(root / RUNTIME_DATASET_MANIFEST)},
        "fixtures": {"path": FIXTURES_PATH.as_posix(), "sha256": _sha(root / FIXTURES_PATH)},
        "implementation": {path.as_posix(): _sha(root / path) for path in IMPLEMENTATION_PATHS},
        "step_9_1": STEP91_HASHES,
        "counts": {"fixture_count": 24, "profile_count": 6, "prediction_count": 144, "failure_count": 0},
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "reference_opened": False,
        "benchmark_gold_opened": False,
        "model_usage": {"provider_requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0, "incremental_cost_usd": 0},
    }


def _final_payloads(applications, fixture_results, sweep, selection, scorecard, checks):
    return {
        "development-decisions.jsonl": b"".join(canonical_json_bytes(item) for item in applications),
        "fixture-results.jsonl": b"".join(canonical_json_bytes(item) for item in fixture_results),
        "threshold-sweep.jsonl": b"".join(canonical_json_bytes(item) for item in sweep),
        "thresholds.json": canonical_json_bytes(selection),
        "scorecard.json": canonical_json_bytes(scorecard),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "failures.jsonl": b"",
        "run.json": canonical_json_bytes({
            "guidance_version": GUIDANCE_VERSION,
            "runtime_version": "answerability_calibration_runtime_v1",
            "input_release_version": "answerability-development-v1",
            "output_release_version": FINAL_RELEASE_VERSION,
            "selected_profile": SELECTED_PROFILE,
            "development_decision_count": 24,
            "fixture_prediction_count": 144,
            "threshold_sweep_count": 6,
            "model": "deterministic_no_model",
            "provider_requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0,
            "incremental_cost_usd": 0, "historical_openai_spend_usd": "0.2314404",
        }),
        "findings.md": _findings().encode("utf-8"),
    }


def _final_manifest(root, runtime, dataset, checks, payloads):
    return {
        "release_version": FINAL_RELEASE_VERSION,
        "guidance": {
            "version": GUIDANCE_VERSION, "sha256": GUIDANCE_SHA256,
            "starting_commit": START_COMMIT,
            "contamination_boundary": "clean_room_no_rejected_prompt_content",
        },
        "configuration": {"path": CONFIG_PATH.as_posix(), "sha256": _sha(root / CONFIG_PATH)},
        "dataset": {"path": DATASET_MANIFEST.as_posix(), "sha256": _sha(root / DATASET_MANIFEST)},
        "runtime_checkpoint": {"path": (RUNTIME_ROOT / "checkpoint_manifest.json").as_posix(), "sha256": _sha(runtime / "checkpoint_manifest.json")},
        "implementation": {path.as_posix(): _sha(root / path) for path in IMPLEMENTATION_PATHS},
        "step_9_1": STEP91_HASHES,
        "prerequisite_manifests": {
            "step_8_2": "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841",
            "step_8_3": "ac936819856939f66597c279fc0b852022a650d5455a4c21240ffbb01f0a524f",
        },
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "counts": asdict(checks),
        "profile_order": list(PROFILE_ORDER),
        "selected_profile": SELECTED_PROFILE,
        "confidence_policy": {
            "value": None, "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        },
        "excluded_inputs": dataset["excluded_inputs"],
        "predecessor_drift": [],
        "model_usage": {
            "model": "deterministic_no_model", "provider_requests": 0, "retries": 0,
            "input_tokens": 0, "output_tokens": 0, "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
        "limitations": [
            "Real development coverage remains zero because every claim is candidate-only.",
            "No real answerability correctness or confidence calibration was measured.",
            "Controlled fixtures test deterministic mechanics and are not a product score.",
            "Step 9.3 has not started.",
        ],
    }


def _findings() -> str:
    return (
        "# Answerability threshold findings\n\n"
        "The real development run still answers none of its 24 cases. Every available claim is a candidate, so the selected threshold leaves all decisions at `abstain`.\n\n"
        "There is no authorized answerability gold for these cases. This run therefore does not measure real answer correctness, abstention precision or recall, selective risk, or confidence calibration. Confidence remains null.\n\n"
        "The invented fixture grid checks the threshold mechanics. It shows how stricter profiles discard useful controlled cases, but it is not a benchmark or product score. The selected profile keeps the ontology's identity, time, authority, conflict, trait, causal, and provenance rules intact without optimizing for maximum abstention.\n\n"
        "Step 9.3 has not started. No interactive case, prompt, database, provider, or model path was opened.\n"
    )


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise CalibrationEvaluationError("ratio denominator is zero")
    return f"{numerator / denominator:.6f}"


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CalibrationEvaluationError("JSONL row is not an object")
                values.append(value)
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationEvaluationError("JSON artifact is not an object")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise CalibrationEvaluationError("output directory must be absent or empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
