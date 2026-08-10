"""Offline scorer for the frozen four-case interactive runtime checkpoint."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .interactive_contracts_v2 import (
    BEHAVIOURS,
    FINAL_RELEASE_VERSION,
    INTERACTIVE_VERSION,
    SCORER_VERSION,
    SYSTEM_VARIANT,
    InteractiveAnsweringError,
    InteractiveBehaviourReference,
    InteractiveChecks,
    InteractiveMetric,
    InteractivePrediction,
    canonical_json_bytes,
    prediction_from_mapping,
    reference_from_mapping,
)


GUIDANCE_SHA256 = "865223b8aea8665b8df8b10839f9d1f1cdb9c9c5a31d17f97e6f47ad27606a77"
CORRECTION_SHA256S = (
    "34a82610f0287d6f5321d7be6c1b75fb6fe475a984c6418cfb67d4b31b160880",
    "5480ebcbe58fbeeb17bfbdb3e30c0aa4a7e9ed41477976233c0efe19a72733a1",
    "4772c7f722bd1545a98c589d665248eaf7aee82f525670be0944f2820f2ca653",
)
STARTING_COMMIT = "d0932a7994153745285c1f4e3d75c36ffbbaf06a"
DATASET_MANIFEST = Path("data/abstention/interactive-answering-development-v2/manifest.json")
REFERENCE_PATH = Path("data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl")
REVIEW_PATH = Path("data/abstention/interactive-answering-development-v2/gold/review.json")
RUNTIME_ROOT = Path("results/abstention/interactive-answering-development-runtime-v2")
FINAL_ROOT = Path("results/abstention/interactive-answering-development-v2")
RUNTIME_CHECKPOINT_SHA256 = "423790b9eb4d0fa5e33aeeafd870cb75f8d78c259c307322d25e276615a199d4"
RUNTIME_RESPONSES_SHA256 = "babf748e7014d7d89b418986b3e3b9daf3d191f04e1a93d4cca8b877a276a977"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
DATASET_MANIFEST_SHA256 = "831529a53365efe0cf932f5ac5204696f0c0f05863b1a460b796533fc64774d6"
REFERENCE_SHA256 = "afa7a0d8d16140a469d79e4c69f57a7b357f7a50ea5247edb176ad69be60c975"
REVIEW_SHA256 = "b75e3f4a9f231a428cdeaae9422f947b021fd01d414d054af2cdab7f3ded97bd"
STEP91_MANIFEST_SHA256 = "15667096c215de19fcf2ac6d897237791b62eef30e37d505adac2e5de4523de3"
STEP92_MANIFEST_SHA256 = "ef957226d62beb44a8cb117e786a2c59f23f96815d8b5daa48b9d22d80c6ef53"
ARTIFACTS = (
    "predictions.jsonl", "per-case.jsonl", "scorecard.json", "checks.json",
    "failures.jsonl", "run.json", "findings.md",
)


def execute_interactive_evaluation(
    output_dir: str | Path = FINAL_ROOT,
    *,
    repo_root: str | Path = ".",
) -> InteractiveChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    dataset = _load_dataset(root)
    _verify_runtime_checkpoint(root)
    predictions = _load_predictions(root / RUNTIME_ROOT / "responses.jsonl")
    references = _load_references(root / REFERENCE_PATH)
    per_case, scorecard = score_interactive(predictions, references)
    checks = _checks(predictions)
    run = _run(checks)
    findings = _findings()
    payloads = {
        "predictions.jsonl": (root / RUNTIME_ROOT / "responses.jsonl").read_bytes(),
        "per-case.jsonl": _serialize(per_case),
        "scorecard.json": canonical_json_bytes(scorecard),
        "checks.json": canonical_json_bytes(checks),
        "failures.jsonl": (root / RUNTIME_ROOT / "failures.jsonl").read_bytes(),
        "run.json": canonical_json_bytes(run),
        "findings.md": findings.encode("utf-8"),
    }
    manifest = _manifest(root, dataset, checks, payloads)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    _write_exclusive(output / "manifest.json", canonical_json_bytes(manifest))
    verify_interactive_release(output, repo_root=root)
    return checks


def score_interactive(
    predictions: Sequence[InteractivePrediction],
    references: Sequence[InteractiveBehaviourReference],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if len(predictions) != 4 or len(references) != 4:
        raise InteractiveAnsweringError("interactive scorer accounting changed")
    by_case = {item.case_id: item for item in references}
    if len(by_case) != 4:
        raise InteractiveAnsweringError("interactive references are duplicated")
    per_case = []
    for prediction in predictions:
        reference = by_case.get(prediction.case_id)
        if reference is None or reference.user_id != prediction.user_id:
            raise InteractiveAnsweringError("prediction and reference identity differ")
        predicted = set(prediction.predicted_behaviours)
        expected = set(reference.expected_behaviours)
        predicted_evidence = set(prediction.predicted_evidence_tuples)
        expected_evidence = set(reference.exact_evidence_tuples)
        per_case.append({
            "case_id": prediction.case_id,
            "user_id": prediction.user_id,
            "capability": prediction.capability,
            "difficulty": prediction.difficulty,
            "split": prediction.split,
            "source_types": reference.source_types,
            "predicted_decision": prediction.decision,
            "expected_decision": reference.expected_decision,
            "decision_exact": prediction.decision == reference.expected_decision,
            "predicted_behaviours": prediction.predicted_behaviours,
            "expected_behaviours": reference.expected_behaviours,
            "behaviour_true_positive_count": len(predicted & expected),
            "predicted_behaviour_count": len(predicted),
            "expected_behaviour_count": len(expected),
            "predicted_evidence_count": len(predicted_evidence),
            "expected_evidence_count": len(expected_evidence),
            "evidence_true_positive_count": len(predicted_evidence & expected_evidence),
            "factual_statement_count": prediction.factual_statement_count,
            "citation_count": prediction.citation_count,
        })
    overall = _metrics_for_rows(tuple(per_case), "overall", "all")
    slices = []
    for dimension in ("capability", "difficulty", "split"):
        values = sorted({str(row[dimension]) for row in per_case})
        for value in values:
            members = tuple(row for row in per_case if row[dimension] == value)
            slices.extend(_slice_metrics(members, dimension, value))
    source_types = sorted({source for row in per_case for source in row["source_types"]})
    for source_type in source_types:
        members = tuple(row for row in per_case if source_type in row["source_types"])
        slices.extend(_slice_metrics(members, "source_type", source_type))
    scorecard = {
        "interactive_version": INTERACTIVE_VERSION,
        "scorer_version": SCORER_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "case_count": 4,
        "composite_score": None,
        "overall_metrics": overall,
        "slice_metrics": slices,
        "excluded_metrics": [
            "answer_correctness", "confidence_calibration", "abstention_precision_recall",
            "selective_risk", "false_answer_rate", "unnecessary_abstention_rate",
            "B6_B7_delta", "emotional_support_style", "semantic_similarity",
            "model_judge_score",
        ],
    }
    return tuple(per_case), scorecard


def verify_interactive_release(
    output_dir: str | Path = FINAL_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    dataset = _load_dataset(root)
    _verify_runtime_checkpoint(root)
    predictions = _load_predictions(output / "predictions.jsonl")
    references = _load_references(root / REFERENCE_PATH)
    per_case, scorecard = score_interactive(predictions, references)
    checks = _checks(predictions)
    expected_payloads = {
        "predictions.jsonl": (root / RUNTIME_ROOT / "responses.jsonl").read_bytes(),
        "per-case.jsonl": _serialize(per_case),
        "scorecard.json": canonical_json_bytes(scorecard),
        "checks.json": canonical_json_bytes(checks),
        "failures.jsonl": (root / RUNTIME_ROOT / "failures.jsonl").read_bytes(),
        "run.json": canonical_json_bytes(_run(checks)),
        "findings.md": _findings().encode("utf-8"),
    }
    for name, expected in expected_payloads.items():
        if (output / name).read_bytes() != expected:
            raise InteractiveAnsweringError("final artifact does not recompute")
    manifest = json.loads((output / "manifest.json").read_bytes())
    expected_manifest = _manifest(root, dataset, checks, expected_payloads)
    if manifest != expected_manifest or canonical_json_bytes(manifest) != (output / "manifest.json").read_bytes():
        raise InteractiveAnsweringError("final manifest does not recompute")


def _metrics_for_rows(
    rows: Sequence[Mapping[str, object]],
    slice_name: str,
    slice_value: str,
) -> list[dict[str, object]]:
    decision_hits = sum(bool(row["decision_exact"]) for row in rows)
    behaviour_tp = sum(int(row["behaviour_true_positive_count"]) for row in rows)
    predicted_behaviour = sum(int(row["predicted_behaviour_count"]) for row in rows)
    expected_behaviour = sum(int(row["expected_behaviour_count"]) for row in rows)
    evidence_tp = sum(int(row["evidence_true_positive_count"]) for row in rows)
    predicted_evidence = sum(int(row["predicted_evidence_count"]) for row in rows)
    expected_evidence = sum(int(row["expected_evidence_count"]) for row in rows)
    result = [
        _metric_row(slice_name, slice_value, "expected_decision_accuracy", decision_hits, len(rows), "no_reviewed_interactive_cases"),
        _metric_row(slice_name, slice_value, "expected_behaviour_micro_precision", behaviour_tp, predicted_behaviour, "no_predicted_behaviours"),
        _metric_row(slice_name, slice_value, "expected_behaviour_micro_recall", behaviour_tp, expected_behaviour, "no_reviewed_interactive_cases"),
        _metric_row(slice_name, slice_value, "expected_behaviour_micro_f1", 2 * behaviour_tp, predicted_behaviour + expected_behaviour, "no_reviewed_interactive_cases"),
    ]
    case_precisions = []
    case_recalls = []
    case_f1s = []
    for row in rows:
        tp = int(row["behaviour_true_positive_count"])
        predicted = int(row["predicted_behaviour_count"])
        expected = int(row["expected_behaviour_count"])
        case_precisions.append(tp / predicted if predicted else 0.0)
        case_recalls.append(tp / expected if expected else 0.0)
        case_f1s.append((2 * tp / (predicted + expected)) if predicted + expected else 0.0)
    for name, values in (
        ("expected_behaviour_macro_precision", case_precisions),
        ("expected_behaviour_macro_recall", case_recalls),
        ("expected_behaviour_macro_f1", case_f1s),
    ):
        result.append(_decimal_metric_row(slice_name, slice_value, name, values, "no_reviewed_interactive_cases"))
    for behaviour in BEHAVIOURS:
        denominator = sum(behaviour in row["expected_behaviours"] for row in rows)
        numerator = sum(
            behaviour in row["expected_behaviours"] and behaviour in row["predicted_behaviours"]
            for row in rows
        )
        result.append(_metric_row(
            slice_name, slice_value, f"{behaviour}_recall", numerator, denominator,
            f"no_reviewed_{behaviour}_cases",
        ))
    result.extend((
        _metric_row(slice_name, slice_value, "exact_evidence_tuple_precision", evidence_tp, predicted_evidence, "no_predicted_evidence"),
        _metric_row(slice_name, slice_value, "exact_evidence_tuple_recall", evidence_tp, expected_evidence, "no_reviewed_evidence"),
        _metric_row(slice_name, slice_value, "exact_evidence_tuple_f1", 2 * evidence_tp, predicted_evidence + expected_evidence, "no_reviewed_evidence"),
        _metric_row(slice_name, slice_value, "factual_provenance_coverage", 0, 0, "no_factual_outputs"),
    ))
    return result


def _slice_metrics(rows, name: str, value: str) -> list[dict[str, object]]:
    all_rows = _metrics_for_rows(rows, name, value)
    keep = {
        "expected_decision_accuracy", "expected_behaviour_micro_recall",
        "exact_evidence_tuple_recall", "factual_provenance_coverage",
    }
    return [row for row in all_rows if row["metric"] in keep]


def _metric_row(
    slice_name: str,
    slice_value: str,
    name: str,
    numerator: int,
    denominator: int,
    null_reason: str,
) -> dict[str, object]:
    metric = InteractiveMetric(
        numerator, denominator,
        None if denominator == 0 else f"{numerator / denominator:.6f}",
        null_reason if denominator == 0 else None,
    )
    return {"slice_name": slice_name, "slice_value": slice_value, "metric": name, **asdict(metric)}


def _decimal_metric_row(
    slice_name: str,
    slice_value: str,
    name: str,
    values: Sequence[float],
    null_reason: str,
) -> dict[str, object]:
    denominator = len(values)
    metric = InteractiveMetric(
        int(round(sum(values) * 1_000_000)),
        denominator * 1_000_000,
        None if not values else f"{sum(values) / denominator:.6f}",
        null_reason if not values else None,
    )
    return {"slice_name": slice_name, "slice_value": slice_value, "metric": name, **asdict(metric)}


def _checks(predictions: Sequence[InteractivePrediction]) -> InteractiveChecks:
    actions = [item.response_action for item in predictions]
    return InteractiveChecks(
        20, 4, 16, False, len(predictions), sum(item.turn_count for item in predictions),
        actions.count("abstained"), actions.count("clarification_requested"),
        actions.count("answered"), actions.count("disputed"),
        actions.count("partially_answered"),
        sum(item.factual_statement_count for item in predictions),
        sum(item.citation_count for item in predictions),
        sum(item.provider_eligible for item in predictions), 0, 0,
    )


def _run(checks: InteractiveChecks) -> dict[str, object]:
    return {
        "interactive_version": INTERACTIVE_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "system_variant": SYSTEM_VARIANT,
        "starting_commit": STARTING_COMMIT,
        "roadmap_target_case_count": 20,
        "development_executed_count": checks.development_case_count,
        "frozen_test_deferred_count": 16,
        "full_step_9_3_roadmap_complete": False,
        "configured_future_answer_model": "gpt-4.1-2025-04-14",
        "provider_returned_model": None,
        "provider_eligible_cases": checks.provider_eligible_case_count,
        "provider_requests": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "max_output_tokens": 0,
        "incremental_cost_usd": 0,
        "historical_openai_spend_usd": 0.2314404,
        "new_cumulative_ceiling_usd": 0.2314404,
        "runtime_cases_transmitted": "none",
        "fields_transmitted": "none",
        "B5_available": False,
        "B6_available": False,
        "B7_label": False,
        "invalid_v1_release": "invalid_uncommitted_backed_up_outside_repository",
        "prior_development_gold_exposure": True,
        "development_gold_used_for_v2_runtime": False,
        "reviewer_frozen_snippet_exposure_disclosed": True,
        "reviewer_frozen_snippet_used": False,
        "runtime_checkpoint_predated_v2_gold": True,
    }


def _findings() -> str:
    return (
        "# Development interactive answering findings\n\n"
        "This version repairs an invalid, uncommitted v1 run. Three aggregate requirement names were "
        "replaced with the registered predicates used by the ontology. The invalid files remain in a "
        "private backup outside the repository.\n\n"
        "The correction was not blind. Development gold had already been seen during v1 review, but it "
        "was not used to build the v2 runtime requirements or predictions. A reviewer separately exposed "
        "a frozen-test snippet after v1 froze; no snippet content, identifier, or rule entered v2. The "
        "corrected runtime checkpoint was frozen before any v2 gold or scorer file existed.\n\n"
        "All four development histories remain candidate-only. The system returned four abstentions, with "
        "no factual statements, citations, provider eligibility, or model calls. Sixteen frozen-test cases "
        "remain deferred, so this release does not complete the roadmap's 20-case Step 9.3 target.\n\n"
        "The scores describe four reviewed development cases. They do not establish answer quality or show "
        "that the abstentions are correct. B5 and B6 inputs are unavailable, this run is not B7, and Step "
        "9.4 has not started.\n"
    )


def _manifest(root: Path, dataset, checks, payloads) -> dict[str, object]:
    implementation = {
        str(path): _sha(root / path) for path in (
            Path("configs/abstention/interactive_answering_v2.json"),
            Path("src/abstention/interactive_contracts_v2.py"),
            Path("src/abstention/interactive_input_v2.py"),
            Path("src/abstention/interactive_runtime_v2.py"),
            Path("src/abstention/interactive_evaluation_v2.py"),
            Path("data/abstention/interactive-answering-development-v2/runtime/manifest.json"),
            Path("data/abstention/interactive-answering-development-v2/runtime/cases.jsonl"),
            Path("data/abstention/interactive-answering-development-v2/runtime/requirements.jsonl"),
            REFERENCE_PATH,
            REVIEW_PATH,
            Path("tests/unit/test_interactive_answering_v2.py"),
            Path("tests/integration/test_interactive_answering_v2.py"),
            Path("tests/integration/test_answer_quality_evaluation.py"),
            Path("tests/integration/test_answerability.py"),
            Path("tests/integration/test_memory_answer.py"),
        )
    }
    return {
        "interactive_version": INTERACTIVE_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "correction_envelope_sha256s": list(CORRECTION_SHA256S),
        "starting_commit": STARTING_COMMIT,
        "dataset_manifest_sha256": _sha(root / DATASET_MANIFEST),
        "runtime_checkpoint_sha256": RUNTIME_CHECKPOINT_SHA256,
        "runtime_responses_sha256": RUNTIME_RESPONSES_SHA256,
        "step_9_1_manifest_sha256": STEP91_MANIFEST_SHA256,
        "step_9_2_manifest_sha256": STEP92_MANIFEST_SHA256,
        "implementation": implementation,
        "outputs": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        "counts": asdict(checks),
        "runtime_preceded_reference": True,
        "invalid_v1_checkpoint_sha256": "0b7be7fe41bb9275115a32ef8998862707a7a0a945e56754bdba77556b00017e",
        "invalid_v1_backup_map_sha256": "ee40e76025fa0d7ae92ee565460c2c49dae1e0134aca4ab9da01319298659748",
        "prior_development_gold_exposure": True,
        "development_gold_used_for_v2_runtime": False,
        "reviewer_frozen_snippet_exposure_disclosed": True,
        "reviewer_frozen_snippet_used": False,
        "reference_records_requested": 4,
        "frozen_test_records_requested": 0,
        "full_step_9_3_roadmap_complete": False,
        "provider_requests": 0,
        "incremental_cost_usd": 0,
        "historical_openai_spend_usd": 0.2314404,
        "predecessor_drift": [
            {
                "path": "tests/integration/test_answer_quality_evaluation.py",
                "old_sha256": "0f66f17abe92b6a244a90cd7b999c48106ffffc0790c24c6fd1feb31c1fced95",
                "new_sha256": "fff42248ee2c3f361caea561341b4d0e28728be98166d11f6f8f9d138694ba2d",
                "reason": "step_9_3_v2_topology_and_nested_authority_adapter",
            },
            {
                "path": "tests/integration/test_answerability.py",
                "old_sha256": "0e502c029eca08d634780f9948215c312608bd0f15fb1dc51bd82d322827dbd3",
                "new_sha256": "73aeedc3f24f76e3f357e99793ef6ed85bc8c7610caf9b87fd298a7e2fb3c9f4",
                "reason": "step_9_3_v2_topology_and_nested_authority_adapter",
            },
            {
                "path": "tests/integration/test_memory_answer.py",
                "old_sha256": "ec3c88940ff9b3e8a8040225c7f444dc114d1281dab17105b4848bd709ac0835",
                "new_sha256": "1ca97a9ff1df5f11d5210b5434f5ded6830205df0b420c4d88570fdaa5f05f1d",
                "reason": "step_9_3_v2_topology_and_nested_authority_adapter",
            },
        ],
        "excluded_inputs": dataset["excluded_inputs"],
        "limitations": [
            "four_development_cases_only", "sixteen_frozen_test_cases_deferred",
            "candidate_only_histories", "not_blind", "not_answer_quality",
            "prior_development_gold_exposure_no_runtime_use",
            "reviewer_frozen_snippet_exposure_no_use",
            "invalid_v1_uncommitted_private_backup",
            "B5_B6_unavailable", "not_B7", "step_9_4_not_started",
        ],
    }


def _verify_runtime_checkpoint(root: Path) -> None:
    checkpoint_path = root / RUNTIME_ROOT / "checkpoint_manifest.json"
    if _sha(checkpoint_path) != RUNTIME_CHECKPOINT_SHA256:
        raise InteractiveAnsweringError("runtime checkpoint changed")
    checkpoint = json.loads(checkpoint_path.read_bytes())
    if (
        checkpoint.get("gold_opened") is not False
        or checkpoint.get("response_count") != 4
        or checkpoint.get("failure_count") != 0
        or checkpoint.get("provider_eligible_case_count") != 0
        or checkpoint.get("development_gold_used_for_v2_runtime") is not False
        or checkpoint.get("reviewer_frozen_snippet_used") is not False
    ):
        raise InteractiveAnsweringError("runtime checkpoint accounting changed")
    artifacts = checkpoint.get("artifacts")
    implementation = checkpoint.get("implementation")
    if not isinstance(artifacts, dict) or not isinstance(implementation, dict):
        raise InteractiveAnsweringError("runtime checkpoint map changed")
    for name, expected in artifacts.items():
        if _sha(root / RUNTIME_ROOT / name) != expected:
            raise InteractiveAnsweringError("runtime artifact changed")
    for name, expected in implementation.items():
        if _sha(root / name) != expected:
            raise InteractiveAnsweringError("runtime implementation changed")
    if _sha(root / RUNTIME_ROOT / "responses.jsonl") != RUNTIME_RESPONSES_SHA256:
        raise InteractiveAnsweringError("runtime responses changed")
    if _sha(root / RUNTIME_ROOT / "failures.jsonl") != EMPTY_SHA256:
        raise InteractiveAnsweringError("runtime failures changed")


def _load_dataset(root: Path) -> dict[str, object]:
    if _sha(root / DATASET_MANIFEST) != DATASET_MANIFEST_SHA256:
        raise InteractiveAnsweringError("dataset manifest changed")
    value = json.loads((root / DATASET_MANIFEST).read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("runtime_checkpoint_sha256") != RUNTIME_CHECKPOINT_SHA256
        or value.get("runtime_preceded_gold") is not True
        or value.get("development_case_count") != 4
        or value.get("frozen_test_deferred_count") != 16
        or value.get("full_step_9_3_roadmap_complete") is not False
    ):
        raise InteractiveAnsweringError("dataset manifest changed")
    if (
        value.get("gold_behaviours_sha256") != REFERENCE_SHA256
        or value.get("gold_review_sha256") != REVIEW_SHA256
        or _sha(root / REFERENCE_PATH) != REFERENCE_SHA256
        or _sha(root / REVIEW_PATH) != REVIEW_SHA256
    ):
        raise InteractiveAnsweringError("reviewed reference changed")
    return value


def _load_predictions(path: Path) -> tuple[InteractivePrediction, ...]:
    rows = tuple(prediction_from_mapping(item) for item in _read_jsonl(path))
    if len(rows) != 4 or len({item.prediction_id for item in rows}) != 4:
        raise InteractiveAnsweringError("prediction accounting changed")
    return rows


def _load_references(path: Path) -> tuple[InteractiveBehaviourReference, ...]:
    rows = tuple(reference_from_mapping(item) for item in _read_jsonl(path))
    if len(rows) != 4 or len({item.case_id for item in rows}) != 4:
        raise InteractiveAnsweringError("reference accounting changed")
    return rows


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise InteractiveAnsweringError("JSONL row lacks newline")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise InteractiveAnsweringError("JSONL row is not an object")
        rows.append(value)
    return tuple(rows)


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(value) for value in values)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise InteractiveAnsweringError("output directory is not empty")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
