"""Score the frozen four-case B6/B7 prerequisite without rerunning runtime work."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

from .b7_evaluation_contracts import (
    BASELINES,
    BASELINE_METRICS,
    CONFIG_VERSION,
    DATASET_VERSION,
    DELTA_METRICS,
    EVALUATION_VERSION,
    FINAL_RELEASE_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    B7EvaluationChecks,
    B7EvaluationError,
    B7EvaluationReference,
    B7PairDelta,
    B7PerCase,
    B7Scorecard,
    MetricValue,
    canonical_json_bytes,
    checks_from_mapping,
    pair_delta_from_mapping,
    per_case_from_mapping,
    reference_from_mapping,
    scorecard_from_mapping,
    stable_sha256,
)
from .comparison_contracts import (
    ComparablePair,
    ComparablePrediction,
    canonical_json_bytes as prerequisite_json_bytes,
    checkpoint_from_mapping,
    pair_from_mapping,
    prediction_from_mapping,
)


STARTING_COMMIT = "7e8fc5337384ac329264e3606507b925bd890d63"
GUIDANCE_SHA256 = "e8c5a5b1381f6314edc13d952c5f77103cef020503de90b5f7904a929573595f"
BOUNDARY_RULING_VERSION = "step-9.4-guidance-v2-boundary-ruling-1"
BOUNDARY_RULING_PREIMAGE = (
    "step-9.4-guidance-v2-boundary-ruling-1|"
    "e8c5a5b1381f6314edc13d952c5f77103cef020503de90b5f7904a929573595f|"
    "7e8fc5337384ac329264e3606507b925bd890d63|"
    "preserve_no_frozen_oracle_reference_boundary|"
    "manifest_only_scaled_validation|safe_unittest_discovery|"
    "unrestricted_full_discovery_not_run|phase10_not_started"
)
BOUNDARY_RULING_SHA256 = "d22d055e9c32cdd0f178f1eadaa77f23574eab50e02e7a6af69f879cbf97d476"
SCALED_MANIFEST_SHA256 = "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3"
SCALED_DATASET_SHA256 = "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61"
CONFIG_PATH = Path("configs/abstention/b7_evaluation_v1.json")
DATASET_ROOT = Path("data/abstention/b7-evaluation-development-v1")
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
REFERENCE_PATH = DATASET_ROOT / "reference/expected.jsonl"
REFERENCE_REVIEW = DATASET_ROOT / "reference/review.json"
RESULT_ROOT = Path("results/abstention/b7-evaluation-development-v1")
PREREQUISITE_ROOT = Path("results/abstention/b6-b7-comparable-development-runtime-v1")
SOURCE_REFERENCE = Path("data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl")
PREREQUISITE_CHECKPOINT_SHA256 = "9902507e5518b2b16fee35ad76be62ef57201bb9271b3de251e2b2063c239186"
PREREQUISITE_PAIRS_SHA256 = "a7b3e018e9fb6c45392d9a678ba2c166990e456c05ac2e43d17b1cc9f640df24"
PREREQUISITE_B6_SHA256 = "49992c38495346a8a88b1ed3e12995c12a965b2ccc525eb6956ee2cb6dd72d9c"
PREREQUISITE_B7_SHA256 = "ad1fc7bfc9eecfdc7792b281afd46b0c729596060e5a2dbdae7a49a515b4fefe"
PREREQUISITE_IMPLEMENTATION = {
    "configs/abstention/b6_b7_comparable_runtime_v1.json": "78934d7ae4837250d2f29452bdb9452287c4a38b01cc483a644fb5a418bac33d",
    "data/abstention/b6-b7-comparable-development-v1/runtime/manifest.json": "81ad18fc14e444b7518ae25d24015fd798f46e9a7962c17edf0bb27e46601a36",
    "src/abstention/comparison_contracts.py": "14db47896f400411e310bd496c05cbe0f2599f3114b154b1e374e0cc2ad6ad3f",
    "src/abstention/comparison_runtime.py": "f6d5320253a3bc2681ddb4fe9ab6e9950ab27f9541a1a333322f7250529f655a",
}
SOURCE_REFERENCE_SHA256 = "afa7a0d8d16140a469d79e4c69f57a7b357f7a50ea5247edb176ad69be60c975"
MEMORY_CONFIG_SHA256 = "98b743d593e17a88216ac74cf17693f08898537027e0d91672d45aaeb4dfcc10"
PROMPT_SHA256 = "69dd688430c55fc35a16369201ef8eea7d470d10f610edec254f5cbd59bf14c1"
ARTIFACTS = (
    "per-case.jsonl",
    "pair-deltas.jsonl",
    "scorecard.json",
    "checks.json",
    "failures.jsonl",
    "run.json",
    "findings.md",
)
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/abstention/b7_evaluation_contracts.py"),
    Path("src/abstention/b7_evaluation.py"),
    DATASET_MANIFEST,
)
ADAPTER_PATHS = (
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
)
NEW_PATHS = (
    "configs/abstention/b7_evaluation_v1.json",
    "data/abstention/b7-evaluation-development-v1/manifest.json",
    "data/abstention/b7-evaluation-development-v1/reference/expected.jsonl",
    "data/abstention/b7-evaluation-development-v1/reference/review.json",
    "results/abstention/b7-evaluation-development-v1/checks.json",
    "results/abstention/b7-evaluation-development-v1/failures.jsonl",
    "results/abstention/b7-evaluation-development-v1/findings.md",
    "results/abstention/b7-evaluation-development-v1/manifest.json",
    "results/abstention/b7-evaluation-development-v1/pair-deltas.jsonl",
    "results/abstention/b7-evaluation-development-v1/per-case.jsonl",
    "results/abstention/b7-evaluation-development-v1/run.json",
    "results/abstention/b7-evaluation-development-v1/scorecard.json",
    "src/abstention/b7_evaluation.py",
    "src/abstention/b7_evaluation_contracts.py",
    "tests/integration/test_b7_evaluation.py",
    "tests/unit/test_b7_evaluation.py",
)
FINAL_AUTHORIZED_PATHS = tuple(sorted((*NEW_PATHS, *ADAPTER_PATHS, "docs/implementation-progress.md")))
SAFE_DISCOVERY_INCLUDED_MODULES = (
    "tests.integration.test_answer_quality_evaluation",
    "tests.integration.test_answerability",
    "tests.integration.test_answerability_calibration",
    "tests.integration.test_b6_b7_comparison_prerequisite",
    "tests.integration.test_belief_resolution",
    "tests.integration.test_belief_resolution_evaluation",
    "tests.integration.test_conflict_candidates",
    "tests.integration.test_conflict_relation_evaluation",
    "tests.integration.test_conflict_relations",
    "tests.integration.test_durative_claim_evaluation",
    "tests.integration.test_durative_claim_persistence",
    "tests.integration.test_evidence_package",
    "tests.integration.test_grounded_summary_evaluation",
    "tests.integration.test_grounded_summary_persistence",
    "tests.integration.test_memory_answer",
    "tests.integration.test_phase5_conflict_evaluation",
    "tests.integration.test_retrieval_baseline_evaluation",
    "tests.integration.test_retrieval_baselines",
    "tests.integration.test_retrieval_index",
    "tests.integration.test_retrieval_query_evaluation",
    "tests.integration.test_retrieval_query_planning",
    "tests.integration.test_sessionization",
    "tests.integration.test_temporal_service",
    "tests.unit.test_answer_quality_evaluation",
    "tests.unit.test_answerability",
    "tests.unit.test_answerability_calibration",
    "tests.unit.test_atomic_extraction_contracts",
    "tests.unit.test_atomic_extraction_runner",
    "tests.unit.test_atomic_extraction_schema",
    "tests.unit.test_atomic_extraction_scoring",
    "tests.unit.test_b6_b7_comparison_prerequisite",
    "tests.unit.test_b7_evaluation",
    "tests.unit.test_belief_resolution_persistence",
    "tests.unit.test_conflict_candidates",
    "tests.unit.test_conflict_classifier",
    "tests.unit.test_conflict_relations",
    "tests.unit.test_conflict_resolver",
    "tests.unit.test_durative_claims",
    "tests.unit.test_evidence_package",
    "tests.unit.test_grounded_summaries",
    "tests.unit.test_ingestion_contracts",
    "tests.unit.test_memory_answer",
    "tests.unit.test_openai_client",
    "tests.unit.test_phase5_conflict_evaluation",
    "tests.unit.test_prediction_contract",
    "tests.unit.test_retrieval_baseline_evaluation",
    "tests.unit.test_retrieval_baselines",
    "tests.unit.test_retrieval_index",
    "tests.unit.test_retrieval_query_evaluation",
    "tests.unit.test_retrieval_query_planning",
    "tests.unit.test_run_config",
    "tests.unit.test_sessionization",
    "tests.unit.test_storage_contracts",
    "tests.unit.test_summary_persistence",
    "tests.unit.test_temporal_contracts",
)
SAFE_DISCOVERY_EXCLUDED_MODULES = (
    "tests.integration.test_b7_evaluation",
    "tests.integration.test_ingestion_service",
    "tests.integration.test_interactive_answering_v2",
    "tests.integration.test_phase4_storage",
    "tests.integration.test_retrieval_index_evaluation",
    "tests.integration.test_retrieval_quality_evaluation",
    "tests.integration.test_summary_quality_evaluation",
    "tests.integration.test_temporal_evaluation",
    "tests.unit.test_atomic_extraction_development_analysis",
    "tests.unit.test_atomic_extraction_failure_analysis",
    "tests.unit.test_atomic_extraction_gold",
    "tests.unit.test_atomic_extraction_pipeline",
    "tests.unit.test_atomic_extraction_prompt",
    "tests.unit.test_atomic_extraction_run_safety",
    "tests.unit.test_atomic_extraction_source",
    "tests.unit.test_atomic_extraction_step34_selection",
    "tests.unit.test_atomic_extraction_v5_development_analysis",
    "tests.unit.test_atomic_extraction_v6_development_analysis",
    "tests.unit.test_atomic_extraction_v7_development_analysis",
    "tests.unit.test_atomic_extraction_v8_development_analysis",
    "tests.unit.test_atomic_extraction_v9_development_analysis",
    "tests.unit.test_b1_full_history",
    "tests.unit.test_belief_resolution_evaluation",
    "tests.unit.test_benchmark_release",
    "tests.unit.test_conflict_candidate_evaluation",
    "tests.unit.test_conflict_relation_evaluation",
    "tests.unit.test_durative_evaluation",
    "tests.unit.test_failure_analysis",
    "tests.unit.test_grounded_summary_evaluation",
    "tests.unit.test_history_prompt",
    "tests.unit.test_interactive_answering_v2",
    "tests.unit.test_load_corpus",
    "tests.unit.test_phase4_fallback_runner",
    "tests.unit.test_phase4_input_contract",
    "tests.unit.test_phase4_input_runner",
    "tests.unit.test_predicate_registry",
    "tests.unit.test_retrieval_index_evaluation",
    "tests.unit.test_retrieval_quality_evaluation",
    "tests.unit.test_scaled_atomic_scoring",
    "tests.unit.test_scaled_atomic_source",
    "tests.unit.test_scaled_release",
    "tests.unit.test_scoring",
    "tests.unit.test_sessionization_evaluation",
    "tests.unit.test_smoke_runner",
    "tests.unit.test_summary_quality_evaluation",
    "tests.unit.test_temporal_evaluation",
)


def load_b7_evaluation_config(
    path: str | Path = CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> tuple[Mapping[str, object], str]:
    root = Path(repo_root).resolve()
    target = _resolve(root, path)
    raw = target.read_bytes()
    value = json.loads(raw)
    expected = {
        "evaluation_version", "schema_version", "config_version", "scorer_version",
        "dataset_version", "final_release_version", "development_case_count",
        "roadmap_target_case_count", "frozen_test_deferred_count", "baseline_order",
        "prerequisite_checkpoint_sha256", "prerequisite_pairs_sha256",
        "prerequisite_b6_predictions_sha256", "prerequisite_b7_predictions_sha256",
        "step9_3_behaviour_reference_sha256", "configured_future_answer_model",
        "memory_answer_config_sha256", "prompt_version", "prompt_sha256",
        "generation_settings", "provider_execution_enabled", "provider_request_count",
        "incremental_cost_usd", "historical_openai_spend_usd",
        "baseline_metric_order", "delta_metric_order", "null_reason_policy",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise B7EvaluationError("evaluation config fields changed")
    if (
        value["evaluation_version"] != EVALUATION_VERSION
        or value["schema_version"] != SCHEMA_VERSION
        or value["config_version"] != CONFIG_VERSION
        or value["scorer_version"] != SCORER_VERSION
        or value["dataset_version"] != DATASET_VERSION
        or value["final_release_version"] != FINAL_RELEASE_VERSION
        or (value["development_case_count"], value["roadmap_target_case_count"], value["frozen_test_deferred_count"]) != (4, 20, 16)
        or value["baseline_order"] != list(BASELINES)
        or value["prerequisite_checkpoint_sha256"] != PREREQUISITE_CHECKPOINT_SHA256
        or value["prerequisite_pairs_sha256"] != PREREQUISITE_PAIRS_SHA256
        or value["prerequisite_b6_predictions_sha256"] != PREREQUISITE_B6_SHA256
        or value["prerequisite_b7_predictions_sha256"] != PREREQUISITE_B7_SHA256
        or value["step9_3_behaviour_reference_sha256"] != SOURCE_REFERENCE_SHA256
        or value["configured_future_answer_model"] != "gpt-4.1-2025-04-14"
        or value["memory_answer_config_sha256"] != MEMORY_CONFIG_SHA256
        or value["prompt_version"] != "memory_answer_prompt_v1"
        or value["prompt_sha256"] != PROMPT_SHA256
        or value["generation_settings"] != {
            "provider": "OpenAI", "api": "responses", "temperature": 0.0,
            "max_output_tokens": 1000, "store": False, "text_format": "json_object",
        }
        or value["provider_execution_enabled"] is not False
        or value["provider_request_count"] != 0
        or value["incremental_cost_usd"] != 0
        or value["historical_openai_spend_usd"] != "0.2314404"
        or value["baseline_metric_order"] != list(BASELINE_METRICS)
        or value["delta_metric_order"] != list(DELTA_METRICS)
        or value["null_reason_policy"] != {
            "answer_accuracy": "no_answered_cases",
            "selective_risk": "no_answered_cases",
            "answer_accuracy_delta": "no_answered_cases_both_baselines",
            "selective_risk_delta": "no_answered_cases_both_baselines",
        }
    ):
        raise B7EvaluationError("evaluation config policy changed")
    return value, hashlib.sha256(raw).hexdigest()


def prepare_b7_evaluation_reference(
    dataset_root: str | Path = DATASET_ROOT,
    *,
    repo_root: str | Path = ".",
    record_reader: Callable[[Path], object] | None = None,
) -> tuple[B7EvaluationReference, ...]:
    """Verify the checkpoint, then copy exactly four scorer-only decision labels."""
    root = Path(repo_root).resolve()
    target = _resolve(root, dataset_root)
    _require_absent_or_empty(target)
    _verify_prerequisite(root)
    config, config_sha256 = load_b7_evaluation_config(repo_root=root)
    reader = record_reader or _four_row_reader
    source = root / SOURCE_REFERENCE
    if _sha(source) != SOURCE_REFERENCE_SHA256:
        raise B7EvaluationError("Step 9.3 behaviour reference changed")
    raw_rows = reader(source)
    if not isinstance(raw_rows, tuple) or len(raw_rows) != 4:
        raise B7EvaluationError("reference reader did not return exactly four rows")
    references = []
    for value in raw_rows:
        if not isinstance(value, dict):
            raise B7EvaluationError("reference source row is invalid")
        projected = {
            "evaluation_version": EVALUATION_VERSION,
            "case_id": value.get("case_id"),
            "user_id": value.get("user_id"),
            "expected_decision": value.get("expected_decision"),
            "review_status": "implementation_reviewed",
        }
        references.append(B7EvaluationReference(reference_id=stable_sha256(projected), **projected))
    _validate_reference_order(tuple(references))
    expected_bytes = _serialize(references)
    review = {
        "review_status": "implementation_reviewed_nonblind_development",
        "development_records_requested": 4,
        "record_five_requested": False,
        "expected_answerable_count": 3,
        "expected_abstain_count": 1,
        "expected_clarify_count": 0,
        "reference_response_included": False,
        "claim_or_evidence_content_included": False,
        "checkpoint_verified_before_reference": True,
        "nonblind_development_evaluation": True,
        "frozen_test_opened": False,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
    }
    review_bytes = canonical_json_bytes(review)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "config_sha256": config_sha256,
        "prerequisite_checkpoint_sha256": PREREQUISITE_CHECKPOINT_SHA256,
        "prerequisite_pairs_sha256": PREREQUISITE_PAIRS_SHA256,
        "prerequisite_b6_predictions_sha256": PREREQUISITE_B6_SHA256,
        "prerequisite_b7_predictions_sha256": PREREQUISITE_B7_SHA256,
        "step9_3_behaviour_reference_sha256": SOURCE_REFERENCE_SHA256,
        "reference_expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "reference_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "development_case_count": 4,
        "expected_decision_counts": {"answerable": 3, "abstain": 1, "clarify": 0},
        "checkpoint_verified_before_reference": True,
        "reference_fields": ["case_id", "expected_decision", "user_id"],
        "nonblind_development_evaluation": True,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
        "excluded_inputs": [
            "answer and relevance gold", "credentials and environment", "frozen test records",
            "network and providers", "oracle and review queues", "reference answers and wording",
            "runtime database and package source content", "scaled record five",
        ],
    }
    target.mkdir(parents=True, exist_ok=True)
    _write_exclusive(target / "reference/expected.jsonl", expected_bytes)
    _write_exclusive(target / "reference/review.json", review_bytes)
    _write_exclusive(target / "manifest.json", canonical_json_bytes(manifest))
    _verify_dataset(root, config)
    return tuple(references)


def execute_b7_evaluation(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[B7PairDelta, ...]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_absent_or_empty(output)
    checkpoint = _verify_prerequisite(root)
    config, config_sha256 = load_b7_evaluation_config(repo_root=root)
    references = _verify_dataset(root, config)
    pairs, predictions = _load_prerequisite_records(root)
    per_case, deltas = _score_cases(references, pairs, predictions)
    scorecard = _scorecard(per_case)
    checks = B7EvaluationChecks(
        EVALUATION_VERSION, FINAL_RELEASE_VERSION,
        4, 4, 8, 4, 4, 4, 4, 4, 0, 0, 3, 1, 0, 0, 0, 3, 3, 0, 0, 0,
        20, 4, 16, False, True, False,
    )
    run = {
        "evaluation_version": EVALUATION_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "prerequisite_checkpoint_sha256": PREREQUISITE_CHECKPOINT_SHA256,
        "prerequisite_case_count": checkpoint.case_count,
        "development_case_count": 4,
        "frozen_test_deferred_count": 16,
        "configured_future_answer_model": config["configured_future_answer_model"],
        "provider_returned_model": None,
        "provider_request_count": 0,
        "retry_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "incremental_cost_usd": 0,
        "step9_4_development_evaluation_complete": True,
        "phase9_exit": False,
    }
    findings = (
        "# Development B6/B7 evaluation\n\n"
        "B6 and B7 both had zero coverage and produced the same abstention on all four nonblind development cases. "
        "The Step 9 gate did not change an output. Three answerable cases were unnecessarily abstained, while the one expected abstention was preserved.\n\n"
        "Answer accuracy and selective risk are null because neither baseline answered a case. This small development result does not show that B7 improves on B6, and it does not establish production quality. "
        "The remaining sixteen frozen cases, a full B6 answer release, and provider execution were not run. Phase 9 is not complete.\n\n"
        "The implementer had prior pilot answer-reference exposure and the reviewer had prior frozen-snippet exposure; neither was used here. The contract derivation used no prohibited material.\n"
        "\nThe unrestricted scaled validator and unrestricted full test discovery were intentionally not run because they can open mixed or prohibited inputs. "
        "The committed scaled manifest was validated without following its file references. A read-trapped safe run included 55 test modules and excluded the 46 modules that directly open mixed scaled, pilot, gold, oracle, or review data; it ran 602 tests with no prohibited open or failure.\n"
    ).encode("utf-8")
    payloads = {
        "per-case.jsonl": _serialize(per_case),
        "pair-deltas.jsonl": _serialize(deltas),
        "scorecard.json": canonical_json_bytes(scorecard),
        "checks.json": canonical_json_bytes(checks),
        "failures.jsonl": b"",
        "run.json": canonical_json_bytes(run),
        "findings.md": findings,
    }
    manifest = _result_manifest(root, config_sha256, payloads, checks)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        _write_exclusive(output / name, payloads[name])
    _write_exclusive(output / "manifest.json", canonical_json_bytes(manifest))
    verify_b7_evaluation(output, repo_root=root)
    return deltas


def verify_b7_evaluation(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> B7EvaluationChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    checkpoint = _verify_prerequisite(root)
    config, config_sha256 = load_b7_evaluation_config(repo_root=root)
    references = _verify_dataset(root, config)
    if not output.is_dir() or {item.name for item in output.iterdir()} != {*ARTIFACTS, "manifest.json"}:
        raise B7EvaluationError("result artifact set changed")
    per_case = tuple(per_case_from_mapping(item) for item in _read_jsonl(output / "per-case.jsonl"))
    deltas = tuple(pair_delta_from_mapping(item) for item in _read_jsonl(output / "pair-deltas.jsonl"))
    scorecard = scorecard_from_mapping(_read_object(output / "scorecard.json"))
    checks = checks_from_mapping(_read_object(output / "checks.json"))
    if _read_jsonl(output / "failures.jsonl"):
        raise B7EvaluationError("evaluation failures are not empty")
    pairs, predictions = _load_prerequisite_records(root)
    expected_per_case, expected_deltas = _score_cases(references, pairs, predictions)
    if per_case != expected_per_case or deltas != expected_deltas or scorecard != _scorecard(per_case):
        raise B7EvaluationError("evaluation output did not recompute")
    expected_checks = B7EvaluationChecks(
        EVALUATION_VERSION, FINAL_RELEASE_VERSION,
        4, 4, 8, 4, 4, 4, 4, 4, 0, 0, 3, 1, 0, 0, 0, 3, 3, 0, 0, 0,
        20, 4, 16, False, True, False,
    )
    if checks != expected_checks or checkpoint.case_count != 4:
        raise B7EvaluationError("evaluation accounting did not recompute")
    manifest = _read_object(output / "manifest.json")
    expected_payloads = {name: (output / name).read_bytes() for name in ARTIFACTS}
    if manifest != _result_manifest(root, config_sha256, expected_payloads, checks):
        raise B7EvaluationError("result manifest did not recompute")
    return checks


def _score_cases(references, pairs, predictions):
    references_by_case = {item.case_id: item for item in references}
    if len(references_by_case) != 4:
        raise B7EvaluationError("reference case IDs are duplicated")
    b6 = {item.prediction_id: item for item in predictions["B6"]}
    b7 = {item.prediction_id: item for item in predictions["B7"]}
    per_case = []
    deltas = []
    for pair in pairs:
        reference = references_by_case.get(pair.case_id)
        left = b6.get(pair.b6_prediction_id)
        right = b7.get(pair.b7_prediction_id)
        if reference is None or left is None or right is None:
            raise B7EvaluationError("pair input is incomplete")
        if pair.user_id != reference.user_id or left.user_id != pair.user_id or right.user_id != pair.user_id:
            raise B7EvaluationError("cross-user pair input")
        pair_hash = hashlib.sha256(prerequisite_json_bytes(pair)).hexdigest()
        left_row = _per_case(reference, pair, pair_hash, left, "B6")
        right_row = _per_case(reference, pair, pair_hash, right, "B7")
        fields = {
            "evaluation_version": EVALUATION_VERSION,
            "schema_version": SCHEMA_VERSION,
            "case_id": pair.case_id,
            "user_id": pair.user_id,
            "pair_id": pair.pair_id,
            "pair_sha256": pair_hash,
            "shared_identity_sha256": pair.shared_identity_sha256,
            "b6_per_case_id": left_row.per_case_id,
            "b6_per_case_sha256": hashlib.sha256(canonical_json_bytes(left_row)).hexdigest(),
            "b7_per_case_id": right_row.per_case_id,
            "b7_per_case_sha256": hashlib.sha256(canonical_json_bytes(right_row)).hexdigest(),
            "expected_decision": reference.expected_decision,
            "b6_answered": left_row.answered,
            "b7_answered": right_row.answered,
            "b6_abstained": left_row.abstained,
            "b7_abstained": right_row.abstained,
            "b6_false_answer": left_row.false_answer,
            "b7_false_answer": right_row.false_answer,
            "b6_unnecessary_abstention": left_row.unnecessary_abstention,
            "b7_unnecessary_abstention": right_row.unnecessary_abstention,
            "gate_changed_output": (
                left_row.answered != right_row.answered or left_row.abstained != right_row.abstained
            ),
        }
        delta = B7PairDelta(pair_delta_id=stable_sha256(fields), **fields)
        per_case.extend((left_row, right_row))
        deltas.append(delta)
    if len(per_case) != 8 or len(deltas) != 4:
        raise B7EvaluationError("evaluation case accounting changed")
    return tuple(per_case), tuple(deltas)


def _per_case(reference, pair, pair_hash, prediction, baseline):
    if prediction.wrapper_baseline_id != baseline or prediction.case_id != reference.case_id:
        raise B7EvaluationError("prediction baseline or case changed")
    prediction_hash = hashlib.sha256(prerequisite_json_bytes(prediction)).hexdigest()
    expected_hash = pair.b6_prediction_sha256 if baseline == "B6" else pair.b7_prediction_sha256
    if prediction_hash != expected_hash:
        raise B7EvaluationError("prediction hash differs from pair")
    answered = prediction.response_status == "answered"
    abstained = prediction.response_status == "abstained"
    fields = {
        "evaluation_version": EVALUATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "scorer_version": SCORER_VERSION,
        "dataset_version": DATASET_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "case_id": reference.case_id,
        "user_id": reference.user_id,
        "baseline_id": baseline,
        "pair_id": pair.pair_id,
        "pair_sha256": pair_hash,
        "shared_identity_sha256": pair.shared_identity_sha256,
        "prediction_id": prediction.prediction_id,
        "prediction_sha256": prediction_hash,
        "expected_decision": reference.expected_decision,
        "predicted_status": prediction.response_status,
        "predicted_action": prediction.response_action,
        "answered": answered,
        "abstained": abstained,
        "answer_correct": (reference.expected_decision == "answerable") if answered else None,
        "false_answer": answered and reference.expected_decision != "answerable",
        "unnecessary_abstention": abstained and reference.expected_decision == "answerable",
    }
    return B7PerCase(per_case_id=stable_sha256(fields), **fields)


def _scorecard(per_case: Sequence[B7PerCase]) -> B7Scorecard:
    rows = []
    baseline_values = {}
    for baseline in BASELINES:
        cases = tuple(item for item in per_case if item.baseline_id == baseline)
        answered = sum(item.answered for item in cases)
        abstained = sum(item.abstained for item in cases)
        expected_abstain = sum(item.expected_decision == "abstain" for item in cases)
        expected_unanswerable = sum(item.expected_decision != "answerable" for item in cases)
        expected_answerable = sum(item.expected_decision == "answerable" for item in cases)
        correct_abstain = sum(item.abstained and item.expected_decision == "abstain" for item in cases)
        correct_answers = sum(item.answer_correct is True for item in cases)
        false_answers = sum(item.false_answer for item in cases)
        unnecessary = sum(item.unnecessary_abstention for item in cases)
        values = {
            "abstention_precision": (correct_abstain, abstained, None),
            "abstention_recall": (correct_abstain, expected_abstain, None),
            "coverage": (answered, len(cases), None),
            "answer_accuracy": (correct_answers, answered, "no_answered_cases"),
            "selective_risk": (answered - correct_answers, answered, "no_answered_cases"),
            "false_answer_rate": (false_answers, expected_unanswerable, None),
            "unnecessary_abstention_rate": (unnecessary, expected_answerable, None),
        }
        baseline_values[baseline] = values
        rows.extend(_metric("baseline", baseline, name, *values[name]) for name in BASELINE_METRICS)
    delta_map = {
        "coverage_delta": _delta(baseline_values, "coverage"),
        "abstention_precision_delta": _delta(baseline_values, "abstention_precision"),
        "abstention_recall_delta": _delta(baseline_values, "abstention_recall"),
        "answer_accuracy_delta": (0, 0, "no_answered_cases_both_baselines"),
        "selective_risk_delta": (0, 0, "no_answered_cases_both_baselines"),
        "false_answer_rate_delta": _delta(baseline_values, "false_answer_rate"),
        "unnecessary_abstention_rate_delta": _delta(baseline_values, "unnecessary_abstention_rate"),
        "gate_changed_output_count": (
            sum(
                left.answered != right.answered or left.abstained != right.abstained
                for left, right in zip(per_case[0::2], per_case[1::2], strict=True)
            ),
            len(per_case) // 2,
            None,
        ),
    }
    rows.extend(_metric("pair_delta", None, name, *delta_map[name]) for name in DELTA_METRICS)
    return B7Scorecard(
        EVALUATION_VERSION, SCHEMA_VERSION, CONFIG_VERSION, SCORER_VERSION,
        DATASET_VERSION, FINAL_RELEASE_VERSION, None, tuple(rows),
    )


def _delta(values, name):
    left_n, left_d, _ = values["B6"][name]
    right_n, right_d, _ = values["B7"][name]
    if not left_d or not right_d:
        raise B7EvaluationError("non-null delta has an empty denominator")
    return right_n * left_d - left_n * right_d, right_d * left_d, None


def _metric(scope, baseline, name, numerator, denominator, null_reason):
    value = None if denominator == 0 else f"{float(Fraction(numerator, denominator)):.6f}"
    return MetricValue(scope, baseline, name, numerator, denominator, value, null_reason if denominator == 0 else None)


def _load_prerequisite_records(root: Path):
    pairs = tuple(pair_from_mapping(item) for item in _read_jsonl(root / PREREQUISITE_ROOT / "pairs.jsonl"))
    b6 = tuple(prediction_from_mapping(item) for item in _read_jsonl(root / PREREQUISITE_ROOT / "b6-predictions.jsonl"))
    b7 = tuple(prediction_from_mapping(item) for item in _read_jsonl(root / PREREQUISITE_ROOT / "b7-predictions.jsonl"))
    if len(pairs) != 4 or len(b6) != 4 or len(b7) != 4:
        raise B7EvaluationError("prerequisite record accounting changed")
    if tuple(item.case_id for item in pairs) != tuple(item.case_id for item in b6) or tuple(item.case_id for item in pairs) != tuple(item.case_id for item in b7):
        raise B7EvaluationError("prerequisite record order changed")
    return pairs, {"B6": b6, "B7": b7}


def _verify_prerequisite(root: Path):
    checkpoint_path = root / PREREQUISITE_ROOT / "checkpoint_manifest.json"
    if _sha(checkpoint_path) != PREREQUISITE_CHECKPOINT_SHA256:
        raise B7EvaluationError("prerequisite checkpoint changed")
    checkpoint = checkpoint_from_mapping(_read_object(checkpoint_path))
    exact = {
        "preflight.json": "f01e96cf028f5d4504033fe52c58273cb6b06d7fca2dc94296ea63733a895219",
        "pairs.jsonl": PREREQUISITE_PAIRS_SHA256,
        "b6-predictions.jsonl": PREREQUISITE_B6_SHA256,
        "b7-predictions.jsonl": PREREQUISITE_B7_SHA256,
        "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    for name, expected in exact.items():
        if _sha(root / PREREQUISITE_ROOT / name) != expected:
            raise B7EvaluationError("prerequisite artifact changed")
    if dict(checkpoint.artifacts) != exact:
        raise B7EvaluationError("prerequisite artifact bindings changed")
    if dict(checkpoint.implementation) != PREREQUISITE_IMPLEMENTATION:
        raise B7EvaluationError("prerequisite implementation bindings changed")
    for path, expected in PREREQUISITE_IMPLEMENTATION.items():
        if _sha(root / path) != expected:
            raise B7EvaluationError("prerequisite implementation changed")
    pairs, predictions = _load_prerequisite_records(root)
    left = {item.prediction_id: item for item in predictions["B6"]}
    right = {item.prediction_id: item for item in predictions["B7"]}
    if len(left) != 4 or len(right) != 4:
        raise B7EvaluationError("prerequisite prediction IDs are duplicated")
    for pair in pairs:
        b6 = left.get(pair.b6_prediction_id)
        b7 = right.get(pair.b7_prediction_id)
        if b6 is None or b7 is None:
            raise B7EvaluationError("prerequisite pair prediction is missing")
        if (
            hashlib.sha256(prerequisite_json_bytes(b6)).hexdigest() != pair.b6_prediction_sha256
            or hashlib.sha256(prerequisite_json_bytes(b7)).hexdigest() != pair.b7_prediction_sha256
            or (pair.case_id, pair.user_id, pair.input_id) != (b6.case_id, b6.user_id, b6.input_id)
            or (pair.case_id, pair.user_id, pair.input_id) != (b7.case_id, b7.user_id, b7.input_id)
        ):
            raise B7EvaluationError("prerequisite pair identity changed")
    return checkpoint


def _verify_dataset(root: Path, config: Mapping[str, object]):
    if not (root / DATASET_MANIFEST).is_file() or not (root / REFERENCE_PATH).is_file() or not (root / REFERENCE_REVIEW).is_file():
        raise B7EvaluationError("evaluation dataset is incomplete")
    references = tuple(reference_from_mapping(item) for item in _read_jsonl(root / REFERENCE_PATH))
    _validate_reference_order(references)
    review = _read_object(root / REFERENCE_REVIEW)
    if review != {
        "review_status": "implementation_reviewed_nonblind_development",
        "development_records_requested": 4,
        "record_five_requested": False,
        "expected_answerable_count": 3,
        "expected_abstain_count": 1,
        "expected_clarify_count": 0,
        "reference_response_included": False,
        "claim_or_evidence_content_included": False,
        "checkpoint_verified_before_reference": True,
        "nonblind_development_evaluation": True,
        "frozen_test_opened": False,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
    }:
        raise B7EvaluationError("reference review changed")
    manifest = _read_object(root / DATASET_MANIFEST)
    expected = {
        "dataset_version": DATASET_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "config_sha256": _sha(root / CONFIG_PATH),
        "prerequisite_checkpoint_sha256": PREREQUISITE_CHECKPOINT_SHA256,
        "prerequisite_pairs_sha256": PREREQUISITE_PAIRS_SHA256,
        "prerequisite_b6_predictions_sha256": PREREQUISITE_B6_SHA256,
        "prerequisite_b7_predictions_sha256": PREREQUISITE_B7_SHA256,
        "step9_3_behaviour_reference_sha256": SOURCE_REFERENCE_SHA256,
        "reference_expected_sha256": _sha(root / REFERENCE_PATH),
        "reference_review_sha256": _sha(root / REFERENCE_REVIEW),
        "development_case_count": 4,
        "expected_decision_counts": {"answerable": 3, "abstain": 1, "clarify": 0},
        "checkpoint_verified_before_reference": True,
        "reference_fields": ["case_id", "expected_decision", "user_id"],
        "nonblind_development_evaluation": True,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
        "excluded_inputs": [
            "answer and relevance gold", "credentials and environment", "frozen test records",
            "network and providers", "oracle and review queues", "reference answers and wording",
            "runtime database and package source content", "scaled record five",
        ],
    }
    if manifest != expected or config["step9_3_behaviour_reference_sha256"] != SOURCE_REFERENCE_SHA256:
        raise B7EvaluationError("evaluation dataset manifest changed")
    return references


def _result_manifest(root, config_sha256, payloads, checks):
    return {
        "evaluation_version": EVALUATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "scorer_version": SCORER_VERSION,
        "dataset_version": DATASET_VERSION,
        "final_release_version": FINAL_RELEASE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "boundary_ruling": {
            "version": BOUNDARY_RULING_VERSION,
            "preimage": BOUNDARY_RULING_PREIMAGE,
            "sha256": BOUNDARY_RULING_SHA256,
        },
        "starting_commit": STARTING_COMMIT,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": _sha(root / DATASET_MANIFEST),
        "prerequisite_checkpoint_sha256": PREREQUISITE_CHECKPOINT_SHA256,
        "artifacts": [[name, hashlib.sha256(payloads[name]).hexdigest()] for name in sorted(payloads)],
        "implementation": [[path.as_posix(), _sha(root / path)] for path in sorted(IMPLEMENTATION_PATHS)],
        "committed_test_authorities": [
            [path, _git_show_sha(root, STARTING_COMMIT, path)] for path in ADAPTER_PATHS
        ],
        "live_test_adapters": [[path, _sha(root / path)] for path in ADAPTER_PATHS],
        "authorized_final_paths": list(FINAL_AUTHORIZED_PATHS),
        "counts": asdict(checks),
        "metric_policy": {
            "baseline_order": list(BASELINES),
            "baseline_metrics": list(BASELINE_METRICS),
            "delta_metrics": list(DELTA_METRICS),
            "null_reasons": ["no_answered_cases", "no_answered_cases_both_baselines"],
            "composite_score": None,
        },
        "provider_request_count": 0,
        "incremental_cost_usd": 0,
        "nonblind_development_evaluation": True,
        "implementer_pilot_answer_reference_exposure": True,
        "implementer_pilot_answer_reference_used": False,
        "reviewer_frozen_snippet_exposure": True,
        "reviewer_frozen_snippet_used": False,
        "root_contract_deriver_prohibited_exposure": False,
        "limitations": [
            "four development cases only", "frozen sixteen deferred", "no answered predictions",
            "no evidence of B7 improvement", "Phase 9 exit is false", "provider was not run",
            "unrestricted full discovery was intentionally not run",
            "validate-scaled-benchmark was intentionally not run",
        ],
        "safe_test_evidence": {
            "scaled_manifest": {
                "path": "data/scaled-v1/manifest.json",
                "sha256": SCALED_MANIFEST_SHA256,
                "dataset_sha256": SCALED_DATASET_SHA256,
                "referenced_files_opened": False,
            },
            "test_module_count": 101,
            "included_module_count": 55,
            "included_modules": list(SAFE_DISCOVERY_INCLUDED_MODULES),
            "excluded_module_count": 46,
            "excluded_modules": list(SAFE_DISCOVERY_EXCLUDED_MODULES),
            "test_count": 602,
            "skip_count": 130,
            "unexpected_prohibited_open_count": 0,
            "failure_count": 0,
            "unrestricted_full_discovery_run": False,
            "validate_scaled_benchmark_run": False,
        },
    }


def _validate_reference_order(references):
    expected = (
        ("scaled_user_001_interactive_extraction_001", "user_001", "answerable"),
        ("scaled_user_001_interactive_temporal_reasoning_002", "user_001", "answerable"),
        ("scaled_user_002_interactive_conflict_detection_001", "user_002", "answerable"),
        ("scaled_user_002_interactive_abstention_002", "user_002", "abstain"),
    )
    if tuple((item.case_id, item.user_id, item.expected_decision) for item in references) != expected:
        raise B7EvaluationError("reference order or decisions changed")


def _four_row_reader(path: Path):
    rows = []
    with path.open("rb") as stream:
        for _ in range(4):
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                raise B7EvaluationError("development reference prefix is incomplete")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise B7EvaluationError("development reference row is invalid")
            rows.append(value)
    return tuple(rows)


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(item) for item in values)


def _read_jsonl(path: Path):
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise B7EvaluationError("JSONL row lacks newline")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise B7EvaluationError("JSONL row is invalid")
        rows.append(value)
    return tuple(rows)


def _read_object(path: Path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise B7EvaluationError("JSON object is invalid")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_absent_or_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise B7EvaluationError("output directory must be empty")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_show_sha(root: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=root, check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()
