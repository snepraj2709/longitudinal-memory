"""Metadata-only finalization for the Step 9.2 threshold release.

The v1 runtime checkpoint and calibration implementation are immutable.  This
module regenerates their deterministic v1 final in a temporary directory,
checks its fixed hashes, and carries the substantive payloads into v2 without
changing them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping

from .calibration_contracts import canonical_json_bytes
from .calibration_evaluation import (
    CalibrationEvaluationError,
    score_threshold_profiles,
    verify_answerability_threshold_release,
    verify_threshold_runtime_checkpoint,
)


V2_RELEASE_VERSION = "answerability-threshold-development-v2"
V2_ROOT = Path("results/abstention/answerability-threshold-development-v2")
RUNTIME_ROOT = Path("results/abstention/answerability-threshold-development-runtime-v1")
CONFIG_PATH = Path("configs/abstention/answerability_thresholds_v1.json")
RUNTIME_DATASET_MANIFEST = Path(
    "data/abstention/answerability-threshold-development-v1/runtime/manifest.json"
)
FIXTURES_PATH = Path(
    "data/abstention/answerability-threshold-development-v1/runtime/fixtures.jsonl"
)
REFERENCE_PATH = Path(
    "data/abstention/answerability-threshold-development-v1/reference/expected.jsonl"
)
DATASET_MANIFEST = Path("data/abstention/answerability-threshold-development-v1/manifest.json")
FINALIZER_PATH = Path("src/abstention/calibration_release_v2.py")

BASE_GUIDANCE_SHA256 = "c815ed3785101c08ce563c42c802c4513ab1bb00f5213e21e77098369d751628"
RULING_1_SHA256 = "04a22f11ce263f3ec9ddb87b4b821881d7bbec72a655fe5cb4fc34cffbb83f81"
RULING_2_SHA256 = "930d0059bbff3323cd38855b48d4a1ca6ae97b4b35baae4653906fd50c1c9139"
RULING_3_SHA256 = "af76ef9b298cb600cb7214242993d63c7c318da5b47326a24b409275a6c52f59"
RULING_4_SHA256 = "4b6c8a864310ff0823b5dd493abd726be8406bbe945f7b1839daf9a42f9afea7"
START_COMMIT = "35d0c64431f19d4243b72af712dadeb8d522128f"
RUNTIME_CHECKPOINT_SHA256 = "5a66f85a0c690879771f36f3f6293d190cd672c9b291ee9c7f93f2965f39ce88"

RUNTIME_IMPLEMENTATION_HASHES = {
    "src/abstention/calibration_contracts.py": "3dcff2419b33dd5958b1d06215ff38f29463444c1695e9be4b4e7b410f3e2e8c",
    "src/abstention/calibration.py": "67407be6be00603b887984b3217a9e3522ce6331db8cc73e7c70a2a819de2daf",
    "src/abstention/calibration_evaluation.py": "51162ea629771f3d30030dc443e82b3b860025ddff1d3f1e036899a408a29077",
}
FROZEN_INPUT_HASHES = {
    "configs/abstention/answerability_thresholds_v1.json": "cf33994fb0a13994748b1d3c67a53a15396e30f51b65dd042a191c18e7127e02",
    "data/abstention/answerability-threshold-development-v1/runtime/manifest.json": "4a8b84836dbe3ea038e83c6396855a32a5217dafabf32f9ef9f4f6d4a675f249",
    "data/abstention/answerability-threshold-development-v1/runtime/fixtures.jsonl": "82cee5781648819fb90094f9ab3244c2d0a8d85528e1fd2bf49df553d4765d51",
    "data/abstention/answerability-threshold-development-v1/reference/expected.jsonl": "b76ff615d9b0b60252691cfe6bdf9361b8f8894f615d8e53c25c9956b0b732a7",
    "data/abstention/answerability-threshold-development-v1/manifest.json": "e08cd19ba33a7bdd7970c1ebaa3ee1920088673450488da2a5ec7c972e149d47",
}
RUNTIME_ARTIFACT_HASHES = {
    "preflight.json": "214aaa6244cb9fad00c722cfcfdd11530c02a2dc752fbc981d488263b21dcc7b",
    "predictions.jsonl": "4b7db7c9427ada99430e49b9a42b14f4c9b92de749f83f876897064eb6b9acd0",
    "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "checkpoint_manifest.json": RUNTIME_CHECKPOINT_SHA256,
}
COPIED_ARTIFACT_HASHES = {
    "checks.json": "91acd39d1d382b40339e6f12b19cf3ccd10129acb6e0fcb56d3b2bc0394cf7bd",
    "development-decisions.jsonl": "28ca79bac049c1ac46d64864902a5647b1f68326ccf5e9bf38ba0032aa81b5fe",
    "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "fixture-results.jsonl": "4c43beb98e4dc4032279125b7c590ecd50a6e9c1ff99ffd74c39bb929fdbd0a0",
    "scorecard.json": "c2e94d82d8ca809fbb34ef5df7366701bdc124d5632c51608d964f50d9e4a3b6",
    "threshold-sweep.jsonl": "e809cc4b2731000fcd11aaa5eda497f98bba2c5b514018fb0bb44a7481127b91",
    "thresholds.json": "26d814b0ecffdcc70d8af66dd462c5c1233b5a231d2f6079198e5199dd11e11a",
}
V1_INTERMEDIATE_HASHES = {
    **COPIED_ARTIFACT_HASHES,
    "run.json": "cedc0d9086e19038b88a9f8ccfd2952291f27e264eac1d561e452da8d2385807",
    "findings.md": "f64c6f403047b3c7bf870e258b50b5ce0aab6f02bf5c4e7f01ec54d00ab3796d",
    "manifest.json": "f96d1142d553249ce2405a4150a2e0c75dd1d0c7cb79f645f6cf5063c19b9fa5",
}
V2_ARTIFACTS = (*COPIED_ARTIFACT_HASHES, "run.json", "findings.md", "manifest.json")


class CalibrationReleaseV2Error(RuntimeError):
    """Reject an invalid metadata-only carry-forward release."""


def finalize_answerability_threshold_release_v2(
    output_dir: str | Path = V2_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    verify_threshold_runtime_checkpoint(repo_root=root)
    _verify_frozen_bytes(root)
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / "v1-final"
        score_threshold_profiles(intermediate, repo_root=root)
        verify_answerability_threshold_release(intermediate, repo_root=root)
        _verify_v1_intermediate(intermediate)
        copied = {name: (intermediate / name).read_bytes() for name in COPIED_ARTIFACT_HASHES}
    run = canonical_json_bytes(_run())
    findings = _findings().encode("utf-8")
    payloads = {**copied, "run.json": run, "findings.md": findings}
    manifest = canonical_json_bytes(_manifest(root, payloads))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_exclusive(output / name, payload)
    _write_exclusive(output / "manifest.json", manifest)
    verify_answerability_threshold_release_v2(output, repo_root=root)


def verify_answerability_threshold_release_v2(
    output_dir: str | Path = V2_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    if not output.is_dir() or {path.name for path in output.iterdir()} != set(V2_ARTIFACTS):
        raise CalibrationReleaseV2Error("v2 final artifact set changed")
    verify_threshold_runtime_checkpoint(repo_root=root)
    _verify_frozen_bytes(root)
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / "v1-final"
        score_threshold_profiles(intermediate, repo_root=root)
        verify_answerability_threshold_release(intermediate, repo_root=root)
        _verify_v1_intermediate(intermediate)
        for name, expected in COPIED_ARTIFACT_HASHES.items():
            if (output / name).read_bytes() != (intermediate / name).read_bytes():
                raise CalibrationReleaseV2Error(f"{name} was not copied byte-for-byte")
            if _sha(output / name) != expected:
                raise CalibrationReleaseV2Error(f"{name} hash changed")
    payloads = {
        name: (output / name).read_bytes()
        for name in (*COPIED_ARTIFACT_HASHES, "run.json", "findings.md")
    }
    if payloads["run.json"] != canonical_json_bytes(_run()):
        raise CalibrationReleaseV2Error("v2 run metadata changed")
    if payloads["findings.md"] != _findings().encode("utf-8"):
        raise CalibrationReleaseV2Error("v2 findings changed")
    if (output / "manifest.json").read_bytes() != canonical_json_bytes(_manifest(root, payloads)):
        raise CalibrationReleaseV2Error("v2 manifest does not recompute")


def _verify_frozen_bytes(root: Path) -> None:
    actual_inputs = {path: _sha(root / path) for path in FROZEN_INPUT_HASHES}
    if actual_inputs != FROZEN_INPUT_HASHES:
        raise CalibrationReleaseV2Error("frozen calibration input changed")
    actual_implementation = {path: _sha(root / path) for path in RUNTIME_IMPLEMENTATION_HASHES}
    if actual_implementation != RUNTIME_IMPLEMENTATION_HASHES:
        raise CalibrationReleaseV2Error("frozen runtime implementation changed")
    actual_runtime = {
        name: _sha(root / RUNTIME_ROOT / name)
        for name in RUNTIME_ARTIFACT_HASHES
    }
    if actual_runtime != RUNTIME_ARTIFACT_HASHES:
        raise CalibrationReleaseV2Error("frozen runtime artifact changed")


def _verify_v1_intermediate(path: Path) -> None:
    actual = {name: _sha(path / name) for name in V1_INTERMEDIATE_HASHES}
    if actual != V1_INTERMEDIATE_HASHES:
        raise CalibrationReleaseV2Error("v1 intermediate final did not reproduce")


def _run() -> Mapping[str, object]:
    return {
        "guidance_version": "step-9.2-guidance-v1",
        "runtime_version": "answerability_calibration_runtime_v1",
        "input_release_version": "answerability-development-v1",
        "output_release_version": V2_RELEASE_VERSION,
        "carry_forward_mode": "metadata_only_no_runtime_change",
        "selected_profile": "ordinary_1_trait_2",
        "development_decision_count": 24,
        "fixture_prediction_count": 144,
        "threshold_sweep_count": 6,
        "runtime_checkpoint_sha256": RUNTIME_CHECKPOINT_SHA256,
        "compatibility_rulings": [
            RULING_1_SHA256,
            RULING_2_SHA256,
            RULING_3_SHA256,
            RULING_4_SHA256,
        ],
        "prior_controlled_reference_exposure": True,
        "model": "deterministic_no_model",
        "provider_requests": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "incremental_cost_usd": 0,
        "historical_openai_spend_usd": "0.2314404",
    }


def _manifest(root: Path, payloads: Mapping[str, bytes]) -> Mapping[str, object]:
    checks = json.loads(payloads["checks.json"])
    return {
        "release_version": V2_RELEASE_VERSION,
        "carry_forward_mode": "metadata_only_no_runtime_change",
        "guidance": {
            "version": "step-9.2-guidance-v1",
            "sha256": BASE_GUIDANCE_SHA256,
            "starting_commit": START_COMMIT,
            "clean_room_boundary": "no_rejected_prompt_content",
            "compatibility_rulings": [
                {"version": "step-9.2-guidance-v1-compatibility-ruling-1", "sha256": RULING_1_SHA256},
                {"version": "step-9.2-guidance-v1-compatibility-ruling-2", "sha256": RULING_2_SHA256},
                {"version": "step-9.2-guidance-v1-compatibility-ruling-3", "sha256": RULING_3_SHA256},
                {"version": "step-9.2-guidance-v1-compatibility-ruling-4", "sha256": RULING_4_SHA256},
            ],
        },
        "frozen_inputs": FROZEN_INPUT_HASHES,
        "runtime_implementation": RUNTIME_IMPLEMENTATION_HASHES,
        "finalization_implementation": {FINALIZER_PATH.as_posix(): _sha(root / FINALIZER_PATH)},
        "runtime_checkpoint": {
            "path": (RUNTIME_ROOT / "checkpoint_manifest.json").as_posix(),
            "sha256": RUNTIME_CHECKPOINT_SHA256,
        },
        "v1_uncommitted_intermediate": {
            "release_version": "answerability-threshold-development-v1",
            "hashes": V1_INTERMEDIATE_HASHES,
            "status": "invalid_uncommitted_metadata_incomplete",
        },
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "copied_substantive_artifacts": COPIED_ARTIFACT_HASHES,
        "counts": checks,
        "selected_profile": "ordinary_1_trait_2",
        "profile_order": [
            "ordinary_1_trait_2",
            "ordinary_1_trait_3",
            "ordinary_2_trait_2",
            "ordinary_2_trait_3",
            "ordinary_3_trait_2",
            "ordinary_3_trait_3",
        ],
        "confidence_policy": {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        },
        "real_metric_null_reason_policy": {
            "answer_accuracy_on_answered_cases": "no_answered_cases_and_no_authorized_answerability_gold",
            "selective_risk": "no_answered_cases",
            "abstention_precision": "no_authorized_answerability_gold",
            "abstention_recall": "no_authorized_answerability_gold",
            "false_answer_rate_on_unanswerable_cases": "no_authorized_answerability_gold",
            "unnecessary_abstention_rate_on_answerable_cases": "no_authorized_answerability_gold",
            "confidence_calibration_error": "no_authorized_answerability_gold",
        },
        "prior_controlled_reference_exposure": True,
        "benchmark_gold_opened": False,
        "runtime_changed": False,
        "threshold_selection_changed": False,
        "substantive_payloads_changed": False,
        "predecessor_drift": [
            {
                "path": "tests/integration/test_answer_quality_evaluation.py",
                "old_sha256": "933a7b9378d99957295004bbc81926707e79e4c7cf6e5f68d3f89d8f76421dc4",
                "new_sha256": "0f66f17abe92b6a244a90cd7b999c48106ffffc0790c24c6fd1feb31c1fced95",
                "reason": "compatibility_ruling_3_step92_topology",
            },
            {
                "path": "tests/integration/test_answerability.py",
                "old_sha256": "97aa053af784bf23626dc65d1720fac48a750f26b0c47b6ec0ec84ced8fe775d",
                "new_sha256": "0e502c029eca08d634780f9948215c312608bd0f15fb1dc51bd82d322827dbd3",
                "reason": "compatibility_rulings_1_2_4_step92_topology_v2_metadata_and_nested_hashes",
            },
            {
                "path": "tests/integration/test_memory_answer.py",
                "old_sha256": "5b68e3cb8037c78e841ce7293498859fc2c1d5397f5fe55b087b1127d3a40818",
                "new_sha256": "ec3c88940ff9b3e8a8040225c7f444dc114d1281dab17105b4848bd709ac0835",
                "reason": "compatibility_ruling_4_step92_topology",
            },
        ],
        "model_usage": {
            "model": "deterministic_no_model",
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
        "limitations": [
            "Real development coverage remains zero because every claim is candidate-only.",
            "No real answerability correctness or confidence calibration was measured.",
            "Controlled fixtures test deterministic mechanics and are not a product score.",
            "The controlled reference was already exposed after the valid runtime checkpoint.",
            "Step 9.3 has not started.",
        ],
    }


def _findings() -> str:
    return (
        "# Answerability threshold findings\n\n"
        "The real development run still answers none of its 24 cases. Every available claim is a candidate, so the selected threshold leaves all decisions at `abstain`.\n\n"
        "There is no authorized answerability gold for these cases. This release does not measure real answer correctness, abstention precision or recall, selective risk, or confidence calibration. Confidence remains null.\n\n"
        "The invented fixture grid checks deterministic threshold mechanics. It is not a benchmark, a product score, or a blind calibration. The selected profile preserves the ontology's identity, time, authority, conflict, trait, causal, and provenance floors without optimizing for maximum abstention. The controlled reference had already been exposed after the valid runtime checkpoint. It did not change the selected profile.\n\n"
        "The uncommitted v1 final was invalid only because compatibility metadata arrived after the runtime freeze. This v2 release carries forward the same checkpoint, controlled reference, selected profile, substantive payloads, metrics, and zero-call policy. No runtime or threshold-selection byte changed.\n\n"
        "Three test-only topology adapters changed under compatibility rulings `04a22f11...`, `930d0059...`, `af76ef9b...`, and `4b6c8a86...`. Their final hashes are `0f66f17a...ed95`, `0e502c02...bd3`, and `ec3c8894...0835`. They only verify predecessor hashes and the Step 9.2 path boundary.\n\n"
        "Step 9.3 has not started. No interactive case, prompt, database, provider, or model path was opened.\n"
    )


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise CalibrationReleaseV2Error("output directory must be absent or empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
