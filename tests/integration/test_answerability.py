from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from abstention.evaluation import (
    AnswerabilityEvaluationError,
    execute_answerability_evaluation,
    verify_answerability_release,
)
from abstention.input import load_answerability_inputs
from abstention.input import evidence_package_from_mapping
from abstention.contracts import AnswerabilityError


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/abstention/answerability-development-v1"
STEP91_START = "a18501a27708c259cccce8bf87948962e672bd41"
STEP91_COMMIT = "35d0c64431f19d4243b72af712dadeb8d522128f"
STEP92_COMMIT = "d0932a7994153745285c1f4e3d75c36ffbbaf06a"
STEP93_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
STEP94_PREREQUISITE_COMMIT = "7e8fc5337384ac329264e3606507b925bd890d63"
STEP94_EVALUATION_COMMIT = "78ed4900fd9a7aecbd7ca8c70b5726b356a07ff4"
STEP101_COMMIT = "b2ae263e1129758325a30db57c03620628c6355e"
AUTHORIZED = (
    "configs/abstention/answerability_v1.json",
    "data/abstention/answerability-development-v1/manifest.json",
    "docs/implementation-progress.md",
    "results/abstention/answerability-development-v1/checks.json",
    "results/abstention/answerability-development-v1/decisions.jsonl",
    "results/abstention/answerability-development-v1/failures.jsonl",
    "results/abstention/answerability-development-v1/findings.md",
    "results/abstention/answerability-development-v1/manifest.json",
    "results/abstention/answerability-development-v1/run.json",
    "src/abstention/__init__.py",
    "src/abstention/contracts.py",
    "src/abstention/evaluation.py",
    "src/abstention/input.py",
    "src/abstention/policy.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_answerability.py",
)
STEP92_AUTHORIZED_DRIFT = (
    "configs/abstention/answerability_thresholds_v1.json",
    "data/abstention/answerability-threshold-development-v1/manifest.json",
    "data/abstention/answerability-threshold-development-v1/reference/expected.jsonl",
    "data/abstention/answerability-threshold-development-v1/runtime/fixtures.jsonl",
    "data/abstention/answerability-threshold-development-v1/runtime/manifest.json",
    "docs/implementation-progress.md",
    "results/abstention/answerability-threshold-development-runtime-v1/checkpoint_manifest.json",
    "results/abstention/answerability-threshold-development-runtime-v1/failures.jsonl",
    "results/abstention/answerability-threshold-development-runtime-v1/predictions.jsonl",
    "results/abstention/answerability-threshold-development-runtime-v1/preflight.json",
    "results/abstention/answerability-threshold-development-v2/checks.json",
    "results/abstention/answerability-threshold-development-v2/development-decisions.jsonl",
    "results/abstention/answerability-threshold-development-v2/failures.jsonl",
    "results/abstention/answerability-threshold-development-v2/findings.md",
    "results/abstention/answerability-threshold-development-v2/fixture-results.jsonl",
    "results/abstention/answerability-threshold-development-v2/manifest.json",
    "results/abstention/answerability-threshold-development-v2/run.json",
    "results/abstention/answerability-threshold-development-v2/scorecard.json",
    "results/abstention/answerability-threshold-development-v2/threshold-sweep.jsonl",
    "results/abstention/answerability-threshold-development-v2/thresholds.json",
    "src/abstention/calibration.py",
    "src/abstention/calibration_contracts.py",
    "src/abstention/calibration_evaluation.py",
    "src/abstention/calibration_release_v2.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_answerability_calibration.py",
    "tests/unit/test_answerability_calibration.py",
)
STEP92_COMPATIBILITY_DRIFT = (
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_memory_answer.py",
)
STEP93_AUTHORIZED_DRIFT = (
    "configs/abstention/interactive_answering_v2.json",
    "data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl",
    "data/abstention/interactive-answering-development-v2/gold/review.json",
    "data/abstention/interactive-answering-development-v2/manifest.json",
    "data/abstention/interactive-answering-development-v2/runtime/cases.jsonl",
    "data/abstention/interactive-answering-development-v2/runtime/manifest.json",
    "data/abstention/interactive-answering-development-v2/runtime/requirements.jsonl",
    "docs/implementation-progress.md",
    "results/abstention/interactive-answering-development-runtime-v2/checkpoint_manifest.json",
    "results/abstention/interactive-answering-development-runtime-v2/decisions.jsonl",
    "results/abstention/interactive-answering-development-runtime-v2/failures.jsonl",
    "results/abstention/interactive-answering-development-runtime-v2/packages.jsonl",
    "results/abstention/interactive-answering-development-runtime-v2/plans.jsonl",
    "results/abstention/interactive-answering-development-runtime-v2/preflight.json",
    "results/abstention/interactive-answering-development-runtime-v2/responses.jsonl",
    "results/abstention/interactive-answering-development-runtime-v2/retrieval-results.jsonl",
    "results/abstention/interactive-answering-development-v2/checks.json",
    "results/abstention/interactive-answering-development-v2/failures.jsonl",
    "results/abstention/interactive-answering-development-v2/findings.md",
    "results/abstention/interactive-answering-development-v2/manifest.json",
    "results/abstention/interactive-answering-development-v2/per-case.jsonl",
    "results/abstention/interactive-answering-development-v2/predictions.jsonl",
    "results/abstention/interactive-answering-development-v2/run.json",
    "results/abstention/interactive-answering-development-v2/scorecard.json",
    "src/abstention/interactive_contracts_v2.py",
    "src/abstention/interactive_evaluation_v2.py",
    "src/abstention/interactive_input_v2.py",
    "src/abstention/interactive_runtime_v2.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_interactive_answering_v2.py",
)
STEP94_PREREQUISITE_AUTHORIZED_DRIFT = (
    "configs/abstention/b6_b7_comparable_runtime_v1.json",
    "data/abstention/b6-b7-comparable-development-v1/runtime/manifest.json",
    "docs/implementation-progress.md",
    "results/abstention/b6-b7-comparable-development-runtime-v1/b6-predictions.jsonl",
    "results/abstention/b6-b7-comparable-development-runtime-v1/b7-predictions.jsonl",
    "results/abstention/b6-b7-comparable-development-runtime-v1/checkpoint_manifest.json",
    "results/abstention/b6-b7-comparable-development-runtime-v1/failures.jsonl",
    "results/abstention/b6-b7-comparable-development-runtime-v1/pairs.jsonl",
    "results/abstention/b6-b7-comparable-development-runtime-v1/preflight.json",
    "src/abstention/comparison_contracts.py",
    "src/abstention/comparison_runtime.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_b6_b7_comparison_prerequisite.py",
)
STEP94_EVALUATION_AUTHORIZED_DRIFT = (
    "configs/abstention/b7_evaluation_v1.json",
    "data/abstention/b7-evaluation-development-v1/manifest.json",
    "data/abstention/b7-evaluation-development-v1/reference/expected.jsonl",
    "data/abstention/b7-evaluation-development-v1/reference/review.json",
    "docs/implementation-progress.md",
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
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_b7_evaluation.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_b7_evaluation.py",
)
STEP101_AUTHORIZED_DRIFT = (
    "configs/evaluation/frozen_comparison_v1.json",
    "configs/evaluation/frozen_prompts_v1.json",
    "data/evaluation/frozen-comparison-v1/manifest.json",
    "docs/implementation-progress.md",
    "results/evaluation/frozen-comparison-v1/checks.json",
    "results/evaluation/frozen-comparison-v1/findings.md",
    "results/evaluation/frozen-comparison-v1/manifest.json",
    "results/evaluation/frozen-comparison-v1/run.json",
    "src/evaluation/comparison_freeze.py",
    "src/evaluation/comparison_freeze_contracts.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_comparison_freeze.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_comparison_freeze.py",
)
STEP102_AUTHORIZED_DRIFT = (
    "configs/evaluation/frozen_answer_run_v1.json",
    "configs/evaluation/frozen_answer_run_v2.json",
    "configs/evaluation/frozen_preflight_v1.json",
    "configs/evaluation/frozen_run_v1.json",
    "data/evaluation/frozen-preflight-v1/manifest.json",
    "data/evaluation/frozen-preflight-v1/runtime/manifest.json",
    "data/evaluation/frozen-preflight-v1/runtime/transmission-plan.jsonl",
    "docs/implementation-progress.md",
    "results/evaluation/frozen-preflight-v1/approval-request.md",
    "results/evaluation/frozen-preflight-v1/batches.jsonl",
    "results/evaluation/frozen-preflight-v1/checks.json",
    "results/evaluation/frozen-preflight-v1/failures.jsonl",
    "results/evaluation/frozen-preflight-v1/findings.md",
    "results/evaluation/frozen-preflight-v1/manifest.json",
    "results/evaluation/frozen-preflight-v1/preflight.json",
    "results/evaluation/frozen-preflight-v1/run.json",
    "results/evaluation/frozen-preflight-v1/token-estimates.jsonl",
    "results/evaluation/frozen-run-v1/batches/batch_01_extraction_all_sources/checkpoint.json",
    "results/evaluation/frozen-run-v1/batches/batch_01_extraction_all_sources/failures.jsonl",
    "results/evaluation/frozen-run-v1/batches/batch_01_extraction_all_sources/predictions.jsonl",
    "results/evaluation/frozen-run-v1/batches/batch_02_B0_qa/checkpoint.json",
    "results/evaluation/frozen-run-v1/batches/batch_02_B0_qa/failures.jsonl",
    "results/evaluation/frozen-run-v1/batches/batch_02_B0_qa/predictions.jsonl",
    "results/evaluation/frozen-run-v2/batches/batch_02_B0_qa/checkpoint.json",
    "results/evaluation/frozen-run-v2/batches/batch_02_B0_qa/failures.jsonl",
    "results/evaluation/frozen-run-v2/batches/batch_02_B0_qa/predictions.jsonl",
    "results/evaluation/openai-step10.3-interrupted-v1/findings.md",
    "results/evaluation/openai-step10.3-interrupted-v1/manifest.json",
    "src/evaluation/frozen_answer_contracts.py",
    "src/evaluation/frozen_answers.py",
    "src/evaluation/frozen_contexts.py",
    "src/evaluation/frozen_preflight.py",
    "src/evaluation/frozen_preflight_contracts.py",
    "src/evaluation/frozen_run.py",
    "src/evaluation/frozen_run_contracts.py",
    "src/evaluation/openai_client.py",
    "src/evaluation/openai_recovery.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_b7_evaluation.py",
    "tests/integration/test_comparison_freeze.py",
    "tests/integration/test_frozen_answers.py",
    "tests/integration/test_frozen_preflight.py",
    "tests/integration/test_frozen_run.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
    "tests/integration/test_openai_recovery.py",
    "tests/unit/test_frozen_answers.py",
    "tests/unit/test_frozen_preflight.py",
    "tests/unit/test_frozen_run.py",
    "tests/unit/test_openai_client.py",
)
PROTECTED = {
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "Makefile": "6c7f965049ab12d4bb5339ddd2a75b701e318abc424be91a7e5d3c46e1dc7e6f",
    "results/answering/evidence-package-development-v1/manifest.json": "8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213",
    "results/answering/memory-answer-contract-development-v1/manifest.json": "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841",
    "results/answering/memory-answer-quality-development-v1/manifest.json": "ac936819856939f66597c279fc0b852022a650d5455a4c21240ffbb01f0a524f",
    "tests/integration/test_answer_quality_evaluation.py": "3b58c11717875032d16ec65e870cf590664eb89d33ee6f2901e51ae037c0975d",
    "tests/integration/test_memory_answer.py": "aa96dcd889ae1c0133954ca29f2c8c103325f0555d23426da18d29887c47a104",
}


class AnswerabilityIntegrationTests(unittest.TestCase):
    def test_checked_release_and_exact_counts(self):
        verify_answerability_release(repo_root=ROOT)
        manifest = json.loads((RESULT / "manifest.json").read_text())
        self.assertEqual(
            [item["path"] for item in manifest["predecessor_drift"]],
            [
                "tests/integration/test_answer_quality_evaluation.py",
                "tests/integration/test_memory_answer.py",
            ],
        )
        checks = json.loads((RESULT / "checks.json").read_text())
        self.assertEqual(
            checks,
            {
                "abstain_count": 24,
                "accepted_evidence_count": 0,
                "answerable_count": 0,
                "b2_count": 8,
                "b3_count": 8,
                "b4_count": 8,
                "candidate_rejection_count": 295,
                "clarify_count": 0,
                "decision_count": 24,
                "failure_count": 0,
                "generation_allowed_count": 0,
                "no_promoted_claims_count": 24,
                "package_count": 24,
                "provider_request_count": 0,
                "rejected_evidence_count": 491,
                "retrieval_rejection_count": 196,
                "uncalibrated_confidence_count": 24,
            },
        )
        decisions = [json.loads(line) for line in (RESULT / "decisions.jsonl").read_text().splitlines()]
        self.assertTrue(all(item["decision"] == "abstain" for item in decisions))
        self.assertTrue(all(item["primary_reason"] == "no_promoted_claims" for item in decisions))
        self.assertTrue(all(item["confidence"] == {
            "calibration_status": "not_calibrated",
            "null_reason": "step_9_2_not_run",
            "value": None,
        } for item in decisions))

    def test_two_clean_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first) / "release"
            second_path = Path(second) / "release"
            execute_answerability_evaluation(first_path, repo_root=ROOT)
            execute_answerability_evaluation(second_path, repo_root=ROOT)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_path.iterdir()},
                {path.name: path.read_bytes() for path in second_path.iterdir()},
            )

    def test_runtime_read_trap_blocks_prohibited_inputs(self):
        original_bytes = Path.read_bytes
        original_text = Path.read_text
        forbidden = (
            "/gold/", "oracle", "review_queue", "answers.jsonl", "scorecard.json",
            "scaled-v1", ".env",
        )

        def guard_bytes(path, *args, **kwargs):
            normalized = path.as_posix().lower()
            if any(token in normalized for token in forbidden):
                raise AssertionError(f"prohibited read: {normalized}")
            return original_bytes(path, *args, **kwargs)

        def guard_text(path, *args, **kwargs):
            normalized = path.as_posix().lower()
            if any(token in normalized for token in forbidden):
                raise AssertionError(f"prohibited read: {normalized}")
            return original_text(path, *args, **kwargs)

        config_sha = hashlib.sha256((ROOT / "configs/abstention/answerability_v1.json").read_bytes()).hexdigest()
        policy_sha = hashlib.sha256((ROOT / "src/abstention/policy.py").read_bytes()).hexdigest()
        with patch.object(Path, "read_bytes", guard_bytes), patch.object(Path, "read_text", guard_text):
            rows = load_answerability_inputs(ROOT, config_sha256=config_sha, policy_sha256=policy_sha)
        self.assertEqual(len(rows), 24)

    def test_tamper_and_nonempty_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "release"
            shutil.copytree(RESULT, copied)
            payload = copied / "checks.json"
            payload.write_bytes(payload.read_bytes().replace(b'"package_count":24', b'"package_count":25'))
            with self.assertRaises(AnswerabilityEvaluationError):
                verify_answerability_release(copied, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "sentinel").write_text("keep")
            with self.assertRaisesRegex(AnswerabilityEvaluationError, "must be empty"):
                execute_answerability_evaluation(output, repo_root=ROOT)

    def test_nested_package_fields_are_strict(self):
        package = json.loads(
            (ROOT / "results/answering/evidence-package-development-v1/packages.jsonl")
            .read_text()
            .splitlines()[0]
        )
        mutations = (
            ("eligibility", package["eligibility"]),
            ("rejected evidence", package["rejected_evidence"][0]),
        )
        for label, target in mutations:
            with self.subTest(label=label):
                changed = json.loads(json.dumps(package))
                if label == "eligibility":
                    changed["eligibility"]["unexpected"] = True
                else:
                    changed["rejected_evidence"][0]["unexpected"] = True
                with self.assertRaisesRegex(AnswerabilityError, "fields changed"):
                    evidence_package_from_mapping(changed)

    def test_exact_allowlist_and_protected_hashes(self):
        committed = subprocess.run(
            ["git", "diff", "--name-only", STEP91_START, STEP91_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(committed), AUTHORIZED)
        committed_step92 = subprocess.run(
            ["git", "diff", "--name-only", STEP91_COMMIT, STEP92_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            tuple(committed_step92),
            tuple(sorted(set(STEP92_AUTHORIZED_DRIFT).union(STEP92_COMPATIBILITY_DRIFT))),
        )
        committed_step93 = subprocess.run(
            ["git", "diff", "--name-only", STEP92_COMMIT, STEP93_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(committed_step93), STEP93_AUTHORIZED_DRIFT)
        prerequisite_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP93_COMMIT, STEP94_PREREQUISITE_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(prerequisite_committed), STEP94_PREREQUISITE_AUTHORIZED_DRIFT)
        evaluation_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_PREREQUISITE_COMMIT, STEP94_EVALUATION_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(evaluation_committed), STEP94_EVALUATION_AUTHORIZED_DRIFT)
        step101_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_EVALUATION_COMMIT, STEP101_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(step101_committed), STEP101_AUTHORIZED_DRIFT)
        tracked = subprocess.run(
            ["git", "diff", "--name-only", STEP101_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = [
            path for path in subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            if not path.startswith("docs/DEMO_") and not path.startswith("docs/IMPLEMENTATION_") and not path.startswith("docs/THINE_")
        ]
        actual = tuple(sorted(set(tracked).union(untracked)))
        self.assertEqual(actual, STEP102_AUTHORIZED_DRIFT)
        for path, expected in PROTECTED.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)

    def test_runtime_source_has_no_database_provider_or_generation_boundary(self):
        source = "\n".join(
            (ROOT / path).read_text()
            for path in (
                "src/abstention/contracts.py", "src/abstention/input.py",
                "src/abstention/policy.py", "src/abstention/evaluation.py",
            )
        ).lower()
        for forbidden in (
            "psycopg", "import openai", "openai_api_key", "requests.", "urlopen", "render_answer_prompt",
            "answers.jsonl", "scorecard.json",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
