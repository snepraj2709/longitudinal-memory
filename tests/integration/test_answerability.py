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
PROTECTED = {
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "Makefile": "6c7f965049ab12d4bb5339ddd2a75b701e318abc424be91a7e5d3c46e1dc7e6f",
    "results/answering/evidence-package-development-v1/manifest.json": "8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213",
    "results/answering/memory-answer-contract-development-v1/manifest.json": "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841",
    "results/answering/memory-answer-quality-development-v1/manifest.json": "ac936819856939f66597c279fc0b852022a650d5455a4c21240ffbb01f0a524f",
    "tests/integration/test_answer_quality_evaluation.py": "933a7b9378d99957295004bbc81926707e79e4c7cf6e5f68d3f89d8f76421dc4",
    "tests/integration/test_memory_answer.py": "5b68e3cb8037c78e841ce7293498859fc2c1d5397f5fe55b087b1127d3a40818",
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
        tracked = subprocess.run(
            ["git", "diff", "--name-only", "a18501a27708c259cccce8bf87948962e672bd41"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        actual = tuple(sorted(set(tracked).union(untracked)))
        self.assertEqual(actual, AUTHORIZED)
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
