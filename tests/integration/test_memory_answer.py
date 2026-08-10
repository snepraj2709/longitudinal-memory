from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from answering.answer_contracts import MemoryAnswerFailure, MemoryAnswerError
from answering.answer_evaluation import (
    RESULT_ROOT,
    MemoryAnswerEvaluationError,
    execute_memory_answer_evaluation,
    verify_memory_answer_release,
)
from answering.answer_input import (
    STEP81_CHECKS_SHA256,
    STEP81_DATASET_SHA256,
    STEP81_MANIFEST_SHA256,
    STEP81_PACKAGES,
    STEP81_PACKAGES_SHA256,
    load_answer_package_views,
)
from answering.memory_answer import ABSTENTION_REASONS, ABSTENTION_TEXT


ROOT = Path(__file__).resolve().parents[2]
CHECKED = ROOT / RESULT_ROOT
START = "8fec075d754dff7f12821947919d5c01f867d949"
STEP82_COMMIT = "9f7625455abafb85b513b8d30a53d793580160ce"
STEP83_COMMIT = "a18501a27708c259cccce8bf87948962e672bd41"
STEP91_COMMIT = "35d0c64431f19d4243b72af712dadeb8d522128f"
STEP92_COMMIT = "d0932a7994153745285c1f4e3d75c36ffbbaf06a"
STEP93_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
STEP82_COMMITTED_PATHS = (
    "configs/answering/memory_answer_v1.json",
    "data/answering/memory-answer-contract-development-v1/manifest.json",
    "docs/implementation-progress.md",
    "results/answering/memory-answer-contract-development-v1/answers.jsonl",
    "results/answering/memory-answer-contract-development-v1/checks.json",
    "results/answering/memory-answer-contract-development-v1/failures.jsonl",
    "results/answering/memory-answer-contract-development-v1/findings.md",
    "results/answering/memory-answer-contract-development-v1/manifest.json",
    "results/answering/memory-answer-contract-development-v1/run.json",
    "src/answering/answer_contracts.py",
    "src/answering/answer_evaluation.py",
    "src/answering/answer_input.py",
    "src/answering/memory_answer.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_memory_answer.py",
)
STEP83_AUTHORIZED_DRIFT = (
    "configs/answering/comparable_answer_run_v1.json",
    "data/answering/memory-answer-quality-development-v1/manifest.json",
    "docs/implementation-progress.md",
    "results/answering/memory-answer-quality-development-runtime-v1/checkpoint_manifest.json",
    "results/answering/memory-answer-quality-development-runtime-v1/failures.jsonl",
    "results/answering/memory-answer-quality-development-runtime-v1/predictions.jsonl",
    "results/answering/memory-answer-quality-development-runtime-v1/preflight.json",
    "results/answering/memory-answer-quality-development-v1/checks.json",
    "results/answering/memory-answer-quality-development-v1/failures.jsonl",
    "results/answering/memory-answer-quality-development-v1/findings.md",
    "results/answering/memory-answer-quality-development-v1/manifest.json",
    "results/answering/memory-answer-quality-development-v1/per-case.jsonl",
    "results/answering/memory-answer-quality-development-v1/run.json",
    "results/answering/memory-answer-quality-development-v1/scorecard.json",
    "src/answering/answer_quality_evaluation.py",
    "src/answering/answer_run_contracts.py",
    "src/answering/comparable_answer_run.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_answer_quality_evaluation.py",
)
STEP91_AUTHORIZED_DRIFT = (
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
    "tests/unit/test_answerability.py",
)
STEP91_COMPATIBILITY_DRIFT = (
    "tests/integration/test_memory_answer.py",
)
STEP92_PRIMARY_DRIFT = (
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


class MemoryAnswerIntegrationTests(unittest.TestCase):
    def _execute(self, output: Path):
        return execute_memory_answer_evaluation(output, repo_root=ROOT)

    def test_checked_release_self_verifies(self) -> None:
        verify_memory_answer_release(CHECKED, repo_root=ROOT)

    def test_release_has_exact_24_structural_abstentions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            checks = self._execute(output)
            answers = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
        self.assertEqual((checks.package_count, checks.answer_count, checks.failure_count), (24, 24, 0))
        self.assertEqual((checks.abstained_count, checks.no_promoted_claims_count), (24, 24))
        self.assertTrue(all(item["status"] == "abstained" for item in answers))
        self.assertTrue(all(item["answer"] == ABSTENTION_TEXT for item in answers))
        self.assertTrue(all(
            item["abstention_reason"] == ABSTENTION_REASONS["no_promoted_claims"]
            for item in answers
        ))
        self.assertTrue(all(
            item["confidence"] == 0 and not item["statements"]
            and item["requested_model"] is None and item["resolved_model"] is None
            for item in answers
        ))

    def test_real_release_never_renders_prompt_or_reaches_provider_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "answering.memory_answer.render_answer_prompt",
            side_effect=AssertionError("prompt reached"),
        ):
            checks = self._execute(Path(directory) / "release")
        self.assertEqual(checks.abstained_count, 24)

    def test_two_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self._execute(first)
            self._execute(second)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_nonempty_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "keep").write_text("keep")
            with self.assertRaisesRegex(MemoryAnswerEvaluationError, "must be empty"):
                self._execute(output)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.write_text("not a directory")
            with self.assertRaisesRegex(MemoryAnswerEvaluationError, "must be empty"):
                self._execute(output)

    def test_tampered_output_and_package_authority_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            (output / "answers.jsonl").write_bytes((output / "answers.jsonl").read_bytes() + b" ")
            with self.assertRaisesRegex(MemoryAnswerEvaluationError, "hash changed"):
                verify_memory_answer_release(output, repo_root=ROOT)
        with patch("answering.answer_input.STEP81_PACKAGES_SHA256", "0" * 64):
            with self.assertRaises(MemoryAnswerError):
                load_answer_package_views(ROOT)

    def test_dataset_manifest_binds_every_step81_authority(self) -> None:
        from answering import answer_evaluation

        original_read = answer_evaluation._read_object

        def tampered(path):
            value = dict(original_read(path))
            if Path(path).resolve() == (ROOT / answer_evaluation.DATASET_MANIFEST).resolve():
                value["input_release"] = dict(value["input_release"])
                value["input_release"]["checks_sha256"] = "0" * 64
            return value

        with patch.object(answer_evaluation, "_read_object", tampered):
            with self.assertRaisesRegex(MemoryAnswerEvaluationError, "dataset manifest"):
                answer_evaluation._dataset(ROOT)
        self.assertEqual(STEP81_CHECKS_SHA256, "9e081b65e5bc8c8f59c8fb320a357e14240eedc06fd5febb67ddbcf588d8c766")

    def test_step81_release_is_verified_before_package_bytes_are_read(self) -> None:
        state = {"verifying": False, "verified": False}
        from answering import answer_input

        original_verify = answer_input.verify_evidence_package_release
        original_read_bytes = Path.read_bytes

        def verified(*args, **kwargs):
            state["verifying"] = True
            try:
                result = original_verify(*args, **kwargs)
            finally:
                state["verifying"] = False
            state["verified"] = True
            return result

        def guarded(path):
            if (
                Path(path).resolve() == (ROOT / STEP81_PACKAGES).resolve()
                and not state["verifying"] and not state["verified"]
            ):
                raise AssertionError("packages read before verifier")
            return original_read_bytes(path)

        with patch.object(answer_input, "verify_evidence_package_release", verified), patch.object(
            Path, "read_bytes", guarded
        ):
            self.assertEqual(len(load_answer_package_views(ROOT)), 24)

    def test_runtime_has_no_prohibited_data_provider_or_environment_access(self) -> None:
        forbidden = (
            "/gold/", "relevance.jsonl", "/oracle", "review_queue", "user_003",
            "/.env", "OPENAI_API_KEY",
        )
        original_open = Path.open

        def guarded(path, *args, **kwargs):
            value = Path(path).as_posix()
            if any(token in value for token in forbidden):
                raise AssertionError(f"forbidden read: {value}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "open", guarded):
            self.assertEqual(self._execute(Path(directory) / "release").answer_count, 24)
        source = "".join((ROOT / path).read_text() for path in (
            "src/answering/answer_contracts.py", "src/answering/answer_input.py",
            "src/answering/memory_answer.py", "src/answering/answer_evaluation.py",
        ))
        self.assertNotIn("import openai", source.lower())
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("os.environ", source)

    def test_failure_contract_is_sanitized(self) -> None:
        failure = MemoryAnswerFailure(
            "1" * 64, None, "2" * 64, "query_safe", "B2", "runtime_failure", "runtime"
        )
        self.assertNotIn("quote", json.dumps(failure.__dict__))
        with self.assertRaises(MemoryAnswerError):
            MemoryAnswerFailure(
                "1" * 64, None, "2" * 64, "query_safe", "B2", "raw source", "runtime"
            )

    def test_predecessor_bytes_and_allowlist_are_exact(self) -> None:
        expected = {
            "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
            "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
            "Makefile": "6c7f965049ab12d4bb5339ddd2a75b701e318abc424be91a7e5d3c46e1dc7e6f",
            "src/answering/__init__.py": "582ee7aa9e8eb133a9b0a43ccbb256ef4c020f0b5fd89b42ff266d74fdb9aafa",
            "results/answering/evidence-package-development-v1/manifest.json": STEP81_MANIFEST_SHA256,
            "results/answering/evidence-package-development-v1/packages.jsonl": STEP81_PACKAGES_SHA256,
            "data/answering/evidence-package-development-v1/manifest.json": STEP81_DATASET_SHA256,
        }
        self.assertEqual(
            {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in expected},
            expected,
        )
        import subprocess
        committed = subprocess.run(
            ["git", "diff", "--name-only", START, STEP82_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(committed, list(STEP82_COMMITTED_PATHS))
        step83_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP82_COMMIT, STEP83_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(step83_committed, list(STEP83_AUTHORIZED_DRIFT))
        step91_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP83_COMMIT, STEP91_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            step91_committed,
            sorted(set(STEP91_AUTHORIZED_DRIFT).union(STEP91_COMPATIBILITY_DRIFT)),
        )
        step92_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP91_COMMIT, STEP92_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            step92_committed,
            sorted(set(STEP92_PRIMARY_DRIFT).union(STEP92_COMPATIBILITY_DRIFT)),
        )
        step93_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP92_COMMIT, STEP93_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(step93_committed, list(STEP93_AUTHORIZED_DRIFT))
        tracked = subprocess.run(
            ["git", "diff", "--name-only", STEP93_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            sorted(set(tracked).union(untracked)),
            list(STEP94_PREREQUISITE_AUTHORIZED_DRIFT),
        )


if __name__ == "__main__":
    unittest.main()
