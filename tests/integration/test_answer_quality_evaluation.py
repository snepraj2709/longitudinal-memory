from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from answering.answer_quality_evaluation import (
    RESULT_ROOT,
    execute_answer_quality_evaluation,
    score_frozen_answers,
    verify_answer_quality_release,
)
from answering.answer_run_contracts import AnswerRunError
from answering.comparable_answer_run import (
    RUNTIME_ROOT,
    STEP82_ANSWERS_SHA256,
    freeze_answer_predictions,
    verify_answer_runtime_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
START = "9f7625455abafb85b513b8d30a53d793580160ce"
STEP83_COMMIT = "a18501a27708c259cccce8bf87948962e672bd41"
STEP91_COMMIT = "35d0c64431f19d4243b72af712dadeb8d522128f"
STEP92_COMMIT = "d0932a7994153745285c1f4e3d75c36ffbbaf06a"
STEP93_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
STEP94_PREREQUISITE_COMMIT = "7e8fc5337384ac329264e3606507b925bd890d63"
STEP94_EVALUATION_COMMIT = "78ed4900fd9a7aecbd7ca8c70b5726b356a07ff4"
STEP101_COMMIT = "b2ae263e1129758325a30db57c03620628c6355e"
STEP102_PREDECESSOR_COMMIT = "1c6332d9865358d1af7045d015beab0339418191"
STEP102_COMMIT = "77b0a28c5b396bd44f1f76c0a41fd3fbec10cd8f"
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


class AnswerQualityIntegrationTests(unittest.TestCase):
    def _execute(self, base: Path):
        runtime = base / "runtime"
        final = base / "final"
        checks = execute_answer_quality_evaluation(runtime, final, repo_root=ROOT)
        return runtime, final, checks

    def test_exact_release_accounting_and_null_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, final, checks = self._execute(Path(directory))
            self.assertEqual((checks.per_case_count, checks.scorecard_row_count), (24, 35))
            self.assertEqual((checks.B2_case_count, checks.B3_case_count, checks.B4_case_count), (8, 8, 8))
            self.assertEqual((checks.B5_case_count, checks.B6_case_count), (0, 0))
            self.assertEqual(checks.abstained_count, 24)
            self.assertEqual(checks.non_null_metric_count, 0)
            self.assertEqual(checks.provider_request_count, 0)
            scorecard = json.loads((final / "scorecard.json").read_text())
            self.assertEqual(len(scorecard["rows"]), 35)
            self.assertTrue(all(row["value"]["value"] is None for row in scorecard["rows"]))
            self.assertEqual(hashlib.sha256((runtime / "predictions.jsonl").read_bytes()).hexdigest(), STEP82_ANSWERS_SHA256)

    def test_predictions_are_exact_step82_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _, _ = self._execute(Path(directory))
            self.assertEqual(
                (runtime / "predictions.jsonl").read_bytes(),
                (ROOT / "results/answering/memory-answer-contract-development-v1/answers.jsonl").read_bytes(),
            )
            self.assertEqual((runtime / "failures.jsonl").read_bytes(), b"")

    def test_step81_and_step82_public_verifiers_run_before_answer_read(self) -> None:
        from answering import comparable_answer_run as module

        order = []
        original_step81 = module.verify_evidence_package_release
        original_step82 = module.verify_memory_answer_release

        def step81(*args, **kwargs):
            order.append("step81")
            return original_step81(*args, **kwargs)

        def step82(*args, **kwargs):
            order.append("step82")
            return original_step82(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "verify_evidence_package_release", step81
        ), patch.object(module, "verify_memory_answer_release", step82):
            freeze_answer_predictions(Path(directory) / "runtime", repo_root=ROOT)
        self.assertEqual(order[:2], ["step81", "step82"])

    def test_checkpoint_is_verified_before_scorer_reads_predictions(self) -> None:
        from answering import answer_quality_evaluation as module

        state = {"verified": False}
        original_verify = module.verify_answer_runtime_checkpoint
        original_answers = module._runtime_answers

        def verified(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            state["verified"] = True
            return result

        def guarded(*args, **kwargs):
            if not state["verified"]:
                raise AssertionError("scorer read predictions before checkpoint verification")
            return original_answers(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            freeze_answer_predictions(runtime, repo_root=ROOT)
            with patch.object(module, "verify_answer_runtime_checkpoint", verified), patch.object(
                module, "_runtime_answers", guarded
            ):
                score_frozen_answers(Path(directory) / "final", runtime_dir=runtime, repo_root=ROOT)

    def test_two_runtime_and_final_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_runtime, first_final, _ = self._execute(Path(directory) / "first")
            second_runtime, second_final, _ = self._execute(Path(directory) / "second")
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_runtime.iterdir()},
                {path.name: path.read_bytes() for path in second_runtime.iterdir()},
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_final.iterdir()},
                {path.name: path.read_bytes() for path in second_final.iterdir()},
            )

    def test_runtime_has_no_prompt_environment_network_or_prohibited_read(self) -> None:
        forbidden = ("/gold/", "/oracle", "review_queue", "user_003", "/.env")
        original_open = Path.open

        def guarded(path, *args, **kwargs):
            rendered = Path(path).as_posix().lower()
            if any(token in rendered for token in forbidden):
                raise AssertionError(f"prohibited read: {rendered}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "open", guarded), patch(
            "answering.memory_answer.render_answer_prompt", side_effect=AssertionError("prompt rendered")
        ), patch.object(os, "getenv", side_effect=AssertionError("environment read")), patch.object(
            socket, "socket", side_effect=AssertionError("network opened")
        ):
            self._execute(Path(directory))

    def test_changed_checkpoint_metric_and_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, final, _ = self._execute(Path(directory))
            checkpoint = json.loads((runtime / "checkpoint_manifest.json").read_text())
            checkpoint["prediction_count"] = 23
            (runtime / "checkpoint_manifest.json").write_text(json.dumps(checkpoint))
            with self.assertRaises(AnswerRunError):
                verify_answer_runtime_checkpoint(runtime, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            runtime, final, _ = self._execute(Path(directory))
            scorecard = json.loads((final / "scorecard.json").read_text())
            scorecard["rows"][0]["value"]["null_reason"] = "wrong_reason"
            (final / "scorecard.json").write_text(json.dumps(scorecard))
            with self.assertRaises(AnswerRunError):
                verify_answer_quality_release(final, runtime_dir=runtime, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            runtime, final, _ = self._execute(Path(directory))
            (final / "findings.md").write_bytes((final / "findings.md").read_bytes() + b"changed")
            with self.assertRaises(AnswerRunError):
                verify_answer_quality_release(final, runtime_dir=runtime, repo_root=ROOT)

    def test_changed_input_authority_prompt_model_settings_and_baseline_fail(self) -> None:
        from answering import comparable_answer_run as module

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "STEP82_ANSWERS_SHA256", "0" * 64
        ):
            with self.assertRaises(AnswerRunError):
                freeze_answer_predictions(Path(directory) / "runtime", repo_root=ROOT)
        original = json.loads((ROOT / "configs/answering/comparable_answer_run_v1.json").read_text())
        for field, value in (
            ("prompt_sha256", "0" * 64),
            ("requested_model", "wrong-model"),
            ("available_baselines", ["B2", "B3", "B4", "B5"]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                changed = dict(original)
                changed[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(AnswerRunError):
                    module.load_comparable_answer_run_config(path, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            changed = dict(original)
            changed["generation_settings"] = dict(changed["generation_settings"])
            changed["generation_settings"]["temperature"] = 0.2
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(AnswerRunError):
                module.load_comparable_answer_run_config(path, repo_root=ROOT)

    def test_provider_eligible_package_and_b5_prediction_are_rejected(self) -> None:
        from dataclasses import replace
        from answering import comparable_answer_run as module

        views = load_views = module.load_answer_package_views(ROOT)
        poisoned = (replace(views[0], answer_allowed=True, structural_blockers=()), *views[1:])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "load_answer_package_views", return_value=poisoned
        ):
            with self.assertRaises(AnswerRunError):
                freeze_answer_predictions(Path(directory) / "runtime", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            freeze_answer_predictions(runtime, repo_root=ROOT)
            raw = (runtime / "predictions.jsonl").read_bytes().replace(
                b'"baseline_id":"B2"', b'"baseline_id":"B5"', 1
            )
            (runtime / "predictions.jsonl").write_bytes(raw)
            with self.assertRaises(AnswerRunError):
                verify_answer_runtime_checkpoint(runtime, repo_root=ROOT)

    def test_nonempty_runtime_and_final_directories_are_refused(self) -> None:
        for target in ("runtime", "final"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                runtime = base / "runtime"
                final = base / "final"
                if target == "runtime":
                    runtime.mkdir()
                    (runtime / "keep").write_text("keep")
                    with self.assertRaisesRegex(AnswerRunError, "must be empty"):
                        freeze_answer_predictions(runtime, repo_root=ROOT)
                else:
                    freeze_answer_predictions(runtime, repo_root=ROOT)
                    final.mkdir()
                    (final / "keep").write_text("keep")
                    with self.assertRaisesRegex(AnswerRunError, "must be empty"):
                        score_frozen_answers(final, runtime_dir=runtime, repo_root=ROOT)

    def test_predecessor_hashes_and_tracked_diff_are_exact(self) -> None:
        expected = {
            "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
            "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
            "Makefile": "6c7f965049ab12d4bb5339ddd2a75b701e318abc424be91a7e5d3c46e1dc7e6f",
            "src/answering/__init__.py": "582ee7aa9e8eb133a9b0a43ccbb256ef4c020f0b5fd89b42ff266d74fdb9aafa",
            "results/answering/evidence-package-development-v1/manifest.json": "8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213",
            "results/answering/memory-answer-contract-development-v1/manifest.json": "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841",
            "results/answering/memory-answer-contract-development-v1/answers.jsonl": STEP82_ANSWERS_SHA256,
            "tests/integration/test_memory_answer.py": "aa96dcd889ae1c0133954ca29f2c8c103325f0555d23426da18d29887c47a104",
        }
        predecessor_hashes = {
            path: hashlib.sha256(subprocess.run(
                ["git", "show", f"{STEP102_PREDECESSOR_COMMIT}:{path}"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout).hexdigest()
            for path in expected
        }
        self.assertEqual(predecessor_hashes, expected)
        committed = subprocess.run(
            ["git", "diff", "--name-only", START, STEP83_COMMIT], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(committed, list(STEP83_AUTHORIZED_DRIFT))
        step91_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP83_COMMIT, STEP91_COMMIT], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            step91_committed,
            sorted(set(STEP91_AUTHORIZED_DRIFT).union(STEP91_COMPATIBILITY_DRIFT)),
        )
        step92_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP91_COMMIT, STEP92_COMMIT], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            step92_committed,
            sorted(set(STEP92_PRIMARY_DRIFT).union(STEP92_COMPATIBILITY_DRIFT)),
        )
        step93_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP92_COMMIT, STEP93_COMMIT], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(step93_committed, list(STEP93_AUTHORIZED_DRIFT))
        prerequisite_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP93_COMMIT, STEP94_PREREQUISITE_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(prerequisite_committed, list(STEP94_PREREQUISITE_AUTHORIZED_DRIFT))
        evaluation_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_PREREQUISITE_COMMIT, STEP94_EVALUATION_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(evaluation_committed, list(STEP94_EVALUATION_AUTHORIZED_DRIFT))
        step101_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_EVALUATION_COMMIT, STEP101_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(step101_committed, list(STEP101_AUTHORIZED_DRIFT))
        step102_committed = [
            path for path in subprocess.run(
            ["git", "diff", "--name-only", STEP101_COMMIT, STEP102_COMMIT], cwd=ROOT,
            check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            if not path.startswith(("docs/DEMO_", "docs/IMPLEMENTATION_", "docs/THINE_"))
        ]
        self.assertEqual(
            step102_committed,
            list(STEP102_AUTHORIZED_DRIFT),
        )


if __name__ == "__main__":
    unittest.main()
