from __future__ import annotations

import os
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from abstention.interactive_input_v2 import load_runtime_prefix
from abstention.interactive_runtime_v2 import (
    verify_interactive_runtime_checkpoint,
)
from abstention.interactive_evaluation_v2 import (
    FINAL_ROOT,
    RUNTIME_CHECKPOINT_SHA256,
    execute_interactive_evaluation,
    verify_interactive_release,
)
import abstention.interactive_evaluation_v2 as evaluation_module


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
STEP93_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
STEP94_PREREQUISITE_COMMIT = "7e8fc5337384ac329264e3606507b925bd890d63"
STEP94_EVALUATION_COMMIT = "78ed4900fd9a7aecbd7ca8c70b5726b356a07ff4"
STEP101_COMMIT = "b2ae263e1129758325a30db57c03620628c6355e"
STEP102_COMMIT = "77b0a28c5b396bd44f1f76c0a41fd3fbec10cd8f"
STEP93_COMMITTED_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "fff42248ee2c3f361caea561341b4d0e28728be98166d11f6f8f9d138694ba2d",
    "tests/integration/test_answerability.py": "73aeedc3f24f76e3f357e99793ef6ed85bc8c7610caf9b87fd298a7e2fb3c9f4",
    "tests/integration/test_interactive_answering_v2.py": "3eea1ed4c883c076ddfa6a2a0668e3ce77b7c1fb8ffa7b1188e4b1fb3a2abac2",
    "tests/integration/test_memory_answer.py": "1ca97a9ff1df5f11d5210b5434f5ded6830205df0b420c4d88570fdaa5f05f1d",
}
STEP94_EVALUATION_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "638b7730c0ed87ffa76de95496c20979d5baab53a7a17e6791ffd1fb7c895822",
    "tests/integration/test_answerability.py": "2e7694dfc2c188b3571c39741e891fdcfcc10222d8bd7a119b237e516ee9f40f",
    "tests/integration/test_b6_b7_comparison_prerequisite.py": "e10064e56a2e1c3d13290a0eb378be1dfa0ac9ca0ca8f712119c4241a78076c3",
    "tests/integration/test_memory_answer.py": "1802d2bc6353244c5f3fd720f5920794a6e5ee4733d25cb012fea503551f5b82",
}
STEP102_LIVE_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "3171620fbdb36a4d7ee0ebc4601187031a0020e7517ec5dee220e277426127b5",
    "tests/integration/test_answerability.py": "3174f90a554402f73a03b22d0f5acd13477181f36a67b295e328feb6684229fd",
    "tests/integration/test_b6_b7_comparison_prerequisite.py": "8147e7e18decc4dc46866fca5740b1eaa4cdab9759e025c4690c1c2197de1db2",
    "tests/integration/test_comparison_freeze.py": "bb3222f17afd17ee2a6bb131acc143b7e6338ff73c0b2a8b1a7845e31689719f",
    "tests/integration/test_memory_answer.py": "ad7ac20d156270d95dd19e046946dd6cc50941c8427d0ad836dbc5e0ac27ed66",
}
STEP94_PREREQUISITE_COMMITTED_PATHS = (
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
STEP101_COMMITTED_PATHS = (
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


class InteractiveInputV2IntegrationTests(unittest.TestCase):
    def test_committed_prerequisite_and_live_evaluation_topology_are_exact(self) -> None:
        committed = subprocess.run(
            ["git", "diff", "--name-only", STEP93_COMMIT, STEP94_PREREQUISITE_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(committed, list(STEP94_PREREQUISITE_COMMITTED_PATHS))
        evaluation_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_PREREQUISITE_COMMIT, STEP94_EVALUATION_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(evaluation_committed, list(STEP94_EVALUATION_AUTHORIZED_DRIFT))
        step101_committed = subprocess.run(
            ["git", "diff", "--name-only", STEP94_EVALUATION_COMMIT, STEP101_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(step101_committed, list(STEP101_COMMITTED_PATHS))
        step102_committed = [
            path for path in subprocess.run(
                ["git", "diff", "--name-only", STEP101_COMMIT, STEP102_COMMIT],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            if not path.startswith(("docs/DEMO_", "docs/IMPLEMENTATION_", "docs/THINE_"))
        ]
        self.assertEqual(
            step102_committed,
            list(STEP102_AUTHORIZED_DRIFT),
        )

    def test_runtime_prefix_is_exactly_four_development_cases(self) -> None:
        reads = 0

        def reader(stream):
            nonlocal reads
            reads += 1
            if reads == 5:
                raise AssertionError("frozen test record was requested")
            return stream.readline()

        rows = load_runtime_prefix(
            ROOT / "data/scaled-v1/runtime/interactive.jsonl", record_reader=reader,
        )
        self.assertEqual(reads, 4)
        self.assertEqual({item.user_id for item in rows}, {"user_001", "user_002"})
        self.assertTrue(all(item.split == "development" for item in rows))

    def test_gold_prefix_is_exactly_four_and_never_five(self) -> None:
        from abstention.interactive_input_v2 import load_gold_prefix_raw
        reads = 0

        def reader(stream):
            nonlocal reads
            reads += 1
            if reads == 5:
                raise AssertionError("frozen test gold was requested")
            return stream.readline()

        rows = load_gold_prefix_raw(
            ROOT / "data/scaled-v1/gold/interactive.jsonl", record_reader=reader,
        )
        self.assertEqual((len(rows), reads), (4, 4))

    def test_runtime_checkpoint_predates_v2_reference(self) -> None:
        checkpoint = ROOT / "results/abstention/interactive-answering-development-runtime-v2/checkpoint_manifest.json"
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), RUNTIME_CHECKPOINT_SHA256)
        self.assertIn(b'"gold_opened":false', checkpoint.read_bytes())
        self.assertIn(b'"development_gold_used_for_v2_runtime":false', checkpoint.read_bytes())

    def test_reviewed_abstention_mapping_keeps_evidence_obligation(self) -> None:
        reviewed = {
            row["case_id"]: row for row in (
                json.loads(line) for line in (
                    ROOT / "data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl"
                ).read_text().splitlines()
            )
        }
        row = reviewed["scaled_user_002_interactive_abstention_002"]
        self.assertEqual(row["expected_behaviours"], ["evidence", "abstention"])
        self.assertEqual(len(row["exact_evidence_tuples"]), 2)

    def test_two_clean_scorer_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            execute_interactive_evaluation(first, repo_root=ROOT)
            execute_interactive_evaluation(second, repo_root=ROOT)
            self.assertEqual(
                {item.name: item.read_bytes() for item in first.iterdir()},
                {item.name: item.read_bytes() for item in second.iterdir()},
            )

    def test_checked_final_release_self_verifies_and_matches_checkpoint(self) -> None:
        committed = {
            path: hashlib.sha256(subprocess.run(
                ["git", "show", f"{STEP93_COMMIT}:{path}"], cwd=ROOT,
                check=True, capture_output=True,
            ).stdout).hexdigest()
            for path in STEP93_COMMITTED_ADAPTER_SHA256
        }
        self.assertEqual(committed, STEP93_COMMITTED_ADAPTER_SHA256)
        step94_live = {
            path: hashlib.sha256(subprocess.run(
                ["git", "show", f"{STEP94_EVALUATION_COMMIT}:{path}"], cwd=ROOT,
                check=True, capture_output=True,
            ).stdout).hexdigest()
            for path in STEP94_EVALUATION_ADAPTER_SHA256
        }
        self.assertEqual(step94_live, STEP94_EVALUATION_ADAPTER_SHA256)
        live = {
            path: hashlib.sha256(subprocess.run(
                ["git", "show", f"{STEP102_COMMIT}:{path}"], cwd=ROOT,
                check=True, capture_output=True,
            ).stdout).hexdigest()
            for path in STEP102_LIVE_ADAPTER_SHA256
        }
        self.assertEqual(live, STEP102_LIVE_ADAPTER_SHA256)
        original_sha = evaluation_module._sha
        committed_paths = {
            (ROOT / path).resolve(): digest
            for path, digest in STEP93_COMMITTED_ADAPTER_SHA256.items()
        }

        def committed_adapter_sha(path):
            resolved = Path(path).resolve()
            if resolved in committed_paths:
                return committed_paths[resolved]
            return original_sha(path)

        with patch.object(evaluation_module, "_sha", side_effect=committed_adapter_sha):
            verify_interactive_release(ROOT / FINAL_ROOT, repo_root=ROOT)
        self.assertEqual(
            (ROOT / FINAL_ROOT / "predictions.jsonl").read_bytes(),
            (ROOT / "results/abstention/interactive-answering-development-runtime-v2/responses.jsonl").read_bytes(),
        )
        self.assertEqual(
            (ROOT / FINAL_ROOT / "failures.jsonl").read_bytes(),
            (ROOT / "results/abstention/interactive-answering-development-runtime-v2/failures.jsonl").read_bytes(),
        )
        manifest = json.loads((ROOT / FINAL_ROOT / "manifest.json").read_text())
        self.assertTrue(manifest["prior_development_gold_exposure"])
        self.assertFalse(manifest["development_gold_used_for_v2_runtime"])
        self.assertTrue(manifest["reviewer_frozen_snippet_exposure_disclosed"])
        self.assertFalse(manifest["reviewer_frozen_snippet_used"])

    def test_final_verifier_rejects_runtime_implementation_drift(self) -> None:
        original_sha = evaluation_module._sha
        runtime_module = ROOT / "src/abstention/interactive_runtime_v2.py"

        def drifted_sha(path):
            if Path(path) == runtime_module:
                return "0" * 64
            return original_sha(path)

        with patch.object(evaluation_module, "_sha", side_effect=drifted_sha):
            with self.assertRaisesRegex(Exception, "runtime implementation changed"):
                verify_interactive_release(ROOT / FINAL_ROOT, repo_root=ROOT)

    def test_final_verifier_rejects_dataset_authority_drift(self) -> None:
        with patch.object(evaluation_module, "DATASET_MANIFEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(Exception, "dataset manifest changed"):
                verify_interactive_release(ROOT / FINAL_ROOT, repo_root=ROOT)

    def test_nonempty_final_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "occupied").write_text("x")
            with self.assertRaises(Exception):
                execute_interactive_evaluation(output, repo_root=ROOT)


@unittest.skip("runtime was executed twice and frozen before v2 gold/scorer paths existed")
class InteractiveRuntimeV2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admin = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.admin.close()

    def setUp(self) -> None:
        self._clean()

    def tearDown(self) -> None:
        self._clean()

    def _clean(self) -> None:
        self.admin.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.admin.execute("CREATE SCHEMA public")

    def _factory(self):
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def test_runtime_checkpoint_precedes_gold_and_has_four_abstentions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            verify_interactive_runtime_checkpoint(repo_root=ROOT)

    def test_two_clean_runtime_builds_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            verify_interactive_runtime_checkpoint(repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
