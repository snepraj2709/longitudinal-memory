from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from evaluation.comparison_freeze import (
    DATA_ROOT,
    RESULT_ROOT,
    build_frozen_comparison,
    verify_frozen_comparison,
)
from evaluation.comparison_freeze_contracts import ComparisonFreezeError


ROOT = Path(__file__).resolve().parents[2]
SCALED_MANIFEST = (ROOT / "data/scaled-v1/manifest.json").resolve()
STEP101_START = "78ed4900fd9a7aecbd7ca8c70b5726b356a07ff4"
STEP101_COMMIT = "b2ae263e1129758325a30db57c03620628c6355e"
PROTECTED = {
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "data/scaled-v1/manifest.json": "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3",
    "results/abstention/b7-evaluation-development-v1/manifest.json": "e8aed9be4455a1dd8b33f390928bcec537e5c69e752a41a1a23265acb5e12fbd",
}
STEP101_COMMITTED = (
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
ALLOWED_NEW = (
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


class ComparisonFreezeIntegrationTests(unittest.TestCase):
    def test_committed_build_verifies_and_counts_are_exact(self):
        hashes = verify_frozen_comparison(repo_root=ROOT)
        self.assertEqual(set(hashes), {
            "data_manifest_sha256", "checks_sha256", "run_sha256",
            "findings_sha256", "result_manifest_sha256",
        })
        import json
        checks = json.loads((ROOT / RESULT_ROOT / "checks.json").read_text())
        self.assertEqual(checks["manifest_file_count"], 27)
        self.assertEqual(checks["baseline_count"], 8)
        self.assertEqual(checks["task_count"], 3)
        self.assertEqual(checks["prerequisite_gap_count"], 8)
        self.assertEqual((checks["prediction_count"], checks["score_count"], checks["provider_request_count"]), (0, 0, 0))
        self.assertFalse(checks["referenced_files_opened"])

    def test_manifest_only_read_trap_and_byte_identical_replay(self):
        opened: list[Path] = []

        def reader(path: Path) -> bytes:
            resolved = path.resolve()
            opened.append(resolved)
            if "data/scaled-v1" in resolved.as_posix() and resolved != SCALED_MANIFEST:
                raise AssertionError(f"scaled referenced file opened: {resolved}")
            lowered = resolved.as_posix().lower()
            for token in ("/.env", "/oracle/", "/gold/", "/review_queues/", "/predictions/"):
                if token in lowered and "data/scaled-v1/manifest.json" not in lowered:
                    raise AssertionError(f"prohibited path opened: {resolved}")
            return resolved.read_bytes()

        snapshots: list[dict[str, bytes]] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                data = base / "data"
                result = base / "result"
                build_frozen_comparison(repo_root=ROOT, data_root=data, result_root=result, reader=reader)
                verify_frozen_comparison(repo_root=ROOT, data_root=data, result_root=result, reader=reader)
                snapshots.append({
                    "data/manifest.json": (data / "manifest.json").read_bytes(),
                    **{f"result/{path.name}": path.read_bytes() for path in sorted(result.iterdir())},
                })
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertIn(SCALED_MANIFEST, opened)
        self.assertFalse(any(path != SCALED_MANIFEST and "data/scaled-v1" in path.as_posix() for path in opened))

    def test_nonempty_output_and_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            result = base / "result"
            data.mkdir()
            (data / "existing").write_text("x")
            with self.assertRaises(ComparisonFreezeError):
                build_frozen_comparison(repo_root=ROOT, data_root=data, result_root=result)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            result = base / "result"
            build_frozen_comparison(repo_root=ROOT, data_root=data, result_root=result)
            (result / "checks.json").write_bytes((result / "checks.json").read_bytes() + b" ")
            with self.assertRaises(ComparisonFreezeError):
                verify_frozen_comparison(repo_root=ROOT, data_root=data, result_root=result)

    def test_protected_hashes_and_exact_live_additions(self):
        actual = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in PROTECTED
        }
        self.assertEqual(actual, PROTECTED)
        import subprocess
        committed = subprocess.run(
            ["git", "diff", "--name-only", STEP101_START, STEP101_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(committed), STEP101_COMMITTED)
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
        self.assertEqual(tuple(sorted(set(tracked).union(untracked))), ALLOWED_NEW)


if __name__ == "__main__":
    unittest.main()
