from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_preflight import (
    DATA_ROOT,
    RESULT_ROOT,
    RUNTIME_INPUTS,
    build_frozen_preflight,
    verify_frozen_preflight,
)
from evaluation.frozen_preflight_contracts import FrozenPreflightError


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_NEW = (
    "configs/evaluation/frozen_preflight_v1.json",
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
    "src/evaluation/frozen_preflight.py",
    "src/evaluation/frozen_preflight_contracts.py",
    "tests/integration/test_answer_quality_evaluation.py",
    "tests/integration/test_answerability.py",
    "tests/integration/test_b6_b7_comparison_prerequisite.py",
    "tests/integration/test_b7_evaluation.py",
    "tests/integration/test_comparison_freeze.py",
    "tests/integration/test_frozen_preflight.py",
    "tests/integration/test_interactive_answering_v2.py",
    "tests/integration/test_memory_answer.py",
    "tests/unit/test_frozen_preflight.py",
)


class FrozenPreflightIntegrationTests(unittest.TestCase):
    def test_two_builds_are_byte_identical_and_deep_verify(self):
        snapshots = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                build_frozen_preflight(repo_root=ROOT, data_root=base / "data", result_root=base / "result")
                verify_frozen_preflight(repo_root=ROOT, data_root=base / "data", result_root=base / "result")
                snapshots.append({
                    path.relative_to(base).as_posix(): path.read_bytes()
                    for path in sorted(base.rglob("*")) if path.is_file()
                })
        self.assertEqual(snapshots[0], snapshots[1])

    def test_runtime_only_read_trap_and_no_provider_or_secret_path(self):
        opened: list[Path] = []
        allowed_runtime = {(ROOT / path).resolve() for path in RUNTIME_INPUTS}

        def reader(path: Path) -> bytes:
            resolved = path.resolve()
            opened.append(resolved)
            lowered = resolved.as_posix().lower()
            if "data/scaled-v1/runtime" in lowered and resolved not in allowed_runtime:
                raise AssertionError(f"unexpected runtime path: {resolved}")
            for token in ("/gold/", "/oracle/", "/review_queues/", "/predictions/", "/.env"):
                if token in lowered:
                    raise AssertionError(f"prohibited path opened: {resolved}")
            return resolved.read_bytes()

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            build_frozen_preflight(repo_root=ROOT, data_root=base / "data", result_root=base / "result", reader=reader)
        self.assertEqual({path for path in opened if path in allowed_runtime}, allowed_runtime)

    def test_nonempty_output_and_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            data.mkdir()
            (data / "existing").write_text("x")
            with self.assertRaises(FrozenPreflightError):
                build_frozen_preflight(repo_root=ROOT, data_root=data, result_root=base / "result")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            build_frozen_preflight(repo_root=ROOT, data_root=base / "data", result_root=base / "result")
            target = base / "result/checks.json"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaises(FrozenPreflightError):
                verify_frozen_preflight(repo_root=ROOT, data_root=base / "data", result_root=base / "result")

    def test_checked_release_verifies_and_has_no_failures(self):
        hashes = verify_frozen_preflight(repo_root=ROOT)
        self.assertEqual(len(hashes), 12)
        self.assertEqual((ROOT / RESULT_ROOT / "failures.jsonl").read_bytes(), b"")

    def test_exact_preapproval_path_allowlist(self):
        import subprocess
        tracked = subprocess.run(
            ["git", "diff", "--name-only", "b2ae263e1129758325a30db57c03620628c6355e"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(tuple(sorted(set(tracked).union(untracked))), ALLOWED_NEW)


if __name__ == "__main__":
    unittest.main()
