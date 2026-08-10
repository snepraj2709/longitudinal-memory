from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from abstention.comparison_runtime import (
    ARTIFACTS,
    RESULT_ROOT,
    ComparisonPrerequisiteError,
    execute_comparison_runtime,
    verify_comparison_runtime,
)
from abstention.comparison_contracts import (
    canonical_json_bytes,
    pair_from_mapping,
    prediction_from_mapping,
    stable_sha256,
)
import abstention.comparison_runtime as comparison_runtime


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
STEP93_COMMIT = "3c45309de43a35b0c7b7b588077f094be2b57934"
STEP94_PREREQUISITE_COMMIT = "7e8fc5337384ac329264e3606507b925bd890d63"
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


class B6B7ComparisonPrerequisiteStaticIntegrationTests(unittest.TestCase):
    def test_committed_prerequisite_and_live_evaluation_topology_are_exact(self) -> None:
        committed = subprocess.run(
            ["git", "diff", "--name-only", STEP93_COMMIT, STEP94_PREREQUISITE_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(committed, list(STEP94_PREREQUISITE_COMMITTED_PATHS))
        tracked = subprocess.run(
            ["git", "diff", "--name-only", STEP94_PREREQUISITE_COMMIT],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(
            sorted(set(tracked).union(untracked)),
            list(STEP94_EVALUATION_AUTHORIZED_DRIFT),
        )

    def test_checked_runtime_self_verifies_when_present(self) -> None:
        checked = ROOT / RESULT_ROOT
        if not checked.exists():
            self.skipTest("runtime checkpoint has not been frozen yet")
        checkpoint = verify_comparison_runtime(checked, repo_root=ROOT)
        self.assertEqual((checkpoint.case_count, checkpoint.pair_count), (4, 4))
        self.assertEqual(checkpoint.provider_eligible_case_count, 0)
        self.assertFalse(checkpoint.step9_4_evaluated)

    def test_runtime_manifest_discloses_nonblind_exposures_without_use(self) -> None:
        manifest = json.loads((
            ROOT / "data/abstention/b6-b7-comparable-development-v1/runtime/manifest.json"
        ).read_bytes())
        self.assertTrue(manifest["implementer_pilot_answer_reference_exposure"])
        self.assertFalse(manifest["implementer_pilot_answer_reference_used"])
        self.assertTrue(manifest["reviewer_frozen_snippet_exposure"])
        self.assertFalse(manifest["reviewer_frozen_snippet_used"])
        self.assertFalse(manifest["root_contract_deriver_prohibited_exposure"])
        self.assertFalse(manifest["reference_available"])
        self.assertFalse(manifest["step9_4_scorer_available"])

    def test_step93_shared_artifacts_remain_exact(self) -> None:
        expected = {
            "plans.jsonl": "02e2afd4001d96fa421c57b2ad5fd329dda1d581e90124cc0de768b496e5f94a",
            "retrieval-results.jsonl": "5b4f5199cff3f73a83deab9f394aff5340f307a7aac7581c5d7a524424e2f203",
            "packages.jsonl": "ccdf6a13fbba02125442cae803c47298e73be88407267c4c32d2367562beefe4",
            "decisions.jsonl": "be32a1d12084a3bba57fc48d76ae5f15e43a70ddc50aa6ff176fef4dff899e0a",
        }
        root = ROOT / "results/abstention/interactive-answering-development-runtime-v2"
        self.assertEqual(
            {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected},
            expected,
        )

    def test_nonempty_output_is_refused_without_opening_predecessors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            output.mkdir()
            (output / "occupied").write_text("x")
            with self.assertRaisesRegex(ComparisonPrerequisiteError, "must be empty"):
                execute_comparison_runtime(lambda: None, output, repo_root=ROOT)

    def test_deep_verifier_rejects_tampered_prediction(self) -> None:
        checked = ROOT / RESULT_ROOT
        if not checked.exists():
            self.skipTest("runtime checkpoint has not been frozen yet")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "runtime"
            copied.mkdir()
            for source in checked.iterdir():
                (copied / source.name).write_bytes(source.read_bytes())
            target = copied / "b6-predictions.jsonl"
            target.write_bytes(target.read_bytes().replace(b'"B6"', b'"B7"', 1))
            with self.assertRaises(ComparisonPrerequisiteError):
                verify_comparison_runtime(copied, repo_root=ROOT)

    def test_deep_verifier_rejects_coordinated_prediction_rehash(self) -> None:
        checked = ROOT / RESULT_ROOT
        if not checked.exists():
            self.skipTest("runtime checkpoint has not been frozen yet")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "runtime"
            copied.mkdir()
            for source in checked.iterdir():
                (copied / source.name).write_bytes(source.read_bytes())
            predictions_by_baseline = {}
            for name in ("b6-predictions.jsonl", "b7-predictions.jsonl"):
                rows = [json.loads(line) for line in (copied / name).read_text().splitlines()]
                for row in rows:
                    row["response_text_sha256"] = "0" * 64
                    row["prediction_id"] = stable_sha256({
                        key: value for key, value in row.items() if key != "prediction_id"
                    })
                predictions = tuple(prediction_from_mapping(row) for row in rows)
                predictions_by_baseline[name] = {
                    item.case_id: item for item in predictions
                }
                (copied / name).write_bytes(b"".join(
                    canonical_json_bytes(item) for item in predictions
                ))
            pair_rows = [json.loads(line) for line in (copied / "pairs.jsonl").read_text().splitlines()]
            for row in pair_rows:
                b6 = predictions_by_baseline["b6-predictions.jsonl"][row["case_id"]]
                b7 = predictions_by_baseline["b7-predictions.jsonl"][row["case_id"]]
                row.update({
                    "b6_prediction_id": b6.prediction_id,
                    "b6_prediction_sha256": hashlib.sha256(canonical_json_bytes(b6)).hexdigest(),
                    "b7_prediction_id": b7.prediction_id,
                    "b7_prediction_sha256": hashlib.sha256(canonical_json_bytes(b7)).hexdigest(),
                })
                row["pair_id"] = stable_sha256({
                    key: value for key, value in row.items() if key != "pair_id"
                })
            pairs = tuple(pair_from_mapping(row) for row in pair_rows)
            (copied / "pairs.jsonl").write_bytes(b"".join(
                canonical_json_bytes(item) for item in pairs
            ))
            checkpoint_path = copied / "checkpoint_manifest.json"
            checkpoint = json.loads(checkpoint_path.read_bytes())
            for item in checkpoint["artifacts"]:
                if item[0] in {"b6-predictions.jsonl", "b7-predictions.jsonl", "pairs.jsonl"}:
                    item[1] = hashlib.sha256((copied / item[0]).read_bytes()).hexdigest()
            checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
            with self.assertRaisesRegex(
                ComparisonPrerequisiteError, "frozen Step 9.3 runtime identity",
            ):
                verify_comparison_runtime(copied, repo_root=ROOT)


@unittest.skipUnless(psycopg is not None and DATABASE_URL, "PostgreSQL is required")
class B6B7ComparisonPrerequisiteLiveIntegrationTests(unittest.TestCase):
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

    def test_two_clean_runs_are_byte_identical_and_no_prompt_is_rendered(self) -> None:
        original_open = Path.open
        forbidden = (
            "data/abstention/interactive-answering-development-v2/gold/",
            "data/pilot/evaluation/",
            "data/scaled-v1/gold/",
            "data/scaled-v1/review_queues/",
            "data/scaled-v1/runtime/interactive.jsonl",
            "results/abstention/interactive-answering-development-v2/per-case.jsonl",
            "results/abstention/interactive-answering-development-v2/scorecard.json",
            ".env",
        )

        def guarded_open(path, *args, **kwargs):
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return original_open(path, *args, **kwargs)
            mode = kwargs.get("mode", args[0] if args else "r")
            if ("r" in mode or "+" in mode) and any(item in relative for item in forbidden):
                raise AssertionError(f"prohibited runtime read: {relative}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            with patch.object(Path, "open", guarded_open), patch(
                "answering.memory_answer.render_answer_prompt",
                side_effect=AssertionError("prompt rendered"),
            ):
                execute_comparison_runtime(self._factory, first, repo_root=ROOT)
            self._clean()
            with patch.object(Path, "open", guarded_open), patch(
                "answering.memory_answer.render_answer_prompt",
                side_effect=AssertionError("prompt rendered"),
            ):
                execute_comparison_runtime(self._factory, second, repo_root=ROOT)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )
            self.assertEqual({path.name for path in first.iterdir()}, {*ARTIFACTS, "checkpoint_manifest.json"})

    def test_clean_run_has_four_exact_pairs_and_zero_provider_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            pairs = execute_comparison_runtime(self._factory, output, repo_root=ROOT)
            checkpoint = verify_comparison_runtime(output, repo_root=ROOT)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(item.response_identity_match for item in pairs))
        self.assertEqual(checkpoint.b6_gate_application_count, 0)
        self.assertEqual(checkpoint.b7_answerability_decision_count, 4)
        self.assertEqual(checkpoint.b7_threshold_application_count, 4)
        self.assertEqual(checkpoint.provider_eligible_case_count, 0)
        self.assertEqual(checkpoint.failure_count, 0)

    def test_generation_eligibility_stops_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            original = comparison_runtime.build_evidence_package

            def allowed(*args, **kwargs):
                from types import SimpleNamespace
                original(*args, **kwargs)
                return SimpleNamespace(answer_allowed=True)

            with patch.object(comparison_runtime, "build_evidence_package", side_effect=allowed):
                with self.assertRaisesRegex(
                    ComparisonPrerequisiteError,
                    "b6_provider_eligibility_requires_new_approval",
                ):
                    execute_comparison_runtime(self._factory, output, repo_root=ROOT)
            self.assertFalse(output.exists())

    def test_b7_generation_eligibility_stops_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            from types import SimpleNamespace
            eligible = SimpleNamespace(output_decision="answerable", generation_allowed=True)
            with patch.object(comparison_runtime, "_threshold_application", return_value=eligible):
                with self.assertRaisesRegex(
                    ComparisonPrerequisiteError,
                    "b7_provider_eligibility_requires_new_approval",
                ):
                    execute_comparison_runtime(self._factory, output, repo_root=ROOT)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
