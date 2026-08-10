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
STEP93_COMMITTED_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "fff42248ee2c3f361caea561341b4d0e28728be98166d11f6f8f9d138694ba2d",
    "tests/integration/test_answerability.py": "73aeedc3f24f76e3f357e99793ef6ed85bc8c7610caf9b87fd298a7e2fb3c9f4",
    "tests/integration/test_interactive_answering_v2.py": "3eea1ed4c883c076ddfa6a2a0668e3ce77b7c1fb8ffa7b1188e4b1fb3a2abac2",
    "tests/integration/test_memory_answer.py": "1ca97a9ff1df5f11d5210b5434f5ded6830205df0b420c4d88570fdaa5f05f1d",
}
STEP94_PREREQUISITE_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "4a56ef43fa1ece155f0403d24f46833fb65752f2997eecaaa6e89eb7becb0202",
    "tests/integration/test_answerability.py": "7f5e14c52001cee93c27294a7f6739a578bd1e30020876f92d7c38e87e1cc584",
    "tests/integration/test_memory_answer.py": "ade88da0465e8c2ee0e5fb2e2ebe4a540fd74854701117b2312d73cb9cba2c8e",
}


class InteractiveInputV2IntegrationTests(unittest.TestCase):
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
        live = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in STEP94_PREREQUISITE_ADAPTER_SHA256
        }
        self.assertEqual(live, STEP94_PREREQUISITE_ADAPTER_SHA256)
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
