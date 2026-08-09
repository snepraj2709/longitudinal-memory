from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from conflicts.resolution_evaluation import (
    DATASET_ROOT,
    execute_resolution_evaluation,
    load_resolution_dataset_runtime,
    run_resolution_cases,
)
from storage.migrations import apply_migrations


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for resolution evaluation tests",
)
class BeliefResolutionEvaluationIntegrationTests(unittest.TestCase):
    def _reset(self) -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    def _factory(self):
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def test_all_eight_cases_and_two_clean_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self._reset()
            release_one = execute_resolution_evaluation(
                self._factory, repo_root=ROOT, result_root=first
            )
            self._reset()
            release_two = execute_resolution_evaluation(
                self._factory, repo_root=ROOT, result_root=second
            )
            self.assertEqual(release_one, release_two)
            self.assertEqual(
                sorted(path.name for path in first.iterdir()),
                [
                    "failures.jsonl", "findings.md", "manifest.json",
                    "predictions.jsonl", "run.json", "scores.json",
                ],
            )
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            predictions = [
                json.loads(line)
                for line in (first / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(predictions), 8)
            self.assertEqual((first / "failures.jsonl").read_bytes(), b"")
            outcomes = {}
            for prediction in predictions:
                outcomes[prediction["outcome"]] = outcomes.get(prediction["outcome"], 0) + 1
                self.assertIsNone(prediction["belief_confidence"])
                self.assertEqual(prediction["execution_mode"], "deterministic")
                self.assertIsNone(prediction["model"])
                self.assertTrue(prediction["decision_evidence_ids"])
            self.assertEqual(
                outcomes,
                {
                    "disputed": 1,
                    "excluded": 2,
                    "no_change": 4,
                    "temporal_change_resolved": 1,
                },
            )
            scores = json.loads((first / "scores.json").read_text())
            self.assertTrue(scores["exact_match_gate_passed"])
            self.assertEqual(scores["cross_user_count"], 0)
            self.assertEqual(scores["supersession_accuracy"]["status"], "not_evaluated")
            self.assertIsNone(scores["supersession_accuracy"]["value"])
            self.assertEqual(release_one["execution"]["model_calls"], 0)

    def test_cross_user_runtime_identity_fails_safely_before_resolution(self) -> None:
        self._reset()
        connection = self._factory()
        try:
            apply_migrations(connection, ROOT / "migrations")
            manifest, cases = load_resolution_dataset_runtime(
                ROOT / DATASET_ROOT, repo_root=ROOT
            )
            relation_manifest = json.loads(
                (ROOT / "data/conflicts/relation-development-v1/manifest.json").read_text()
            )
            case = replace(cases[0], user_id="user_002")
            predictions, failures = run_resolution_cases(
                connection,
                (case,),
                repo_root=ROOT,
                reference_runtime_path=ROOT / relation_manifest["reference_runtime"]["path"],
            )
            self.assertEqual(predictions, ())
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0].code, "case_execution_failed")
            self.assertEqual(failures[0].location, "case")
            self.assertNotIn("user_001", str(failures[0]))
        finally:
            connection.close()

    def test_nonempty_result_directory_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "existing").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                execute_resolution_evaluation(
                    self._factory, repo_root=ROOT, result_root=output
                )
            self.assertEqual((output / "existing").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
