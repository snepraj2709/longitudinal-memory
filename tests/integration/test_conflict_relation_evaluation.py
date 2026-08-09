from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import ast

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from conflicts.relation_evaluation import (
    CASE_IDS,
    PRESENT_GOLD_LABELS,
    RelationEvaluationError,
    file_sha256,
    load_relation_dataset_runtime,
    persist_outputs_before_gold,
    run_relation_cases,
)
from storage.migrations import apply_migrations


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RELATION_DATASET = ROOT / "data/conflicts/relation-development-v1"
RELATION_RESULT = ROOT / "results/conflicts/relation-classification-development-v1"
RELATION_ARTIFACT_SHA256 = {
    "predictions.jsonl": "f41a15fa83df9602c0536976aaebbc7aef9d7e33d1353f057dbeb52a6b3cf201",
    "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "scores.json": "4efc0bbc243e9bca92898da6aeabf7d0693ef479b7cfeef39afb647c2ecd49bd",
    "run.json": "b0adae4c6bd9f801fbb43546e62312765b65d7601833448804624a3e9aaaba3a",
    "findings.md": "a7f4c8586d2f0ea4f94fd51b0ba5091d63b2ab0cc65a5334b6b658cb719cfb6b",
    "manifest.json": "2f29edd580957192cc7808e2e4a454b7fa7c0b89bfc9ea3851efbf9e1c76179e",
}


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for relation evaluation tests",
)
class ConflictRelationEvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def reset_database(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        apply_migrations(self.connection, ROOT / "migrations")

    def run_current_lower_seams(self, output: Path):
        manifest, runtime = load_relation_dataset_runtime(
            RELATION_DATASET, repo_root=ROOT
        )
        predictions, failures = run_relation_cases(
            self.connection,
            runtime,
            repo_root=ROOT,
            reference_runtime_path=ROOT / manifest["reference_runtime"]["path"],
        )
        scorecard = persist_outputs_before_gold(
            output,
            runtime,
            predictions,
            failures,
            RELATION_DATASET / manifest["gold"]["path"],
            expected_gold_sha256=manifest["gold"]["sha256"],
            runtime_resources_closed=True,
        )
        return scorecard

    def test_all_eight_pairs_classify_and_persist_without_leaking_state(self) -> None:
        self.reset_database()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            scorecard = self.run_current_lower_seams(output)
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text().splitlines()
            ]
            failures = (output / "failures.jsonl").read_text()
            scores = json.loads((output / "scores.json").read_text())
        self.assertEqual({item["case_id"] for item in predictions}, set(CASE_IDS))
        self.assertEqual(
            {
                label: sum(item["label"] == label for item in predictions)
                for label in PRESENT_GOLD_LABELS
            },
            PRESENT_GOLD_LABELS,
        )
        self.assertEqual(failures, "")
        self.assertEqual(scores["overall_label_accuracy"]["value"], 1)
        self.assertEqual(scores["exact_relation_set_accuracy"]["value"], 1)
        self.assertIsNone(scores["direction_accuracy"]["value"])
        self.assertTrue(scores["exact_match_gate_passed"])
        self.assertEqual(scorecard.cross_user_relation_count, 0)
        self.assertEqual(scorecard.model_prediction_count, 0)
        for table in (
            "memory_users", "source_events", "source_spans", "claims",
            "claim_versions", "evidence_links", "conflict_decisions",
            "claim_relations", "conflict_decision_evidence",
        ):
            self.assertEqual(
                self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0],
                0,
            )

    def test_two_clean_database_runs_are_byte_identical_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = (Path(directory) / "first", Path(directory) / "second")
            releases: list[dict[str, bytes]] = []
            for output in roots:
                self.reset_database()
                self.run_current_lower_seams(output)
                releases.append(
                    {
                        name: (output / name).read_bytes()
                        for name in (
                            "predictions.jsonl", "failures.jsonl", "scores.json",
                        )
                    }
                )
            self.assertEqual(releases[0], releases[1])
            for name, payload in releases[0].items():
                self.assertEqual(
                    file_sha256(RELATION_RESULT / name),
                    RELATION_ARTIFACT_SHA256[name],
                )
                self.assertEqual(payload, (RELATION_RESULT / name).read_bytes())
            for name in ("run.json", "findings.md", "manifest.json"):
                self.assertEqual(
                    file_sha256(RELATION_RESULT / name),
                    RELATION_ARTIFACT_SHA256[name],
                )
            with self.assertRaisesRegex(RelationEvaluationError, "absent or empty"):
                self.run_current_lower_seams(roots[1])

    def test_runtime_import_graph_has_no_forbidden_gold_or_test_user_path(self) -> None:
        source = (ROOT / "src/conflicts/relation_evaluation.py").read_text()
        imported = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(
            any("gold" in name or "oracle" in name or "review" in name for name in imported)
        )
        self.assertNotIn("temporal-development-v1/gold", source)
        self.assertNotIn("scaled-v1/gold", source)
        self.assertNotIn("review_queues", source)
        for user_number in range(3, 11):
            self.assertNotIn(f"user_{user_number:03d}", source)


if __name__ == "__main__":
    unittest.main()
