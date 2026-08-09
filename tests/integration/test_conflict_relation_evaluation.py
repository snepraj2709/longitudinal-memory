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
    execute_relation_evaluation,
)
from storage.migrations import apply_migrations


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")


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

    def test_all_eight_pairs_classify_and_persist_without_leaking_state(self) -> None:
        self.reset_database()
        with tempfile.TemporaryDirectory() as directory:
            release = execute_relation_evaluation(
                self.connection,
                repo_root=ROOT,
                result_root=Path(directory) / "release",
            )
            output = Path(directory) / "release"
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
        self.assertEqual(release["execution"]["cross_user_relation_count"], 0)
        self.assertEqual(release["execution"]["model_calls"], 0)
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
                execute_relation_evaluation(
                    self.connection, repo_root=ROOT, result_root=output
                )
                releases.append(
                    {
                        name: (output / name).read_bytes()
                        for name in (
                            "predictions.jsonl", "failures.jsonl", "scores.json",
                            "run.json", "findings.md", "manifest.json",
                        )
                    }
                )
            self.assertEqual(releases[0], releases[1])
            with self.assertRaisesRegex(RelationEvaluationError, "absent or empty"):
                execute_relation_evaluation(
                    self.connection, repo_root=ROOT, result_root=roots[1]
                )

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
