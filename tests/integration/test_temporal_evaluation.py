from __future__ import annotations

from dataclasses import replace
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
import hashlib

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from evaluation import run_temporal as runner
from evaluation.temporal import (
    TemporalEvaluationError,
    load_temporal_gold,
    load_temporal_runtime,
    record,
    run_temporal_cases,
    score_temporal,
    serialize_jsonl,
)
from storage.migrations import apply_migrations

if psycopg is not None:
    from storage.repository import StorageRepository


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
TEMPORAL_DATASET = REPO_ROOT / "data/phase4/temporal-development-v1"
TEMPORAL_RESULT = REPO_ROOT / "results/phase4/step4.4-temporal-evaluation-v1"
TEMPORAL_ARTIFACT_SHA256 = {
    "predictions.jsonl": "641e2b1221f6b0123c7521b95b997fa7d4a321faf7aa49f4fc8ab1713726c8b1",
    "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "scores.json": "23ce6b37522e1367d91e72959b3acfb1a6558597a2667f53c0da9bd970a53d9e",
    "run.json": "1613ed40d8e9a73c2263aa651400e2240fda9a3ca46174e76d903d49e44cb285",
    "findings.md": "1317ef934f24f8b3f7eb08b49703bde0bc23855ae1cde4e419d1d223d047159f",
    "manifest.json": "8f7cc49fbe5620094c618eaaaa98c27ce7a337fdb2747ca9918d8bcfe6d4d644",
}


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for temporal evaluation integration tests",
)
class TemporalEvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def reset_database(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def _run_current_lower_seams(self, output: Path) -> object:
        apply_migrations(self.connection, REPO_ROOT / "migrations")
        runner._require_clean_database(self.connection)
        runtime = load_temporal_runtime(TEMPORAL_DATASET / "runtime/cases.jsonl")
        predictions, failures = run_temporal_cases(
            self.connection, runtime, REPO_ROOT
        )
        runner._require_case_accounting(runtime, predictions, failures)
        runner._require_empty_output(output)
        output.mkdir(parents=True)
        prediction_path = output / "predictions.jsonl"
        failure_path = output / "failures.jsonl"
        runner._write_exclusive(prediction_path, serialize_jsonl(predictions))
        runner._write_exclusive(failure_path, serialize_jsonl(failures))
        runner._verify_persisted_accounting(runtime, prediction_path, failure_path)
        gold = load_temporal_gold(TEMPORAL_DATASET / "gold/cases.jsonl")
        scores = score_temporal(predictions, failures, gold)
        runner._write_exclusive(
            output / "scores.json", runner._json_bytes(record(scores))
        )
        return scores

    def test_all_cases_score_on_clean_database_and_repeat_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self.reset_database()
            first_scores = self._run_current_lower_seams(first)
            predictions = [json.loads(line) for line in first.joinpath("predictions.jsonl").read_text().splitlines()]
            failures = first.joinpath("failures.jsonl").read_text()
            scores = json.loads(first.joinpath("scores.json").read_text())
            self.assertEqual(len(predictions), 12)
            self.assertEqual(failures, "")
            self.assertEqual(first_scores.failure_count, 0)
            for name in (
                "event_ordering_accuracy",
                "date_normalization_accuracy",
                "interval_relation_accuracy",
                "current_state_accuracy",
                "historical_state_accuracy",
                "correction_visibility_accuracy",
            ):
                self.assertEqual(scores[name]["value"], 1.0, name)
            self.assertEqual(scores["mean_interval_iou"]["value"], 0.027027)

            by_case = {item["case_id"]: item for item in predictions}
            self.assertEqual(
                by_case["t44_u1_correction_before"]["current_claim_ids"],
                ["t44_before_old_claim"],
            )
            self.assertEqual(
                by_case["t44_u1_correction_at"]["current_claim_ids"],
                ["t44_at_new_claim"],
            )
            self.assertEqual(
                by_case["t44_u1_normal_change"]["historical_claim_ids"],
                ["t44_normal_old_claim"],
            )
            self.assertEqual(
                by_case["t44_u2_out_of_order"]["ordered_source_ids"],
                ["t44_order_conversation", "t44_order_email"],
            )
            runtime = load_temporal_runtime(
                REPO_ROOT / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
            )
            repeated = next(item for item in runtime if item.case_id == "t44_u1_repeated_evidence")
            self.assertEqual(repeated.claims[0]["span_ids"], ["t44_repeat_span_a", "t44_repeat_span_b"])
            self.assertEqual(
                self.connection.execute("SELECT count(*) FROM claims").fetchone()[0],
                0,
            )

            artifact_names = (
                "predictions.jsonl",
                "failures.jsonl",
                "scores.json",
            )
            first_artifacts = {
                name: first.joinpath(name).read_bytes() for name in artifact_names
            }
            self.reset_database()
            second_scores = self._run_current_lower_seams(second)
            for name, payload in first_artifacts.items():
                self.assertEqual(second.joinpath(name).read_bytes(), payload, name)
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    TEMPORAL_ARTIFACT_SHA256[name],
                    name,
                )
                self.assertEqual(TEMPORAL_RESULT.joinpath(name).read_bytes(), payload)
            self.assertEqual(record(first_scores), record(second_scores))
            for name in ("run.json", "findings.md", "manifest.json"):
                self.assertEqual(
                    hashlib.sha256(TEMPORAL_RESULT.joinpath(name).read_bytes()).hexdigest(),
                    TEMPORAL_ARTIFACT_SHA256[name],
                    name,
                )

    def test_cross_user_runtime_reference_becomes_sanitized_failure(self) -> None:
        self.reset_database()
        apply_migrations(self.connection, REPO_ROOT / "migrations")
        runtime = load_temporal_runtime(
            REPO_ROOT / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
        )
        changed = replace(runtime[0], user_id="user_002")
        predictions, failures = run_temporal_cases(
            self.connection, (changed,), REPO_ROOT
        )
        self.assertEqual(predictions, ())
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].case_id, changed.case_id)
        self.assertEqual(failures[0].location, "case")
        self.assertNotIn("Asha", failures[0].code)

    def test_runner_refuses_a_database_with_existing_runtime_rows(self) -> None:
        self.reset_database()
        apply_migrations(self.connection, REPO_ROOT / "migrations")
        self.connection.execute(
            "INSERT INTO memory_users (user_id, created_at) VALUES (%s, %s)",
            ("existing_user", "2026-01-01T00:00:00Z"),
        )
        with self.assertRaisesRegex(TemporalEvaluationError, "clean database"):
            runner._require_clean_database(self.connection)

    def test_runtime_import_graph_has_no_scaled_gold_or_oracle_dependency(self) -> None:
        source_path = REPO_ROOT / "src/evaluation/temporal.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any("scaled_scoring" in name for name in imports))
        self.assertNotIn("review_queues", source)
        self.assertNotIn("data/scaled-v1/gold", source)
        self.assertNotIn("data/scaled-v1/oracle", source)

    def test_existing_twenty_source_thirty_three_claim_replay_is_idempotent(self) -> None:
        from tests.integration.test_phase4_storage import (
            Phase4StorageIntegrationTests,
            _read_jsonl_all,
            _read_jsonl_prefix,
        )

        self.reset_database()
        apply_migrations(self.connection, REPO_ROOT / "migrations")
        sources = _read_jsonl_prefix(
            REPO_ROOT / "data/scaled-v1/runtime/sources.jsonl", 20
        )
        claims = _read_jsonl_all(
            REPO_ROOT
            / "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl"
        )
        self.assertEqual((len(sources), len(claims)), (20, 33))
        harness = Phase4StorageIntegrationTests(
            "test_phase3_handoff_loads_20_sources_and_33_weak_candidates_then_rolls_back"
        )
        harness.connection = self.connection
        harness.repository = StorageRepository(self.connection)
        harness._load_phase3_handoff(sources, claims)
        first = self._phase3_snapshot()
        harness._load_phase3_handoff(sources, claims)
        second = self._phase3_snapshot()
        self.assertEqual(first, second)
        self.assertEqual(first["source_count"], 20)
        self.assertEqual(first["claim_count"], 33)
        self.assertEqual(len(first["candidate"]), 33)
        self.assertEqual(first["current"], ())
        self.assertEqual(first["historical"], ())

    def _phase3_snapshot(self) -> dict[str, object]:
        statuses = {
            status: tuple(
                row[0]
                for row in self.connection.execute(
                    "SELECT claim_id FROM claim_versions WHERE lifecycle_status = %s ORDER BY claim_id",
                    (status,),
                ).fetchall()
            )
            for status in ("candidate", "current", "historical")
        }
        return {
            "source_count": self.connection.execute("SELECT count(*) FROM source_events").fetchone()[0],
            "claim_count": self.connection.execute("SELECT count(*) FROM claims").fetchone()[0],
            **statuses,
        }


if __name__ == "__main__":
    unittest.main()
