from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from storage.contracts import MemoryUser, SourceEventRecord
from storage.migrations import apply_migrations
from storage.repository import StorageRepository
from summaries.contracts import SessionizationRequest
from summaries.repository import SessionSourceRepository
from summaries.sessions import SessionizationService
from summaries.sessionization_evaluation import (
    EXPECTED_TYPE_COUNTS,
    RESULT_ROOT,
    canonical_json_bytes,
    load_sessionization_runtime,
    materialize_runtime_sources,
    run_sessionization,
    score_sessionization,
    serialize_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)
ARTIFACTS = (
    "predictions.jsonl",
    "failures.jsonl",
    "scores.json",
    "run.json",
    "findings.md",
    "manifest.json",
)
FROZEN_RELEASE_SHA256 = "34611525b22cb9dcf8b5c9eb4affd2422d778b58b4b443c90913ce29c8f9365c"


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for sessionization integration tests",
)
class SessionizationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._reset_database()

    def tearDown(self) -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    def _reset_database(self) -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    @staticmethod
    def _factory():
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def _prepared_connection(self):
        connection = self._factory()
        apply_migrations(connection, ROOT / "migrations")
        return connection

    def _historical_release_bytes(self):
        _, sources = load_sessionization_runtime(repo_root=ROOT)
        connection = self._factory()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                migration_root = Path(temporary)
                for name in (
                    "0001_phase4_storage.sql",
                    "0002_ingestion_reprocessing.sql",
                    "0003_temporal_lifecycle.sql",
                    "0004_conflict_relations.sql",
                    "0005_belief_resolution.sql",
                ):
                    shutil.copyfile(ROOT / "migrations" / name, migration_root / name)
                apply_migrations(connection, migration_root)
            materialize_runtime_sources(connection, sources)
            predictions, failures = run_sessionization(connection, sources)
            scores = score_sessionization(sources, predictions, failures)
            self.assertEqual(scores.source_coverage["value"], 1.0)
            self.assertEqual(scores.session_count, 20)
            self.assertEqual(scores.session_counts_by_type, EXPECTED_TYPE_COUNTS)
            self.assertEqual(scores.failure_count, 0)
            self.assertEqual(scores.invalid_session_count, 0)
            self.assertEqual(scores.cross_user_count, 0)
            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name LIKE %s
                    """,
                    ("%session%",),
                ).fetchall()
            }
            self.assertEqual(tables, set())
            return (
                serialize_jsonl(predictions),
                serialize_jsonl(failures),
                canonical_json_bytes(scores.__dict__),
                canonical_json_bytes(
                    {
                        "artifact_version": "sessionization_development_v1",
                        "dataset_version": "sessionization_development_v1",
                        "starting_commit": "548b12750142eb07c8749f0a8f7834ba4c7b7c3b",
                        "source_count": len(sources),
                        "prediction_count": len(predictions),
                        "failure_count": len(failures),
                        "request_count": 0,
                        "retry_count": 0,
                        "model_calls": 0,
                        "cost_usd": 0,
                        "execution_mode": "deterministic",
                        "persisted_session_rows": 0,
                    }
                ),
                (
                    "# Sessionization development findings\n\n"
                    "All 20 development sources were assigned once, producing 20 deterministic "
                    "sessions with no failures or cross-user membership. The frozen sources use "
                    "unique declared thread IDs, so the grouping edge cases remain covered by tests.\n\n"
                    "This run measures boundary reproducibility and source coverage. It does not "
                    "evaluate summary quality or production traffic.\n"
                ).encode("utf-8"),
            )
        finally:
            connection.close()

    @staticmethod
    def _insert_source(
        repository: StorageRepository,
        source_id: str,
        produced_at: datetime,
        *,
        ingested_at: datetime | None = None,
        thread_id: str | None = None,
    ) -> None:
        raw = f"synthetic source {source_id}"
        repository.insert_source_event(
            SourceEventRecord(
                source_id,
                "user_dev",
                "chat",
                None,
                f"session-test:{source_id}",
                produced_at,
                ingested_at or produced_at,
                raw,
                ["user_dev"],
                {} if thread_id is None else {"thread_id": thread_id},
                hashlib.sha256(raw.encode()).hexdigest(),
            )
        )

    def test_full_release_is_byte_identical_on_two_clean_databases(self) -> None:
        checked = ROOT / RESULT_ROOT
        self.assertTrue(checked.is_dir())
        self.assertEqual(
            hashlib.sha256((checked / "manifest.json").read_bytes()).hexdigest(),
            FROZEN_RELEASE_SHA256,
        )
        first = self._historical_release_bytes()
        self._reset_database()
        second = self._historical_release_bytes()
        self.assertEqual(first, second)
        for name, content in zip(ARTIFACTS[:-1], first):
            self.assertEqual(content, (checked / name).read_bytes())
        for name in ARTIFACTS:
            self.assertTrue((checked / name).is_file())

    def test_exact_cutoff_is_inclusive_and_queries_are_user_scoped(self) -> None:
        _, sources = load_sessionization_runtime(repo_root=ROOT)
        connection = self._prepared_connection()
        try:
            materialize_runtime_sources(connection, sources)
            service = SessionizationService(SessionSourceRepository(connection))
            last = max(item.ingested_at for item in sources if item.user_id == "user_001")
            at_cutoff = service.define_sessions(
                SessionizationRequest("user_001", last)
            )
            before = service.define_sessions(
                SessionizationRequest("user_001", last - timedelta(microseconds=1))
            )
            self.assertEqual(sum(len(item.source_ids) for item in at_cutoff.definitions), 10)
            self.assertEqual(sum(len(item.source_ids) for item in before.definitions), 9)
            self.assertTrue(
                all("user_002" not in source_id for item in at_cutoff.definitions for source_id in item.source_ids)
            )
        finally:
            connection.close()

    def test_late_chat_joins_bridges_and_then_splits_at_1801(self) -> None:
        connection = self._prepared_connection()
        try:
            repository = StorageRepository(connection)
            repository.insert_user(MemoryUser("user_dev", BASE))
            self._insert_source(repository, "chat_a", BASE)
            self._insert_source(repository, "chat_c", BASE + timedelta(seconds=3600), ingested_at=BASE)
            self._insert_source(
                repository,
                "chat_b",
                BASE + timedelta(seconds=1800),
                ingested_at=BASE + timedelta(hours=2),
            )
            service = SessionizationService(SessionSourceRepository(connection))
            before = service.define_sessions(
                SessionizationRequest("user_dev", BASE + timedelta(hours=1))
            )
            after = service.define_sessions(
                SessionizationRequest("user_dev", BASE + timedelta(hours=2))
            )
            self.assertEqual([item.source_ids for item in before.definitions], [("chat_a",), ("chat_c",)])
            self.assertEqual([item.source_ids for item in after.definitions], [("chat_a", "chat_b", "chat_c")])
            self._insert_source(repository, "chat_d", BASE + timedelta(seconds=5401))
            split = service.define_sessions(
                SessionizationRequest("user_dev", BASE + timedelta(hours=3))
            )
            self.assertEqual(
                [item.source_ids for item in split.definitions],
                [("chat_a", "chat_b", "chat_c"), ("chat_d",)],
            )
        finally:
            connection.close()

    def test_deletion_recomputes_bridge_last_declared_survivor_and_tombstone(self) -> None:
        connection = self._prepared_connection()
        try:
            repository = StorageRepository(connection)
            repository.insert_user(MemoryUser("user_dev", BASE))
            for source_id, offset in (("chat_a", 0), ("chat_b", 1800), ("chat_c", 3600)):
                self._insert_source(repository, source_id, BASE + timedelta(seconds=offset))
            self._insert_source(repository, "chat_d", BASE + timedelta(seconds=10), thread_id="declared")
            self._insert_source(repository, "chat_e", BASE + timedelta(seconds=20), thread_id="declared")
            service = SessionizationService(SessionSourceRepository(connection))
            full = service.define_sessions(SessionizationRequest("user_dev", BASE + timedelta(hours=2)))
            declared_id = next(item.definition_id for item in full.definitions if item.source_ids == ("chat_d", "chat_e"))
            connection.execute("DELETE FROM source_events WHERE user_id = %s AND source_id = %s", ("user_dev", "chat_b"))
            bridgeless = service.define_sessions(SessionizationRequest("user_dev", BASE + timedelta(hours=2)))
            self.assertIn(("chat_a",), [item.source_ids for item in bridgeless.definitions])
            self.assertIn(("chat_c",), [item.source_ids for item in bridgeless.definitions])
            connection.execute("DELETE FROM source_events WHERE user_id = %s AND source_id = %s", ("user_dev", "chat_c"))
            connection.execute("DELETE FROM source_events WHERE user_id = %s AND source_id = %s", ("user_dev", "chat_d"))
            survivor = service.define_sessions(SessionizationRequest("user_dev", BASE + timedelta(hours=2)))
            declared = next(item for item in survivor.definitions if item.source_ids == ("chat_e",))
            self.assertEqual(declared.definition_id, declared_id)
            self.assertIn(("chat_a",), [item.source_ids for item in survivor.definitions])
            connection.execute(
                """
                INSERT INTO source_tombstones
                    (user_id, source_id, idempotency_key, content_hash, deleted_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                ("user_dev", "chat_b", "session-test:chat_b", "b" * 64, BASE + timedelta(hours=3)),
            )
            tombstoned = service.define_sessions(SessionizationRequest("user_dev", BASE + timedelta(hours=3)))
            self.assertTrue(all("chat_b" not in item.source_ids for item in tombstoned.definitions))
        finally:
            connection.close()

    def test_current_schema_adds_only_derived_summary_session_tables(self) -> None:
        connection = self._prepared_connection()
        try:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ).fetchall()
            }
            self.assertEqual(
                {name for name in names if "session" in name},
                {
                    "session_summaries",
                    "session_summary_sources",
                    "session_summary_statements",
                    "session_summary_statement_evidence",
                },
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
