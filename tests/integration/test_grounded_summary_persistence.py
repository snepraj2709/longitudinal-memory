from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from ingestion.service import IngestionError, IngestionService
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations
from storage.repository import StorageRepository
from summaries.contracts import SessionizationRequest
from summaries.grounded import GroundedSummaryCoordinator, plan_grounded_summary
from summaries.repository import SessionSourceRepository
from summaries.sessions import SessionizationService
from summaries.summary_contracts import GroundedSummaryRequest
from summaries.summary_repository import SessionSummaryRepository, SummaryPersistenceError


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = BASE + timedelta(days=1)
SHA = "a" * 64


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for summary integration tests",
)
class GroundedSummaryPersistenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        self.applied = apply_migrations(self.connection, ROOT / "migrations")
        self.repository = StorageRepository(self.connection)
        self.summary_repository = SessionSummaryRepository(self.connection)
        for user_id in ("user_001", "user_002"):
            self.repository.insert_user(MemoryUser(user_id, BASE))
        self.repository.insert_extraction_version(
            ExtractionVersionRecord(
                "extractor_001",
                "model",
                "prompt",
                SHA,
                "schema",
                SHA,
                "predicate_registry_v2",
                SHA,
                SHA,
                BASE,
            )
        )

    def tearDown(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def _source(self, source_id: str, offset: int = 0) -> None:
        raw = "Delhi"
        self.repository.insert_source_event(
            SourceEventRecord(
                source_id,
                "user_001",
                "chat",
                "thread_001",
                f"ingest:{source_id}",
                BASE + timedelta(minutes=offset),
                BASE,
                raw,
                ["user_001"],
                {},
                hashlib.sha256(raw.encode()).hexdigest(),
            )
        )
        self.repository.insert_source_span(
            SourceSpanRecord(
                f"span_{source_id}",
                "user_001",
                source_id,
                f"message_{source_id}",
                "user_001",
                raw,
                0,
                len(raw),
            )
        )

    def _graph(
        self,
        *,
        two_sources: bool = False,
        second_source_without_evidence: bool = False,
    ) -> None:
        self._source("source_a")
        if two_sources:
            self._source("source_b", 1)
        claim = ClaimRecord(
            "claim_001",
            "user_001",
            "user_001",
            "user_001",
            "lives_in",
            "predicate_registry_v2",
            {"state": "Delhi"},
            "positive",
            "asserted",
            date(2026, 1, 1),
            None,
            None,
            None,
            "day",
            1,
            "durative",
            "standard",
            "extractor_001",
        )
        self.repository.insert_claim(claim)
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                "version_001",
                "user_001",
                claim.claim_id,
                "candidate",
                BASE,
                valid_from_date=date(2026, 1, 1),
                time_precision="day",
            )
        )
        evidence_sources = (
            ("source_a",)
            if second_source_without_evidence or not two_sources
            else ("source_a", "source_b")
        )
        for source_id in evidence_sources:
            self.repository.insert_evidence_link(
                EvidenceLinkRecord(
                    "user_001",
                    claim.claim_id,
                    f"span_{source_id}",
                    "supports",
                    1,
                )
            )

    def test_summary_membership_keeps_a_source_without_claim_evidence(self) -> None:
        self._graph(two_sources=True, second_source_without_evidence=True)
        definition, planned = self._plan("membership:no-claim-source")
        self.assertEqual(definition.source_ids, ("source_a", "source_b"))
        outcome = self.summary_repository.persist(planned, definition.source_ids)
        mapped = self.connection.execute(
            """
            SELECT source_id FROM session_summary_sources
            WHERE user_id = %s AND summary_id = %s ORDER BY source_order
            """,
            ("user_001", outcome.summary_id),
        ).fetchall()
        self.assertEqual(mapped, [("source_a",), ("source_b",)])

    def _definition(self, at: datetime = AS_OF):
        result = SessionizationService(
            SessionSourceRepository(self.connection)
        ).define_sessions(SessionizationRequest("user_001", at))
        self.assertEqual(len(result.definitions), 1)
        return result.definitions[0]

    def _plan(self, key: str, at: datetime = AS_OF):
        definition = self._definition(at)
        evidence = self.summary_repository.load_visible_evidence(definition, at)
        return definition, plan_grounded_summary(
            GroundedSummaryRequest(
                "user_001",
                definition.definition_id,
                definition.membership_sha256,
                at,
                key,
            ),
            evidence,
        )

    def _count(self, table: str) -> int:
        return self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def test_migration_repeats_and_enforces_composite_ownership_and_json(self) -> None:
        self.assertEqual(self.applied[-1], "0006_session_summaries.sql")
        self.assertEqual(apply_migrations(self.connection, ROOT / "migrations"), ())
        tables = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public' AND tablename LIKE 'session_summar%'
                """
            ).fetchall()
        }
        self.assertEqual(
            tables,
            {
                "session_summaries",
                "session_summary_sources",
                "session_summary_statements",
                "session_summary_statement_evidence",
            },
        )
        self._graph()
        definition, planned = self._plan("summary:create")
        self.summary_repository.persist(planned, definition.source_ids)
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE session_summary_statement_evidence
                SET user_id = 'user_002'
                WHERE user_id = 'user_001'
                """
            )
        with self.assertRaises(psycopg.errors.UniqueViolation):
            self.connection.execute(
                """
                INSERT INTO session_summaries (
                    summary_id, user_id, session_definition_id,
                    session_membership_sha256, renderer_version,
                    idempotency_key, input_snapshot_sha256, summary_text,
                    valid_time_kind, contains_sensitive, transaction_from
                )
                SELECT %s, user_id, session_definition_id,
                       session_membership_sha256, renderer_version,
                       'another-key', %s, summary_text,
                       'unknown', false, transaction_from
                FROM session_summaries WHERE summary_id = %s
                """,
                ("b" * 64, "c" * 64, planned.summary.summary_id),
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                """
                UPDATE session_summaries
                SET transaction_to = transaction_from
                WHERE summary_id = %s
                """,
                (planned.summary.summary_id,),
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                "UPDATE session_summary_statements SET claim_ids = '{}'::jsonb"
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE session_summary_statement_evidence
                SET claim_version_id = 'missing_version'
                """
            )

    def test_replay_unchanged_successor_empty_and_stale_are_exact(self) -> None:
        self._graph()
        definition, first = self._plan("summary:first")
        created = self.summary_repository.persist(first, definition.source_ids)
        self.assertTrue(created.created)
        replay = self.summary_repository.persist(first, definition.source_ids)
        self.assertTrue(replay.replayed)
        same_definition, unchanged = self._plan("summary:unchanged")
        no_row = self.summary_repository.persist(unchanged, same_definition.source_ids)
        self.assertFalse(no_row.created)
        self.assertFalse(no_row.replayed)
        self.assertEqual(self._count("session_summaries"), 1)

        changed_at = AS_OF + timedelta(hours=1)
        self.connection.execute(
            """
            UPDATE claim_versions SET transaction_to = %s
            WHERE user_id = 'user_001' AND version_id = 'version_001'
            """,
            (changed_at,),
        )
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                "version_002",
                "user_001",
                "claim_001",
                "confirmed",
                changed_at,
                valid_from_date=date(2026, 1, 1),
                time_precision="day",
            )
        )
        second_at = AS_OF + timedelta(hours=2)
        second_definition, second = self._plan("summary:second", second_at)
        successor = self.summary_repository.persist(second, second_definition.source_ids)
        self.assertTrue(successor.created)
        self.assertEqual(self._count("session_summaries"), 2)
        intervals = self.connection.execute(
            """
            SELECT transaction_from, transaction_to FROM session_summaries
            ORDER BY transaction_from
            """
        ).fetchall()
        self.assertEqual(intervals[0][1], second_at)
        self.assertIsNone(intervals[1][1])
        with self.assertRaisesRegex(SummaryPersistenceError, "stale_summary_read"):
            stale_definition, stale = self._plan("summary:stale", AS_OF)
            self.summary_repository.persist(stale, stale_definition.source_ids)
        with self.assertRaisesRegex(SummaryPersistenceError, "idempotency_drift"):
            drifted = second.__class__(second.plan_id, second.input_snapshot_sha256, first.request, second.summary)
            self.summary_repository.persist(drifted, second_definition.source_ids)

        excluded_at = second_at + timedelta(hours=1)
        self.connection.execute(
            """
            UPDATE claim_versions SET transaction_to = %s
            WHERE user_id = 'user_001' AND version_id = 'version_002'
            """,
            (excluded_at,),
        )
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                "version_003",
                "user_001",
                "claim_001",
                "excluded",
                excluded_at,
                valid_from_date=date(2026, 1, 1),
                time_precision="day",
            )
        )
        empty_at = excluded_at + timedelta(hours=1)
        empty_definition, empty = self._plan("summary:empty", empty_at)
        self.assertEqual(empty.summary.observed_facts, ())
        retired = self.summary_repository.persist(empty, empty_definition.source_ids)
        self.assertTrue(retired.retired)
        self.assertIsNone(retired.summary_id)
        self.assertEqual(self._count("session_summaries"), 2)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM session_summaries WHERE transaction_to IS NULL"
            ).fetchone()[0],
            0,
        )

    def test_transaction_failure_rolls_back_and_concurrent_replay_is_once_only(self) -> None:
        self._graph()
        self._source("source_unrelated", 2)
        definition, planned = self._plan("summary:rollback")
        with self.assertRaisesRegex(SummaryPersistenceError, "summary_write_failed"):
            self.summary_repository.persist(planned, ("source_unrelated",))
        self.assertEqual(self._count("session_summaries"), 0)

        def persist_once():
            connection = psycopg.connect(DATABASE_URL, autocommit=True)
            try:
                return SessionSummaryRepository(connection).persist(
                    planned,
                    definition.source_ids,
                )
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(lambda _: persist_once(), range(2)))
        self.assertEqual(sum(item.created for item in outcomes), 1)
        self.assertEqual(sum(item.replayed for item in outcomes), 1)
        self.assertEqual(self._count("session_summaries"), 1)

    def test_summary_delete_preserves_base_rows_and_source_delete_purges_then_rebuilds(self) -> None:
        self._graph(two_sources=True)
        definition, planned = self._plan("summary:delete")
        self.summary_repository.persist(planned, definition.source_ids)
        self.connection.execute(
            """
            DELETE FROM evidence_links
            WHERE user_id = 'user_001' AND span_id = 'span_source_a'
            """
        )
        self.assertEqual(self._count("session_summaries"), 0)
        self.assertEqual(self._count("source_events"), 2)
        self.assertEqual(self._count("claims"), 1)
        self.repository.insert_evidence_link(
            EvidenceLinkRecord(
                "user_001", "claim_001", "span_source_a", "supports", 1
            )
        )
        self.summary_repository.persist(planned, definition.source_ids)
        self.connection.execute(
            "DELETE FROM session_summaries WHERE user_id = 'user_001'"
        )
        self.assertEqual(self._count("source_events"), 2)
        self.assertEqual(self._count("claims"), 1)
        self.assertEqual(self._count("evidence_links"), 2)
        self.summary_repository.persist(planned, definition.source_ids)
        with self.assertRaises(IngestionError):
            IngestionService(self.connection).delete_source(
                "user_002", "source_a", AS_OF + timedelta(hours=1)
            )
        self.assertEqual(self._count("session_summaries"), 1)

        service = IngestionService(self.connection)
        deleted_at = AS_OF + timedelta(hours=2)
        service.delete_source("user_001", "source_a", deleted_at)
        self.assertEqual(self._count("session_summaries"), 0)
        event_id = self.connection.execute(
            """
            SELECT event_id FROM processing_outbox
            WHERE user_id = 'user_001' AND event_type = 'source_deleted'
              AND aggregate_id = 'source_a'
            """
        ).fetchone()[0]
        event = self.repository.get_outbox("user_001", event_id)
        self.assertIsNotNone(event)
        rebuilt = GroundedSummaryCoordinator(self.connection).process(event)
        self.assertEqual(rebuilt.created_count, 1)
        self.assertEqual(self._count("session_summaries"), 1)
        self.assertEqual(self._count("source_events"), 1)
        self.assertEqual(self._count("claims"), 1)

        final_at = deleted_at + timedelta(hours=1)
        service.delete_source("user_001", "source_b", final_at)
        self.assertEqual(self._count("session_summaries"), 0)
        final_event_id = self.connection.execute(
            """
            SELECT event_id FROM processing_outbox
            WHERE user_id = 'user_001' AND event_type = 'source_deleted'
              AND aggregate_id = 'source_b'
            """
        ).fetchone()[0]
        final_event = self.repository.get_outbox("user_001", final_event_id)
        self.assertIsNotNone(final_event)
        final = GroundedSummaryCoordinator(self.connection).process(final_event)
        self.assertEqual(final.session_count, 0)
        self.assertEqual(self._count("session_summaries"), 0)


if __name__ == "__main__":
    unittest.main()
