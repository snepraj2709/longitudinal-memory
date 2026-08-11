from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from ingestion.service import IngestionService
from retrieval.contracts import (
    AtomicIndexInput,
    ClaimVersionLineage,
    RelationLineage,
    SessionIndexInput,
    SessionStatementInput,
    SourceSpanLineage,
    TransactionTime,
    ValidTime,
    load_index_config,
)
from retrieval.embeddings import DeterministicTokenHashEmbedder
from retrieval.indexing import build_atomic_index_record, build_session_index_record
from retrieval.index_evaluation import _atomic_inputs
from retrieval.repository import (
    IndexBuildRequest,
    RetrievalIndexRepository,
    RetrievalPersistenceConflict,
)
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


ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "migrations"
CONFIG_PATH = ROOT / "configs/retrieval/index_v1.json"
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 8, 1, 9, tzinfo=UTC)
AS_OF = BASE + timedelta(days=1)
SHA = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for retrieval integration tests",
)
class RetrievalIndexIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        self.applied = apply_migrations(self.connection, MIGRATIONS)
        self.storage = StorageRepository(self.connection)
        self.repository = RetrievalIndexRepository(
            self.connection, config_path=CONFIG_PATH
        )
        self.config = load_index_config(CONFIG_PATH)
        self.embedder = DeterministicTokenHashEmbedder()
        self._insert_graph()

    def tearDown(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def _insert_graph(self) -> None:
        for user_id in ("user_001", "user_002"):
            self.storage.insert_user(MemoryUser(user_id, BASE))
        self.storage.insert_extraction_version(
            ExtractionVersionRecord(
                "extractor_001",
                "deterministic",
                "prompt_v1",
                SHA,
                "schema_v1",
                SHA,
                "predicate_registry_v2",
                SHA,
                SHA,
                BASE,
            )
        )
        for source_id, produced in (("source_001", BASE), ("source_002", BASE + timedelta(hours=1))):
            raw = f"Evidence for {source_id}"
            self.storage.insert_source_event(
                SourceEventRecord(
                    source_id,
                    "user_001",
                    "conversation",
                    None,
                    f"ingest:{source_id}",
                    produced,
                    BASE,
                    raw,
                    ["user_001"],
                    {},
                    digest(raw),
                )
            )
            self.storage.insert_source_span(
                SourceSpanRecord(
                    f"span_{source_id[-3:]}",
                    "user_001",
                    source_id,
                    f"message_{source_id[-3:]}",
                    "user_001",
                    raw,
                    0,
                    len(raw),
                )
            )
        raw = "Foreign evidence"
        self.storage.insert_source_event(
            SourceEventRecord(
                "source_foreign",
                "user_002",
                "conversation",
                None,
                "ingest:foreign",
                BASE,
                BASE,
                raw,
                ["user_002"],
                {},
                digest(raw),
            )
        )
        self.storage.insert_source_span(
            SourceSpanRecord(
                "span_foreign",
                "user_002",
                "source_foreign",
                "message_foreign",
                "user_002",
                raw,
                0,
                len(raw),
            )
        )
        self._claim(
            "claim_001", "version_001", "candidate", "remote work", ("001", "002")
        )
        self._claim(
            "claim_002", "version_002", "superseded", "office work", ("002",)
        )
        self.connection.execute(
            """
            INSERT INTO conflict_decisions (
                decision_id, user_id, classifier_version, rule_version,
                pair_id, left_claim_id, right_claim_id, left_version_id,
                right_version_id, input_snapshot_sha256, transaction_as_of,
                matched_rule, label, classified_at
            ) VALUES (
                'decision_001', 'user_001', 'relation_classifier_v1',
                'conflict_relation_rules_v1', 'pair_001', 'claim_001',
                'claim_002', 'version_001', 'version_002', %s, %s,
                'explicit_correction', 'explicit_correction', %s
            )
            """,
            (SHA, AS_OF, AS_OF),
        )
        self.connection.execute(
            """
            INSERT INTO claim_relations (
                relation_id, user_id, decision_id, classifier_version,
                source_claim_id, target_claim_id, relation_type, confidence,
                input_snapshot_sha256, created_at
            ) VALUES (
                'relation_001', 'user_001', 'decision_001',
                'relation_classifier_v1', 'claim_002', 'claim_001',
                'corrects', 1, %s, %s
            )
            """,
            (SHA, AS_OF),
        )
        self.connection.execute(
            """
            INSERT INTO session_summaries (
                summary_id, user_id, session_definition_id,
                session_membership_sha256, renderer_version, idempotency_key,
                input_snapshot_sha256, summary_text, valid_time_kind,
                contains_sensitive, transaction_from
            ) VALUES (
                %s, 'user_001', %s, %s, 'session_summary_renderer_v1',
                'summary:index-fixture', %s,
                'Observed facts\n- Candidate: user_001 works remotely.\n\nUnresolved questions\n- None.',
                'unknown', false, %s
            )
            """,
            (
                digest("summary_001"),
                digest("session_001"),
                digest("membership_001"),
                digest("summary_snapshot_001"),
                BASE,
            ),
        )
        for order, source_id in enumerate(("source_001", "source_002")):
            self.connection.execute(
                """
                INSERT INTO session_summary_sources (
                    user_id, summary_id, source_id, source_order
                ) VALUES ('user_001', %s, %s, %s)
                """,
                (digest("summary_001"), source_id, order),
            )

    def _claim(
        self,
        claim_id: str,
        version_id: str,
        lifecycle_status: str,
        value: str,
        source_suffixes: tuple[str, ...],
    ) -> None:
        self.storage.insert_claim(
            ClaimRecord(
                claim_id,
                "user_001",
                "user_001",
                "user_001",
                "work_preference",
                "predicate_registry_v2",
                value,
                "positive",
                "asserted",
                date(2026, 8, 1),
                None,
                None,
                None,
                "day",
                0.9,
                None,
                "standard",
                "extractor_001",
            )
        )
        self.storage.insert_claim_version(
            ClaimVersionRecord(
                version_id,
                "user_001",
                claim_id,
                lifecycle_status,
                BASE,
                valid_from_date=date(2026, 8, 1),
                time_precision="day",
            )
        )
        for suffix in source_suffixes:
            self.storage.insert_evidence_link(
                EvidenceLinkRecord(
                    "user_001", claim_id, f"span_{suffix}", "supports", 0.9
                )
            )

    def _atomic(
        self,
        claim_id: str = "claim_001",
        version_id: str = "version_001",
        lifecycle_status: str = "candidate",
        value: str = "remote work",
        source_suffixes: tuple[str, ...] = ("001", "002"),
        *,
        relation: bool = True,
    ):
        relation_lineage = (
            RelationLineage(
                "relation_001",
                "user_001",
                "claim_002",
                "claim_001",
                "corrects",
                "incoming",
                0,
            ),
        ) if relation else ()
        value_input = AtomicIndexInput(
            "user_001",
            claim_id,
            version_id,
            "user_001",
            "user_001",
            "work_preference",
            value,
            "positive",
            "asserted",
            None,
            lifecycle_status,
            ValidTime("day", valid_from_date=date(2026, 8, 1)),
            TransactionTime(BASE),
            "standard",
            (
                ClaimVersionLineage(
                    "user_001", claim_id, version_id, lifecycle_status, 0
                ),
            ),
            tuple(
                SourceSpanLineage(
                    "user_001",
                    claim_id,
                    version_id,
                    f"source_{suffix}",
                    f"span_{suffix}",
                    "supports",
                    order,
                )
                for order, suffix in enumerate(source_suffixes)
            ),
            relation_lineage,
        )
        return build_atomic_index_record(
            "user_001",
            AS_OF,
            value_input,
            config=self.config,
            embedder=self.embedder,
        )

    def _session(self):
        statement = SessionStatementInput(
            "statement_001",
            "observed_fact",
            "candidate",
            "Candidate: user_001 works remotely.",
            (
                ClaimVersionLineage(
                    "user_001", "claim_001", "version_001", "candidate", 0
                ),
            ),
            (
                SourceSpanLineage(
                    "user_001",
                    "claim_001",
                    "version_001",
                    "source_001",
                    "span_001",
                    "supports",
                    0,
                ),
            ),
            0,
        )
        value = SessionIndexInput(
            "user_001",
            digest("summary_001"),
            digest("session_001"),
            "session_summary_renderer_v1",
            "Observed facts\n- Candidate: user_001 works remotely.\n\nUnresolved questions\n- None.",
            (statement,),
            ("source_001", "source_002"),
            ValidTime("unknown"),
            TransactionTime(BASE),
            "standard",
            False,
        )
        return build_session_index_record(
            "user_001",
            AS_OF,
            value,
            config=self.config,
            embedder=self.embedder,
        )

    def _records(self):
        records = (
            self._atomic(),
            self._atomic(
                "claim_002",
                "version_002",
                "superseded",
                "office work",
                ("002",),
                relation=False,
            ),
            self._session(),
        )
        self.assertTrue(all(record is not None for record in records))
        return tuple(records)

    def _request(self, key: str = "retrieval:build:001", snapshot: str = SHA):
        return IndexBuildRequest(
            "user_001",
            "retrieval_index_v1",
            self.repository.config_sha256,
            AS_OF,
            key,
            snapshot,
            AS_OF,
            AS_OF,
        )

    def _count(self, table: str) -> int:
        return self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def test_migration_repeats_with_exact_tables_indexes_and_pgvector(self) -> None:
        self.assertEqual(self.applied[-1], "0008_retrieval_indexes.sql")
        self.assertEqual(apply_migrations(self.connection, MIGRATIONS), ())
        self.assertEqual(
            {
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public' AND tablename LIKE 'retrieval_index_%'
                    """
                ).fetchall()
            },
            {
                "retrieval_index_runs",
                "retrieval_index_records",
                "retrieval_index_claim_links",
                "retrieval_index_source_links",
                "retrieval_index_relation_links",
            },
        )
        indexes = {
            row[0]: row[1]
            for row in self.connection.execute(
                """
                SELECT indexname, indexdef FROM pg_indexes
                WHERE indexname LIKE 'retrieval_records_%'
                """
            ).fetchall()
        }
        for name in (
            "retrieval_records_atomic_fts_gin",
            "retrieval_records_session_fts_gin",
        ):
            self.assertIn("USING gin", indexes[name])
        for name in (
            "retrieval_records_atomic_vector_hnsw",
            "retrieval_records_session_vector_hnsw",
        ):
            self.assertIn("USING hnsw", indexes[name])
            self.assertIn("vector_cosine_ops", indexes[name])
        self.assertEqual(
            self.connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0],
            "0.8.6",
        )

    def test_atomic_session_lineage_exact_replay_and_drift(self) -> None:
        records = self._records()
        request = self._request()
        created = self.repository.persist(request, records)
        replayed = self.repository.persist(request, tuple(reversed(records)))
        self.assertTrue(created.created)
        self.assertTrue(replayed.replayed)
        self.assertEqual(created.record_ids, replayed.record_ids)
        self.assertEqual(self._count("retrieval_index_records"), 3)
        self.assertEqual(self._count("retrieval_index_claim_links"), 3)
        self.assertEqual(self._count("retrieval_index_source_links"), 4)
        self.assertEqual(self._count("retrieval_index_relation_links"), 1)
        kinds = self.connection.execute(
            """
            SELECT record_kind, lifecycle_statuses, vector_dims(embedding),
                   length(search_document::text) > 0
            FROM retrieval_index_records ORDER BY record_kind, index_record_id
            """
        ).fetchall()
        self.assertEqual([row[0] for row in kinds], ["atomic", "atomic", "session"])
        self.assertTrue(all(row[2:] == (256, True) for row in kinds))
        self.assertIn(["superseded"], [row[1] for row in kinds])
        with self.assertRaisesRegex(RetrievalPersistenceConflict, "idempotency_drift"):
            self.repository.persist(
                self._request(snapshot="b" * 64), records
            )

    def test_relation_lineage_respects_the_build_cutoff(self) -> None:
        future = AS_OF + timedelta(days=1)
        self.connection.execute(
            """
            INSERT INTO conflict_decisions (
                decision_id, user_id, classifier_version, rule_version,
                pair_id, left_claim_id, right_claim_id, left_version_id,
                right_version_id, input_snapshot_sha256, transaction_as_of,
                matched_rule, label, classified_at
            ) VALUES (
                'decision_002', 'user_001', 'relation_classifier_v1',
                'conflict_relation_rules_v1', 'pair_002', 'claim_001',
                'claim_002', 'version_001', 'version_002', %s, %s,
                'explicit_correction', 'explicit_correction', %s
            )
            """,
            ("b" * 64, future, future),
        )
        self.connection.execute(
            """
            INSERT INTO claim_relations (
                relation_id, user_id, decision_id, classifier_version,
                source_claim_id, target_claim_id, relation_type, confidence,
                input_snapshot_sha256, created_at
            ) VALUES (
                'relation_002', 'user_001', 'decision_002',
                'relation_classifier_v1', 'claim_002', 'claim_001',
                'corrects', 1, %s, %s
            )
            """,
            ("b" * 64, future),
        )
        inputs = _atomic_inputs(self.connection, "user_001", AS_OF)
        relation_ids = {
            item.relation_id
            for value in inputs
            for item in value.relation_lineage
        }
        self.assertIn("relation_001", relation_ids)
        self.assertNotIn("relation_002", relation_ids)

    def test_cross_user_and_database_constraints_reject_before_leakage(self) -> None:
        record = self._atomic()
        assert record is not None
        with self.assertRaisesRegex(RetrievalPersistenceConflict, "stable_id_drift"):
            self.repository.persist(
                self._request(),
                (
                    replace(
                        record,
                        source_lineage=(
                            SourceSpanLineage(
                                "user_001",
                                "claim_001",
                                "version_001",
                                "source_foreign",
                                "span_foreign",
                                "supports",
                                0,
                            ),
                        ),
                    ),
                ),
            )
        self.assertEqual(self._count("retrieval_index_runs"), 0)
        self.repository.persist(self._request("retrieval:constraints"), (record,))
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                """
                UPDATE retrieval_index_records
                SET record_kind = 'session'
                WHERE index_record_id = %s
                """,
                (record.index_record_id,),
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE retrieval_index_source_links SET user_id = 'user_002'
                WHERE index_record_id = %s
                """,
                (record.index_record_id,),
            )
        with self.assertRaises(psycopg.errors.DataException):
            self.connection.execute(
                """
                UPDATE retrieval_index_records SET embedding = '[1,2]'::vector
                WHERE index_record_id = %s
                """,
                (record.index_record_id,),
            )

    def test_concurrent_exact_build_creates_once(self) -> None:
        records = self._records()
        request = self._request("retrieval:concurrent")

        def persist_once(_: int):
            connection = psycopg.connect(DATABASE_URL, autocommit=True)
            try:
                repository = RetrievalIndexRepository(
                    connection, config_path=CONFIG_PATH
                )
                return repository.persist(request, records)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(persist_once, range(2)))
        self.assertEqual(sum(item.created for item in outcomes), 1)
        self.assertEqual(sum(item.replayed for item in outcomes), 1)
        self.assertEqual(self._count("retrieval_index_runs"), 1)
        self.assertEqual(self._count("retrieval_index_records"), 3)

    def test_failed_build_rolls_back_records_and_recovery_is_separate(self) -> None:
        valid = self._atomic()
        bad = self._atomic(source_suffixes=("missing",), relation=False)
        assert valid is not None and bad is not None
        request = self._request("retrieval:failure")
        with self.assertRaisesRegex(RetrievalPersistenceConflict, "stable_id_drift"):
            self.repository.persist(request, (valid, bad))
        self.assertEqual(self._count("retrieval_index_runs"), 0)
        self.assertEqual(self._count("retrieval_index_records"), 0)
        failed = self.repository.record_failure(request, "foreign_key_failure")
        self.assertTrue(failed.failed)
        recovered = self.repository.persist(
            self._request("retrieval:recovered", snapshot="b" * 64), (valid,)
        )
        self.assertTrue(recovered.created)
        self.assertEqual(self._count("retrieval_index_runs"), 2)
        self.assertEqual(self._count("retrieval_index_records"), 1)

    def test_index_versions_coexist_without_overwrite(self) -> None:
        record = self._atomic()
        assert record is not None
        self.repository.persist(self._request("retrieval:v1"), (record,))
        run_v2 = digest("run:v2")
        record_v2 = digest("record:v2")
        original_run = self.connection.execute(
            "SELECT run_id FROM retrieval_index_runs"
        ).fetchone()[0]
        self.connection.execute(
            """
            INSERT INTO retrieval_index_runs (
                run_id, user_id, index_version, content_renderer_version,
                embedding_version, embedding_dimension, config_sha256,
                idempotency_key, transaction_as_of, input_snapshot_sha256,
                records_snapshot_sha256, status, atomic_count, session_count,
                record_count, started_at, completed_at
            ) SELECT %s, user_id, 'retrieval_index_v2', content_renderer_version,
                     embedding_version, embedding_dimension, config_sha256,
                     'retrieval:v2', transaction_as_of, input_snapshot_sha256,
                     records_snapshot_sha256, status, 1, 0, 1,
                     started_at, completed_at
              FROM retrieval_index_runs WHERE run_id = %s
            """,
            (run_v2, original_run),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_records (
                index_record_id, user_id, run_id, index_version, record_kind,
                atomic_claim_id, atomic_claim_version_id, subject_id, speaker_id,
                predicate, content_text, content_sha256, embedding_version,
                embedding_dimension, embedding, lifecycle_statuses, memory_kind,
                epistemic_status, time_precision, valid_from_date,
                transaction_from, sensitivity, contains_sensitive,
                input_snapshot_sha256
            ) SELECT %s, user_id, %s, 'retrieval_index_v2', record_kind,
                     atomic_claim_id, atomic_claim_version_id, subject_id,
                     speaker_id, predicate, content_text, content_sha256,
                     embedding_version, embedding_dimension, embedding,
                     lifecycle_statuses, memory_kind, epistemic_status,
                     time_precision, valid_from_date, transaction_from,
                     sensitivity, contains_sensitive, input_snapshot_sha256
              FROM retrieval_index_records WHERE index_record_id = %s
            """,
            (record_v2, run_v2, record.index_record_id),
        )
        self.assertEqual(
            self.connection.execute(
                """
                SELECT index_version, count(*) FROM retrieval_index_records
                GROUP BY index_version ORDER BY index_version
                """
            ).fetchall(),
            [("retrieval_index_v1", 1), ("retrieval_index_v2", 1)],
        )

    def test_evidence_relation_and_summary_deletion_remove_stale_rows_and_rebuild(self) -> None:
        records = self._records()
        self.repository.persist(self._request("retrieval:delete"), records)
        atomic_current, atomic_old, session_record = records
        self.connection.execute(
            """
            DELETE FROM evidence_links
            WHERE user_id = 'user_001' AND claim_id = 'claim_002'
              AND span_id = 'span_002' AND support_type = 'supports'
            """
        )
        self.assertNotIn(
            atomic_old.index_record_id,
            self.repository.record_ids("user_001", "retrieval_index_v1"),
        )
        IngestionService(self.connection).delete_source(
            "user_001", "source_001", AS_OF + timedelta(hours=1)
        )
        remaining = self.repository.record_ids("user_001", "retrieval_index_v1")
        self.assertNotIn(atomic_current.index_record_id, remaining)
        self.assertNotIn(session_record.index_record_id, remaining)
        self.assertEqual(
            self.connection.execute(
                """
                SELECT count(*) FROM retrieval_index_records
                WHERE search_document @@ plainto_tsquery('simple', 'remote')
                """
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM evidence_links WHERE claim_id = 'claim_001'"
            ).fetchone()[0],
            1,
        )
        rebuilt = self._atomic(source_suffixes=("002",))
        assert rebuilt is not None
        self.repository.persist(
            self._request("retrieval:rebuild", snapshot="c" * 64), (rebuilt,)
        )
        self.assertIn(
            rebuilt.index_record_id,
            self.repository.record_ids("user_001", "retrieval_index_v1"),
        )
        self.connection.execute(
            "DELETE FROM claim_relations WHERE relation_id = 'relation_001'"
        )
        self.assertNotIn(
            rebuilt.index_record_id,
            self.repository.record_ids("user_001", "retrieval_index_v1"),
        )


if __name__ == "__main__":
    unittest.main()
