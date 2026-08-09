from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

import conflicts.candidates as candidate_module
from conflicts.candidates import CandidatePair, CandidateSignals
from conflicts.classifier import (
    CLASSIFIER_VERSION,
    ConflictClassifier,
    ConflictClassifierError,
    ClassificationRequest,
)
from conflicts.relations import ConflictRelationConflict, ConflictRelationService
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import MigrationError, apply_migrations
from temporal.contracts import TemporalQuery

if psycopg is not None:
    from ingestion.service import IngestionService
    from storage.repository import StorageRepository
    from temporal.service import TemporalService


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for relation integration tests",
)
class ConflictRelationIntegrationTests(unittest.TestCase):
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
        self.repository.insert_user(MemoryUser("user_001", BASE))
        self.repository.insert_user(MemoryUser("user_002", BASE))
        self.repository.insert_extraction_version(
            ExtractionVersionRecord(
                "relations_test_v1", "deterministic", "none", "0" * 64,
                "relations_test", "0" * 64, "predicate_registry_v2",
                "0" * 64, "0" * 64, BASE,
            )
        )
        self.classifier = ConflictClassifier(repo_root=ROOT)
        self.service = ConflictRelationService(self.connection, repo_root=ROOT)

    def _source(
        self,
        source_id: str,
        claim_id: str,
        *,
        user_id: str = "user_001",
        ingested_at: datetime = BASE,
    ) -> str:
        content = f"evidence for {source_id} and {claim_id}"
        self.repository.insert_source_event(
            SourceEventRecord(
                source_id, user_id, "conversation", None,
                f"key_{user_id}_{source_id}", ingested_at, ingested_at, content,
                [user_id], {}, hashlib.sha256(content.encode()).hexdigest(),
            )
        )
        span_id = f"span_{source_id}_{claim_id}"
        self.repository.insert_source_span(
            SourceSpanRecord(
                span_id, user_id, source_id, f"message_{claim_id}", user_id, content
            )
        )
        return span_id

    def _claim(
        self,
        claim_id: str,
        spans: tuple[str, ...],
        *,
        user_id: str = "user_001",
        value: str = "Delhi",
        status: str = "candidate",
    ) -> None:
        self.repository.insert_claim(
            ClaimRecord(
                claim_id, user_id, user_id, user_id, "lives_in",
                "predicate_registry_v2", value, "positive", "asserted",
                date(2026, 1, 1), None, date(2026, 12, 31), None, "day",
                1, "durative", "standard", "relations_test_v1",
            )
        )
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                f"version_{claim_id}", user_id, claim_id, status, BASE,
                valid_from_date=date(2026, 1, 1),
                valid_to_date=date(2026, 12, 31), time_precision="day",
            )
        )
        for span_id in spans:
            self.repository.insert_evidence_link(
                EvidenceLinkRecord(user_id, claim_id, span_id, "supports", 1)
            )

    def _request(
        self,
        left_id: str,
        right_id: str,
        *,
        statuses: frozenset[str] = frozenset(
            {"candidate", "confirmed", "current", "historical", "disputed", "superseded"}
        ),
    ) -> ClassificationRequest:
        visible = {
            item.claim.claim_id: item
            for item in TemporalService(self.connection).query(
                TemporalQuery("user_001", AS_OF, statuses=statuses)
            )
        }
        left, right = visible[left_id], visible[right_id]
        source_ids = tuple(
            sorted(
                {
                    evidence.source_id
                    for item in (left, right)
                    for evidence in item.evidence
                }
            )
        )
        ingested = tuple(
            self.connection.execute(
                """
                SELECT source_id, ingested_at FROM source_events
                WHERE user_id = 'user_001' AND source_id = ANY(%s)
                ORDER BY source_id
                """,
                (list(source_ids),),
            ).fetchall()
        )
        return ClassificationRequest(
            CLASSIFIER_VERSION,
            "user_001",
            AS_OF,
            CandidatePair(
                candidate_module._pair_id(
                    "candidate_linker_v1", "user_001", left_id, right_id
                ),
                "candidate_linker_v1",
                "user_001",
                left_id,
                right_id,
                CandidateSignals(True, True, (), "overlap", 0, False, 1.0),
                source_ids,
            ),
            left,
            right,
            ingested,
        )

    def _pair(self, prefix: str = "base") -> ClassificationRequest:
        left_id, right_id = f"claim_{prefix}_a", f"claim_{prefix}_b"
        self._claim(left_id, (self._source(f"{prefix}_a", left_id),), value="Delhi")
        self._claim(right_id, (self._source(f"{prefix}_b", right_id),), value="Mumbai")
        return self._request(left_id, right_id)

    def test_clean_repeat_migration_and_composite_database_constraints(self) -> None:
        self.assertEqual(self.applied[-1], "0005_belief_resolution.sql")
        self.assertEqual(apply_migrations(self.connection, ROOT / "migrations"), ())
        tables = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public' AND tablename LIKE 'conflict_%'
                """
            ).fetchall()
        }
        self.assertEqual(
            tables, {"conflict_decisions", "conflict_decision_evidence"}
        )
        request = self._pair()
        decision = self.classifier.classify(request)
        stored = self.service.persist(request, decision, AS_OF + timedelta(days=1))
        for relation_type in (
            "supports", "supersedes", "same_event_as", "caused_by", "hindered_by"
        ):
            self.connection.execute(
                """
                INSERT INTO claim_relations (
                    relation_id, user_id, decision_id, classifier_version,
                    source_claim_id, target_claim_id, relation_type, confidence,
                    input_snapshot_sha256, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s)
                """,
                (
                    f"ontology_{relation_type}", request.user_id,
                    decision.decision_id, decision.classifier_version,
                    request.left.claim.claim_id, request.right.claim.claim_id,
                    relation_type, decision.input_snapshot_sha256,
                    AS_OF + timedelta(days=1),
                ),
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                """
                INSERT INTO claim_relations (
                    relation_id, user_id, decision_id, classifier_version,
                    source_claim_id, target_claim_id, relation_type, confidence,
                    input_snapshot_sha256, created_at
                ) VALUES ('reverse_symmetric', %s, %s, %s, %s, %s,
                          'same_event_as', 1, %s, %s)
                """,
                (
                    request.user_id, decision.decision_id,
                    decision.classifier_version, request.right.claim.claim_id,
                    request.left.claim.claim_id, decision.input_snapshot_sha256,
                    AS_OF + timedelta(days=1),
                ),
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                INSERT INTO conflict_decision_evidence (
                    decision_evidence_id, user_id, decision_id, claim_id, span_id,
                    support_type, input_snapshot_sha256
                ) VALUES ('cross', 'user_002', %s, %s, %s, 'supports', %s)
                """,
                (
                    decision.decision_id,
                    request.left.claim.claim_id,
                    request.left.evidence[0].span_id,
                    decision.input_snapshot_sha256,
                ),
            )
        self.assertEqual(stored.decision.user_id, "user_001")

    def test_migration_checksum_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "0004_conflict_relations.sql"
            changed.write_text("SELECT 1;\n", encoding="utf-8")
            with self.assertRaisesRegex(MigrationError, "checksum changed"):
                apply_migrations(self.connection, directory)

    def test_atomic_persist_exact_replay_drift_and_no_lifecycle_mutation(self) -> None:
        request = self._pair()
        decision = self.classifier.classify(request)
        before = tuple(
            self.connection.execute(
                "SELECT version_id, lifecycle_status FROM claim_versions ORDER BY version_id"
            ).fetchall()
        )
        classified_at = AS_OF + timedelta(days=1)
        first = self.service.persist(request, decision, classified_at)
        replay = self.service.persist(request, decision, classified_at)
        self.assertFalse(first.replayed)
        self.assertTrue(replay.replayed)
        self.assertEqual(len(first.relations), 1)
        self.assertEqual(len(first.evidence), 2)
        with self.assertRaisesRegex(ConflictRelationConflict, "stable_id_drift"):
            self.service.persist(request, decision, classified_at + timedelta(seconds=1))
        after = tuple(
            self.connection.execute(
                "SELECT version_id, lifecycle_status FROM claim_versions ORDER BY version_id"
            ).fetchall()
        )
        self.assertEqual(after, before)

        rollback = self._pair("rollback")
        rollback_decision = self.classifier.classify(rollback)
        with mock.patch.object(
            self.service._repository,
            "insert_claim_relation",
            side_effect=RuntimeError("forced"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                self.service.persist(rollback, rollback_decision, classified_at)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM conflict_decisions WHERE decision_id = %s",
                (rollback_decision.decision_id,),
            ).fetchone()[0],
            0,
        )

    def test_future_excluded_and_deleted_inputs_cannot_persist(self) -> None:
        request = self._pair()
        future = ClassificationRequest(
            request.classifier_version, request.user_id, request.transaction_as_of,
            request.pair, request.left, request.right,
            tuple((source_id, AS_OF + timedelta(seconds=1)) for source_id in request.pair.source_ids),
        )
        with self.assertRaisesRegex(ConflictClassifierError, "future evidence"):
            self.classifier.classify(future)

        excluded_id = "claim_excluded"
        self._claim(
            excluded_id,
            (self._source("excluded", excluded_id),),
            value="Pune",
            status="excluded",
        )
        excluded_request = self._request(
            request.left.claim.claim_id,
            excluded_id,
            statuses=frozenset({"candidate", "excluded"}),
        )
        with self.assertRaisesRegex(ConflictClassifierError, "not classifier-visible"):
            self.classifier.classify(excluded_request)

        decision = self.classifier.classify(request)
        self.service.persist(request, decision, AS_OF + timedelta(days=1))
        IngestionService(self.connection).delete_source(
            "user_001", request.left.evidence[0].source_id, AS_OF + timedelta(days=2)
        )
        with self.assertRaisesRegex(ConflictRelationConflict, "source_snapshot_drift"):
            self.service.persist(request, decision, AS_OF + timedelta(days=1))

    def test_deletion_removes_decision_and_recomputes_only_surviving_pair(self) -> None:
        left_id, right_id = "claim_keep_a", "claim_keep_b"
        removed_span = self._source("removed", left_id)
        left_keep = self._source("left_keep", left_id)
        right_keep = self._source("right_keep", right_id)
        self._claim(left_id, (removed_span, left_keep), value="Delhi")
        self._claim(right_id, (right_keep,), value="Mumbai")
        request = self._request(left_id, right_id)
        decision = self.classifier.classify(request)
        self.service.persist(request, decision, AS_OF + timedelta(days=1))
        versions_before = self.connection.execute(
            "SELECT version_id, lifecycle_status FROM claim_versions WHERE claim_id = ANY(%s) ORDER BY version_id",
            ([left_id, right_id],),
        ).fetchall()
        IngestionService(self.connection).delete_source(
            "user_001", "removed", AS_OF + timedelta(days=2)
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM conflict_decisions WHERE decision_id = %s",
                (decision.decision_id,),
            ).fetchone()[0],
            0,
        )
        event = self.connection.execute(
            """
            SELECT payload FROM processing_outbox
            WHERE event_type = 'conflict_recompute_required'
              AND aggregate_id = %s
            """,
            (request.pair.pair_id,),
        ).fetchone()[0]
        self.assertEqual(
            set(event),
            {"pair_id", "left_claim_id", "right_claim_id", "deleted_source_id"},
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT version_id, lifecycle_status FROM claim_versions WHERE claim_id = ANY(%s) ORDER BY version_id",
                ([left_id, right_id],),
            ).fetchall(),
            versions_before,
        )

        sole = self._pair("sole")
        sole_decision = self.classifier.classify(sole)
        self.service.persist(sole, sole_decision, AS_OF + timedelta(days=1))
        IngestionService(self.connection).delete_source(
            "user_001", sole.left.evidence[0].source_id, AS_OF + timedelta(days=3)
        )
        self.assertEqual(
            self.connection.execute(
                """
                SELECT count(*) FROM processing_outbox
                WHERE event_type = 'conflict_recompute_required'
                  AND aggregate_id = %s
                """,
                (sole.pair.pair_id,),
            ).fetchone()[0],
            0,
        )
        self.assertIsNone(
            self.repository.get_claim("user_001", sole.left.claim.claim_id)
        )


if __name__ == "__main__":
    unittest.main()
