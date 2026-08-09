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

from ingestion.service import IngestionService
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    ProcessingOutboxRecord,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations
from storage.repository import StorageRepository
from summaries.durative import infer_durative_claim
from summaries.durative_contracts import (
    DurativeInferenceRequest,
    DurativePropositionPlan,
    DurativePropositionRequest,
)
from summaries.durative_repository import (
    DurativeClaimCoordinator,
    DurativeClaimRepository,
    DurativePersistenceError,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, 9, tzinfo=UTC)
AS_OF = BASE + timedelta(days=30)
SHA = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for durative integration tests",
)
class DurativeClaimPersistenceIntegrationTests(unittest.TestCase):
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
        self.storage = StorageRepository(self.connection)
        self.repository = DurativeClaimRepository(self.connection)
        for user_id in ("user_001", "user_002"):
            self.storage.insert_user(MemoryUser(user_id, BASE))
        self.storage.insert_extraction_version(
            ExtractionVersionRecord(
                "episodic_extractor",
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

    def _support(
        self,
        label: str,
        *,
        user_id: str = "user_001",
        value: str = "remote work",
        polarity: str = "positive",
        support_type: str = "supports",
        lifecycle_status: str = "confirmed",
        day: int | None = None,
        closed_interval: bool = True,
        memory_kind: str | None = "episodic",
    ) -> None:
        index = int(label[-1]) if label[-1:].isdigit() else 1
        source_id = f"source_{label}"
        span_id = f"span_{label}"
        claim_id = f"claim_{label}"
        raw = f"Evidence {label}"
        self.storage.insert_source_event(
            SourceEventRecord(
                source_id,
                user_id,
                "conversation",
                None,
                f"ingest:{user_id}:{source_id}",
                BASE + timedelta(hours=index),
                BASE,
                raw,
                [user_id],
                {},
                hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )
        )
        self.storage.insert_source_span(
            SourceSpanRecord(
                span_id,
                user_id,
                source_id,
                f"message_{label}",
                user_id,
                raw,
                0,
                len(raw),
            )
        )
        valid_day = date(2026, 1, day if day is not None else index)
        self.storage.insert_claim(
            ClaimRecord(
                claim_id,
                user_id,
                user_id,
                user_id,
                "work_preference",
                "predicate_registry_v2",
                value,
                polarity,
                "asserted",
                valid_day,
                None,
                valid_day if closed_interval else None,
                None,
                "day",
                0.9,
                memory_kind,
                "standard",
                "episodic_extractor",
            )
        )
        self.storage.insert_claim_version(
            ClaimVersionRecord(
                f"version_{label}",
                user_id,
                claim_id,
                lifecycle_status,
                BASE,
                valid_from_date=valid_day,
                valid_to_date=valid_day if closed_interval else None,
                time_precision="day",
            )
        )
        self.storage.insert_evidence_link(
            EvidenceLinkRecord(
                user_id,
                claim_id,
                span_id,
                support_type,
                0.9,
            )
        )

    def _plan(self, *, key: str = "durative:test", at: datetime = AS_OF):
        episodes = self.repository.load_visible_episodes("user_001", at)
        return infer_durative_claim(
            DurativePropositionRequest(
                "user_001",
                "user_001",
                "work_preference",
                "predicate_registry_v2",
                "remote work",
                "positive",
                at,
                key,
            ),
            episodes,
        )

    def _persist(self, planned: DurativePropositionPlan):
        request = DurativeInferenceRequest(
            planned.request.user_id,
            planned.request.transaction_as_of,
            planned.request.idempotency_key,
        )
        episodes = self.repository.load_visible_episodes(
            request.user_id, request.transaction_as_of,
        )
        return self.repository.persist(request, (planned,), episodes)

    def _event(
        self,
        label: str,
        *,
        at: datetime = AS_OF,
        event_type: str = "claims_changed",
    ) -> ProcessingOutboxRecord:
        return ProcessingOutboxRecord(
            digest(f"event:{label}"),
            "user_001",
            event_type,
            f"aggregate_{label}",
            f"event:{label}",
            {"source_id": f"source_{label}"},
            "pending",
            at,
        )

    def _count(self, table: str, where: str = "TRUE") -> int:
        return self.connection.execute(
            f"SELECT count(*) FROM {table} WHERE {where}"
        ).fetchone()[0]

    def test_migration_repeats_and_composite_ownership_is_enforced(self) -> None:
        self.assertEqual(self.applied[-1], "0007_durative_claims.sql")
        self.assertEqual(apply_migrations(self.connection, ROOT / "migrations"), ())
        self.assertEqual(
            {
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public' AND tablename LIKE 'durative_%'
                    """
                ).fetchall()
            },
            {
                "durative_inference_runs",
                "durative_inference_decisions",
                "durative_inference_evidence",
            },
        )
        self._support("own1")
        self._support("own2")
        planned = self._plan()
        self._persist(planned)
        self.assertEqual(
            {
                row[0] for row in self.connection.execute(
                    "SELECT DISTINCT role FROM durative_inference_evidence"
                ).fetchall()
            },
            {"supports"},
        )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                "UPDATE durative_inference_evidence SET role = 'ignored'"
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                UPDATE durative_inference_evidence SET user_id = 'user_002'
                WHERE user_id = 'user_001'
                """
            )

    def test_coordinator_persists_multiple_decisions_in_one_user_run(self) -> None:
        self._support("aggregate1", value="remote work")
        self._support("aggregate2", value="remote work")
        self._support("aggregate3", value="office first")
        self._support("aggregate4", value="office first")
        result = DurativeClaimCoordinator(self.connection).process(
            self._event("aggregate")
        )
        self.assertEqual(result.proposition_count, 2)
        self.assertEqual(self._count("durative_inference_runs"), 1)
        self.assertEqual(self._count("durative_inference_decisions"), 2)
        self.assertEqual(
            self.connection.execute(
                """
                SELECT decision_count, accepted_count, rejected_count
                FROM durative_inference_runs
                """
            ).fetchone(),
            (2, 2, 0),
        )
        self.assertEqual(
            self._count("durative_inference_evidence", "role = 'ignored'"),
            0,
        )

    def test_persist_creates_ordinary_candidate_lineage_and_exact_replay(self) -> None:
        self._support("persist1")
        self._support("persist2")
        planned = self._plan()
        outcome = self._persist(planned)
        self.assertEqual((outcome.created_count, outcome.replayed_count), (1, 0))
        claim_id = outcome.decisions[0].claim_id
        self.assertIsNotNone(claim_id)
        self.assertEqual(
            self.connection.execute(
                """
                SELECT memory_kind, epistemic_status, speaker_id
                FROM claims WHERE user_id = 'user_001' AND claim_id = %s
                """,
                (claim_id,),
            ).fetchone(),
            ("durative", "inferred", "memory_system"),
        )
        self.assertEqual(
            self.connection.execute(
                """
                SELECT lifecycle_status, belief_confidence
                FROM claim_versions
                WHERE user_id = 'user_001' AND claim_id = %s
                """,
                (claim_id,),
            ).fetchone(),
            ("candidate", None),
        )
        self.assertEqual(
            self._count("evidence_links", f"claim_id = '{claim_id}'"),
            2,
        )
        self.assertEqual(self._count("durative_inference_decisions"), 1)
        self.assertEqual(self._count("durative_inference_evidence"), 2)
        self.assertEqual(
            self.connection.execute(
                """
                SELECT model_version FROM extraction_versions
                WHERE version_id = (
                    SELECT extraction_version_id FROM durative_inference_runs
                    WHERE run_id = %s
                )
                """,
                (outcome.run_id,),
            ).fetchone(),
            ("deterministic",),
        )
        self.assertEqual(
            self._count("processing_outbox", "event_type = 'claims_changed'"),
            1,
        )
        replay = self._persist(planned)
        self.assertEqual((replay.created_count, replay.replayed_count), (0, 1))
        self.assertEqual(self._count("durative_inference_runs"), 1)
        self.connection.execute(
            """
            UPDATE durative_inference_evidence SET role = 'counter_evidence'
            WHERE user_id = 'user_001' AND decision_id = %s
            """,
            (planned.decision.decision_id,),
        )
        with self.assertRaisesRegex(DurativePersistenceError, "idempotency_drift"):
            self._persist(planned)

    def test_same_idempotency_key_with_new_episode_is_rejected_without_partial_write(self) -> None:
        self._support("drift1")
        self._support("drift2")
        self._persist(self._plan(key="durative:drift"))
        self._support("drift3")
        drifted = self._plan(key="durative:drift")
        with self.assertRaisesRegex(DurativePersistenceError, "idempotency_drift"):
            self._persist(drifted)
        self.assertEqual(self._count("durative_inference_runs"), 1)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 1)

    def test_added_evidence_appends_version_on_same_semantic_claim(self) -> None:
        self._support("append1", closed_interval=False)
        self._support("append2", closed_interval=False)
        first = self._persist(self._plan(key="durative:append:first"))
        self._support("append3", closed_interval=False)
        later = self._persist(
            self._plan(
                key="durative:append:later",
                at=AS_OF + timedelta(minutes=1),
            )
        )
        first_claim = first.decisions[0].claim_id
        later_claim = later.decisions[0].claim_id
        self.assertEqual(first_claim, later_claim)
        self.assertEqual(
            self._count("claim_versions", f"claim_id = '{first_claim}'"),
            2,
        )
        self.assertEqual(
            self._count(
                "claim_versions",
                f"claim_id = '{first_claim}' AND transaction_to IS NULL",
            ),
            1,
        )
        self.assertEqual(
            self._count("evidence_links", f"claim_id = '{first_claim}'"),
            3,
        )

    def test_changed_valid_interval_creates_replacement_claim(self) -> None:
        self._support("replace1", day=1)
        self._support("replace2", day=2)
        first = self._persist(self._plan(key="durative:replace:first"))
        self._support("replace3", day=3)
        later = self._persist(
            self._plan(
                key="durative:replace:later",
                at=AS_OF + timedelta(minutes=1),
            )
        )
        self.assertNotEqual(first.decisions[0].claim_id, later.decisions[0].claim_id)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 2)
        self.assertEqual(
            self._count("claim_versions", "transaction_to IS NULL"),
            5,
        )

    def test_concurrent_added_evidence_creates_one_successor(self) -> None:
        self._support("race1", closed_interval=False)
        self._support("race2", closed_interval=False)
        first = self._persist(self._plan(key="durative:race:first"))
        self._support("race3", closed_interval=False)
        at = AS_OF + timedelta(minutes=1)
        episodes = self.repository.load_visible_episodes("user_001", at)
        plans = tuple(
            infer_durative_claim(
                DurativePropositionRequest(
                    "user_001", "user_001", "work_preference",
                    "predicate_registry_v2", "remote work", "positive",
                    at, f"durative:race:{index}",
                ),
                episodes,
            )
            for index in (1, 2)
        )

        def persist(plan):
            with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
                repository = DurativeClaimRepository(connection)
                request = DurativeInferenceRequest(
                    plan.request.user_id,
                    plan.request.transaction_as_of,
                    plan.request.idempotency_key,
                )
                visible = repository.load_visible_episodes(
                    request.user_id, request.transaction_as_of,
                )
                return repository.persist(request, (plan,), visible)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(persist, plans))
        first_claim = first.decisions[0].claim_id
        self.assertEqual(
            {item.decisions[0].claim_id for item in outcomes}, {first_claim}
        )
        self.assertEqual(sum(item.created_count for item in outcomes), 2)
        self.assertEqual(
            self._count("claim_versions", f"claim_id = '{first_claim}'"),
            2,
        )

    def test_persisted_unresolved_relation_is_loaded_as_counterevidence(self) -> None:
        self._support("relation1")
        self._support("relation2")
        left_claim, right_claim = sorted(("claim_relation1", "claim_relation2"))
        versions = {
            "claim_relation1": "version_relation1",
            "claim_relation2": "version_relation2",
        }
        decision_id = digest("durative:relation:decision")
        self.connection.execute(
            """
            INSERT INTO conflict_decisions (
                decision_id, user_id, classifier_version, rule_version,
                pair_id, left_claim_id, right_claim_id, left_version_id,
                right_version_id, input_snapshot_sha256, transaction_as_of,
                matched_rule, label, classified_at
            ) VALUES (
                %s, 'user_001', 'classifier', 'rule', %s, %s, %s, %s, %s,
                %s, %s, 'temporal_change', 'temporal_change', %s
            )
            """,
            (
                decision_id, digest("durative:relation:pair"),
                left_claim, right_claim, versions[left_claim],
                versions[right_claim], digest("durative:relation:snapshot"),
                BASE, BASE,
            ),
        )
        self.connection.execute(
            """
            INSERT INTO claim_relations (
                relation_id, user_id, decision_id, classifier_version,
                source_claim_id, target_claim_id, relation_type, confidence,
                input_snapshot_sha256, created_at
            ) VALUES (%s, 'user_001', %s, 'classifier', %s, %s,
                      'corrects', 1, %s, %s)
            """,
            (
                digest("durative:relation"), decision_id, right_claim,
                left_claim, digest("durative:relation:snapshot"), BASE,
            ),
        )
        episodes = self.repository.load_visible_episodes("user_001", AS_OF)
        self.assertTrue(any(item.relation_types == ("corrects",) for item in episodes))
        planned = self._plan(key="durative:persisted-relation")
        self.assertEqual(planned.decision.reason, "counterevidence")

    def test_rejected_run_is_persisted_without_claim_version_or_outbox(self) -> None:
        self._support("single1", closed_interval=False)
        planned = self._plan()
        self.assertEqual(planned.decision.status, "rejected")
        outcome = self._persist(planned)
        self.assertEqual(outcome.created_count, 1)
        self.assertIsNone(outcome.decisions[0].claim_id)
        self.assertEqual(self._count("durative_inference_runs"), 1)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 0)
        self.assertEqual(
            self._count("processing_outbox", "event_type = 'claims_changed'"),
            0,
        )

    def test_null_memory_kind_is_counted_and_rejected_instead_of_dropped(self) -> None:
        self._support("null_kind1", memory_kind=None)
        episodes = self.repository.load_visible_episodes("user_001", AS_OF)
        self.assertEqual(len(episodes), 1)
        planned = self._plan(key="durative:null-kind")
        self.assertEqual(planned.decision.status, "rejected")
        self.assertEqual(planned.decision.reason, "memory_kind_ineligible")
        self._persist(planned)
        self.assertEqual(self._count("durative_inference_runs"), 1)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 0)

    def test_write_failure_rolls_back_claim_version_evidence_and_run(self) -> None:
        self._support("rollback1")
        self._support("rollback2")
        planned = self._plan()

        class FailingRepository(DurativeClaimRepository):
            def _insert_run(self, *args, **kwargs):
                raise RuntimeError("injected failure")

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            FailingRepository(self.connection).persist(
                DurativeInferenceRequest(
                    planned.request.user_id,
                    planned.request.transaction_as_of,
                    planned.request.idempotency_key,
                ),
                (planned,),
                self.repository.load_visible_episodes("user_001", AS_OF),
            )
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 0)
        self.assertEqual(self._count("durative_inference_runs"), 0)
        self.assertEqual(
            self._count("extraction_versions", "model_version = 'deterministic'"),
            0,
        )

    def test_coordinator_is_idempotent_and_does_not_recurse_on_durative_claims(self) -> None:
        self._support("event1")
        self._support("event2")
        coordinator = DurativeClaimCoordinator(self.connection)
        event = self._event("first")
        first = coordinator.process(event)
        self.assertEqual(
            (first.proposition_count, first.accepted_count, first.created_count),
            (1, 1, 1),
        )
        replay = coordinator.process(event)
        self.assertEqual(replay.replayed_count, 1)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 1)
        self.assertEqual(self._count("durative_inference_runs"), 1)

        later = coordinator.process(
            self._event("later", at=AS_OF + timedelta(minutes=1))
        )
        self.assertEqual((later.accepted_count, later.created_count), (1, 1))
        self.assertEqual(self._count("durative_inference_runs"), 2)
        self.assertEqual(
            self._count("processing_outbox", "event_type = 'claims_changed'"),
            1,
        )

    def test_source_deletion_purges_then_rebuilds_from_surviving_support(self) -> None:
        for label in ("delete1", "delete2", "delete3"):
            self._support(label)
        coordinator = DurativeClaimCoordinator(self.connection)
        initial = coordinator.process(self._event("initial"))
        self.assertEqual(initial.created_count, 1)
        old_claim = self.connection.execute(
            "SELECT claim_id FROM claims WHERE memory_kind = 'durative'"
        ).fetchone()[0]

        deleted_at = AS_OF + timedelta(hours=1)
        outcome = IngestionService(self.connection).delete_source(
            "user_001", "source_delete1", deleted_at
        )
        self.assertTrue(outcome.deleted)
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 0)
        row = self.connection.execute(
            """
            SELECT event_id, user_id, event_type, aggregate_id, dedupe_key,
                   payload, state, created_at, published_at
            FROM processing_outbox
            WHERE user_id = 'user_001' AND event_type = 'source_deleted'
              AND aggregate_id = 'source_delete1'
            """
        ).fetchone()
        rebuilt = coordinator.process(ProcessingOutboxRecord(*row))
        self.assertEqual((rebuilt.accepted_count, rebuilt.created_count), (1, 1))
        new_claim = self.connection.execute(
            "SELECT claim_id FROM claims WHERE memory_kind = 'durative'"
        ).fetchone()[0]
        self.assertEqual(old_claim, new_claim)
        self.assertEqual(
            self._count("evidence_links", f"claim_id = '{new_claim}'"),
            2,
        )

    def test_source_deletion_with_one_survivor_leaves_no_unsupported_claim(self) -> None:
        self._support("sole1", closed_interval=False)
        self._support("sole2", closed_interval=False)
        coordinator = DurativeClaimCoordinator(self.connection)
        coordinator.process(self._event("sole_initial"))
        deleted_at = AS_OF + timedelta(hours=1)
        IngestionService(self.connection).delete_source(
            "user_001", "source_sole1", deleted_at
        )
        row = self.connection.execute(
            """
            SELECT event_id, user_id, event_type, aggregate_id, dedupe_key,
                   payload, state, created_at, published_at
            FROM processing_outbox
            WHERE user_id = 'user_001' AND event_type = 'source_deleted'
              AND aggregate_id = 'source_sole1'
            """
        ).fetchone()
        result = coordinator.process(ProcessingOutboxRecord(*row))
        self.assertEqual((result.accepted_count, result.rejected_count), (0, 1))
        self.assertEqual(self._count("claims", "memory_kind = 'durative'"), 0)


if __name__ == "__main__":
    unittest.main()
