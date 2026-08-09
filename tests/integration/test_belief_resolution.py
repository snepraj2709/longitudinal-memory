from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
from threading import Barrier
import unittest
from unittest import mock

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

import conflicts.candidates as candidate_module
from conflicts.candidates import CandidatePair, CandidateSignals
from conflicts.classifier import CLASSIFIER_VERSION, ClassificationRequest, ConflictClassifier
from conflicts.relations import ConflictRelationService
from conflicts.resolution import BeliefResolutionConflict, BeliefResolutionService
from conflicts.resolver import RESOLVER_VERSION, ResolutionRequest
from ingestion.service import IngestionService
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
from storage.repository import StorageConflictError, StorageRepository
from temporal.contracts import TemporalQuery
from temporal.service import TemporalService


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
RESOLVED = datetime(2026, 6, 2, tzinfo=UTC)


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for resolver integration tests",
)
class BeliefResolutionIntegrationTests(unittest.TestCase):
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
        for user_id in ("user_001", "user_002"):
            self.repository.insert_user(MemoryUser(user_id, BASE))
        self.repository.insert_extraction_version(
            ExtractionVersionRecord(
                "resolver_test_v1", "deterministic", "none", "0" * 64,
                "resolver_test", "0" * 64, "predicate_registry_v2",
                "0" * 64, "0" * 64, BASE,
            )
        )
        self.classifier = ConflictClassifier(repo_root=ROOT)
        self.relations = ConflictRelationService(self.connection, repo_root=ROOT)
        self.resolver = BeliefResolutionService(self.connection, repo_root=ROOT)

    def _source(
        self,
        source_id: str,
        claim_id: str,
        *,
        source_type: str = "conversation",
        user_id: str = "user_001",
        ingested_at: datetime = BASE,
    ) -> str:
        content = f"evidence for {source_id} and {claim_id}"
        self.repository.insert_source_event(
            SourceEventRecord(
                source_id, user_id, source_type, None,
                f"key_{user_id}_{source_id}", ingested_at, ingested_at,
                content, [user_id], {}, hashlib.sha256(content.encode()).hexdigest(),
            )
        )
        span_id = f"span_{source_id}_{claim_id}"
        self.repository.insert_source_span(
            SourceSpanRecord(
                span_id, user_id, source_id, f"message_{claim_id}",
                user_id, content,
            )
        )
        return span_id

    def _claim(
        self,
        claim_id: str,
        spans: tuple[str, ...],
        *,
        value: object = "Delhi",
        predicate: str = "lives_in",
        epistemic_status: str = "asserted",
        valid_from: date = date(2026, 1, 1),
        valid_to: date = date(2026, 12, 31),
        user_id: str = "user_001",
        status: str = "candidate",
    ) -> None:
        self.repository.insert_claim(
            ClaimRecord(
                claim_id, user_id, user_id, user_id, predicate,
                "predicate_registry_v2", value, "positive", epistemic_status,
                valid_from, None, valid_to, None, "day", 1, "durative",
                "standard", "resolver_test_v1",
            )
        )
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                f"version_{claim_id}", user_id, claim_id, status, BASE,
                valid_from_date=valid_from, valid_to_date=valid_to,
                time_precision="day",
            )
        )
        for span_id in spans:
            self.repository.insert_evidence_link(
                EvidenceLinkRecord(user_id, claim_id, span_id, "supports", 1)
            )

    def _decision(
        self,
        left_id: str,
        right_id: str,
        *,
        target: str | None = None,
        cutoff: datetime = AS_OF,
    ) -> str:
        visible = {
            item.claim.claim_id: item
            for item in TemporalService(self.connection).query(
                TemporalQuery(
                    "user_001",
                    cutoff,
                    statuses=frozenset(
                        {"candidate", "confirmed", "current", "historical", "disputed", "superseded"}
                    ),
                )
            )
        }
        left, right = visible[left_id], visible[right_id]
        sources = tuple(
            sorted(
                {
                    value.source_id
                    for item in (left, right)
                    for value in item.evidence
                }
            )
        )
        ingestion = tuple(
            self.connection.execute(
                """
                SELECT source_id, ingested_at FROM source_events
                WHERE user_id = 'user_001' AND source_id = ANY(%s)
                ORDER BY source_id
                """,
                (list(sources),),
            ).fetchall()
        )
        request = ClassificationRequest(
            CLASSIFIER_VERSION,
            "user_001",
            cutoff,
            CandidatePair(
                candidate_module._pair_id(
                    "candidate_linker_v1", "user_001", left_id, right_id
                ),
                "candidate_linker_v1",
                "user_001",
                left_id,
                right_id,
                CandidateSignals(True, True, (), "overlap", 0, False, 1),
                sources,
            ),
            left,
            right,
            ingestion,
            target,
        )
        decision = self.classifier.classify(request)
        self.relations.persist(request, decision, cutoff)
        return decision.decision_id

    def _pair(
        self,
        prefix: str,
        *,
        label: str = "hard",
        extra_left_source: bool = False,
        left_source_type: str = "conversation",
    ) -> tuple[str, str, str, str]:
        left_id, right_id = f"claim_{prefix}_a", f"claim_{prefix}_b"
        left_source = f"source_{prefix}_a"
        left_spans = [
            self._source(
                left_source, left_id, source_type=left_source_type
            )
        ]
        if extra_left_source:
            left_spans.append(self._source(f"source_{prefix}_a2", left_id))
        right_source = f"source_{prefix}_b"
        right_span = self._source(right_source, right_id)
        if label == "temporal":
            self._claim(
                left_id, tuple(left_spans), value="Delhi",
                valid_from=date(2026, 1, 1), valid_to=date(2026, 3, 31),
            )
            self._claim(
                right_id, (right_span,), value="Mumbai",
                valid_from=date(2026, 4, 1), valid_to=date(2026, 12, 31),
            )
        elif label == "correction":
            self._claim(
                left_id, tuple(left_spans), value="Delhi", status="confirmed"
            )
            self._claim(
                right_id, (right_span,), value="Mumbai",
                epistemic_status="corrected",
            )
        elif label == "unrelated":
            self._claim(
                left_id, tuple(left_spans), predicate="work_preference", value="remote"
            )
            self._claim(
                right_id, (right_span,), predicate="work_preference", value="quiet office"
            )
        else:
            self._claim(left_id, tuple(left_spans), value="Delhi")
            self._claim(right_id, (right_span,), value="Mumbai")
        target = left_id if label == "correction" else None
        decision_id = self._decision(left_id, right_id, target=target)
        return decision_id, left_id, right_id, left_source

    def _request(
        self,
        decision_id: str,
        *,
        key: str = "resolve_1",
        transaction_as_of: datetime = AS_OF,
        valid_at: date | datetime | None = date(2026, 6, 1),
        resolved_at: datetime = RESOLVED,
    ) -> ResolutionRequest:
        return ResolutionRequest(
            "user_001", decision_id, transaction_as_of, valid_at,
            resolved_at, key, RESOLVER_VERSION,
        )

    def _open_status(self, claim_id: str) -> str:
        return self.connection.execute(
            """
            SELECT lifecycle_status FROM claim_versions
            WHERE user_id = 'user_001' AND claim_id = %s
              AND transaction_to IS NULL
            """,
            (claim_id,),
        ).fetchone()[0]

    def _count(self, table: str) -> int:
        return self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def _two_decision_chain(
        self, prefix: str
    ) -> tuple[str, str, str, str, str, str, object]:
        left_id = f"claim_{prefix}_a"
        first_right_id = f"claim_{prefix}_b"
        second_right_id = f"claim_{prefix}_c"
        left_span = self._source(f"source_{prefix}_a", left_id)
        first_right_span = self._source(
            f"source_{prefix}_b", first_right_id
        )
        self._claim(left_id, (left_span,), value="Delhi")
        self._claim(first_right_id, (first_right_span,), value="Mumbai")
        surviving_decision = self._decision(left_id, first_right_id)
        first_resolution = self.resolver.resolve(
            self._request(
                surviving_decision,
                key=f"{prefix}_surviving",
                resolved_at=AS_OF + timedelta(days=1),
            )
        )
        deleted_source = f"source_{prefix}_deleted"
        deleted_span = self._source(
            deleted_source,
            left_id,
            ingested_at=AS_OF + timedelta(days=1, hours=1),
        )
        self.repository.insert_evidence_link(
            EvidenceLinkRecord(
                "user_001", left_id, deleted_span, "supports", 1
            )
        )
        second_right_span = self._source(
            f"source_{prefix}_c", second_right_id
        )
        self._claim(second_right_id, (second_right_span,), value="Pune")
        invalidated_decision = self._decision(
            left_id,
            second_right_id,
            cutoff=AS_OF + timedelta(days=2),
        )
        self.resolver.resolve(
            self._request(
                invalidated_decision,
                key=f"{prefix}_invalidated",
                transaction_as_of=AS_OF + timedelta(days=2),
                resolved_at=AS_OF + timedelta(days=3),
            )
        )
        return (
            left_id,
            first_right_id,
            second_right_id,
            deleted_source,
            surviving_decision,
            invalidated_decision,
            first_resolution,
        )

    def test_migration_repeat_constraints_and_composite_ownership(self) -> None:
        self.assertEqual(self.applied[-1], "0006_session_summaries.sql")
        self.assertEqual(apply_migrations(self.connection, ROOT / "migrations"), ())
        tables = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public' AND tablename LIKE 'belief_resolution%'
                ORDER BY tablename
                """
            ).fetchall()
        }
        self.assertEqual(
            tables,
            {
                "belief_resolutions", "belief_resolution_actions",
                "belief_resolution_evidence",
            },
        )
        decision_id, _, _, _ = self._pair("constraints", label="unrelated")
        stored = self.resolver.resolve(self._request(decision_id))
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                INSERT INTO belief_resolution_evidence (
                    user_id, resolution_id, decision_id, decision_evidence_id
                ) VALUES ('user_002', %s, %s, %s)
                """,
                (
                    stored.resolution.resolution_id,
                    decision_id,
                    stored.evidence[0].decision_evidence_id,
                ),
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                "UPDATE belief_resolutions SET belief_confidence = 0.5 WHERE resolution_id = %s",
                (stored.resolution.resolution_id,),
            )

    def test_temporal_change_executes_atomically_and_candidate_becomes_historical(self) -> None:
        decision_id, left_id, right_id, _ = self._pair("temporal", label="temporal")
        stored = self.resolver.resolve(self._request(decision_id, key="temporal"))
        self.assertFalse(stored.replayed)
        self.assertEqual(self._open_status(left_id), "historical")
        self.assertEqual(self._open_status(right_id), "current")
        self.assertEqual([value.action_order for value in stored.actions], [1, 2])
        self.assertEqual(stored.relations, ())
        self.assertEqual(stored.resolution.belief_confidence, None)
        self.assertEqual(self._count("belief_resolution_evidence"), 2)
        payload = self.connection.execute(
            "SELECT payload FROM processing_outbox WHERE event_type = 'belief_resolved'"
        ).fetchone()[0]
        self.assertNotIn("outcome", payload)
        self.assertNotIn("reason", payload)

    def test_correction_persists_resolver_supersedes_and_replays_exactly(self) -> None:
        decision_id, left_id, right_id, _ = self._pair("correction", label="correction")
        request = self._request(decision_id, key="correction")
        first = self.resolver.resolve(request)
        replay = self.resolver.resolve(request)
        self.assertTrue(replay.replayed)
        self.assertEqual(first.resolution, replay.resolution)
        self.assertEqual(self._open_status(left_id), "superseded")
        self.assertEqual(self._open_status(right_id), "current")
        self.assertEqual(len(first.relations), 1)
        self.assertEqual(first.relations[0].relation_type, "supersedes")
        self.assertEqual(first.relations[0].resolver_version, RESOLVER_VERSION)
        self.assertEqual(first.relations[0].resolution_id, first.resolution.resolution_id)
        self.assertEqual(self._count("belief_resolutions"), 1)
        with self.assertRaisesRegex(BeliefResolutionConflict, "idempotency_drift"):
            self.resolver.resolve(
                self._request(
                    decision_id, key="correction", valid_at=date(2026, 7, 1)
                )
            )

    def test_unrelated_persists_no_change_without_lifecycle_mutation(self) -> None:
        decision_id, left_id, right_id, _ = self._pair("unrelated", label="unrelated")
        stored = self.resolver.resolve(self._request(decision_id, key="unrelated"))
        self.assertEqual(stored.resolution.outcome, "no_change")
        self.assertEqual(stored.actions, ())
        self.assertEqual(self._open_status(left_id), "candidate")
        self.assertEqual(self._open_status(right_id), "candidate")

    def test_failure_rolls_back_resolution_actions_versions_and_outbox(self) -> None:
        decision_id, left_id, right_id, _ = self._pair("rollback", label="temporal")
        before_versions = self._count("claim_versions")
        with mock.patch.object(
            self.resolver._repository,
            "insert_belief_resolution_action",
            side_effect=StorageConflictError("forced"),
        ):
            with self.assertRaisesRegex(BeliefResolutionConflict, "stable_id_drift"):
                self.resolver.resolve(self._request(decision_id, key="rollback"))
        self.assertEqual(self._count("belief_resolutions"), 0)
        self.assertEqual(self._count("belief_resolution_actions"), 0)
        self.assertEqual(self._count("claim_versions"), before_versions)
        self.assertEqual(self._open_status(left_id), "candidate")
        self.assertEqual(self._open_status(right_id), "candidate")
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM processing_outbox WHERE event_type = 'belief_resolved'"
            ).fetchone()[0],
            0,
        )

    def test_concurrent_same_key_creates_once_and_replays_once(self) -> None:
        decision_id, _, _, _ = self._pair("concurrent", label="correction")
        request = self._request(decision_id, key="concurrent")
        barrier = Barrier(2)

        def resolve_once(_: int) -> bool:
            connection = psycopg.connect(DATABASE_URL, autocommit=True)
            try:
                barrier.wait()
                return BeliefResolutionService(
                    connection, repo_root=ROOT
                ).resolve(request).replayed
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(resolve_once, range(2)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(self._count("belief_resolutions"), 1)

    def test_user_and_cutoff_fail_closed_before_writes(self) -> None:
        decision_id, _, _, _ = self._pair("scope", label="unrelated")
        with self.assertRaisesRegex(Exception, "decision_not_found"):
            self.resolver.resolve(
                ResolutionRequest(
                    "user_002", decision_id, AS_OF, date(2026, 6, 1),
                    RESOLVED, "wrong_user", RESOLVER_VERSION,
                )
            )
        with self.assertRaisesRegex(BeliefResolutionConflict, "future_decision"):
            self.resolver.resolve(
                self._request(
                    decision_id,
                    key="early",
                    transaction_as_of=AS_OF - timedelta(seconds=1),
                )
            )
        self.assertEqual(self._count("belief_resolutions"), 0)

    def test_delete_invalidates_resolution_reverses_suffix_and_preserves_survivors(self) -> None:
        decision_id, left_id, right_id, deleted_source = self._pair(
            "delete", extra_left_source=True
        )
        stored = self.resolver.resolve(self._request(decision_id, key="delete"))
        self.assertEqual(
            {self._open_status(left_id), self._open_status(right_id)}, {"disputed"}
        )
        deleted = IngestionService(self.connection).delete_source(
            "user_001", deleted_source, RESOLVED + timedelta(days=1)
        )
        self.assertTrue(deleted.deleted)
        self.assertEqual(self._open_status(left_id), "candidate")
        self.assertEqual(self._open_status(right_id), "candidate")
        self.assertEqual(self._count("belief_resolutions"), 0)
        self.assertEqual(self._count("belief_resolution_actions"), 0)
        self.assertIsNone(
            self.repository.get_conflict_decision("user_001", decision_id)
        )
        self.assertIsNotNone(self.repository.get_claim("user_001", left_id))
        self.assertIsNotNone(self.repository.get_claim("user_001", right_id))
        recompute = self.connection.execute(
            """
            SELECT payload FROM processing_outbox
            WHERE event_type = 'conflict_recompute_required'
            """
        ).fetchone()[0]
        self.assertEqual(
            set(recompute),
            {"pair_id", "left_claim_id", "right_claim_id", "deleted_source_id"},
        )
        repeated = IngestionService(self.connection).delete_source(
            "user_001", deleted_source, RESOLVED + timedelta(days=2)
        )
        self.assertFalse(repeated.deleted)
        self.assertNotEqual(stored.resolution.resolution_id, "")

    def test_delete_rewinds_to_baseline_and_replays_surviving_decision_in_order(self) -> None:
        (
            left_id,
            first_right_id,
            second_right_id,
            deleted_source,
            surviving_decision,
            invalidated_decision,
            original,
        ) = self._two_decision_chain("replay")
        deleted = IngestionService(self.connection).delete_source(
            "user_001", deleted_source, AS_OF + timedelta(days=4)
        )
        self.assertTrue(deleted.deleted)
        replayed = self.repository.get_belief_resolution(
            "user_001", original.resolution.resolution_id
        )
        self.assertEqual(replayed, original.resolution)
        self.assertIsNotNone(
            self.repository.get_conflict_decision(
                "user_001", surviving_decision
            )
        )
        self.assertIsNone(
            self.repository.get_conflict_decision(
                "user_001", invalidated_decision
            )
        )
        self.assertEqual(self._open_status(left_id), "disputed")
        self.assertEqual(self._open_status(first_right_id), "disputed")
        self.assertEqual(self._open_status(second_right_id), "candidate")
        self.assertEqual(self._count("belief_resolutions"), 1)
        unsupported = self.connection.execute(
            """
            SELECT count(*)
            FROM claim_versions AS version
            WHERE version.user_id = 'user_001'
              AND version.transaction_to IS NULL
              AND version.lifecycle_status IN (
                  'current', 'disputed', 'superseded'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM evidence_links AS evidence
                  WHERE evidence.user_id = version.user_id
                    AND evidence.claim_id = version.claim_id
              )
            """
        ).fetchone()[0]
        self.assertEqual(unsupported, 0)
        repeated = IngestionService(self.connection).delete_source(
            "user_001", deleted_source, AS_OF + timedelta(days=5)
        )
        self.assertFalse(repeated.deleted)
        self.assertEqual(self._count("belief_resolutions"), 1)

    def test_delete_replay_failure_rolls_back_and_retry_is_deterministic(self) -> None:
        (
            left_id,
            first_right_id,
            second_right_id,
            deleted_source,
            _,
            _,
            original,
        ) = self._two_decision_chain("replay_rollback")
        with mock.patch(
            "conflicts.resolution.BeliefResolutionService.resolve_in_transaction",
            side_effect=RuntimeError("forced replay failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced replay failure"):
                IngestionService(self.connection).delete_source(
                    "user_001", deleted_source, AS_OF + timedelta(days=4)
                )
        self.assertIsNotNone(
            self.repository.get_source_event("user_001", deleted_source)
        )
        self.assertIsNone(
            self.repository.get_tombstone("user_001", deleted_source)
        )
        self.assertEqual(self._count("belief_resolutions"), 2)
        self.assertEqual(self._open_status(left_id), "disputed")
        self.assertEqual(self._open_status(first_right_id), "disputed")
        self.assertEqual(self._open_status(second_right_id), "disputed")
        result = IngestionService(self.connection).delete_source(
            "user_001", deleted_source, AS_OF + timedelta(days=4)
        )
        self.assertTrue(result.deleted)
        self.assertEqual(
            self.repository.get_belief_resolution(
                "user_001", original.resolution.resolution_id
            ),
            original.resolution,
        )


if __name__ == "__main__":
    unittest.main()
