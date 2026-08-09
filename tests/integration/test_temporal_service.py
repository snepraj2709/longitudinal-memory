from __future__ import annotations

from datetime import date, datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

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
from temporal.contracts import CorrectionRequest, TemporalQuery, TransitionRequest

if psycopg is not None:
    from ingestion.service import IngestionService
    from storage.repository import StorageRepository
    from temporal.service import TemporalConflict, TemporalError, TemporalService


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)
SHA = "a" * 64


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for temporal integration tests",
)
class TemporalServiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        self.assertEqual(
            apply_migrations(self.connection, REPO_ROOT / "migrations"),
            (
                "0001_phase4_storage.sql",
                "0002_ingestion_reprocessing.sql",
                "0003_temporal_lifecycle.sql",
                "0004_conflict_relations.sql",
                "0005_belief_resolution.sql",
                "0006_session_summaries.sql",
                "0007_durative_claims.sql",
            ),
        )
        self.repository = StorageRepository(self.connection)
        self.service = TemporalService(self.connection)
        self.repository.insert_user(MemoryUser("user_1", BASE))
        self.repository.insert_user(MemoryUser("user_2", BASE))
        self.repository.insert_extraction_version(
            ExtractionVersionRecord(
                "extractor", "model", "prompt", SHA, "schema", SHA,
                "registry", SHA, SHA, BASE,
            )
        )

    def test_migration_repeats_and_enforces_nonoverlap_and_outbox_type(self) -> None:
        self.assertEqual(apply_migrations(self.connection, REPO_ROOT / "migrations"), ())
        extensions = {
            row[0]
            for row in self.connection.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'btree_gist')"
            ).fetchall()
        }
        self.assertEqual(extensions, {"vector", "btree_gist"})
        self._graph("claim_1")
        with self.assertRaises(psycopg.errors.ExclusionViolation):
            self.repository.insert_claim_version(
                ClaimVersionRecord(
                    "overlap", "user_1", "claim_1", "candidate", BASE,
                    BASE.replace(day=3), valid_from_date=date(2026, 1, 1),
                    time_precision="day",
                )
            )
        self._graph("claim_2")
        transition = self.service.transition(
            TransitionRequest(
                "user_1", "claim_2", "claim-2-confirmed", "confirmed",
                "accepted", BASE.replace(day=3),
            )
        ).transition
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute(
                """
                INSERT INTO lifecycle_transitions (
                    transition_id, user_id, idempotency_key, claim_id,
                    from_version_id, to_version_id, target_status, reason,
                    replacement_claim_id, transitioned_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
                """,
                (
                    "wrong-claim-transition", "user_1", "wrong-claim-key",
                    "claim_1", transition.from_version_id,
                    transition.to_version_id, "confirmed", "invalid",
                    BASE.replace(day=3),
                ),
            )

    def test_incremental_migration_backfills_a_promoted_version_snapshot(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        with tempfile.TemporaryDirectory() as temporary:
            migration_directory = Path(temporary)
            for name in (
                "0001_phase4_storage.sql",
                "0002_ingestion_reprocessing.sql",
            ):
                (migration_directory / name).write_bytes(
                    (REPO_ROOT / "migrations" / name).read_bytes()
                )
            self.assertEqual(
                apply_migrations(self.connection, migration_directory),
                ("0001_phase4_storage.sql", "0002_ingestion_reprocessing.sql"),
            )
        self.connection.execute(
            "INSERT INTO memory_users (user_id, created_at) VALUES (%s, %s)",
            ("user_1", BASE),
        )
        self.connection.execute(
            """
            INSERT INTO extraction_versions (
                version_id, model_version, prompt_version, prompt_hash,
                schema_version, schema_hash, registry_version, registry_hash,
                input_manifest_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            ("extractor", "model", "prompt", SHA, "schema", SHA,
             "registry", SHA, SHA, BASE),
        )
        self.connection.execute(
            """
            INSERT INTO claims (
                claim_id, user_id, subject_id, speaker_id, predicate,
                predicate_registry_version, object_json, polarity,
                epistemic_status, valid_from_date, valid_to_date,
                time_precision, extraction_confidence, memory_kind,
                sensitivity, extraction_version_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, '"engineer"'::jsonb, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                "claim_1", "user_1", "user_1", "user_1", "role",
                "registry", "positive", "asserted", date(2026, 1, 1),
                date(2026, 1, 31), "day", 0.8, "episodic", "standard",
                "extractor",
            ),
        )
        self.connection.execute(
            """
            INSERT INTO claim_versions (
                version_id, user_id, claim_id, lifecycle_status,
                transaction_from, belief_confidence
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            ("version_1", "user_1", "claim_1", "current", BASE, 0.9),
        )

        self.assertEqual(
            apply_migrations(self.connection, REPO_ROOT / "migrations"),
            (
                "0003_temporal_lifecycle.sql",
                "0004_conflict_relations.sql",
                "0005_belief_resolution.sql",
                "0006_session_summaries.sql",
                "0007_durative_claims.sql",
            ),
        )
        self.assertEqual(
            self.connection.execute(
                """
                SELECT lifecycle_status, belief_confidence, valid_from_date,
                       valid_to_date, time_precision
                FROM claim_versions
                WHERE user_id = 'user_1' AND version_id = 'version_1'
                """
            ).fetchone(),
            ("current", 0.9, date(2026, 1, 1), date(2026, 1, 31), "day"),
        )

    def test_transition_preserves_snapshot_replays_and_rejects_drift_terminal_and_stale(self) -> None:
        original = self._graph("claim_1", valid_to=date(2026, 1, 31))
        request = TransitionRequest(
            "user_1", "claim_1", "promote", "current", "accepted evidence",
            BASE.replace(day=3), 0.9,
        )
        result = self.service.transition(request)
        self.assertEqual(result.version.lifecycle_status, "current")
        self.assertEqual(result.version.valid_from_date, original.valid_from_date)
        self.assertEqual(result.version.valid_to_date, original.valid_to_date)
        self.assertEqual(self.service.transition(request).replayed, True)
        with self.assertRaisesRegex(TemporalConflict, "idempotency_drift"):
            self.service.transition(
                TransitionRequest(
                    "user_1", "claim_1", "promote", "current", "different",
                    BASE.replace(day=3), 0.9,
                )
            )
        ended = self.service.transition(
            TransitionRequest(
                "user_1", "claim_1", "end", "historical", "normal ending",
                BASE.replace(day=4), 0.85,
            )
        )
        self.assertEqual(ended.version.lifecycle_status, "historical")
        with self.assertRaisesRegex(TemporalConflict, "stale_transition"):
            self.service.transition(
                TransitionRequest(
                    "user_1", "claim_1", "stale", "excluded", "stale",
                    BASE.replace(day=4), 0.8,
                )
            )
        excluded = self.service.transition(
            TransitionRequest(
                "user_1", "claim_1", "exclude", "excluded", "policy",
                BASE.replace(day=5), 0.1,
            )
        )
        self.assertEqual(excluded.version.lifecycle_status, "excluded")
        with self.assertRaisesRegex(TemporalConflict, "transition_not_allowed"):
            self.service.transition(
                TransitionRequest(
                    "user_1", "claim_1", "terminal", "disputed", "late",
                    BASE.replace(day=6), 0.2,
                )
            )

    def test_unknown_and_null_kind_candidates_are_not_promoted(self) -> None:
        self._graph("weak", memory_kind=None, precision="unknown", valid_from=None)
        with self.assertRaisesRegex(TemporalConflict, "memory_kind_required"):
            self.service.transition(
                TransitionRequest(
                    "user_1", "weak", "weak-promotion", "confirmed", "reviewed",
                    BASE.replace(day=3),
                )
            )
        self.assertEqual(
            self.connection.execute(
                "SELECT lifecycle_status FROM claim_versions WHERE claim_id = 'weak'"
            ).fetchone()[0],
            "candidate",
        )

    def test_correction_is_atomic_user_scoped_and_names_both_claims(self) -> None:
        self._graph("old", valid_to=date(2026, 1, 31))
        self._graph("new", valid_to=date(2026, 2, 28))
        self.service.transition(
            TransitionRequest(
                "user_1", "old", "old-current", "current", "accepted",
                BASE.replace(day=3), 0.8,
            )
        )
        request = CorrectionRequest(
            "user_1", "old", "new", "correction", "confirmed",
            "explicit correction", BASE.replace(day=4), 0.2, 0.95,
        )
        result = self.service.correct(request)
        self.assertEqual(result.replaced.version.lifecycle_status, "superseded")
        self.assertEqual(result.replacement.version.lifecycle_status, "confirmed")
        self.assertEqual(result.replaced.transition.replacement_claim_id, "new")
        self.assertTrue(self.service.correct(request).replayed)
        self._graph("other", user_id="user_2", valid_to=date(2026, 1, 31))
        with self.assertRaises(TemporalError):
            self.service.correct(
                CorrectionRequest(
                    "user_1", "new", "other", "cross-user", "confirmed",
                    "invalid", BASE.replace(day=5),
                )
            )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM lifecycle_transitions WHERE user_id = 'user_1'"
            ).fetchone()[0],
            3,
        )

    def test_as_of_query_uses_half_open_versions_inclusive_valid_time_and_visible_sources(self) -> None:
        self._graph("claim_1", valid_to=date(2026, 1, 31))
        self.service.transition(
            TransitionRequest(
                "user_1", "claim_1", "promote", "current", "accepted",
                BASE.replace(day=3), 0.9,
            )
        )
        self.service.transition(
            TransitionRequest(
                "user_1", "claim_1", "end", "historical", "ended",
                BASE.replace(day=4), 0.8,
            )
        )
        at_start = self.service.query(
            TemporalQuery("user_1", BASE.replace(day=3), date(2026, 1, 1))
        )
        self.assertEqual(at_start[0].version.lifecycle_status, "current")
        at_boundary = self.service.query(
            TemporalQuery("user_1", BASE.replace(day=4), date(2026, 1, 31))
        )
        self.assertEqual(at_boundary[0].version.lifecycle_status, "historical")
        self.assertEqual(
            self.service.query(
                TemporalQuery("user_1", BASE.replace(day=4), date(2026, 2, 1))
            ),
            (),
        )
        self._graph(
            "late", source_ingested=BASE.replace(day=10),
            version_from=BASE.replace(day=2), valid_to=date(2026, 1, 31),
        )
        self.assertEqual(
            self.service.query(
                TemporalQuery(
                    "user_1", BASE.replace(day=5),
                    statuses=frozenset({"candidate"}),
                )
            ),
            (),
        )

    def test_outbox_and_transition_roll_back_together(self) -> None:
        self._graph("claim_1", valid_to=date(2026, 1, 31))
        self.connection.execute(
            """
            CREATE FUNCTION reject_lifecycle_outbox() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'reject'; END $$;
            CREATE TRIGGER reject_lifecycle_outbox BEFORE INSERT ON processing_outbox
            FOR EACH ROW WHEN (NEW.event_type = 'claim_lifecycle_changed')
            EXECUTE FUNCTION reject_lifecycle_outbox();
            """
        )
        with self.assertRaises(psycopg.errors.RaiseException):
            self.service.transition(
                TransitionRequest(
                    "user_1", "claim_1", "rollback", "confirmed", "accepted",
                    BASE.replace(day=3), 0.8,
                )
            )
        row = self.connection.execute(
            "SELECT lifecycle_status, transaction_to FROM claim_versions WHERE claim_id = 'claim_1'"
        ).fetchone()
        self.assertEqual(row, ("candidate", None))
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM lifecycle_transitions").fetchone()[0],
            0,
        )

    def test_concurrent_same_request_replays_and_competing_request_has_one_successor(self) -> None:
        self._graph("claim_1", valid_to=date(2026, 1, 31))
        same = TransitionRequest(
            "user_1", "claim_1", "same", "confirmed", "accepted",
            BASE.replace(day=3), 0.8,
        )

        def run(request: TransitionRequest) -> object:
            connection = psycopg.connect(DATABASE_URL, autocommit=True)
            try:
                return TemporalService(connection).transition(request)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            same_results = tuple(pool.map(run, (same, same)))
        self.assertEqual({item.replayed for item in same_results}, {False, True})
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM claim_versions WHERE claim_id = 'claim_1' AND transaction_to IS NULL"
            ).fetchone()[0],
            1,
        )

        self._graph("claim_2", valid_to=date(2026, 1, 31))
        competing = (
            TransitionRequest(
                "user_1", "claim_2", "a", "confirmed", "accepted",
                BASE.replace(day=3), 0.8,
            ),
            TransitionRequest(
                "user_1", "claim_2", "b", "current", "accepted",
                BASE.replace(day=3), 0.8,
            ),
        )
        outcomes: list[object] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, item) for item in competing]
            for future in futures:
                try:
                    outcomes.append(future.result())
                except TemporalConflict as error:
                    outcomes.append(error)
        self.assertEqual(sum(not isinstance(item, Exception) for item in outcomes), 1)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM claim_versions WHERE claim_id = 'claim_2' AND transaction_to IS NULL"
            ).fetchone()[0],
            1,
        )

    def test_deleted_source_and_tombstone_dominate_earlier_as_of(self) -> None:
        self._graph("claim_1", valid_to=date(2026, 1, 31))
        self.service.transition(
            TransitionRequest(
                "user_1", "claim_1", "confirmed", "confirmed", "accepted",
                BASE.replace(day=3), 0.8,
            )
        )
        cutoff = BASE.replace(day=4)
        self.assertEqual(len(self.service.query(TemporalQuery("user_1", cutoff))), 1)
        deleted = IngestionService(self.connection).delete_source(
            "user_1", "source_user_1_claim_1", BASE.replace(day=5)
        )
        self.assertTrue(deleted.deleted)
        self.assertEqual(self.service.query(TemporalQuery("user_1", cutoff)), ())
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM source_tombstones WHERE user_id = 'user_1'"
            ).fetchone()[0],
            1,
        )

    def _graph(
        self,
        claim_id: str,
        *,
        user_id: str = "user_1",
        memory_kind: str | None = "episodic",
        precision: str = "day",
        valid_from: date | None = date(2026, 1, 1),
        valid_to: date | None = None,
        source_ingested: datetime = BASE,
        version_from: datetime = BASE.replace(day=2),
    ) -> ClaimVersionRecord:
        source_id = f"source_{user_id}_{claim_id}"
        span_id = f"span_{user_id}_{claim_id}"
        source = SourceEventRecord(
            source_id, user_id, "conversation", None, f"key_{claim_id}",
            BASE, source_ingested, f"quote {claim_id}", [user_id], {},
            hashlib.sha256(f"quote {claim_id}".encode()).hexdigest(),
        )
        self.repository.insert_source_event(source)
        self.repository.insert_source_span(
            SourceSpanRecord(span_id, user_id, source_id, "message_1", user_id, source.raw_content)
        )
        claim = ClaimRecord(
            claim_id, user_id, user_id, user_id, "role", "registry",
            claim_id, "positive", "asserted", valid_from, None, valid_to, None,
            precision, 0.8, memory_kind, "standard", "extractor",
        )
        self.repository.insert_claim(claim)
        self.repository.insert_evidence_link(
            EvidenceLinkRecord(user_id, claim_id, span_id, "supports", 0.8)
        )
        version = ClaimVersionRecord(
            f"version_{user_id}_{claim_id}", user_id, claim_id, "candidate",
            version_from, belief_confidence=None, valid_from_date=valid_from,
            valid_to_date=valid_to, time_precision=precision,
        )
        self.repository.insert_claim_version(version)
        return version


if __name__ == "__main__":
    unittest.main()
