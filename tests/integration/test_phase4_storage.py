from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
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
    ProcessingAttemptRecord,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import MigrationError, apply_migrations

if psycopg is not None:
    from storage.repository import StorageConflictError, StorageRepository


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = REPO_ROOT / "migrations"
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
SHA_A = "a" * 64
SHA_B = "b" * 64


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for storage integration tests",
)
class Phase4StorageIntegrationTests(unittest.TestCase):
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
            apply_migrations(self.connection, MIGRATIONS),
            (
                "0001_phase4_storage.sql",
                "0002_ingestion_reprocessing.sql",
                "0003_temporal_lifecycle.sql",
            ),
        )
        self.repository = StorageRepository(self.connection)

    def test_clean_and_repeated_migration_enable_vector_and_all_tables(self) -> None:
        self.assertEqual(apply_migrations(self.connection, MIGRATIONS), ())
        tables = {
            row[0]
            for row in self.connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        self.assertEqual(
            tables,
            {
                "schema_migrations",
                "memory_users",
                "source_events",
                "source_spans",
                "extraction_versions",
                "processing_attempts",
                "claims",
                "claim_versions",
                "evidence_links",
                "claim_extractions",
                "processing_outbox",
                "source_tombstones",
                "lifecycle_transitions",
            },
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0],
            "0.8.6",
        )
        columns = self.connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name IN ('claims', 'source_events')
            """
        ).fetchall()
        self.assertNotIn(("embedding",), columns)
        index = self.connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'claim_versions_one_open_per_claim'"
        ).fetchone()[0]
        self.assertIn("WHERE (transaction_to IS NULL)", index)

    def test_migration_checksum_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "0001_phase4_storage.sql"
            changed.write_text("SELECT 1;\n", encoding="utf-8")
            with self.assertRaisesRegex(MigrationError, "checksum changed"):
                apply_migrations(self.connection, temporary)

    def test_same_user_roundtrip_and_duplicate_same_content_are_harmless(self) -> None:
        records = self._insert_complete_graph()
        for insert, get, record, identity in (
            (self.repository.insert_user, self.repository.get_user, records["user"], ("user_1",)),
            (self.repository.insert_source_event, self.repository.get_source_event, records["source"], ("user_1", "source_1")),
            (self.repository.insert_source_span, self.repository.get_source_span, records["span"], ("user_1", "span_1")),
            (self.repository.insert_extraction_version, self.repository.get_extraction_version, records["extraction"], ("extractor_1",)),
            (self.repository.insert_processing_attempt, self.repository.get_processing_attempt, records["attempt"], ("user_1", "attempt_1")),
            (self.repository.insert_claim, self.repository.get_claim, records["claim"], ("user_1", "claim_1")),
            (self.repository.insert_claim_version, self.repository.get_claim_version, records["version"], ("user_1", "version_1")),
        ):
            self.assertEqual(get(*identity), record)
            self.assertEqual(insert(record), record)
        link = records["evidence"]
        self.assertEqual(
            self.repository.get_evidence_link(
                link.user_id, link.claim_id, link.span_id, link.support_type
            ),
            link,
        )
        self.assertEqual(self.repository.insert_evidence_link(link), link)
        self.assertEqual(records["source"].metadata, {"title": None, "label": "café"})
        self.assertEqual(records["claim"].object_json, {"goal": "निर्माण", "priority": 2})

    def test_cross_user_unknown_and_restricted_deletes_fail_in_database(self) -> None:
        records = self._insert_complete_graph()
        self.repository.insert_user(MemoryUser("user_2", datetime(2026, 1, 1, tzinfo=UTC)))
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.repository.insert_evidence_link(
                EvidenceLinkRecord("user_2", "claim_1", "span_1", "supports", 0.8)
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.repository.insert_source_span(
                SourceSpanRecord("missing_span", "user_1", "missing_source", None, "user_1", "quote")
            )
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute("DELETE FROM source_events WHERE source_id = 'source_1'")
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.connection.execute("DELETE FROM claims WHERE claim_id = 'claim_1'")
        self.assertEqual(
            self.repository.get_claim("user_1", records["claim"].claim_id),
            records["claim"],
        )

    def test_stable_id_idempotency_open_version_and_checks_reject_drift(self) -> None:
        records = self._insert_complete_graph()
        with self.assertRaises(StorageConflictError):
            self.repository.insert_source_event(
                SourceEventRecord(
                    **{**records["source"].__dict__, "raw_content": "different"}
                )
            )
        with self.assertRaises(psycopg.errors.UniqueViolation):
            self.repository.insert_source_event(
                SourceEventRecord(
                    **{**records["source"].__dict__, "source_id": "source_2"}
                )
            )
        with self.assertRaises(psycopg.errors.UniqueViolation):
            self.repository.insert_claim_version(
                ClaimVersionRecord(
                    "version_2", "user_1", "claim_1", "candidate",
                    datetime(2026, 1, 3, tzinfo=UTC),
                )
            )
        invalid_updates = (
            ("UPDATE source_events SET participants = '{}'::jsonb WHERE source_id = 'source_1'", psycopg.errors.CheckViolation),
            ("UPDATE source_events SET metadata = '[]'::jsonb WHERE source_id = 'source_1'", psycopg.errors.CheckViolation),
            ("UPDATE source_spans SET start_offset = 4, end_offset = 2 WHERE span_id = 'span_1'", psycopg.errors.CheckViolation),
            ("UPDATE claims SET extraction_confidence = 2 WHERE claim_id = 'claim_1'", psycopg.errors.CheckViolation),
            ("UPDATE claims SET object_json = 'null'::jsonb WHERE claim_id = 'claim_1'", psycopg.errors.CheckViolation),
            ("UPDATE claims SET polarity = 'maybe' WHERE claim_id = 'claim_1'", psycopg.errors.CheckViolation),
            ("UPDATE claim_versions SET lifecycle_status = 'active' WHERE version_id = 'version_1'", psycopg.errors.CheckViolation),
            ("UPDATE claim_versions SET transaction_to = transaction_from WHERE version_id = 'version_1'", psycopg.errors.CheckViolation),
        )
        for statement, error_type in invalid_updates:
            with self.subTest(statement=statement):
                with self.assertRaises(error_type):
                    self.connection.execute(statement)

    def test_failed_transaction_leaves_no_partial_graph(self) -> None:
        try:
            with self.connection.transaction():
                self.repository.insert_user(
                    MemoryUser("rollback_user", datetime(2026, 1, 1, tzinfo=UTC))
                )
                self.repository.insert_source_span(
                    SourceSpanRecord(
                        "rollback_span", "rollback_user", "missing", None,
                        "rollback_user", "quote",
                    )
                )
        except psycopg.errors.ForeignKeyViolation:
            pass
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM memory_users WHERE user_id = 'rollback_user'"
            ).fetchone()[0],
            0,
        )

    def test_phase3_handoff_loads_20_sources_and_33_weak_candidates_then_rolls_back(self) -> None:
        sources = _read_jsonl_prefix(
            REPO_ROOT / "data/scaled-v1/runtime/sources.jsonl", 20
        )
        claims = _read_jsonl_all(
            REPO_ROOT
            / "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl"
        )
        self.assertEqual(len(sources), 20)
        self.assertEqual(len(claims), 33)
        source_map = {item["source_id"]: item for item in sources}
        self.assertEqual(set(source_map), {item["source_id"] for item in sources})

        class RollbackLoad(Exception):
            pass

        try:
            with self.connection.transaction():
                self._load_phase3_handoff(sources, claims)
                counts = {
                    table: self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    for table in (
                        "memory_users", "source_events", "source_spans", "processing_attempts",
                        "claims", "claim_versions", "evidence_links",
                    )
                }
                self.assertEqual(counts["memory_users"], 2)
                self.assertEqual(counts["source_events"], 20)
                self.assertEqual(counts["processing_attempts"], 20)
                self.assertEqual(counts["claims"], 33)
                self.assertEqual(counts["claim_versions"], 33)
                self.assertEqual(counts["evidence_links"], 33)
                self.assertGreater(counts["source_spans"], 0)
                self.assertEqual(
                    self.connection.execute(
                        "SELECT count(*) FROM claim_versions WHERE lifecycle_status <> 'candidate'"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    self.connection.execute(
                        "SELECT count(*) FROM claims WHERE memory_kind IS NOT NULL OR sensitivity IS NOT NULL"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    self.connection.execute(
                        "SELECT count(*) FROM claim_versions WHERE belief_confidence IS NOT NULL"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    self.connection.execute(
                        """
                        SELECT count(*)
                        FROM claim_versions AS version
                        JOIN claims AS claim
                          ON claim.user_id = version.user_id
                         AND claim.claim_id = version.claim_id
                        WHERE version.valid_from_date IS DISTINCT FROM claim.valid_from_date
                           OR version.valid_from_timestamp IS DISTINCT FROM claim.valid_from_timestamp
                           OR version.valid_to_date IS DISTINCT FROM claim.valid_to_date
                           OR version.valid_to_timestamp IS DISTINCT FROM claim.valid_to_timestamp
                           OR version.time_precision IS DISTINCT FROM claim.time_precision
                        """
                    ).fetchone()[0],
                    0,
                )
                stored_ids = {
                    row[0]
                    for row in self.connection.execute("SELECT claim_id FROM claims").fetchall()
                }
                self.assertEqual(stored_ids, {item["claim_id"] for item in claims})
                raise RollbackLoad
        except RollbackLoad:
            pass
        for table in (
            "memory_users", "source_events", "source_spans", "processing_attempts",
            "claims", "claim_versions", "evidence_links",
        ):
            self.assertEqual(
                self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0
            )

    def _insert_complete_graph(self) -> dict[str, object]:
        user = MemoryUser("user_1", datetime(2026, 1, 1, tzinfo=UTC))
        source = SourceEventRecord(
            "source_1", "user_1", "chat", "session_1", "key_1",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            "नमस्ते", ["user_1"], {"title": None, "label": "café"}, SHA_A,
        )
        span = SourceSpanRecord(
            "span_1", "user_1", "source_1", "message_1", "user_1", "नमस्ते", 0, 6
        )
        extraction = ExtractionVersionRecord(
            "extractor_1", "model_1", "prompt_1", SHA_A, "schema_1", SHA_A,
            "registry_1", SHA_A, SHA_B, datetime(2026, 1, 2, tzinfo=UTC),
        )
        attempt = ProcessingAttemptRecord(
            "attempt_1", "user_1", "source_1", "extractor_1", 1, "succeeded",
            datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 2, 0, 1, tzinfo=UTC),
        )
        claim = ClaimRecord(
            "claim_1", "user_1", "user_1", "user_1", "career_goal", "registry_1",
            {"goal": "निर्माण", "priority": 2}, "positive", "asserted",
            date(2026, 1, 1), None, date(2026, 1, 31), None, "day", 0.8,
            None, None, "extractor_1",
        )
        version = ClaimVersionRecord(
            "version_1", "user_1", "claim_1", "candidate",
            datetime(2026, 1, 2, tzinfo=UTC), None, None,
            valid_from_date=claim.valid_from_date,
            valid_from_timestamp=claim.valid_from_timestamp,
            valid_to_date=claim.valid_to_date,
            valid_to_timestamp=claim.valid_to_timestamp,
            time_precision=claim.time_precision,
        )
        evidence = EvidenceLinkRecord("user_1", "claim_1", "span_1", "supports", 0.8)
        for insert, record in (
            (self.repository.insert_user, user),
            (self.repository.insert_source_event, source),
            (self.repository.insert_source_span, span),
            (self.repository.insert_extraction_version, extraction),
            (self.repository.insert_processing_attempt, attempt),
            (self.repository.insert_claim, claim),
            (self.repository.insert_claim_version, version),
            (self.repository.insert_evidence_link, evidence),
        ):
            insert(record)
        return {
            "user": user, "source": source, "span": span, "extraction": extraction,
            "attempt": attempt, "claim": claim, "version": version, "evidence": evidence,
        }

    def _load_phase3_handoff(
        self, sources: list[dict[str, object]], claims: list[dict[str, object]]
    ) -> None:
        for user_id in ("user_001", "user_002"):
            produced = min(
                datetime.fromisoformat(item["created_at"])
                for item in sources if item["user_id"] == user_id
            )
            self.repository.insert_user(MemoryUser(user_id, produced))
        extraction = ExtractionVersionRecord(
            "step35_gpt41_fallback_v1", "gpt-4.1-2025-04-14",
            "atomic-extraction-v3", "1a6a371a73b043807249a4c130b5b309b8a29fbcbea34f6c2e912f6eded38e59",
            "atomic_extraction_v1", "f694616f02143cc4e5595fe0658b100f29562c9c4d4df716af362fa10989aac1",
            "predicate_registry_v2", "5cb9ba2f2b7a81aa2ce61d52d9425fbf1f2d39964a3a66286d5679ebb2e5175d",
            "f5127cfdb7720ecf84e320da613396d11d71629b709813e2e25ea2ab00df0876",
            max(datetime.fromisoformat(item["ingested_at"]) for item in sources),
        )
        self.repository.insert_extraction_version(extraction)
        for source in sources:
            metadata = source["metadata"]
            session_id = metadata.get("thread_id") or metadata.get("event_id") or source["source_id"]
            event = SourceEventRecord(
                source["source_id"], source["user_id"], source["source_type"], session_id,
                f"step35:{source['source_id']}", datetime.fromisoformat(source["created_at"]),
                datetime.fromisoformat(source["ingested_at"]), source["content"],
                source["participants"], metadata,
                hashlib.sha256(source["content"].encode("utf-8")).hexdigest(),
            )
            self.repository.insert_source_event(event)
            self.repository.insert_processing_attempt(
                ProcessingAttemptRecord(
                    f"attempt_{source['source_id']}", source["user_id"], source["source_id"],
                    extraction.version_id, 1, "succeeded", event.ingested_at, event.ingested_at,
                )
            )
        spans: dict[tuple[object, ...], SourceSpanRecord] = {}
        for item in claims:
            evidence = item["evidence"][0]
            source = next(
                record
                for record in sources
                if record["source_id"] == evidence["source_id"]
            )
            if evidence["message_id"] is None:
                text = source["content"]
                speaker_id = source["user_id"]
            else:
                message = next(record for record in source["messages"] if record["message_id"] == evidence["message_id"])
                text = message["text"]
                speaker_id = message["speaker_id"]
            start = text.find(evidence["quote"])
            self.assertGreaterEqual(start, 0)
            identity = (item["user_id"], evidence["source_id"], evidence["message_id"], evidence["quote"])
            digest = hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            span = spans.setdefault(
                identity,
                SourceSpanRecord(
                    f"span_{digest}", item["user_id"], evidence["source_id"],
                    evidence["message_id"], speaker_id, evidence["quote"],
                    start, start + len(evidence["quote"]),
                ),
            )
            self.repository.insert_source_span(span)
            valid_from_date, valid_from_timestamp = _boundary(item["valid_from"], item["time_precision"])
            valid_to_date, valid_to_timestamp = _boundary(item["valid_to"], item["time_precision"])
            claim = ClaimRecord(
                item["claim_id"], item["user_id"], item["subject_id"], item["speaker_id"],
                item["predicate"], item["predicate_registry_version"], item["object"],
                item["polarity"], item["epistemic_status"], valid_from_date,
                valid_from_timestamp, valid_to_date, valid_to_timestamp,
                item["time_precision"], item["confidence"], None, None,
                extraction.version_id,
            )
            self.repository.insert_claim(claim)
            transaction_from = datetime.fromisoformat(source["ingested_at"])
            self.repository.insert_claim_version(
                ClaimVersionRecord(
                    f"version_{item['claim_id']}", item["user_id"], item["claim_id"],
                    "candidate", transaction_from, None, None,
                    valid_from_date=valid_from_date,
                    valid_from_timestamp=valid_from_timestamp,
                    valid_to_date=valid_to_date,
                    valid_to_timestamp=valid_to_timestamp,
                    time_precision=item["time_precision"],
                )
            )
            self.repository.insert_evidence_link(
                EvidenceLinkRecord(
                    item["user_id"], item["claim_id"], span.span_id,
                    "supports", item["confidence"],
                )
            )


def _read_jsonl_prefix(path: Path, count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for _ in range(count):
            line = handle.readline()
            if not line:
                raise AssertionError(f"{path} ended before {count} records")
            records.append(json.loads(line))
    return records


def _read_jsonl_all(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _boundary(value: str | None, precision: str) -> tuple[date | None, datetime | None]:
    if value is None:
        return None, None
    if precision == "timestamp":
        return None, datetime.fromisoformat(value)
    return date.fromisoformat(value), None


if __name__ == "__main__":
    unittest.main()
