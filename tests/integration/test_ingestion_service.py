from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import Barrier
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from ingestion.contracts import ClaimWrite, IngestRequest, IngestionConflict, IngestionError, stable_id
from storage.contracts import (
    ClaimRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations

if psycopg is not None:
    from ingestion.service import IngestionService
    from storage.repository import StorageRepository


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
NOW = datetime(2026, 1, 10, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for ingestion integration tests",
)
class IngestionServiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        apply_migrations(self.connection, REPO_ROOT / "migrations")
        self.repository = StorageRepository(self.connection)
        self.service = IngestionService(self.connection)
        for user_id in ("user_1", "user_2"):
            self.repository.insert_user(MemoryUser(user_id, NOW))
        for version_id in ("extractor_1", "extractor_2"):
            self.repository.insert_extraction_version(_extractor(version_id))

    def test_ingest_is_exactly_idempotent_and_user_scoped(self) -> None:
        source = _source("user_1", "source_1", "shared_key", ingested_at=NOW)
        first = self.service.ingest(IngestRequest(source, "extractor_1"))
        duplicate = self.service.ingest(IngestRequest(source, "extractor_1"))
        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertIsNone(duplicate.attempt_id)
        self.assertEqual(self._count("source_events"), 1)
        self.assertEqual(self._count("processing_attempts"), 1)
        self.assertEqual(self._count("processing_outbox"), 1)
        stored = self.repository.get_source_event("user_1", "source_1")
        self.assertEqual(stored.produced_at, source.produced_at)
        self.assertEqual(stored.ingested_at, source.ingested_at)
        self.assertEqual(stored.raw_content, source.raw_content)
        self.assertEqual(stored.metadata, source.metadata)

        other_user = _source("user_2", "source_2", "shared_key", ingested_at=NOW)
        self.assertTrue(
            self.service.ingest(IngestRequest(other_user, "extractor_1")).created
        )
        with self.assertRaisesRegex(IngestionConflict, "source_conflict"):
            self.service.ingest(
                IngestRequest(
                    _source("user_1", "source_3", "shared_key", content="drift"),
                    "extractor_1",
                )
            )
        self.assertIsNone(self.repository.get_source_event("user_1", "source_2"))
        self.assertEqual(self.repository.get_source_event("user_2", "source_2"), other_user)

    def test_concurrent_duplicate_ingest_creates_one_source_attempt_and_event(self) -> None:
        source = _source("user_1", "source_1", "key_1")
        ready = Barrier(2)

        def ingest_once() -> bool:
            connection = psycopg.connect(DATABASE_URL, autocommit=True)
            try:
                service = IngestionService(connection)
                ready.wait()
                return service.ingest(
                    IngestRequest(source, "extractor_1")
                ).created
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            created = tuple(executor.map(lambda _: ingest_once(), range(2)))

        self.assertEqual(sorted(created), [False, True])
        self.assertEqual(self._count("source_events"), 1)
        self.assertEqual(self._count("processing_attempts"), 1)
        self.assertEqual(self._count("processing_outbox"), 1)

    def test_out_of_order_delivery_preserves_source_and_ingestion_times(self) -> None:
        late_fact = _source(
            "user_1",
            "source_late",
            "key_late",
            produced_at=NOW - timedelta(days=30),
            ingested_at=NOW + timedelta(days=2),
        )
        recent_fact = _source(
            "user_1",
            "source_recent",
            "key_recent",
            produced_at=NOW - timedelta(days=1),
            ingested_at=NOW,
        )
        self.service.ingest(IngestRequest(recent_fact, "extractor_1"))
        self.service.ingest(IngestRequest(late_fact, "extractor_1"))
        rows = self.connection.execute(
            "SELECT source_id FROM source_events ORDER BY produced_at"
        ).fetchall()
        self.assertEqual(rows, [("source_late",), ("source_recent",)])

    def test_skip_locked_leasing_and_expired_retry_are_database_owned(self) -> None:
        for number in (1, 2):
            self.service.ingest(
                IngestRequest(
                    _source("user_1", f"source_{number}", f"key_{number}"),
                    "extractor_1",
                )
            )
        second_connection = psycopg.connect(DATABASE_URL, autocommit=True)
        try:
            second_service = IngestionService(second_connection)
            with self.connection.transaction():
                locked = self.connection.execute(
                    """
                    SELECT attempt_id FROM processing_attempts
                    WHERE source_id = 'source_1' FOR UPDATE
                    """
                ).fetchone()[0]
                leased = second_service.lease_next(
                    "worker_2", NOW, timedelta(seconds=30)
                )
                self.assertNotEqual(leased.attempt_id, locked)
                self.assertEqual(leased.source_id, "source_2")
        finally:
            second_connection.close()

        first = self.service.lease_next("worker_1", NOW, timedelta(seconds=1))
        self.assertEqual(first.source_id, "source_1")
        self.assertEqual(
            self.service.recover_expired_leases(NOW + timedelta(seconds=2)), 1
        )
        old = self.repository.get_processing_attempt("user_1", first.attempt_id)
        self.assertEqual(old.state, "failed")
        self.assertEqual(old.sanitized_error_code, "lease_expired")
        retry = self.service.lease_next(
            "worker_3", NOW + timedelta(seconds=2), timedelta(seconds=30)
        )
        self.assertEqual(retry.source_id, "source_1")
        self.assertEqual(retry.attempt_number, 2)

    def test_worker_failure_rolls_back_and_validation_is_not_retryable(self) -> None:
        source = _source("user_1", "source_1", "key_1", content="exact quote")
        self.service.ingest(IngestRequest(source, "extractor_1"))
        lease = self.service.lease_next("worker_1", NOW, timedelta(minutes=1))
        valid = _write(source, "extractor_1", "claim_1", "exact quote")
        invalid = _write(source, "extractor_1", "claim_2", "missing quote")
        with self.assertRaisesRegex(IngestionError, "provenance_failure"):
            self.service.complete_attempt(
                "user_1", lease.attempt_id, "worker_1", NOW, (valid, invalid)
            )
        self.assertEqual(self._count("claims"), 0)
        self.assertEqual(self._count("source_spans"), 0)
        self.assertEqual(
            self.repository.get_processing_attempt("user_1", lease.attempt_id).state,
            "running",
        )
        failed = self.service.fail_attempt(
            "user_1",
            lease.attempt_id,
            "worker_1",
            "provenance_failure",
            NOW,
            retryable=True,
        )
        self.assertFalse(failed.retryable)
        with self.assertRaisesRegex(IngestionConflict, "attempt_not_retryable"):
            self.service.enqueue_reprocessing(
                "user_1", "source_1", "extractor_1", NOW
            )

    def test_retryable_worker_failure_creates_only_the_next_attempt(self) -> None:
        source = _source("user_1", "source_1", "key_1")
        self.service.ingest(IngestRequest(source, "extractor_1"))
        first = self.service.lease_next("worker_1", NOW, timedelta(minutes=1))
        failed = self.service.fail_attempt(
            "user_1",
            first.attempt_id,
            "worker_1",
            "worker_unavailable",
            NOW,
            retryable=True,
        )
        self.assertTrue(failed.retryable)
        retry = self.service.enqueue_reprocessing(
            "user_1", "source_1", "extractor_1", NOW + timedelta(seconds=1)
        )
        self.assertTrue(retry.created)
        self.assertEqual(
            self.repository.get_processing_attempt("user_1", retry.attempt_id).attempt_number,
            2,
        )
        duplicate = self.service.enqueue_reprocessing(
            "user_1", "source_1", "extractor_1", NOW + timedelta(seconds=2)
        )
        self.assertFalse(duplicate.created)
        self.assertEqual(self._count("processing_attempts"), 2)

    def test_same_extractor_is_noop_and_new_extractor_reuses_canonical_claim(self) -> None:
        source = _source("user_1", "source_1", "key_1", content="exact quote")
        self.service.ingest(IngestRequest(source, "extractor_1"))
        lease = self.service.lease_next("worker_1", NOW, timedelta(minutes=1))
        first = _write(source, "extractor_1", "claim_1", "exact quote")
        self.service.complete_attempt(
            "user_1", lease.attempt_id, "worker_1", NOW, (first,)
        )
        same = self.service.enqueue_reprocessing(
            "user_1", "source_1", "extractor_1", NOW + timedelta(minutes=1)
        )
        self.assertFalse(same.created)

        new = self.service.enqueue_reprocessing(
            "user_1", "source_1", "extractor_2", NOW + timedelta(minutes=1)
        )
        self.assertTrue(new.created)
        lease2 = self.service.lease_next(
            "worker_2", NOW + timedelta(minutes=1), timedelta(minutes=1)
        )
        reused = _write(source, "extractor_2", "claim_1", "exact quote")
        changed = _write(source, "extractor_2", "claim_2", "exact quote")
        result = self.service.complete_attempt(
            "user_1",
            lease2.attempt_id,
            "worker_2",
            NOW + timedelta(minutes=1),
            (reused, changed),
        )
        self.assertEqual(result.created_claim_ids, ("claim_2",))
        self.assertEqual(self._count("claims"), 2)
        self.assertEqual(self._count("claim_versions"), 2)
        self.assertEqual(self._count("claim_extractions"), 3)
        origin = self.repository.get_claim("user_1", "claim_1")
        self.assertEqual(origin.extraction_version_id, "extractor_1")
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM evidence_links WHERE claim_id = 'claim_1'"
            ).fetchone()[0],
            1,
        )

    def test_source_deletion_retires_sole_claims_and_recomputes_shared_claims(self) -> None:
        source1 = _source("user_1", "source_1", "key_1", content="shared sole")
        source2 = _source("user_1", "source_2", "key_2", content="shared")
        for source in (source1, source2):
            self.service.ingest(IngestRequest(source, "extractor_1"))
        lease1 = self.service.lease_next("worker_1", NOW, timedelta(minutes=1))
        self.service.complete_attempt(
            "user_1",
            lease1.attempt_id,
            "worker_1",
            NOW,
            (
                _write(source1, "extractor_1", "claim_shared", "shared"),
                _write(source1, "extractor_1", "claim_sole", "sole"),
            ),
        )
        lease2 = self.service.lease_next("worker_2", NOW, timedelta(minutes=1))
        self.service.complete_attempt(
            "user_1",
            lease2.attempt_id,
            "worker_2",
            NOW,
            (_write(source2, "extractor_1", "claim_shared", "shared"),),
        )

        with self.assertRaisesRegex(IngestionError, "source_not_found"):
            self.service.delete_source("user_2", "source_1", NOW)
        deleted = self.service.delete_source("user_1", "source_1", NOW)
        self.assertEqual(deleted.affected_claim_ids, ("claim_shared", "claim_sole"))
        self.assertEqual(deleted.retired_claim_ids, ("claim_sole",))
        self.assertIsNotNone(self.repository.get_claim("user_1", "claim_shared"))
        self.assertIsNone(self.repository.get_claim("user_1", "claim_sole"))
        self.assertEqual(
            self.connection.execute(
                """
                SELECT count(*) FROM claims AS claim
                WHERE NOT EXISTS (
                    SELECT 1 FROM evidence_links AS evidence
                    WHERE evidence.user_id = claim.user_id
                      AND evidence.claim_id = claim.claim_id
                )
                """
            ).fetchone()[0],
            0,
        )
        tombstone_columns = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'source_tombstones'
                """
            ).fetchall()
        }
        self.assertNotIn("raw_content", tombstone_columns)
        self.assertFalse(
            self.service.delete_source("user_1", "source_1", NOW).deleted
        )
        with self.assertRaisesRegex(IngestionConflict, "source_tombstoned"):
            self.service.ingest(IngestRequest(source1, "extractor_1"))
        self.assertEqual(
            self.connection.execute(
                """
                SELECT count(*) FROM processing_outbox
                WHERE event_type = 'claim_recompute_required'
                """
            ).fetchone()[0],
            1,
        )

    def test_phase3_development_replay_twice_is_count_stable(self) -> None:
        self.connection.execute("TRUNCATE memory_users CASCADE")
        sources = _jsonl_prefix(
            REPO_ROOT / "data/scaled-v1/runtime/sources.jsonl", 20
        )
        claims = _jsonl_all(
            REPO_ROOT
            / "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl"
        )
        for user_id in ("user_001", "user_002"):
            created = min(
                datetime.fromisoformat(item["created_at"])
                for item in sources
                if item["user_id"] == user_id
            )
            self.repository.insert_user(MemoryUser(user_id, created))
        extraction = _extractor("step35_gpt41_fallback_v1")
        self.repository.insert_extraction_version(extraction)
        records = [_runtime_source(item) for item in sources]
        replay_at = max(item.ingested_at for item in records) + timedelta(seconds=1)
        for record in records:
            self.service.ingest(IngestRequest(record, extraction.version_id))
        claims_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in claims:
            claims_by_source[item["evidence"][0]["source_id"]].append(item)
        source_map = {item.source_id: item for item in records}
        while True:
            lease = self.service.lease_next(
                "replay_worker", replay_at, timedelta(minutes=5)
            )
            if lease is None:
                break
            writes = tuple(
                _phase3_write(source_map[lease.source_id], extraction.version_id, item)
                for item in claims_by_source[lease.source_id]
            )
            self.service.complete_attempt(
                lease.user_id,
                lease.attempt_id,
                "replay_worker",
                replay_at,
                writes,
            )
        first = self._pipeline_counts()
        for record in records:
            result = self.service.ingest(IngestRequest(record, extraction.version_id))
            self.assertFalse(result.created)
        self.assertEqual(self._pipeline_counts(), first)
        self.assertEqual(first["source_events"], 20)
        self.assertEqual(first["processing_attempts"], 20)
        self.assertEqual(first["claims"], 33)
        self.assertEqual(first["claim_versions"], 33)
        self.assertEqual(first["claim_extractions"], 33)
        self.assertEqual(first["evidence_links"], 33)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM claim_versions WHERE lifecycle_status <> 'candidate'"
            ).fetchone()[0],
            0,
        )

    def _count(self, table: str) -> int:
        return self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def _pipeline_counts(self) -> dict[str, int]:
        return {
            table: self._count(table)
            for table in (
                "source_events",
                "processing_attempts",
                "processing_outbox",
                "source_spans",
                "claims",
                "claim_versions",
                "claim_extractions",
                "evidence_links",
            )
        }


def _extractor(version_id: str) -> ExtractionVersionRecord:
    return ExtractionVersionRecord(
        version_id,
        "model_1",
        "prompt_1",
        SHA_A,
        "schema_1",
        SHA_A,
        "registry_1",
        SHA_A,
        SHA_B,
        NOW,
    )


def _source(
    user_id: str,
    source_id: str,
    key: str,
    *,
    content: str = "exact quote",
    produced_at: datetime = NOW - timedelta(days=1),
    ingested_at: datetime = NOW,
) -> SourceEventRecord:
    return SourceEventRecord(
        source_id,
        user_id,
        "chat",
        f"session_{source_id}",
        key,
        produced_at,
        ingested_at,
        content,
        [user_id],
        {"title": None},
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _write(
    source: SourceEventRecord,
    extraction_version_id: str,
    claim_id: str,
    quote: str,
) -> ClaimWrite:
    start = source.raw_content.find(quote)
    span_id = stable_id("span", source.user_id, source.source_id, quote)
    span = SourceSpanRecord(
        span_id,
        source.user_id,
        source.source_id,
        f"message_{source.source_id}",
        source.user_id,
        quote,
        None if start < 0 else start,
        None if start < 0 else start + len(quote),
    )
    claim = ClaimRecord(
        claim_id,
        source.user_id,
        source.user_id,
        source.user_id,
        "career_goal",
        "registry_1",
        "build useful systems" if "shared" not in claim_id else "shared goal",
        "positive",
        "asserted",
        date(2026, 1, 1),
        None,
        None,
        None,
        "day",
        0.8,
        None,
        None,
        extraction_version_id,
    )
    return ClaimWrite(
        claim,
        (span,),
        (EvidenceLinkRecord(source.user_id, claim_id, span_id, "supports", 0.8),),
    )


def _runtime_source(item: dict[str, object]) -> SourceEventRecord:
    metadata = item["metadata"]
    session_id = (
        metadata.get("thread_id")
        or metadata.get("event_id")
        or item["source_id"]
    )
    return SourceEventRecord(
        item["source_id"],
        item["user_id"],
        item["source_type"],
        session_id,
        f"step42:{item['source_id']}",
        datetime.fromisoformat(item["created_at"]),
        datetime.fromisoformat(item["ingested_at"]),
        item["content"],
        item["participants"],
        metadata,
        hashlib.sha256(item["content"].encode("utf-8")).hexdigest(),
    )


def _phase3_write(
    source: SourceEventRecord,
    extraction_version_id: str,
    item: dict[str, object],
) -> ClaimWrite:
    evidence = item["evidence"][0]
    quote = evidence["quote"]
    start = source.raw_content.find(quote)
    span_id = stable_id(
        "span",
        item["user_id"],
        evidence["source_id"],
        evidence["message_id"],
        quote,
    )
    span = SourceSpanRecord(
        span_id,
        item["user_id"],
        evidence["source_id"],
        evidence["message_id"],
        item["speaker_id"],
        quote,
        start,
        start + len(quote),
    )
    valid_from_date, valid_from_timestamp = _boundary(
        item["valid_from"], item["time_precision"]
    )
    valid_to_date, valid_to_timestamp = _boundary(
        item["valid_to"], item["time_precision"]
    )
    claim = ClaimRecord(
        item["claim_id"],
        item["user_id"],
        item["subject_id"],
        item["speaker_id"],
        item["predicate"],
        item["predicate_registry_version"],
        item["object"],
        item["polarity"],
        item["epistemic_status"],
        valid_from_date,
        valid_from_timestamp,
        valid_to_date,
        valid_to_timestamp,
        item["time_precision"],
        item["confidence"],
        None,
        None,
        extraction_version_id,
    )
    return ClaimWrite(
        claim,
        (span,),
        (
            EvidenceLinkRecord(
                item["user_id"], item["claim_id"], span_id, "supports", item["confidence"]
            ),
        ),
    )


def _boundary(value: str | None, precision: str) -> tuple[date | None, datetime | None]:
    if value is None:
        return None, None
    if precision == "timestamp":
        return None, datetime.fromisoformat(value)
    return date.fromisoformat(value), None


def _jsonl_prefix(path: Path, count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for _ in range(count):
            records.append(json.loads(next(handle)))
    return records


def _jsonl_all(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if __name__ == "__main__":
    unittest.main()
