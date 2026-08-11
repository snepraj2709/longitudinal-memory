from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import unittest

from ingestion.contracts import (
    ClaimWrite,
    IngestionConflict,
    NONRETRYABLE_FAILURE_CODES,
    attempt_id,
    claim_version_id,
    outbox_id,
)
from storage.contracts import (
    ClaimRecord,
    EvidenceLinkRecord,
    ProcessingAttemptRecord,
    ProcessingOutboxRecord,
    SourceSpanRecord,
    SourceTombstoneRecord,
    StorageValidationError,
)


UTC = timezone.utc
NOW = datetime(2026, 1, 2, tzinfo=UTC)
SHA256 = "a" * 64


def claim() -> ClaimRecord:
    return ClaimRecord(
        "claim_1",
        "user_1",
        "user_1",
        "user_1",
        "career_goal",
        "registry_1",
        "build useful systems",
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
        "extractor_1",
    )


class IngestionContractTests(unittest.TestCase):
    def test_stable_ids_are_deterministic_and_scope_attempts(self) -> None:
        first = attempt_id("user_1", "source_1", "extractor_1", 1)
        self.assertEqual(first, attempt_id("user_1", "source_1", "extractor_1", 1))
        self.assertNotEqual(first, attempt_id("user_2", "source_1", "extractor_1", 1))
        self.assertNotEqual(first, attempt_id("user_1", "source_1", "extractor_2", 1))
        self.assertEqual(
            claim_version_id("user_1", "claim_1"),
            claim_version_id("user_1", "claim_1"),
        )
        self.assertNotEqual(
            outbox_id("user_1", "source_ingested", "source_1"),
            outbox_id("user_2", "source_ingested", "source_1"),
        )

    def test_claim_write_requires_owned_unique_provenance(self) -> None:
        record = claim()
        span = SourceSpanRecord(
            "span_1", "user_1", "source_1", "message_1", "user_1", "quote"
        )
        link = EvidenceLinkRecord("user_1", "claim_1", "span_1", "supports", 0.8)
        self.assertEqual(ClaimWrite(record, (span,), (link,)).claim, record)
        with self.assertRaisesRegex(StorageValidationError, "evidence references"):
            ClaimWrite(
                record,
                (span,),
                (EvidenceLinkRecord("user_1", "claim_1", "missing", "supports", 0.8),),
            )

    def test_processing_attempt_state_and_lease_fields_are_consistent(self) -> None:
        pending = ProcessingAttemptRecord(
            "attempt_1",
            "user_1",
            "source_1",
            "extractor_1",
            1,
            "pending",
            NOW,
            retryable=True,
        )
        self.assertTrue(pending.retryable)
        running = ProcessingAttemptRecord(
            "attempt_1",
            "user_1",
            "source_1",
            "extractor_1",
            1,
            "running",
            NOW,
            lease_owner="worker_1",
            lease_expires_at=NOW + timedelta(minutes=1),
            retryable=True,
        )
        self.assertEqual(running.lease_owner, "worker_1")
        with self.assertRaisesRegex(StorageValidationError, "running attempt"):
            ProcessingAttemptRecord(
                "attempt_1",
                "user_1",
                "source_1",
                "extractor_1",
                1,
                "running",
                NOW,
                retryable=True,
            )

    def test_failure_classes_freeze_nonretryable_boundaries(self) -> None:
        self.assertEqual(
            NONRETRYABLE_FAILURE_CODES,
            {
                "validation_failure",
                "provenance_failure",
                "user_isolation_failure",
                "stable_id_conflict",
            },
        )
        error = IngestionConflict("source_conflict", "source")
        self.assertEqual(str(error), "source_conflict at source")
        self.assertNotIn("raw", str(error))

    def test_outbox_and_tombstone_are_content_free_and_strict(self) -> None:
        event = ProcessingOutboxRecord(
            "event_1",
            "user_1",
            "source_deleted",
            "source_1",
            "source_deleted:source_1",
            {"source_id": "source_1"},
            "pending",
            NOW,
        )
        self.assertNotIn("raw_content", event.__dict__)
        tombstone = SourceTombstoneRecord(
            "user_1", "source_1", "key_1", SHA256, NOW
        )
        self.assertNotIn("raw_content", tombstone.__dict__)
        with self.assertRaisesRegex(StorageValidationError, "inconsistent"):
            ProcessingOutboxRecord(
                "event_2",
                "user_1",
                "source_deleted",
                "source_1",
                "source_deleted:source_2",
                {},
                "published",
                NOW,
            )


if __name__ == "__main__":
    unittest.main()
