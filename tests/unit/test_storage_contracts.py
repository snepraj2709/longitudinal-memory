from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    SourceEventRecord,
    SourceSpanRecord,
    StorageValidationError,
    safe_json,
)
from storage.migrations import MigrationError, load_migrations


UTC = timezone.utc
SHA256 = "a" * 64


def claim(**changes: object) -> ClaimRecord:
    values: dict[str, object] = {
        "claim_id": "claim_1",
        "user_id": "user_1",
        "subject_id": "user_1",
        "speaker_id": "user_1",
        "predicate": "career_goal",
        "predicate_registry_version": "registry_v1",
        "object_json": "build useful systems",
        "polarity": "positive",
        "epistemic_status": "asserted",
        "valid_from_date": date(2026, 1, 1),
        "valid_from_timestamp": None,
        "valid_to_date": date(2026, 1, 31),
        "valid_to_timestamp": None,
        "time_precision": "day",
        "extraction_confidence": 0.8,
        "memory_kind": None,
        "sensitivity": None,
        "extraction_version_id": "extractor_1",
    }
    values.update(changes)
    return ClaimRecord(**values)


class StorageContractTests(unittest.TestCase):
    def test_json_accepts_unicode_scalars_lists_objects_and_nested_null(self) -> None:
        values = ["नमस्ते", 3, 2.5, True, ["é", 1], {"title": None, "city": "Pune"}]
        for value in values:
            self.assertEqual(safe_json(value, "value"), value)
        event = SourceEventRecord(
            source_id="source_1",
            user_id="user_1",
            source_type="chat",
            session_id=None,
            idempotency_key="key_1",
            produced_at=datetime(2026, 1, 1, tzinfo=UTC),
            ingested_at=datetime(2026, 1, 2, tzinfo=UTC),
            raw_content="नमस्ते",
            participants=["user_1"],
            metadata={"title": None, "label": "café"},
            content_hash=SHA256,
        )
        self.assertEqual(event.metadata["label"], "café")

    def test_json_rejects_non_json_keys_values_nan_and_infinity(self) -> None:
        for value in ({1: "value"}, {"bad": object()}, float("nan"), float("inf"), None):
            with self.subTest(value=repr(value)):
                with self.assertRaises(StorageValidationError):
                    safe_json(value, "value")

    def test_valid_dates_are_inclusive_and_never_mixed_with_timestamps(self) -> None:
        record = claim()
        self.assertTrue(record.valid_contains(date(2026, 1, 1)))
        self.assertTrue(record.valid_contains(date(2026, 1, 31)))
        self.assertFalse(record.valid_contains(date(2026, 2, 1)))
        with self.assertRaisesRegex(StorageValidationError, "cannot mix"):
            claim(valid_from_timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        with self.assertRaisesRegex(StorageValidationError, "unknown precision"):
            claim(time_precision="unknown")

    def test_timestamp_boundaries_require_offsets_and_remain_timestamps(self) -> None:
        with self.assertRaisesRegex(StorageValidationError, "timezone-aware"):
            claim(
                valid_from_date=None,
                valid_to_date=None,
                valid_from_timestamp=datetime(2026, 1, 1),
                time_precision="timestamp",
            )
        record = claim(
            valid_from_date=None,
            valid_to_date=None,
            valid_from_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            valid_to_timestamp=datetime(2026, 1, 1, 1, tzinfo=UTC),
            time_precision="timestamp",
        )
        self.assertTrue(record.valid_contains(datetime(2026, 1, 1, 1, tzinfo=UTC)))

    def test_transaction_intervals_are_start_inclusive_and_end_exclusive(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        version = ClaimVersionRecord(
            version_id="version_1",
            user_id="user_1",
            claim_id="claim_1",
            lifecycle_status="candidate",
            transaction_from=start,
            transaction_to=end,
        )
        self.assertTrue(version.transaction_contains(start))
        self.assertFalse(version.transaction_contains(end))
        with self.assertRaisesRegex(StorageValidationError, r"\[from, to\)"):
            ClaimVersionRecord(
                version_id="version_2",
                user_id="user_1",
                claim_id="claim_1",
                lifecycle_status="candidate",
                transaction_from=start,
                transaction_to=start,
            )

    def test_weak_claim_stays_candidate_with_nullable_phase4_fields(self) -> None:
        weak = claim(
            valid_from_date=None,
            valid_to_date=None,
            time_precision="unknown",
            memory_kind=None,
            sensitivity=None,
        )
        version = ClaimVersionRecord(
            version_id="version_1",
            user_id=weak.user_id,
            claim_id=weak.claim_id,
            lifecycle_status="candidate",
            transaction_from=datetime(2026, 1, 1, tzinfo=UTC),
            belief_confidence=None,
        )
        self.assertEqual(version.lifecycle_status, "candidate")
        self.assertIsNone(version.belief_confidence)
        self.assertFalse(weak.valid_contains(date(2026, 1, 1)))
        self.assertFalse(
            weak.valid_contains(datetime(2026, 1, 1, tzinfo=UTC))
        )
        with self.assertRaises(FrozenInstanceError):
            weak.claim_id = "changed"  # type: ignore[misc]

    def test_valid_time_queries_cannot_cross_date_representations(self) -> None:
        with self.assertRaisesRegex(StorageValidationError, "require a date query"):
            claim().valid_contains(datetime(2026, 1, 1, tzinfo=UTC))
        timestamp_claim = claim(
            valid_from_date=None,
            valid_to_date=None,
            valid_from_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            valid_to_timestamp=None,
            time_precision="timestamp",
        )
        with self.assertRaisesRegex(StorageValidationError, "timezone-aware"):
            timestamp_claim.valid_contains(date(2026, 1, 1))

    def test_enums_confidence_and_offsets_are_strict(self) -> None:
        with self.assertRaises(StorageValidationError):
            claim(polarity="maybe")
        for value in (-0.1, 1.1, Decimal("NaN"), True):
            with self.subTest(value=value):
                with self.assertRaises(StorageValidationError):
                    claim(extraction_confidence=value)
        with self.assertRaises(StorageValidationError):
            SourceSpanRecord(
                span_id="span_1",
                user_id="user_1",
                source_id="source_1",
                message_id=None,
                speaker_id="user_1",
                verbatim_quote="quote",
                start_offset=3,
                end_offset=3,
            )

    def test_migration_files_are_ordered_and_checksum_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "0002_second.sql").write_text("SELECT 2;\n", encoding="utf-8")
            (directory / "0001_first.sql").write_text("SELECT 1;\n", encoding="utf-8")
            migrations = load_migrations(directory)
            self.assertEqual([item.version for item in migrations], ["0001_first.sql", "0002_second.sql"])
            self.assertEqual(len(migrations[0].checksum), 64)
            (directory / "notes.txt").write_text("not a migration", encoding="utf-8")
            with self.assertRaises(MigrationError):
                load_migrations(directory)


if __name__ == "__main__":
    unittest.main()
