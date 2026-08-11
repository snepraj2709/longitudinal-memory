from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from storage.contracts import (
    ClaimVersionRecord,
    LifecycleTransitionRecord,
    StorageValidationError,
)
from temporal.contracts import (
    ACCEPTED_STATUSES,
    CorrectionRequest,
    TRANSITION_MATRIX,
    TemporalQuery,
    TransitionRequest,
)


UTC = timezone.utc


class TemporalContractTests(unittest.TestCase):
    def test_transition_matrix_is_exact_and_terminal_states_have_no_successors(self) -> None:
        self.assertEqual(
            TRANSITION_MATRIX,
            {
                "candidate": frozenset(
                    {"confirmed", "current", "historical", "disputed", "excluded"}
                ),
                "confirmed": frozenset({"disputed", "superseded", "excluded"}),
                "current": frozenset({"historical", "disputed", "superseded", "excluded"}),
                "historical": frozenset({"disputed", "superseded", "excluded"}),
                "disputed": frozenset({"confirmed", "current", "historical", "superseded", "excluded"}),
                "superseded": frozenset(),
                "excluded": frozenset(),
            },
        )
        with self.assertRaisesRegex(StorageValidationError, "target_status is invalid"):
            LifecycleTransitionRecord(
                "transition", "user", "key", "claim", "from", "to",
                "candidate", "reason", None, datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_transition_requires_explicit_bounded_reason_and_aware_time(self) -> None:
        with self.assertRaises(StorageValidationError):
            TransitionRequest("u", "c", "k", "current", "", datetime(2026, 1, 1))
        with self.assertRaisesRegex(StorageValidationError, "at most 500"):
            TransitionRequest(
                "u", "c", "k", "current", "x" * 501,
                datetime(2026, 1, 1, tzinfo=UTC),
            )
        with self.assertRaises(StorageValidationError):
            TransitionRequest(
                "u", "c", "k", "candidate", "reason",
                datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_correction_names_distinct_claims_and_only_accepted_replacement(self) -> None:
        with self.assertRaises(StorageValidationError):
            CorrectionRequest(
                "u", "c", "c", "k", "current", "reason",
                datetime(2026, 1, 1, tzinfo=UTC),
            )
        with self.assertRaises(StorageValidationError):
            CorrectionRequest(
                "u", "old", "new", "k", "disputed", "reason",
                datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_version_snapshots_use_inclusive_valid_and_half_open_transaction_time(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        version = ClaimVersionRecord(
            "v", "u", "c", "historical", start, end, 0.7,
            date(2026, 1, 3), None, date(2026, 1, 5), None, "day",
        )
        self.assertTrue(version.transaction_contains(start))
        self.assertFalse(version.transaction_contains(end))
        self.assertTrue(version.valid_contains(date(2026, 1, 3)))
        self.assertTrue(version.valid_contains(date(2026, 1, 5)))
        self.assertFalse(version.valid_contains(date(2026, 1, 6)))

    def test_null_valid_time_never_matches_and_cannot_be_current(self) -> None:
        unknown = ClaimVersionRecord(
            "v", "u", "c", "candidate", datetime(2026, 1, 1, tzinfo=UTC)
        )
        self.assertFalse(unknown.valid_contains(date(2026, 1, 1)))
        with self.assertRaisesRegex(StorageValidationError, "known valid time"):
            ClaimVersionRecord(
                "v2", "u", "c", "current", datetime(2026, 1, 1, tzinfo=UTC)
            )

    def test_query_defaults_to_accepted_states_and_rejects_mixed_time_types(self) -> None:
        query = TemporalQuery("u", datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(query.statuses, ACCEPTED_STATUSES)
        with self.assertRaises(StorageValidationError):
            TemporalQuery("u", datetime(2026, 1, 1))
        with self.assertRaises(StorageValidationError):
            TemporalQuery(
                "u", datetime(2026, 1, 1, tzinfo=UTC), valid_at="2026-01-01"  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
