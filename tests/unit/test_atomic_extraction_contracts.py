from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from extraction import AtomicClaimV1, AtomicClaimValidationError, validate_atomic_claim


class AtomicExtractionContractTests(unittest.TestCase):
    def claim_record(self) -> dict[str, object]:
        return {
            "claim_id": "claim_001",
            "subject_id": "i_am_maya",
            "speaker_id": "maya100",
            "predicate": "will_have_job_role",
            "object": "marketing associate",
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": "2026-05-04",
            "valid_to": None,
            "confidence": 0.98,
            "evidence": [
                {
                    "source_id": "conv_002",
                    "message_id": "msg_conv_002_002",
                    "quote": "I join Infinity Learn on May 4 as a marketing associate.",
                }
            ],
        }

    def assert_invalid(
        self, record: object, message: str
    ) -> AtomicClaimValidationError:
        with self.assertRaises(AtomicClaimValidationError) as raised:
            validate_atomic_claim(record)
        self.assertIn(message, raised.exception.errors)
        return raised.exception

    def test_valid_claim(self) -> None:
        claim = validate_atomic_claim(self.claim_record())

        self.assertIsInstance(claim, AtomicClaimV1)
        self.assertEqual(claim.valid_from, "2026-05-04")
        self.assertIsNone(claim.valid_to)
        self.assertEqual(claim.evidence[0].source_id, "conv_002")

        with self.assertRaises(FrozenInstanceError):
            claim.subject_id = "someone_else"
        with self.assertRaises(FrozenInstanceError):
            claim.evidence[0].quote = "changed"

    def test_calendar_evidence_allows_null_message_id(self) -> None:
        record = self.claim_record()
        record["evidence"][0].update(
            source_id="cal_001",
            message_id=None,
            quote="Final semester examinations.",
        )

        claim = validate_atomic_claim(record)

        self.assertIsNone(claim.evidence[0].message_id)

    def test_times_allow_null_dates_and_timezone_aware_datetimes(self) -> None:
        record = self.claim_record()
        record.update(valid_from=None, valid_to=None)
        claim = validate_atomic_claim(record)
        self.assertIsNone(claim.valid_from)
        self.assertIsNone(claim.valid_to)

        record.update(
            valid_from="2026-05-04T08:30:00+05:30",
            valid_to="2026-05-04T09:30:00+05:30",
        )
        claim = validate_atomic_claim(record)
        self.assertEqual(claim.valid_from, record["valid_from"])
        self.assertEqual(claim.valid_to, record["valid_to"])

    def test_missing_extra_and_empty_fields_are_reported_together(self) -> None:
        record = self.claim_record()
        del record["claim_id"]
        record["subject_id"] = ""
        record["status"] = "current"
        del record["evidence"][0]["quote"]
        record["evidence"][0]["span_id"] = "span_001"

        error = self.assert_invalid(record, "claim is missing required field: claim_id")

        self.assertIn("claim contains unknown field: status", error.errors)
        self.assertIn("subject_id must be a non-empty string", error.errors)
        self.assertIn("evidence[0] is missing required field: quote", error.errors)
        self.assertIn("evidence[0] contains unknown field: span_id", error.errors)

    def test_invalid_enums_are_rejected(self) -> None:
        record = self.claim_record()
        record["polarity"] = "neutral"
        record["epistemic_status"] = "known"
        record["predicate"] = "free_form_predicate"

        error = self.assert_invalid(
            record, "polarity must be one of: negative, positive"
        )
        self.assertIn(
            "epistemic_status must be one of: asserted, corrected, denied, "
            "hypothetical, inferred, reported_by_other, uncertain",
            error.errors,
        )
        self.assertTrue(
            any(error.startswith("predicate must be one of:") for error in error.errors)
        )

    def test_invalid_confidence_is_rejected(self) -> None:
        cases = (
            (True, "confidence must be a real number"),
            (float("inf"), "confidence must be finite"),
            (-0.01, "confidence must be between 0 and 1 inclusive"),
            (1.01, "confidence must be between 0 and 1 inclusive"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                record = self.claim_record()
                record["confidence"] = value
                self.assert_invalid(record, message)

    def test_invalid_or_naive_times_are_rejected(self) -> None:
        for value in ("May 4, 2026", "2026-02-30", "2026-05-04T08:30:00"):
            with self.subTest(value=value):
                record = self.claim_record()
                record["valid_from"] = value
                self.assert_invalid(
                    record,
                    "valid_from must be an ISO date, timezone-aware ISO datetime, "
                    "or null",
                )

    def test_reversed_time_ranges_are_rejected(self) -> None:
        ranges = (
            ("2026-05-05", "2026-05-04"),
            (
                "2026-05-04T09:00:00+05:30",
                "2026-05-04T08:59:59+05:30",
            ),
        )
        for valid_from, valid_to in ranges:
            with self.subTest(valid_from=valid_from, valid_to=valid_to):
                record = self.claim_record()
                record.update(valid_from=valid_from, valid_to=valid_to)
                self.assert_invalid(record, "valid_to must not be before valid_from")

    def test_empty_evidence_is_rejected(self) -> None:
        record = self.claim_record()
        record["evidence"] = []

        self.assert_invalid(record, "evidence must contain at least one item")

    def test_duplicate_evidence_references_are_rejected(self) -> None:
        record = self.claim_record()
        record["evidence"].append(
            {
                "source_id": "conv_002",
                "message_id": "msg_conv_002_002",
                "quote": "A different quote does not make the reference unique.",
            }
        )

        self.assert_invalid(
            record,
            "evidence[1] duplicates evidence reference "
            "('conv_002', 'msg_conv_002_002')",
        )


if __name__ == "__main__":
    unittest.main()
