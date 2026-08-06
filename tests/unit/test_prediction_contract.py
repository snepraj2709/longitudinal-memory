from __future__ import annotations

import unittest

from evaluation import (
    BaselinePrediction,
    PredictionValidationError,
    validate_prediction,
)


class PredictionContractTests(unittest.TestCase):
    def answered_record(self) -> dict[str, object]:
        return {
            "case_id": "temporal_003",
            "status": "answered",
            "answer": "Aryan started his Bengaluru job on May 18, 2026.",
            "confidence": 0.96,
            "evidence": [
                {
                    "source_id": "conv_004",
                    "message_id": "msg_conv_004_002",
                    "quote": "I actually joined on the 18th, not the 11th.",
                }
            ],
            "abstention_reason": None,
        }

    def assert_invalid(self, record: object, message: str) -> PredictionValidationError:
        with self.assertRaises(PredictionValidationError) as raised:
            validate_prediction(record)
        self.assertIn(message, raised.exception.errors)
        return raised.exception

    def test_valid_answered_prediction(self) -> None:
        prediction = validate_prediction(self.answered_record())

        self.assertIsInstance(prediction, BaselinePrediction)
        self.assertEqual(prediction.status, "answered")
        self.assertEqual(prediction.evidence[0].source_id, "conv_004")

    def test_valid_abstention(self) -> None:
        record = self.answered_record()
        record.update(
            status="abstained",
            answer="The available history does not name Aryan's employer.",
            confidence=0,
            evidence=[],
            abstention_reason="no_source_evidence",
        )

        prediction = validate_prediction(record)

        self.assertEqual(prediction.status, "abstained")
        self.assertEqual(prediction.evidence, ())

    def test_valid_disputed_prediction(self) -> None:
        record = self.answered_record()
        record["status"] = "disputed"

        self.assertEqual(validate_prediction(record).status, "disputed")

    def test_valid_partially_answered_prediction(self) -> None:
        record = self.answered_record()
        record["status"] = "partially_answered"

        self.assertEqual(validate_prediction(record).status, "partially_answered")

    def test_valid_calendar_evidence_with_null_message_id(self) -> None:
        record = self.answered_record()
        record["evidence"] = [
            {
                "source_id": "cal_001",
                "message_id": None,
                "quote": "Final semester examinations.",
            }
        ]

        prediction = validate_prediction(record)

        self.assertIsNone(prediction.evidence[0].message_id)

    def test_unknown_status(self) -> None:
        record = self.answered_record()
        record["status"] = "complete"

        error = self.assert_invalid(
            record,
            "status must be one of: abstained, answered, disputed, partially_answered",
        )
        self.assertNotIn("answer_status", str(error))

    def test_non_string_status_reports_a_validation_error(self) -> None:
        record = self.answered_record()
        record["status"] = []

        self.assert_invalid(
            record,
            "status must be one of: abstained, answered, disputed, partially_answered",
        )

    def test_missing_required_field(self) -> None:
        record = self.answered_record()
        del record["case_id"]

        self.assert_invalid(record, "prediction is missing required field: case_id")

    def test_unknown_extra_field(self) -> None:
        record = self.answered_record()
        record["answer_status"] = "answered"

        self.assert_invalid(
            record, "prediction contains unknown field: answer_status"
        )

    def test_empty_required_strings(self) -> None:
        cases = {
            "case_id": ("case_id", "case_id must be a non-empty string"),
            "answer": ("answer", "answer must be a non-empty string"),
            "source_id": (
                "source_id",
                "evidence[0].source_id must be a non-empty string",
            ),
            "quote": ("quote", "evidence[0].quote must be a non-empty string"),
        }

        for name, (field, message) in cases.items():
            with self.subTest(name=name):
                record = self.answered_record()
                if field in {"source_id", "quote"}:
                    record["evidence"][0][field] = ""
                else:
                    record[field] = ""
                self.assert_invalid(record, message)

    def test_invalid_confidence_values(self) -> None:
        cases = (
            (-0.01, "confidence must be between 0 and 1 inclusive"),
            (1.01, "confidence must be between 0 and 1 inclusive"),
            (True, "confidence must be a real number"),
            (float("nan"), "confidence must be finite"),
            (float("inf"), "confidence must be finite"),
        )

        for confidence, message in cases:
            with self.subTest(confidence=confidence):
                record = self.answered_record()
                record["confidence"] = confidence
                self.assert_invalid(record, message)

    def test_answered_prediction_requires_evidence(self) -> None:
        record = self.answered_record()
        record["evidence"] = []

        self.assert_invalid(
            record, "answered predictions must include at least one evidence item"
        )

    def test_abstention_rejects_evidence(self) -> None:
        record = self.answered_record()
        record.update(status="abstained", abstention_reason="no_source_evidence")

        self.assert_invalid(
            record, "abstained predictions must have an empty evidence list"
        )

    def test_abstention_requires_reason(self) -> None:
        for reason in (None, ""):
            with self.subTest(reason=reason):
                record = self.answered_record()
                record.update(
                    status="abstained", evidence=[], abstention_reason=reason
                )
                self.assert_invalid(
                    record,
                    "abstained predictions must have a non-empty abstention_reason",
                )

    def test_non_abstention_rejects_abstention_reason(self) -> None:
        for status in ("answered", "disputed", "partially_answered"):
            with self.subTest(status=status):
                record = self.answered_record()
                record.update(status=status, abstention_reason="insufficient_evidence")
                self.assert_invalid(
                    record, f"{status} predictions must have a null abstention_reason"
                )

    def test_duplicate_evidence_references(self) -> None:
        record = self.answered_record()
        record["evidence"].append(
            {
                "source_id": "conv_004",
                "message_id": "msg_conv_004_002",
                "quote": "A different quote cannot make the reference unique.",
            }
        )

        self.assert_invalid(
            record,
            "evidence[1] duplicates evidence reference "
            "('conv_004', 'msg_conv_004_002')",
        )

    def test_malformed_input_that_is_not_an_object(self) -> None:
        for record in (None, [], "prediction"):
            with self.subTest(record=record):
                self.assert_invalid(record, "prediction must be an object")

    def test_evidence_must_be_a_list(self) -> None:
        record = self.answered_record()
        record["evidence"] = {}

        self.assert_invalid(record, "evidence must be a list")

    def test_evidence_item_shape_is_strict(self) -> None:
        record = self.answered_record()
        record["evidence"] = [
            {"source_id": "conv_004", "message_id": "msg_conv_004_002", "extra": 1}
        ]

        error = self.assert_invalid(
            record, "evidence[0] is missing required field: quote"
        )
        self.assertIn("evidence[0] contains unknown field: extra", error.errors)

    def test_empty_message_id_is_rejected(self) -> None:
        record = self.answered_record()
        record["evidence"][0]["message_id"] = ""

        self.assert_invalid(
            record,
            "evidence[0].message_id must be a non-empty string or null",
        )

    def test_validation_reports_all_record_errors(self) -> None:
        record = self.answered_record()
        del record["case_id"]
        record["answer"] = ""
        record["unexpected"] = True

        with self.assertRaises(PredictionValidationError) as raised:
            validate_prediction(record)

        self.assertEqual(
            raised.exception.errors,
            (
                "prediction is missing required field: case_id",
                "prediction contains unknown field: unexpected",
                "answer must be a non-empty string",
            ),
        )


if __name__ == "__main__":
    unittest.main()
