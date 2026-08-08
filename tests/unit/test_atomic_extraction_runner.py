from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation.history import HistoryObservation
from evaluation.openai_client import OpenAIResponseMetadata
from extraction.atomic import (
    AtomicExtractionValidationError,
    extract_atomic_claims,
    sanitized_validation_diagnostics,
)
from extraction.contracts import AtomicClaimV1
from extraction.prompt import (
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
)
from extraction.source import ExtractionSource, KnownEntity


class FakeAtomicClient:
    def __init__(self, response: str | BaseException) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self.metadata = OpenAIResponseMetadata(
            response_id="resp_atomic_001",
            returned_model="fake-atomic-model",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response, self.metadata


class AtomicExtractionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = HistoryObservation(
            observed_at=datetime.fromisoformat("2026-04-05T19:30:00+05:30"),
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
            author_id="i_am_maya",
            author_name="Maya",
            text="I accepted the Marketing Associate role.",
        )
        self.second = HistoryObservation(
            observed_at=datetime.fromisoformat("2026-04-05T19:35:00+05:30"),
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_002",
            author_id="person_asha",
            author_name="Asha",
            text="Maya told me she accepted the role.",
        )
        self.source = ExtractionSource(
            source_id="conv_001",
            source_type="conversation",
            observations=(self.first, self.second),
            known_entities=(
                KnownEntity("i_am_maya", "Maya"),
                KnownEntity("person_asha", "Asha"),
            ),
        )

    def claim(
        self,
        *,
        claim_id: str = "claim_001",
        speaker_id: str = "i_am_maya",
        source_id: str = "conv_001",
        message_id: str | None = "msg_001",
        quote: str = "accepted the Marketing Associate role",
    ) -> dict[str, object]:
        return {
            "claim_id": claim_id,
            "subject_id": "i_am_maya",
            "speaker_id": speaker_id,
            "predicate": "accepted_offer",
            "object": "Marketing Associate role",
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": None,
            "valid_to": None,
            "confidence": 0.9,
            "evidence": [
                {
                    "source_id": source_id,
                    "message_id": message_id,
                    "quote": quote,
                }
            ],
        }

    def response(self, claims: object) -> str:
        return json.dumps({"claims": claims})

    def assert_invalid(
        self,
        response: str,
        message: str,
        source: ExtractionSource | None = None,
    ) -> AtomicExtractionValidationError:
        with self.assertRaises(AtomicExtractionValidationError) as raised:
            extract_atomic_claims(source or self.source, FakeAtomicClient(response))
        self.assertEqual(raised.exception.source_id, (source or self.source).source_id)
        self.assertIn(message, str(raised.exception))
        return raised.exception

    def test_valid_claim_uses_prompt_and_returns_metadata(self) -> None:
        client = FakeAtomicClient(self.response([self.claim()]))

        result = extract_atomic_claims(self.source, client)

        self.assertEqual(result.source_id, "conv_001")
        self.assertEqual(len(result.claims), 1)
        self.assertIsInstance(result.claims[0], AtomicClaimV1)
        self.assertEqual(result.response_metadata, client.metadata)
        self.assertEqual(
            client.calls,
            [
                (
                    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
                    build_atomic_extraction_prompt(self.source),
                )
            ],
        )
        self.assertFalse(hasattr(result, "raw_response"))
        with self.assertRaises(FrozenInstanceError):
            result.source_id = "conv_002"

    def test_multiple_claims_are_ordered_by_evidence_time_then_claim_id(self) -> None:
        later = self.claim(
            claim_id="claim_later",
            speaker_id="person_asha",
            message_id="msg_002",
            quote="Maya told me she accepted the role",
        )
        first_b = self.claim(claim_id="claim_b")
        first_a = self.claim(claim_id="claim_a")

        result = extract_atomic_claims(
            self.source,
            FakeAtomicClient(self.response([later, first_b, first_a])),
        )

        self.assertEqual(
            [claim.claim_id for claim in result.claims],
            ["claim_a", "claim_b", "claim_later"],
        )

    def test_empty_claims_are_valid(self) -> None:
        result = extract_atomic_claims(
            self.source, FakeAtomicClient(self.response([]))
        )

        self.assertEqual(result.claims, ())

    def test_normalizes_unicode_punctuation_to_the_unique_exact_source_span(self) -> None:
        source = ExtractionSource(
            source_id="conv_quoted",
            source_type="conversation",
            observations=(
                HistoryObservation(
                    observed_at=datetime.fromisoformat("2026-05-10T20:17:00+05:30"),
                    source_type="conversation",
                    source_id="conv_quoted",
                    message_id="msg_quoted_001",
                    author_id="i_am_maya",
                    author_name="Maya",
                    text="He's coming here? I want to talk to him.",
                ),
            ),
            known_entities=(KnownEntity("i_am_maya", "Maya"),),
        )
        claim = self.claim(
            source_id="conv_quoted",
            message_id="msg_quoted_001",
            quote="He’s coming here?",
        )

        result = extract_atomic_claims(
            source,
            FakeAtomicClient(self.response([claim])),
            evidence_normalization_version="unicode_punctuation_v1",
        )

        self.assertEqual(result.claims[0].evidence[0].quote, "He's coming here?")
        self.assertEqual(
            [(item.code, item.location) for item in result.normalization_diagnostics],
            [("evidence_quote_unicode_punctuation_normalized", "claims[0].evidence[0].quote")],
        )

    def test_does_not_normalize_an_ambiguous_unicode_quote(self) -> None:
        repeated = HistoryObservation(
            observed_at=datetime.fromisoformat("2026-05-10T20:17:00+05:30"),
            source_type="conversation",
            source_id="conv_repeated",
            message_id="msg_repeated_001",
            author_id="i_am_maya",
            author_name="Maya",
            text="He's ready. He's ready.",
        )
        source = ExtractionSource(
            "conv_repeated",
            "conversation",
            (repeated,),
            (KnownEntity("i_am_maya", "Maya"),),
        )
        claim = self.claim(
            source_id="conv_repeated",
            message_id="msg_repeated_001",
            quote="He’s ready.",
        )

        with self.assertRaisesRegex(
            AtomicExtractionValidationError,
            "quote is not an exact substring",
        ):
            extract_atomic_claims(
                source,
                FakeAtomicClient(self.response([claim])),
                evidence_normalization_version="unicode_punctuation_v1",
            )

    def test_source_span_fallback_uses_the_cited_observation_and_is_audited(self) -> None:
        claim = self.claim(quote="A paraphrase that is not a source substring.")

        result = extract_atomic_claims(
            self.source,
            FakeAtomicClient(self.response([claim])),
            evidence_normalization_version="source_span_v1",
        )

        self.assertEqual(result.claims[0].evidence[0].quote, self.first.text)
        self.assertEqual(
            [(item.code, item.location) for item in result.normalization_diagnostics],
            [
                (
                    "evidence_quote_replaced_with_cited_observation",
                    "claims[0].evidence[0].quote",
                )
            ],
        )

    def test_normalizes_false_boolean_object_into_negative_polarity(self) -> None:
        claim = self.claim()
        claim.update(
            {
                "predicate": "has_handoff_cover",
                "object": False,
                "polarity": "positive",
            }
        )

        result = extract_atomic_claims(
            self.source,
            FakeAtomicClient(self.response([claim])),
            evidence_normalization_version="source_span_boolean_polarity_v2",
        )

        self.assertIs(result.claims[0].object, True)
        self.assertEqual(result.claims[0].polarity, "negative")
        self.assertEqual(
            [(item.code, item.location) for item in result.normalization_diagnostics],
            [("claim_boolean_polarity_normalized", "claims[0].object")],
        )

    def test_rejects_malformed_json_without_exposing_raw_response(self) -> None:
        raw_response = "not-json-sensitive-provider-output"

        error = self.assert_invalid(raw_response, "response is not valid JSON")

        self.assertNotIn(raw_response, str(error))
        self.assertTrue(error.__suppress_context__)

    def test_validation_diagnostics_keep_only_codes_and_locations(self) -> None:
        error = AtomicExtractionValidationError(
            "conv_001",
            [
                "claims[0]: polarity must be one of: negative, positive",
                "claims[0].evidence[0].quote is not an exact substring of the cited text",
            ],
        )

        self.assertEqual(
            [
                {"code": item.code, "location": item.location}
                for item in sanitized_validation_diagnostics(error)
            ],
            [
                {"code": "claim_invalid_polarity", "location": "claims[0].polarity"},
                {
                    "code": "evidence_inexact_quote",
                    "location": "claims[0].evidence[0].quote",
                },
            ],
        )

    def test_rejects_non_object_response(self) -> None:
        self.assert_invalid("[]", "response must be an object")

    def test_rejects_missing_and_extra_top_level_fields(self) -> None:
        self.assert_invalid("{}", "missing required field: claims")
        self.assert_invalid(
            json.dumps({"claims": [], "status": "ok"}),
            "unknown field: status",
        )

    def test_rejects_non_list_claims(self) -> None:
        self.assert_invalid(self.response({}), "claims must be a list")

    def test_rejects_invalid_step_1_claim_data(self) -> None:
        claim = self.claim()
        claim["polarity"] = "neutral"

        self.assert_invalid(
            self.response([claim]),
            "claims[0]: polarity must be one of: negative, positive",
        )

    def test_rejects_duplicate_claim_ids(self) -> None:
        self.assert_invalid(
            self.response([self.claim(), self.claim()]),
            "duplicates claim_id 'claim_001'",
        )

    def test_rejects_unknown_source_and_message_ids(self) -> None:
        self.assert_invalid(
            self.response([self.claim(source_id="conv_unknown")]),
            "source_id must match source 'conv_001'",
        )
        self.assert_invalid(
            self.response([self.claim(message_id="msg_unknown")]),
            "message_id does not exist in source 'conv_001'",
        )

    def test_rejects_invalid_calendar_message_reference(self) -> None:
        observation = HistoryObservation(
            observed_at=datetime.fromisoformat("2026-04-01T09:00:00+05:30"),
            source_type="calendar",
            source_id="cal_001",
            message_id=None,
            author_id="i_am_maya",
            author_name="Maya",
            text="Final semester examinations.",
        )
        source = ExtractionSource(
            "cal_001",
            "calendar",
            (observation,),
            (KnownEntity("i_am_maya", "Maya"),),
        )
        valid = self.claim(
            source_id="cal_001",
            message_id=None,
            quote="Final semester examinations",
        )
        invalid = self.claim(
            source_id="cal_001",
            message_id="calendar_msg",
            quote="Final semester examinations",
        )

        result = extract_atomic_claims(
            source, FakeAtomicClient(self.response([valid]))
        )
        self.assertIsNone(result.claims[0].evidence[0].message_id)
        self.assert_invalid(
            self.response([invalid]),
            "calendar evidence must use message_id: null",
            source,
        )

    def test_rejects_quote_mismatch(self) -> None:
        self.assert_invalid(
            self.response([self.claim(quote="This quote is absent")]),
            "quote is not an exact substring of the cited text",
        )

    def test_rejects_unknown_and_unsupported_speaker_ids(self) -> None:
        self.assert_invalid(
            self.response([self.claim(speaker_id="person_unknown")]),
            "speaker_id 'person_unknown' does not exist in the source",
        )
        self.assert_invalid(
            self.response(
                [
                    self.claim(
                        speaker_id="person_asha",
                        message_id="msg_001",
                    )
                ]
            ),
            "speaker_id 'person_asha' does not match any cited observation",
        )

    def test_rejects_unknown_subject_id(self) -> None:
        claim = self.claim()
        claim["subject_id"] = "person_unknown"

        self.assert_invalid(
            self.response([claim]),
            "subject_id 'person_unknown' does not exist in known_entities",
        )

    def test_provider_failures_propagate(self) -> None:
        failure = RuntimeError("provider unavailable")

        with self.assertRaises(RuntimeError) as raised:
            extract_atomic_claims(self.source, FakeAtomicClient(failure))

        self.assertIs(raised.exception, failure)

    def test_extraction_writes_no_files(self) -> None:
        client = FakeAtomicClient(self.response([self.claim()]))

        with patch("builtins.open") as open_file, patch.object(
            Path, "write_text"
        ) as write_text, patch.object(Path, "write_bytes") as write_bytes:
            result = extract_atomic_claims(self.source, client)

        self.assertEqual(len(result.claims), 1)
        open_file.assert_not_called()
        write_text.assert_not_called()
        write_bytes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
