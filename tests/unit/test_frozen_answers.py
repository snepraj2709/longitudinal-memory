from __future__ import annotations

import unittest

from evaluation.frozen_answer_contracts import validate_answer_output
from evaluation.frozen_answers import _render_user_prompt, _safe_failure_code
from evaluation.openai_client import OpenAIResponseError
from evaluation.frozen_run_contracts import FrozenRunError


class FrozenAnswerContractTests(unittest.TestCase):
    def test_v2_serialized_input_explicitly_requests_json(self):
        prompt = _render_user_prompt(
            "qa",
            {
                "case_id": "case_1", "user_id": "user_001",
                "as_of": "2026-08-10T00:00:00Z", "question": "What changed?",
            },
            [],
        )
        self.assertIn('"response_format":"JSON object"', prompt)

    def test_http_failure_code_uses_only_safe_provider_parameter(self):
        error = OpenAIResponseError(
            "OpenAI API returned HTTP 400",
            status_code=400,
            provider_code="unsupported_value",
            provider_param="text.format.type",
            provider_reason="unsupported_value",
        )
        self.assertEqual(
            _safe_failure_code(error, "provider"),
            "provider_http_400_text_format_type_unsupported_value",
        )

    def test_grounded_output_requires_exact_context_citation(self):
        evidence = {("source_1", "message_1", "Exact quote."): object()}
        output = {
            "status": "answered", "answer": "A grounded fact.", "confidence": 0.8,
            "statements": ["A grounded fact."],
            "citations": [{"source_id": "source_1", "message_id": "message_1", "quote": "Exact quote."}],
            "unresolved_parts": [], "abstention_reason": None,
        }
        self.assertEqual(
            validate_answer_output(output, task="qa", evidence_index=evidence)["status"],
            "answered",
        )
        changed = {**output, "citations": [{"source_id": "source_1", "message_id": "message_1", "quote": "Changed."}]}
        with self.assertRaises(FrozenRunError):
            validate_answer_output(changed, task="qa", evidence_index=evidence)

    def test_abstention_contract_is_exact(self):
        output = {
            "status": "abstained", "summary": "Not enough reliable memory.",
            "confidence": 0, "statements": [], "citations": [],
            "unresolved_parts": [], "abstention_reason": "insufficient_evidence",
        }
        validate_answer_output(output, task="summary", evidence_index={})
        changed = {**output, "confidence": 0.1}
        with self.assertRaises(FrozenRunError):
            validate_answer_output(changed, task="summary", evidence_index={})

    def test_unknown_fields_and_untracked_prose_are_rejected(self):
        output = {
            "status": "answered", "response": "Extra prose. Fact.", "confidence": 0.7,
            "statements": ["Different fact."],
            "citations": [{"source_id": "s", "message_id": None, "quote": "q"}],
            "unresolved_parts": [], "abstention_reason": None,
        }
        with self.assertRaises(FrozenRunError):
            validate_answer_output(output, task="interactive", evidence_index={("s", None, "q"): object()})
        output["unknown"] = True
        with self.assertRaises(FrozenRunError):
            validate_answer_output(output, task="interactive", evidence_index={})


if __name__ == "__main__":
    unittest.main()
