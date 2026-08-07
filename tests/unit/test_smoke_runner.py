from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.history import EvaluationQuestion, HistoryObservation
from evaluation.smoke import (
    ModelRunConfig,
    SMOKE_CASE_IDS,
    SmokeTestSelectionError,
    main,
    run_smoke_test,
    select_smoke_questions,
    write_smoke_test_artifacts,
)


class FakeModelClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        case_id = user_prompt.splitlines()[0].removeprefix("Case ID: ")
        return self.responses[case_id]


class SmokeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        observed_at = datetime.fromisoformat("2026-04-01T10:00:00+05:30")
        as_of = datetime.fromisoformat("2026-08-01T10:00:00+05:30")
        self.observations = (
            HistoryObservation(
                observed_at=observed_at,
                source_type="conversation",
                source_id="conv_001",
                message_id="msg_001",
                author_id="i_am_maya",
                author_name="Maya",
                text="I accepted the Marketing Associate role.",
            ),
            HistoryObservation(
                observed_at=observed_at,
                source_type="calendar",
                source_id="cal_001",
                message_id=None,
                author_id="i_am_maya",
                author_name="Maya",
                text="Final semester examinations.",
            ),
        )
        self.questions = tuple(
            EvaluationQuestion(
                case_id=case_id,
                question=f"Question for {case_id}?",
                as_of=as_of,
            )
            for case_id in reversed(SMOKE_CASE_IDS)
        )
        self.model_config = ModelRunConfig(
            provider="fake",
            model="fake-answer-model",
            settings={"temperature": 0},
        )

    def prediction_json(
        self,
        case_id: str,
        *,
        source_id: str = "conv_001",
        message_id: str | None = "msg_001",
        quote: str = "Marketing Associate role",
    ) -> str:
        return json.dumps(
            {
                "case_id": case_id,
                "status": "answered",
                "answer": "Supported answer.",
                "confidence": 0.8,
                "evidence": [
                    {
                        "source_id": source_id,
                        "message_id": message_id,
                        "quote": quote,
                    }
                ],
                "abstention_reason": None,
            }
        )

    def valid_responses(self) -> dict[str, str]:
        return {case_id: self.prediction_json(case_id) for case_id in SMOKE_CASE_IDS}

    def test_selects_exact_five_cases_in_fixed_order(self) -> None:
        selected = select_smoke_questions(self.questions)

        self.assertEqual(tuple(item.case_id for item in selected), SMOKE_CASE_IDS)

    def test_rejects_missing_and_duplicated_cases(self) -> None:
        missing = tuple(
            item for item in self.questions if item.case_id != "temporal_003"
        )
        duplicated = self.questions + (self.questions[0],)

        with self.assertRaisesRegex(
            SmokeTestSelectionError, "missing smoke-test case 'temporal_003'"
        ):
            select_smoke_questions(missing)
        with self.assertRaisesRegex(
            SmokeTestSelectionError, "appears 2 times"
        ):
            select_smoke_questions(duplicated)

    def test_successful_json_contract_and_prompt_calls(self) -> None:
        client = FakeModelClient(self.valid_responses())

        run = run_smoke_test(
            self.questions, self.observations, client, self.model_config
        )

        self.assertEqual(run.exit_code, 0)
        self.assertTrue(all(result.passed for result in run.results))
        self.assertEqual(len(client.calls), 5)
        for system_prompt, user_prompt in client.calls:
            self.assertIn("Return exactly one JSON object", system_prompt)
            self.assertIn("<history>", user_prompt)

    def test_invalid_json_fails_with_raw_response_visible(self) -> None:
        responses = self.valid_responses()
        responses["conflict_004"] = "not json"

        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )
        result = run.results[2]

        self.assertEqual(run.exit_code, 1)
        self.assertFalse(result.valid_json)
        self.assertEqual(result.raw_response, "not json")
        self.assertIn("invalid JSON", result.validation_error)

    def test_wrong_returned_case_id_fails(self) -> None:
        responses = self.valid_responses()
        responses["temporal_003"] = self.prediction_json("wrong_case")

        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )
        result = run.results[1]

        self.assertTrue(result.valid_contract)
        self.assertFalse(result.case_id_matches)
        self.assertIn("does not match requested case", result.validation_error)

    def test_unknown_source_or_message_citation_fails(self) -> None:
        responses = self.valid_responses()
        responses["extraction_001"] = self.prediction_json(
            "extraction_001", message_id="unknown_message"
        )

        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )

        self.assertFalse(run.results[0].exact_evidence)
        self.assertIn("unknown source/message pair", run.results[0].validation_error)

    def test_quote_must_be_an_exact_substring(self) -> None:
        responses = self.valid_responses()
        responses["extraction_001"] = self.prediction_json(
            "extraction_001", quote="marketing associate role"
        )

        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )

        self.assertFalse(run.results[0].exact_evidence)
        self.assertIn("not an exact substring", run.results[0].validation_error)

    def test_calendar_evidence_with_null_message_id_is_valid(self) -> None:
        responses = self.valid_responses()
        responses["extraction_001"] = self.prediction_json(
            "extraction_001",
            source_id="cal_001",
            message_id=None,
            quote="semester examinations",
        )

        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )

        self.assertTrue(run.results[0].exact_evidence)

    def test_contract_failure_is_visible_in_diagnostics(self) -> None:
        responses = self.valid_responses()
        malformed = json.loads(responses["abstention_001"])
        del malformed["confidence"]
        responses["abstention_001"] = json.dumps(malformed)
        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_smoke_test_artifacts(run, directory)
            diagnostics = [
                json.loads(line)
                for line in artifacts.diagnostics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        failed = diagnostics[4]
        self.assertEqual(failed["raw_response"], responses["abstention_001"])
        self.assertIn("missing required field: confidence", failed["validation_error"])

    def test_runtime_prompts_contain_no_evaluation_side_gold_fields(self) -> None:
        client = FakeModelClient(self.valid_responses())

        run_smoke_test(self.questions, self.observations, client, self.model_config)

        prompt_text = "\n".join(
            system_prompt + "\n" + user_prompt
            for system_prompt, user_prompt in client.calls
        )
        for forbidden in (
            "answer_status",
            "expected_evidence",
            "capability",
            "difficulty",
            "failure_tags",
            "oracle_event_ids",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prompt_text)

    def test_jsonl_output_is_in_fixed_order_and_contains_only_passes(self) -> None:
        responses = self.valid_responses()
        responses["conflict_004"] = "invalid"
        run = run_smoke_test(
            self.questions,
            self.observations,
            FakeModelClient(responses),
            self.model_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_smoke_test_artifacts(run, directory)
            predictions = [
                json.loads(line)
                for line in artifacts.predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            report = artifacts.report_path.read_text(encoding="utf-8")

        self.assertEqual(
            [item["case_id"] for item in predictions],
            [case_id for case_id in SMOKE_CASE_IDS if case_id != "conflict_004"],
        )
        self.assertIn("Provider: fake", report)
        self.assertIn("Model: fake-answer-model", report)
        self.assertIn("| conflict_004 | no | no | no |", report)

    def test_dry_run_builds_only_five_prompts_without_model_calls(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        output = io.StringIO()

        exit_code = main(
            [
                "--dry-run",
                "--source-dir",
                str(repo_root / "data" / "pilot" / "sources"),
                "--questions",
                str(
                    repo_root
                    / "data"
                    / "pilot"
                    / "evaluation"
                    / "eval_questions.jsonl"
                ),
            ],
            stdout=output,
        )

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(exit_code, 0)
        self.assertEqual([item["case_id"] for item in records], list(SMOKE_CASE_IDS))
        self.assertEqual(len(records), 5)


if __name__ == "__main__":
    unittest.main()
