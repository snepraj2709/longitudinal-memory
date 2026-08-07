from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.b1_full_history import (
    B1QuestionError,
    B1RunError,
    PILOT_CASE_IDS,
    ensure_output_directory_safe,
    execute_b1_pipeline,
    require_all_pilot_questions,
    run_b1_questions,
    verify_completed_run_matches_config,
)
from evaluation.history import load_evaluation_questions, load_history_observations
from evaluation.openai_client import OpenAIResponseMetadata
from evaluation.run_config import (
    RunConfigurationError,
    canonical_sha256,
    load_frozen_config,
    validate_frozen_config,
    verify_frozen_content,
)


class FakeB1Client:
    def __init__(self, model: str, responses: dict[str, object] | None = None) -> None:
        self.model = model
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        self.calls.append((system_prompt, user_prompt))
        case_id = next(
            line[len("Case ID: ") :]
            for line in user_prompt.splitlines()
            if line.startswith("Case ID: ")
        )
        response = self.responses.get(case_id, _abstention_response(case_id))
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            raw_response, returned_model = response
        else:
            raw_response, returned_model = response, self.model
        position = len(self.calls)
        return raw_response, OpenAIResponseMetadata(
            response_id=f"resp_fake_{position:03d}",
            returned_model=returned_model,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )


def _abstention_response(case_id: str) -> str:
    return json.dumps(
        {
            "case_id": case_id,
            "status": "abstained",
            "answer": "The supplied history does not establish this answer.",
            "confidence": 0.0,
            "evidence": [],
            "abstention_reason": "The supplied history does not contain enough evidence.",
        }
    )


def _answered_response(
    case_id: str,
    *,
    source_id: str,
    message_id: str | None,
    quote: str,
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


class FullHistoryB1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.questions_path = (
            cls.repo_root / "data/pilot/evaluation/eval_questions.jsonl"
        )
        cls.gold_path = cls.repo_root / "data/pilot/evaluation/eval_answer.jsonl"
        cls.questions = load_evaluation_questions(cls.questions_path)
        cls.observations = load_history_observations(
            cls.repo_root / "data/pilot/sources"
        )
        cls.config = load_frozen_config(
            cls.repo_root / "configs/full_history_baseline_v1.json"
        )

    def _run(self, responses: dict[str, object] | None = None):
        client = FakeB1Client(self.config.resolved_model, responses)
        results = run_b1_questions(
            self.questions,
            self.observations,
            client,
            self.config,
        )
        return client, results

    def _execute(
        self,
        directory: str,
        *,
        responses: dict[str, object] | None = None,
        scorer=None,
    ):
        client = FakeB1Client(self.config.resolved_model, responses)
        artifacts = execute_b1_pipeline(
            repo_root=directory,
            output_dir="results/pilot/b1-full-history",
            questions_path=self.questions_path,
            gold_path=self.gold_path,
            questions=self.questions,
            observations=self.observations,
            config=self.config,
            client=client,
            repository_commit="a" * 40,
            scorer=scorer,
            now=lambda: datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
        )
        return client, artifacts

    def test_selects_all_25_unique_questions_in_file_order(self) -> None:
        selected = require_all_pilot_questions(self.questions)

        self.assertEqual(len(selected), 25)
        self.assertEqual(tuple(item.case_id for item in selected), PILOT_CASE_IDS)

    def test_rejects_missing_duplicate_and_reordered_case_ids(self) -> None:
        with self.subTest("missing"):
            with self.assertRaisesRegex(B1QuestionError, "exactly 25"):
                require_all_pilot_questions(self.questions[:-1])
        with self.subTest("duplicate"):
            duplicated = (*self.questions[:-1], self.questions[0])
            with self.assertRaises(B1QuestionError) as raised:
                require_all_pilot_questions(duplicated)
            self.assertIn("duplicate case IDs", str(raised.exception))
            self.assertIn("missing case IDs", str(raised.exception))
        with self.subTest("order"):
            reordered = list(self.questions)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            with self.assertRaisesRegex(B1QuestionError, "order"):
                require_all_pilot_questions(reordered)

    def test_rejects_frozen_provider_model_and_settings_mismatch(self) -> None:
        valid = {
            "provider": self.config.provider,
            "requested_model": self.config.requested_model,
            "returned_model": self.config.resolved_model,
            "temperature": self.config.temperature,
            "generation_settings": self.config.generation_settings,
        }
        for field, value in (
            ("provider", "Other"),
            ("requested_model", "floating-model"),
            ("returned_model", "different-snapshot"),
            ("temperature", 0.5),
            ("generation_settings", {"api": "other"}),
        ):
            with self.subTest(field=field):
                values = dict(valid)
                values[field] = value
                with self.assertRaisesRegex(B1RunError, field.replace("_", " ")):
                    verify_completed_run_matches_config(self.config, **values)

    def test_rejects_prompt_dataset_and_ordering_drift(self) -> None:
        with self.subTest("prompt"):
            changed = replace(self.config, prompt_sha256="0" * 64)
            with self.assertRaisesRegex(RunConfigurationError, "prompt text"):
                verify_frozen_content(changed, self.repo_root)
        with self.subTest("dataset"):
            changed = replace(self.config, dataset_sha256="0" * 64)
            with self.assertRaisesRegex(RunConfigurationError, "pilot data"):
                verify_frozen_content(changed, self.repo_root)
        with self.subTest("ordering"):
            record = json.loads(
                (self.repo_root / "configs/full_history_baseline_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            record["source_ordering_rule"] = "different ordering"
            record["configuration_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in record.items()
                    if key != "configuration_sha256"
                }
            )
            with self.assertRaisesRegex(RunConfigurationError, "ordering_rule"):
                validate_frozen_config(record)

    def test_successful_fake_client_executes_all_25_sequentially(self) -> None:
        client, results = self._run()

        self.assertEqual(len(client.calls), 25)
        self.assertTrue(all(item.passed for item in results))
        self.assertEqual(tuple(item.case_id for item in results), PILOT_CASE_IDS)

    def test_partial_provider_failure_does_not_retry_or_reorder(self) -> None:
        client, results = self._run(
            {"temporal_003": RuntimeError("provider unavailable")}
        )

        failed = next(item for item in results if item.case_id == "temporal_003")
        self.assertEqual(len(client.calls), 25)
        self.assertEqual(failed.failure_stage, "provider")
        self.assertEqual(failed.error, "provider unavailable")
        self.assertEqual(tuple(item.case_id for item in results), PILOT_CASE_IDS)

    def test_provider_model_mismatch_stops_later_calls(self) -> None:
        response = (
            _abstention_response("extraction_001"),
            "different-model-snapshot",
        )
        client, results = self._run({"extraction_001": response})

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(results[0].failure_stage, "provider_model_mismatch")
        self.assertEqual(results[1].failure_stage, "not_attempted")
        self.assertFalse(results[1].attempted)

    def test_invalid_json_records_raw_response(self) -> None:
        _, results = self._run({"extraction_001": "not json"})
        failed = results[0]

        self.assertEqual(failed.failure_stage, "json")
        self.assertEqual(failed.raw_response, "not json")
        self.assertFalse(failed.valid_json)

    def test_invalid_prediction_contract_is_visible(self) -> None:
        _, results = self._run(
            {"extraction_001": json.dumps({"case_id": "extraction_001"})}
        )

        self.assertEqual(results[0].failure_stage, "prediction_contract")
        self.assertTrue(results[0].valid_json)
        self.assertIn("missing required field", results[0].error)

    def test_incorrect_returned_case_id_is_rejected(self) -> None:
        _, results = self._run(
            {"extraction_001": _abstention_response("wrong_case")}
        )

        self.assertEqual(results[0].failure_stage, "case_id")
        self.assertTrue(results[0].valid_contract)

    def test_unknown_evidence_reference_is_rejected(self) -> None:
        response = _answered_response(
            "extraction_001",
            source_id="unknown",
            message_id="missing",
            quote="Unknown quote.",
        )
        _, results = self._run({"extraction_001": response})

        self.assertEqual(results[0].failure_stage, "evidence")
        self.assertIn("unknown source/message pair", results[0].error)

    def test_quote_mismatch_is_rejected(self) -> None:
        response = _answered_response(
            "extraction_001",
            source_id="email_001",
            message_id="msg_email_001_001",
            quote="This quote was never in the email.",
        )
        _, results = self._run({"extraction_001": response})

        self.assertEqual(results[0].failure_stage, "evidence")
        self.assertIn("exact substring", results[0].error)

    def test_calendar_citation_with_null_message_id_passes(self) -> None:
        response = _answered_response(
            "temporal_001",
            source_id="cal_001",
            message_id=None,
            quote=(
                "Final semester examinations. April 20, 9:00 AM to April 29, "
                "12:00 PM."
            ),
        )
        _, results = self._run({"temporal_001": response})
        result = next(item for item in results if item.case_id == "temporal_001")

        self.assertTrue(result.passed)

    def test_runtime_prompts_contain_no_gold_or_oracle_fields(self) -> None:
        client, _ = self._run()
        prompts = "\n".join(system + "\n" + user for system, user in client.calls)

        for forbidden in (
            "reference_answer",
            "acceptable_answers",
            "answer_status",
            "oracle_fact_ids",
            "failure_tags",
            "difficulty",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prompts)

    def test_pipeline_writes_predictions_failures_scores_and_run_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client, artifacts = self._execute(
                directory,
                responses={"temporal_003": RuntimeError("temporary provider failure")},
            )
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            predictions = _read_jsonl(output_dir / "predictions.jsonl")
            failures = _read_jsonl(output_dir / "failures.jsonl")
            scores = json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))
            run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
            artifact_names = sorted(path.name for path in output_dir.iterdir())

        self.assertEqual(len(client.calls), 25)
        self.assertEqual(artifacts.status, "completed_with_failures")
        self.assertEqual(len(predictions), 24)
        self.assertEqual(
            [item["case_id"] for item in predictions],
            [case_id for case_id in PILOT_CASE_IDS if case_id != "temporal_003"],
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["case_id"], "temporal_003")
        self.assertEqual(failures[0]["question_position"], 8)
        self.assertEqual(failures[0]["failure_stage"], "provider")
        self.assertEqual(scores["valid_prediction_count"], 24)
        self.assertEqual(scores["failed_case_count"], 1)
        self.assertIsNone(scores["semantic_answer_score"])
        self.assertEqual(run["calls_attempted"], 25)
        self.assertTrue(run["gold_answer_hash_unchanged"])
        self.assertEqual(
            artifact_names,
            ["failures.jsonl", "predictions.jsonl", "run.json", "scores.json"],
        )

    def test_deterministic_scores_are_transparent_without_semantic_grading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, artifacts = self._execute(directory)
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            scores = json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))
            failures_text = (output_dir / "failures.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(artifacts.status, "completed")
        self.assertEqual(scores["total_cases"], 25)
        self.assertEqual(scores["valid_prediction_count"], 25)
        self.assertEqual(scores["failed_case_count"], 0)
        self.assertEqual(scores["json_valid_rate"]["value"], 1.0)
        self.assertEqual(scores["prediction_contract_valid_rate"]["value"], 1.0)
        self.assertEqual(scores["exact_citation_valid_rate"]["value"], 1.0)
        self.assertEqual(scores["answered_coverage"]["value"], 0.0)
        self.assertEqual(scores["status_counts"]["abstained"], 25)
        self.assertEqual(scores["abstention_precision"]["value"], 0.2)
        self.assertEqual(scores["abstention_recall"]["value"], 1.0)
        self.assertEqual(scores["abstention_accuracy"]["value"], 0.2)
        self.assertEqual(scores["evidence_reference_recall"]["value"], 0.0)
        self.assertEqual(len(scores["results_by_capability"]), 5)
        self.assertEqual(len(scores["cases_requiring_manual_semantic_review"]), 25)
        self.assertIsNone(scores["semantic_answer_score"])
        self.assertTrue(scores["semantic_answer_score_reason"])
        self.assertEqual(failures_text, "")

    def test_scoring_runs_only_after_generation_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            client = FakeB1Client(self.config.resolved_model)

            def scorer(questions_path, gold_path, attempts):
                self.assertEqual(len(client.calls), 25)
                self.assertTrue((output_dir / "predictions.jsonl").is_file())
                self.assertTrue((output_dir / "failures.jsonl").is_file())
                return {"total_cases": len(attempts), "semantic_answer_score": None}

            execute_b1_pipeline(
                repo_root=directory,
                output_dir="results/pilot/b1-full-history",
                questions_path=self.questions_path,
                gold_path=self.gold_path,
                questions=self.questions,
                observations=self.observations,
                config=self.config,
                client=client,
                repository_commit="a" * 40,
                scorer=scorer,
                now=lambda: datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
            )

    def test_gold_file_remains_byte_for_byte_unchanged(self) -> None:
        before = hashlib.sha256(self.gold_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            self._execute(directory)
            run = json.loads(
                (
                    Path(directory)
                    / "results/pilot/b1-full-history/run.json"
                ).read_text(encoding="utf-8")
            )
        after = hashlib.sha256(self.gold_path.read_bytes()).hexdigest()

        self.assertEqual(before, after)
        self.assertEqual(run["gold_answer_hash_before"], before)
        self.assertEqual(run["gold_answer_hash_after"], after)
        self.assertTrue(run["gold_answer_hash_unchanged"])

    def test_safe_rerun_refuses_existing_same_or_different_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            output_dir.mkdir(parents=True)
            (output_dir / "run.json").write_text(
                json.dumps(
                    {"frozen_configuration_hash": self.config.configuration_sha256}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(B1RunError, "same frozen configuration"):
                ensure_output_directory_safe(
                    output_dir, self.config.configuration_sha256
                )
            (output_dir / "run.json").write_text(
                json.dumps({"frozen_configuration_hash": "0" * 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(B1RunError, "different frozen configuration"):
                ensure_output_directory_safe(
                    output_dir, self.config.configuration_sha256
                )

    def test_interrupted_run_resumes_only_remaining_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            interrupted_client = FakeB1Client(
                self.config.resolved_model,
                {"temporal_003": KeyboardInterrupt("simulated interruption")},
            )
            with self.assertRaisesRegex(KeyboardInterrupt, "simulated"):
                execute_b1_pipeline(
                    repo_root=directory,
                    output_dir="results/pilot/b1-full-history",
                    questions_path=self.questions_path,
                    gold_path=self.gold_path,
                    questions=self.questions,
                    observations=self.observations,
                    config=self.config,
                    client=interrupted_client,
                    repository_commit="a" * 40,
                    now=lambda: datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
                )

            output_dir = Path(directory) / "results/pilot/b1-full-history"
            checkpoint = json.loads(
                (output_dir / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["run_status"], "running")
            self.assertEqual(checkpoint["recorded_cases"], 7)
            self.assertEqual(checkpoint["calls_attempted"], 7)
            self.assertEqual(len(_read_jsonl(output_dir / "predictions.jsonl")), 7)
            self.assertFalse((output_dir / "scores.json").exists())

            resumed_client = FakeB1Client(self.config.resolved_model)
            artifacts = execute_b1_pipeline(
                repo_root=directory,
                output_dir="results/pilot/b1-full-history",
                questions_path=self.questions_path,
                gold_path=self.gold_path,
                questions=self.questions,
                observations=self.observations,
                config=self.config,
                client=resumed_client,
                repository_commit="a" * 40,
                now=lambda: datetime(2026, 8, 7, 12, 1, tzinfo=timezone.utc),
                resume=True,
            )
            final_run = json.loads(
                (output_dir / "run.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(interrupted_client.calls), 8)
        self.assertEqual(len(resumed_client.calls), 18)
        self.assertEqual(artifacts.status, "completed")
        self.assertEqual(final_run["calls_attempted"], 25)
        self.assertEqual(final_run["resume_count"], 1)
        self.assertEqual(final_run["remaining_questions"], 0)

    def test_resume_rejects_incompatible_or_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            output_dir.mkdir(parents=True)
            (output_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_status": "running",
                        "frozen_configuration_hash": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(B1RunError, "different frozen configuration"):
                ensure_output_directory_safe(
                    output_dir,
                    self.config.configuration_sha256,
                    allow_compatible_resume=True,
                )
            (output_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_status": "completed",
                        "frozen_configuration_hash": self.config.configuration_sha256,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(B1RunError, "same frozen configuration"):
                ensure_output_directory_safe(
                    output_dir,
                    self.config.configuration_sha256,
                    allow_compatible_resume=True,
                )

    def test_resume_retries_only_provider_failures_without_model_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_client, first_artifacts = self._execute(
                directory,
                responses={"temporal_003": RuntimeError("rate limited")},
            )
            resumed_client = FakeB1Client(self.config.resolved_model)
            resumed_artifacts = execute_b1_pipeline(
                repo_root=directory,
                output_dir="results/pilot/b1-full-history",
                questions_path=self.questions_path,
                gold_path=self.gold_path,
                questions=self.questions,
                observations=self.observations,
                config=self.config,
                client=resumed_client,
                repository_commit="a" * 40,
                now=lambda: datetime(2026, 8, 7, 12, 1, tzinfo=timezone.utc),
                resume=True,
            )
            run = json.loads(
                (
                    Path(directory)
                    / "results/pilot/b1-full-history/run.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(first_artifacts.status, "completed_with_failures")
        self.assertEqual(len(first_client.calls), 25)
        self.assertEqual(len(resumed_client.calls), 1)
        self.assertEqual(resumed_artifacts.status, "completed")
        self.assertEqual(run["calls_attempted"], 25)
        self.assertEqual(run["provider_requests_attempted"], 26)
        self.assertEqual(len(run["prior_provider_failures"]), 1)

    def test_cost_cap_stops_before_a_request_that_would_exceed_it(self) -> None:
        client = FakeB1Client(self.config.resolved_model)

        results = run_b1_questions(
            self.questions,
            self.observations,
            client,
            self.config,
            max_cost_usd=Decimal("0.000001"),
        )

        self.assertEqual(client.calls, [])
        self.assertEqual(results[0].failure_stage, "cost_cap")
        self.assertTrue(all(not item.attempted for item in results))

    def test_artifacts_do_not_contain_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._execute(directory)
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            combined = "\n".join(
                path.read_text(encoding="utf-8") for path in output_dir.iterdir()
            )

        self.assertNotIn("OPENAI_API_KEY", combined)
        self.assertNotIn("api_key", combined.lower())
        self.assertNotIn("authorization", combined.lower())
        self.assertNotIn("Bearer ", combined)
        self.assertNotIn("sk-", combined)
        self.assertNotIn('"system_prompt"', combined)
        self.assertNotIn('"user_prompt"', combined)
        self.assertNotIn("<history>", combined)

    def test_secret_shaped_raw_failure_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RunConfigurationError):
                self._execute(
                    directory,
                    responses={"extraction_001": "sk-sensitive-looking-value"},
                )
            output_dir = Path(directory) / "results/pilot/b1-full-history"
            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output_dir.iterdir()
                if path.is_file()
            )

        self.assertNotIn("sk-sensitive-looking-value", combined)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
