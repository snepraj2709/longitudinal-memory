from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from evaluation.score_b1 import score_completed_b1
from evaluation.scoring import (
    CAPABILITIES,
    abstention_metrics,
    aggregate_reference_metrics,
    answer_accuracy,
    answer_result,
    capability_metrics,
    compare_references,
    deterministic_answer_match,
    outdated_fact_error,
    rate,
    unsupported_claim_metrics,
)


class ScoringPrimitiveTests(unittest.TestCase):
    def test_equivalent_date_formats_match(self) -> None:
        for value in ("May 18, 2026", "18 May 2026", "2026-05-18"):
            with self.subTest(value=value):
                self.assertTrue(
                    deterministic_answer_match(value, "May 18, 2026.", [])
                )

    def test_different_dates_do_not_match(self) -> None:
        self.assertFalse(
            deterministic_answer_match("May 11, 2026", "May 18, 2026", [])
        )

    def test_normalized_name_comparison_is_exact(self) -> None:
        self.assertTrue(deterministic_answer_match("  PRAVIN. ", "Pravin", []))
        self.assertFalse(deterministic_answer_match("Pravin Kumar", "Pravin", []))

    def test_status_comparison_is_exact(self) -> None:
        self.assertTrue(deterministic_answer_match("APPROVED.", "approved", []))
        self.assertFalse(deterministic_answer_match("not approved", "approved", []))

    def test_acceptable_answers_are_compared_conservatively(self) -> None:
        self.assertTrue(
            deterministic_answer_match(
                "a marketing associate role",
                "Marketing Associate",
                ["a marketing associate role"],
            )
        )
        self.assertFalse(
            deterministic_answer_match(
                "She might consider product work later",
                "Product work",
                ["product work"],
            )
        )

    def test_false_positive_substrings_are_not_accepted(self) -> None:
        self.assertFalse(
            deterministic_answer_match("Pravin rejected it", "Pravin", [])
        )

    def test_source_reference_tp_fp_and_fn(self) -> None:
        result = compare_references({"a", "b"}, {"b", "c"})
        self.assertEqual(result["true_positives"], {"b"})
        self.assertEqual(result["false_positives"], {"a"})
        self.assertEqual(result["false_negatives"], {"c"})

    def test_source_micro_and_macro_metrics(self) -> None:
        aggregate = aggregate_reference_metrics(
            [
                compare_references({"a", "b"}, {"b", "c"}),
                compare_references({"d"}, {"d"}),
            ]
        )
        self.assertEqual(aggregate["micro_precision"]["value"], 0.666667)
        self.assertEqual(aggregate["micro_recall"]["value"], 0.666667)
        self.assertEqual(aggregate["macro_precision"]["numerator"], 1.5)
        self.assertEqual(aggregate["macro_precision"]["denominator"], 2)

    def test_message_pairs_do_not_compare_message_id_alone(self) -> None:
        result = compare_references(
            {("source_a", "message_1")}, {("source_b", "message_1")}
        )
        self.assertEqual(result["true_positives"], set())

    def test_calendar_null_message_id_is_a_normal_reference(self) -> None:
        result = compare_references({("cal_001", None)}, {("cal_001", None)})
        self.assertEqual(result["true_positives"], {("cal_001", None)})

    def test_abstention_confusion_matrix(self) -> None:
        result = abstention_metrics(
            [True, True, False, False], [True, False, True, False]
        )
        self.assertEqual(
            result["confusion_matrix"],
            {
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_negative": 1,
            },
        )
        self.assertEqual(result["precision"]["value"], 0.5)
        self.assertEqual(result["recall"]["value"], 0.5)

    def test_zero_denominator_returns_null_and_reason(self) -> None:
        result = rate(0, 0, "nothing to divide")
        self.assertIsNone(result["value"])
        self.assertEqual(result["null_reason"], "nothing to divide")

    def test_unsupported_claim_calculation(self) -> None:
        result = unsupported_claim_metrics(4, 1)
        self.assertEqual(result["numerator"], 1)
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["value"], 0.25)

    def test_current_fact_uses_corrected_value(self) -> None:
        self.assertTrue(
            outdated_fact_error(
                question="What is the corrected start date?",
                predicted_answer="May 11, 2026",
                reference_answer="May 18, 2026",
                acceptable_answers=[],
            )
        )

    def test_historical_question_is_not_marked_outdated(self) -> None:
        self.assertIsNone(
            outdated_fact_error(
                question="When did Maya start working at Infinity Learn?",
                predicted_answer="May 4, 2026",
                reference_answer="May 4, 2026",
                acceptable_answers=[],
            )
        )

    def test_all_five_capability_groups_are_required(self) -> None:
        scores = [
            _case_score(capability, "correct") for capability in CAPABILITIES
        ]
        result = capability_metrics(scores)
        self.assertEqual(tuple(result), CAPABILITIES)
        self.assertEqual(sum(item["total_cases"] for item in result.values()), 5)

    def test_strict_and_lenient_accuracy_are_distinct(self) -> None:
        result = answer_accuracy(
            [
                _case_score("extraction", "correct"),
                _case_score("extraction", "partial"),
                _case_score("extraction", "incorrect"),
                _case_score("extraction", "unresolved"),
            ]
        )
        self.assertEqual(result["strict_accuracy"]["value"], 0.25)
        self.assertEqual(result["lenient_accuracy"]["value"], 0.375)
        self.assertEqual(result["reviewed_accuracy"]["value"], 0.333333)

    def test_failed_prediction_is_incorrect(self) -> None:
        correctness, method, _ = answer_result(
            capability="extraction",
            question={"question": "Who?"},
            gold={"should_abstain": False},
            prediction=None,
            manual_review=None,
            execution_failure={"stage": "provider", "error": "failed"},
        )
        self.assertEqual((correctness, method), ("incorrect", "deterministic"))


class CompletedRunScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def _copy_run(self, destination: Path) -> None:
        shutil.copytree(self.repo_root / "data", destination / "data")
        shutil.copytree(self.repo_root / "configs", destination / "configs")
        target = destination / "results/pilot/b1-full-history"
        target.mkdir(parents=True)
        for name in ("run.json", "predictions.jsonl", "failures.jsonl"):
            shutil.copy2(
                self.repo_root / "results/pilot/b1-full-history" / name,
                target / name,
            )

    def test_unresolved_reviews_make_scores_provisional_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_run(root)
            result = score_completed_b1(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            case_scores = _read_jsonl(
                root / "results/pilot/b1-full-history/case_scores.jsonl"
            )
            questions = _read_jsonl(root / "data/pilot/evaluation/eval_questions.jsonl")

        self.assertEqual(result["result_status"], "provisional")
        self.assertGreater(result["unresolved_manual_review_count"], 0)
        self.assertEqual(
            [item["case_id"] for item in case_scores],
            [item["case_id"] for item in questions],
        )

    def test_scoring_preserves_inputs_and_repeats_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_run(root)
            predictions = root / "results/pilot/b1-full-history/predictions.jsonl"
            gold = root / "data/pilot/evaluation/eval_answer.jsonl"
            before = (_sha256(predictions), _sha256(gold))
            first = score_completed_b1(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            first_cases = (root / "results/pilot/b1-full-history/case_scores.jsonl").read_bytes()
            second = score_completed_b1(
                repo_root=root,
                now=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc),
            )
            second_cases = (root / "results/pilot/b1-full-history/case_scores.jsonl").read_bytes()
            after = (_sha256(predictions), _sha256(gold))

        self.assertEqual(before, after)
        first.pop("scored_at_utc")
        second.pop("scored_at_utc")
        self.assertEqual(first, second)
        self.assertEqual(first_cases, second_cases)

    def test_missing_prediction_counts_as_an_incomplete_execution_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_run(root)
            result_dir = root / "results/pilot/b1-full-history"
            predictions = _read_jsonl(result_dir / "predictions.jsonl")[:-1]
            (result_dir / "predictions.jsonl").write_text(
                "\n".join(json.dumps(item, separators=(",", ":")) for item in predictions)
                + "\n"
            )
            run = json.loads((result_dir / "run.json").read_text())
            run["attempts"] = run["attempts"][:-1]
            run["run_status"] = "completed_with_failures"
            run["recorded_cases"] = 24
            run["remaining_questions"] = 1
            run["successful_predictions"] = 24
            run["failed_predictions"] = 1
            (result_dir / "run.json").write_text(json.dumps(run))

            scores = score_completed_b1(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            cases = _read_jsonl(result_dir / "case_scores.jsonl")

        self.assertEqual(scores["result_status"], "incomplete")
        self.assertEqual(scores["execution_failure_count"], 1)
        self.assertEqual(scores["execution_failure_case_ids"], ["abstention_005"])
        self.assertEqual(cases[-1]["answer_correctness"], "incorrect")
        self.assertEqual(
            cases[-1]["execution_failure"]["stage"], "missing_prediction"
        )

    def test_gold_is_not_read_before_generation_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_run(root)
            run_path = root / "results/pilot/b1-full-history/run.json"
            run = json.loads(run_path.read_text())
            run["run_status"] = "running"
            run_path.write_text(json.dumps(run))
            (root / "data/pilot/evaluation/eval_answer.jsonl").unlink()

            with self.assertRaisesRegex(
                ValueError, "Step 5 run is not complete; gold data was not read"
            ):
                score_completed_b1(repo_root=root)

    def test_scorer_has_no_model_prompt_or_oracle_dependency(self) -> None:
        scoring_text = (
            (self.repo_root / "src/evaluation/scoring.py").read_text()
            + (self.repo_root / "src/evaluation/score_b1.py").read_text()
        ).casefold()
        self.assertNotIn("openai", scoring_text)
        self.assertNotIn("oracle-event", scoring_text)
        self.assertNotIn("build_history_prompt", scoring_text)


def _case_score(capability: str, correctness: str) -> dict[str, object]:
    return {
        "capability": capability,
        "answer_correctness": correctness,
        "execution_failure": None,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
