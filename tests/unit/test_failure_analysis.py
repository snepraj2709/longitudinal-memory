from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from evaluation.failure_analysis import (
    CAPABILITIES,
    FAILURE_TAXONOMY,
    FailureAnalysisError,
    attempt_has_execution_failure,
    expected_failure_case_ids,
    run_failure_analysis,
    structural_citation_failure,
    summarize_failures,
    validate_failure_record,
)


class FailureTaxonomyTests(unittest.TestCase):
    def test_every_required_taxonomy_value_is_present(self) -> None:
        self.assertEqual(
            FAILURE_TAXONOMY,
            (
                "missed_evidence",
                "wrong_date",
                "used_stale_information",
                "failed_to_apply_correction",
                "treated_assumption_as_fact",
                "unsupported_user_inference",
                "should_have_abstained",
                "abstained_despite_sufficient_evidence",
                "invalid_citation",
            ),
        )

    def test_unknown_taxonomy_value_is_rejected(self) -> None:
        record = _failure_record("missed_evidence")
        record["primary_classification"] = "other_failure"
        with self.assertRaisesRegex(FailureAnalysisError, "unknown failure"):
            validate_failure_record(record)

    def test_reasoning_failure_requires_one_primary_classification(self) -> None:
        record = _failure_record("missed_evidence")
        record["primary_classification"] = None
        with self.assertRaisesRegex(FailureAnalysisError, "one primary"):
            validate_failure_record(record)

    def test_multiple_contributing_classifications_are_allowed(self) -> None:
        record = _failure_record("missed_evidence")
        record["contributing_classifications"] = ["wrong_date", "used_stale_information"]
        record["wrong_date"] = {"predicted": "May 11", "expected": "May 18"}
        record["stale_evidence"] = [{"source_id": "conv_1", "message_id": "msg_1"}]
        self.assertEqual(
            validate_failure_record(record)["contributing_classifications"],
            ["wrong_date", "used_stale_information"],
        )

    def test_missed_evidence_classification(self) -> None:
        validate_failure_record(_failure_record("missed_evidence"))

    def test_wrong_date_classification(self) -> None:
        validate_failure_record(_failure_record("wrong_date"))

    def test_stale_information_classification(self) -> None:
        validate_failure_record(_failure_record("used_stale_information"))

    def test_explicit_correction_classification(self) -> None:
        validate_failure_record(_failure_record("failed_to_apply_correction"))

    def test_assumption_as_fact_classification(self) -> None:
        validate_failure_record(_failure_record("treated_assumption_as_fact"))

    def test_unsupported_user_inference_classification(self) -> None:
        validate_failure_record(_failure_record("unsupported_user_inference"))

    def test_should_have_abstained_classification(self) -> None:
        validate_failure_record(_failure_record("should_have_abstained"))

    def test_abstained_despite_evidence_classification(self) -> None:
        validate_failure_record(
            _failure_record("abstained_despite_sufficient_evidence")
        )

    def test_invalid_citation_classification(self) -> None:
        validate_failure_record(_failure_record("invalid_citation"))

    def test_semantic_mismatch_is_not_a_structural_citation_failure(self) -> None:
        attempt = {
            "valid_json": True,
            "valid_contract": True,
            "case_id_matches": True,
            "exact_evidence": True,
            "prediction": {"answer": "unsupported by this otherwise valid quote"},
        }
        self.assertFalse(structural_citation_failure(attempt))

    def test_malformed_attempt_is_an_execution_not_citation_failure(self) -> None:
        attempt = {
            "valid_json": False,
            "valid_contract": False,
            "case_id_matches": False,
            "exact_evidence": False,
            "failure_stage": "json_parse",
        }
        self.assertTrue(attempt_has_execution_failure(attempt))
        self.assertFalse(structural_citation_failure(attempt))

    def test_execution_failure_stays_outside_reasoning_taxonomy(self) -> None:
        record = _failure_record("missed_evidence")
        record["primary_classification"] = None
        record["missed_evidence"] = []
        record["execution_failure"] = {"stage": "provider", "error": "failed"}
        validate_failure_record(record)
        record["primary_classification"] = "missed_evidence"
        record["missed_evidence"] = [_evidence_detail()]
        with self.assertRaisesRegex(FailureAnalysisError, "remain separate"):
            validate_failure_record(record)

    def test_failure_selection_is_ordered_and_excludes_successes(self) -> None:
        case_scores = [
            _case_score("case_1"),
            _case_score("case_2", answer="partial"),
            _case_score("case_3", missed=True),
        ]
        attempts = {
            case_id: {
                "valid_json": True,
                "valid_contract": True,
                "case_id_matches": True,
                "exact_evidence": True,
            }
            for case_id in ("case_1", "case_2", "case_3")
        }
        self.assertEqual(
            expected_failure_case_ids(case_scores, attempts), ["case_2", "case_3"]
        )


class FailureSummaryTests(unittest.TestCase):
    def test_category_capability_and_multi_category_counts_match_records(self) -> None:
        first = _failure_record("missed_evidence", case_id="case_1")
        first["contributing_classifications"] = ["wrong_date"]
        first["wrong_date"] = {"predicted": "May 11", "expected": "May 18"}
        second = _failure_record("invalid_citation", case_id="case_2")
        second["capability"] = "temporal_reasoning"
        summary = _summary([first, second])

        self.assertEqual(summary["failed_cases"], 2)
        self.assertEqual(summary["category_counts"]["missed_evidence"], 1)
        self.assertEqual(summary["category_counts"]["wrong_date"], 1)
        self.assertEqual(summary["category_counts"]["invalid_citation"], 1)
        self.assertEqual(summary["failures_by_capability"]["extraction"]["count"], 1)
        self.assertEqual(
            summary["failures_by_capability"]["temporal_reasoning"]["count"], 1
        )
        self.assertEqual(summary["multi_category_cases"], ["case_1"])

    def test_unresolved_review_makes_summary_provisional(self) -> None:
        record = _failure_record("missed_evidence")
        record["reviewer_status"] = "unresolved"
        record["reviewer_identifier"] = None
        record["reviewed_at"] = None
        validate_failure_record(record)
        self.assertEqual(_summary([record])["report_status"], "provisional")

    def test_completed_reviews_make_summary_final(self) -> None:
        self.assertEqual(
            _summary([_failure_record("missed_evidence")])["report_status"],
            "final",
        )


class CompletedFailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def _copy_inputs(self, destination: Path) -> None:
        shutil.copytree(self.repo_root / "data", destination / "data")
        shutil.copytree(self.repo_root / "configs", destination / "configs")
        target = destination / "results/pilot/b1-full-history"
        target.mkdir(parents=True)
        for name in (
            "run.json",
            "predictions.jsonl",
            "failures.jsonl",
            "scores.json",
            "case_scores.jsonl",
            "manual_review.jsonl",
            "failure_analysis.jsonl",
        ):
            shutil.copy2(
                self.repo_root / "results/pilot/b1-full-history" / name,
                target / name,
            )

    def test_every_failed_case_appears_once_and_successes_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_inputs(root)
            summary = run_failure_analysis(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            records = _read_jsonl(
                root / "results/pilot/b1-full-history/failure_analysis.jsonl"
            )

        ids = [item["case_id"] for item in records]
        self.assertEqual(len(ids), len(set(ids)), 12)
        self.assertNotIn("extraction_001", ids)
        self.assertEqual(summary["failed_cases"], len(records))
        self.assertEqual(
            sum(item["count"] for item in summary["failures_by_capability"].values()),
            len(records),
        )

    def test_all_protected_hashes_remain_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_inputs(root)
            protected = _protected_files(root)
            before = {str(path): _sha256(path) for path in protected}
            summary = run_failure_analysis(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            after = {str(path): _sha256(path) for path in protected}

        self.assertEqual(before, after)
        self.assertTrue(summary["protected_artifacts_unchanged"])

    def test_repeat_analysis_changes_only_the_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_inputs(root)
            first = run_failure_analysis(
                repo_root=root,
                now=lambda: datetime(2026, 8, 7, tzinfo=timezone.utc),
            )
            first_findings = (
                root / "results/pilot/b1-full-history/baseline_findings.md"
            ).read_text()
            second = run_failure_analysis(
                repo_root=root,
                now=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc),
            )
            second_findings = (
                root / "results/pilot/b1-full-history/baseline_findings.md"
            ).read_text()

        first.pop("analysis_date_utc")
        second.pop("analysis_date_utc")
        self.assertEqual(first, second)
        self.assertEqual(first_findings, second_findings)

    def test_failure_analysis_uses_no_model_or_api_client(self) -> None:
        source = (
            self.repo_root / "src/evaluation/failure_analysis.py"
        ).read_text().casefold()
        self.assertNotIn("openai_client", source)
        self.assertNotIn("build_history_prompt", source)
        self.assertNotIn("complete_with_metadata", source)


def _failure_record(category: str, case_id: str = "case_1") -> dict[str, object]:
    record = {
        "case_id": case_id,
        "capability": "extraction",
        "question": "What happened?",
        "as_of": "2026-08-07T00:00:00Z",
        "result": "incorrect",
        "predicted_status": "answered",
        "predicted_answer": "Prediction.",
        "gold_status": "answered",
        "reference_answer": "Reference.",
        "primary_classification": category,
        "contributing_classifications": [],
        "execution_failure": None,
        "predicted_evidence": [],
        "expected_evidence": [],
        "relevant_evidence_inspected": [],
        "missed_evidence": [],
        "wrong_date": None,
        "stale_evidence": [],
        "correcting_evidence": [],
        "assumptions_treated_as_fact": [],
        "unsupported_claims": [],
        "abstention_error": None,
        "invalid_citations": [],
        "predicted_abstention": False,
        "expected_abstention": False,
        "concise_explanation": "The prediction failed for a specific reviewed reason.",
        "reviewer_status": "completed",
        "reviewer_identifier": "reviewer",
        "reviewed_at": "2026-08-07T00:00:00Z",
    }
    detail_field = {
        "missed_evidence": "missed_evidence",
        "wrong_date": "wrong_date",
        "used_stale_information": "stale_evidence",
        "failed_to_apply_correction": "correcting_evidence",
        "treated_assumption_as_fact": "assumptions_treated_as_fact",
        "unsupported_user_inference": "unsupported_claims",
        "should_have_abstained": "abstention_error",
        "abstained_despite_sufficient_evidence": "abstention_error",
        "invalid_citation": "invalid_citations",
    }[category]
    record[detail_field] = (
        {"predicted": "wrong", "expected": "right"}
        if detail_field in {"wrong_date", "abstention_error"}
        else [_evidence_detail()]
    )
    return record


def _evidence_detail() -> dict[str, object]:
    return {
        "source_id": "conv_1",
        "message_id": "msg_1",
        "quote": "Evidence.",
        "reason": "It resolves the question.",
    }


def _case_score(
    case_id: str, *, answer: str = "correct", missed: bool = False
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "answer_correctness": answer,
        "execution_failure": None,
        "message_false_negatives": ([{"source_id": "a", "message_id": "b"}] if missed else []),
        "unsupported_claim_count": 0,
        "current_vs_outdated_fact_error": None,
        "expected_abstention": False,
        "predicted_abstention": False,
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    return summarize_failures(
        records=records,
        total_questions=25,
        run={
            "baseline_id": "b1-full-history",
            "frozen_configuration_hash": "c" * 64,
            "dataset_hash": "d" * 64,
        },
        scores={
            "answer_correctness": {
                "fully_correct": 20,
                "partially_correct": 2,
                "incorrect": 3,
                "unresolved": 0,
            }
        },
        predictions_hash="p" * 64,
        scores_hash="s" * 64,
        protected_hashes={},
        reviewed_non_failures=[],
        analyzed_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
    )


def _protected_files(root: Path) -> list[Path]:
    result = root / "results/pilot/b1-full-history"
    return [
        root / "configs/full_history_baseline_v1.json",
        *sorted((root / "data/pilot").rglob("*.jsonl")),
        *(result / name for name in (
            "run.json",
            "predictions.jsonl",
            "failures.jsonl",
            "scores.json",
            "case_scores.jsonl",
            "manual_review.jsonl",
        )),
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
