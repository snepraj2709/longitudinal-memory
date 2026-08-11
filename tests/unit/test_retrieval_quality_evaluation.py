from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from retrieval.quality_contracts import (
    MetricValue,
    RelevanceAnnotation,
    RelevanceCase,
    RetrievalQualityError,
)
from retrieval.quality_evaluation import (
    DATA_MANIFEST,
    RELEVANCE_PATH,
    _aggregate_metric,
    _latency_aggregates,
    _quality_aggregates,
    _score_one,
    _write_exclusive,
    load_relevance_cases,
)
from retrieval.quality_runtime import LatencySample


ROOT = Path(__file__).resolve().parents[2]


def annotation(number: int, kind: str, grade: int, *, stale: bool = False) -> RelevanceAnnotation:
    return RelevanceAnnotation(
        f"{number:064x}", kind, grade, stale, "corrected" if stale else None,
        ("conversation",), "implementation_reviewed",
    )


def case(*annotations: RelevanceAnnotation) -> RelevanceCase:
    ordered = tuple(sorted(annotations, key=lambda value: value.index_record_id))
    return RelevanceCase(
        "baseline_case_001", "baseline_query_001", "user_001", "development",
        "current_state", "temporal_reasoning", "focused", ordered,
    )


class RetrievalQualityContractTests(unittest.TestCase):
    def test_annotation_rejects_bad_grade_kind_stale_and_source_order(self) -> None:
        with self.assertRaises(RetrievalQualityError):
            annotation(1, "other", 1)
        with self.assertRaises(RetrievalQualityError):
            annotation(1, "atomic", 3)
        with self.assertRaises(RetrievalQualityError):
            RelevanceAnnotation("1" * 64, "atomic", 1, True, None, ("email",), "implementation_reviewed")
        with self.assertRaises(RetrievalQualityError):
            RelevanceAnnotation("1" * 64, "atomic", 1, False, None, ("email", "chat"), "implementation_reviewed")

    def test_case_requires_stable_unique_annotation_order(self) -> None:
        first, second = annotation(1, "atomic", 1), annotation(2, "session", 1)
        with self.assertRaisesRegex(RetrievalQualityError, "stable-ID"):
            RelevanceCase(
                "baseline_case_001", "baseline_query_001", "user_001", "development",
                "current_state", "extraction", "focused", (second, first),
            )

    def test_metric_null_and_defined_contract(self) -> None:
        self.assertIsNone(MetricValue(0, 0, None, "no_relevant_records").value)
        with self.assertRaises(RetrievalQualityError):
            MetricValue(0, 0, None, None)
        with self.assertRaises(RetrievalQualityError):
            MetricValue(1, 1, "1.000000", "bad")

    def test_checked_gold_has_exact_eight_cases_and_200_cells(self) -> None:
        cases = load_relevance_cases(ROOT / RELEVANCE_PATH)
        self.assertEqual(len(cases), 8)
        self.assertEqual(sum(len(value.annotations) for value in cases), 200)
        self.assertEqual([len(value.annotations) for value in cases], [24, 26] * 4)
        self.assertTrue(all(value.split == "development" for value in cases))

    def test_dataset_manifest_discloses_nonblind_boundary(self) -> None:
        manifest = json.loads((ROOT / DATA_MANIFEST).read_text())
        self.assertFalse(manifest["blind_evaluation"])
        self.assertTrue(manifest["prior_ranked_result_exposure_possible"])
        self.assertEqual(manifest["labeling_boundary"], "stable_same_user_candidate_order_without_rankings")

    def test_loader_rejects_missing_annotation_cell(self) -> None:
        rows = [json.loads(line) for line in (ROOT / RELEVANCE_PATH).read_text().splitlines()]
        rows[0]["annotations"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relevance.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(RetrievalQualityError, "universe"):
                load_relevance_cases(path)


class RetrievalQualityMetricTests(unittest.TestCase):
    def test_recall_k_truncation_and_no_hit_mrr(self) -> None:
        target = annotation(11, "atomic", 2)
        distractors = tuple(annotation(number, "atomic", 0) for number in range(1, 11))
        value = case(*distractors, target)
        score = _score_one(value, "B2", tuple(item.index_record_id for item in distractors) + (target.index_record_id,))
        self.assertEqual(score.recall_at_5.value, "0.000000")
        self.assertEqual(score.recall_at_10.value, "0.000000")
        self.assertEqual(score.mrr.value, "0.090909")

    def test_grade_two_has_larger_dcg_gain_than_grade_one(self) -> None:
        low, high = annotation(1, "atomic", 1), annotation(2, "atomic", 2)
        value = case(low, high)
        high_first = _score_one(value, "B2", (high.index_record_id, low.index_record_id))
        low_first = _score_one(value, "B2", (low.index_record_id, high.index_record_id))
        self.assertGreater(high_first.ndcg_at_10.value, low_first.ndcg_at_10.value)

    def test_no_relevant_records_returns_explicit_nulls(self) -> None:
        value = _score_one(case(annotation(1, "atomic", 0)), "B2", ())
        self.assertEqual(value.recall_at_10.null_reason, "no_relevant_records")
        self.assertEqual(value.ndcg_at_10.null_reason, "no_relevant_records")
        self.assertEqual(value.mrr.null_reason, "no_relevant_records")

    def test_b2_session_recall_has_baseline_null_reason(self) -> None:
        session = annotation(1, "session", 2)
        score = _score_one(case(session), "B2", ())
        self.assertEqual(score.relevant_session_recall.null_reason, "baseline_has_no_session_path")

    def test_session_recall_and_stale_rate_denominators(self) -> None:
        stale = annotation(1, "session", 2, stale=True)
        other = annotation(2, "session", 0)
        score = _score_one(case(stale, other), "B3", (stale.index_record_id, other.index_record_id))
        self.assertEqual(score.relevant_session_recall.value, "1.000000")
        self.assertEqual(score.stale_memory_rate.value, "0.500000")
        empty = _score_one(case(stale), "B3", ())
        self.assertEqual(empty.stale_memory_rate.null_reason, "no_accepted_records")

    def test_macro_quality_and_micro_stale_aggregation_differ(self) -> None:
        relevant = annotation(1, "atomic", 2, stale=True)
        first = _score_one(case(relevant), "B2", (relevant.index_record_id,))
        second = replace(first, case_id="baseline_case_002", accepted_count=9, stale_accepted_count=0,
                         stale_memory_rate=MetricValue(0, 9, "0.000000", None))
        rows = _quality_aggregates((first, second))
        overall = next(row for row in rows if row["baseline_id"] == "B2" and row["slice_dimension"] == "overall")
        self.assertEqual(overall["metrics"]["stale_memory_rate"]["value"], "0.100000")

    def test_source_type_slices_are_multi_membership(self) -> None:
        item = RelevanceAnnotation("1" * 64, "atomic", 2, False, None,
                                   ("chat", "email"), "implementation_reviewed")
        score = _score_one(case(item), "B2", (item.index_record_id,))
        rows = _quality_aggregates((score,))
        source_values = {row["slice_value"] for row in rows if row["slice_dimension"] == "source_type"}
        self.assertEqual(source_values, {"chat", "email"})

    def test_fixed_quality_and_latency_rounding(self) -> None:
        item = annotation(1, "atomic", 2)
        score = _score_one(case(item), "B2", (item.index_record_id,))
        aggregate = _aggregate_metric((score,), "recall_at_10", micro=False)
        self.assertEqual(aggregate["value"], "1.000000")
        samples = tuple(
            LatencySample("baseline_case_001", "baseline_query_001", "user_001", "B2", trial, 1_234_500 + trial, "a" * 64)
            for trial in range(1, 11)
        )
        rows = _latency_aggregates(samples, {"baseline_case_001": case(item)})
        overall = next(row for row in rows if row["slice_dimension"] == "overall")
        self.assertRegex(overall["mean_ms"], r"^\d+\.\d{3}$")

    def test_checked_calendar_title_is_irrelevant_for_happened_query(self) -> None:
        cases = load_relevance_cases(ROOT / RELEVANCE_PATH)
        happened = cases[3]
        calendar = {item.index_record_id[:10]: item for item in happened.annotations}
        self.assertEqual(calendar["cf5c83eae6"].relevance_grade, 0)
        self.assertEqual(calendar["fd7390ba12"].relevance_grade, 0)

    def test_checked_corrected_start_date_counts_stale_exactly(self) -> None:
        cases = load_relevance_cases(ROOT / RELEVANCE_PATH)
        evidence = cases[6]
        stale = [item for item in evidence.annotations if item.stale_for_query]
        self.assertEqual(len(stale), 2)
        self.assertEqual({item.stale_reason for item in stale}, {"corrected"})

    def test_session_without_required_qualification_cannot_substitute(self) -> None:
        qualified = annotation(1, "atomic", 2)
        unqualified = annotation(2, "session", 0)
        score = _score_one(case(qualified, unqualified), "B4", (unqualified.index_record_id,))
        self.assertEqual(score.recall_at_10.value, "0.000000")

    def test_cross_user_high_scorer_fails_closed(self) -> None:
        value = case(annotation(1, "atomic", 2))
        with self.assertRaisesRegex(RetrievalQualityError, "outside the relevance universe"):
            _score_one(value, "B2", ("f" * 64,))

    def test_checked_reported_by_other_claim_is_scored_for_its_subject(self) -> None:
        cases = load_relevance_cases(ROOT / RELEVANCE_PATH)
        historical_office = cases[1]
        records = {
            row["index_record_id"]: row
            for row in (
                json.loads(line)
                for line in (ROOT / "results/retrieval/index-development-v1/records.jsonl").read_text().splitlines()
            )
        }
        sofia_report = next(
            item
            for item in historical_office.annotations
            if records[item.index_record_id].get("speaker_id") == "sofia"
            and records[item.index_record_id].get("predicate") == "office_base"
        )
        self.assertEqual(records[sofia_report.index_record_id]["subject_id"], "user_002")
        self.assertEqual(sofia_report.relevance_grade, 2)

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            _write_exclusive(path, b"one")
            with self.assertRaisesRegex(RetrievalQualityError, "exists"):
                _write_exclusive(path, b"two")
            self.assertEqual(path.read_bytes(), b"one")


if __name__ == "__main__":
    unittest.main()
