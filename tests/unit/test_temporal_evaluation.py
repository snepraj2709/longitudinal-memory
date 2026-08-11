from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from extraction.scaled_source import load_scaled_development_sources
from evaluation import run_temporal as runner

from evaluation.temporal import (
    DATASET_VERSION,
    TemporalEvaluationError,
    TemporalFailure,
    TemporalGoldCase,
    TemporalPrediction,
    interval_relation_and_iou,
    load_temporal_gold,
    load_temporal_runtime,
    score_temporal,
    serialize_jsonl,
)


def runtime_record(position: int) -> dict[str, object]:
    user_id = "user_001" if position < 6 else "user_002"
    return {
        "case_id": f"temporal_case_{position + 1:03d}",
        "dataset_version": DATASET_VERSION,
        "split": "development",
        "user_id": user_id,
        "tags": ["boundary"],
        "sources": [],
        "spans": [],
        "claims": [],
        "commands": [],
        "query": {},
        "interval_pairs": [],
    }


def gold_case(case_id: str = "case_1", user_id: str = "user_001") -> TemporalGoldCase:
    return TemporalGoldCase(
        case_id,
        DATASET_VERSION,
        "development",
        user_id,
        "approved",
        ("source_1",),
        (
            {
                "claim_id": "claim_1",
                "valid_from": "2026-01-01",
                "valid_to": "2026-01-10",
                "time_precision": "day",
            },
        ),
        (
            {
                "left_claim_id": "claim_1",
                "right_claim_id": "claim_2",
                "relation": "overlaps",
                "iou": 0.05,
                "iou_reason": None,
            },
        ),
        ("claim_1",),
        (),
        ({"claim_id": "claim_1", "version_id": "version_1", "status": "current"},),
    )


def prediction(case_id: str = "case_1", user_id: str = "user_001") -> TemporalPrediction:
    expected = gold_case(case_id, user_id)
    return TemporalPrediction(
        case_id,
        user_id,
        expected.expected_ordered_source_ids or (),
        expected.expected_normalized_claims or (),
        expected.expected_intervals or (),
        expected.expected_current_claim_ids or (),
        expected.expected_historical_claim_ids or (),
        expected.expected_visible_versions or (),
    )


class TemporalEvaluationUnitTests(unittest.TestCase):
    def write_jsonl(self, records: list[dict[str, object]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cases.jsonl"
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
        return path

    def test_runtime_schema_enforces_release_identity_exact_ids_and_six_per_user(self) -> None:
        records = [runtime_record(index) for index in range(12)]
        cases = load_temporal_runtime(self.write_jsonl(records))
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(case.user_id == "user_001" for case in cases), 6)
        self.assertEqual(sum(case.user_id == "user_002" for case in cases), 6)

        invalid = [dict(item) for item in records]
        invalid[0]["extra"] = True
        with self.assertRaisesRegex(TemporalEvaluationError, "fields are invalid"):
            load_temporal_runtime(self.write_jsonl(invalid))

        duplicate = [dict(item) for item in records]
        duplicate[1]["case_id"] = duplicate[0]["case_id"]
        with self.assertRaisesRegex(TemporalEvaluationError, "must be unique"):
            load_temporal_runtime(self.write_jsonl(duplicate))

    def test_runtime_rejects_test_users_wrong_split_and_duplicate_nested_ids(self) -> None:
        records = [runtime_record(index) for index in range(12)]
        for field, value, message in (
            ("user_id", "user_003", "development users"),
            ("split", "test", "release identity"),
        ):
            changed = [dict(item) for item in records]
            changed[0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(TemporalEvaluationError, message):
                    load_temporal_runtime(self.write_jsonl(changed))

        changed = [dict(item) for item in records]
        changed[0]["sources"] = [
            {"source_id": "same"},
            {"source_id": "same"},
        ]
        with self.assertRaisesRegex(TemporalEvaluationError, "source IDs"):
            load_temporal_runtime(self.write_jsonl(changed))

    def test_gold_schema_requires_approved_records_and_valid_interval_values(self) -> None:
        records = []
        for index in range(12):
            case = gold_case(
                f"temporal_case_{index + 1:03d}",
                "user_001" if index < 6 else "user_002",
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "dataset_version": case.dataset_version,
                    "split": case.split,
                    "user_id": case.user_id,
                    "review_status": case.review_status,
                    "expected_ordered_source_ids": list(case.expected_ordered_source_ids or ()),
                    "expected_normalized_claims": list(case.expected_normalized_claims or ()),
                    "expected_intervals": list(case.expected_intervals or ()),
                    "expected_current_claim_ids": list(case.expected_current_claim_ids or ()),
                    "expected_historical_claim_ids": list(case.expected_historical_claim_ids or ()),
                    "expected_visible_versions": list(case.expected_visible_versions or ()),
                }
            )
        self.assertEqual(len(load_temporal_gold(self.write_jsonl(records))), 12)
        records[0]["review_status"] = "draft"
        with self.assertRaisesRegex(TemporalEvaluationError, "approved"):
            load_temporal_gold(self.write_jsonl(records))
        records[0]["review_status"] = "approved"
        records[0]["expected_intervals"][0]["iou"] = float("nan")  # type: ignore[index]
        with self.assertRaisesRegex(TemporalEvaluationError, "IoU is invalid"):
            load_temporal_gold(self.write_jsonl(records))

    def test_date_relations_use_inclusive_endpoints_and_iou(self) -> None:
        left = {
            "valid_from": "2026-01-01",
            "valid_to": "2026-01-10",
            "time_precision": "day",
        }
        touching = {
            "valid_from": "2026-01-10",
            "valid_to": "2026-01-20",
            "time_precision": "day",
        }
        relation, iou, reason = interval_relation_and_iou(left, touching)
        self.assertEqual((relation, iou, reason), ("overlaps", 0.05, None))
        contained = dict(touching, valid_from="2026-01-03", valid_to="2026-01-04")
        self.assertEqual(interval_relation_and_iou(left, contained)[0], "contains")
        endpoint = dict(touching, valid_from="2026-01-10", valid_to="2026-01-10")
        self.assertEqual(interval_relation_and_iou(left, endpoint)[0], "overlaps")
        after = dict(touching, valid_from="2026-02-01", valid_to="2026-02-02")
        self.assertEqual(interval_relation_and_iou(left, after)[0], "before")

    def test_timestamp_relations_normalize_offsets_and_score_points(self) -> None:
        utc_point = {
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": "2026-01-01T00:00:00Z",
            "time_precision": "timestamp",
        }
        offset_point = {
            "valid_from": "2026-01-01T05:30:00+05:30",
            "valid_to": "2026-01-01T05:30:00+05:30",
            "time_precision": "timestamp",
        }
        self.assertEqual(
            interval_relation_and_iou(utc_point, offset_point),
            ("equal", 1.0, None),
        )
        later = dict(
            offset_point,
            valid_from="2026-01-02T05:30:00+05:30",
            valid_to="2026-01-02T05:30:00+05:30",
        )
        self.assertEqual(interval_relation_and_iou(utc_point, later), ("before", 0.0, None))

    def test_unknown_or_open_intervals_have_explicit_null_reason(self) -> None:
        unknown = {"valid_from": None, "valid_to": None, "time_precision": "unknown"}
        known = {
            "valid_from": "2026-01-01",
            "valid_to": "2026-01-02",
            "time_precision": "day",
        }
        self.assertEqual(
            interval_relation_and_iou(unknown, known),
            ("unknown", None, "unknown_or_open_interval"),
        )
        open_interval = dict(known, valid_to=None)
        self.assertEqual(
            interval_relation_and_iou(open_interval, known),
            ("unknown", None, "unknown_or_open_interval"),
        )

    def test_failures_remain_in_every_applicable_denominator(self) -> None:
        expected = gold_case()
        failure = TemporalFailure("failure_1", "case_1", "user_001", "query_failed", "query")
        scores = score_temporal((), (failure,), (expected,))
        for name in (
            "event_ordering_accuracy",
            "date_normalization_accuracy",
            "interval_relation_accuracy",
            "mean_interval_iou",
            "current_state_accuracy",
            "historical_state_accuracy",
            "correction_visibility_accuracy",
        ):
            metric = getattr(scores, name)
            self.assertEqual(metric["denominator"], 1)
            self.assertEqual(metric["value"], 0.0)

    def test_current_historical_and_correction_metrics_compare_exact_sets(self) -> None:
        expected = replace(
            gold_case(),
            expected_current_claim_ids=("claim_1", "claim_2"),
            expected_historical_claim_ids=("claim_3", "claim_4"),
            expected_visible_versions=(
                {"claim_id": "claim_1", "version_id": "version_1", "status": "current"},
                {"claim_id": "claim_2", "version_id": "version_2", "status": "superseded"},
            ),
        )
        actual = replace(
            prediction(),
            current_claim_ids=("claim_2", "claim_1"),
            historical_claim_ids=("claim_4", "claim_3"),
            visible_versions=tuple(reversed(expected.expected_visible_versions or ())),
        )
        scores = score_temporal((actual,), (), (expected,))
        self.assertEqual(scores.current_state_accuracy["value"], 1.0)
        self.assertEqual(scores.historical_state_accuracy["value"], 1.0)
        self.assertEqual(scores.correction_visibility_accuracy["value"], 1.0)

    def test_zero_denominators_are_null_with_reason(self) -> None:
        expected = replace(
            gold_case(),
            expected_ordered_source_ids=None,
            expected_normalized_claims=None,
            expected_intervals=None,
            expected_current_claim_ids=None,
            expected_historical_claim_ids=None,
            expected_visible_versions=None,
        )
        scores = score_temporal((prediction(),), (), (expected,))
        for name in (
            "event_ordering_accuracy",
            "date_normalization_accuracy",
            "interval_relation_accuracy",
            "mean_interval_iou",
            "current_state_accuracy",
            "historical_state_accuracy",
            "correction_visibility_accuracy",
        ):
            self.assertEqual(
                getattr(scores, name),
                {"value": None, "numerator": 0, "denominator": 0, "null_reason": "zero_denominator"},
            )

    def test_case_coverage_and_ids_are_strict(self) -> None:
        expected = gold_case()
        with self.assertRaisesRegex(TemporalEvaluationError, "cover every gold case"):
            score_temporal((), (), (expected,))
        with self.assertRaisesRegex(TemporalEvaluationError, "both prediction and failure"):
            score_temporal(
                (prediction(),),
                (TemporalFailure("failure_1", "case_1", "user_001", "failed", "case"),),
                (expected,),
            )
        with self.assertRaisesRegex(TemporalEvaluationError, "sanitized token"):
            TemporalFailure("failure 1", "case_1", "user_001", "failed", "case")

    def test_serialization_is_stable_sorted_utf8_jsonl(self) -> None:
        value = prediction()
        first = serialize_jsonl((value,))
        second = serialize_jsonl((value,))
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        decoded = json.loads(first)
        self.assertEqual(list(decoded), sorted(decoded))

    def test_frozen_fixture_is_reviewed_source_backed_and_split_six_per_user(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runtime_path = root / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
        gold_path = root / "data/phase4/temporal-development-v1/gold/cases.jsonl"
        runtime = load_temporal_runtime(runtime_path)
        gold = load_temporal_gold(gold_path)
        self.assertEqual(tuple(item.case_id for item in runtime), tuple(item.case_id for item in gold))
        self.assertEqual(len(runtime), 12)
        self.assertEqual([sum(item.user_id == user for item in runtime) for user in ("user_001", "user_002")], [6, 6])
        tags = {tag for item in runtime for tag in item.tags}
        self.assertTrue(
            {
                "correction", "repeated_evidence", "normal_change", "approximate",
                "utc_equivalence", "out_of_order_ingestion", "separate_valid_periods",
                "inclusive_valid", "transaction_boundary", "half_open",
            } <= tags
        )

        catalog = {
            (item.user_id, item.source_id): item
            for item in load_scaled_development_sources(root)
        }
        for case in runtime:
            sources = {item["source_id"]: item for item in case.sources}
            for source in sources.values():
                adapted = catalog[(case.user_id, source["source_ref"])]
                self.assertIn(case.user_id, {entity.entity_id for entity in adapted.known_entities})
            for span in case.spans:
                adapted = catalog[(case.user_id, sources[span["source_id"]]["source_ref"])]
                observation = next(
                    item
                    for item in adapted.observations
                    if item.message_id == span["message_id"]
                )
                self.assertEqual(observation.author_id, span["speaker_id"])
                self.assertEqual(observation.text, span["quote"])

    def test_runtime_and_gold_are_separate_and_runtime_load_does_not_open_gold(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runtime_path = root / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
        gold_path = root / "data/phase4/temporal-development-v1/gold/cases.jsonl"
        runtime_text = runtime_path.read_text(encoding="utf-8")
        gold_text = gold_path.read_text(encoding="utf-8")
        self.assertNotIn("expected_", runtime_text)
        self.assertNotIn("review_status", runtime_text)
        self.assertNotIn('"commands"', gold_text)
        self.assertNotIn('"sources"', gold_text)

        original_open = Path.open
        opened: list[Path] = []

        def guarded_open(path: Path, *args: object, **kwargs: object):
            resolved = path.resolve()
            opened.append(resolved)
            if resolved == gold_path.resolve():
                raise AssertionError("runtime loader reached gold")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            self.assertEqual(len(load_temporal_runtime(runtime_path)), 12)
        self.assertNotIn(gold_path.resolve(), opened)

    def test_runner_persists_every_case_before_hashing_or_loading_gold(self) -> None:
        root = Path(__file__).resolve().parents[2]
        prediction_path = (
            root / "results/phase4/step4.4-temporal-evaluation-v1/predictions.jsonl"
        )
        runtime = load_temporal_runtime(
            root / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
        )
        predictions = []
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            predictions.append(
                TemporalPrediction(
                    item["case_id"], item["user_id"],
                    tuple(item["ordered_source_ids"]),
                    tuple(item["normalized_claims"]), tuple(item["intervals"]),
                    tuple(item["current_claim_ids"]),
                    tuple(item["historical_claim_ids"]),
                    tuple(item["visible_versions"]),
                )
            )
        predictions = tuple(predictions)
        original_sha256 = runner._sha256
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            gold_path = root / "data/phase4/temporal-development-v1/gold/cases.jsonl"

            def assert_persisted_before_gold(path: Path) -> str:
                if path.resolve() == (
                    root / "data/phase4/temporal-development-v1/gold/cases.jsonl"
                ).resolve():
                    prediction_records = output.joinpath("predictions.jsonl").read_text().splitlines()
                    failure_records = output.joinpath("failures.jsonl").read_text().splitlines()
                    self.assertEqual(len(prediction_records) + len(failure_records), 12)
                return original_sha256(path)

            def load_after_persist(path: Path):
                self.assertTrue(output.joinpath("predictions.jsonl").exists())
                self.assertTrue(output.joinpath("failures.jsonl").exists())
                return load_temporal_gold(path)

            runner._require_case_accounting(runtime, predictions, ())
            runner._require_empty_output(output)
            output.mkdir()
            runner._write_exclusive(
                output / "predictions.jsonl", serialize_jsonl(predictions)
            )
            runner._write_exclusive(output / "failures.jsonl", b"")
            runner._verify_persisted_accounting(
                runtime, output / "predictions.jsonl", output / "failures.jsonl"
            )
            with mock.patch.object(
                runner, "_sha256", side_effect=assert_persisted_before_gold
            ):
                runner._require_hash(gold_path, runner.GOLD_SHA256)
            gold = load_after_persist(gold_path)
            scores = score_temporal(predictions, (), gold)
            runner._write_exclusive(
                output / "scores.json", runner._json_bytes(runner.record(scores))
            )
            self.assertEqual(scores.prediction_count, 12)
            self.assertTrue(output.joinpath("scores.json").exists())

    def test_runner_refuses_nonempty_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            output.joinpath("existing").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(TemporalEvaluationError, "absent or empty"):
                runner._require_empty_output(output)


if __name__ == "__main__":
    unittest.main()
