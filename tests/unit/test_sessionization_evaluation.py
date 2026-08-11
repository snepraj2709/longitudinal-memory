from __future__ import annotations

from dataclasses import asdict, replace
from datetime import timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summaries.sessionization_evaluation import (
    DATASET_MANIFEST,
    DEVELOPMENT_USERS,
    EXPECTED_SOURCE_IDS,
    EXPECTED_TYPE_COUNTS,
    SessionizationEvaluationError,
    SessionizationFailure,
    SessionizationPrediction,
    _read_prefix,
    _validate_manifest,
    _validate_runtime,
    canonical_json_bytes,
    load_sessionization_runtime,
    run_sessionization,
    score_sessionization,
    serialize_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]


def prediction_for(item, cutoff) -> SessionizationPrediction:
    identity = hashlib.sha256(
        f"{item.user_id}:{item.source_id}:definition".encode()
    ).hexdigest()
    membership = hashlib.sha256(
        f"{item.user_id}:{item.source_id}:membership".encode()
    ).hexdigest()
    return SessionizationPrediction(
        identity,
        item.user_id,
        item.source_type,
        "source",
        "session_boundaries_v1",
        (item.source_id,),
        item.produced_at,
        item.produced_at,
        cutoff,
        membership,
    )


class SessionizationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.sources = load_sessionization_runtime(repo_root=ROOT)

    def test_loader_returns_exact_development_prefix(self) -> None:
        self.assertEqual(len(self.sources), 20)
        self.assertEqual(tuple(item.source_id for item in self.sources), EXPECTED_SOURCE_IDS)
        self.assertEqual(
            {item.user_id for item in self.sources}, set(DEVELOPMENT_USERS)
        )
        self.assertEqual(
            dict(sorted(__import__("collections").Counter(item.source_type for item in self.sources).items())),
            EXPECTED_TYPE_COUNTS,
        )

    def test_prefix_reader_does_not_touch_record_after_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prefix.jsonl"
            with path.open("wb") as handle:
                for index in range(20):
                    handle.write(json.dumps({"index": index}).encode() + b"\n")
                handle.write(b"this is a frozen test record and not JSON\n")
            rows, payload = _read_prefix(path, 20)
        self.assertEqual(len(rows), 20)
        self.assertNotIn(b"frozen test", payload)

    def test_manifest_has_runtime_only_bindings_and_frozen_identity(self) -> None:
        serialized = json.dumps(self.manifest, sort_keys=True)
        self.assertNotIn("gold", serialized)
        self.assertNotIn("oracle", serialized)
        self.assertNotIn("review_queue", serialized)
        self.assertEqual(self.manifest["source_count"], 20)
        changed = dict(self.manifest)
        changed["source_count"] = 19
        with self.assertRaisesRegex(SessionizationEvaluationError, "identity"):
            _validate_manifest(changed)
        changed = json.loads(json.dumps(self.manifest))
        changed["inputs"]["sources"]["development_prefix_sha256"] = "0" * 64
        with self.assertRaisesRegex(SessionizationEvaluationError, "hashes"):
            _validate_manifest(changed)

    def test_runtime_rejects_empty_duplicate_reordered_and_cross_user(self) -> None:
        invalid = (
            (),
            (*self.sources[:-1], self.sources[0]),
            (self.sources[1], self.sources[0], *self.sources[2:]),
            (
                replace(
                    self.sources[0],
                    user_id="user_002",
                    participants=("user_002", "nikhil"),
                ),
                *self.sources[1:],
            ),
        )
        for values in invalid:
            with self.subTest(length=len(values)):
                with self.assertRaises(SessionizationEvaluationError):
                    _validate_runtime(values)

    def test_prefix_and_config_hashes_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / DATASET_MANIFEST
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            for binding in self.manifest["inputs"].values():
                target = root / binding["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"changed\n")
            with self.assertRaises(SessionizationEvaluationError):
                load_sessionization_runtime(repo_root=root)


class SessionizationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.sources = load_sessionization_runtime(repo_root=ROOT)
        cls.cutoffs = {
            user_id: max(item.ingested_at for item in cls.sources if item.user_id == user_id)
            for user_id in DEVELOPMENT_USERS
        }
        cls.predictions = tuple(
            prediction_for(item, cls.cutoffs[item.user_id]) for item in cls.sources
        )

    def test_perfect_source_coverage_and_counts(self) -> None:
        score = score_sessionization(self.sources, self.predictions, ())
        self.assertEqual(score.source_coverage["value"], 1.0)
        self.assertEqual(score.session_count, 20)
        self.assertEqual(score.session_counts_by_type, EXPECTED_TYPE_COUNTS)
        self.assertEqual(score.session_counts_by_user, {"user_001": 10, "user_002": 10})
        self.assertEqual(score.invalid_session_count, 0)
        self.assertEqual(score.duplicate_source_count, 0)
        self.assertEqual(score.cross_user_count, 0)
        self.assertEqual(score.model_prediction_count, 0)

    def test_duplicate_cross_user_and_failure_remain_visible(self) -> None:
        cross = replace(
            self.predictions[10],
            user_id="user_001",
            source_ids=(self.sources[10].source_id,),
        )
        predictions = (*self.predictions[:10], cross, *self.predictions[11:])
        score = score_sessionization(self.sources, predictions, ())
        self.assertEqual(score.invalid_session_count, 1)
        self.assertEqual(score.cross_user_count, 1)
        duplicate = replace(
            self.predictions[1], source_ids=self.predictions[0].source_ids
        )
        score = score_sessionization(
            self.sources, (self.predictions[0], duplicate, *self.predictions[2:]), ()
        )
        self.assertEqual(score.duplicate_source_count, 1)
        user_two = tuple(item for item in self.predictions if item.user_id == "user_002")
        failure = SessionizationFailure(
            "failure_" + "a" * 64,
            "user_001",
            "sessionization_failed",
            "service",
        )
        score = score_sessionization(self.sources, user_two, (failure,))
        self.assertEqual(score.failure_count, 1)
        self.assertEqual(score.source_coverage["numerator"], 10)
        self.assertEqual(score.source_coverage["denominator"], 20)

    def test_empty_outcomes_and_overlapping_failure_are_rejected(self) -> None:
        with self.assertRaisesRegex(SessionizationEvaluationError, "requires"):
            score_sessionization(self.sources, (), ())
        failure = SessionizationFailure(
            "failure_" + "a" * 64, "user_001", "failed", "service"
        )
        with self.assertRaisesRegex(SessionizationEvaluationError, "requires"):
            score_sessionization(self.sources, self.predictions, (failure,))

    def test_failure_details_are_sanitized(self) -> None:
        class UnsafeError(Exception):
            code = "bad value with source text"
            location = "raw/content"

        with patch(
            "summaries.sessionization_evaluation.SessionizationService.define_sessions",
            side_effect=UnsafeError("private source text"),
        ):
            predictions, failures = run_sessionization(object(), self.sources)
        self.assertEqual(predictions, ())
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(item.code == "sessionization_failed" for item in failures))
        self.assertTrue(all(item.location == "service" for item in failures))
        self.assertNotIn("private", serialize_jsonl(failures).decode())

    def test_serialization_is_byte_stable_and_utc_normalized(self) -> None:
        first = serialize_jsonl(self.predictions)
        second = serialize_jsonl(tuple(self.predictions))
        self.assertEqual(first, second)
        self.assertIn(b'"execution_mode":"deterministic"', first)
        self.assertIn(b"Z", first)
        self.assertEqual(
            canonical_json_bytes({"b": 1, "a": 2}),
            b'{"a":2,"b":1}\n',
        )

    def test_prediction_rejects_model_and_bad_hash(self) -> None:
        with self.assertRaises(SessionizationEvaluationError):
            replace(self.predictions[0], definition_id="bad")
        with self.assertRaises(SessionizationEvaluationError):
            replace(self.predictions[0], model="gpt")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
