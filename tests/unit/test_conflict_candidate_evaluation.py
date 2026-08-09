from __future__ import annotations

from dataclasses import asdict, fields
from datetime import date, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import conflicts.evaluation as evaluation_module
from conflicts.candidates import CandidateRequest, generate_candidate_pairs, load_candidate_config
from conflicts.evaluation import (
    CASE_IDS,
    CandidateFailure,
    CandidateGoldCase,
    CandidatePrediction,
    CandidateRuntimeCase,
    CandidateScorecard,
    ConflictEvaluationError,
    PROTECTED_SHA256,
    SCORER_ONLY_PROTECTED_PATH,
    file_sha256,
    load_candidate_dataset_runtime,
    load_candidate_gold,
    load_candidate_runtime,
    persist_outputs_before_gold,
    score_candidate_generation,
    serialize_jsonl,
)
from extraction.predicate_registry import load_predicate_registry
from storage.contracts import ClaimRecord, ClaimVersionRecord
from temporal.contracts import TemporalClaim, VisibleEvidence


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/conflicts/candidate-development-v1"
REFERENCE_RUNTIME = ROOT / "data/phase4/temporal-development-v1/runtime/cases.jsonl"
GOLD_PATH = DATASET_ROOT / "gold/required_pairs.jsonl"
STEP51_RESULT_ROOT = ROOT / "results/conflicts/candidate-generation-development-v1"
STEP51_MANIFEST_SHA256 = "e089dd87b4361982988cd6df37a150e678f3b9245c1a14a007f90202de6b6c18"
STEP51_ARTIFACT_SHA256 = {
    "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "findings.md": "882dc375f582cd3247392f90aaad1419ac365fca326d0b8058acdefe1ea69297",
    "predictions.jsonl": "2d8d0c790c3aa136735b2ac8bb1f2fcca5eeeb9732acca2af5b4dd5dd35870ae",
    "run.json": "a6dbd095cff5f3be62cda6484a77cc5f92b700ca067101d135db305108faa722",
    "scores.json": "57dc7c12e6e3665978862158fd8d592c86481479778d00546c6f742fe05214fa",
}
STEP51_IMPLEMENTATION_SHA256 = {
    "Makefile": "28bb69d856d8c64a04b56969097e58842e8d30257b985e6818a0870d49a54bf3",
    "src/conflicts/__init__.py": "a6da0cd3a4b39ee895da493c12aa64aab6b58e2cc03814db725ba74085ce16f1",
    "src/conflicts/candidates.py": "9d99f71d628fbf12366252825830ffc9ff724b69cd52318ea011e276753b1cec",
    "src/conflicts/evaluation.py": "0202c6f27ff5665b3a3838829c4f3c5c2c374889c191ff589c5f9562f18915d2",
    "tests/integration/test_conflict_candidates.py": "77cbc669b43f39b8061c581c77fb6da320cf8b832dc7876b1731f32e646de55d",
    "tests/unit/test_conflict_candidate_evaluation.py": "a641e2b3bc5a43094b0eabbdf28238aa31376358ba5976f24a3ac8845d30e1b6",
    "tests/unit/test_conflict_candidates.py": "c13787154bde8cf4688bbd36ca51595bdacd584123fa629097660a54af18c103",
}
STEP52_AUTHORIZED_PREDECESSOR_DRIFT = frozenset(
    {
        "Makefile",
        "src/conflicts/__init__.py",
        "src/ingestion/service.py",
        "src/storage/contracts.py",
        "src/storage/repository.py",
        "tests/integration/test_conflict_candidates.py",
        "tests/integration/test_phase4_storage.py",
        "tests/integration/test_temporal_evaluation.py",
        "tests/integration/test_temporal_service.py",
        "tests/unit/test_conflict_candidate_evaluation.py",
        "tests/unit/test_temporal_evaluation.py",
    }
)
STEP52_AUTHORIZED_PROTECTED_DRIFT = frozenset(
    {
        "src/ingestion/service.py",
        "src/storage/contracts.py",
        "src/storage/repository.py",
        "tests/integration/test_phase4_storage.py",
        "tests/integration/test_temporal_evaluation.py",
        "tests/integration/test_temporal_service.py",
        "tests/unit/test_temporal_evaluation.py",
    }
)
STEP53_ALLOWED_PREDECESSOR_DRIFT = STEP52_AUTHORIZED_PREDECESSOR_DRIFT | {
    "src/temporal/contracts.py",
    "src/temporal/service.py",
    "tests/unit/test_temporal_contracts.py",
}
STEP53_ALLOWED_PROTECTED_DRIFT = STEP52_AUTHORIZED_PROTECTED_DRIFT | {
    "src/temporal/contracts.py",
    "src/temporal/service.py",
    "tests/unit/test_temporal_contracts.py",
}


def _temporal_claim(value: object) -> TemporalClaim:
    date_from = value.valid_from if type(value.valid_from) is date else None
    date_to = value.valid_to if type(value.valid_to) is date else None
    timestamp_from = value.valid_from if isinstance(value.valid_from, datetime) else None
    timestamp_to = value.valid_to if isinstance(value.valid_to, datetime) else None
    claim = ClaimRecord(
        claim_id=value.claim_id,
        user_id=value.user_id,
        subject_id=value.subject_id,
        speaker_id=value.subject_id,
        predicate=value.predicate,
        predicate_registry_version="predicate_registry_v2",
        object_json=value.object_json,
        polarity="positive",
        epistemic_status="asserted",
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=value.time_precision,
        extraction_confidence=1,
        memory_kind="durative",
        sensitivity="standard",
        extraction_version_id="candidate_eval_v1",
    )
    version = ClaimVersionRecord(
        version_id=f"version_{value.claim_id}",
        user_id=value.user_id,
        claim_id=value.claim_id,
        lifecycle_status=value.lifecycle_status,
        transaction_from=value.transaction_from,
        transaction_to=value.transaction_to,
        belief_confidence=None,
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=value.time_precision,
    )
    evidence = tuple(
        VisibleEvidence(
            source_id=item.source_id,
            span_id=item.span_id,
            message_id=None,
            speaker_id=value.subject_id,
            quote="reference runtime evidence",
        )
        for item in value.evidence_refs
    )
    return TemporalClaim(claim, version, evidence)


class ConflictCandidateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.runtime = load_candidate_dataset_runtime(DATASET_ROOT)
        cls.config = load_candidate_config(
            ROOT / "configs/conflicts/candidate_linker_v1.json", repo_root=ROOT
        )
        cls.registry = load_predicate_registry(
            ROOT / "configs/extraction/predicate_registry_v2.json"
        )
        cls.predictions = tuple(
            CandidatePrediction(
                case.case_id,
                case.user_id,
                generate_candidate_pairs(
                    cls.config,
                    CandidateRequest(
                        case.user_id, case.transaction_as_of, case.incoming_claim_ids
                    ),
                    tuple(
                        _temporal_claim(claim)
                        for claim in case.claims
                        if claim.user_id == case.user_id
                    ),
                    cls.registry,
                ),
            )
            for case in cls.runtime
        )

    def _persist_and_load_gold(
        self,
        output: Path,
        predictions: tuple[CandidatePrediction, ...] | None = None,
        failures: tuple[CandidateFailure, ...] = (),
    ) -> tuple[CandidateScorecard, tuple[CandidateGoldCase, ...]]:
        loaded: list[tuple[CandidateGoldCase, ...]] = []

        def loader(path: str | Path) -> tuple[CandidateGoldCase, ...]:
            prediction_path = output / "predictions.jsonl"
            failure_path = output / "failures.jsonl"
            self.assertTrue(prediction_path.is_file())
            self.assertTrue(failure_path.is_file())
            persisted = {
                item["case_id"]
                for artifact in (prediction_path, failure_path)
                for item in (
                    json.loads(line)
                    for line in artifact.read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(persisted, set(CASE_IDS))
            self.assertEqual(
                file_sha256(path), self.manifest["gold"]["sha256"]
            )
            value = load_candidate_gold(path)
            self.assertNotIn('"relation"', Path(path).read_text(encoding="utf-8"))
            loaded.append(value)
            return value

        score = persist_outputs_before_gold(
            output,
            self.runtime,
            self.predictions if predictions is None else predictions,
            failures,
            GOLD_PATH,
            expected_gold_sha256=self.manifest["gold"]["sha256"],
            gold_loader=loader,
        )
        return score, loaded[0]

    def test_dataset_is_exact_reviewed_eight_case_runtime_release(self) -> None:
        self.assertEqual(tuple(case.case_id for case in self.runtime), CASE_IDS)
        self.assertEqual(
            [case.user_id for case in self.runtime].count("user_001"), 4
        )
        self.assertEqual(
            [case.user_id for case in self.runtime].count("user_002"), 4
        )
        self.assertTrue(all(len(case.claims) >= 3 for case in self.runtime))
        tags = {tag for case in self.runtime for tag in case.tags}
        self.assertTrue(
            {
                "same_subject_family", "repeated_evidence", "separate_periods",
                "ninety_day_boundary", "shared_entity", "lexical_threshold",
                "approximate", "unknown_time", "cross_user_distractor",
                "inclusive_overlap",
            } <= tags
        )
        repeat = self.runtime[0].claims[0]
        self.assertEqual(len(repeat.evidence_refs), 2)
        cross = self.runtime[-1]
        self.assertEqual({claim.user_id for claim in cross.claims}, {"user_001", "user_002"})
        runtime_text = (DATASET_ROOT / "runtime/cases.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("required_pairs", runtime_text)
        self.assertNotIn("review_status", runtime_text)
        self.assertNotIn("user_003", runtime_text)

    def test_runtime_loading_never_reads_or_hashes_gold(self) -> None:
        opened: list[Path] = []
        hashed: list[Path] = []
        original_read = evaluation_module._read_jsonl
        original_hash = evaluation_module.file_sha256

        def tracked_read(path: str | Path) -> object:
            opened.append(Path(path).resolve())
            return original_read(path)

        def tracked_hash(path: str | Path) -> str:
            hashed.append(Path(path).resolve())
            return original_hash(path)

        with mock.patch.object(evaluation_module, "_read_jsonl", side_effect=tracked_read), mock.patch.object(
            evaluation_module, "file_sha256", side_effect=tracked_hash
        ):
            manifest, cases = load_candidate_dataset_runtime(DATASET_ROOT)
        self.assertEqual(len(cases), 8)
        self.assertEqual(manifest["gold"]["path"], "gold/required_pairs.jsonl")
        self.assertNotIn(GOLD_PATH.resolve(), opened)
        self.assertNotIn(GOLD_PATH.resolve(), hashed)

    def test_all_protected_inputs_are_bound_and_prior_gold_is_deferred(self) -> None:
        self.assertEqual(len(PROTECTED_SHA256), 42)
        predecessor_path = STEP51_RESULT_ROOT / "manifest.json"
        self.assertEqual(file_sha256(predecessor_path), STEP51_MANIFEST_SHA256)
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        self.assertEqual(predecessor["protected_inputs"], PROTECTED_SHA256)
        self.assertEqual(
            predecessor["implementation_hashes"], STEP51_IMPLEMENTATION_SHA256
        )
        self.assertEqual(predecessor["artifacts"], STEP51_ARTIFACT_SHA256)
        for name, expected in STEP51_ARTIFACT_SHA256.items():
            self.assertEqual(file_sha256(STEP51_RESULT_ROOT / name), expected)
        predecessor_files = {
            **PROTECTED_SHA256,
            **STEP51_IMPLEMENTATION_SHA256,
        }
        actual_drift = {
            path
            for path, expected in predecessor_files.items()
            if file_sha256(ROOT / path) != expected
        }
        self.assertLessEqual(STEP52_AUTHORIZED_PREDECESSOR_DRIFT, actual_drift)
        self.assertLessEqual(actual_drift, STEP53_ALLOWED_PREDECESSOR_DRIFT)
        protected_drift = {
            path
            for path, expected in PROTECTED_SHA256.items()
            if file_sha256(ROOT / path) != expected
        }
        self.assertLessEqual(STEP52_AUTHORIZED_PROTECTED_DRIFT, protected_drift)
        self.assertLessEqual(protected_drift, STEP53_ALLOWED_PROTECTED_DRIFT)
        self.assertEqual(
            file_sha256(ROOT / SCORER_ONLY_PROTECTED_PATH),
            PROTECTED_SHA256[SCORER_ONLY_PROTECTED_PATH],
        )

    def test_reference_claim_and_evidence_ownership_guards(self) -> None:
        source = (DATASET_ROOT / "runtime/cases.jsonl").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            bad_claim = Path(directory) / "bad_claim.jsonl"
            bad_claim.write_text(
                source.replace('"reference_claim_id":"t44_repeat_claim"', '"reference_claim_id":"missing_claim"', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConflictEvaluationError, "claim reference"):
                load_candidate_runtime(bad_claim, reference_runtime_path=REFERENCE_RUNTIME)
            bad_evidence = Path(directory) / "bad_evidence.jsonl"
            bad_evidence.write_text(
                source.replace('"span_id":"t44_repeat_span_a"', '"span_id":"missing_span"', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConflictEvaluationError, "evidence reference"):
                load_candidate_runtime(bad_evidence, reference_runtime_path=REFERENCE_RUNTIME)

    def test_all_required_pairs_are_generated_with_expected_signal_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            score, gold = self._persist_and_load_gold(Path(directory) / "results")
        predicted = {
            item.case_id: {(pair.left_claim_id, pair.right_claim_id) for pair in item.pairs}
            for item in self.predictions
        }
        for case in gold:
            self.assertTrue(set(case.required_pairs) <= predicted[case.case_id])
        self.assertEqual(score.candidate_recall["value"], 1.0)
        self.assertEqual(score.required_pair_count, 8)
        self.assertEqual(score.generated_pair_count, 8)
        self.assertEqual(score.all_possible_same_user_pair_count, 10)
        self.assertEqual(score.pair_reduction["value"], 0.2)
        self.assertEqual(score.cross_user_candidate_count, 0)
        self.assertGreater(score.candidate_counts_by_signal["same_subject"], 0)
        self.assertGreater(score.candidate_counts_by_signal["same_predicate_family"], 0)
        self.assertGreater(score.candidate_counts_by_signal["shared_entity"], 0)
        self.assertGreater(score.candidate_counts_by_signal["temporal_overlap"], 0)
        self.assertGreater(score.candidate_counts_by_signal["temporal_within_gap"], 0)
        self.assertGreater(score.candidate_counts_by_signal["approximate_time"], 0)
        self.assertGreater(score.candidate_counts_by_signal["lexical_at_or_above_threshold"], 0)

    def test_failure_placeholder_misses_pair_but_stays_in_denominators(self) -> None:
        failures = (
            CandidateFailure(
                "failure_c51_u2_cross", "c51_u2_cross_user_distractor",
                "user_002", "execution_failed", "case",
            ),
        )
        predictions = self.predictions[:-1]
        with tempfile.TemporaryDirectory() as directory:
            score, _ = self._persist_and_load_gold(
                Path(directory) / "results", predictions, failures
            )
        self.assertEqual(score.prediction_count, 7)
        self.assertEqual(score.failure_count, 1)
        self.assertEqual(score.candidate_recall["numerator"], 7)
        self.assertEqual(score.candidate_recall["denominator"], 8)
        self.assertEqual(score.required_pair_count, 8)
        self.assertEqual(score.all_possible_same_user_pair_count, 10)

    def test_zero_denominators_are_null_with_reasons(self) -> None:
        score = score_candidate_generation((), (), (), ())
        self.assertEqual(
            score.candidate_recall,
            {"value": None, "numerator": 0, "denominator": 0, "null_reason": "no_required_pairs"},
        )
        self.assertEqual(
            score.pair_reduction,
            {"value": None, "numerator": 0, "denominator": 0, "null_reason": "no_possible_same_user_pairs"},
        )

    def test_serialization_and_two_persisted_runs_are_byte_stable(self) -> None:
        self.assertEqual(
            serialize_jsonl(self.predictions),
            serialize_jsonl(tuple(reversed(self.predictions))),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_score, _ = self._persist_and_load_gold(first)
            second_score, _ = self._persist_and_load_gold(second)
            self.assertEqual(asdict(first_score), asdict(second_score))
            for name in ("predictions.jsonl", "failures.jsonl", "scores.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_nonempty_output_is_rejected_before_gold_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            (output / "marker").write_text("occupied", encoding="utf-8")
            loader = mock.Mock()
            with self.assertRaisesRegex(ConflictEvaluationError, "must be empty"):
                persist_outputs_before_gold(
                    output,
                    self.runtime,
                    self.predictions,
                    (),
                    GOLD_PATH,
                    gold_loader=loader,
                )
            loader.assert_not_called()

    def test_record_fields_and_scorecard_exclude_precision(self) -> None:
        self.assertEqual(
            {field.name for field in fields(CandidateGoldCase)},
            {"case_id", "dataset_version", "split", "user_id", "review_status", "required_pairs"},
        )
        score_fields = {field.name for field in fields(CandidateScorecard)}
        self.assertNotIn("precision", score_fields)
        self.assertEqual(
            score_fields,
            {
                "dataset_version", "case_count", "prediction_count", "failure_count",
                "candidate_recall", "required_pair_count", "generated_pair_count",
                "all_possible_same_user_pair_count", "pair_reduction",
                "candidate_counts_by_signal", "cross_user_candidate_count",
            },
        )


if __name__ == "__main__":
    unittest.main()
