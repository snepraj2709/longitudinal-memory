from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from abstention import calibration_evaluation as evaluation
from abstention import calibration_release_v2 as release_v2
from abstention.calibration_contracts import CalibrationError, ThresholdApplication
from abstention.calibration_evaluation import (
    CalibrationEvaluationError,
    freeze_threshold_runtime,
    score_threshold_profiles,
    verify_answerability_threshold_release,
    verify_threshold_runtime_checkpoint,
)
from abstention.calibration_release_v2 import (
    COPIED_ARTIFACT_HASHES,
    CalibrationReleaseV2Error,
    finalize_answerability_threshold_release_v2,
    verify_answerability_threshold_release_v2,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "results/abstention/answerability-threshold-development-runtime-v1"
FINAL = ROOT / "results/abstention/answerability-threshold-development-v2"


class AnswerabilityCalibrationIntegrationTests(unittest.TestCase):
    def test_checked_runtime_and_final_releases_self_verify(self) -> None:
        verify_threshold_runtime_checkpoint(repo_root=ROOT)
        verify_answerability_threshold_release_v2(repo_root=ROOT)
        checks = json.loads((FINAL / "checks.json").read_text())
        self.assertEqual(
            checks,
            {
                "accepted_evidence_count": 0,
                "development_abstain_count": 24,
                "development_answerable_count": 0,
                "development_clarify_count": 0,
                "development_decision_count": 24,
                "development_generation_allowed_count": 0,
                "development_null_confidence_count": 24,
                "failure_count": 0,
                "fixture_result_count": 24,
                "input_rejection_count": 491,
                "provider_request_count": 0,
                "selected_profile": "ordinary_1_trait_2",
                "threshold_sweep_count": 6,
            },
        )

    def test_runtime_checkpoint_predates_reference_and_remains_unchanged(self) -> None:
        preflight = json.loads((RUNTIME / "preflight.json").read_text())
        checkpoint = json.loads((RUNTIME / "checkpoint_manifest.json").read_text())
        self.assertFalse(preflight["reference_present"])
        self.assertFalse(preflight["reference_opened"])
        self.assertFalse(preflight["final_result_present"])
        self.assertFalse(checkpoint["reference_opened"])
        self.assertEqual(
            hashlib.sha256((RUNTIME / "checkpoint_manifest.json").read_bytes()).hexdigest(),
            "5a66f85a0c690879771f36f3f6293d190cd672c9b291ee9c7f93f2965f39ce88",
        )
        self.assertEqual(
            hashlib.sha256((RUNTIME / "predictions.jsonl").read_bytes()).hexdigest(),
            "4b7db7c9427ada99430e49b9a42b14f4c9b92de749f83f876897064eb6b9acd0",
        )

    def test_runtime_has_144_stable_predictions_and_zero_failures(self) -> None:
        predictions = [json.loads(line) for line in (RUNTIME / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(predictions), 144)
        self.assertEqual(len({item["prediction_id"] for item in predictions}), 144)
        self.assertEqual((RUNTIME / "failures.jsonl").read_bytes(), b"")
        self.assertEqual(
            [sum(item["output_decision"] == "answerable" for item in predictions if item["profile_id"] == profile) for profile in evaluation.PROFILE_ORDER],
            [8, 7, 6, 5, 4, 3],
        )

    def test_real_applications_preserve_all_abstentions_rejections_and_null_confidence(self) -> None:
        values = [json.loads(line) for line in (FINAL / "development-decisions.jsonl").read_text().splitlines()]
        self.assertEqual(len(values), 24)
        self.assertEqual({item["output_decision"] for item in values}, {"abstain"})
        self.assertEqual({item["primary_reason"] for item in values}, {"no_promoted_claims"})
        self.assertEqual(sum(item["rejected_evidence_count"] for item in values), 491)
        self.assertEqual(sum(item["accepted_evidence_count"] for item in values), 0)
        self.assertTrue(all(item["confidence"] == {
            "value": None,
            "calibration_status": "not_calibrated",
            "null_reason": "no_authorized_answerability_gold",
        } for item in values))

    def test_scorecard_has_honest_real_nulls_and_controlled_fixture_scope(self) -> None:
        value = json.loads((FINAL / "scorecard.json").read_text())
        development = value["development_metrics"]
        controlled = value["controlled_fixture_metrics"]
        self.assertIsNone(value["composite_score"])
        self.assertTrue(all(item["scope"] == "real_development" for item in development))
        self.assertTrue(all(item["scope"] == "controlled_fixture_only" for item in controlled))
        nulls = {item["metric"]: item["null_reason"] for item in development if item["denominator"] == 0}
        self.assertEqual(nulls["selective_risk"], "no_answered_cases")
        self.assertEqual(nulls["confidence_calibration_error"], "no_authorized_answerability_gold")
        self.assertEqual(next(item for item in development if item["metric"] == "coverage" and item["slice_name"] == "overall")["value"], "0.000000")

    def test_runtime_can_reproduce_twice_with_scorer_paths_logically_absent(self) -> None:
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            absent = Path("data/abstention/answerability-threshold-development-v1/reference/not-present.jsonl")
            absent_dataset = Path("data/abstention/answerability-threshold-development-v1/not-present.json")
            absent_final = Path("results/abstention/answerability-threshold-development-not-present")
            with (
                patch.object(evaluation, "REFERENCE_PATH", absent),
                patch.object(evaluation, "DATASET_MANIFEST", absent_dataset),
                patch.object(evaluation, "FINAL_ROOT", absent_final),
            ):
                first = Path(one) / "runtime"
                second = Path(two) / "runtime"
                freeze_threshold_runtime(first, repo_root=ROOT)
                freeze_threshold_runtime(second, repo_root=ROOT)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_two_v2_finalizations_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = Path(one) / "final"
            second = Path(two) / "final"
            finalize_answerability_threshold_release_v2(first, repo_root=ROOT)
            finalize_answerability_threshold_release_v2(second, repo_root=ROOT)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_v2_copies_substantive_payloads_and_discloses_metadata_carry_forward(self) -> None:
        self.assertFalse((ROOT / "results/abstention/answerability-threshold-development-v1").exists())
        self.assertEqual(
            {name: hashlib.sha256((FINAL / name).read_bytes()).hexdigest() for name in COPIED_ARTIFACT_HASHES},
            COPIED_ARTIFACT_HASHES,
        )
        manifest = json.loads((FINAL / "manifest.json").read_text())
        self.assertEqual(manifest["release_version"], "answerability-threshold-development-v2")
        self.assertTrue(manifest["prior_controlled_reference_exposure"])
        self.assertFalse(manifest["benchmark_gold_opened"])
        self.assertFalse(manifest["runtime_changed"])
        self.assertFalse(manifest["threshold_selection_changed"])
        self.assertFalse(manifest["substantive_payloads_changed"])
        self.assertEqual(
            [item["sha256"] for item in manifest["guidance"]["compatibility_rulings"]],
            [
                release_v2.RULING_1_SHA256,
                release_v2.RULING_2_SHA256,
                release_v2.RULING_3_SHA256,
                release_v2.RULING_4_SHA256,
            ],
        )

    def test_public_step91_verifier_runs_before_decision_rows_are_read(self) -> None:
        events = []
        original_verify = evaluation.verify_answerability_release
        original_read = evaluation._read_jsonl

        def checked_verify(*args, **kwargs):
            events.append("verify")
            return original_verify(*args, **kwargs)

        def checked_read(path):
            if Path(path).name == "decisions.jsonl":
                events.append("decisions")
            return original_read(path)

        with patch.object(evaluation, "verify_answerability_release", checked_verify), patch.object(evaluation, "_read_jsonl", checked_read):
            with tempfile.TemporaryDirectory() as directory:
                score_threshold_profiles(Path(directory) / "final", repo_root=ROOT)
        self.assertLess(events.index("verify"), events.index("decisions"))

    def test_runtime_and_scorer_read_traps_block_prohibited_paths(self) -> None:
        source = "\n".join((ROOT / path).read_text() for path in (
            "src/abstention/calibration.py",
            "src/abstention/calibration_evaluation.py",
            "src/abstention/calibration_release_v2.py",
        ))
        for token in ("import psycopg", "from psycopg", "import openai", "from openai", "OPENAI_API_KEY", "os.environ", "requests.", "urllib.", "socket."):
            self.assertNotIn(token, source)
        forbidden = ("benchmark_gold", "answer_gold", "relevance_gold", "oracle", "review_queues", "user_003", ".env")
        opened = []
        original = Path.open

        def guarded(path, *args, **kwargs):
            text = Path(path).as_posix()
            if any(token in text for token in forbidden):
                raise AssertionError(f"prohibited read: {text}")
            opened.append(text)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            verify_threshold_runtime_checkpoint(repo_root=ROOT)
        self.assertFalse(any("reference/expected.jsonl" in path for path in opened))

    def test_tamper_and_nonempty_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            shutil.copytree(RUNTIME, runtime)
            raw = (runtime / "predictions.jsonl").read_bytes()
            (runtime / "predictions.jsonl").write_bytes(raw.replace(b'"profile_id":"ordinary_1_trait_2"', b'"profile_id":"ordinary_1_trait_3"', 1))
            with self.assertRaises((CalibrationEvaluationError, CalibrationError)):
                verify_threshold_runtime_checkpoint(runtime, repo_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            output.mkdir()
            (output / "keep").write_text("keep")
            with self.assertRaisesRegex(CalibrationReleaseV2Error, "must be absent or empty"):
                finalize_answerability_threshold_release_v2(output, repo_root=ROOT)

    def test_v2_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            shutil.copytree(FINAL, output)
            (output / "run.json").write_bytes((output / "run.json").read_bytes().replace(b'"provider_requests":0', b'"provider_requests":1'))
            with self.assertRaises(CalibrationReleaseV2Error):
                verify_answerability_threshold_release_v2(output, repo_root=ROOT)

    def test_real_confidence_value_is_rejected(self) -> None:
        value = json.loads((FINAL / "development-decisions.jsonl").read_text().splitlines()[0])
        value["confidence"] = {"value": "0.500000", "calibration_status": "controlled_fixture_only", "null_reason": None}
        with self.assertRaises(CalibrationError):
            ThresholdApplication(**value)


if __name__ == "__main__":
    unittest.main()
