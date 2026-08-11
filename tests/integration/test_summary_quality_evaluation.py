from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summaries.summary_quality_evaluation as quality_evaluation
from summaries.summary_quality_evaluation import (
    SummaryQualityEvaluationError,
    canonical_json,
    load_scorer_inputs,
    match_case_events,
    score_summary_quality,
)
from summaries.summary_quality_runtime import TemporalFailure


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "results/summaries/summary-quality-development-runtime-v2"
RESULT = ROOT / "results/summaries/summary-quality-development-v2"
GOLD_CASES = ROOT / "data/summaries/summary-quality-development-v2/gold/cases.jsonl"
GOLD_CLAIMS = ROOT / "data/summaries/summary-quality-development-v2/gold/claims.jsonl"
EVENT_MAP = ROOT / "data/summaries/summary-quality-development-v2/gold/event_map.jsonl"
CONFIG = ROOT / "configs/summaries/summary_quality_scorer_v2.json"
DATASET_MANIFEST = ROOT / "data/summaries/summary-quality-development-v2/manifest.json"
RUNTIME_CASES = ROOT / "data/summaries/summary-quality-development-v2/runtime/cases.jsonl"
RUNTIME_MODULE = ROOT / "src/summaries/summary_quality_runtime.py"
EVALUATOR_MODULE = ROOT / "src/summaries/summary_quality_evaluation.py"
ARTIFACTS = (
    "predictions.jsonl",
    "failures.jsonl",
    "scores.json",
    "run.json",
    "findings.md",
)
FINDINGS = """# Step 6.4 summary-quality findings

The frozen runtime produced a bundle for all 10 cases without a failure. All 170 statement instances kept exact statement-to-Claim-version-to-span provenance.

Strict Claim-and-evidence matching found no event matches. Recall was 0/22 and precision was 0/165. Evidence recall was 0/31 and precision was 0/165. Because no event matched, the current-versus-historical metric is null rather than zero. The correction and uncertainty checks were 0/2 and 0/9.

Two inherited limits explain the low score. The baseline returns every visible session summary instead of selecting for the instruction, so one user's broad bundle repeats across cases. The extracted Claims also differ from the reviewed Claims on at least one exact semantic or evidence field. The scorer does not use lexical or semantic similarity to hide those differences.

There were no cross-user, stale, unsupported, or runtime-failure records. Durative output remained empty. No model ran; historical OpenAI spend remains $0.2314404.

This is not a blind evaluation. The implementing agent had already seen the authorized v1 development gold before the whitespace-only runtime correction. The v2 runtime and predictions were frozen while all scorer and gold files were absent, and the carried gold bytes were not reinterpreted or changed.
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def render_release(output_dir: Path) -> dict[str, str]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SummaryQualityEvaluationError("result directory is not empty")
    predictions, failures, cases, claims, event_map = load_scorer_inputs(
        CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP
    )
    scores = score_summary_quality(predictions, failures, cases, claims, event_map)
    run = {
        "artifact_version": "summary_quality_development_v2",
        "case_count": len(cases),
        "cost_usd": 0,
        "execution_mode": "deterministic_no_model",
        "failure_count": len(failures),
        "gold_claim_count": len(claims),
        "gold_event_count": sum(len(item.events) for item in event_map),
        "historical_openai_spend_usd": 0.2314404,
        "input_tokens": 0,
        "model_calls": 0,
        "output_tokens": 0,
        "prediction_count": len(predictions),
        "request_count": 0,
        "retry_count": 0,
        "scorer_version": "summary_quality_scorer_v2",
        "starting_commit": "d52b4a2a9a1178ab37354fc65fd1d202519185cc",
    }
    payloads = {
        "predictions.jsonl": (CHECKPOINT / "predictions.jsonl").read_bytes(),
        "failures.jsonl": (CHECKPOINT / "failures.jsonl").read_bytes(),
        "scores.json": canonical_json(scores),
        "run.json": json_bytes(run),
        "findings.md": FINDINGS.encode("utf-8"),
    }
    artifact_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    manifest = {
        "artifact_version": "summary_quality_development_v2",
        "artifacts": artifact_hashes,
        "guidance": {
            "sha256": "3b5673d169a1ad58cffda7ff45bd83ffe0a541da9ed38e870be99503842955b9",
            "version": "step-6.4-guidance-v2",
        },
        "inputs": {
            "checkpoint_failures_sha256": sha(CHECKPOINT / "failures.jsonl"),
            "checkpoint_manifest_sha256": sha(CHECKPOINT / "manifest.json"),
            "checkpoint_preflight_sha256": sha(CHECKPOINT / "checkpoint_preflight.json"),
            "checkpoint_tree_manifest_sha256": sha(CHECKPOINT / "checkpoint_manifest.json"),
            "checkpoint_predictions_sha256": sha(CHECKPOINT / "predictions.jsonl"),
            "checkpoint_run_sha256": sha(CHECKPOINT / "run.json"),
            "dataset_manifest_sha256": sha(DATASET_MANIFEST),
            "event_map_sha256": sha(EVENT_MAP),
            "required_claims_sha256": sha(GOLD_CLAIMS),
            "runtime_cases_sha256": sha(RUNTIME_CASES),
            "runtime_module_sha256": sha(RUNTIME_MODULE),
            "scorer_cases_sha256": sha(GOLD_CASES),
            "scorer_config_sha256": sha(CONFIG),
            "scorer_module_sha256": sha(EVALUATOR_MODULE),
        },
        "model": {
            "calls": 0,
            "cost_usd": 0,
            "historical_openai_spend_usd": 0.2314404,
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
            "retries": 0,
        },
        "prior_v1_development_gold_exposure": {
            "blind_evaluation": False,
            "carried_gold_reinterpreted": False,
            "occurred": True,
            "runtime_checkpoint_frozen_with_gold_and_scorer_absent": True,
        },
        "predecessor": {
            "durative_manifest_sha256": "1d3f1c78d95bd96399224581bec21143c4b52562517d4779b74850e42d26fdbb",
            "grounded_summaries_sha256": "6ce2ddbdc3ade54154e4db0bd136ff6f12e1037ebf4b214d560a2247fb674b22",
            "step6_2_manifest_sha256": "ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761",
        },
    }
    manifest_bytes = json_bytes(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {**payloads, "manifest.json": manifest_bytes}.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(payload)
    return {**artifact_hashes, "manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}


class SummaryQualityReleaseIntegrationTests(unittest.TestCase):
    def test_checked_release_is_self_verifying_and_reuses_checkpoint_bytes(self) -> None:
        self.assertEqual(
            {item.name for item in RESULT.iterdir()}, {*ARTIFACTS, "manifest.json"}
        )
        manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["inputs"]["dataset_manifest_sha256"], sha(DATASET_MANIFEST))
        self.assertEqual(manifest["inputs"]["runtime_cases_sha256"], sha(RUNTIME_CASES))
        self.assertTrue(manifest["prior_v1_development_gold_exposure"]["occurred"])
        self.assertFalse(manifest["prior_v1_development_gold_exposure"]["blind_evaluation"])
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(sha(RESULT / name), expected)
        self.assertEqual(
            (RESULT / "predictions.jsonl").read_bytes(),
            (CHECKPOINT / "predictions.jsonl").read_bytes(),
        )
        self.assertEqual(
            (RESULT / "failures.jsonl").read_bytes(),
            (CHECKPOINT / "failures.jsonl").read_bytes(),
        )
        findings = (RESULT / "findings.md").read_text(encoding="utf-8")
        self.assertIn("not a blind evaluation", findings)
        self.assertIn("already seen the authorized v1 development gold", findings)
        predictions, failures, cases, claims, event_map = load_scorer_inputs(
            CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP
        )
        expected_scores = canonical_json(
            score_summary_quality(predictions, failures, cases, claims, event_map)
        )
        self.assertEqual((RESULT / "scores.json").read_bytes(), expected_scores)

    def test_two_clean_scorer_runs_are_byte_identical_and_nonempty_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_hashes = render_release(first)
            second_hashes = render_release(second)
            self.assertEqual(first_hashes, second_hashes)
            for name in (*ARTIFACTS, "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual((first / name).read_bytes(), (RESULT / name).read_bytes())
            with self.assertRaisesRegex(SummaryQualityEvaluationError, "not empty"):
                render_release(first)

    def test_checked_metrics_keep_exact_denominators_and_null_reason(self) -> None:
        scores = json.loads((RESULT / "scores.json").read_text(encoding="utf-8"))
        expected = {
            "gold_event_micro_precision": (0, 165, 0.0, None),
            "gold_event_micro_recall": (0, 22, 0.0, None),
            "supporting_evidence_micro_precision": (0, 165, 0.0, None),
            "supporting_evidence_micro_recall": (0, 31, 0.0, None),
            "current_historical_accuracy": (
                0,
                0,
                None,
                "no_matched_reviewed_state_events",
            ),
            "correction_preservation": (0, 2, 0.0, None),
            "uncertainty_preservation": (0, 9, 0.0, None),
            "case_accounting": (10, 10, 1.0, None),
            "provenance_coverage": (170, 170, 1.0, None),
        }
        for name, value in expected.items():
            metric = scores[name]
            self.assertEqual(
                (
                    metric["numerator"],
                    metric["denominator"],
                    metric["value"],
                    metric["null_reason"],
                ),
                value,
            )
        self.assertNotIn("composite", scores)
        self.assertEqual(scores["cross_user_prediction_count"], 0)
        self.assertEqual(scores["stale_reference_count"], 0)
        self.assertEqual(scores["unsupported_statement_count"], 0)
        self.assertEqual(scores["runtime_failure_count"], 0)

    def test_matching_is_one_to_one_and_failure_remains_in_all_denominators(self) -> None:
        predictions, failures, cases, claims, event_map = load_scorer_inputs(
            CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP
        )
        claim_by_id = {item.claim_id: item for item in claims}
        for prediction, mapped in zip(predictions, event_map, strict=True):
            matches = match_case_events(prediction, mapped, claim_by_id)
            self.assertLessEqual(len(matches), len(mapped.events))
            self.assertEqual(len({item.statement_id for item in matches}), len(matches))
            self.assertEqual(len({item.event_id for item in matches}), len(matches))
        failed_prediction = predictions[0]
        synthetic_failure = TemporalFailure(
            case_id=failed_prediction.case_id,
            user_id=failed_prediction.user_id,
            code="runtime_failure",
            location="case_bundle",
        )
        score = score_summary_quality(
            predictions[1:], (synthetic_failure,), cases, claims, event_map
        )
        self.assertEqual(score.case_accounting.value, 1.0)
        self.assertEqual(score.runtime_failure_count, 1)
        self.assertEqual(score.gold_event_micro_recall.denominator, 22)
        self.assertEqual(score.supporting_evidence_micro_recall.denominator, 31)

    def test_checkpoint_finishes_before_scorer_inputs_and_imports_are_isolated(self) -> None:
        ready = False
        original_loader = quality_evaluation.load_runtime_checkpoint
        original_open = Path.open
        protected = {GOLD_CASES.resolve(), GOLD_CLAIMS.resolve(), EVENT_MAP.resolve()}

        def checked_loader(root):
            nonlocal ready
            result = original_loader(root)
            self.assertEqual(len(result[0]) + len(result[1]), 10)
            ready = True
            return result

        def guarded_open(path, *args, **kwargs):
            if path.resolve() in protected and not ready:
                raise AssertionError("scorer input opened before runtime completion")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(
            quality_evaluation, "load_runtime_checkpoint", side_effect=checked_loader
        ), mock.patch.object(Path, "open", guarded_open):
            load_scorer_inputs(CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP)
        self.assertTrue(ready)
        runtime_source = RUNTIME_MODULE.read_text(encoding="utf-8").casefold()
        for token in ("g" + "old", "or" + "acle", "re" + "view", "sco" + "rer"):
            self.assertNotIn(token, runtime_source)
        evaluator_source = EVALUATOR_MODULE.read_text(encoding="utf-8").casefold()
        for token in ("or" + "acle", "review" + "_queue", "test" + "_user"):
            self.assertNotIn(token, evaluator_source)
        imports = {
            node.module
            for node in ast.walk(ast.parse(evaluator_source))
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertTrue(all("oracle" not in item and "review" not in item for item in imports))


if __name__ == "__main__":
    unittest.main()
