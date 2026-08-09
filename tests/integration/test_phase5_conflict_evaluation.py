from __future__ import annotations

from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from conflicts.evaluation import (
    CandidatePrediction,
    _require_clean_database,
    load_candidate_runtime,
    run_candidate_cases,
)
from conflicts.phase_evaluation import (
    CASE_IDS,
    DATASET_ROOT,
    Phase5EvaluationError,
    Phase5Prediction,
    canonical_json_bytes,
    file_sha256,
    load_phase5_runtime,
    persist_outputs_before_gold,
    run_phase5_cases,
)
from conflicts.relation_evaluation import (
    RelationPrediction,
    RelationRuntimeCase,
    run_relation_cases,
)
from conflicts.resolution_evaluation import (
    ResolutionPrediction,
    ResolutionRuntimeCase,
    run_resolution_cases,
)
from storage.migrations import apply_migrations


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
RESULT_ROOT = ROOT / "results/conflicts/phase5-conflict-evaluation-development-v1"
STEP53_RESULT = ROOT / "results/conflicts/belief-resolution-development-v1"
STEP53_MANIFEST_SHA256 = "df01c8fbf9494e3f2eb0898e6f2c18aae3ee1cc57e8b2e3fe39ab0006e9f887d"
ARTIFACT_NAMES = (
    "predictions.jsonl",
    "failures.jsonl",
    "scores.json",
    "run.json",
    "findings.md",
)


class _FreshStagePipeline:
    def __init__(
        self,
        candidate: dict[str, CandidatePrediction],
        relation: dict[str, RelationPrediction],
        resolution: dict[str, ResolutionPrediction],
    ) -> None:
        self.candidate = candidate
        self.relation = relation
        self.resolution = resolution

    def generate_candidates(self, case):
        try:
            return self.candidate[case.case_id]
        except KeyError as error:
            raise RuntimeError("fresh candidate stage failed") from error

    def classify_relation(self, case, pair):
        del pair
        try:
            return self.relation[case.case_id]
        except KeyError as error:
            raise RuntimeError("fresh relation stage failed") from error

    def resolve_belief(self, case, decision):
        del decision
        try:
            return self.resolution[case.case_id]
        except KeyError as error:
            raise RuntimeError("fresh resolution stage failed") from error


def _fresh_outputs(connection, phase_cases):
    """Run all three production stages without loading predecessor predictions."""

    candidate_manifest = json.loads(
        (ROOT / "data/conflicts/candidate-development-v1/manifest.json").read_text()
    )
    candidate_cases = load_candidate_runtime(
        ROOT / candidate_manifest["runtime"]["path"].replace("runtime/", "data/conflicts/candidate-development-v1/runtime/"),
        reference_runtime_path=ROOT / "data" / candidate_manifest["reference_runtime"]["path"],
    )
    by_candidate_case = {item.case_id: item for item in candidate_cases}
    reference_path = ROOT / "data" / candidate_manifest["reference_runtime"]["path"]
    candidate_predictions, candidate_failures = run_candidate_cases(
        connection,
        candidate_cases,
        repo_root=ROOT,
        reference_runtime_path=reference_path,
    )
    candidate_by_id = {item.case_id: item for item in candidate_predictions}

    relation_cases = []
    for phase in phase_cases:
        candidate = candidate_by_id.get(phase.case_id)
        if candidate is None or len(candidate.pairs) != 1:
            continue
        relation_cases.append(
            RelationRuntimeCase(
                phase.case_id,
                "relation_development_v1",
                "development",
                phase.user_id,
                phase.case_id,
                candidate.pairs[0],
                phase.explicit_target_claim_id,
                by_candidate_case[phase.case_id],
            )
        )
    relation_predictions, relation_failures = run_relation_cases(
        connection,
        tuple(relation_cases),
        repo_root=ROOT,
        reference_runtime_path=reference_path,
    )
    relation_by_id = {item.case_id: item for item in relation_predictions}
    relation_case_by_id = {item.case_id: item for item in relation_cases}

    resolution_cases = []
    for phase in phase_cases:
        relation = relation_by_id.get(phase.case_id)
        relation_case = relation_case_by_id.get(phase.case_id)
        if relation is None or relation_case is None:
            continue
        resolution_cases.append(
            ResolutionRuntimeCase(
                phase.case_id,
                "belief_resolution_development_v1",
                "development",
                phase.user_id,
                phase.case_id,
                relation.pair_id,
                relation.decision_id,
                relation.input_snapshot_sha256,
                relation.label,
                phase.valid_at_strategy,
                phase.resolved_offset_seconds,
                phase.resolution_idempotency_key,
                relation_case,
                relation,
            )
        )
    resolution_predictions, resolution_failures = run_resolution_cases(
        connection,
        tuple(resolution_cases),
        repo_root=ROOT,
        reference_runtime_path=reference_path,
    )
    pipeline = _FreshStagePipeline(
        candidate_by_id,
        relation_by_id,
        {item.case_id: item for item in resolution_predictions},
    )
    predictions, failures = run_phase5_cases(phase_cases, pipeline)
    stage_failure_count = (
        len(candidate_failures) + len(relation_failures) + len(resolution_failures)
    )
    if stage_failure_count and not failures:
        raise Phase5EvaluationError("fresh stage failures were not carried forward")
    return predictions, failures, pipeline


def execute_phase5_evaluation(connection_factory, output: Path):
    if output.exists() and any(output.iterdir()):
        raise Phase5EvaluationError("result directory must be empty")
    manifest, runtime = load_phase5_runtime(ROOT / DATASET_ROOT / "manifest.json", repo_root=ROOT)
    connection = connection_factory()
    try:
        apply_migrations(connection, ROOT / "migrations")
        _require_clean_database(connection)
        predictions, failures, _ = _fresh_outputs(connection, runtime)
        for table in (
            "memory_users", "source_events", "source_spans", "claims",
            "claim_versions", "evidence_links", "conflict_decisions",
            "claim_relations", "conflict_decision_evidence",
            "belief_resolutions", "belief_resolution_actions",
            "belief_resolution_evidence",
        ):
            if connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != 0:
                raise Phase5EvaluationError("case transaction leaked database state")
    finally:
        connection.close()
    scorecard = persist_outputs_before_gold(
        output,
        manifest,
        runtime,
        predictions,
        failures,
        repo_root=ROOT,
        runtime_resources_closed=True,
    )
    run = {
        "artifact_version": "phase5_conflict_evaluation_development_v1",
        "dataset_version": "phase5_evaluation_development_v1",
        "starting_commit": "5dd1eed3fd61a99eb1c234b08651b38fe9e34067",
        "case_count": 8,
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "execution_mode": "deterministic",
        "fresh_production_services": True,
        "predecessor_predictions_loaded": False,
        "runtime_resources_closed_before_gold": True,
        "database_closed_before_gold": True,
        "model_fallback": False,
        "model_calls": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "new_cost_usd": 0,
        "hosted_writes": 0,
    }
    _write_exclusive(output / "run.json", canonical_json_bytes(run))
    _write_exclusive(output / "findings.md", _findings().encode("utf-8"))
    predecessor = _step53_attestation()
    artifact_hashes = {name: file_sha256(output / name) for name in ARTIFACT_NAMES}
    release = {
        "artifact_version": "phase5_conflict_evaluation_development_v1",
        "guidance_version": "step-5.4-guidance-v1",
        "decision_envelope_sha256": "512dcbffc4d6b8dab30ba9c55045bb0b22eb8fea16942f9b9c50ea61896043e3",
        "dataset": {
            "manifest_sha256": file_sha256(ROOT / DATASET_ROOT / "manifest.json"),
            "runtime_inputs": manifest["runtime_inputs"],
            "scorer_gold": manifest["scorer_gold"],
            "case_count": 8,
            "user_counts": {"user_001": 4, "user_002": 4},
        },
        "execution": run,
        "metrics": asdict(scorecard),
        "predecessor": predecessor,
        "implementation_hashes": {
            path: file_sha256(ROOT / path)
            for path in (
                "Makefile",
                "data/conflicts/phase5-evaluation-development-v1/manifest.json",
                "src/conflicts/phase_evaluation.py",
                "tests/integration/test_phase5_conflict_evaluation.py",
                "tests/unit/test_phase5_conflict_evaluation.py",
            )
        },
        "artifacts": artifact_hashes,
        "known_limitations": [
            "The fixed development set contains two conflict-positive pairs across eight cases.",
            "There is no correction or supersession case, so both denominators remain not evaluated.",
            "This deterministic development run does not estimate production accuracy.",
        ],
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(release))
    return release


def _step53_attestation():
    path = STEP53_RESULT / "manifest.json"
    if file_sha256(path) != STEP53_MANIFEST_SHA256:
        raise Phase5EvaluationError("Step 5.3 manifest changed")
    manifest = json.loads(path.read_text())
    for name, expected in manifest["artifacts"].items():
        if file_sha256(STEP53_RESULT / name) != expected:
            raise Phase5EvaluationError("Step 5.3 artifact changed")
    drift = []
    for relative, expected in manifest["implementation_hashes"].items():
        current = file_sha256(ROOT / relative)
        if current == expected:
            continue
        if relative != "Makefile":
            raise Phase5EvaluationError("Step 5.3 implementation changed")
        drift.append(
            {
                "path": relative,
                "predecessor_sha256": expected,
                "step5_4_sha256": current,
                "reason": "adds_step5_4_conflict_evaluation_test_target",
            }
        )
    if len(drift) != 1:
        raise Phase5EvaluationError("Step 5.4 predecessor drift changed")
    return {
        "manifest": {
            "path": "results/conflicts/belief-resolution-development-v1/manifest.json",
            "sha256": STEP53_MANIFEST_SHA256,
        },
        "artifacts": manifest["artifacts"],
        "implementation_hashes": manifest["implementation_hashes"],
        "input_hashes": manifest["input_hashes"],
        "protected_maps": manifest["predecessor"],
        "authorized_drift": drift,
    }


def _write_exclusive(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise Phase5EvaluationError("result artifact already exists") from error


def _findings() -> str:
    return (
        "# Phase 5 conflict evaluation findings\n\n"
        "All eight development cases ran through the candidate linker, relation classifier, and belief resolver with no model calls or failures. "
        "The two conflict-positive pairs were found and typed correctly. Current, historical, disputed, and evidence checks also matched the reviewed expectations.\n\n"
        "The fixed cases contain no correction or supersession example. Those scorecard entries are not evaluated rather than counted as perfect.\n"
    )


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for Phase 5 evaluation tests",
)
class Phase5ConflictEvaluationIntegrationTests(unittest.TestCase):
    def _reset(self) -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    def _factory(self):
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def test_all_eight_cases_run_fresh_with_expected_component_metrics(self) -> None:
        self._reset()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            release = execute_phase5_evaluation(self._factory, output)
            scores = json.loads((output / "scores.json").read_text())
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), 8)
            self.assertEqual((output / "failures.jsonl").read_bytes(), b"")
            for name in (
                "candidate_recall", "conflict_pair_precision", "conflict_pair_recall",
                "conflict_pair_f1", "conflict_type_accuracy",
                "current_belief_selection_accuracy", "historical_preservation_accuracy",
                "unresolved_dispute_accuracy", "evidence_trace_coverage",
            ):
                self.assertEqual(scores[name]["value"], 1.0, name)
            self.assertEqual(scores["conflict_pair_recall"]["denominator"], 2)
            self.assertEqual(scores["false_contradiction_rate"]["denominator"], 8)
            self.assertEqual(scores["correction_link_accuracy"]["reason"], "no_reviewed_correction_links")
            self.assertEqual(scores["superseded_preservation_accuracy"]["reason"], "no_reviewed_superseded_claims")
            self.assertEqual(scores["cross_user_count"], 0)
            self.assertEqual(release["execution"]["model_calls"], 0)
            by_id = {item["case_id"]: item for item in predictions}
            temporal = by_id["c51_u1_separate_periods"]
            disputed = by_id["c51_u2_mixed_unknown"]
            self.assertEqual(len(temporal["current_claim_ids"]), 1)
            self.assertEqual(len(temporal["historical_claim_ids"]), 1)
            self.assertEqual(len(disputed["disputed_claim_ids"]), 2)
            self.assertEqual(
                sum(item["resolution_outcome"] == "excluded" for item in predictions),
                2,
            )
            self.assertTrue(all(not item["superseded_claim_ids"] for item in predictions))
            for name in (*ARTIFACT_NAMES, "manifest.json"):
                self.assertEqual((output / name).read_bytes(), (RESULT_ROOT / name).read_bytes())

    def test_two_clean_runs_are_byte_identical_and_leave_no_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = (Path(directory) / "first", Path(directory) / "second")
            for output in outputs:
                self._reset()
                execute_phase5_evaluation(self._factory, output)
            for name in (*ARTIFACT_NAMES, "manifest.json"):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())

    def test_fresh_extra_candidate_is_a_case_failure_and_later_stages_are_absent(self) -> None:
        self._reset()
        manifest, runtime = load_phase5_runtime(ROOT / DATASET_ROOT / "manifest.json", repo_root=ROOT)
        del manifest
        connection = self._factory()
        try:
            apply_migrations(connection, ROOT / "migrations")
            predictions, failures, pipeline = _fresh_outputs(connection, runtime)
        finally:
            connection.close()
        self.assertEqual((len(predictions), len(failures)), (8, 0))
        first = runtime[0]
        second_pair = pipeline.candidate[runtime[1].case_id].pairs[0]
        original = pipeline.candidate[first.case_id]
        pairs = tuple(sorted((original.pairs[0], second_pair), key=lambda item: (item.left_claim_id, item.right_claim_id)))
        pipeline.candidate[first.case_id] = CandidatePrediction(first.case_id, first.user_id, pairs)
        predictions, failures = run_phase5_cases(runtime, pipeline)
        self.assertEqual((len(predictions), len(failures)), (7, 1))
        self.assertEqual((failures[0].case_id, failures[0].location), (first.case_id, "candidate"))

    def test_clean_database_and_immutable_result_guards(self) -> None:
        self._reset()
        connection = self._factory()
        try:
            apply_migrations(connection, ROOT / "migrations")
            connection.execute(
                "INSERT INTO memory_users (user_id, created_at) VALUES ('occupied', now())"
            )
        finally:
            connection.close()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "clean database"):
                execute_phase5_evaluation(self._factory, Path(directory) / "clean_guard")
            output = Path(directory) / "nonempty"
            output.mkdir()
            (output / "keep").write_text("preserve")
            with self.assertRaisesRegex(Phase5EvaluationError, "must be empty"):
                execute_phase5_evaluation(self._factory, output)
            self.assertEqual((output / "keep").read_text(), "preserve")

    def test_checked_release_is_hash_bound_and_predecessor_predictions_are_not_inputs(self) -> None:
        manifest = json.loads((RESULT_ROOT / "manifest.json").read_text())
        self.assertEqual(file_sha256(STEP53_RESULT / "manifest.json"), STEP53_MANIFEST_SHA256)
        self.assertFalse(manifest["execution"]["predecessor_predictions_loaded"])
        self.assertEqual(set(manifest["predecessor"]["authorized_drift"][0]), {"path", "predecessor_sha256", "step5_4_sha256", "reason"})
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(file_sha256(RESULT_ROOT / name), expected)


if __name__ == "__main__":
    unittest.main()
