from __future__ import annotations

from dataclasses import fields, FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from conflicts.candidates import CandidatePair
from conflicts.classifier import CLASSIFIER_VERSION
from conflicts.evaluation import CandidatePrediction
from conflicts.phase_evaluation import (
    CASE_IDS,
    DATASET_VERSION,
    Phase5Action,
    Phase5EvaluationError,
    Phase5Failure,
    Phase5GoldCase,
    Phase5Prediction,
    Phase5Relation,
    canonical_json_bytes,
    load_phase5_gold,
    load_phase5_runtime,
    persist_outputs_before_gold,
    run_phase5_cases,
    score_phase5_conflicts,
    serialize_jsonl,
)
from conflicts.relation_evaluation import RelationPrediction, RelationPredictionRelation
from conflicts.resolution_evaluation import ResolutionPrediction
from conflicts.resolver import POLICY_VERSION, RESOLVER_VERSION


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/conflicts/phase5-evaluation-development-v1/manifest.json"


class _Pipeline:
    def __init__(self, *, extra: bool = False, snapshot: str | None = None, fail: str | None = None) -> None:
        self.extra = extra
        self.snapshot = snapshot
        self.fail = fail
        self.calls: list[str] = []

    def generate_candidates(self, case):
        self.calls.append("candidate")
        if self.fail == "candidate":
            raise RuntimeError("raw secret must not escape")
        pairs = (case.pair,)
        if self.extra:
            other = _CASES[1].pair if case.case_id == _CASES[0].case_id else _CASES[0].pair
            pairs = tuple(sorted((case.pair, other), key=lambda item: (item.left_claim_id, item.right_claim_id)))
        return CandidatePrediction(case.case_id, case.user_id, pairs)

    def classify_relation(self, case, pair):
        self.calls.append("relation")
        if self.fail == "relation":
            raise RuntimeError("raw relation detail")
        relations = ()
        if case.relation_label == "temporal_change":
            relations = (
                RelationPredictionRelation(
                    "relation_" + "1" * 64,
                    pair.left_claim_id,
                    pair.right_claim_id,
                    "same_topic_as",
                    1.0,
                ),
            )
        return RelationPrediction(
            case.case_id,
            case.user_id,
            pair.pair_id,
            CLASSIFIER_VERSION,
            "conflict_relation_rules_v1",
            case.decision_id,
            self.snapshot or case.decision_snapshot_sha256,
            case.relation_label,
            relations,
        )

    def resolve_belief(self, case, decision):
        self.calls.append("resolution")
        if self.fail == "resolution":
            raise RuntimeError("raw resolver detail")
        pair = case.pair
        temporal = case.relation_label == "temporal_change"
        unresolved = case.relation_label == "unresolved_ambiguity"
        return ResolutionPrediction(
            case.case_id,
            case.user_id,
            pair.pair_id,
            decision.decision_id,
            "resolution_" + hashlib.sha256(case.case_id.encode()).hexdigest(),
            RESOLVER_VERSION,
            POLICY_VERSION,
            hashlib.sha256((case.case_id + ":resolution").encode()).hexdigest(),
            "temporal_change_resolved" if temporal else ("disputed" if unresolved else "no_change"),
            pair.left_claim_id if temporal else None,
            (),
            (pair.left_claim_id,) if temporal else (),
            (pair.right_claim_id,) if temporal else (),
            tuple(sorted((pair.left_claim_id, pair.right_claim_id))) if unresolved else (),
            (),
            ("evidence_" + hashlib.sha256(case.case_id.encode()).hexdigest(),),
            pair.source_ids,
            tuple(item.relation_id for item in decision.relations),
            None,
        )


def _gold(cases):
    result = []
    for case in cases:
        temporal = case.case_id == "c51_u1_separate_periods"
        unresolved = case.case_id == "c51_u2_mixed_unknown"
        label = "temporal_change" if temporal else ("unresolved_ambiguity" if unresolved else "unrelated")
        relations = (
            (Phase5Relation(case.pair.left_claim_id, case.pair.right_claim_id, "same_topic_as"),)
            if temporal
            else ()
        )
        result.append(
            Phase5GoldCase(
                case.case_id,
                case.user_id,
                (case.pair.left_claim_id, case.pair.right_claim_id),
                label,
                relations,
                "temporal_change_resolved" if temporal else ("disputed" if unresolved else "no_change"),
                case.pair.left_claim_id if temporal else None,
                (),
                (case.pair.right_claim_id,) if temporal else (),
                tuple(sorted((case.pair.left_claim_id, case.pair.right_claim_id))) if unresolved else (),
                (),
                case.pair.source_ids,
            )
        )
    return tuple(result)


def _predictions(cases):
    return run_phase5_cases(cases, _Pipeline())[0]


_, _CASES = load_phase5_runtime(MANIFEST, repo_root=ROOT)


class Phase5ConflictEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest, self.cases = load_phase5_runtime(MANIFEST, repo_root=ROOT)
        self.gold = _gold(self.cases)
        self.predictions = _predictions(self.cases)

    def _copy_inputs(self, directory: Path) -> Path:
        for relative in (
            "data/conflicts/candidate-development-v1/runtime/cases.jsonl",
            "data/conflicts/relation-development-v1/runtime/cases.jsonl",
            "data/conflicts/belief-resolution-development-v1/runtime/cases.jsonl",
        ):
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        target_manifest = directory / "data/conflicts/phase5-evaluation-development-v1/manifest.json"
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MANIFEST, target_manifest)
        return target_manifest

    def _rewrite_runtime(self, directory: Path, stage: str, mutate) -> None:
        relative = self.manifest["runtime_inputs"][stage]["path"]
        path = directory / relative
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
        path.write_text(content)
        manifest_path = directory / "data/conflicts/phase5-evaluation-development-v1/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime_inputs"][stage]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def test_records_keep_runtime_prediction_and_gold_fields_separate(self) -> None:
        runtime_fields = {item.name for item in fields(type(self.cases[0]))}
        prediction_fields = {item.name for item in fields(Phase5Prediction)}
        gold_fields = {item.name for item in fields(Phase5GoldCase)}
        self.assertNotIn("expected_label", runtime_fields | prediction_fields)
        self.assertNotIn("decision_snapshot_sha256", gold_fields)
        with self.assertRaises(FrozenInstanceError):
            self.predictions[0].case_id = "changed"

    def test_runtime_loader_never_opens_or_hashes_gold(self) -> None:
        gold_paths = {str((ROOT / value["path"]).resolve()) for value in self.manifest["scorer_gold"].values()}
        original_text = Path.read_text
        original_bytes = Path.read_bytes

        def guarded_text(path, *args, **kwargs):
            if str(path.resolve()) in gold_paths:
                raise AssertionError("gold text opened during runtime")
            return original_text(path, *args, **kwargs)

        def guarded_bytes(path, *args, **kwargs):
            if str(path.resolve()) in gold_paths:
                raise AssertionError("gold hashed during runtime")
            return original_bytes(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_text), patch.object(Path, "read_bytes", guarded_bytes):
            _, cases = load_phase5_runtime(MANIFEST, repo_root=ROOT)
        self.assertEqual(tuple(item.case_id for item in cases), CASE_IDS)

    def test_gold_refuses_access_until_runtime_resources_close(self) -> None:
        with patch("conflicts.phase_evaluation._require_hash") as require_hash:
            with self.assertRaisesRegex(Phase5EvaluationError, "resources must close"):
                load_phase5_gold(self.manifest, repo_root=ROOT, runtime_resources_closed=False)
        require_hash.assert_not_called()

    def test_manifest_paths_hashes_split_users_order_and_identity_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._copy_inputs(root)
            self._rewrite_runtime(root, "candidate", lambda rows: rows[0].update(split="test"))
            with self.assertRaises(Phase5EvaluationError):
                load_phase5_runtime(manifest, repo_root=root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._copy_inputs(root)
            self._rewrite_runtime(root, "relation", lambda rows: rows[0].update(user_id="user_002"))
            with self.assertRaises(Phase5EvaluationError):
                load_phase5_runtime(manifest, repo_root=root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._copy_inputs(root)
            self._rewrite_runtime(root, "resolution", lambda rows: rows.reverse())
            with self.assertRaises(Phase5EvaluationError):
                load_phase5_runtime(manifest, repo_root=root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._copy_inputs(root)
            path = root / self.manifest["runtime_inputs"]["candidate"]["path"]
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(Phase5EvaluationError, "hash mismatch"):
                load_phase5_runtime(manifest, repo_root=root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._copy_inputs(root)
            raw = json.loads(manifest.read_text())
            raw["runtime_inputs"]["candidate"]["path"] = "data/elsewhere.jsonl"
            manifest.write_text(json.dumps(raw))
            with self.assertRaisesRegex(Phase5EvaluationError, "binding identity"):
                load_phase5_runtime(manifest, repo_root=root)

    def test_cross_stage_pair_snapshot_and_case_identity_are_checked(self) -> None:
        for stage, mutate in (
            ("relation", lambda rows: rows[0].update(candidate_case_id="c51_u1_separate_periods")),
            ("resolution", lambda rows: rows[0].update(pair_id="0" * 64)),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self._copy_inputs(root)
                self._rewrite_runtime(root, stage, mutate)
                with self.assertRaises(Phase5EvaluationError):
                    load_phase5_runtime(manifest, repo_root=root)

    def test_fresh_pipeline_runs_all_stages_and_sanitizes_stage_failure(self) -> None:
        pipeline = _Pipeline()
        predictions, failures = run_phase5_cases(self.cases, pipeline)
        self.assertEqual((len(predictions), len(failures)), (8, 0))
        self.assertEqual(pipeline.calls[:3], ["candidate", "relation", "resolution"])
        self.assertEqual(len(pipeline.calls), 24)
        pipeline = _Pipeline(fail="candidate")
        predictions, failures = run_phase5_cases(self.cases, pipeline)
        self.assertFalse(predictions)
        self.assertEqual(pipeline.calls, ["candidate"] * 8)
        self.assertEqual(len(failures), 8)
        self.assertEqual((failures[0].code, failures[0].location), ("candidate_execution_failed", "candidate"))
        self.assertNotIn("secret", serialize_jsonl(failures).decode())

    def test_extra_candidate_and_snapshot_mismatch_fail_at_their_stage(self) -> None:
        _, failures = run_phase5_cases(self.cases, _Pipeline(extra=True))
        self.assertEqual(len(failures), 8)
        self.assertEqual((failures[0].code, failures[0].location), ("candidate_pair_count_mismatch", "candidate"))
        pipeline = _Pipeline(snapshot="0" * 64)
        _, failures = run_phase5_cases(self.cases, pipeline)
        self.assertEqual(len(failures), 8)
        self.assertEqual((failures[0].code, failures[0].location), ("relation_snapshot_mismatch", "relation"))
        self.assertEqual(pipeline.calls, [item for _ in range(8) for item in ("candidate", "relation")])

    def test_normal_scorecard_has_required_denominators_and_null_reasons(self) -> None:
        score = score_phase5_conflicts(self.cases, self.predictions, (), self.gold)
        for name in (
            "candidate_recall", "conflict_pair_precision", "conflict_pair_recall",
            "conflict_pair_f1", "conflict_type_accuracy", "current_belief_selection_accuracy",
            "historical_preservation_accuracy", "unresolved_dispute_accuracy",
            "evidence_trace_coverage",
        ):
            self.assertEqual(getattr(score, name)["value"], 1.0, name)
        self.assertEqual(score.conflict_pair_recall["denominator"], 2)
        self.assertEqual(score.conflict_pair_f1["numerator"], 4)
        self.assertEqual(score.conflict_pair_f1["denominator"], 4)
        self.assertEqual(score.false_contradiction_rate, {"status": "evaluated", "numerator": 0, "denominator": 8, "value": 0.0, "reason": None})
        self.assertEqual(score.correction_link_accuracy["reason"], "no_reviewed_correction_links")
        self.assertEqual(score.superseded_preservation_accuracy["reason"], "no_reviewed_superseded_claims")
        self.assertEqual((score.failure_count, score.cross_user_count, score.model_prediction_count), (0, 0, 0))

    def test_evidence_coverage_requires_exact_source_lineage(self) -> None:
        prediction = self.predictions[0]
        with_extra_source = replace(
            prediction,
            source_ids=tuple(sorted((*prediction.source_ids, "unexpected_source"))),
        )
        score = score_phase5_conflicts(
            self.cases,
            (with_extra_source,) + self.predictions[1:],
            (),
            self.gold,
        )
        self.assertEqual(score.evidence_trace_coverage["numerator"], 7)
        self.assertEqual(score.evidence_trace_coverage["denominator"], 8)

    def test_false_positive_false_negative_and_upstream_failure_remain_in_denominators(self) -> None:
        unrelated = self.predictions[0]
        false_positive = replace(
            unrelated,
            relation_label="hard_contradiction",
            relations=(Phase5Relation(unrelated.left_claim_id, unrelated.right_claim_id, "contradicts"),),
        )
        changed = (false_positive,) + self.predictions[1:]
        score = score_phase5_conflicts(self.cases, changed, (), self.gold)
        self.assertEqual(score.conflict_pair_precision["denominator"], 3)
        self.assertEqual(score.conflict_pair_f1["numerator"], 4)
        self.assertEqual(score.conflict_pair_f1["denominator"], 5)
        self.assertEqual(score.false_contradiction_rate["numerator"], 1)
        temporal_index = CASE_IDS.index("c51_u1_separate_periods")
        remaining = tuple(item for index, item in enumerate(self.predictions) if index != temporal_index)
        failure = Phase5Failure("failure_" + "1" * 64, self.cases[temporal_index].case_id, "user_001", "candidate_execution_failed", "candidate")
        score = score_phase5_conflicts(self.cases, remaining, (failure,), self.gold)
        self.assertEqual(score.candidate_recall, {"status": "evaluated", "numerator": 7, "denominator": 8, "value": 0.875, "reason": None})
        self.assertEqual(score.conflict_pair_recall["denominator"], 2)
        self.assertEqual(score.conflict_pair_recall["numerator"], 1)
        self.assertEqual(score.current_belief_selection_accuracy["numerator"], 0)

    def test_synthetic_correction_and_supersession_get_real_denominators(self) -> None:
        expected = replace(
            self.gold[0],
            expected_label="explicit_correction",
            expected_relations=(Phase5Relation(self.gold[0].required_pair[0], self.gold[0].required_pair[1], "corrects"),),
            expected_superseded_claim_ids=(self.gold[0].required_pair[1],),
        )
        prediction = replace(
            self.predictions[0],
            relation_label="explicit_correction",
            relations=expected.expected_relations,
            superseded_claim_ids=expected.expected_superseded_claim_ids,
        )
        score = score_phase5_conflicts(
            self.cases,
            (prediction,) + self.predictions[1:],
            (),
            (expected,) + self.gold[1:],
        )
        self.assertEqual(score.correction_link_accuracy["value"], 1.0)
        self.assertEqual(score.superseded_preservation_accuracy["value"], 1.0)

    def test_persistence_writes_every_outcome_then_loads_gold_and_refuses_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            opened = []

            def loader(manifest, *, repo_root, runtime_resources_closed):
                self.assertTrue(runtime_resources_closed)
                self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 8)
                self.assertEqual((output / "failures.jsonl").read_bytes(), b"")
                opened.append(True)
                return self.gold

            score = persist_outputs_before_gold(
                output,
                self.manifest,
                self.cases,
                self.predictions,
                (),
                repo_root=ROOT,
                runtime_resources_closed=True,
                gold_loader=loader,
            )
            self.assertEqual((opened, score.case_count), ([True], 8))
            with self.assertRaisesRegex(Phase5EvaluationError, "must be empty"):
                persist_outputs_before_gold(
                    output,
                    self.manifest,
                    self.cases,
                    self.predictions,
                    (),
                    repo_root=ROOT,
                    runtime_resources_closed=True,
                    gold_loader=loader,
                )

    def test_serialization_is_utf8_canonical_and_byte_stable(self) -> None:
        first = serialize_jsonl(self.predictions)
        second = serialize_jsonl(tuple(self.predictions))
        self.assertEqual(first, second)
        self.assertEqual(first.count(b"\n"), 8)
        self.assertEqual(canonical_json_bytes({"z": "caf\u00e9", "a": 1}), b'{"a":1,"z":"caf\xc3\xa9"}\n')


if __name__ == "__main__":
    unittest.main()
