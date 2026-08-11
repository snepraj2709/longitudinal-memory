from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from extraction.predicate_registry import load_predicate_registry
from summaries.durative_evaluation import (
    AUTHORIZED_PREDECESSOR_DRIFT,
    DATASET_MANIFEST,
    DATASET_VERSION,
    DurativeDecisionPrediction,
    DurativeEpisodePrediction,
    DurativeEvaluationError,
    DurativeEvaluationFailure,
    _durative_predicate,
    _expected_proposition_counts,
    _span_id,
    load_durative_runtime,
    score_durative_claims,
    serialize_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
FRESH_ONLY_PREDECESSOR_DRIFT = {
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "src/summaries/grounded_evaluation.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DurativeEvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.sources, cls.definitions, cls.claims = load_durative_runtime(
            repo_root=ROOT
        )

    def test_runtime_is_the_exact_bound_development_slice(self) -> None:
        self.assertEqual((len(self.sources), len(self.definitions), len(self.claims)), (20, 20, 33))
        self.assertEqual({item.user_id for item in self.sources}, {"user_001", "user_002"})
        registry = load_predicate_registry(ROOT / "configs/extraction/predicate_registry_v2.json")
        self.assertEqual(_expected_proposition_counts(self.claims, registry), {"user_001": 13, "user_002": 13})
        self.assertEqual(
            sum(not _durative_predicate(registry, item["predicate"]) for item in self.claims),
            7,
        )

    def test_manifest_has_no_gold_or_test_boundary(self) -> None:
        serialized = json.dumps(self.manifest["inputs"], sort_keys=True)
        for forbidden in ("gold", "oracle", "review_queue", "user_003", "test_user"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(self.manifest["expected"]["accepted_claim_count"], 0)
        self.assertEqual(self.manifest["evaluation_contract"]["model_calls"], 0)

    def test_bound_input_hash_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            changed = json.loads(json.dumps(self.manifest))
            changed["inputs"]["rules"]["sha256"] = "0" * 64
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(DurativeEvaluationError, "bound input hash"):
                load_durative_runtime(path, repo_root=ROOT)

    def test_runtime_module_does_not_import_scorer_only_inputs(self) -> None:
        source = (ROOT / "src/summaries/durative_evaluation.py").read_text(encoding="utf-8")
        imports = "\n".join(
            line for line in source.splitlines() if line.startswith(("import ", "from "))
        )
        for forbidden in ("gold", "oracle", "review"):
            self.assertNotIn(forbidden, imports)

    def test_release_records_exact_step6_2_compatibility_drift(self) -> None:
        release = json.loads(
            (ROOT / "results/summaries/durative-claim-development-v1/manifest.json").read_text()
        )
        self.assertEqual(
            {item["path"] for item in release["predecessor"]["authorized_drift"]},
            set(AUTHORIZED_PREDECESSOR_DRIFT) - FRESH_ONLY_PREDECESSOR_DRIFT,
        )
        self.assertEqual(release["predecessor"]["protected_file_count"], 79)
        self.assertEqual(len(release["predecessor"]["effective_protected_file_hashes"]), 79)
        self.assertEqual(release["predecessor"]["implementation_file_count"], 13)
        self.assertEqual(release["predecessor"]["effective_file_count"], 91)


class DurativeEvaluationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.sources, cls.definitions, cls.claims = load_durative_runtime(repo_root=ROOT)
        cls.registry = load_predicate_registry(ROOT / "configs/extraction/predicate_registry_v2.json")
        cls.predictions = cls._predictions()

    @classmethod
    def _predictions(cls) -> tuple[DurativeDecisionPrediction, ...]:
        by_source = {
            source_id: definition
            for definition in cls.definitions
            for source_id in definition.source_ids
        }
        eligible = [
            item for item in cls.claims
            if _durative_predicate(cls.registry, item["predicate"])
        ]
        results = []
        for index, claim in enumerate(eligible):
            evidence = claim["evidence"][0]
            definition = by_source[evidence["source_id"]]
            episode = DurativeEpisodePrediction(
                digest(f"episode:{claim['claim_id']}"),
                "counter_evidence",
                claim["claim_id"],
                f"version_{claim['claim_id']}",
                definition.definition_id,
                evidence["source_id"],
                _span_id(claim),
                "supports",
            )
            results.append(
                DurativeDecisionPrediction(
                    digest(f"plan:{index}"),
                    digest(f"decision:{index}"),
                    claim["user_id"],
                    "durative_claim_rules_v1",
                    digest(f"snapshot:{index}"),
                    "rejected",
                    "memory_kind_ineligible",
                    None,
                    None,
                    definition.transaction_as_of,
                    (episode,),
                )
            )
        return tuple(results)

    def test_complete_rejections_have_visible_denominators_and_no_model_cost(self) -> None:
        score = score_durative_claims(
            self.definitions,
            self.claims,
            self.predictions,
            (),
            (),
            idempotency_replay_pass=True,
            deletion_recompute_pass=True,
        )
        self.assertEqual(score.dataset_version, DATASET_VERSION)
        self.assertEqual(
            (score.input_claim_count, score.durative_proposition_count, score.decision_count),
            (33, 26, 26),
        )
        self.assertEqual((score.accepted_count, score.rejected_count, score.failure_count), (0, 26, 0))
        self.assertEqual(score.input_claim_accounting["value"], 1.0)
        self.assertEqual(score.provenance_integrity["value"], 1.0)
        self.assertEqual((score.model_calls, score.retry_count, score.cost_usd), (0, 0, 0))

    def test_failures_remain_in_every_proposition_denominator(self) -> None:
        failures = tuple(
            DurativeEvaluationFailure(
                digest(f"failure:{user_id}"), user_id, 13, "evaluation_failed", "coordinator"
            )
            for user_id in ("user_001", "user_002")
        )
        score = score_durative_claims(
            self.definitions,
            self.claims,
            (),
            (),
            failures,
            idempotency_replay_pass=False,
            deletion_recompute_pass=False,
        )
        self.assertEqual((score.failure_count, score.failed_proposition_count), (2, 26))
        self.assertEqual(score.decision_accounting["value"], 1.0)
        self.assertEqual(score.provenance_integrity["status"], "not_evaluated")
        self.assertEqual(score.provenance_integrity["reason"], "no_decision_provenance")

    def test_cross_user_and_stale_provenance_are_not_hidden(self) -> None:
        first = self.predictions[0]
        cross_user = replace(first, user_id="user_002")
        stale_episode = replace(self.predictions[1].episodes[0], span_id="stale_span")
        stale = replace(self.predictions[1], episodes=(stale_episode,))
        score = score_durative_claims(
            self.definitions,
            self.claims,
            (cross_user, stale, *self.predictions[2:]),
            (),
            (),
            idempotency_replay_pass=True,
            deletion_recompute_pass=True,
        )
        self.assertEqual(score.cross_user_count, 1)
        self.assertEqual(score.stale_reference_count, 1)
        self.assertLess(score.provenance_integrity["value"], 1.0)

    def test_records_reject_model_execution_and_unsanitized_failure(self) -> None:
        with self.assertRaisesRegex(DurativeEvaluationError, "model execution"):
            replace(self.predictions[0], model="forbidden")
        with self.assertRaisesRegex(DurativeEvaluationError, "sanitized"):
            DurativeEvaluationFailure(digest("failure"), "user_001", 1, "raw value", "worker")

    def test_serialization_is_stable_and_order_preserving(self) -> None:
        first = serialize_jsonl(self.predictions)
        second = serialize_jsonl(tuple(self.predictions))
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())


if __name__ == "__main__":
    unittest.main()
