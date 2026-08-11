from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summaries.grounded_evaluation import (
    AUTHORIZED_PREDECESSOR_DRIFT,
    DATASET_MANIFEST,
    DATASET_VERSION,
    EmptySessionPrediction,
    EvidencePath,
    GroundedStatementPrediction,
    GroundedSummaryEvaluationError,
    GroundedSummaryFailure,
    GroundedSummaryPrediction,
    canonical_json_bytes,
    load_grounded_runtime,
    score_grounded_summaries,
    serialize_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_RELEASE_MANIFEST_SHA256 = "ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761"
AUTHORIZED_STEP63_DRIFT = {
    "Makefile",
    "src/summaries/grounded_evaluation.py",
    "tests/integration/test_grounded_summary_persistence.py",
    "tests/integration/test_grounded_summary_evaluation.py",
    "tests/unit/test_grounded_summary_evaluation.py",
}
FRESH_ONLY_PREDECESSOR_DRIFT = {
    "compose.yaml",
    "src/conflicts/resolution_evaluation.py",
    "tests/unit/test_conflict_candidate_evaluation.py",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class GroundedSummaryEvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.sources, cls.definitions, cls.claims = load_grounded_runtime(
            repo_root=ROOT
        )

    def test_runtime_has_exact_bound_development_counts(self) -> None:
        self.assertEqual(len(self.sources), 20)
        self.assertEqual(len(self.definitions), 20)
        self.assertEqual(len(self.claims), 33)
        self.assertEqual({item.user_id for item in self.sources}, {"user_001", "user_002"})
        self.assertEqual(
            {source_id for item in self.definitions for source_id in item.source_ids},
            {item.source_id for item in self.sources},
        )

    def test_manifest_binds_runtime_only_and_phase5_implementation_map(self) -> None:
        serialized = json.dumps(self.manifest, sort_keys=True)
        for forbidden in ("oracle", "review_queue", "test_user"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            set(self.manifest["inputs"]["phase5_release"]["implementation_hashes"]),
            {
                "Makefile",
                "data/conflicts/phase5-evaluation-development-v1/manifest.json",
                "src/conflicts/phase_evaluation.py",
                "tests/integration/test_phase5_conflict_evaluation.py",
                "tests/unit/test_phase5_conflict_evaluation.py",
            },
        )
        release_path = ROOT / "results/summaries/grounded-summary-development-v1/manifest.json"
        self.assertEqual(hashlib.sha256(release_path.read_bytes()).hexdigest(), FROZEN_RELEASE_MANIFEST_SHA256)
        release = json.loads(release_path.read_text())
        for name, expected in release["artifacts"].items():
            self.assertEqual(
                hashlib.sha256((release_path.parent / name).read_bytes()).hexdigest(),
                expected,
            )
        current_drift = {
            path
            for path, expected in release["implementation_hashes"].items()
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected
        }
        self.assertEqual(current_drift, AUTHORIZED_STEP63_DRIFT)
        predecessor = release["predecessor"]
        self.assertEqual(predecessor["protected_file_count"], 79)
        self.assertEqual(predecessor["unchanged_file_count"], 72)
        self.assertEqual(
            {item["path"] for item in predecessor["authorized_drift"]},
            set(AUTHORIZED_PREDECESSOR_DRIFT) - FRESH_ONLY_PREDECESSOR_DRIFT,
        )

    def test_runtime_module_has_no_scorer_data_import(self) -> None:
        source = (ROOT / "src/summaries/grounded_evaluation.py").read_text(encoding="utf-8")
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        joined = "\n".join(import_lines)
        self.assertNotIn("gold", joined)
        self.assertNotIn("oracle", joined)
        self.assertNotIn("review", joined)

    def test_bound_hash_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / DATASET_MANIFEST
            manifest_path.parent.mkdir(parents=True)
            changed = json.loads(json.dumps(self.manifest))
            changed["inputs"]["renderer"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(GroundedSummaryEvaluationError):
                load_grounded_runtime(repo_root=root)


class GroundedSummaryEvaluationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.sources, cls.definitions, cls.claims = load_grounded_runtime(repo_root=ROOT)
        by_source: dict[str, list[dict[str, object]]] = {
            item.source_id: [] for item in cls.sources
        }
        for claim in cls.claims:
            by_source[claim["evidence"][0]["source_id"]].append(claim)
        predictions = []
        empty = []
        for definition in cls.definitions:
            claims = [
                claim
                for source_id in definition.source_ids
                for claim in by_source[source_id]
            ]
            if not claims:
                empty.append(
                    EmptySessionPrediction(
                        definition.user_id,
                        definition.definition_id,
                        definition.membership_sha256,
                        definition.source_ids,
                    )
                )
                continue
            statements = tuple(
                GroundedStatementPrediction(
                    digest(f"statement:{claim['claim_id']}"),
                    "observed_fact",
                    "candidate",
                    f"Candidate claim {claim['claim_id']}",
                    (claim["claim_id"],),
                    (
                        EvidencePath(
                            claim["claim_id"],
                            f"version_{claim['claim_id']}",
                            claim["evidence"][0]["source_id"],
                            "span_" + hashlib.sha256(
                                canonical_json_bytes(
                                    (
                                        claim["user_id"],
                                        claim["evidence"][0]["source_id"],
                                        claim["evidence"][0]["message_id"],
                                        claim["evidence"][0]["quote"],
                                    )
                                )[:-1]
                            ).hexdigest(),
                            "supports",
                        ),
                    ),
                    False,
                )
                for claim in claims
            )
            predictions.append(
                GroundedSummaryPrediction(
                    digest(f"summary:{definition.definition_id}"),
                    definition.user_id,
                    definition.definition_id,
                    definition.membership_sha256,
                    "session_summary_renderer_v1",
                    digest(f"snapshot:{definition.definition_id}"),
                    definition.source_ids,
                    tuple(sorted(claim["claim_id"] for claim in claims)),
                    "Observed facts\n- candidate",
                    statements,
                    False,
                )
            )
        cls.predictions = tuple(predictions)
        cls.empty = tuple(empty)

    def test_perfect_structure_has_all_three_one_scores(self) -> None:
        checks = score_grounded_summaries(
            self.definitions,
            self.claims,
            self.predictions,
            self.empty,
            (),
            deletion_recompute_pass=True,
        )
        self.assertEqual(checks.dataset_version, DATASET_VERSION)
        self.assertEqual((checks.summary_count, checks.empty_session_count), (17, 3))
        self.assertEqual(checks.statement_count, 33)
        self.assertEqual(checks.provenance_integrity["value"], 1.0)
        self.assertEqual(checks.membership_integrity["value"], 1.0)
        self.assertEqual(checks.eligible_claim_accounting["value"], 1.0)
        self.assertEqual(
            (
                checks.cross_user_count,
                checks.stale_reference_count,
                checks.duplicate_statement_count,
                checks.unsupported_statement_count,
            ),
            (0, 0, 0, 0),
        )

    def test_bad_provenance_and_membership_remain_visible(self) -> None:
        first = self.predictions[0]
        bad_path = replace(first.statements[0].evidence[0], claim_version_id="stale")
        bad_statement = replace(first.statements[0], evidence=(bad_path,))
        bad = replace(first, source_ids=("wrong_source",), statements=(bad_statement, *first.statements[1:]))
        checks = score_grounded_summaries(
            self.definitions,
            self.claims,
            (bad, *self.predictions[1:]),
            self.empty,
            (),
            deletion_recompute_pass=False,
        )
        self.assertLess(checks.provenance_integrity["value"], 1.0)
        self.assertLess(checks.membership_integrity["value"], 1.0)
        self.assertEqual(checks.stale_reference_count, 1)
        self.assertFalse(checks.deletion_recompute_pass)

    def test_exact_span_and_statement_claim_lineage_are_enforced(self) -> None:
        first = self.predictions[0]
        bad_path = replace(first.statements[0].evidence[0], span_id="wrong_span")
        bad_statement = replace(first.statements[0], evidence=(bad_path,))
        bad = replace(first, statements=(bad_statement, *first.statements[1:]))
        checks = score_grounded_summaries(
            self.definitions,
            self.claims,
            (bad, *self.predictions[1:]),
            self.empty,
            (),
            deletion_recompute_pass=True,
        )
        self.assertLess(checks.provenance_integrity["value"], 1.0)
        self.assertEqual(checks.stale_reference_count, 1)
        with self.assertRaisesRegex(GroundedSummaryEvaluationError, "match exactly"):
            replace(first.statements[0], claim_ids=("claim_not_in_evidence",))

    def test_failure_counts_in_session_accounting(self) -> None:
        removed = self.predictions[0]
        failure = GroundedSummaryFailure(
            digest("failure"),
            removed.user_id,
            removed.session_definition_id,
            "summary_failed",
            "session",
        )
        checks = score_grounded_summaries(
            self.definitions,
            self.claims,
            self.predictions[1:],
            self.empty,
            (failure,),
            deletion_recompute_pass=True,
        )
        self.assertEqual(checks.failure_count, 1)
        self.assertLess(checks.eligible_claim_accounting["value"], 1.0)

    def test_missing_or_overlapping_session_outcomes_are_rejected(self) -> None:
        with self.assertRaisesRegex(GroundedSummaryEvaluationError, "incomplete"):
            score_grounded_summaries(
                self.definitions,
                self.claims,
                self.predictions[:-1],
                self.empty,
                (),
                deletion_recompute_pass=True,
            )
        duplicate = GroundedSummaryFailure(
            digest("duplicate"),
            self.predictions[0].user_id,
            self.predictions[0].session_definition_id,
            "summary_failed",
            "session",
        )
        with self.assertRaisesRegex(GroundedSummaryEvaluationError, "duplicated"):
            score_grounded_summaries(
                self.definitions,
                self.claims,
                self.predictions,
                self.empty,
                (duplicate,),
                deletion_recompute_pass=True,
            )

    def test_serialization_is_stable_and_sanitized(self) -> None:
        self.assertEqual(serialize_jsonl(self.predictions), serialize_jsonl(tuple(self.predictions)))
        self.assertEqual(canonical_json_bytes({"b": 1, "a": 2}), b'{"a":2,"b":1}\n')
        failure = GroundedSummaryFailure(
            digest("safe"), "user_001", self.definitions[0].definition_id,
            "summary_failed", "session",
        )
        self.assertNotIn("source text", serialize_jsonl((failure,)).decode())


if __name__ == "__main__":
    unittest.main()
