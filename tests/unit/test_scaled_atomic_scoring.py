from __future__ import annotations

import unittest

from extraction.phase4_input import build_phase4_source_claims
from extraction.predicate_registry import load_predicate_registry
from extraction.scaled_scoring import (
    load_scaled_development_gold,
    qualification_gate,
    score_scaled_development,
)
from extraction.scaled_source import (
    QUALIFICATION_SOURCE_REFS,
    load_scaled_development_sources,
    select_scaled_sources,
)


def _atomic_record(claim: object) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "subject_id": claim.subject_id,
        "speaker_id": claim.speaker_id,
        "predicate": claim.predicate,
        "object": claim.object,
        "polarity": claim.polarity,
        "epistemic_status": claim.epistemic_status,
        "valid_from": claim.valid_from,
        "valid_to": claim.valid_to,
        "confidence": claim.confidence,
        "evidence": [
            {"source_id": item.source_id, "message_id": item.message_id, "quote": item.quote}
            for item in claim.evidence
        ],
    }


class ScaledAtomicScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_predicate_registry(
            "configs/extraction/predicate_registry_v2.json"
        )
        cls.all_sources = load_scaled_development_sources()

    def test_loads_only_development_gold_after_source_selection(self) -> None:
        full = load_scaled_development_gold(
            ".", self.all_sources, self.registry
        )
        qualification_sources = select_scaled_sources(
            self.all_sources, QUALIFICATION_SOURCE_REFS
        )
        qualification = load_scaled_development_gold(
            ".",
            self.all_sources,
            self.registry,
            selected_source_ids={item.source_id for item in qualification_sources},
        )

        self.assertEqual(len(full), 20)
        self.assertEqual(sum(len(case.expected_claims) for case in full), 32)
        self.assertEqual(len(qualification), 4)

    def test_perfect_predictions_score_overall_and_null_slices(self) -> None:
        gold = load_scaled_development_gold(".", self.all_sources, self.registry)
        source_map = {source.source_id: source for source in self.all_sources}
        predictions = {
            case.source_id: build_phase4_source_claims(
                source_map[case.source_id],
                [_atomic_record(claim) for claim in case.expected_claims],
                self.registry,
            )
            for case in gold
        }
        scores = score_scaled_development(
            gold, predictions, self.all_sources, self.registry
        )

        self.assertEqual(scores["micro_claim_f1"]["value"], 1.0)
        self.assertEqual(scores["provenance_span_recall"]["value"], 1.0)
        self.assertIsNone(
            scores["slices"]["predicate_family"]["identity"]["micro_claim_f1"]["value"]
        )

    def test_qualification_gate_allows_at_most_one_match_gap(self) -> None:
        scores = {
            "case_results": [{"true_positives": 5}],
            "total_unsupported_claims": 0,
            "provenance_span_precision": {"numerator": 5},
            "subject_accuracy": {"value": 1.0},
            "speaker_accuracy": {"value": 1.0},
        }
        mini = {
            **scores,
            "case_results": [{"true_positives": 4}],
            "provenance_span_precision": {"numerator": 4},
        }
        zero = {name: 0 for name in ("execution", "persistence", "cross_user", "provenance")}

        self.assertTrue(
            qualification_gate(
                scores, mini, full_failures=zero, mini_failures=zero
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
