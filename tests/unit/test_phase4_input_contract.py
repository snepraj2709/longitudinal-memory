from __future__ import annotations

from copy import deepcopy
import unittest

from extraction.phase4_input import (
    PHASE4_CLAIM_FIELDS,
    Phase4InputValidationError,
    build_phase4_source_claims,
    phase4_claim_record,
    validate_phase4_claim_record,
)
from extraction.predicate_registry import load_predicate_registry
from extraction.scaled_source import load_scaled_development_sources


class Phase4InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_predicate_registry(
            "configs/extraction/predicate_registry_v2.json"
        )
        cls.source = load_scaled_development_sources()[0]
        observation = cls.source.observations[0]
        cls.atomic_record = {
            "claim_id": "provider_claim_1",
            "subject_id": "user_001",
            "speaker_id": observation.author_id,
            "predicate": "accepted_role",
            "object": "product engineer",
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": observation.observed_at.isoformat(),
            "valid_to": None,
            "confidence": 0.9,
            "evidence": [
                {
                    "source_id": cls.source.source_id,
                    "message_id": observation.message_id,
                    "quote": observation.text,
                }
            ],
        }

    def test_builds_exact_phase4_shape_and_canonical_id(self) -> None:
        first = build_phase4_source_claims(
            self.source, [self.atomic_record], self.registry
        )[0]
        second = build_phase4_source_claims(
            self.source, [deepcopy(self.atomic_record)], self.registry
        )[0]
        record = phase4_claim_record(first)

        self.assertEqual(first.claim_id, second.claim_id)
        self.assertRegex(first.claim_id, r"^claim_[0-9a-f]{64}$")
        self.assertEqual(set(record), PHASE4_CLAIM_FIELDS)
        self.assertEqual(record["user_id"], "user_001")
        self.assertEqual(record["predicate_registry_version"], "predicate_registry_v2")
        self.assertEqual(record["time_precision"], "timestamp")
        self.assertEqual(
            validate_phase4_claim_record(
                record, {self.source.source_id: self.source}, self.registry
            ),
            first,
        )

    def test_rejects_inexact_unknown_and_cross_user_evidence(self) -> None:
        inexact = deepcopy(self.atomic_record)
        inexact["evidence"][0]["quote"] = "not an exact quote"
        with self.assertRaises(Phase4InputValidationError):
            build_phase4_source_claims(self.source, [inexact], self.registry)

        unknown = deepcopy(self.atomic_record)
        unknown["evidence"][0]["source_id"] = "unknown"
        with self.assertRaises(Phase4InputValidationError):
            build_phase4_source_claims(self.source, [unknown], self.registry)

        claim = build_phase4_source_claims(
            self.source, [self.atomic_record], self.registry
        )[0]
        persisted = phase4_claim_record(claim)
        persisted["user_id"] = "user_002"
        with self.assertRaisesRegex(Phase4InputValidationError, "user boundary"):
            validate_phase4_claim_record(
                persisted, {self.source.source_id: self.source}, self.registry
            )

    def test_rejects_mixed_precision_and_duplicate_canonical_claims(self) -> None:
        mixed = deepcopy(self.atomic_record)
        mixed["valid_to"] = "2026-01-09"
        with self.assertRaisesRegex(Phase4InputValidationError, "mixed precision"):
            build_phase4_source_claims(self.source, [mixed], self.registry)
        duplicate = deepcopy(self.atomic_record)
        duplicate["claim_id"] = "provider_claim_2"
        with self.assertRaisesRegex(Phase4InputValidationError, "collide or duplicate"):
            build_phase4_source_claims(
                self.source, [self.atomic_record, duplicate], self.registry
            )


if __name__ == "__main__":
    unittest.main()
