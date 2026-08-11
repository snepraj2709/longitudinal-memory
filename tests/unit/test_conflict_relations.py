from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from conflicts.relations import conflict_decision_evidence_id
from storage.contracts import (
    CLAIM_RELATION_TYPES,
    ClaimRelationRecord,
    ConflictDecisionEvidenceRecord,
    ConflictDecisionRecord,
    StorageValidationError,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 1, tzinfo=UTC)
SHA = "a" * 64


def _decision() -> ConflictDecisionRecord:
    return ConflictDecisionRecord(
        "decision_1",
        "user_001",
        "relation_classifier_v1",
        "relation_rules_v1",
        "pair_1",
        "claim_a",
        "claim_b",
        "version_a",
        "version_b",
        SHA,
        NOW,
        "hard_contradiction",
        "hard_contradiction",
        NOW,
    )


class ConflictRelationContractTests(unittest.TestCase):
    def test_decision_record_freezes_pair_versions_rule_and_aware_times(self) -> None:
        record = _decision()
        self.assertEqual(record.left_version_id, "version_a")
        self.assertEqual(record.matched_rule, record.label)
        with self.assertRaisesRegex(StorageValidationError, "canonical"):
            replace(record, left_claim_id="claim_z")
        with self.assertRaisesRegex(StorageValidationError, "matched_rule"):
            replace(record, matched_rule="unrelated")
        with self.assertRaisesRegex(StorageValidationError, "timezone-aware"):
            replace(record, transaction_as_of=datetime(2026, 8, 1))

    def test_relation_vocabulary_orientation_and_confidence_are_strict(self) -> None:
        relation = ClaimRelationRecord(
            "relation_1", "user_001", "decision_1", "relation_classifier_v1",
            "claim_a", "claim_b", "contradicts", 1, SHA, NOW,
        )
        self.assertEqual(relation.confidence, 1.0)
        with self.assertRaisesRegex(StorageValidationError, "canonical"):
            replace(relation, source_claim_id="claim_z")
        self.assertEqual(
            CLAIM_RELATION_TYPES,
            {
                "supports", "contradicts", "corrects", "supersedes", "refines",
                "same_event_as", "caused_by", "hindered_by", "same_topic_as",
            },
        )
        self.assertEqual(replace(relation, relation_type="supersedes").relation_type, "supersedes")
        with self.assertRaisesRegex(StorageValidationError, "canonical"):
            replace(
                relation,
                relation_type="same_event_as",
                source_claim_id="claim_z",
            )
        with self.assertRaisesRegex(StorageValidationError, "invalid"):
            replace(relation, relation_type="unknown_relation")
        with self.assertRaisesRegex(StorageValidationError, "must be one"):
            replace(relation, confidence=0.9)

    def test_decision_evidence_id_is_stable_and_user_snapshot_scoped(self) -> None:
        first = conflict_decision_evidence_id(
            "user_001", "decision_1", "claim_a", "span_1", "supports", SHA
        )
        self.assertEqual(
            first,
            conflict_decision_evidence_id(
                "user_001", "decision_1", "claim_a", "span_1", "supports", SHA
            ),
        )
        self.assertNotEqual(
            first,
            conflict_decision_evidence_id(
                "user_002", "decision_1", "claim_a", "span_1", "supports", SHA
            ),
        )
        record = ConflictDecisionEvidenceRecord(
            first, "user_001", "decision_1", "claim_a", "span_1", "supports", SHA
        )
        self.assertEqual(record.input_snapshot_sha256, SHA)
        with self.assertRaisesRegex(StorageValidationError, "support_type"):
            replace(record, support_type="mentions")


if __name__ == "__main__":
    unittest.main()
