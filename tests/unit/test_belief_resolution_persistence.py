from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime, timezone
import unittest

from conflicts.resolution import PersistedBeliefResolution
from storage.contracts import (
    BeliefResolutionActionRecord,
    BeliefResolutionEvidenceRecord,
    BeliefResolutionRecord,
    ClaimRelationRecord,
    StorageValidationError,
)
from temporal.contracts import TRANSITION_MATRIX


UTC = timezone.utc
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
RESOLVED = datetime(2026, 6, 2, tzinfo=UTC)
SHA = "a" * 64


def _resolution(**changes: object) -> BeliefResolutionRecord:
    values = {
        "resolution_id": "resolution_1",
        "user_id": "user_001",
        "resolver_version": "belief_resolver_v1",
        "policy_version": "belief_resolution_rules_v1",
        "decision_id": "decision_1",
        "idempotency_key": "resolve_1",
        "input_snapshot_sha256": SHA,
        "transaction_as_of": AS_OF,
        "valid_at_date": date(2026, 6, 1),
        "valid_at_timestamp": None,
        "resolved_at": RESOLVED,
        "outcome": "correction_resolved",
        "selected_current_claim_id": "claim_b",
        "authority_reason": "not_applicable",
        "belief_confidence": None,
    }
    values.update(changes)
    return BeliefResolutionRecord(**values)


class BeliefResolutionPersistenceTests(unittest.TestCase):
    def test_records_are_typed_and_keep_belief_confidence_null(self) -> None:
        resolution = _resolution()
        action = BeliefResolutionActionRecord(
            "action_1", "user_001", resolution.resolution_id, 1,
            "claim_a", "candidate", "superseded", "claim_b", "correction",
            "version_a", "version_a_next", "transition_a",
        )
        evidence = BeliefResolutionEvidenceRecord(
            "user_001", resolution.resolution_id, "decision_1", "evidence_1"
        )
        result = PersistedBeliefResolution(
            resolution, (action,), (evidence,), (), False
        )
        self.assertIsNone(result.resolution.belief_confidence)
        self.assertEqual(result.actions[0].replacement_claim_id, "claim_b")
        self.assertEqual(
            {value.name for value in fields(BeliefResolutionEvidenceRecord)},
            {"user_id", "resolution_id", "decision_id", "decision_evidence_id"},
        )

    def test_valid_at_representation_and_current_selection_fail_closed(self) -> None:
        with self.assertRaisesRegex(StorageValidationError, "cannot mix"):
            _resolution(valid_at_timestamp=AS_OF)
        with self.assertRaisesRegex(StorageValidationError, "requires valid_at"):
            _resolution(
                valid_at_date=None,
                selected_current_claim_id="claim_b",
            )
        with self.assertRaisesRegex(StorageValidationError, "remain null"):
            _resolution(belief_confidence=0.8)

    def test_action_and_resolver_relation_provenance_validation(self) -> None:
        with self.assertRaisesRegex(StorageValidationError, "change status"):
            BeliefResolutionActionRecord(
                "action_1", "user_001", "resolution_1", 1,
                "claim_a", "confirmed", "confirmed", None, "reason",
                "version_a", "version_b", "transition_a",
            )
        with self.assertRaisesRegex(StorageValidationError, "wholly"):
            ClaimRelationRecord(
                "relation_1", "user_001", "decision_1", "classifier_v1",
                "claim_b", "claim_a", "supersedes", 1, SHA, RESOLVED,
                resolver_version="belief_resolver_v1",
            )

    def test_candidate_to_historical_is_the_only_matrix_change(self) -> None:
        self.assertEqual(
            TRANSITION_MATRIX["candidate"],
            frozenset(
                {"confirmed", "current", "historical", "disputed", "excluded"}
            ),
        )
        self.assertEqual(TRANSITION_MATRIX["superseded"], frozenset())
        self.assertEqual(TRANSITION_MATRIX["excluded"], frozenset())
        self.assertNotIn("current", TRANSITION_MATRIX["confirmed"])
        self.assertNotIn("confirmed", TRANSITION_MATRIX["historical"])


if __name__ == "__main__":
    unittest.main()
