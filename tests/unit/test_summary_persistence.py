from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import unittest

from storage.contracts import ProcessingOutboxRecord
from summaries.contracts import SessionDefinition, SessionizationResult
from summaries.grounded import GroundedSummaryCoordinator
from summaries.summary_contracts import GroundedSummaryError, SummaryEvidence, load_summary_renderer_config
from summaries.summary_repository import (
    SummaryPersistenceError,
    SummaryPersistenceResult,
    _source_ids,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/summaries/session_summary_renderer_v1.json"
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def evidence(definition_id: str) -> SummaryEvidence:
    return SummaryEvidence(
        evidence_id=digest("evidence"),
        user_id="user_001",
        session_definition_id=definition_id,
        claim_id="claim_001",
        claim_version_id="version_001",
        subject_id="user_001",
        speaker_id="user_001",
        predicate="lives_in",
        object_json={"state": "Delhi"},
        epistemic_status="asserted",
        lifecycle_status="confirmed",
        sensitivity="standard",
        support_type="supports",
        time_precision="day",
        valid_from_date=date(2026, 1, 1),
        valid_from_timestamp=None,
        valid_to_date=None,
        valid_to_timestamp=None,
        transaction_from=BASE,
        transaction_to=None,
        source_id="source_001",
        source_type="chat",
        source_produced_at=BASE,
        source_ingested_at=BASE,
        source_order=0,
        span_id="span_001",
        span_start_offset=0,
        span_end_offset=5,
        exact_quote="Delhi",
    )


class _Connection:
    def transaction(self):
        return nullcontext()


class _SessionService:
    def __init__(self, result: SessionizationResult) -> None:
        self.result = result

    def define_sessions(self, request):
        if request.user_id != self.result.user_id:
            raise AssertionError("user changed")
        return self.result


class _SummaryRepository:
    def __init__(self, values: tuple[SummaryEvidence, ...]) -> None:
        self.values = values
        self.planned = []
        self.retired = 0

    def load_visible_evidence(self, definition, transaction_as_of):
        if definition.transaction_as_of != transaction_as_of:
            raise AssertionError("cutoff changed")
        return self.values

    def persist_in_transaction(self, planned, source_ids):
        self.planned.append((planned, tuple(source_ids)))
        return SummaryPersistenceResult(planned.summary.summary_id, True, False, False)

    def retire_definition(self, *args):
        self.retired += 1
        return True

    def retire_missing_current(self, *args):
        return 0


class SummaryPersistenceUnitTests(unittest.TestCase):
    def definition(self) -> SessionDefinition:
        return SessionDefinition(
            digest("definition"),
            "user_001",
            "chat",
            "declared_thread",
            "session_boundaries_v1",
            ("source_001",),
            BASE,
            BASE,
            BASE + timedelta(days=1),
            digest("membership"),
        )

    def event(self) -> ProcessingOutboxRecord:
        return ProcessingOutboxRecord(
            "event_001",
            "user_001",
            "claims_changed",
            "source_001",
            "claims_changed:source_001",
            {"source_id": "source_001"},
            "pending",
            BASE + timedelta(days=1),
        )

    def coordinator(self, values: tuple[SummaryEvidence, ...]):
        definition = self.definition()
        coordinator = GroundedSummaryCoordinator.__new__(GroundedSummaryCoordinator)
        coordinator.connection = _Connection()
        coordinator.config = load_summary_renderer_config(CONFIG)
        coordinator.session_service = _SessionService(
            SessionizationResult(
                "user_001",
                definition.transaction_as_of,
                "session_boundaries_v1",
                (definition,),
            )
        )
        coordinator.summary_repository = _SummaryRepository(values)
        return coordinator

    def test_source_ids_require_nonempty_unique_ordered_values(self) -> None:
        self.assertEqual(_source_ids(("source_b", "source_a")), ("source_b", "source_a"))
        for invalid in ((), ("source", "source"), ("",), "source"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(GroundedSummaryError):
                    _source_ids(invalid)

    def test_coordinator_uses_existing_event_and_persists_the_definition(self) -> None:
        definition = self.definition()
        coordinator = self.coordinator((evidence(definition.definition_id),))
        result = coordinator.process(self.event())
        self.assertEqual(result.session_count, 1)
        self.assertEqual(result.created_count, 1)
        self.assertEqual(result.replayed_count, 0)
        planned, sources = coordinator.summary_repository.planned[0]
        self.assertEqual(sources, definition.source_ids)
        self.assertEqual(planned.request.transaction_as_of, self.event().created_at)
        self.assertEqual(
            planned.request.idempotency_key,
            f"summary:{self.event().event_id}:{definition.definition_id}",
        )

    def test_coordinator_retires_a_session_with_no_visible_evidence(self) -> None:
        coordinator = self.coordinator(())
        result = coordinator.process(self.event())
        self.assertEqual(result.created_count, 0)
        self.assertEqual(result.retired_count, 1)
        self.assertEqual(coordinator.summary_repository.planned, [])

    def test_coordinator_rejects_an_unrelated_existing_event_type(self) -> None:
        coordinator = self.coordinator(())
        event = ProcessingOutboxRecord(
            "event_001",
            "user_001",
            "conflict_recompute_required",
            "pair_001",
            "conflict:pair_001",
            {},
            "pending",
            BASE + timedelta(days=1),
        )
        coordinator.process(event)
        unrelated = object.__new__(ProcessingOutboxRecord)
        object.__setattr__(unrelated, "event_type", "future_event")
        with self.assertRaises(GroundedSummaryError):
            coordinator.process(unrelated)

    def test_persistence_errors_are_sanitized(self) -> None:
        error = SummaryPersistenceError("stale_summary_read", "evidence")
        self.assertEqual(error.code, "stale_summary_read")
        self.assertEqual(error.location, "evidence")
        self.assertNotIn("Delhi", str(error))


if __name__ == "__main__":
    unittest.main()
