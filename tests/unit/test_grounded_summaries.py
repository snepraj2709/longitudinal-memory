from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import unittest

from summaries.grounded import plan_grounded_summary, replay_grounded_summary
from summaries.summary_contracts import (
    GroundedSummaryError,
    GroundedSummaryRequest,
    SummaryEvidence,
    canonical_json,
    load_summary_renderer_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/summaries/session_summary_renderer_v1.json"
UTC = timezone.utc
BASE = datetime(2026, 1, 1, 9, tzinfo=UTC)
SESSION_ID = "a" * 64
MEMBERSHIP = "b" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def request(**changes) -> GroundedSummaryRequest:
    values = {
        "user_id": "user_001",
        "session_definition_id": SESSION_ID,
        "session_membership_sha256": MEMBERSHIP,
        "transaction_as_of": BASE + timedelta(days=30),
        "idempotency_key": "summary:user_001:session_001",
    }
    values.update(changes)
    return GroundedSummaryRequest(**values)


def evidence(label: str, **changes) -> SummaryEvidence:
    values = {
        "evidence_id": digest(f"evidence:{label}"),
        "user_id": "user_001",
        "session_definition_id": SESSION_ID,
        "claim_id": f"claim_{label}",
        "claim_version_id": f"version_{label}",
        "subject_id": "user_001",
        "speaker_id": "user_001",
        "predicate": "has_goal",
        "object_json": {"goal": label, "priority": 1},
        "epistemic_status": "asserted",
        "lifecycle_status": "confirmed",
        "sensitivity": "standard",
        "support_type": "supports",
        "time_precision": "day",
        "valid_from_date": date(2026, 1, 1),
        "valid_from_timestamp": None,
        "valid_to_date": date(2026, 1, 31),
        "valid_to_timestamp": None,
        "transaction_from": BASE,
        "transaction_to": None,
        "source_id": f"source_{label}",
        "source_type": "conversation",
        "source_produced_at": BASE,
        "source_ingested_at": BASE,
        "source_order": 0,
        "span_id": f"span_{label}",
        "span_start_offset": 0,
        "span_end_offset": 12,
        "exact_quote": f"source wording for {label}",
        "linked_claim_ids": (),
    }
    values.update(changes)
    return SummaryEvidence(**values)


class GroundedSummaryContractTests(unittest.TestCase):
    def test_frozen_config_identity_and_hash(self) -> None:
        config = load_summary_renderer_config(CONFIG)
        self.assertEqual(config.renderer_version, "session_summary_renderer_v1")
        self.assertEqual(config.observed_facts_heading, "Observed facts")
        self.assertEqual(config.question_triggers, ("uncertain", "disputed"))
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "1e15e3359c292095f7563de1f00f0d348d43030eebe34f68545b0f851e8797a5",
        )

    def test_request_requires_aware_cutoff_hashes_and_version(self) -> None:
        with self.assertRaisesRegex(GroundedSummaryError, "timezone-aware"):
            request(transaction_as_of=datetime(2026, 1, 1))
        with self.assertRaisesRegex(GroundedSummaryError, "SHA-256"):
            request(session_definition_id="bad")
        with self.assertRaisesRegex(GroundedSummaryError, "unsupported"):
            request(renderer_version="renderer_v2")

    def test_object_json_is_detached_unicode_safe_and_canonical(self) -> None:
        value = {"z": ["café", True], "a": 1}
        item = evidence("json", object_json=value)
        value["a"] = 2
        self.assertEqual(item.object_json["a"], 1)
        self.assertEqual(canonical_json(item.object_json), '{"a":1,"z":["café",true]}')
        with self.assertRaisesRegex(GroundedSummaryError, "JSON-safe"):
            evidence("nan", object_json={"bad": float("nan")})
        with self.assertRaisesRegex(GroundedSummaryError, "JSON-safe"):
            evidence("key", object_json={1: "bad"})  # type: ignore[dict-item]

    def test_current_claim_cannot_have_unknown_or_null_valid_time(self) -> None:
        with self.assertRaisesRegex(GroundedSummaryError, "current evidence"):
            evidence(
                "current",
                lifecycle_status="current",
                time_precision="unknown",
                valid_from_date=None,
                valid_to_date=None,
            )

    def test_evidence_time_offsets_and_dispute_links_are_strict(self) -> None:
        with self.assertRaisesRegex(GroundedSummaryError, "offsets"):
            evidence("offset", span_start_offset=0, span_end_offset=None)
        with self.assertRaisesRegex(GroundedSummaryError, "link every side"):
            evidence("dispute", lifecycle_status="disputed")
        with self.assertRaisesRegex(GroundedSummaryError, "only disputed"):
            evidence("linked", linked_claim_ids=("claim_linked", "claim_other"))


class GroundedSummaryRenderingTests(unittest.TestCase):
    def test_claim_fields_render_without_source_paraphrase(self) -> None:
        item = evidence(
            "render",
            predicate="has_relocation_plan",
            object_json={"city": "Pune", "year": 2027},
            exact_quote="I might perhaps travel somewhere later",
        )
        result = plan_grounded_summary(request(), (item,), config_path=CONFIG)
        statement = result.summary.observed_facts[0]
        self.assertEqual(
            statement.text,
            'user_001 has relocation plan {"city":"Pune","year":2027}.',
        )
        self.assertNotIn(item.exact_quote, statement.text)
        self.assertEqual(statement.evidence, (item,))

    def test_candidate_reported_uncertain_denied_and_corrected_qualifiers(self) -> None:
        items = (
            evidence("candidate", lifecycle_status="candidate", speaker_id="manager"),
            evidence("reported", epistemic_status="reported_by_other", speaker_id="manager", source_order=1),
            evidence("uncertain", epistemic_status="uncertain", speaker_id="user_001", source_order=2),
            evidence("denied", epistemic_status="denied", speaker_id="user_001", source_order=3),
            evidence("corrected", epistemic_status="corrected", speaker_id="user_001", source_order=4),
        )
        result = plan_grounded_summary(request(), items, config_path=CONFIG)
        texts = [item.text for item in result.summary.observed_facts]
        self.assertTrue(texts[0].startswith("Candidate, attributed to manager:"))
        self.assertTrue(texts[1].startswith("Reported by manager:"))
        self.assertTrue(texts[2].startswith("user_001 was uncertain:"))
        self.assertTrue(texts[3].startswith("user_001 denied:"))
        self.assertTrue(texts[4].startswith("user_001 corrected the record:"))
        self.assertEqual(len(result.summary.unresolved_questions), 1)
        self.assertEqual(
            result.summary.unresolved_questions[0].claim_ids,
            ("claim_uncertain",),
        )

    def test_historical_and_superseded_statements_remain_visible(self) -> None:
        historical = evidence("historical", lifecycle_status="historical")
        superseded = evidence("superseded", lifecycle_status="superseded", source_order=1)
        result = plan_grounded_summary(request(), (historical, superseded), config_path=CONFIG)
        self.assertEqual(len(result.summary.observed_facts), 2)
        self.assertTrue(result.summary.observed_facts[0].text.startswith("Historical:"))
        self.assertTrue(result.summary.observed_facts[1].text.startswith("Superseded:"))
        self.assertIs(result.summary.historical_view[0], result.summary.observed_facts[0])
        self.assertIs(result.summary.historical_view[1], result.summary.observed_facts[1])

    def test_dispute_keeps_both_sides_and_one_exact_question(self) -> None:
        links = ("claim_left", "claim_right")
        left = evidence(
            "left",
            lifecycle_status="disputed",
            linked_claim_ids=links,
            object_json={"office": "Pune"},
        )
        right = evidence(
            "right",
            lifecycle_status="disputed",
            linked_claim_ids=links,
            object_json={"office": "Mumbai"},
            source_order=1,
        )
        result = plan_grounded_summary(request(), (left, right), config_path=CONFIG)
        self.assertEqual(len(result.summary.disputed_view), 2)
        self.assertEqual(len(result.summary.unresolved_questions), 1)
        question = result.summary.unresolved_questions[0]
        self.assertEqual(question.claim_ids, links)
        self.assertEqual(question.evidence, (left, right))
        self.assertIn('{"office":"Pune"}', question.text)
        self.assertIn('{"office":"Mumbai"}', question.text)

    def test_missing_or_inconsistent_dispute_side_fails(self) -> None:
        links = ("claim_left", "claim_right")
        left = evidence("left", lifecycle_status="disputed", linked_claim_ids=links)
        with self.assertRaisesRegex(GroundedSummaryError, "missing"):
            plan_grounded_summary(request(), (left,), config_path=CONFIG)
        right = evidence(
            "right",
            lifecycle_status="disputed",
            linked_claim_ids=("claim_left", "claim_right", "claim_third"),
        )
        with self.assertRaisesRegex(GroundedSummaryError, "missing|inconsistent"):
            plan_grounded_summary(request(), (left, right), config_path=CONFIG)

    def test_only_explicit_uncertainty_and_dispute_create_questions(self) -> None:
        statuses = ("asserted", "inferred", "reported_by_other", "denied", "corrected")
        items = tuple(
            evidence(f"status_{index}", epistemic_status=status, source_order=index)
            for index, status in enumerate(statuses)
        )
        result = plan_grounded_summary(request(), items, config_path=CONFIG)
        self.assertEqual(result.summary.unresolved_questions, ())

    def test_exclusions_and_sensitive_flag_are_explicit(self) -> None:
        items = (
            evidence("excluded", lifecycle_status="excluded"),
            evidence("hypothetical", epistemic_status="hypothetical", source_order=1),
            evidence("restricted", sensitivity="restricted", source_order=2),
            evidence("sensitive", sensitivity="sensitive", source_order=3),
        )
        result = plan_grounded_summary(request(), items, config_path=CONFIG)
        self.assertEqual(result.summary.claim_ids, ("claim_sensitive",))
        self.assertTrue(result.summary.contains_sensitive)
        self.assertTrue(result.summary.observed_facts[0].sensitive)
        self.assertNotIn("restricted", result.summary.summary_text)

    def test_summary_uses_exact_headings_statement_texts_and_empty_marker(self) -> None:
        item = evidence("heading")
        result = plan_grounded_summary(request(), (item,), config_path=CONFIG)
        text = result.summary.observed_facts[0].text
        self.assertEqual(
            result.summary.summary_text,
            f"Observed facts\n- {text}\n\nUnresolved questions\n- None.",
        )
        filtered = evidence("filtered", lifecycle_status="excluded")
        empty = plan_grounded_summary(request(), (filtered,), config_path=CONFIG)
        self.assertEqual(
            empty.summary.summary_text,
            "Observed facts\n- None.\n\nUnresolved questions\n- None.",
        )

    def test_statement_order_uses_source_time_order_offset_claim_and_kind(self) -> None:
        items = (
            evidence("late", source_produced_at=BASE + timedelta(hours=1)),
            evidence("offset_b", source_order=1, span_start_offset=20, span_end_offset=30),
            evidence("offset_a", source_order=1, span_start_offset=10, span_end_offset=20),
            evidence("order", source_order=0),
        )
        result = plan_grounded_summary(request(), items, config_path=CONFIG)
        self.assertEqual(
            [item.claim_ids[0] for item in result.summary.observed_facts],
            ["claim_order", "claim_offset_a", "claim_offset_b", "claim_late"],
        )


class GroundedSummaryTimeAndIdentityTests(unittest.TestCase):
    def test_date_aggregation_and_missing_boundary(self) -> None:
        items = (
            evidence("date_a", valid_from_date=date(2026, 1, 2), valid_to_date=date(2026, 1, 4)),
            evidence("date_b", valid_from_date=date(2026, 1, 1), valid_to_date=date(2026, 1, 8), source_order=1),
        )
        summary = plan_grounded_summary(request(), items, config_path=CONFIG).summary
        self.assertEqual(
            (summary.valid_time_kind, summary.valid_time_start_date, summary.valid_time_end_date),
            ("date", date(2026, 1, 1), date(2026, 1, 8)),
        )
        missing = replace(items[1], valid_to_date=None)
        summary = plan_grounded_summary(request(), (items[0], missing), config_path=CONFIG).summary
        self.assertEqual(summary.valid_time_kind, "date")
        self.assertIsNone(summary.valid_time_end_date)

    def test_timestamp_unknown_and_mixed_aggregation(self) -> None:
        timestamp = evidence(
            "timestamp",
            time_precision="timestamp",
            valid_from_date=None,
            valid_to_date=None,
            valid_from_timestamp=BASE,
            valid_to_timestamp=BASE + timedelta(hours=2),
        )
        summary = plan_grounded_summary(request(), (timestamp,), config_path=CONFIG).summary
        self.assertEqual(summary.valid_time_kind, "timestamp")
        self.assertEqual(summary.valid_time_start_timestamp, BASE)
        unknown = evidence(
            "unknown",
            time_precision="unknown",
            valid_from_date=None,
            valid_to_date=None,
            source_order=1,
        )
        summary = plan_grounded_summary(request(), (unknown,), config_path=CONFIG).summary
        self.assertEqual(summary.valid_time_kind, "unknown")
        self.assertIsNone(summary.valid_time_start_date)
        summary = plan_grounded_summary(request(), (timestamp, unknown), config_path=CONFIG).summary
        self.assertEqual(summary.valid_time_kind, "mixed")
        self.assertIsNone(summary.valid_time_start_timestamp)

    def test_visibility_user_session_and_half_open_cutoff_fail_closed(self) -> None:
        item = evidence("visible")
        invalid = (
            replace(item, user_id="user_002"),
            replace(item, session_definition_id="c" * 64),
            replace(item, source_ingested_at=request().transaction_as_of + timedelta(seconds=1)),
            replace(item, transaction_from=request().transaction_as_of + timedelta(seconds=1)),
            replace(item, transaction_to=request().transaction_as_of),
        )
        for changed in invalid:
            with self.subTest(changed=changed.source_id):
                with self.assertRaises(GroundedSummaryError):
                    plan_grounded_summary(request(), (changed,), config_path=CONFIG)

    def test_snapshot_and_ids_bind_rendered_inputs_not_control_fields(self) -> None:
        item = evidence("stable")
        first = plan_grounded_summary(request(), (item,), config_path=CONFIG)
        second = plan_grounded_summary(request(), (item,), config_path=CONFIG)
        self.assertEqual(first, second)
        offset = timezone(timedelta(hours=5, minutes=30))
        equivalent_request = request(
            transaction_as_of=request().transaction_as_of.astimezone(offset)
        )
        equivalent_item = replace(
            item,
            transaction_from=item.transaction_from.astimezone(offset),
            source_produced_at=item.source_produced_at.astimezone(offset),
            source_ingested_at=item.source_ingested_at.astimezone(offset),
        )
        equivalent = plan_grounded_summary(
            equivalent_request, (equivalent_item,), config_path=CONFIG
        )
        self.assertEqual(first.input_snapshot_sha256, equivalent.input_snapshot_sha256)
        self.assertEqual(first.summary.summary_id, equivalent.summary.summary_id)
        excluded = evidence("excluded_snapshot", lifecycle_status="excluded", source_order=1)
        changed = plan_grounded_summary(request(), (item, excluded), config_path=CONFIG)
        self.assertEqual(first.input_snapshot_sha256, changed.input_snapshot_sha256)
        self.assertEqual(first.summary.summary_text, changed.summary.summary_text)
        later_request = request(
            transaction_as_of=request().transaction_as_of + timedelta(hours=1),
            idempotency_key="later-control-key",
        )
        later = plan_grounded_summary(later_request, (item,), config_path=CONFIG)
        self.assertEqual(first.input_snapshot_sha256, later.input_snapshot_sha256)
        self.assertEqual(first.summary.summary_id, later.summary.summary_id)

    def test_idempotent_replay_returns_existing_and_rejects_drift(self) -> None:
        first = plan_grounded_summary(request(), (evidence("replay"),), config_path=CONFIG)
        replay = plan_grounded_summary(request(), (evidence("replay"),), config_path=CONFIG)
        value, created = replay_grounded_summary(first, replay)
        self.assertIs(value, first)
        self.assertFalse(created)
        drift = plan_grounded_summary(
            request(),
            (replace(evidence("replay"), object_json={"goal": "changed"}),),
            config_path=CONFIG,
        )
        with self.assertRaisesRegex(GroundedSummaryError, "input drift"):
            replay_grounded_summary(first, drift)
        different_key = plan_grounded_summary(
            request(idempotency_key="different"), (evidence("replay"),), config_path=CONFIG
        )
        with self.assertRaisesRegex(GroundedSummaryError, "different idempotency"):
            replay_grounded_summary(first, different_key)

    def test_duplicate_evidence_and_claim_semantic_drift_fail(self) -> None:
        item = evidence("duplicate")
        with self.assertRaisesRegex(GroundedSummaryError, "duplicated"):
            plan_grounded_summary(request(), (item, item), config_path=CONFIG)
        other = replace(
            item,
            evidence_id=digest("second evidence"),
            span_id="span_other",
            object_json={"goal": "different"},
        )
        with self.assertRaisesRegex(GroundedSummaryError, "inconsistent semantics"):
            plan_grounded_summary(request(), (item, other), config_path=CONFIG)

    def test_statement_claim_ids_must_match_evidence_claims_exactly(self) -> None:
        item = evidence("exact")
        other = evidence("other", source_order=1)
        statement = plan_grounded_summary(
            request(), (item,), config_path=CONFIG
        ).summary.observed_facts[0]
        with self.assertRaisesRegex(GroundedSummaryError, "match exactly"):
            replace(statement, evidence=(item, other))


if __name__ == "__main__":
    unittest.main()
