from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import unittest

from summaries.durative import infer_durative_claim
from summaries.durative_contracts import (
    DurativeClaimError,
    DurativeEpisode,
    DurativeInferenceRequest,
    DurativeInferenceResult,
    DurativePropositionRequest,
    canonical_json,
    load_durative_rules_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/summaries/durative_claim_rules_v1.json"
UTC = timezone.utc
BASE = datetime(2026, 1, 1, 9, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request(**changes: object) -> DurativePropositionRequest:
    values: dict[str, object] = {
        "user_id": "user_001",
        "subject_id": "user_001",
        "predicate": "work_preference",
        "predicate_registry_version": "predicate_registry_v2",
        "object_json": "remote work",
        "polarity": "positive",
        "transaction_as_of": BASE + timedelta(days=30),
        "idempotency_key": "durative:user_001:remote-work",
    }
    values.update(changes)
    return DurativePropositionRequest(**values)  # type: ignore[arg-type]


def episode(label: str, **changes: object) -> DurativeEpisode:
    index = int(label[-1]) if label[-1:].isdigit() else 0
    values: dict[str, object] = {
        "episode_id": digest(f"episode:{label}"),
        "user_id": "user_001",
        "claim_id": f"claim_{label}",
        "claim_version_id": f"claim_version_{label}",
        "session_definition_id": digest(f"session:{label}"),
        "source_id": f"source_{label}",
        "span_id": f"span_{label}",
        "subject_id": "user_001",
        "speaker_id": "user_001",
        "predicate": "work_preference",
        "predicate_registry_version": "predicate_registry_v2",
        "object_json": "remote work",
        "polarity": "positive",
        "epistemic_status": "asserted",
        "lifecycle_status": "confirmed",
        "memory_kind": "episodic",
        "sensitivity": "standard",
        "extraction_confidence": 0.9,
        "belief_confidence": None,
        "support_type": "supports",
        "time_precision": "unknown",
        "valid_from_date": None,
        "valid_from_timestamp": None,
        "valid_to_date": None,
        "valid_to_timestamp": None,
        "episode_at": date(2026, 1, 1) + timedelta(days=index),
        "transaction_from": BASE,
        "transaction_to": None,
    }
    values.update(changes)
    return DurativeEpisode(**values)  # type: ignore[arg-type]


class DurativeContractTests(unittest.TestCase):
    def test_public_inference_contract_is_run_level(self) -> None:
        self.assertEqual(
            tuple(item.name for item in fields(DurativeInferenceRequest)),
            ("user_id", "transaction_as_of", "idempotency_key", "rule_version"),
        )
        self.assertEqual(
            tuple(item.name for item in fields(DurativeInferenceResult)),
            (
                "run_id", "user_id", "rule_version", "input_snapshot_sha256",
                "decisions", "created_count", "replayed_count",
            ),
        )

    def test_frozen_config_identity_registry_and_hash(self) -> None:
        config = load_durative_rules_config(CONFIG)
        self.assertEqual(config.rules_version, "durative_claim_rules_v1")
        self.assertEqual(
            config.allowed_predicate_families,
            ("belief", "goal", "preference", "relationship", "role", "state"),
        )
        self.assertEqual(config.minimum_distinct_sessions, 2)
        self.assertEqual(config.minimum_distinct_sources, 2)
        self.assertEqual(config.minimum_distinct_episode_times, 2)
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "680de008e33de6824b8fded16f8e6fdc6877130d9a2caa35908be1c45c1a0b7b",
        )

    def test_request_requires_aware_cutoff_and_json_safe_object(self) -> None:
        with self.assertRaisesRegex(DurativeClaimError, "timezone-aware"):
            request(transaction_as_of=datetime(2026, 1, 1))
        with self.assertRaisesRegex(DurativeClaimError, "JSON-safe"):
            request(object_json={"bad": float("nan")})
        with self.assertRaisesRegex(DurativeClaimError, "JSON-safe"):
            request(object_json={1: "bad"})

    def test_episode_requires_valid_time_and_aware_transaction_interval(self) -> None:
        with self.assertRaisesRegex(DurativeClaimError, "cannot mix"):
            episode(
                "mix", time_precision="day",
                valid_from_date=date(2026, 1, 1),
                valid_to_date=date(2026, 1, 2),
                valid_from_timestamp=BASE,
            )
        with self.assertRaisesRegex(DurativeClaimError, "timezone-aware"):
            episode("naive", transaction_from=datetime(2026, 1, 1))

    def test_canonical_json_preserves_unicode_and_sorting(self) -> None:
        self.assertEqual(
            canonical_json({"z": ["café", True], "a": 1}),
            '{"a":1,"z":["café",true]}',
        )


class DurativeInferenceTests(unittest.TestCase):
    def test_repeated_exact_proposition_creates_candidate_durative_plan(self) -> None:
        first = episode("support1", extraction_confidence=0.85)
        second = episode(
            "support2", epistemic_status="corrected",
            lifecycle_status="historical", sensitivity="sensitive",
            extraction_confidence=0.7,
        )
        result = infer_durative_claim(request(), (second, first), config_path=CONFIG)

        self.assertEqual(result.decision.status, "accepted")
        self.assertEqual(result.decision.reason, "accepted_repeated_episodes")
        self.assertIsNotNone(result.claim)
        claim = result.claim
        assert claim is not None
        self.assertEqual(
            (claim.memory_kind, claim.epistemic_status, claim.lifecycle_status),
            ("durative", "inferred", "candidate"),
        )
        self.assertEqual(claim.speaker_id, "memory_system")
        self.assertEqual(claim.extraction_confidence, 0.7)
        self.assertIsNone(claim.belief_confidence)
        self.assertEqual(claim.sensitivity, "sensitive")
        self.assertEqual(
            {item.source_id for item in claim.evidence},
            {first.source_id, second.source_id},
        )
        self.assertEqual(claim.extraction.extractor_kind, "deterministic_rules")
        self.assertEqual(
            claim.extraction.input_snapshot_sha256,
            result.input_snapshot_sha256,
        )

    def test_one_explicit_closed_interval_is_sufficient_and_copied(self) -> None:
        support = episode(
            "closed", time_precision="day",
            valid_from_date=date(2025, 4, 1),
            valid_to_date=date(2025, 7, 31),
            episode_at=None,
        )
        result = infer_durative_claim(request(), (support,), config_path=CONFIG)
        self.assertEqual(result.decision.reason, "accepted_explicit_closed_interval")
        assert result.claim is not None
        self.assertEqual(result.claim.valid_from_date, date(2025, 4, 1))
        self.assertEqual(result.claim.valid_to_date, date(2025, 7, 31))
        self.assertEqual(result.claim.time_precision, "day")

    def test_repetition_requires_two_sessions_sources_and_episode_times(self) -> None:
        first = episode("repeat1")
        variants = (
            replace(episode("repeat2"), session_definition_id=first.session_definition_id),
            replace(episode("repeat2"), source_id=first.source_id),
            replace(episode("repeat2"), episode_at=first.episode_at),
        )
        for second in variants:
            with self.subTest(second=second):
                result = infer_durative_claim(request(), (first, second), config_path=CONFIG)
                self.assertEqual(result.decision.reason, "insufficient_repetition")

    def test_non_durative_or_disallowed_family_is_rejected(self) -> None:
        cases = (
            ("rated_first_work_week", "good"),
            ("product_work_sentiment", "good"),
        )
        for predicate, value in cases:
            with self.subTest(predicate=predicate):
                req = request(predicate=predicate, object_json=value)
                item = episode("family", predicate=predicate, object_json=value)
                result = infer_durative_claim(req, (item,), config_path=CONFIG)
                self.assertEqual(result.decision.reason, "predicate_not_durative")

    def test_belief_interval_family_is_allowed(self) -> None:
        predicate = "believes_marketing_is_right_fit"
        items = (
            episode("belief1", predicate=predicate, object_json=True),
            episode("belief2", predicate=predicate, object_json=True),
        )
        result = infer_durative_claim(
            request(predicate=predicate, object_json=True), items, config_path=CONFIG
        )
        self.assertEqual(result.decision.status, "accepted")

    def test_no_exact_support_and_input_accounting(self) -> None:
        other = episode("other", object_json="office work")
        result = infer_durative_claim(request(), (other,), config_path=CONFIG)
        self.assertEqual(result.decision.reason, "no_exact_support")
        self.assertEqual(result.decision.support_episode_ids, ())
        self.assertEqual(result.decision.ignored_episode_ids, (other.episode_id,))

    def test_restricted_and_recursive_evidence_are_ineligible(self) -> None:
        cases = (
            (episode("restricted", sensitivity="restricted"), "restricted_evidence"),
            (episode("recursive", memory_kind="durative"), "recursive_durative"),
            (episode("untyped", memory_kind=None), "memory_kind_ineligible"),
        )
        for item, reason in cases:
            with self.subTest(reason=reason):
                result = infer_durative_claim(request(), (item,), config_path=CONFIG)
                self.assertEqual(result.decision.reason, reason)
                self.assertIsNone(result.claim)

    def test_visibility_is_inclusive_at_start_and_half_open_at_end(self) -> None:
        cutoff = request().transaction_as_of
        start = episode("visible", transaction_from=cutoff)
        self.assertNotEqual(
            infer_durative_claim(request(), (start,), config_path=CONFIG).decision.reason,
            "not_visible",
        )
        ended = episode("ended", transaction_to=cutoff)
        result = infer_durative_claim(request(), (ended,), config_path=CONFIG)
        self.assertEqual(result.decision.reason, "not_visible")

    def test_ineligible_exact_assertions_are_independent_counterevidence(self) -> None:
        cases = (
            episode("candidate", lifecycle_status="candidate"),
            episode("disputed", lifecycle_status="disputed"),
            episode("denied", epistemic_status="denied"),
            episode("uncertain", epistemic_status="uncertain"),
            episode("hypothetical", epistemic_status="hypothetical"),
        )
        for item in cases:
            with self.subTest(item=item):
                result = infer_durative_claim(request(), (item,), config_path=CONFIG)
                self.assertEqual(result.decision.reason, "counterevidence")
                self.assertEqual(result.decision.counter_episode_ids, (item.episode_id,))

    def test_explicit_contradiction_or_correction_blocks_inference(self) -> None:
        supports = (episode("support1"), episode("support2"))
        for support_type in ("contradicts", "corrects"):
            counter = episode(f"counter_{support_type}", support_type=support_type)
            result = infer_durative_claim(
                request(), (*supports, counter), config_path=CONFIG
            )
            self.assertEqual(result.decision.reason, "counterevidence")
            self.assertEqual(result.decision.counter_episode_ids, (counter.episode_id,))

    def test_opposite_polarity_same_object_blocks_inference(self) -> None:
        counter = episode("opposite", polarity="negative")
        result = infer_durative_claim(
            request(), (episode("support1"), episode("support2"), counter),
            config_path=CONFIG,
        )
        self.assertEqual(result.decision.reason, "counterevidence")

    def test_single_value_different_object_is_counter_but_multi_value_is_not(self) -> None:
        single_request = request(predicate="office_base", object_json="Pune")
        single_supports = (
            episode("office1", predicate="office_base", object_json="Pune"),
            episode("office2", predicate="office_base", object_json="Pune"),
        )
        single_counter = episode(
            "office_counter", predicate="office_base", object_json="Mumbai"
        )
        blocked = infer_durative_claim(
            single_request, (*single_supports, single_counter), config_path=CONFIG
        )
        self.assertEqual(blocked.decision.reason, "counterevidence")

        multi_other = episode("multi_other", object_json="quiet mornings")
        accepted = infer_durative_claim(
            request(), (episode("support1"), episode("support2"), multi_other),
            config_path=CONFIG,
        )
        self.assertEqual(accepted.decision.status, "accepted")
        self.assertIn(multi_other.episode_id, accepted.decision.ignored_episode_ids)

    def test_unresolved_persisted_conflict_or_correction_blocks(self) -> None:
        supports = (episode("support1"), episode("support2"))
        cases = (
            episode("conflict", conflict_labels=("unresolved_ambiguity",)),
            episode("correction", relation_types=("corrects",)),
        )
        for counter in cases:
            with self.subTest(counter=counter):
                result = infer_durative_claim(
                    request(), (*supports, counter), config_path=CONFIG
                )
                self.assertEqual(result.decision.reason, "counterevidence")

    def test_incompatible_value_only_blocks_overlapping_or_unknown_time(self) -> None:
        supports = (
            episode(
                "office1", predicate="office_base", object_json="Pune",
                time_precision="day", valid_from_date=date(2025, 1, 1),
                valid_to_date=date(2025, 1, 2),
            ),
            episode(
                "office2", predicate="office_base", object_json="Pune",
                time_precision="day", valid_from_date=date(2025, 1, 3),
                valid_to_date=date(2025, 1, 4),
            ),
        )
        req = request(predicate="office_base", object_json="Pune")
        disjoint = episode(
            "office_old", predicate="office_base", object_json="Mumbai",
            time_precision="day", valid_from_date=date(2025, 2, 1),
            valid_to_date=date(2025, 2, 2),
            conflict_labels=("temporal_change",),
        )
        self.assertEqual(
            infer_durative_claim(req, (*supports, disjoint), config_path=CONFIG).decision.status,
            "accepted",
        )
        overlap = replace(
            disjoint, episode_id=digest("episode:office_overlap"),
            source_id="source_office_overlap", span_id="span_office_overlap",
            session_definition_id=digest("session:office_overlap"),
            valid_from_date=date(2025, 1, 4), valid_to_date=date(2025, 1, 5),
            conflict_labels=(),
        )
        self.assertEqual(
            infer_durative_claim(req, (*supports, overlap), config_path=CONFIG).decision.reason,
            "counterevidence",
        )

    def test_complete_date_boundaries_form_approximate_inclusive_range(self) -> None:
        first = episode(
            "date1", episode_at=date(2025, 3, 5), time_precision="day",
            valid_from_date=date(2025, 3, 1), valid_to_date=date(2025, 3, 10),
        )
        second = episode(
            "date2", episode_at=date(2025, 1, 2), time_precision="day",
            valid_from_date=date(2025, 1, 1), valid_to_date=date(2025, 1, 5),
        )
        result = infer_durative_claim(request(), (first, second), config_path=CONFIG)
        assert result.claim is not None
        self.assertEqual(result.claim.time_precision, "approximate")
        self.assertEqual(result.claim.valid_from_date, date(2025, 1, 1))
        self.assertEqual(result.claim.valid_to_date, date(2025, 3, 10))

    def test_complete_timestamp_boundaries_form_approximate_range(self) -> None:
        first = episode(
            "time1", episode_at=BASE + timedelta(hours=4),
            time_precision="timestamp", valid_from_timestamp=BASE,
            valid_to_timestamp=BASE + timedelta(hours=5),
        )
        second = episode(
            "time2", episode_at=BASE + timedelta(hours=1),
            time_precision="timestamp", valid_from_timestamp=BASE - timedelta(hours=2),
            valid_to_timestamp=BASE + timedelta(hours=2),
        )
        result = infer_durative_claim(request(), (first, second), config_path=CONFIG)
        assert result.claim is not None
        self.assertEqual(result.claim.time_precision, "approximate")
        self.assertEqual(result.claim.valid_from_timestamp, BASE - timedelta(hours=2))
        self.assertEqual(result.claim.valid_to_timestamp, BASE + timedelta(hours=5))

    def test_missing_or_mixed_episode_times_produce_unknown_time(self) -> None:
        cases = (
            (episode("known1"), episode("known2"), episode("missing", episode_at=None)),
            (episode("date1"), episode("time2", episode_at=BASE + timedelta(days=2))),
        )
        for items in cases:
            with self.subTest(items=items):
                result = infer_durative_claim(request(), items, config_path=CONFIG)
                assert result.claim is not None
                self.assertEqual(result.claim.time_precision, "unknown")
                self.assertIsNone(result.claim.valid_from_date)
                self.assertIsNone(result.claim.valid_from_timestamp)

    def test_user_isolation_is_enforced_before_inference(self) -> None:
        with self.assertRaisesRegex(DurativeClaimError, "cross-user"):
            infer_durative_claim(
                request(), (episode("foreign", user_id="user_002"),),
                config_path=CONFIG,
            )

    def test_unknown_predicate_and_invalid_object_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(DurativeClaimError, "unknown"):
            infer_durative_claim(
                request(predicate="not_registered"), (), config_path=CONFIG
            )
        with self.assertRaisesRegex(DurativeClaimError, "does not match"):
            infer_durative_claim(
                request(object_json={"not": "text"}), (), config_path=CONFIG
            )

    def test_replay_order_ids_and_idempotency_intent_are_stable(self) -> None:
        first = episode("stable1")
        second = episode("stable2")
        left = infer_durative_claim(request(), (first, second), config_path=CONFIG)
        reordered = infer_durative_claim(request(), (second, first), config_path=CONFIG)
        self.assertEqual(left, reordered)

        drifted = infer_durative_claim(
            request(), (replace(first, extraction_confidence=0.8), second),
            config_path=CONFIG,
        )
        self.assertEqual(left.plan_id, drifted.plan_id)
        self.assertNotEqual(left.input_snapshot_sha256, drifted.input_snapshot_sha256)
        self.assertEqual(left.claim.claim_id, drifted.claim.claim_id)  # type: ignore[union-attr]

        new_intent = infer_durative_claim(
            request(idempotency_key="durative:user_001:new-intent"),
            (first, second), config_path=CONFIG,
        )
        self.assertNotEqual(left.plan_id, new_intent.plan_id)
        self.assertNotEqual(left.input_snapshot_sha256, new_intent.input_snapshot_sha256)
        self.assertEqual(left.claim.claim_id, new_intent.claim.claim_id)  # type: ignore[union-attr]

    def test_every_input_episode_is_accounted_for_once(self) -> None:
        supports = (episode("support1"), episode("support2"))
        ignored = episode("irrelevant", subject_id="other_person")
        result = infer_durative_claim(
            request(), (*supports, ignored), config_path=CONFIG
        )
        groups = (
            result.decision.support_episode_ids
            + result.decision.counter_episode_ids
            + result.decision.ignored_episode_ids
        )
        self.assertEqual(set(groups), {item.episode_id for item in (*supports, ignored)})
        self.assertEqual(len(groups), len(set(groups)))


if __name__ == "__main__":
    unittest.main()
