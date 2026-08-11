from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import unittest
from unittest import mock

import conflicts.candidates as candidate_module
from conflicts.candidates import (
    CandidateConfig,
    CandidatePair,
    CandidateRequest,
    CandidateSignals,
    ConflictCandidateError,
    ConflictCandidateService,
    generate_candidate_pairs,
    load_candidate_config,
)
from extraction.predicate_registry import load_predicate_registry
from storage.contracts import ClaimRecord, ClaimVersionRecord
from temporal.contracts import TemporalClaim, TemporalQuery, VisibleEvidence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/conflicts/candidate_linker_v1.json"
REGISTRY_PATH = ROOT / "configs/extraction/predicate_registry_v2.json"
UTC = timezone.utc
TRANSACTION_FROM = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def _claim(
    claim_id: str,
    *,
    user_id: str = "user_001",
    subject_id: str = "person_a",
    predicate: str = "employer",
    object_json: object = "Acme",
    status: str = "candidate",
    precision: str = "day",
    valid_from: date | datetime | None = date(2026, 1, 1),
    valid_to: date | datetime | None = date(2026, 1, 1),
    transaction_from: datetime = TRANSACTION_FROM,
    transaction_to: datetime | None = None,
    source_ids: tuple[str, ...] = ("source_001",),
    quote: str = "source-backed quote",
) -> TemporalClaim:
    date_from = valid_from if type(valid_from) is date else None
    date_to = valid_to if type(valid_to) is date else None
    timestamp_from = valid_from if isinstance(valid_from, datetime) else None
    timestamp_to = valid_to if isinstance(valid_to, datetime) else None
    claim = ClaimRecord(
        claim_id=claim_id,
        user_id=user_id,
        subject_id=subject_id,
        speaker_id=subject_id,
        predicate=predicate,
        predicate_registry_version="predicate_registry_v2",
        object_json=object_json,
        polarity="positive",
        epistemic_status="asserted",
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=precision,
        extraction_confidence=0.9,
        memory_kind="durative",
        sensitivity="standard",
        extraction_version_id="extractor_v1",
    )
    version = ClaimVersionRecord(
        version_id=f"version_{claim_id}",
        user_id=user_id,
        claim_id=claim_id,
        lifecycle_status=status,
        transaction_from=transaction_from,
        transaction_to=transaction_to,
        belief_confidence=None,
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=precision,
    )
    evidence = tuple(
        VisibleEvidence(
            source_id=source_id,
            span_id=f"span_{claim_id}_{index}",
            message_id=f"message_{index}",
            speaker_id=subject_id,
            quote=quote,
        )
        for index, source_id in enumerate(source_ids)
    )
    return TemporalClaim(claim=claim, version=version, evidence=evidence)


class CandidateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_candidate_config(CONFIG_PATH, repo_root=ROOT)
        cls.registry = load_predicate_registry(REGISTRY_PATH)

    def _generate(
        self,
        claims: tuple[TemporalClaim, ...],
        incoming: tuple[str, ...] = ("claim_a",),
    ) -> tuple[CandidatePair, ...]:
        return generate_candidate_pairs(
            self.config,
            CandidateRequest("user_001", AS_OF, incoming),
            claims,
            self.registry,
        )

    def test_frozen_config_and_record_fields(self) -> None:
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "c21fe89467f64d31098f42940cb0b7a62d8a213bd93d7ac04d1d91f2280d9f7f",
        )
        self.assertEqual(self.config.linker_version, "candidate_linker_v1")
        self.assertEqual(self.config.rule_version, "candidate_rules_v1")
        self.assertEqual(self.config.predicate_registry_version, "predicate_registry_v2")
        self.assertEqual(self.config.predicate_registry_sha256, self.registry.content_sha256)
        self.assertEqual(self.config.maximum_temporal_gap_days, 90)
        self.assertEqual(self.config.lexical_jaccard_threshold, 0.5)
        self.assertEqual(
            self.config.eligible_lifecycle_statuses,
            ("candidate", "confirmed", "current", "historical", "disputed", "superseded"),
        )
        self.assertEqual(
            {field.name for field in fields(CandidateRequest)},
            {"user_id", "transaction_as_of", "incoming_claim_ids"},
        )
        self.assertEqual(
            {field.name for field in fields(CandidateSignals)},
            {
                "same_subject", "same_predicate_family", "shared_entities",
                "temporal_relation", "temporal_gap_days", "approximate_time",
                "lexical_jaccard",
            },
        )
        self.assertEqual(
            {field.name for field in fields(CandidatePair)},
            {
                "pair_id", "linker_version", "user_id", "left_claim_id",
                "right_claim_id", "signals", "source_ids",
            },
        )

    def test_config_identity_and_registry_binding_reject_drift(self) -> None:
        with self.assertRaisesRegex(ConflictCandidateError, "identity changed"):
            replace(self.config, rule_version="candidate_rules_v2")
        drifted_registry = replace(self.registry, content_sha256="0" * 64)
        with self.assertRaisesRegex(ConflictCandidateError, "registry binding changed"):
            generate_candidate_pairs(
                self.config,
                CandidateRequest("user_001", AS_OF, ("claim_a",)),
                (_claim("claim_a"),),
                drifted_registry,
            )

    def test_request_requires_user_aware_cutoff_and_unique_incoming_tuple(self) -> None:
        invalid = (
            ("", AS_OF, ("claim_a",)),
            ("user_001", datetime(2026, 1, 1), ("claim_a",)),
            ("user_001", AS_OF, ()),
            ("user_001", AS_OF, ["claim_a"]),
            ("user_001", AS_OF, ("claim_a", "claim_a")),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ConflictCandidateError):
                CandidateRequest(*arguments)

    def test_pair_id_claim_order_sources_and_dedup_are_stable(self) -> None:
        claims = (
            _claim("claim_c", source_ids=("source_c",)),
            _claim("claim_a", source_ids=("source_a", "source_shared")),
            _claim("claim_b", source_ids=("source_b",)),
        )
        pairs = self._generate(claims, ("claim_c", "claim_a"))
        self.assertEqual(
            [(pair.left_claim_id, pair.right_claim_id) for pair in pairs],
            [("claim_a", "claim_b"), ("claim_a", "claim_c"), ("claim_b", "claim_c")],
        )
        self.assertTrue(all({pair.left_claim_id, pair.right_claim_id} & {"claim_a", "claim_c"} for pair in pairs))
        self.assertEqual(pairs[0].source_ids, ("source_a", "source_b", "source_shared"))
        self.assertRegex(pairs[0].pair_id, r"^[0-9a-f]{64}$")
        self.assertEqual(pairs, self._generate(tuple(reversed(claims)), ("claim_a", "claim_c")))

    def test_same_subject_and_registry_family_qualifies(self) -> None:
        pairs = self._generate(
            (
                _claim("claim_a", predicate="employer", object_json="Acme", precision="unknown", valid_from=None, valid_to=None),
                _claim("claim_b", predicate="accepted_role", object_json="NewCo", precision="unknown", valid_from=None, valid_to=None),
            )
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].signals.same_subject)
        self.assertTrue(pairs[0].signals.same_predicate_family)
        self.assertEqual(pairs[0].signals.temporal_relation, "unknown")

    def test_entities_use_subjects_and_complete_normalized_string_leaves(self) -> None:
        entities = candidate_module._entities(
            "PersonＡ",
            {"nested": ["Résumé Team", {"organization": "ＡＣＭＥ Inc."}], "count": 3},
        )
        self.assertEqual(entities, frozenset({"persona", "résumé team", "acme inc"}))
        self.assertNotIn("organization", entities)
        self.assertNotIn("acme", entities)
        pairs = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json={"org": "ＡＣＭＥ Inc."}),
                _claim("claim_b", subject_id="person_b", predicate="career_goal", object_json=["acme inc."]),
            )
        )
        self.assertEqual(pairs[0].signals.shared_entities, ("acme inc",))

    def test_inclusive_overlap_exact_ninety_day_gap_and_approximate_signal(self) -> None:
        overlap = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json="shared", valid_from=date(2026, 1, 1), valid_to=date(2026, 1, 10)),
                _claim("claim_b", subject_id="person_b", predicate="career_goal", object_json="shared", valid_from=date(2026, 1, 10), valid_to=date(2026, 1, 12)),
            )
        )
        self.assertEqual(overlap[0].signals.temporal_relation, "overlap")
        exactly_ninety = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json="shared", precision="approximate", valid_from=date(2026, 1, 1), valid_to=date(2026, 1, 1)),
                _claim("claim_b", subject_id="person_b", predicate="career_goal", object_json="shared", valid_from=date(2026, 4, 1), valid_to=date(2026, 4, 1)),
            )
        )
        self.assertEqual(exactly_ninety[0].signals.temporal_relation, "within_gap")
        self.assertEqual(exactly_ninety[0].signals.temporal_gap_days, 90.0)
        self.assertTrue(exactly_ninety[0].signals.approximate_time)
        beyond = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json="shared", valid_from=date(2026, 1, 1), valid_to=date(2026, 1, 1)),
                _claim("claim_b", subject_id="person_b", predicate="career_goal", object_json="shared", valid_from=date(2026, 4, 2), valid_to=date(2026, 4, 2)),
            )
        )
        self.assertEqual(beyond, ())

    def test_unknown_open_and_mixed_time_do_not_activate_temporal_rule(self) -> None:
        unknown = _claim("claim_a", subject_id="person_a", predicate="employer", object_json="shared", precision="unknown", valid_from=None, valid_to=None)
        open_ended = _claim("claim_b", subject_id="person_b", predicate="career_goal", object_json="shared", valid_from=date(2026, 1, 1), valid_to=None)
        timestamp = _claim(
            "claim_c", subject_id="person_c", predicate="career_goal", object_json="shared",
            precision="timestamp", valid_from=datetime(2026, 1, 1, tzinfo=UTC),
            valid_to=datetime(2026, 1, 2, tzinfo=UTC),
        )
        day = _claim("claim_d", subject_id="person_d", predicate="employer", object_json="shared")
        self.assertEqual(self._generate((unknown, open_ended)), ())
        self.assertEqual(self._generate((day, timestamp), ("claim_d",)), ())
        relation, gap = candidate_module._temporal_signal(unknown.version, open_ended.version, 90)
        self.assertEqual((relation, gap), ("unknown", None))

    def test_lexical_threshold_is_inclusive_and_quotes_are_never_features(self) -> None:
        at_threshold = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json="alpha", precision="unknown", valid_from=None, valid_to=None, quote="irrelevant one"),
                _claim("claim_b", subject_id="person_b", predicate="accepted_role", object_json="alpha", precision="unknown", valid_from=None, valid_to=None, quote="irrelevant two"),
            )
        )
        self.assertEqual(at_threshold[0].signals.lexical_jaccard, 0.5)
        below = self._generate(
            (
                _claim("claim_a", subject_id="person_a", predicate="employer", object_json="alpha one", precision="unknown", valid_from=None, valid_to=None, quote="same secret quote"),
                _claim("claim_b", subject_id="person_b", predicate="accepted_role", object_json="alpha two", precision="unknown", valid_from=None, valid_to=None, quote="same secret quote"),
            )
        )
        self.assertEqual(below, ())

    def test_user_status_transaction_and_support_filters_run_before_features(self) -> None:
        valid_incoming = _claim("claim_a")
        valid_existing = _claim("claim_b")
        filtered = (
            _claim("claim_excluded", status="excluded"),
            _claim("claim_future", transaction_from=AS_OF + timedelta(days=1)),
            _claim("claim_unsupported", source_ids=()),
        )
        with mock.patch.object(
            candidate_module,
            "_candidate_signals",
            wraps=candidate_module._candidate_signals,
        ) as signals:
            pairs = self._generate((valid_incoming, valid_existing, *filtered))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(signals.call_count, 1)
        left, right = signals.call_args.args[1:3]
        self.assertEqual({left.claim.claim_id, right.claim.claim_id}, {"claim_a", "claim_b"})
        with mock.patch.object(candidate_module, "_candidate_signals") as blocked:
            with self.assertRaisesRegex(ConflictCandidateError, "not visible and supported"):
                self._generate((valid_existing, filtered[-1]), ("claim_unsupported",))
        blocked.assert_not_called()

    def test_cross_user_input_is_rejected_before_feature_work(self) -> None:
        for cross_user in (
            _claim("claim_cross", user_id="user_002"),
            replace(
                _claim("claim_cross_version"),
                version=replace(_claim("claim_cross_version").version, user_id="user_002"),
            ),
        ):
            with self.subTest(claim_id=cross_user.claim.claim_id), mock.patch.object(
                candidate_module, "_candidate_signals"
            ) as signals:
                with self.assertRaisesRegex(ConflictCandidateError, "cross-user"):
                    self._generate((_claim("claim_a"), cross_user))
            signals.assert_not_called()

    def test_unknown_predicate_is_rejected_before_pairing(self) -> None:
        with mock.patch.object(candidate_module, "_candidate_signals") as signals:
            with self.assertRaisesRegex(ConflictCandidateError, "predicate is unknown"):
                self._generate((_claim("claim_a", predicate="unknown_predicate"),))
        signals.assert_not_called()

    def test_service_uses_exact_user_scoped_temporal_query_and_deletion_visibility(self) -> None:
        incoming = _claim("claim_a")
        existing = _claim("claim_b")
        temporal = mock.Mock()
        temporal.query.return_value = (existing, incoming)
        with mock.patch.object(candidate_module, "TemporalService", return_value=temporal):
            service = ConflictCandidateService(object(), repo_root=ROOT)
            request = CandidateRequest("user_001", AS_OF, ("claim_a",))
            pairs = service.generate(request)
            temporal.query.assert_called_once_with(
                TemporalQuery(
                    user_id="user_001",
                    transaction_as_of=AS_OF,
                    statuses=frozenset(self.config.eligible_lifecycle_statuses),
                )
            )
            self.assertEqual(len(pairs), 1)
            temporal.query.reset_mock()
            temporal.query.return_value = (incoming,)
            self.assertEqual(service.generate(request), ())


if __name__ == "__main__":
    unittest.main()
