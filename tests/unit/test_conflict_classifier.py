from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import unittest

import conflicts.candidates as candidate_module
from conflicts.candidates import CandidatePair, CandidateSignals
from conflicts.classifier import (
    CLASSIFIER_VERSION,
    CONFLICT_LABELS,
    EMITTED_RELATIONS,
    CheckedRelation,
    ClassificationRequest,
    ConflictClassifier,
    ConflictClassifierError,
    ConflictDecision,
    classify_conflict,
    load_classifier_config,
)
from extraction.predicate_registry import load_predicate_registry
from storage.contracts import ClaimRecord, ClaimVersionRecord, StorageValidationError
from temporal.contracts import TemporalClaim, VisibleEvidence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/conflicts/relation_classifier_v1.json"
REGISTRY_PATH = ROOT / "configs/extraction/predicate_registry_v2.json"
UTC = timezone.utc
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
TX_FROM = datetime(2026, 1, 1, tzinfo=UTC)


def _claim(
    claim_id: str,
    *,
    user_id: str = "user_001",
    subject_id: str = "user_001",
    speaker_id: str = "user_001",
    predicate: str = "lives_in",
    object_json: object = "Delhi",
    polarity: str = "positive",
    epistemic_status: str = "asserted",
    status: str = "candidate",
    precision: str = "day",
    valid_from: date | datetime | None = date(2026, 1, 1),
    valid_to: date | datetime | None = date(2026, 12, 31),
    transaction_from: datetime = TX_FROM,
    transaction_to: datetime | None = None,
    source_id: str | None = None,
    evidence_speaker_id: str | None = None,
    quote: str = "source-backed assertion",
) -> TemporalClaim:
    source_id = source_id or f"source_{claim_id}"
    date_from = valid_from if type(valid_from) is date else None
    date_to = valid_to if type(valid_to) is date else None
    timestamp_from = valid_from if isinstance(valid_from, datetime) else None
    timestamp_to = valid_to if isinstance(valid_to, datetime) else None
    claim = ClaimRecord(
        claim_id=claim_id,
        user_id=user_id,
        subject_id=subject_id,
        speaker_id=speaker_id,
        predicate=predicate,
        predicate_registry_version="predicate_registry_v2",
        object_json=object_json,
        polarity=polarity,
        epistemic_status=epistemic_status,
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=precision,
        extraction_confidence=1,
        memory_kind="durative",
        sensitivity="standard",
        extraction_version_id="classifier_test_v1",
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
    evidence = (
        VisibleEvidence(
            source_id=source_id,
            span_id=f"span_{claim_id}",
            message_id=f"message_{claim_id}",
            speaker_id=evidence_speaker_id or speaker_id,
            quote=quote,
        ),
    )
    return TemporalClaim(claim, version, evidence)


def _request(
    *,
    left: TemporalClaim | None = None,
    right: TemporalClaim | None = None,
    target: str | None = None,
    user_id: str = "user_001",
    as_of: datetime = AS_OF,
    ingested_at: datetime | None = None,
    lexical_jaccard: float = 0.0,
) -> ClassificationRequest:
    left = left or _claim("claim_a")
    right = right or _claim("claim_b", object_json="Mumbai")
    source_ids = tuple(
        sorted(
            {
                evidence.source_id
                for item in (left, right)
                for evidence in item.evidence
            }
        )
    )
    pair = CandidatePair(
        pair_id=candidate_module._pair_id(
            "candidate_linker_v1", user_id, left.claim.claim_id, right.claim.claim_id
        ),
        linker_version="candidate_linker_v1",
        user_id=user_id,
        left_claim_id=left.claim.claim_id,
        right_claim_id=right.claim.claim_id,
        signals=CandidateSignals(
            same_subject=left.claim.subject_id == right.claim.subject_id,
            same_predicate_family=True,
            shared_entities=(),
            temporal_relation="unknown",
            temporal_gap_days=None,
            approximate_time=False,
            lexical_jaccard=lexical_jaccard,
        ),
        source_ids=source_ids,
    )
    observed = ingested_at or as_of - timedelta(days=1)
    return ClassificationRequest(
        classifier_version=CLASSIFIER_VERSION,
        user_id=user_id,
        transaction_as_of=as_of,
        pair=pair,
        left=left,
        right=right,
        source_ingested_at=tuple((source_id, observed) for source_id in source_ids),
        explicit_target_claim_id=target,
    )


class ConflictClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_classifier_config(CONFIG_PATH, repo_root=ROOT)
        cls.registry = load_predicate_registry(REGISTRY_PATH)
        cls.classifier = ConflictClassifier(repo_root=ROOT)

    def classify(self, request: ClassificationRequest) -> ConflictDecision:
        return classify_conflict(self.config, request, self.registry)

    def test_frozen_config_vocabularies_records_and_hash(self) -> None:
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "fab3171a55ad21e25e303e89f44e57b7d0b926eb8f82283ce1dcb910d8960659",
        )
        self.assertEqual(self.config.labels, CONFLICT_LABELS)
        self.assertEqual(self.config.emitted_relation_types, EMITTED_RELATIONS)
        self.assertIn("supersedes", self.config.accepted_relation_vocabulary)
        self.assertNotIn("supersedes", self.config.emitted_relation_types)
        self.assertEqual(
            {field.name for field in fields(ClassificationRequest)},
            {
                "classifier_version", "user_id", "transaction_as_of", "pair",
                "left", "right", "source_ingested_at", "explicit_target_claim_id",
            },
        )
        self.assertEqual(
            {field.name for field in fields(CheckedRelation)},
            {
                "relation_id", "classifier_version", "user_id", "source_claim_id",
                "target_claim_id", "relation_type", "confidence",
                "input_snapshot_sha256",
            },
        )
        self.assertEqual(
            {field.name for field in fields(ConflictDecision)},
            {
                "decision_id", "classifier_version", "user_id", "pair_id", "label",
                "confidence", "input_snapshot_sha256", "relations",
            },
        )

    def test_all_eight_labels_and_relation_types(self) -> None:
        overlap_left = _claim("claim_a", object_json="Delhi")
        overlap_right = _claim("claim_b", object_json="Mumbai")
        hard = self.classify(_request(left=overlap_left, right=overlap_right))

        different_speaker = _claim(
            "claim_b", object_json="Mumbai", speaker_id="colleague"
        )
        disagreement = self.classify(
            _request(left=overlap_left, right=different_speaker)
        )

        earlier = _claim(
            "claim_a", object_json="Delhi",
            valid_from=date(2026, 1, 1), valid_to=date(2026, 2, 28),
        )
        later = _claim(
            "claim_b", object_json="Mumbai",
            valid_from=date(2026, 3, 1), valid_to=date(2026, 4, 1),
        )
        temporal = self.classify(_request(left=earlier, right=later))

        correction = self.classify(
            _request(
                left=overlap_left,
                right=_claim(
                    "claim_b", object_json="Mumbai", epistemic_status="corrected"
                ),
                target="claim_a",
            )
        )
        retraction = self.classify(
            _request(
                left=earlier,
                right=_claim(
                    "claim_b", object_json="Mumbai", epistemic_status="denied",
                    valid_from=date(2026, 3, 1), valid_to=date(2026, 4, 1),
                ),
                target="claim_a",
            )
        )
        broad = _claim(
            "claim_a", predicate="requested_leave_dates",
            object_json=["2026-07-01"],
        )
        specific = _claim(
            "claim_b", predicate="requested_leave_dates",
            object_json=["2026-07-01", "2026-07-02"],
        )
        refinement = self.classify(_request(left=broad, right=specific))

        unresolved = self.classify(
            _request(
                left=_claim(
                    "claim_a", precision="unknown", valid_from=None, valid_to=None
                ),
                right=_claim(
                    "claim_b", object_json="Mumbai", precision="unknown",
                    valid_from=None, valid_to=None,
                ),
            )
        )
        unrelated = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="work_preference", object_json="remote"
                ),
                right=_claim(
                    "claim_b", predicate="work_preference", object_json="quiet office"
                ),
            )
        )
        decisions = (
            hard, temporal, correction, refinement, disagreement, retraction,
            unresolved, unrelated,
        )
        self.assertEqual({item.label for item in decisions}, set(CONFLICT_LABELS))
        self.assertEqual(hard.relations[0].relation_type, "contradicts")
        self.assertEqual(disagreement.relations[0].relation_type, "contradicts")
        self.assertEqual(temporal.relations[0].relation_type, "same_topic_as")
        self.assertEqual(correction.relations[0].relation_type, "corrects")
        self.assertEqual(retraction.relations[0].relation_type, "corrects")
        self.assertEqual(refinement.relations[0].relation_type, "refines")
        self.assertEqual(unresolved.relations, ())
        self.assertEqual(unrelated.relations, ())
        self.assertTrue(all(item.confidence == 1 for item in decisions))

    def test_targeted_retraction_and_correction_have_precedence_and_direction(self) -> None:
        old = _claim(
            "claim_a", object_json="Delhi",
            valid_from=date(2026, 1, 1), valid_to=date(2026, 2, 1),
        )
        denied = _claim(
            "claim_b", object_json="Mumbai", epistemic_status="denied",
            valid_from=date(2026, 4, 1), valid_to=date(2026, 5, 1),
        )
        retraction = self.classify(
            _request(left=old, right=denied, target="claim_a")
        )
        self.assertEqual(retraction.label, "retraction")
        self.assertEqual(
            (
                retraction.relations[0].source_claim_id,
                retraction.relations[0].target_claim_id,
            ),
            ("claim_b", "claim_a"),
        )
        corrected = replace(
            denied,
            claim=replace(denied.claim, epistemic_status="corrected"),
        )
        correction = self.classify(
            _request(left=old, right=corrected, target="claim_a")
        )
        self.assertEqual(correction.label, "explicit_correction")
        self.assertEqual(
            (
                correction.relations[0].source_claim_id,
                correction.relations[0].target_claim_id,
            ),
            ("claim_b", "claim_a"),
        )
        unchanged = replace(
            corrected,
            claim=replace(corrected.claim, object_json="Delhi"),
        )
        self.assertEqual(
            self.classify(_request(left=old, right=unchanged, target="claim_a")).label,
            "unresolved_ambiguity",
        )

    def test_symmetric_relations_are_once_and_canonical_refinement_is_directed(self) -> None:
        contradiction = self.classify(_request())
        self.assertEqual(len(contradiction.relations), 1)
        self.assertEqual(
            (
                contradiction.relations[0].source_claim_id,
                contradiction.relations[0].target_claim_id,
            ),
            ("claim_a", "claim_b"),
        )
        refinement = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="requested_leave_dates",
                    object_json=["2026-07-01"],
                ),
                right=_claim(
                    "claim_b", predicate="requested_leave_dates",
                    object_json=["2026-07-01", "2026-07-02"],
                ),
            )
        )
        self.assertEqual(
            (
                refinement.relations[0].source_claim_id,
                refinement.relations[0].target_claim_id,
            ),
            ("claim_b", "claim_a"),
        )
        with self.assertRaisesRegex(ConflictClassifierError, "not emitted"):
            replace(refinement.relations[0], relation_type="supersedes")

    def test_registry_single_multi_repeatable_rules_are_distinct(self) -> None:
        single = self.classify(_request())
        multi = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="work_preference", object_json="remote"
                ),
                right=_claim(
                    "claim_b", predicate="work_preference", object_json="quiet office"
                ),
                lexical_jaccard=1,
            )
        )
        repeatable = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="project_review_date",
                    object_json="2026-07-01",
                ),
                right=_claim(
                    "claim_b", predicate="project_review_date",
                    object_json="2026-07-02",
                ),
                lexical_jaccard=1,
            )
        )
        self.assertEqual(single.label, "hard_contradiction")
        self.assertEqual(multi.label, "unrelated")
        self.assertEqual(repeatable.label, "unrelated")

    def test_inclusive_endpoint_overlaps_but_nonoverlap_is_temporal_change(self) -> None:
        endpoint = self.classify(
            _request(
                left=_claim(
                    "claim_a", valid_from=date(2026, 1, 1),
                    valid_to=date(2026, 1, 31),
                ),
                right=_claim(
                    "claim_b", object_json="Mumbai",
                    valid_from=date(2026, 1, 31), valid_to=date(2026, 2, 15),
                ),
            )
        )
        separate = self.classify(
            _request(
                left=_claim(
                    "claim_a", valid_from=date(2026, 1, 1),
                    valid_to=date(2026, 1, 31),
                ),
                right=_claim(
                    "claim_b", object_json="Mumbai",
                    valid_from=date(2026, 2, 1), valid_to=date(2026, 2, 15),
                ),
            )
        )
        self.assertEqual(endpoint.label, "hard_contradiction")
        self.assertEqual(separate.label, "temporal_change")

    def test_unknown_open_mixed_and_approximate_time_are_unresolved(self) -> None:
        cases = (
            (
                _claim("claim_a", precision="unknown", valid_from=None, valid_to=None),
                _claim(
                    "claim_b", object_json="Mumbai", precision="unknown",
                    valid_from=None, valid_to=None,
                ),
            ),
            (
                _claim("claim_a", valid_to=None),
                _claim("claim_b", object_json="Mumbai", valid_to=None),
            ),
            (
                _claim("claim_a"),
                _claim(
                    "claim_b", object_json="Mumbai", precision="timestamp",
                    valid_from=datetime(2026, 1, 1, tzinfo=UTC),
                    valid_to=datetime(2026, 2, 1, tzinfo=UTC),
                ),
            ),
            (
                _claim("claim_a", precision="approximate"),
                _claim("claim_b", object_json="Mumbai", precision="approximate"),
            ),
        )
        for left, right in cases:
            with self.subTest(precision=(left.version.time_precision, right.version.time_precision)):
                self.assertEqual(
                    self.classify(_request(left=left, right=right)).label,
                    "unresolved_ambiguity",
                )

    def test_missing_target_and_prose_only_targeting_are_unresolved(self) -> None:
        structured_without_target = _claim(
            "claim_b", object_json="Mumbai", epistemic_status="corrected"
        )
        self.assertEqual(
            self.classify(_request(right=structured_without_target)).label,
            "unresolved_ambiguity",
        )
        prose_only = _claim(
            "claim_b", object_json="Mumbai",
            quote="Actually, I need to correct that earlier statement.",
        )
        self.assertEqual(
            self.classify(_request(right=prose_only)).label,
            "unresolved_ambiguity",
        )

    def test_snapshot_and_ids_are_json_safe_deterministic_and_value_bound(self) -> None:
        left_object = {"marketing_percent": 40, "product_percent": 60}
        reordered = {"product_percent": 60, "marketing_percent": 40}
        first = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="work_allocation", object_json=left_object
                ),
                right=_claim(
                    "claim_b", predicate="work_allocation", object_json=left_object
                ),
            )
        )
        second = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="work_allocation", object_json=reordered
                ),
                right=_claim(
                    "claim_b", predicate="work_allocation", object_json=reordered
                ),
            )
        )
        self.assertEqual(first, second)
        self.assertRegex(first.decision_id, r"^[0-9a-f]{64}$")
        self.assertRegex(first.input_snapshot_sha256, r"^[0-9a-f]{64}$")
        changed = self.classify(
            _request(
                left=_claim(
                    "claim_a", predicate="work_allocation",
                    object_json={"marketing_percent": 30, "product_percent": 70},
                ),
                right=_claim(
                    "claim_b", predicate="work_allocation", object_json=left_object
                ),
            )
        )
        self.assertNotEqual(first.input_snapshot_sha256, changed.input_snapshot_sha256)
        with self.assertRaises(StorageValidationError):
            _claim("claim_a", object_json=float("nan"))

    def test_request_rejects_user_future_excluded_transaction_and_provenance_drift(self) -> None:
        cross_user = _claim("claim_a", user_id="user_002")
        with self.assertRaisesRegex(ConflictClassifierError, "ownership"):
            self.classify(_request(left=cross_user))

        future = _request(ingested_at=AS_OF + timedelta(seconds=1))
        with self.assertRaisesRegex(ConflictClassifierError, "future evidence"):
            self.classify(future)

        excluded = _claim("claim_a", status="excluded")
        with self.assertRaisesRegex(ConflictClassifierError, "not classifier-visible"):
            self.classify(_request(left=excluded))

        closed = _claim("claim_a", transaction_to=AS_OF)
        with self.assertRaisesRegex(ConflictClassifierError, "not visible as of cutoff"):
            self.classify(_request(left=closed))

        wrong_speaker = _claim(
            "claim_a", evidence_speaker_id="different_speaker"
        )
        with self.assertRaisesRegex(ConflictClassifierError, "evidence speaker"):
            self.classify(_request(left=wrong_speaker))

        request = _request()
        with self.assertRaisesRegex(ConflictClassifierError, "canonical pair order"):
            replace(request, left=request.right, right=request.left)

        changed_linker = replace(
            request.pair,
            pair_id=candidate_module._pair_id(
                "candidate_linker_v2",
                request.user_id,
                request.pair.left_claim_id,
                request.pair.right_claim_id,
            ),
            linker_version="candidate_linker_v2",
        )
        with self.assertRaisesRegex(ConflictClassifierError, "linker version"):
            self.classify(replace(request, pair=changed_linker))

    def test_classifier_does_not_mutate_claim_or_lifecycle_snapshots(self) -> None:
        request = _request()
        before = asdict(request)
        decision = self.classifier.classify(request)
        self.assertEqual(asdict(request), before)
        self.assertEqual(request.left.version.lifecycle_status, "candidate")
        self.assertEqual(request.right.version.lifecycle_status, "candidate")
        self.assertNotIn("supersedes", {item.relation_type for item in decision.relations})


if __name__ == "__main__":
    unittest.main()
