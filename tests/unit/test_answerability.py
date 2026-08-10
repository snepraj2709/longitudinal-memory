from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import json
import math
import unittest

from answering.contracts import canonical_json_bytes as package_bytes
from retrieval.query_contracts import RequestedValidTime

from abstention.contracts import (
    ANSWERABILITY_VERSION,
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    POLICY_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    AnswerabilityError,
    AnswerabilityRequest,
    AnswerabilityRequirement,
    DecisionConfidence,
    answerability_decision_from_mapping,
    canonical_json_bytes,
    stable_sha256,
)
from abstention.policy import AnswerabilityPolicyError, decide_answerability


UTC = timezone.utc
NOW = datetime(2026, 1, 10, tzinfo=UTC)
SHA = "a" * 64


@dataclass(frozen=True)
class Anchor:
    index_record_id: str
    retrieval_rank: int


@dataclass(frozen=True)
class Relation:
    relation_id: str
    relation_type: str
    direction: str
    source_claim_id: str
    target_claim_id: str


@dataclass(frozen=True)
class Claim:
    user_id: str
    claim_id: str
    claim_version_id: str
    subject_id: str
    speaker_id: str
    predicate: str
    object_json: object
    polarity: str
    epistemic_status: str
    memory_kind: str | None
    lifecycle_status: str
    valid_from_date: date | None
    valid_from_timestamp: datetime | None
    valid_to_date: date | None
    valid_to_timestamp: datetime | None
    transaction_from: datetime
    transaction_to: datetime | None
    sensitivity: str | None
    retrieval_anchors: tuple[Anchor, ...]
    evidence_ids: tuple[str, ...]
    checked_relations: tuple[Relation, ...]


@dataclass(frozen=True)
class Span:
    evidence_id: str
    claim_id: str
    claim_version_id: str
    span_id: str
    speaker_id: str


@dataclass(frozen=True)
class Source:
    user_id: str
    source_id: str
    source_type: str
    session_id: str | None
    produced_at: datetime
    ingested_at: datetime
    evidence_spans: tuple[Span, ...]


@dataclass(frozen=True)
class Rejection:
    rejection_id: str
    index_record_id: str
    retrieval_rank: int | None
    claim_id: str | None
    claim_version_id: str | None
    stage: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    query_id: str


@dataclass(frozen=True)
class Plan:
    plan_id: str


@dataclass(frozen=True)
class Coverage:
    complete: bool


@dataclass(frozen=True)
class Package:
    package_id: str
    user_id: str
    query: Query
    baseline_id: str
    execution_id: str
    plan: Plan
    snapshot_run_id: str
    index_version: str
    as_of: datetime
    requested_valid_time: RequestedValidTime | None
    query_type: str
    current_claims: tuple[Claim, ...]
    historical_claims: tuple[Claim, ...]
    conflicting_claims: tuple[Claim, ...]
    relevant_sources: tuple[Source, ...]
    rejected_evidence: tuple[object, ...]
    structural_blockers: tuple[str, ...]
    answer_allowed: bool
    evidence_coverage: Coverage


def requirement(
    *,
    kind="fact",
    subjects=("invented_user",),
    speakers=(),
    requested=None,
    authority="exact_supported",
    predicate="preference",
    object_json={"value": "tea"},
    conflict=False,
    partial=False,
    subparts=(),
):
    fields = {
        "information_kind": kind,
        "target_subject_ids": tuple(subjects),
        "permitted_speaker_ids": tuple(speakers),
        "requested_valid_time": requested,
        "required_authority_class": authority,
        "required_predicate": predicate,
        "required_object_json": object_json,
        "required_polarity": "positive",
        "complete_subpart_coverage": True,
        "conflict_reporting_requested": conflict,
        "partial_response_requested": partial,
        "subrequirement_ids": tuple(subparts),
    }
    return AnswerabilityRequirement(requirement_id=stable_sha256(fields), **fields)


def claim(
    suffix="1",
    *,
    user="invented_user",
    subject="invented_user",
    speaker="invented_user",
    predicate="preference",
    object_json={"value": "tea"},
    memory_kind="episodic",
    lifecycle="current",
    epistemic="asserted",
    valid_from=date(2026, 1, 1),
    valid_to=date(2026, 1, 31),
    transaction_to=None,
    sensitivity="standard",
    relations=(),
):
    claim_id = f"invented_claim_{suffix}"
    version_id = f"invented_version_{suffix}"
    evidence_id = hashlib.sha256(f"evidence-{suffix}".encode()).hexdigest()
    return Claim(
        user, claim_id, version_id, subject, speaker, predicate, object_json, "positive",
        epistemic, memory_kind, lifecycle, valid_from, None, valid_to, None,
        datetime(2025, 1, 1, tzinfo=UTC), transaction_to, sensitivity,
        (Anchor(f"invented_record_{suffix}", 1),), (evidence_id,), tuple(relations),
    )


def package(claims, *, category="current", query_type="fact", requested=None, answer_allowed=True):
    current = tuple(claims) if category == "current" else ()
    historical = tuple(claims) if category == "historical" else ()
    conflicting = tuple(claims) if category == "conflicting" else ()
    sources = []
    for item in claims:
        span = Span(
            item.evidence_ids[0], item.claim_id, item.claim_version_id,
            f"invented_span_{item.claim_id}", item.speaker_id,
        )
        sources.append(Source(
            item.user_id,
            f"invented_source_{item.claim_id}",
            "conversation",
            f"invented_session_{item.claim_id}",
            datetime(2026, 1, 1, tzinfo=UTC),
            NOW,
            (span,),
        ))
    return Package(
        SHA, "invented_user", Query("invented_query"), "B4", "b" * 64,
        Plan("c" * 64), "invented_snapshot", "retrieval_index_v1", NOW, requested,
        query_type, current, historical, conflicting, tuple(sources), (),
        () if answer_allowed else ("no_promoted_claims",), answer_allowed, Coverage(True),
    )


def request(pkg, requirements, *, query_type=None, subjects=("invented_user",), speakers=()):
    fields = {
        "answerability_version": ANSWERABILITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_sha256": "d" * 64,
        "policy_version": POLICY_VERSION,
        "policy_sha256": "e" * 64,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": INPUT_RELEASE_VERSION,
        "input_release_manifest_sha256": "f" * 64,
        "package_id": pkg.package_id,
        "package_sha256": hashlib.sha256(package_bytes(pkg)).hexdigest(),
        "user_id": pkg.user_id,
        "query_id": pkg.query.query_id,
        "baseline_id": pkg.baseline_id,
        "execution_id": pkg.execution_id,
        "plan_id": pkg.plan.plan_id,
        "snapshot_run_id": pkg.snapshot_run_id,
        "index_version": pkg.index_version,
        "as_of": pkg.as_of,
        "requested_valid_time": pkg.requested_valid_time,
        "query_type": query_type or pkg.query_type,
        "subject_scope": tuple(subjects),
        "speaker_scope": tuple(speakers),
        "sensitivity_scope": "standard",
        "requirements": tuple(sorted(requirements, key=lambda item: item.requirement_id)),
    }
    return AnswerabilityRequest(request_id=stable_sha256(fields), **fields)


class AnswerabilityContractTests(unittest.TestCase):
    def test_confidence_is_mandatorily_uncalibrated(self):
        self.assertEqual(DecisionConfidence(None, "not_calibrated", "step_9_2_not_run").value, None)
        with self.assertRaises(AnswerabilityError):
            DecisionConfidence(0.5, "not_calibrated", "step_9_2_not_run")

    def test_requirement_rejects_unsafe_json_and_nonfinite_value(self):
        with self.assertRaises(AnswerabilityError):
            AnswerabilityRequirement(
                "0" * 64, "fact", ("invented_user",), (), None, "exact_supported",
                "preference", {"value": math.nan}, "positive", True, False, False, (),
            )

    def test_requirement_id_and_sorted_scope_are_strict(self):
        with self.assertRaises(AnswerabilityError):
            requirement(subjects=("z", "a"))
        with self.assertRaises(AnswerabilityError):
            replace(requirement(), requirement_id="0" * 64)

    def test_decision_parser_rejects_unknown_nested_fields(self):
        pkg = package((claim(),))
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        value = json.loads(canonical_json_bytes(decision))
        value["accepted_evidence"][0]["unexpected"] = True
        with self.assertRaisesRegex(AnswerabilityError, "accepted evidence fields"):
            answerability_decision_from_mapping(value)

    def test_decision_rejects_incoherent_assessment_and_status(self):
        pkg = package((claim(),))
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        with self.assertRaisesRegex(AnswerabilityError, "decision and assessment reasons"):
            replace(decision, assessment=replace(
                decision.assessment,
                applicable_reasons=("requested_information_absent",),
            ))
        with self.assertRaisesRegex(AnswerabilityError, "disputed decision"):
            replace(decision, permitted_answer_statuses=("disputed",))


class AnswerabilityPolicyTests(unittest.TestCase):
    def test_direct_subject_fact_is_answerable(self):
        pkg = package((claim(),))
        decision = decide_answerability(request(pkg, (requirement(authority="direct_subject"),)), pkg)
        self.assertEqual((decision.decision, decision.permitted_answer_statuses), ("answerable", ("answered",)))
        self.assertEqual(len(decision.accepted_evidence), 1)

    def test_inclusive_historical_endpoint_and_half_open_transaction(self):
        point = RequestedValidTime("point", date(2026, 1, 31), None, None, None, None, None)
        item = claim(lifecycle="historical")
        pkg = package((item,), category="historical", query_type="historical_state", requested=point)
        req = request(pkg, (requirement(kind="historical_state", requested=point),), query_type="historical_state")
        self.assertEqual(decide_answerability(req, pkg).decision, "answerable")
        closed = replace(item, transaction_to=NOW)
        stale_pkg = package((closed,), category="historical", query_type="historical_state", requested=point)
        stale = decide_answerability(request(stale_pkg, (requirement(kind="historical_state", requested=point),), query_type="historical_state"), stale_pkg)
        self.assertEqual(stale.primary_reason, "stale_evidence")

    def test_explicit_conflict_report_permits_only_disputed(self):
        left, right = claim("left"), claim("right", object_json={"value": "coffee"})
        pkg = package((left, right), category="conflicting", query_type="evidence_request")
        req = request(pkg, (requirement(kind="evidence_request", object_json=None, conflict=True),), query_type="evidence_request")
        decision = decide_answerability(req, pkg)
        self.assertEqual((decision.decision, decision.permitted_answer_statuses), ("answerable", ("disputed",)))

    def test_ordinary_request_with_conflict_abstains(self):
        pkg = package((claim("left"), claim("right")), category="conflicting")
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        self.assertEqual(decision.primary_reason, "unresolved_conflict")

    def test_wrong_person_precedes_information_and_authority_reasons(self):
        pkg = package((claim(subject="invented_friend"),))
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        self.assertEqual(decision.primary_reason, "wrong_person_risk")

    def test_other_speaker_and_uncertain_source_are_insufficient(self):
        other = package((claim(speaker="invented_friend"),))
        direct = decide_answerability(request(other, (requirement(authority="direct_subject"),)), other)
        self.assertEqual(direct.primary_reason, "insufficient_source_authority")
        uncertain = package((claim(epistemic="uncertain"),))
        self.assertEqual(
            decide_answerability(request(uncertain, (requirement(),)), uncertain).primary_reason,
            "insufficient_source_authority",
        )

    def test_participant_and_official_authority_are_scoped_to_source_kind(self):
        participant_package = package((claim(speaker="invented_friend"),))
        participant_request = request(
            participant_package,
            (requirement(authority="firsthand_participant"),),
        )
        self.assertEqual(
            decide_answerability(participant_request, participant_package).decision,
            "answerable",
        )
        indirect = replace(
            participant_package,
            relevant_sources=tuple(
                replace(source, source_type="email")
                for source in participant_package.relevant_sources
            ),
        )
        indirect_request = request(
            indirect,
            (requirement(authority="firsthand_participant"),),
        )
        self.assertEqual(
            decide_answerability(indirect_request, indirect).primary_reason,
            "insufficient_source_authority",
        )
        wrong_speaker = replace(
            participant_package,
            relevant_sources=tuple(
                replace(
                    source,
                    evidence_spans=tuple(
                        replace(span, speaker_id="invented_third_party")
                        for span in source.evidence_spans
                    ),
                )
                for source in participant_package.relevant_sources
            ),
        )
        wrong_speaker_request = request(
            wrong_speaker,
            (requirement(authority="firsthand_participant"),),
        )
        self.assertEqual(
            decide_answerability(wrong_speaker_request, wrong_speaker).primary_reason,
            "insufficient_source_authority",
        )

        official = replace(
            package((claim(),)),
            relevant_sources=tuple(
                replace(source, source_type="calendar")
                for source in package((claim(),)).relevant_sources
            ),
        )
        official_request = request(
            official,
            (requirement(authority="official_record"),),
        )
        self.assertEqual(decide_answerability(official_request, official).decision, "answerable")
        ordinary = package((claim(),))
        ordinary_request = request(
            ordinary,
            (requirement(authority="official_record"),),
        )
        self.assertEqual(
            decide_answerability(ordinary_request, ordinary).primary_reason,
            "insufficient_source_authority",
        )

    def test_accepted_durative_authority_rejects_an_episodic_claim(self):
        durable = package((claim(memory_kind="durative"),))
        durable_request = request(
            durable,
            (requirement(authority="accepted_durative"),),
        )
        self.assertEqual(decide_answerability(durable_request, durable).decision, "answerable")
        episodic = package((claim(),))
        episodic_request = request(
            episodic,
            (requirement(authority="accepted_durative"),),
        )
        self.assertEqual(
            decide_answerability(episodic_request, episodic).primary_reason,
            "insufficient_source_authority",
        )

    def test_explicit_speaker_scope_rejects_another_speaker(self):
        pkg = package((claim(speaker="invented_friend"),))
        req = request(
            pkg,
            (requirement(speakers=("invented_user",)),),
            speakers=("invented_user",),
        )
        self.assertEqual(decide_answerability(req, pkg).primary_reason, "insufficient_speaker_authority")

    def test_historical_claim_cannot_answer_current_state(self):
        point = RequestedValidTime("point", date(2026, 1, 10), None, None, None, None, None)
        pkg = package((claim(lifecycle="historical"),), category="historical", query_type="current_state", requested=point)
        req = request(pkg, (requirement(kind="current_state", requested=point),), query_type="current_state")
        self.assertEqual(decide_answerability(req, pkg).primary_reason, "stale_evidence")

    def test_unknown_valid_time_cannot_prove_current(self):
        point = RequestedValidTime("point", date(2026, 1, 10), None, None, None, None, None)
        pkg = package((claim(valid_from=None, valid_to=None),), query_type="current_state", requested=point)
        req = request(pkg, (requirement(kind="current_state", requested=point),), query_type="current_state")
        self.assertEqual(decide_answerability(req, pkg).primary_reason, "requested_time_not_covered")

    def test_isolated_episode_does_not_establish_stable_trait(self):
        pkg = package((claim(),), query_type="stable_trait")
        req = request(pkg, (requirement(kind="stable_trait"),), query_type="stable_trait")
        self.assertEqual(decide_answerability(req, pkg).primary_reason, "stable_trait_support_insufficient")

    def test_durative_interval_establishes_stable_trait(self):
        pkg = package((claim(memory_kind="durative"),), query_type="stable_trait")
        req = request(pkg, (requirement(kind="stable_trait"),), query_type="stable_trait")
        self.assertEqual(decide_answerability(req, pkg).decision, "answerable")

    def test_durative_repetition_requires_distinct_sessions(self):
        base = claim(memory_kind="durative", valid_from=None, valid_to=None)
        second_evidence = hashlib.sha256(b"evidence-second").hexdigest()
        item = replace(base, evidence_ids=tuple(sorted((base.evidence_ids[0], second_evidence))))

        def repeated_package():
            first_span = Span(
                item.evidence_ids[0], item.claim_id, item.claim_version_id,
                "invented_span_first", item.speaker_id,
            )
            second_span = Span(
                item.evidence_ids[1], item.claim_id, item.claim_version_id,
                "invented_span_second", item.speaker_id,
            )
            value = package((item,), query_type="stable_trait")
            return replace(value, relevant_sources=(
                Source(
                    item.user_id, "invented_source_first", "conversation",
                    "invented_session_same", datetime(2026, 1, 1, tzinfo=UTC), NOW, (first_span,),
                ),
                Source(
                    item.user_id, "invented_source_second", "conversation",
                    "invented_session_same", datetime(2026, 1, 2, tzinfo=UTC), NOW, (second_span,),
                ),
            ))

        same_session = repeated_package()
        same_sources = same_session.relevant_sources
        same_request = request(
            same_session, (requirement(kind="stable_trait"),), query_type="stable_trait",
        )
        self.assertEqual(
            decide_answerability(same_request, same_session).primary_reason,
            "stable_trait_support_insufficient",
        )

        distinct_same_time = replace(same_session, relevant_sources=(
            replace(same_sources[0], session_id="invented_session_first"),
            replace(
                same_sources[1],
                session_id="invented_session_second",
                produced_at=same_sources[0].produced_at,
            ),
        ))
        same_time_request = request(
            distinct_same_time,
            (requirement(kind="stable_trait"),),
            query_type="stable_trait",
        )
        self.assertEqual(
            decide_answerability(same_time_request, distinct_same_time).primary_reason,
            "stable_trait_support_insufficient",
        )

        distinct = replace(same_session, relevant_sources=(
            replace(same_sources[0], session_id="invented_session_first"),
            replace(same_sources[1], session_id="invented_session_second"),
        ))
        distinct_request = request(
            distinct, (requirement(kind="stable_trait"),), query_type="stable_trait",
        )
        self.assertEqual(decide_answerability(distinct_request, distinct).decision, "answerable")

    def test_temporal_adjacency_is_not_causality(self):
        pkg = package((claim(),), query_type="causal")
        req = request(pkg, (requirement(kind="causal"),), query_type="causal")
        self.assertEqual(decide_answerability(req, pkg).primary_reason, "causal_support_insufficient")

    def test_checked_causal_relation_is_required(self):
        relation = Relation(
            "invented_relation", "caused_by", "outgoing",
            "invented_claim_1", "invented_claim_2",
        )
        first = claim("1", relations=(relation,))
        second = claim("2")
        pkg = package((first, second), query_type="causal")
        req = request(pkg, (requirement(kind="causal", object_json=None),), query_type="causal")
        decision = decide_answerability(req, pkg)
        self.assertEqual(decision.decision, "answerable")
        self.assertEqual(
            {item.claim_id for item in decision.accepted_evidence},
            {"invented_claim_1", "invented_claim_2"},
        )
        self.assertIn(
            "invented_relation",
            next(
                item.relation_ids
                for item in decision.accepted_evidence
                if item.claim_id == "invented_claim_1"
            ),
        )

    def test_causal_relation_requires_directed_visible_categorized_endpoints(self):
        relation = Relation(
            "invented_relation", "caused_by", "outgoing",
            "invented_claim_1", "invented_claim_2",
        )
        first = claim("1", relations=(relation,))
        closed = claim("2", transaction_to=NOW)
        stale = package((first, closed), query_type="causal")
        stale_request = request(
            stale, (requirement(kind="causal", object_json=None),), query_type="causal",
        )
        stale_decision = decide_answerability(stale_request, stale)
        self.assertEqual(stale_decision.decision, "abstain")
        self.assertIn("causal_support_insufficient", stale_decision.reasons)

        wrong_direction = replace(
            first,
            checked_relations=(replace(relation, direction="incoming"),),
        )
        wrong = package((wrong_direction, claim("2")), query_type="causal")
        wrong_request = request(
            wrong, (requirement(kind="causal", object_json=None),), query_type="causal",
        )
        self.assertEqual(
            decide_answerability(wrong_request, wrong).primary_reason,
            "causal_support_insufficient",
        )

        candidate = claim("2")
        rejected_fields = {
            "index_record_id": "invented_record_2",
            "retrieval_rank": 2,
            "claim_id": candidate.claim_id,
            "claim_version_id": candidate.claim_version_id,
            "stage": "package_validation",
            "reasons": ("candidate_not_promoted",),
        }
        rejected = Rejection(stable_sha256(rejected_fields), **rejected_fields)
        rejected_package = replace(
            package((first,), query_type="causal"),
            rejected_evidence=(rejected,),
        )
        rejected_request = request(
            rejected_package,
            (requirement(kind="causal", object_json=None),),
            query_type="causal",
        )
        self.assertEqual(
            decide_answerability(rejected_request, rejected_package).primary_reason,
            "causal_support_insufficient",
        )

    def test_related_information_does_not_answer_requested_fact(self):
        pkg = package((claim(predicate="different_fact"),))
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        self.assertEqual(decision.primary_reason, "requested_information_absent")

    def test_explicit_unknown_semantics_clarify(self):
        pkg = package((claim(),), query_type="unknown")
        req = request(pkg, (requirement(kind="unknown", predicate=None, object_json=None),), query_type="unknown")
        self.assertEqual(decide_answerability(req, pkg).decision, "clarify")

    def test_missing_exact_predicate_clarifies_instead_of_guessing(self):
        pkg = package((claim(),))
        req = request(pkg, (requirement(predicate=None, object_json=None),))
        decision = decide_answerability(req, pkg)
        self.assertEqual((decision.decision, decision.primary_reason), ("clarify", "clarification_required"))

    def test_explicit_partial_subrequirements_allow_partial_status(self):
        supported = requirement(partial=True, subparts=("part_a", "part_b"))
        missing = requirement(predicate="missing", object_json=None, partial=True, subparts=("part_a", "part_b"))
        pkg = package((claim(),))
        decision = decide_answerability(request(pkg, (supported, missing)), pkg)
        self.assertEqual((decision.decision, decision.permitted_answer_statuses), ("answerable", ("partially_answered",)))

    def test_structurally_blocked_package_cannot_be_overridden(self):
        pkg = package((claim(),), answer_allowed=False)
        decision = decide_answerability(request(pkg, (requirement(),)), pkg)
        self.assertEqual((decision.decision, decision.primary_reason), ("abstain", "no_promoted_claims"))

    def test_restricted_cross_user_tampered_and_incomplete_are_failures(self):
        restricted = package((claim(sensitivity="restricted"),))
        with self.assertRaisesRegex(AnswerabilityPolicyError, "restricted_evidence"):
            decide_answerability(request(restricted, (requirement(),)), restricted)
        cross = package((claim(user="other_user"),))
        with self.assertRaisesRegex(AnswerabilityPolicyError, "cross_user_evidence"):
            decide_answerability(request(cross, (requirement(),)), cross)
        normal = package((claim(),))
        with self.assertRaisesRegex(AnswerabilityError, "request ID changed"):
            replace(request(normal, (requirement(),)), package_sha256="1" * 64)
        incomplete = replace(normal, relevant_sources=())
        with self.assertRaisesRegex(AnswerabilityPolicyError, "incomplete_provenance"):
            decide_answerability(request(incomplete, (requirement(),)), incomplete)


if __name__ == "__main__":
    unittest.main()
