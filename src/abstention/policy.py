"""Deterministic semantic answerability policy over verified evidence packages."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib

from answering.contracts import EvidenceClaim, EvidencePackage, canonical_json_bytes as package_bytes

from .contracts import (
    ANSWERABILITY_VERSION,
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    POLICY_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    AnswerabilityDecision,
    AnswerabilityError,
    AnswerabilityEvidenceReference,
    AnswerabilityRequest,
    CoverageRiskAssessment,
    DecisionConfidence,
    RejectedAnswerabilityEvidence,
    decision_id_from_decision,
    evidence_reference_id,
    ordered_reasons,
    stable_sha256,
)


class AnswerabilityPolicyError(AnswerabilityError):
    """A sanitized hard failure at the semantic policy boundary."""

    def __init__(self, code: str, location: str) -> None:
        super().__init__(code)
        self.code = code
        self.location = location


def decide_answerability(
    request: AnswerabilityRequest,
    package: EvidencePackage,
) -> AnswerabilityDecision:
    """Return one complete deterministic decision or fail without a partial result."""

    _validate_identity(request, package)
    groups = (
        ("current_claims", package.current_claims),
        ("historical_claims", package.historical_claims),
        ("conflicting_claims", package.conflicting_claims),
    )
    claims = tuple((category, claim) for category, values in groups for claim in values)
    _validate_package_ownership(request, package, claims)
    carried = tuple(sorted(
        (
            RejectedAnswerabilityEvidence(
                item.rejection_id,
                item.index_record_id,
                item.retrieval_rank,
                item.claim_id,
                item.claim_version_id,
                item.stage,
                item.reasons,
            )
            for item in package.rejected_evidence
        ),
        key=_rejection_key,
    ))

    structural_reasons = tuple(
        reason for reason in package.structural_blockers
        if reason in {
            "no_retrieved_claims", "incomplete_evidence", "no_promoted_claims",
            "clarification_required",
        }
    )
    if not package.answer_allowed:
        if not structural_reasons:
            raise AnswerabilityPolicyError("structural_flag_invalid", "package")
        reasons = ordered_reasons(structural_reasons)
        assessment = CoverageRiskAssessment(
            package.evidence_coverage.complete,
            len(claims),
            len(request.requirements),
            0,
            True,
            True,
            request.requested_valid_time is None,
            0,
            False,
            False,
            len(package.conflicting_claims),
            None,
            None,
            False,
            tuple(requirement.requirement_id for requirement in request.requirements),
            reasons,
        )
        return _decision(
            request,
            "clarify" if reasons[0] == "clarification_required" else "abstain",
            (),
            assessment,
            reasons,
            (),
            carried,
        )

    source_index = _source_index(package)
    accepted: list[AnswerabilityEvidenceReference] = []
    semantic_rejected: list[RejectedAnswerabilityEvidence] = []
    supported_requirements: set[str] = set()
    reasons_seen: set[str] = set()
    stale_count = 0
    unresolved_conflicts = 0
    identity_match = True
    transaction_visible = True
    valid_time_covered = True
    speaker_ok = True
    source_ok = True
    trait_supported: bool | None = None
    causal_supported: bool | None = None
    information_present = True

    for requirement in request.requirements:
        if requirement.information_kind == "unknown" or requirement.required_predicate is None:
            reasons_seen.add("clarification_required")
            continue
        matching = [item for item in claims if _proposition_matches(requirement, item[1])]
        if not matching:
            information_present = False
            reasons_seen.add("requested_information_absent")
            continue
        subject_matching = [item for item in matching if _subject_matches(requirement, item[1])]
        if not subject_matching:
            identity_match = False
            reasons_seen.add("wrong_person_risk")
            _reject_claims(semantic_rejected, matching, "wrong_person_risk")
            continue
        visible = [item for item in subject_matching if _transaction_visible(item[1], request.as_of)]
        if len(visible) != len(subject_matching):
            transaction_visible = False
            stale_count += len(subject_matching) - len(visible)
            reasons_seen.add("stale_evidence")
            _reject_claims(
                semantic_rejected,
                [item for item in subject_matching if item not in visible],
                "stale_evidence",
            )
        timed = [item for item in visible if _valid_time_matches(requirement.requested_valid_time, item[1])]
        if requirement.requested_valid_time is not None and len(timed) != len(visible):
            valid_time_covered = False
            reasons_seen.add("requested_time_not_covered")
            _reject_claims(
                semantic_rejected,
                [item for item in visible if item not in timed],
                "requested_time_not_covered",
            )
        candidates = timed if requirement.requested_valid_time is not None else visible
        if requirement.information_kind == "current_state":
            historical = [item for item in candidates if item[0] == "historical_claims"]
            if historical:
                stale_count += len(historical)
                reasons_seen.add("stale_evidence")
                _reject_claims(semantic_rejected, historical, "stale_evidence")
                candidates = [item for item in candidates if item[0] != "historical_claims"]
        speakers = [item for item in candidates if _speaker_matches(requirement, item[1])]
        if candidates and not speakers:
            speaker_ok = False
            reasons_seen.add("insufficient_speaker_authority")
            _reject_claims(semantic_rejected, candidates, "insufficient_speaker_authority")
            continue
        authoritative = [
            item for item in speakers
            if _authority_matches(requirement.required_authority_class, item[1], package, source_index)
        ]
        if speakers and not authoritative:
            source_ok = False
            reasons_seen.add("insufficient_source_authority")
            _reject_claims(semantic_rejected, speakers, "insufficient_source_authority")
            continue

        conflicts = [item for item in authoritative if item[0] == "conflicting_claims"]
        ordinary = [item for item in authoritative if item[0] != "conflicting_claims"]
        if conflicts:
            if requirement.conflict_reporting_requested and len({item[1].claim_version_id for item in conflicts}) >= 2:
                for item in conflicts:
                    accepted.append(_reference(item[0], item[1], source_index))
                supported_requirements.add(requirement.requirement_id)
                continue
            unresolved_conflicts += len(conflicts)
            reasons_seen.add("unresolved_conflict")
            _reject_claims(semantic_rejected, conflicts, "unresolved_conflict")
            continue

        if requirement.information_kind == "stable_trait":
            trait_matches = [item for item in ordinary if _durative_support(item[1], source_index)]
            trait_supported = bool(trait_matches)
            if not trait_matches:
                reasons_seen.add("stable_trait_support_insufficient")
                _reject_claims(semantic_rejected, ordinary, "stable_trait_support_insufficient")
                continue
            ordinary = trait_matches
        if requirement.information_kind == "causal":
            causal_matches = _causal_supporting_claims(
                ordinary,
                claims,
                requirement,
                request.as_of,
                source_index,
            )
            causal_supported = bool(causal_matches)
            if not causal_matches:
                reasons_seen.add("causal_support_insufficient")
                _reject_claims(semantic_rejected, ordinary, "causal_support_insufficient")
                continue
            ordinary = causal_matches
        if ordinary:
            for item in ordinary:
                accepted.append(_reference(item[0], item[1], source_index))
            supported_requirements.add(requirement.requirement_id)

    unsupported = tuple(sorted(
        requirement.requirement_id for requirement in request.requirements
        if requirement.requirement_id not in supported_requirements
    ))
    if unsupported and any(requirement.complete_subpart_coverage for requirement in request.requirements):
        if any(requirement.partial_response_requested for requirement in request.requirements) and supported_requirements:
            pass
        elif not reasons_seen:
            reasons_seen.add("requested_information_absent")
    reasons = ordered_reasons(tuple(reasons_seen))
    accepted_tuple = tuple(sorted(set(accepted), key=lambda item: item.reference_id))
    rejected_tuple = tuple(sorted(set(carried + tuple(semantic_rejected)), key=_rejection_key))
    assessment = CoverageRiskAssessment(
        package.evidence_coverage.complete,
        len(claims),
        len(request.requirements),
        len(supported_requirements),
        identity_match,
        transaction_visible,
        valid_time_covered,
        stale_count,
        speaker_ok,
        source_ok,
        unresolved_conflicts,
        trait_supported,
        causal_supported,
        information_present,
        unsupported,
        reasons,
    )
    if reasons:
        decision = "clarify" if reasons[0] == "clarification_required" else "abstain"
        statuses: tuple[str, ...] = ()
        if supported_requirements and any(item.partial_response_requested for item in request.requirements):
            decision, statuses = "answerable", ("partially_answered",)
    else:
        decision = "answerable"
        if accepted_tuple and all(item.category == "conflicting_claims" for item in accepted_tuple):
            statuses = ("disputed",)
        else:
            statuses = ("answered",)
    return _decision(
        request, decision, statuses, assessment, reasons, accepted_tuple, rejected_tuple,
    )


def _decision(
    request: AnswerabilityRequest,
    decision: str,
    statuses: tuple[str, ...],
    assessment: CoverageRiskAssessment,
    reasons: tuple[str, ...],
    accepted: tuple[AnswerabilityEvidenceReference, ...],
    rejected: tuple[RejectedAnswerabilityEvidence, ...],
) -> AnswerabilityDecision:
    fields = {
        "answerability_version": ANSWERABILITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_sha256": request.config_sha256,
        "policy_version": POLICY_VERSION,
        "policy_sha256": request.policy_sha256,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": INPUT_RELEASE_VERSION,
        "input_release_manifest_sha256": request.input_release_manifest_sha256,
        "package_id": request.package_id,
        "package_sha256": request.package_sha256,
        "request_id": request.request_id,
        "user_id": request.user_id,
        "query_id": request.query_id,
        "baseline_id": request.baseline_id,
        "execution_id": request.execution_id,
        "plan_id": request.plan_id,
        "snapshot_run_id": request.snapshot_run_id,
        "index_version": request.index_version,
        "as_of": request.as_of,
        "requested_valid_time": request.requested_valid_time,
        "query_type": request.query_type,
        "decision": decision,
        "permitted_answer_statuses": statuses,
        "generation_allowed": decision == "answerable",
        "confidence": DecisionConfidence(None, "not_calibrated", "step_9_2_not_run"),
        "assessment": assessment,
        "primary_reason": reasons[0] if reasons else None,
        "reasons": reasons,
        "accepted_evidence": accepted,
        "rejected_evidence": rejected,
    }
    return AnswerabilityDecision(decision_id=stable_sha256(fields), **fields)


def _validate_identity(request: AnswerabilityRequest, package: EvidencePackage) -> None:
    expected = (
        package.package_id,
        package.user_id,
        package.query.query_id,
        package.baseline_id,
        package.execution_id,
        package.plan.plan_id,
        package.snapshot_run_id,
        package.index_version,
        package.as_of,
        package.requested_valid_time,
        package.query_type,
    )
    actual = (
        request.package_id,
        request.user_id,
        request.query_id,
        request.baseline_id,
        request.execution_id,
        request.plan_id,
        request.snapshot_run_id,
        request.index_version,
        request.as_of,
        request.requested_valid_time,
        request.query_type,
    )
    if actual != expected:
        raise AnswerabilityPolicyError("input_identity_mismatch", "request")
    if hasattr(package.plan, "entity_ids") and request.subject_scope != tuple(package.plan.entity_ids):
        raise AnswerabilityPolicyError("input_identity_mismatch", "subject_scope")
    if hasattr(package.plan, "speaker_ids") and request.speaker_scope != tuple(package.plan.speaker_ids):
        raise AnswerabilityPolicyError("input_identity_mismatch", "speaker_scope")
    if hasattr(package.plan, "allow_sensitive"):
        expected_sensitivity = "sensitive" if package.plan.allow_sensitive else "standard"
        if request.sensitivity_scope != expected_sensitivity:
            raise AnswerabilityPolicyError("input_identity_mismatch", "sensitivity_scope")
    if hashlib.sha256(package_bytes(package)).hexdigest() != request.package_sha256:
        raise AnswerabilityPolicyError("package_hash_mismatch", "package")


def _validate_package_ownership(request, package, claims) -> None:
    if any(claim.user_id != request.user_id for _, claim in claims):
        raise AnswerabilityPolicyError("cross_user_evidence", "claim")
    if any(source.user_id != request.user_id for source in package.relevant_sources):
        raise AnswerabilityPolicyError("cross_user_evidence", "source")
    if any(claim.sensitivity == "restricted" for _, claim in claims):
        raise AnswerabilityPolicyError("restricted_evidence", "claim")
    evidence_by_claim: dict[tuple[str, str], set[str]] = {}
    for source in package.relevant_sources:
        for span in source.evidence_spans:
            evidence_by_claim.setdefault((span.claim_id, span.claim_version_id), set()).add(span.evidence_id)
    if any(
        evidence_by_claim.get((claim.claim_id, claim.claim_version_id), set()) != set(claim.evidence_ids)
        for _, claim in claims
    ):
        raise AnswerabilityPolicyError("incomplete_provenance", "evidence")
    claim_ids = {claim.claim_id for _, claim in claims}
    claim_ids.update(
        item.claim_id for item in package.rejected_evidence if item.claim_id is not None
    )
    for _, claim in claims:
        for relation in claim.checked_relations:
            if relation.source_claim_id not in claim_ids or relation.target_claim_id not in claim_ids:
                raise AnswerabilityPolicyError("relation_ownership_invalid", "relation")


def _source_index(package: EvidencePackage) -> dict[tuple[str, str], tuple[tuple[object, object], ...]]:
    index: dict[tuple[str, str], list[tuple[object, object]]] = {}
    for source in package.relevant_sources:
        if source.ingested_at > package.as_of:
            raise AnswerabilityPolicyError("source_after_cutoff", "source")
        for span in source.evidence_spans:
            index.setdefault((span.claim_id, span.claim_version_id), []).append((source, span))
    return {key: tuple(value) for key, value in index.items()}


def _reference(category, claim, source_index) -> AnswerabilityEvidenceReference:
    pairs = source_index.get((claim.claim_id, claim.claim_version_id), ())
    if not pairs or {span.evidence_id for _, span in pairs} != set(claim.evidence_ids):
        raise AnswerabilityPolicyError("incomplete_provenance", "evidence")
    fields = {
        "claim_id": claim.claim_id,
        "claim_version_id": claim.claim_version_id,
        "category": category,
        "source_ids": tuple(sorted({source.source_id for source, _ in pairs})),
        "span_ids": tuple(sorted({span.span_id for _, span in pairs})),
        "evidence_ids": tuple(sorted(claim.evidence_ids)),
        "relation_ids": tuple(sorted({relation.relation_id for relation in claim.checked_relations})),
    }
    return AnswerabilityEvidenceReference(reference_id=stable_sha256(fields), **fields)


def _proposition_matches(requirement, claim) -> bool:
    if requirement.required_predicate is not None and claim.predicate != requirement.required_predicate:
        return False
    if requirement.required_object_json is not None and claim.object_json != requirement.required_object_json:
        return False
    if requirement.required_polarity is not None and claim.polarity != requirement.required_polarity:
        return False
    return True


def _subject_matches(requirement, claim) -> bool:
    return not requirement.target_subject_ids or claim.subject_id in requirement.target_subject_ids


def _speaker_matches(requirement, claim) -> bool:
    return not requirement.permitted_speaker_ids or claim.speaker_id in requirement.permitted_speaker_ids


def _transaction_visible(claim: EvidenceClaim, as_of: datetime) -> bool:
    return claim.transaction_from <= as_of and (claim.transaction_to is None or as_of < claim.transaction_to)


def _valid_time_matches(requested, claim: EvidenceClaim) -> bool:
    if requested is None:
        return True
    start, end = _claim_bounds(claim)
    if start is None and end is None:
        return False
    req_start, req_end = _requested_bounds(requested)
    return (end is None or req_start <= end) and (start is None or start <= req_end)


def _claim_bounds(claim: EvidenceClaim) -> tuple[datetime | None, datetime | None]:
    start = claim.valid_from_timestamp
    end = claim.valid_to_timestamp
    if claim.valid_from_date is not None:
        start = datetime.combine(claim.valid_from_date, time.min, timezone.utc)
    if claim.valid_to_date is not None:
        end = datetime.combine(claim.valid_to_date, time.max, timezone.utc)
    return start, end


def _requested_bounds(requested) -> tuple[datetime, datetime]:
    if requested.kind == "point":
        if requested.point_timestamp is not None:
            return requested.point_timestamp, requested.point_timestamp
        point = datetime.combine(requested.point_date, time.min, timezone.utc)
        return point, datetime.combine(requested.point_date, time.max, timezone.utc)
    if requested.range_start_timestamp is not None:
        return requested.range_start_timestamp, requested.range_end_timestamp
    return (
        datetime.combine(requested.range_start_date, time.min, timezone.utc),
        datetime.combine(requested.range_end_date, time.max, timezone.utc),
    )


def _authority_matches(authority, claim, package, source_index) -> bool:
    pairs = source_index.get((claim.claim_id, claim.claim_version_id), ())
    if not pairs:
        return False
    if claim.epistemic_status in {"hypothetical", "uncertain", "reported_by_other"}:
        return False
    if authority == "direct_subject":
        return claim.speaker_id == claim.subject_id and any(
            span.speaker_id == claim.speaker_id for _, span in pairs
        )
    if authority == "firsthand_participant":
        return any(
            source.source_type in {"conversation", "chat"}
            and span.speaker_id == claim.speaker_id
            for source, span in pairs
        )
    if authority == "official_record":
        return any(
            source.source_type in {"email", "calendar"}
            and span.speaker_id == claim.speaker_id
            for source, span in pairs
        )
    if authority == "accepted_durative":
        return _durative_support(claim, source_index)
    if authority == "checked_causal":
        return any(relation.relation_type == "caused_by" for relation in claim.checked_relations)
    return True


def _durative_support(claim, source_index) -> bool:
    pairs = source_index.get((claim.claim_id, claim.claim_version_id), ())
    explicit_interval = any(
        value is not None for value in (
            claim.valid_from_date, claim.valid_from_timestamp,
        )
    ) and any(value is not None for value in (claim.valid_to_date, claim.valid_to_timestamp))
    source_ids = {source.source_id for source, _ in pairs}
    session_ids = {source.session_id for source, _ in pairs if source.session_id is not None}
    episode_times = {source.produced_at for source, _ in pairs}
    repeated_episodes = (
        len(source_ids) >= 2
        and len(session_ids) >= 2
        and len(episode_times) >= 2
    )
    return claim.memory_kind == "durative" and (repeated_episodes or explicit_interval)


def _causal_supporting_claims(
    matching,
    categorized,
    requirement,
    as_of,
    source_index,
):
    """Return exact directed effect/cause endpoints that remain eligible together."""

    supported = []
    for source_item in matching:
        source_claim = source_item[1]
        for relation in source_claim.checked_relations:
            if (
                relation.relation_type != "caused_by"
                or relation.direction != "outgoing"
                or relation.source_claim_id != source_claim.claim_id
                or relation.target_claim_id == source_claim.claim_id
            ):
                continue
            targets = [
                item for item in categorized
                if item[0] != "conflicting_claims"
                and item[1].claim_id == relation.target_claim_id
                and _transaction_visible(item[1], as_of)
                and _valid_time_matches(requirement.requested_valid_time, item[1])
                and _authority_matches("exact_supported", item[1], None, source_index)
            ]
            if len(targets) != 1:
                continue
            supported.extend((source_item, targets[0]))
    unique = []
    seen = set()
    for item in supported:
        key = (item[0], item[1].claim_id, item[1].claim_version_id)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _reject_claims(target, values, reason) -> None:
    for category, claim in values:
        anchors = claim.retrieval_anchors
        fields = {
            "index_record_id": anchors[0].index_record_id,
            "retrieval_rank": anchors[0].retrieval_rank,
            "claim_id": claim.claim_id,
            "claim_version_id": claim.claim_version_id,
            "stage": "semantic_policy",
            "reasons": (reason,),
        }
        target.append(RejectedAnswerabilityEvidence(rejection_id=stable_sha256(fields), **fields))


def _rejection_key(value: RejectedAnswerabilityEvidence) -> tuple[int, int, str, str, str]:
    order = {"pre_filter": 0, "post_rank": 1, "package_validation": 2, "semantic_policy": 3}
    return (
        order[value.stage], value.retrieval_rank or 0, value.index_record_id,
        value.claim_id or "", value.claim_version_id or "",
    )
