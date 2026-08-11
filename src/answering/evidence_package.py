"""Pure assembly of validated evidence packages from frozen retrieval results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .contracts import (
    BLOCKERS,
    CATEGORY_BY_STATUS,
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    EvidenceClaim,
    EvidenceCoverage,
    EvidencePackage,
    EvidencePackageBuildRequest,
    EvidencePackageError,
    EvidenceSource,
    RejectedEvidence,
    blocker_tuple,
    package_id,
    rejection_id,
)
from .repository import HydratedPackageInput, _transaction_visible, _valid_time_matches


CONFIG_PATH = Path("configs/answering/evidence_package_v1.json")


def load_evidence_package_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[Mapping[str, object], str]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidencePackageError("evidence package config is invalid") from error
    expected = {
        "answer_allowed_rule", "category_order", "config_version",
        "evidence_package_version", "failure_codes", "factual_lifecycle_categories",
        "input_release_version", "non_promoted_lifecycle_status", "rejection_reasons",
        "rejection_stages", "runtime_version", "schema_version", "structural_blockers",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidencePackageError("evidence package config fields changed")
    if (
        value["config_version"] != CONFIG_VERSION
        or value["evidence_package_version"] != PACKAGE_VERSION
        or value["schema_version"] != SCHEMA_VERSION
        or value["runtime_version"] != RUNTIME_VERSION
        or value["input_release_version"] != INPUT_RELEASE_VERSION
        or value["factual_lifecycle_categories"] != CATEGORY_BY_STATUS
        or value["non_promoted_lifecycle_status"] != "candidate"
        or tuple(value["structural_blockers"]) != BLOCKERS
    ):
        raise EvidencePackageError("evidence package policy changed")
    return value, hashlib.sha256(raw).hexdigest()


def build_evidence_package(
    request: EvidencePackageBuildRequest,
    hydrated: HydratedPackageInput,
) -> EvidencePackage:
    if any(item.user_id != request.query.user_id for item in hydrated.claims) or any(
        item.user_id != request.query.user_id for item in hydrated.sources
    ):
        raise EvidencePackageError("hydrated package contains another user")

    current: list[EvidenceClaim] = []
    historical: list[EvidenceClaim] = []
    conflicting: list[EvidenceClaim] = []
    candidate_rejections: list[RejectedEvidence] = []
    categorized_evidence: dict[str, object] = {}
    for item in hydrated.claims:
        checked_relations = []
        claim_key = (item.claim_id, item.claim_version_id)
        for relation in item.relations:
            endpoints = {
                (relation.left_claim_id, relation.left_version_id),
                (relation.right_claim_id, relation.right_version_id),
            }
            reference = relation.reference
            if (
                relation.user_id != request.query.user_id
                or claim_key not in endpoints
                or {reference.source_claim_id, reference.target_claim_id}
                != {relation.left_claim_id, relation.right_claim_id}
            ):
                raise EvidencePackageError("relation endpoint ownership is invalid")
            if max(
                relation.created_at,
                relation.decision_transaction_as_of,
                relation.classified_at,
            ) > request.query.as_of:
                raise EvidencePackageError("relation cutoff is invalid")
            if not _transaction_visible(
                request.query.as_of,
                relation.left_transaction_from,
                relation.left_transaction_to,
            ) or not _transaction_visible(
                request.query.as_of,
                relation.right_transaction_from,
                relation.right_transaction_to,
            ):
                raise EvidencePackageError("relation endpoint transaction is invalid")
            if not _valid_time_matches(
                request.plan.requested_valid_time,
                (
                    relation.left_valid_from_date,
                    relation.left_valid_from_timestamp,
                    relation.left_valid_to_date,
                    relation.left_valid_to_timestamp,
                    relation.left_time_precision,
                ),
            ) or not _valid_time_matches(
                request.plan.requested_valid_time,
                (
                    relation.right_valid_from_date,
                    relation.right_valid_from_timestamp,
                    relation.right_valid_to_date,
                    relation.right_valid_to_timestamp,
                    relation.right_time_precision,
                ),
            ):
                raise EvidencePackageError("relation endpoint valid time is invalid")
            checked_relations.append(reference)
        claim = EvidenceClaim(
            item.user_id,
            item.claim_id,
            item.claim_version_id,
            item.subject_id,
            item.speaker_id,
            item.predicate,
            item.object_json,
            item.polarity,
            item.epistemic_status,
            item.memory_kind,
            item.lifecycle_status,
            item.extraction_confidence,
            item.belief_confidence,
            item.valid_from_date,
            item.valid_from_timestamp,
            item.valid_to_date,
            item.valid_to_timestamp,
            item.time_precision,
            item.transaction_from,
            item.transaction_to,
            item.sensitivity,
            item.anchors,
            tuple(sorted(span.evidence_id for span in item.evidence)),
            tuple(checked_relations),
        )
        if item.lifecycle_status == "candidate":
            if not item.evidence:
                raise EvidencePackageError("candidate evidence is incomplete")
            for anchor in item.anchors:
                reasons = ("candidate_not_promoted",)
                candidate_rejections.append(RejectedEvidence(
                    rejection_id(
                        anchor.index_record_id,
                        anchor.retrieval_rank,
                        item.claim_id,
                        item.claim_version_id,
                        "package_validation",
                        reasons,
                    ),
                    anchor.index_record_id,
                    anchor.retrieval_rank,
                    item.claim_id,
                    item.claim_version_id,
                    "package_validation",
                    reasons,
                ))
            continue
        category = CATEGORY_BY_STATUS.get(item.lifecycle_status)
        if category is None:
            raise EvidencePackageError("excluded lifecycle reached package assembly")
        for span in item.evidence:
            previous = categorized_evidence.setdefault(span.evidence_id, span)
            if previous != span:
                raise EvidencePackageError("evidence ID collision")
        {"current_claims": current, "historical_claims": historical, "conflicting_claims": conflicting}[category].append(claim)

    def claim_key(item: EvidenceClaim) -> tuple[int, str, str]:
        return (
            min(anchor.retrieval_rank for anchor in item.retrieval_anchors),
            item.claim_id,
            item.claim_version_id,
        )

    for group in (current, historical, conflicting):
        group.sort(key=claim_key)

    relevant_sources: list[EvidenceSource] = []
    for source in hydrated.sources:
        spans = tuple(sorted(
            (
                span for span in categorized_evidence.values()
                if getattr(span, "source_id") == source.source_id
            ),
            key=lambda span: (
                span.span_id, span.claim_id, span.claim_version_id, span.support_type
            ),
        ))
        if spans:
            relevant_sources.append(EvidenceSource(
                source.user_id,
                source.source_id,
                source.source_type,
                source.session_id,
                source.produced_at,
                source.ingested_at,
                source.content_hash,
                spans,
            ))
    relevant_sources.sort(key=lambda item: item.source_id)

    categorized_count = len(current) + len(historical) + len(conflicting)
    candidate_keys = {
        (item.claim_id, item.claim_version_id) for item in candidate_rejections
    }
    denominator = categorized_count + len(candidate_keys)
    supported = sum(bool(item.evidence_ids) for item in (*current, *historical, *conflicting)) + len(candidate_keys)
    coverage = EvidenceCoverage(
        denominator,
        supported,
        len(relevant_sources),
        sum(len(source.evidence_spans) for source in relevant_sources),
        None if denominator == 0 else f"{supported / denominator:.6f}",
        "no_retrieved_claims" if denominator == 0 else None,
        denominator > 0 and supported == denominator,
    )
    blockers = blocker_tuple(
        coverage,
        bool(current or historical or conflicting),
        request.plan.clarification_required,
    )

    carried: list[RejectedEvidence] = []
    for item in request.retrieval_result.rejected:
        carried.append(RejectedEvidence(
            rejection_id(item.index_record_id, None, None, None, item.stage, item.reasons),
            item.index_record_id,
            None,
            None,
            None,
            item.stage,
            item.reasons,
        ))
    stage_order = {name: index for index, name in enumerate(("pre_filter", "post_rank", "package_validation"))}
    rejected = tuple(sorted(
        {*carried, *candidate_rejections},
        key=lambda item: (
            stage_order[item.stage],
            item.retrieval_rank or 0,
            item.index_record_id,
            item.claim_id or "",
            item.claim_version_id or "",
        ),
    ))
    return EvidencePackage(
        package_id(request),
        request.package_version,
        request.schema_version,
        request.config_version,
        request.config_sha256,
        request.runtime_version,
        request.input_release_version,
        request.input_release_manifest_sha256,
        request.input_release_checkpoint_sha256,
        request.input_results_sha256,
        request.query,
        request.query.user_id,
        request.retrieval_result.baseline_id,
        request.retrieval_result.execution_id,
        request.plan,
        request.eligibility,
        request.retrieval_result.snapshot_run_id,
        request.query.index_version,
        request.retrieval_result_sha256,
        request.plan.primary_label,
        request.plan.requested_valid_time,
        request.query.as_of,
        tuple(current),
        tuple(historical),
        tuple(conflicting),
        tuple(relevant_sources),
        rejected,
        coverage,
        blockers,
        not blockers,
    )
