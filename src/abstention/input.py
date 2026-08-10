"""Verified artifact-only input boundary for answerability decisions."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping

from answering.contracts import (
    CheckedRelationReference,
    EvidenceAnchor,
    EvidenceClaim,
    EvidenceCoverage,
    EvidencePackage,
    EvidenceSource,
    EvidenceSpan,
    FrozenEligibilitySnapshot,
    RejectedEvidence,
)
from answering.evaluation import verify_evidence_package_release
from retrieval.baseline_contracts import RejectedRetrievalItem
from retrieval.query_contracts import QueryPlan, RequestedValidTime, parse_retrieval_query_request

from .contracts import (
    ANSWERABILITY_VERSION,
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    POLICY_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    AnswerabilityError,
    AnswerabilityRequest,
    AnswerabilityRequirement,
    stable_sha256,
)


STEP81_DATASET = Path("data/answering/evidence-package-development-v1/manifest.json")
STEP81_ROOT = Path("results/answering/evidence-package-development-v1")
STEP81_PACKAGES = STEP81_ROOT / "packages.jsonl"
STEP82_MANIFEST = Path("results/answering/memory-answer-contract-development-v1/manifest.json")
STEP83_MANIFEST = Path("results/answering/memory-answer-quality-development-v1/manifest.json")
STEP81_DATASET_SHA256 = "016b34eccba3260974e5c8eb2be58d6fb2023ad4399919634577b82c5c4bc7f4"
STEP81_MANIFEST_SHA256 = "8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213"
STEP81_PACKAGES_SHA256 = "bb57898bea51417b2ecad1252b451669ae2c748033c9cef88360f16a825f8186"
STEP81_CHECKS_SHA256 = "9e081b65e5bc8c8f59c8fb320a357e14240eedc06fd5febb67ddbcf588d8c766"
STEP81_RUN_SHA256 = "87c10978b82cf309e01982b580c41a9e48371dd262f9d8cc28aa7a4d5afd3a1d"
STEP81_FAILURES_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
STEP82_MANIFEST_SHA256 = "d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841"
STEP83_MANIFEST_SHA256 = "ac936819856939f66597c279fc0b852022a650d5455a4c21240ffbb01f0a524f"
BASELINE_ORDER = {"B2": 0, "B3": 1, "B4": 2}


def load_answerability_inputs(
    repo_root: str | Path,
    *,
    config_sha256: str,
    policy_sha256: str,
) -> tuple[tuple[AnswerabilityRequest, EvidencePackage], ...]:
    """Verify the frozen release, then parse its canonical package records once."""

    root = Path(repo_root).resolve()
    verify_evidence_package_release(root / STEP81_ROOT, repo_root=root)
    authorities = (
        (STEP81_DATASET, STEP81_DATASET_SHA256),
        (STEP81_ROOT / "manifest.json", STEP81_MANIFEST_SHA256),
        (STEP81_PACKAGES, STEP81_PACKAGES_SHA256),
        (STEP81_ROOT / "checks.json", STEP81_CHECKS_SHA256),
        (STEP81_ROOT / "run.json", STEP81_RUN_SHA256),
        (STEP81_ROOT / "failures.jsonl", STEP81_FAILURES_SHA256),
        (STEP82_MANIFEST, STEP82_MANIFEST_SHA256),
        (STEP83_MANIFEST, STEP83_MANIFEST_SHA256),
    )
    for path, expected in authorities:
        if _sha(root / path) != expected:
            raise AnswerabilityError("input authority changed")
    raw = (root / STEP81_PACKAGES).read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != 24 or any(not line.endswith(b"\n") for line in lines):
        raise AnswerabilityError("package accounting changed")
    pairs = []
    for line in lines:
        value = json.loads(line)
        package = evidence_package_from_mapping(value)
        request = _request_from_package(
            package,
            hashlib.sha256(line).hexdigest(),
            config_sha256,
            policy_sha256,
        )
        pairs.append((request, package))
    expected = tuple(sorted(
        pairs,
        key=lambda pair: (pair[0].query_id, BASELINE_ORDER[pair[0].baseline_id]),
    ))
    if tuple(pairs) != expected or len({item[0].package_id for item in pairs}) != 24:
        raise AnswerabilityError("packages are not canonical")
    return tuple(pairs)


def evidence_package_from_mapping(value: object) -> EvidencePackage:
    """Parse through the public Step 8.1 contracts with strict field checks."""

    mapping = _mapping(value, "package")
    expected = {
        "answer_allowed", "as_of", "baseline_id", "config_sha256", "config_version",
        "conflicting_claims", "current_claims", "eligibility", "evidence_coverage",
        "execution_id", "historical_claims", "index_version",
        "input_release_checkpoint_sha256", "input_release_manifest_sha256",
        "input_release_version", "input_results_sha256", "package_id", "package_version",
        "plan", "query", "query_type", "rejected_evidence", "relevant_sources",
        "requested_valid_time", "retrieval_result_sha256", "runtime_version",
        "schema_version", "snapshot_run_id", "structural_blockers", "user_id",
    }
    if set(mapping) != expected:
        raise AnswerabilityError("package fields changed")
    query = parse_retrieval_query_request(_mapping(mapping["query"], "query"))
    plan = _plan(_mapping(mapping["plan"], "plan"))
    eligibility_value = _exact_mapping(
        mapping["eligibility"], set(FrozenEligibilitySnapshot.__dataclass_fields__), "eligibility",
    )
    pre = tuple(
        RejectedRetrievalItem(item["index_record_id"], item["stage"], tuple(item["reasons"]))
        for raw in _mapping_list(eligibility_value["pre_filter_rejections"], "pre-filter rejection")
        for item in (_exact_mapping(
            raw, set(RejectedRetrievalItem.__dataclass_fields__), "pre-filter rejection",
        ),)
    )
    eligibility = FrozenEligibilitySnapshot(
        eligibility_value["plan_id"], eligibility_value["user_id"], eligibility_value["index_version"],
        eligibility_value["snapshot_run_id"], tuple(eligibility_value["eligible_record_ids"]), pre,
        eligibility_value["snapshot_sha256"],
    )
    groups = {
        name: tuple(_claim(item) for item in _mapping_list(mapping[name], name))
        for name in ("current_claims", "historical_claims", "conflicting_claims")
    }
    sources = tuple(_source(item) for item in _mapping_list(mapping["relevant_sources"], "source"))
    rejected = tuple(
        RejectedEvidence(
            item["rejection_id"], item["index_record_id"], item["retrieval_rank"],
            item["claim_id"], item["claim_version_id"], item["stage"], tuple(item["reasons"]),
        )
        for raw in _mapping_list(mapping["rejected_evidence"], "rejected evidence")
        for item in (_exact_mapping(
            raw, set(RejectedEvidence.__dataclass_fields__), "rejected evidence",
        ),)
    )
    coverage = EvidenceCoverage(**_exact_mapping(
        mapping["evidence_coverage"], set(EvidenceCoverage.__dataclass_fields__), "coverage",
    ))
    return EvidencePackage(
        mapping["package_id"], mapping["package_version"], mapping["schema_version"],
        mapping["config_version"], mapping["config_sha256"], mapping["runtime_version"],
        mapping["input_release_version"], mapping["input_release_manifest_sha256"],
        mapping["input_release_checkpoint_sha256"], mapping["input_results_sha256"], query,
        mapping["user_id"], mapping["baseline_id"], mapping["execution_id"], plan, eligibility,
        mapping["snapshot_run_id"], mapping["index_version"], mapping["retrieval_result_sha256"],
        mapping["query_type"], _valid_time(mapping["requested_valid_time"]), _datetime(mapping["as_of"]),
        groups["current_claims"], groups["historical_claims"], groups["conflicting_claims"],
        sources, rejected, coverage, tuple(mapping["structural_blockers"]), mapping["answer_allowed"],
    )


def _request_from_package(
    package: EvidencePackage,
    package_sha256: str,
    config_sha256: str,
    policy_sha256: str,
) -> AnswerabilityRequest:
    requirement_fields = {
        "information_kind": package.query_type,
        "target_subject_ids": tuple(package.plan.entity_ids),
        "permitted_speaker_ids": tuple(package.plan.speaker_ids),
        "requested_valid_time": package.requested_valid_time,
        "required_authority_class": "exact_supported",
        "required_predicate": None,
        "required_object_json": None,
        "required_polarity": None,
        "complete_subpart_coverage": True,
        "conflict_reporting_requested": package.query_type == "evidence_request",
        "partial_response_requested": False,
        "subrequirement_ids": (),
    }
    requirement = AnswerabilityRequirement(
        requirement_id=stable_sha256(requirement_fields), **requirement_fields,
    )
    request_fields = {
        "answerability_version": ANSWERABILITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_sha256": config_sha256,
        "policy_version": POLICY_VERSION,
        "policy_sha256": policy_sha256,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": INPUT_RELEASE_VERSION,
        "input_release_manifest_sha256": STEP81_MANIFEST_SHA256,
        "package_id": package.package_id,
        "package_sha256": package_sha256,
        "user_id": package.user_id,
        "query_id": package.query.query_id,
        "baseline_id": package.baseline_id,
        "execution_id": package.execution_id,
        "plan_id": package.plan.plan_id,
        "snapshot_run_id": package.snapshot_run_id,
        "index_version": package.index_version,
        "as_of": package.as_of,
        "requested_valid_time": package.requested_valid_time,
        "query_type": package.query_type,
        "subject_scope": tuple(package.plan.entity_ids),
        "speaker_scope": tuple(package.plan.speaker_ids),
        "sensitivity_scope": "sensitive" if package.plan.allow_sensitive else "standard",
        "requirements": (requirement,),
    }
    return AnswerabilityRequest(request_id=stable_sha256(request_fields), **request_fields)


def _plan(value: Mapping[str, object]) -> QueryPlan:
    value = _exact_mapping(value, set(QueryPlan.__dataclass_fields__), "plan")
    return QueryPlan(
        value["plan_id"], value["query_id"], value["user_id"], value["planner_version"],
        value["planner_config_sha256"], value["index_version"], value["primary_label"],
        _datetime(value["as_of"]), tuple(value["enabled_record_kinds"]),
        _valid_time(value["requested_valid_time"]), tuple(value["speaker_ids"]),
        tuple(value["entity_ids"]), tuple(value["allowed_lifecycle_statuses"]),
        value["allow_sensitive"], value["allow_unclassified_sensitivity"],
        value["include_previous_versions"], value["checked_relation_expansion_intent"],
        tuple(value["relation_expansion_types"]), value["source_evidence_intent"],
        value["unresolved_time"], value["clarification_required"],
    )


def _claim(value: Mapping[str, object]) -> EvidenceClaim:
    value = _exact_mapping(value, set(EvidenceClaim.__dataclass_fields__), "claim")
    anchors = tuple(
        EvidenceAnchor(**_exact_mapping(item, set(EvidenceAnchor.__dataclass_fields__), "anchor"))
        for item in _mapping_list(value["retrieval_anchors"], "anchor")
    )
    relations = tuple(
        CheckedRelationReference(**_exact_mapping(
            item, set(CheckedRelationReference.__dataclass_fields__), "relation",
        ))
        for item in _mapping_list(value["checked_relations"], "relation")
    )
    return EvidenceClaim(
        value["user_id"], value["claim_id"], value["claim_version_id"], value["subject_id"],
        value["speaker_id"], value["predicate"], value["object_json"], value["polarity"],
        value["epistemic_status"], value["memory_kind"], value["lifecycle_status"],
        value["extraction_confidence"], value["belief_confidence"], _date(value["valid_from_date"]),
        _datetime_or_none(value["valid_from_timestamp"]), _date(value["valid_to_date"]),
        _datetime_or_none(value["valid_to_timestamp"]), value["time_precision"],
        _datetime(value["transaction_from"]), _datetime_or_none(value["transaction_to"]),
        value["sensitivity"], anchors, tuple(value["evidence_ids"]), relations,
    )


def _source(value: Mapping[str, object]) -> EvidenceSource:
    value = _exact_mapping(value, set(EvidenceSource.__dataclass_fields__), "source")
    spans = tuple(
        EvidenceSpan(**_exact_mapping(item, set(EvidenceSpan.__dataclass_fields__), "span"))
        for item in _mapping_list(value["evidence_spans"], "span")
    )
    return EvidenceSource(
        value["user_id"], value["source_id"], value["source_type"], value["session_id"],
        _datetime(value["produced_at"]), _datetime(value["ingested_at"]), value["content_hash"], spans,
    )


def _valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    item = _exact_mapping(value, set(RequestedValidTime.__dataclass_fields__), "valid time")
    return RequestedValidTime(
        item["kind"], _date(item["point_date"]), _datetime_or_none(item["point_timestamp"]),
        _date(item["range_start_date"]), _date(item["range_end_date"]),
        _datetime_or_none(item["range_start_timestamp"]), _datetime_or_none(item["range_end_timestamp"]),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AnswerabilityError(f"{name} is invalid")
    return value


def _exact_mapping(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    item = _mapping(value, name)
    if set(item) != fields:
        raise AnswerabilityError(f"{name} fields changed")
    return item


def _mapping_list(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise AnswerabilityError(f"{name} list is invalid")
    return tuple(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise AnswerabilityError("datetime is invalid")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _datetime_or_none(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnswerabilityError("date is invalid")
    return date.fromisoformat(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
