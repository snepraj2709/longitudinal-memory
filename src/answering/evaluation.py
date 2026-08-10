"""Deterministic development release for validated evidence packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

from retrieval.baseline_contracts import (
    BASELINE_RECORD_KINDS,
    BaselineRetrievalResult,
    ExpansionPath,
    RejectedRetrievalItem,
    RetrievedItem,
    RRFContribution,
    SearchChannelHit,
)
from retrieval.baseline_evaluation import BASELINES, CASE_IDS, load_development_queries
from retrieval.index_evaluation import execute_index_evaluation
from retrieval.query_contracts import (
    LIFECYCLE_STATUSES,
    PLANNER_VERSION,
    RELATION_EXPANSION_TYPES,
    QueryPlan,
    RequestedValidTime,
    parse_retrieval_query_request,
    request_canonical_value,
    stable_sha256 as retrieval_sha256,
)

from .contracts import (
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    CheckedRelationReference,
    EvidenceAnchor,
    EvidenceClaim,
    EvidenceCoverage,
    EvidencePackage,
    EvidencePackageBuildRequest,
    EvidencePackageFailure,
    EvidenceSource,
    EvidenceSpan,
    FrozenEligibilitySnapshot,
    RejectedEvidence,
    canonical_json_bytes,
    stable_sha256,
)
from .evidence_package import build_evidence_package, load_evidence_package_config
from .repository import EvidencePackageRepository, EvidencePackageRepositoryError


DATASET_VERSION = "evidence_package_development_v1"
GUIDANCE_VERSION = "step-8.1-guidance-v1"
GUIDANCE_SHA256 = "6af1d3eb7d227c430a93b89abfce8fcc1853c5d29027f828650b5c8a58dec229"
STARTING_COMMIT = "1df7e3c0846e9bebb873debe4cc2ae1f532a2cb1"
DATASET_MANIFEST = Path("data/answering/evidence-package-development-v1/manifest.json")
RESULT_ROOT = Path("results/answering/evidence-package-development-v1")
BASELINE_RESULTS = Path("results/retrieval/baseline-execution-development-v1/results.jsonl")
BASELINE_MANIFEST_SHA256 = "ab45d4a51766d9d0edcc39c77f8b0ccd4abbcb1254437153f7759418cfe01703"
BASELINE_CHECKPOINT_SHA256 = "d99a2ee8720f16fb40867f92d1740582536ecced84c7eadfffc5f1907065e3b4"
BASELINE_RESULTS_SHA256 = "e1e69fe8ac64a81f27e171cdd7082b7648e2a339659f11e6e4f2a1702a1dbb60"
PLANNER_CONFIG_SHA256 = "538af5ceb41f50c752dc086c9f6ef39ee6b42b4ec0616948b3ac38192f66c654"
ALLOWED_USERS = ("user_001", "user_002")
ARTIFACT_NAMES = ("packages.jsonl", "failures.jsonl", "checks.json", "run.json", "findings.md")
IMPLEMENTATION_PATHS = (
    "configs/answering/evidence_package_v1.json",
    "src/answering/__init__.py",
    "src/answering/contracts.py",
    "src/answering/repository.py",
    "src/answering/evidence_package.py",
    "src/answering/evaluation.py",
)
PROTECTED_AUTHORITIES = {
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "results/retrieval/baseline-execution-development-v1/manifest.json": BASELINE_MANIFEST_SHA256,
    "results/retrieval/baseline-execution-development-v1/runtime-checkpoint.json": BASELINE_CHECKPOINT_SHA256,
    "results/retrieval/baseline-execution-development-v1/results.jsonl": BASELINE_RESULTS_SHA256,
    "results/retrieval/retrieval-quality-development-v1/manifest.json": "6e89700beb6a483ce0b23c3033927122c2170897b81a366ffde31fc7785a63e4",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidencePackageEvaluationError(RuntimeError):
    """Reject changed inputs, incomplete packages, or a mutable release."""


@dataclass(frozen=True)
class EvidencePackageChecks:
    dataset_version: str
    query_count: int
    user_count: int
    result_count: int
    package_count: int
    b2_package_count: int
    b3_package_count: int
    b4_package_count: int
    failure_count: int
    accepted_record_count: int
    carried_rejection_count: int
    candidate_rejection_count: int
    unique_claim_version_count: int
    categorized_claim_version_count: int
    relevant_source_count: int
    relevant_span_count: int
    coverage_complete_count: int
    answer_allowed_count: int
    duplicate_package_count: int
    cross_user_count: int
    restricted_count: int
    post_cutoff_count: int
    missing_lineage_count: int
    quote_mismatch_count: int
    partial_package_count: int
    provider_request_count: int
    retry_count: int
    input_token_count: int
    output_token_count: int
    incremental_cost_usd: int

    def __post_init__(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise EvidencePackageEvaluationError("check version changed")
        for name, value in asdict(self).items():
            if name != "dataset_version" and (type(value) is not int or value < 0):
                raise EvidencePackageEvaluationError("check count is invalid")


def execute_evidence_package_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> EvidencePackageChecks:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    dataset = _load_dataset_manifest(root)
    _verify_protected(root)
    _, config_sha256 = load_evidence_package_config(root / "configs/answering/evidence_package_v1.json")
    cases = load_development_queries(root / "data/retrieval/baseline-execution-development-v1/queries.jsonl")
    rows = _read_jsonl(root / BASELINE_RESULTS)
    if len(rows) != 24:
        raise EvidencePackageEvaluationError("frozen result accounting changed")

    with tempfile.TemporaryDirectory() as directory:
        execute_index_evaluation(connection_factory, Path(directory) / "index-release", repo_root=root)

    packages: list[EvidencePackage] = []
    failures: list[EvidencePackageFailure] = []
    connection = connection_factory()
    try:
        repository = EvidencePackageRepository(connection)
        case_map = {item.case_id: item for item in cases}
        for row in rows:
            case_id = str(row.get("case_id"))
            case = case_map.get(case_id)
            result_value = row.get("result")
            primary_label = row.get("primary_label")
            if case is None or not isinstance(result_value, dict) or not isinstance(primary_label, str):
                raise EvidencePackageEvaluationError("frozen result identity changed")
            result = _parse_result(result_value)
            query = replace(case.request, enabled_record_kinds=BASELINE_RECORD_KINDS[result.baseline_id])
            plan = _frozen_plan(query, primary_label, result.plan_id)
            eligibility = _frozen_eligibility(query.index_version, result)
            request = EvidencePackageBuildRequest(
                PACKAGE_VERSION,
                SCHEMA_VERSION,
                CONFIG_VERSION,
                config_sha256,
                RUNTIME_VERSION,
                INPUT_RELEASE_VERSION,
                BASELINE_MANIFEST_SHA256,
                BASELINE_CHECKPOINT_SHA256,
                BASELINE_RESULTS_SHA256,
                query,
                plan,
                eligibility,
                result,
                stable_sha256(result),
            )
            try:
                packages.append(build_evidence_package(request, repository.hydrate(request)))
            except Exception as error:
                failures.append(_failure(case_id, query.user_id, query.query_id, result.baseline_id, error))
    finally:
        connection.close()

    packages.sort(key=lambda item: (CASE_IDS.index(_case_id(item.query.query_id)), BASELINES.index(item.baseline_id)))
    failures.sort(key=lambda item: (item.query_id, BASELINES.index(item.baseline_id)))
    checks = score_evidence_packages(rows, packages, failures)
    payloads = _artifact_payloads(checks, packages, failures)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        _write_exclusive(output / name, payloads[name])
    _write_exclusive(
        output / "manifest.json",
        canonical_json_bytes(_manifest_payload(root, dataset, payloads, checks)),
    )
    verify_evidence_package_release(output, repo_root=root)
    return checks


def score_evidence_packages(
    result_rows: Sequence[Mapping[str, object]],
    packages: Sequence[EvidencePackage],
    failures: Sequence[EvidencePackageFailure],
) -> EvidencePackageChecks:
    package_keys = [(item.query.query_id, item.baseline_id) for item in packages]
    failure_keys = [(item.query_id, item.baseline_id) for item in failures]
    result_keys = [
        (str(row["result"]["query_id"]), str(row["result"]["baseline_id"]))
        for row in result_rows
    ]
    accepted = sum(len(row["result"]["accepted"]) for row in result_rows)
    carried = sum(len(row["result"]["rejected"]) for row in result_rows)
    unique_claims = {
        (claim.claim_id, claim.claim_version_id)
        for package in packages
        for group in (package.current_claims, package.historical_claims, package.conflicting_claims)
        for claim in group
    } | {
        (item.claim_id, item.claim_version_id)
        for package in packages
        for item in package.rejected_evidence
        if item.stage == "package_validation" and item.claim_id is not None
    }
    all_claims = [
        claim
        for package in packages
        for group in (package.current_claims, package.historical_claims, package.conflicting_claims)
        for claim in group
    ]
    all_sources = [source for package in packages for source in package.relevant_sources]
    known_keys = set(package_keys) | set(failure_keys)
    checks = EvidencePackageChecks(
        DATASET_VERSION,
        len({str(row["result"]["query_id"]) for row in result_rows}),
        len({str(row["result"]["user_id"]) for row in result_rows}),
        len(result_rows),
        len(packages),
        sum(item.baseline_id == "B2" for item in packages),
        sum(item.baseline_id == "B3" for item in packages),
        sum(item.baseline_id == "B4" for item in packages),
        len(failures),
        accepted,
        carried,
        sum(item.stage == "package_validation" for package in packages for item in package.rejected_evidence),
        len(unique_claims),
        len(all_claims),
        len(all_sources),
        sum(len(source.evidence_spans) for source in all_sources),
        sum(package.evidence_coverage.complete for package in packages),
        sum(package.answer_allowed for package in packages),
        len(package_keys) - len(set(package_keys)),
        sum(
            claim.user_id != package.user_id
            for package in packages
            for claim in [
                *package.current_claims,
                *package.historical_claims,
                *package.conflicting_claims,
            ]
        )
        + sum(
            source.user_id != package.user_id
            for package in packages
            for source in package.relevant_sources
        ),
        sum(claim.sensitivity == "restricted" for claim in all_claims),
        sum(
            claim.transaction_from > package.as_of
            or (claim.transaction_to is not None and package.as_of >= claim.transaction_to)
            for package in packages
            for claim in [
                *package.current_claims,
                *package.historical_claims,
                *package.conflicting_claims,
            ]
        ) + sum(source.ingested_at > package.as_of for package in packages for source in package.relevant_sources),
        sum(not package.evidence_coverage.complete for package in packages),
        sum(item.location in {"quote", "offset"} for item in failures),
        len(set(result_keys) - known_keys),
        sum(item.code == "provider_request" for item in failures),
        sum(item.code == "retry" for item in failures),
        sum(item.code == "input_tokens" for item in failures),
        sum(item.code == "output_tokens" for item in failures),
        sum(item.code == "incremental_cost" for item in failures),
    )
    expected_zero = (
        checks.failure_count,
        checks.categorized_claim_version_count,
        checks.relevant_source_count,
        checks.relevant_span_count,
        checks.answer_allowed_count,
        checks.duplicate_package_count,
        checks.cross_user_count,
        checks.restricted_count,
        checks.post_cutoff_count,
        checks.missing_lineage_count,
        checks.quote_mismatch_count,
        checks.partial_package_count,
        checks.provider_request_count,
        checks.retry_count,
        checks.input_token_count,
        checks.output_token_count,
        checks.incremental_cost_usd,
    )
    if (
        (checks.query_count, checks.user_count, checks.result_count, checks.package_count) != (8, 2, 24, 24)
        or (checks.b2_package_count, checks.b3_package_count, checks.b4_package_count) != (8, 8, 8)
        or checks.accepted_record_count != 204
        or checks.unique_claim_version_count != 33
        or checks.coverage_complete_count != 24
        or any(expected_zero)
    ):
        raise EvidencePackageEvaluationError("evidence package structural checks failed")
    return checks


def verify_evidence_package_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    expected_names = {*ARTIFACT_NAMES, "manifest.json"}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected_names:
        raise EvidencePackageEvaluationError("release artifact set changed")
    dataset = _load_dataset_manifest(root)
    _verify_protected(root)
    rows = _read_jsonl(root / BASELINE_RESULTS)
    package_values = _read_jsonl(output / "packages.jsonl")
    failure_values = _read_jsonl(output / "failures.jsonl")
    packages = tuple(_parse_package(value) for value in package_values)
    failures = tuple(_parse_failure(value) for value in failure_values)
    checks = score_evidence_packages(rows, packages, failures)
    expected_payloads = _artifact_payloads(checks, packages, failures)
    for name, expected in expected_payloads.items():
        if (output / name).read_bytes() != expected:
            raise EvidencePackageEvaluationError(f"{name} does not recompute")
    manifest = _read_object(output / "manifest.json")
    expected_manifest = _manifest_payload(root, dataset, expected_payloads, checks)
    if manifest != expected_manifest:
        raise EvidencePackageEvaluationError("release manifest does not recompute")
    by_key = {(item.query.query_id, item.baseline_id): item for item in packages}
    cases = {
        item.request.query_id: item
        for item in load_development_queries(root / "data/retrieval/baseline-execution-development-v1/queries.jsonl")
    }
    for row in rows:
        result = _parse_result(row["result"])
        package = by_key.get((result.query_id, result.baseline_id))
        case = cases.get(result.query_id)
        if package is None or case is None:
            raise EvidencePackageEvaluationError("package input accounting changed")
        query = replace(case.request, enabled_record_kinds=BASELINE_RECORD_KINDS[result.baseline_id])
        plan = _frozen_plan(query, str(row["primary_label"]), result.plan_id)
        eligibility = _frozen_eligibility(query.index_version, result)
        if (
            package.query != query
            or package.plan != plan
            or package.eligibility != eligibility
            or package.retrieval_result_sha256 != stable_sha256(result)
        ):
            raise EvidencePackageEvaluationError("package predecessor metadata changed")


def _frozen_plan(query, primary_label: str, expected_plan_id: str) -> QueryPlan:
    """Reconstruct the frozen plan value without invoking classification or filtering."""

    if primary_label not in {
        "current_state", "historical_state", "change_over_time", "specific_event",
        "relationship", "commitment", "evidence_request", "unknown",
    }:
        raise EvidencePackageEvaluationError("frozen query label changed")
    requested = query.requested_valid_time
    if primary_label == "current_state" and requested is None:
        raise EvidencePackageEvaluationError("frozen current plan lacks structured time")
    include_previous = primary_label in {"historical_state", "change_over_time"}
    relation_intent = primary_label == "change_over_time"
    allowed = (
        ("candidate", "confirmed", "current", "disputed")
        if primary_label == "current_state"
        else LIFECYCLE_STATUSES
    )
    plan_value = {
        "request": request_canonical_value(query),
        "planner_version": PLANNER_VERSION,
        "planner_config_sha256": PLANNER_CONFIG_SHA256,
        "primary_label": primary_label,
        "resolved_requested_valid_time": _valid_time_value(requested),
        "allowed_lifecycle_statuses": allowed,
        "include_previous_versions": include_previous,
        "checked_relation_expansion_intent": relation_intent,
        "source_evidence_intent": primary_label == "evidence_request",
        "unresolved_time": False,
    }
    plan = QueryPlan(
        retrieval_sha256(plan_value),
        query.query_id,
        query.user_id,
        PLANNER_VERSION,
        PLANNER_CONFIG_SHA256,
        query.index_version,
        primary_label,
        query.as_of,
        query.enabled_record_kinds,
        requested,
        query.speaker_ids,
        query.entity_ids,
        allowed,
        query.sensitivity_scope == "sensitive",
        query.allow_unclassified_sensitivity,
        include_previous,
        relation_intent,
        RELATION_EXPANSION_TYPES if relation_intent else (),
        primary_label == "evidence_request",
        False,
        False,
    )
    if plan.plan_id != expected_plan_id:
        raise EvidencePackageEvaluationError("frozen plan ID changed")
    return plan


def _frozen_eligibility(index_version: str, result: BaselineRetrievalResult) -> FrozenEligibilitySnapshot:
    eligible = tuple(sorted({
        *(item.index_record_id for item in result.accepted),
        *(item.index_record_id for item in result.rejected if item.stage == "post_rank"),
    }))
    pre = tuple(sorted(
        (item for item in result.rejected if item.stage == "pre_filter"),
        key=lambda item: item.index_record_id,
    ))
    value = {
        "plan_id": result.plan_id,
        "user_id": result.user_id,
        "index_version": index_version,
        "snapshot_run_id": result.snapshot_run_id,
        "eligible_record_ids": eligible,
        "pre_filter_rejections": pre,
    }
    return FrozenEligibilitySnapshot(
        result.plan_id,
        result.user_id,
        index_version,
        result.snapshot_run_id,
        eligible,
        pre,
        stable_sha256(value),
    )


def _artifact_payloads(
    checks: EvidencePackageChecks,
    packages: Sequence[EvidencePackage],
    failures: Sequence[EvidencePackageFailure],
) -> dict[str, bytes]:
    return {
        "packages.jsonl": _serialize(packages),
        "failures.jsonl": _serialize(failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(_run_payload(checks)),
        "findings.md": _findings(checks).encode("utf-8"),
    }


def _run_payload(checks: EvidencePackageChecks) -> Mapping[str, object]:
    return {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": INPUT_RELEASE_VERSION,
        "query_count": checks.query_count,
        "result_count": checks.result_count,
        "package_count": checks.package_count,
        "failure_count": checks.failure_count,
        "execution_mode": "deterministic_no_model",
        "provider_requests": checks.provider_request_count,
        "retries": checks.retry_count,
        "input_tokens": checks.input_token_count,
        "output_tokens": checks.output_token_count,
        "incremental_cost_usd": checks.incremental_cost_usd,
        "historical_openai_spend_usd": "0.2314404",
    }


def _manifest_payload(
    root: Path,
    dataset: Mapping[str, object],
    payloads: Mapping[str, bytes],
    checks: EvidencePackageChecks,
) -> Mapping[str, object]:
    return {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "input_release_version": INPUT_RELEASE_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "dataset": {"path": DATASET_MANIFEST.as_posix(), "sha256": _sha(root / DATASET_MANIFEST)},
        "runtime_inputs": dataset["inputs"],
        "implementation_hashes": {path: _sha(root / path) for path in IMPLEMENTATION_PATHS},
        "protected_authorities": PROTECTED_AUTHORITIES,
        "predecessor_drift": [],
        "artifacts": {name: hashlib.sha256(payloads[name]).hexdigest() for name in ARTIFACT_NAMES},
        "checks": asdict(checks),
        "limitations": [
            "All 33 development claims remain candidates, so no package contains a promoted factual claim.",
            "Evidence coverage measures lineage only. It does not measure relevance, truth, or answer quality.",
            "The release builds packages from frozen development retrieval results and does not reuse them after source deletion.",
        ],
        "excluded_inputs": dataset["excluded_inputs"],
        "model_usage": {
            "provider_requests": checks.provider_request_count,
            "retries": checks.retry_count,
            "input_tokens": checks.input_token_count,
            "output_tokens": checks.output_token_count,
            "incremental_cost_usd": checks.incremental_cost_usd,
            "historical_openai_spend_usd": "0.2314404",
        },
    }


def _parse_result(value: Mapping[str, object]) -> BaselineRetrievalResult:
    expected = {
        "accepted", "baseline_id", "channel_notices", "execution_id", "plan_id",
        "query_id", "rejected", "snapshot_run_id", "user_id",
    }
    if set(value) != expected or not isinstance(value["accepted"], list) or not isinstance(value["rejected"], list):
        raise EvidencePackageEvaluationError("retrieval result fields changed")
    accepted = []
    for item in value["accepted"]:
        hits = tuple(SearchChannelHit(**hit) for hit in item["component_hits"])
        contributions = tuple(RRFContribution(**entry) for entry in item["rrf_contributions"])
        paths = tuple(ExpansionPath(**entry) for entry in item["expansion_paths"])
        accepted.append(RetrievedItem(
            item["index_record_id"], item["record_kind"], item["atomic_claim_version_id"],
            item["session_summary_id"], item["content_sha256"], tuple(item["lifecycle_statuses"]),
            tuple(item["claim_ids"]), tuple(item["claim_version_ids"]), tuple(item["source_ids"]),
            tuple(item["span_ids"]), hits, contributions, paths, item["final_score"], item["rank"],
        ))
    rejected = tuple(
        RejectedRetrievalItem(item["index_record_id"], item["stage"], tuple(item["reasons"]))
        for item in value["rejected"]
    )
    return BaselineRetrievalResult(
        value["execution_id"], value["baseline_id"], value["query_id"], value["user_id"],
        value["plan_id"], value["snapshot_run_id"], tuple(accepted), rejected,
        tuple(value["channel_notices"]),
    )


def _parse_package(value: Mapping[str, object]) -> EvidencePackage:
    query = parse_retrieval_query_request(value["query"])
    plan = _parse_plan(value["plan"])
    eligibility_value = value["eligibility"]
    pre = tuple(
        RejectedRetrievalItem(item["index_record_id"], item["stage"], tuple(item["reasons"]))
        for item in eligibility_value["pre_filter_rejections"]
    )
    eligibility = FrozenEligibilitySnapshot(
        eligibility_value["plan_id"], eligibility_value["user_id"], eligibility_value["index_version"],
        eligibility_value["snapshot_run_id"], tuple(eligibility_value["eligible_record_ids"]), pre,
        eligibility_value["snapshot_sha256"],
    )
    groups = {}
    for name in ("current_claims", "historical_claims", "conflicting_claims"):
        groups[name] = tuple(_parse_claim(item) for item in value[name])
    sources = tuple(_parse_source(item) for item in value["relevant_sources"])
    rejected = tuple(RejectedEvidence(
        item["rejection_id"], item["index_record_id"], item["retrieval_rank"], item["claim_id"],
        item["claim_version_id"], item["stage"], tuple(item["reasons"]),
    ) for item in value["rejected_evidence"])
    coverage = EvidenceCoverage(**value["evidence_coverage"])
    return EvidencePackage(
        value["package_id"], value["package_version"], value["schema_version"],
        value["config_version"], value["config_sha256"], value["runtime_version"],
        value["input_release_version"], value["input_release_manifest_sha256"],
        value["input_release_checkpoint_sha256"], value["input_results_sha256"],
        query, value["user_id"], value["baseline_id"], value["execution_id"], plan, eligibility,
        value["snapshot_run_id"], value["index_version"], value["retrieval_result_sha256"],
        value["query_type"], _parse_valid_time(value["requested_valid_time"]), _datetime(value["as_of"]),
        groups["current_claims"], groups["historical_claims"], groups["conflicting_claims"],
        sources, rejected, coverage, tuple(value["structural_blockers"]), value["answer_allowed"],
    )


def _parse_plan(value: Mapping[str, object]) -> QueryPlan:
    return QueryPlan(
        value["plan_id"], value["query_id"], value["user_id"], value["planner_version"],
        value["planner_config_sha256"], value["index_version"], value["primary_label"],
        _datetime(value["as_of"]), tuple(value["enabled_record_kinds"]),
        _parse_valid_time(value["requested_valid_time"]), tuple(value["speaker_ids"]),
        tuple(value["entity_ids"]), tuple(value["allowed_lifecycle_statuses"]),
        value["allow_sensitive"], value["allow_unclassified_sensitivity"],
        value["include_previous_versions"], value["checked_relation_expansion_intent"],
        tuple(value["relation_expansion_types"]), value["source_evidence_intent"],
        value["unresolved_time"], value["clarification_required"],
    )


def _parse_claim(value: Mapping[str, object]) -> EvidenceClaim:
    anchors = tuple(EvidenceAnchor(**item) for item in value["retrieval_anchors"])
    relations = tuple(CheckedRelationReference(**item) for item in value["checked_relations"])
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


def _parse_source(value: Mapping[str, object]) -> EvidenceSource:
    spans = tuple(EvidenceSpan(**item) for item in value["evidence_spans"])
    return EvidenceSource(
        value["user_id"], value["source_id"], value["source_type"], value["session_id"],
        _datetime(value["produced_at"]), _datetime(value["ingested_at"]), value["content_hash"], spans,
    )


def _parse_failure(value: Mapping[str, object]) -> EvidencePackageFailure:
    return EvidencePackageFailure(
        value["failure_id"], value["user_id"], value["query_id"], value["baseline_id"],
        value["code"], value["location"],
    )


def _parse_valid_time(value: object) -> RequestedValidTime | None:
    if value is None:
        return None
    return RequestedValidTime(
        value["kind"], _date(value["point_date"]), _datetime_or_none(value["point_timestamp"]),
        _date(value["range_start_date"]), _date(value["range_end_date"]),
        _datetime_or_none(value["range_start_timestamp"]), _datetime_or_none(value["range_end_timestamp"]),
    )


def _valid_time_value(value: RequestedValidTime | None) -> object:
    if value is None:
        return None
    return {name: getattr(value, name) for name in RequestedValidTime.__dataclass_fields__}


def _load_dataset_manifest(root: Path) -> Mapping[str, object]:
    manifest = _read_object(root / DATASET_MANIFEST)
    if (
        manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("allowed_users") != list(ALLOWED_USERS)
        or manifest.get("runtime_boundary") != "phase_7_runtime_only_no_relevance_gold_or_scorecard"
        or manifest.get("predecessor_drift") != []
    ):
        raise EvidencePackageEvaluationError("dataset manifest identity changed")
    for binding in manifest.get("inputs", {}).values():
        if not isinstance(binding, dict):
            raise EvidencePackageEvaluationError("input binding is invalid")
        for key, value in binding.items():
            if key == "path" or key.endswith("_path"):
                hash_key = "sha256" if key == "path" else key.removesuffix("_path") + "_sha256"
                if hash_key not in binding or _sha(root / str(value)) != binding[hash_key]:
                    raise EvidencePackageEvaluationError("bound runtime input changed")
    return manifest


def _failure(case_id: str, user_id: str, query_id: str, baseline_id: str, error: Exception) -> EvidencePackageFailure:
    code, location = "invariant_violation", "package"
    if isinstance(error, EvidencePackageRepositoryError):
        code, location = error.code, error.location
    return EvidencePackageFailure(
        stable_sha256({
            "case_id": case_id, "query_id": query_id, "baseline_id": baseline_id,
            "code": code, "location": location,
        }),
        user_id, query_id, baseline_id, code, location,
    )


def _case_id(query_id: str) -> str:
    return f"baseline_case_{query_id.rsplit('_', 1)[-1]}"


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(value) for value in values)


def _findings(checks: EvidencePackageChecks) -> str:
    return (
        "# Evidence package development findings\n\n"
        f"All {checks.package_count} frozen retrieval results produced a package with no runtime failures. "
        "Each retrieved claim version has a validated source and span path.\n\n"
        "All 33 development claims are still candidates. The package builder keeps them out of factual "
        "claim categories, so all 24 packages set `answer_allowed=false`. This result is expected. The "
        "release checks package structure and provenance, not answer correctness.\n\n"
        "No model or provider was used.\n"
    )


def _verify_protected(root: Path) -> None:
    for path, expected in PROTECTED_AUTHORITIES.items():
        if _sha(root / path) != expected:
            raise EvidencePackageEvaluationError("protected authority changed")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise EvidencePackageEvaluationError("result directory must be empty")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise EvidencePackageEvaluationError("JSONL record is invalid")
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise EvidencePackageEvaluationError("JSON object is invalid")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise EvidencePackageEvaluationError("datetime is invalid")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _datetime_or_none(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidencePackageEvaluationError("date is invalid")
    return date.fromisoformat(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
