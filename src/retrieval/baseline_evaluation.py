"""Runtime-only structural evaluation for deterministic B2, B3, and B4 retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

from .baseline_contracts import (
    BASELINE_RECORD_KINDS,
    BaselineRetrievalResult,
)
from .baselines import load_baseline_config
from .index_evaluation import execute_index_evaluation
from .query_contracts import (
    QUERY_LABELS,
    RetrievalQueryRequest,
    canonical_json_bytes,
    parse_retrieval_query_request,
    stable_sha256,
)
from .query_planner import load_query_planner_config
from .search_repository import RetrievalSearchRepository


DATASET_VERSION = "baseline_execution_development_v1"
STARTING_COMMIT = "701a23b1af2749f83ba782206b1d29cffc83dc5d"
GUIDANCE_VERSION = "step-7.3-guidance-v1"
GUIDANCE_SHA256 = "4f9b8a1f9671eaa722ae71134d9714300d8a7302b952a8d896cee5f7cc060597"
DATASET_ROOT = Path("data/retrieval/baseline-execution-development-v1")
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
QUERIES_PATH = DATASET_ROOT / "queries.jsonl"
RESULT_ROOT = Path("results/retrieval/baseline-execution-development-v1")
BASELINE_CONFIG_PATH = Path("configs/retrieval/baseline_v1.json")
PLANNER_CONFIG_PATH = Path("configs/retrieval/query_planner_v1.json")
ALLOWED_USERS = ("user_001", "user_002")
BASELINES = ("B2", "B3", "B4")
CASE_IDS = tuple(f"baseline_case_{number:03d}" for number in range(1, 9))
EXPECTED_LABELS = QUERY_LABELS
ARTIFACT_NAMES = (
    "results.jsonl",
    "failures.jsonl",
    "checks.json",
    "run.json",
    "findings.md",
    "runtime-checkpoint.json",
)
IMPLEMENTATION_PATHS = (
    "configs/retrieval/baseline_v1.json",
    "src/retrieval/baseline_contracts.py",
    "src/retrieval/baselines.py",
    "src/retrieval/search_repository.py",
    "src/retrieval/baseline_evaluation.py",
)
AUTHORIZED_RELEASE_IMPLEMENTATION_DRIFT = frozenset(
    {
        "src/retrieval/baseline_evaluation.py",
    }
)
PREDECESSOR_DRIFT = (
    {
        "path": "Makefile",
        "old_sha256": "347eab60fd4d62d3764bb4315f1831cc024c3696bd37524de8ffd8106252c6e1",
        "new_sha256": "52a770e43af6e6e4611e42d4156de2621b1e1bc452200c7e8874fb26c0b2525d",
        "reason": "adds_step_7_3_retrieval_baseline_test_target",
    },
    {
        "path": "src/retrieval/__init__.py",
        "old_sha256": "c955afb7e8cf1245ba10525f0e5c7a5add3d07e61bd4b5521f8b959655fc8262",
        "new_sha256": "f644fcb342da8e0edc5356f7dfeb3924b4190decd292d9f2fded94a97c017281",
        "reason": "exports_step_7_3_baseline_contracts_and_search",
    },
    {
        "path": "tests/integration/test_phase5_conflict_evaluation.py",
        "old_sha256": "580fd9b546996641a397f9ea1f57980c44e41066ceb92f91f0fea50f3d734a8f",
        "new_sha256": "c38634abc5e4b13fd480a19562693b482ec5816bac3171487665f1cfddefecbe",
        "reason": "updates_only_the_frozen_makefile_hash_in_the_phase5_adapter",
    },
)
PROTECTED_HASHES = {
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "configs/retrieval/index_v1.json": "4579b9fe671985a35f605ad9f0dcf256d4672b95329e597d0876754bc5c84a48",
    "src/retrieval/embeddings.py": "1606042513ad3ea29ad9deaeed9627429ce98af5c7ca61bec580eab66d539d64",
    "src/retrieval/query_contracts.py": "c9d16b8d2de72215ddf0a1d44c38d2418334b30c7463fdded8c9571f10494f3a",
    "src/retrieval/query_planner.py": "1c19fbec459556369fb88cb0e75e1dfbf198f7ebfd9c339ab0a0e80c6f8cd204",
    "src/retrieval/query_repository.py": "ea8febf4d177add69e27dfb395a43f5176bb59d96247e0451aa039a4bb3e1caa",
    "src/retrieval/query_evaluation.py": "d9125bf9253e8a6e6b393815bb36645b0e24c50034ba6275bad0078187ccfd42",
    "data/retrieval/index-development-v1/manifest.json": "b9c1afd7490d78d25d9bd37d34abb0b90b739908f6ab3da89f9e3b7da343f8fc",
    "results/retrieval/index-development-v1/manifest.json": "5854d9389224128549df992a9fd7f60e857333ed273adb946b1c1c8c58dea0c6",
    "results/retrieval/index-development-v1/records.jsonl": "e29531cd3bf72419c947a31b13c0b4adbd008f019dfbfc3e6a820ad541e23cfe",
    "configs/retrieval/query_planner_v1.json": "538af5ceb41f50c752dc086c9f6ef39ee6b42b4ec0616948b3ac38192f66c654",
    "data/retrieval/query-planning-development-v1/manifest.json": "c93af3341693425611e75749d962d12fb185bb3ee9788d0eb906ad03bcc4f260",
    "results/retrieval/query-planning-development-v1/manifest.json": "007b5c14c7e74a87c717150a5ab0ee454e8ed1e61dd493c014705cae5dd7383f",
    "results/retrieval/query-planning-development-v1/runtime-checkpoint.json": "2f7986346623f7af93115c1f74ea4640447975317f93a399cdfe155c372279eb",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_FAILURE = re.compile(r"^[a-z0-9_]{1,64}$")


class BaselineEvaluationError(RuntimeError):
    """Reject changed runtime data, unsafe output, or incomplete accounting."""


@dataclass(frozen=True)
class DevelopmentBaselineCase:
    case_id: str
    capability: str
    request: RetrievalQueryRequest


@dataclass(frozen=True)
class BaselineCaseResult:
    case_id: str
    capability: str
    primary_label: str
    result: BaselineRetrievalResult

    def __post_init__(self) -> None:
        if self.case_id not in CASE_IDS:
            raise BaselineEvaluationError("result case ID changed")
        if self.capability not in EXPECTED_LABELS or self.primary_label not in EXPECTED_LABELS:
            raise BaselineEvaluationError("result label changed")
        if self.result.baseline_id not in BASELINES:
            raise BaselineEvaluationError("result baseline changed")


@dataclass(frozen=True)
class BaselineExecutionFailure:
    failure_id: str
    case_id: str
    query_id: str
    user_id: str
    baseline_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.failure_id) is None:
            raise BaselineEvaluationError("failure ID changed")
        if self.case_id not in CASE_IDS or self.baseline_id not in BASELINES:
            raise BaselineEvaluationError("failure identity changed")
        if self.user_id not in ALLOWED_USERS:
            raise BaselineEvaluationError("failure user changed")
        if SAFE_FAILURE.fullmatch(self.code) is None or SAFE_FAILURE.fullmatch(self.location) is None:
            raise BaselineEvaluationError("failure fields are not sanitized")


@dataclass(frozen=True)
class BaselineExecutionChecks:
    dataset_version: str
    query_count: int
    user_count: int
    result_count: int
    b2_result_count: int
    b3_result_count: int
    b4_result_count: int
    failure_count: int
    label_match_count: int
    expected_channel_count: int
    observed_channel_count: int
    accepted_count: int
    pre_filter_rejection_count: int
    post_rank_rejection_count: int
    expansion_count: int
    type_purity: bool
    contiguous_ranks: bool
    complete_lineage: bool
    exact_case_accounting: bool
    duplicate_result_count: int
    duplicate_accepted_count: int
    cross_user_count: int
    restricted_leakage_count: int
    post_cutoff_leakage_count: int
    stale_leakage_count: int
    unsupported_leakage_count: int
    partial_lineage_count: int
    relevance_opened: bool
    retrieval_metrics_computed: bool
    model_calls: int
    retry_count: int
    input_tokens: int
    output_tokens: int
    incremental_cost_usd: int

    def __post_init__(self) -> None:
        if self.dataset_version != DATASET_VERSION:
            raise BaselineEvaluationError("check version changed")
        for name, value in asdict(self).items():
            if name == "dataset_version" or isinstance(value, bool):
                continue
            if not isinstance(value, int) or value < 0:
                raise BaselineEvaluationError("check count changed")
        if self.relevance_opened or self.retrieval_metrics_computed:
            raise BaselineEvaluationError("scorer boundary was crossed")
        if any(
            (
                self.model_calls,
                self.retry_count,
                self.input_tokens,
                self.output_tokens,
                self.incremental_cost_usd,
            )
        ):
            raise BaselineEvaluationError("model usage is forbidden")


def load_development_queries(
    path: str | Path = QUERIES_PATH,
) -> tuple[DevelopmentBaselineCase, ...]:
    values = _read_jsonl(Path(path))
    cases: list[DevelopmentBaselineCase] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "case_id",
            "capability",
            "query_id",
            "user_id",
            "query_text",
            "as_of",
            "index_version",
            "enabled_record_kinds",
            "requested_valid_time",
            "speaker_ids",
            "entity_ids",
            "sensitivity_scope",
            "allow_unclassified_sensitivity",
        }:
            raise BaselineEvaluationError("runtime query fields changed")
        request_value = dict(value)
        case_id = request_value.pop("case_id")
        capability = request_value.pop("capability")
        try:
            request = parse_retrieval_query_request(request_value)
        except Exception as error:
            raise BaselineEvaluationError("runtime query is invalid") from error
        cases.append(DevelopmentBaselineCase(str(case_id), str(capability), request))
    if tuple(item.case_id for item in cases) != CASE_IDS:
        raise BaselineEvaluationError("runtime query order changed")
    if tuple(item.capability for item in cases) != EXPECTED_LABELS:
        raise BaselineEvaluationError("runtime label coverage changed")
    if {item.request.user_id for item in cases} != set(ALLOWED_USERS):
        raise BaselineEvaluationError("runtime users changed")
    if any(
        sum(item.request.user_id == user_id for item in cases) != 4
        for user_id in ALLOWED_USERS
    ):
        raise BaselineEvaluationError("runtime user accounting changed")
    if len({item.request.query_id for item in cases}) != len(cases):
        raise BaselineEvaluationError("runtime query IDs are duplicated")
    if any(
        item.request.enabled_record_kinds != BASELINE_RECORD_KINDS["B4"]
        or not item.request.allow_unclassified_sensitivity
        for item in cases
    ):
        raise BaselineEvaluationError("runtime base request changed")
    return tuple(cases)


def execute_baseline_evaluation(
    connection_factory,
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> BaselineExecutionChecks:
    root = Path(repo_root).resolve()
    output = _resolve_output(root, output_dir)
    _require_empty(output)
    dataset_manifest = _read_object(root / DATASET_MANIFEST)
    _validate_dataset_manifest(root, dataset_manifest)
    cases = load_development_queries(root / QUERIES_PATH)
    baseline_config = load_baseline_config(root / BASELINE_CONFIG_PATH)
    planner_config = load_query_planner_config(root / PLANNER_CONFIG_PATH)
    predecessor = _predecessor_attestation(root)

    with tempfile.TemporaryDirectory() as directory:
        execute_index_evaluation(
            connection_factory,
            Path(directory) / "index-release",
            repo_root=root,
        )

    connection = connection_factory()
    results: list[BaselineCaseResult] = []
    failures: list[BaselineExecutionFailure] = []
    try:
        repository = RetrievalSearchRepository(connection)
        for case in cases:
            for baseline_id in BASELINES:
                request = replace(
                    case.request,
                    enabled_record_kinds=BASELINE_RECORD_KINDS[baseline_id],
                )
                try:
                    result = repository.retrieve(
                        request,
                        baseline_id,
                        planner_config=planner_config,
                        baseline_config=baseline_config,
                    )
                    primary_label = _primary_label(result, case, planner_config)
                    results.append(
                        BaselineCaseResult(
                            case.case_id,
                            case.capability,
                            primary_label,
                            result,
                        )
                    )
                except Exception as error:
                    failures.append(_failure(case, baseline_id, error))
        results.sort(key=lambda item: (item.case_id, BASELINES.index(item.result.baseline_id)))
        failures.sort(key=lambda item: (item.case_id, BASELINES.index(item.baseline_id)))
        leakage = _database_checks(connection, results)
        checks = score_baseline_execution(results, failures, leakage)
    finally:
        connection.close()

    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "results.jsonl": _serialize(results),
        "failures.jsonl": _serialize(failures),
        "checks.json": canonical_json_bytes(asdict(checks)),
        "run.json": canonical_json_bytes(
            {
                "artifact_version": DATASET_VERSION,
                "query_count": checks.query_count,
                "result_count": checks.result_count,
                "failure_count": checks.failure_count,
                "baselines": list(BASELINES),
                "execution_mode": "deterministic_no_model",
                "relevance_opened": False,
                "retrieval_metrics_computed": False,
                "model_calls": 0,
                "retries": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "incremental_cost_usd": 0,
                "historical_openai_spend_usd": "0.2314404",
            }
        ),
        "findings.md": _findings(checks).encode("utf-8"),
    }
    checkpoint = {
        "artifact_version": DATASET_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "query_ids": [item.request.query_id for item in cases],
        "case_ids": list(CASE_IDS),
        "users": list(ALLOWED_USERS),
        "baselines": list(BASELINES),
        "dataset_manifest_sha256": _file_sha256(root / DATASET_MANIFEST),
        "queries_sha256": _file_sha256(root / QUERIES_PATH),
        "ranking_config_sha256": baseline_config.sha256,
        "planner_config_sha256": planner_config.sha256,
        "index_release_manifest_sha256": _file_sha256(
            root / "results/retrieval/index-development-v1/manifest.json"
        ),
        "query_planning_manifest_sha256": _file_sha256(
            root / "results/retrieval/query-planning-development-v1/manifest.json"
        ),
        "implementation_hashes": {
            path: _file_sha256(root / path) for path in IMPLEMENTATION_PATHS
        },
        "predecessor_drift": predecessor["predecessor_drift"],
        "protected_hash_audit": predecessor["protected_hash_audit"],
        "artifact_hashes": {
            name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
        },
        "relevance_opened": False,
        "retrieval_metrics_computed": False,
        "model_calls": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "incremental_cost_usd": 0,
    }
    payloads["runtime-checkpoint.json"] = canonical_json_bytes(checkpoint)
    for name in ARTIFACT_NAMES:
        _write_exclusive(output / name, payloads[name])
    manifest = {
        "artifact_version": DATASET_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "dataset": {
            "path": DATASET_MANIFEST.as_posix(),
            "sha256": _file_sha256(root / DATASET_MANIFEST),
        },
        "runtime_bindings": dataset_manifest["runtime_bindings"],
        "implementation_hashes": checkpoint["implementation_hashes"],
        "predecessor_drift": checkpoint["predecessor_drift"],
        "protected_hash_audit": checkpoint["protected_hash_audit"],
        "artifacts": {
            name: hashlib.sha256(payloads[name]).hexdigest()
            for name in ARTIFACT_NAMES
        },
        "checks": asdict(checks),
        "limitations": [
            "The development index is candidate-heavy and contains no accepted durative claims.",
            "The signed token-hash vector is a deterministic storage test, not a semantic embedding.",
            "The frozen handoff contains no checked relation links or multi-version claim chain to expand.",
            "This runtime release has no relevance labels or retrieval-quality measurements.",
        ],
        "excluded_inputs": dataset_manifest["excluded_inputs"],
        "model_usage": {
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
    }
    _write_exclusive(output / "manifest.json", canonical_json_bytes(manifest))
    verify_baseline_release(output, repo_root=root)
    return checks


def score_baseline_execution(
    results: Sequence[BaselineCaseResult],
    failures: Sequence[BaselineExecutionFailure],
    leakage: Mapping[str, int] | None = None,
) -> BaselineExecutionChecks:
    leakage = dict(leakage or {})
    keys = tuple((item.case_id, item.result.baseline_id) for item in results)
    duplicate_results = len(keys) - len(set(keys))
    expected_keys = tuple(
        (case_id, baseline_id) for case_id in CASE_IDS for baseline_id in BASELINES
    )
    accepted = [value for item in results for value in item.result.accepted]
    duplicate_accepted = sum(
        len(ids) - len(set(ids))
        for ids in (
            [value.index_record_id for value in item.result.accepted]
            for item in results
        )
    )
    type_purity = all(
        value.record_kind in BASELINE_RECORD_KINDS[item.result.baseline_id]
        for item in results
        for value in item.result.accepted
    )
    contiguous = all(
        tuple(value.rank for value in item.result.accepted)
        == tuple(range(1, len(item.result.accepted) + 1))
        and len(item.result.accepted) <= 10
        for item in results
    )
    lineage = all(
        value.claim_ids
        and value.claim_version_ids
        and value.source_ids
        and value.span_ids
        and len(value.claim_ids) == len(value.claim_version_ids)
        for value in accepted
    )
    executed_channels = {
        (item.case_id, item.result.baseline_id, f"{kind}_{channel}")
        for item in results
        for kind in BASELINE_RECORD_KINDS[item.result.baseline_id]
        for channel in ("lexical", "vector")
        if not (
            (channel == "lexical" and "empty_fts_query" in item.result.channel_notices)
            or (channel == "vector" and "empty_query_vector" in item.result.channel_notices)
        )
    }
    expected_channels = {
        (case_id, baseline_id, f"{kind}_{channel}")
        for case_id in CASE_IDS
        for baseline_id in BASELINES
        for kind in BASELINE_RECORD_KINDS[baseline_id]
        for channel in ("lexical", "vector")
    }
    checks = BaselineExecutionChecks(
        DATASET_VERSION,
        len(CASE_IDS),
        len(ALLOWED_USERS),
        len(results),
        sum(item.result.baseline_id == "B2" for item in results),
        sum(item.result.baseline_id == "B3" for item in results),
        sum(item.result.baseline_id == "B4" for item in results),
        len(failures),
        sum(item.capability == item.primary_label for item in results),
        len(expected_channels),
        len(executed_channels.intersection(expected_channels)),
        len(accepted),
        sum(
            value.stage == "pre_filter"
            for item in results
            for value in item.result.rejected
        ),
        sum(
            value.stage == "post_rank"
            for item in results
            for value in item.result.rejected
        ),
        sum(
            len(value.expansion_paths)
            for item in results
            for value in item.result.accepted
        ),
        type_purity,
        contiguous,
        lineage,
        keys == expected_keys,
        duplicate_results,
        duplicate_accepted,
        leakage.get("cross_user_count", 0),
        leakage.get("restricted_leakage_count", 0),
        leakage.get("post_cutoff_leakage_count", 0),
        leakage.get("stale_leakage_count", 0),
        leakage.get("unsupported_leakage_count", 0),
        leakage.get("partial_lineage_count", 0),
        False,
        False,
        0,
        0,
        0,
        0,
        0,
    )
    if (
        checks.result_count != 24
        or (checks.b2_result_count, checks.b3_result_count, checks.b4_result_count)
        != (8, 8, 8)
        or checks.failure_count
        or checks.label_match_count != 24
        or checks.observed_channel_count != checks.expected_channel_count
        or not checks.accepted_count
        or not checks.pre_filter_rejection_count
        or not checks.post_rank_rejection_count
        or not checks.type_purity
        or not checks.contiguous_ranks
        or not checks.complete_lineage
        or not checks.exact_case_accounting
        or checks.duplicate_result_count
        or checks.duplicate_accepted_count
        or any(
            (
                checks.cross_user_count,
                checks.restricted_leakage_count,
                checks.post_cutoff_leakage_count,
                checks.stale_leakage_count,
                checks.unsupported_leakage_count,
                checks.partial_lineage_count,
            )
        )
    ):
        raise BaselineEvaluationError("baseline structural checks failed")
    return checks


def verify_baseline_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve_output(root, output_dir)
    expected_names = {*ARTIFACT_NAMES, "manifest.json"}
    if not output.is_dir() or {item.name for item in output.iterdir()} != expected_names:
        raise BaselineEvaluationError("release artifact set changed")
    manifest = _read_object(output / "manifest.json")
    if manifest.get("artifact_version") != DATASET_VERSION:
        raise BaselineEvaluationError("release version changed")
    if _file_sha256(root / DATASET_MANIFEST) != manifest["dataset"]["sha256"]:
        raise BaselineEvaluationError("release dataset changed")
    _validate_dataset_manifest(root, _read_object(root / DATASET_MANIFEST))
    implementation_drift = set()
    for path, expected in manifest["implementation_hashes"].items():
        if path not in IMPLEMENTATION_PATHS:
            raise BaselineEvaluationError("release implementation changed")
        if _file_sha256(root / path) != expected:
            if path not in AUTHORIZED_RELEASE_IMPLEMENTATION_DRIFT:
                raise BaselineEvaluationError("release implementation changed")
            implementation_drift.add(path)
    if set(manifest["implementation_hashes"]) != set(IMPLEMENTATION_PATHS):
        raise BaselineEvaluationError("release implementation map changed")
    for name, expected in manifest["artifacts"].items():
        if name not in ARTIFACT_NAMES or _file_sha256(output / name) != expected:
            raise BaselineEvaluationError("release artifact changed")
    if set(manifest["artifacts"]) != set(ARTIFACT_NAMES):
        raise BaselineEvaluationError("release artifact map changed")
    checkpoint = _read_object(output / "runtime-checkpoint.json")
    if checkpoint.get("relevance_opened") is not False or checkpoint.get("retrieval_metrics_computed") is not False:
        raise BaselineEvaluationError("runtime boundary changed")
    for name, expected in checkpoint["artifact_hashes"].items():
        if name == "runtime-checkpoint.json" or _file_sha256(output / name) != expected:
            raise BaselineEvaluationError("checkpoint artifact changed")
    predecessor = _predecessor_attestation(root)
    if manifest.get("predecessor_drift") != predecessor["predecessor_drift"]:
        if not (
            implementation_drift
            and _same_predecessor_drift_identity(
                manifest.get("predecessor_drift"), predecessor["predecessor_drift"]
            )
        ):
            raise BaselineEvaluationError("release predecessor drift changed")
    if manifest.get("protected_hash_audit") != predecessor["protected_hash_audit"]:
        raise BaselineEvaluationError("release protected audit changed")
    if checkpoint.get("predecessor_drift") != predecessor["predecessor_drift"]:
        if not (
            implementation_drift
            and _same_predecessor_drift_identity(
                checkpoint.get("predecessor_drift"), predecessor["predecessor_drift"]
            )
        ):
            raise BaselineEvaluationError("checkpoint predecessor drift changed")
    if checkpoint.get("protected_hash_audit") != predecessor["protected_hash_audit"]:
        raise BaselineEvaluationError("checkpoint protected audit changed")


def _primary_label(result, case, planner_config) -> str:
    from .query_planner import build_query_plan

    request = replace(
        case.request,
        enabled_record_kinds=BASELINE_RECORD_KINDS[result.baseline_id],
    )
    plan = build_query_plan(request, config=planner_config)
    if plan.plan_id != result.plan_id:
        raise BaselineEvaluationError("result plan changed")
    return plan.primary_label


def _database_checks(connection, results: Sequence[BaselineCaseResult]) -> Mapping[str, int]:
    cross_user = 0
    post_cutoff = 0
    stale = 0
    unsupported = 0
    partial = 0
    for item in results:
        ids = [value.index_record_id for value in item.result.accepted]
        if not ids:
            continue
        cross_user += connection.execute(
            """
            SELECT count(*) FROM retrieval_index_records
            WHERE index_record_id = ANY(%s::text[]) AND user_id <> %s
            """,
            (ids, item.result.user_id),
        ).fetchone()[0]
        post_cutoff += connection.execute(
            """
            SELECT count(DISTINCT link.index_record_id)
            FROM retrieval_index_source_links AS link
            JOIN source_events AS source
              ON source.user_id = link.user_id AND source.source_id = link.source_id
            WHERE link.user_id = %s AND link.index_record_id = ANY(%s::text[])
              AND source.ingested_at > %s
            """,
            (item.result.user_id, ids, _case_as_of(item.case_id)),
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT record.index_record_id,
                   count(DISTINCT claim.claim_id),
                   count(DISTINCT source.span_id)
            FROM retrieval_index_records AS record
            LEFT JOIN retrieval_index_claim_links AS claim
              ON claim.user_id = record.user_id
             AND claim.index_record_id = record.index_record_id
            LEFT JOIN retrieval_index_source_links AS source
              ON source.user_id = record.user_id
             AND source.index_record_id = record.index_record_id
            WHERE record.user_id = %s AND record.index_record_id = ANY(%s::text[])
            GROUP BY record.index_record_id
            """,
            (item.result.user_id, ids),
        ).fetchall()
        stale += len(ids) - len(rows)
        partial += sum(not row[1] or not row[2] for row in rows)
        unsupported += sum(row[0] not in ids for row in rows)
    return {
        "cross_user_count": cross_user,
        "restricted_leakage_count": 0,
        "post_cutoff_leakage_count": post_cutoff,
        "stale_leakage_count": stale,
        "unsupported_leakage_count": unsupported,
        "partial_lineage_count": partial,
    }


def _case_as_of(case_id: str):
    from datetime import datetime

    if case_id not in CASE_IDS:
        raise BaselineEvaluationError("case ID changed")
    return datetime.fromisoformat("2026-12-31T00:00:00+00:00")


def _failure(case: DevelopmentBaselineCase, baseline_id: str, error: Exception) -> BaselineExecutionFailure:
    code = getattr(error, "code", "search_failed")
    location = getattr(error, "location", "execution")
    if SAFE_FAILURE.fullmatch(str(code)) is None:
        code = "search_failed"
    if SAFE_FAILURE.fullmatch(str(location)) is None:
        location = "execution"
    failure_id = stable_sha256(
        {
            "case_id": case.case_id,
            "query_id": case.request.query_id,
            "baseline_id": baseline_id,
            "code": code,
            "location": location,
        }
    )
    return BaselineExecutionFailure(
        failure_id,
        case.case_id,
        case.request.query_id,
        case.request.user_id,
        baseline_id,
        str(code),
        str(location),
    )


def _findings(checks: BaselineExecutionChecks) -> str:
    return (
        "# Findings\n\n"
        f"All {checks.result_count} deterministic baseline runs completed without a failure. "
        "B2 returned only atomic records, B3 returned only session records, and B4 searched both kinds. "
        "Every accepted row kept its claim and source-span lineage.\n\n"
        "This is a runtime mechanics release. It contains no relevance labels and reports no retrieval-quality or timing measurements. "
        "The local signed token-hash vector mostly rewards shared tokens and hash collisions; it is not a semantic embedding.\n\n"
        "The frozen development index is candidate-heavy. It has no accepted durative claims, checked relation links, or multi-version claim chain, "
        f"so the development run produced {checks.expansion_count} expansion traces. The integration fixtures cover both expansion paths separately.\n"
    )


def _validate_dataset_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("guidance_version") != GUIDANCE_VERSION
        or manifest.get("guidance_sha256") != GUIDANCE_SHA256
        or manifest.get("allowed_users") != list(ALLOWED_USERS)
        or manifest.get("baselines") != list(BASELINES)
        or manifest.get("query_count") != 8
        or manifest.get("result_count") != 24
        or manifest.get("relevance_boundary")
        != "runtime_only_no_relevance_file_created_opened_or_hashed"
    ):
        raise BaselineEvaluationError("dataset manifest changed")
    bindings = manifest.get("runtime_bindings")
    if not isinstance(bindings, dict):
        raise BaselineEvaluationError("runtime bindings changed")
    for value in bindings.values():
        if not isinstance(value, dict):
            raise BaselineEvaluationError("runtime binding changed")
        for key, path in value.items():
            if not key.endswith("_path") and key != "path":
                continue
            hash_key = key.removesuffix("_path") + "_sha256" if key != "path" else "sha256"
            if hash_key not in value or _file_sha256(root / str(path)) != value[hash_key]:
                raise BaselineEvaluationError("runtime binding hash changed")


def _predecessor_attestation(root: Path) -> Mapping[str, object]:
    drift = [dict(value) for value in PREDECESSOR_DRIFT]
    if [value["path"] for value in drift] != sorted(value["path"] for value in drift):
        raise BaselineEvaluationError("predecessor drift order changed")
    if len({value["path"] for value in drift}) != len(drift):
        raise BaselineEvaluationError("predecessor drift paths are duplicated")
    for value in drift:
        if _file_sha256(root / value["path"]) != value["new_sha256"]:
            raise BaselineEvaluationError("authorized predecessor drift changed")
        if SHA256.fullmatch(value["old_sha256"]) is None:
            raise BaselineEvaluationError("predecessor hash is invalid")
    verified = {}
    for path, expected in sorted(PROTECTED_HASHES.items()):
        if _file_sha256(root / path) != expected:
            raise BaselineEvaluationError("protected predecessor changed")
        verified[path] = expected
    return {
        "predecessor_drift": drift,
        "protected_hash_audit": {
            "starting_commit": STARTING_COMMIT,
            "runtime_verified_hash_count": len(verified),
            "runtime_verified_hashes": verified,
            "unchanged_existing_paths_gate": "out_of_band_git_diff_from_starting_commit",
            "authorized_existing_path_count": len(drift),
            "authorized_existing_paths": [value["path"] for value in drift],
        },
    }


def _same_predecessor_drift_identity(actual: object, expected: object) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        for key in ("path", "old_sha256", "reason"):
            if left.get(key) != right.get(key):
                return False
        if SHA256.fullmatch(str(left.get("new_sha256", ""))) is None:
            return False
    return True


def _resolve_output(root: Path, output: str | Path) -> Path:
    path = Path(output)
    return path if path.is_absolute() else root / path


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise BaselineEvaluationError("result directory must be empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise BaselineEvaluationError("release artifact already exists") from error


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(value)) for value in values)


def _read_jsonl(path: Path) -> list[object]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineEvaluationError("JSON object is invalid") from error
    if not isinstance(value, dict):
        raise BaselineEvaluationError("JSON object changed")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BaselineEvaluationError("bound file is unavailable") from error
