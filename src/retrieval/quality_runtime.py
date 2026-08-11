"""Freeze warm local retrieval latency before relevance data exists."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Mapping, Sequence

from .baseline_contracts import BASELINE_RECORD_KINDS, BaselineRetrievalResult
from .baseline_evaluation import load_development_queries
from .baselines import load_baseline_config
from .index_evaluation import execute_index_evaluation
from .query_contracts import canonical_json_bytes
from .query_planner import load_query_planner_config
from .search_repository import RetrievalSearchRepository


RUNTIME_VERSION = "retrieval_quality_runtime_v1"
EVALUATION_VERSION = "retrieval_quality_development_v1"
STARTING_COMMIT = "d849ea1140f97066edb408acd8704268655c7abe"
GUIDANCE_VERSION = "step-7.4-guidance-v1"
GUIDANCE_SHA256 = "17d8609f5e0799661ea4a7d6b3a1de3493d267a6cff90efa4d21e17de4df96ae"
CONFIG_PATH = Path("configs/retrieval/quality_evaluation_v1.json")
RUNTIME_MANIFEST_PATH = Path(
    "data/retrieval/retrieval-quality-development-v1/runtime/manifest.json"
)
OUTPUT_ROOT = Path("results/retrieval/retrieval-quality-development-runtime-v1")
BASELINE_RESULTS_PATH = Path(
    "results/retrieval/baseline-execution-development-v1/results.jsonl"
)
BASELINES = ("B2", "B3", "B4")
ALLOWED_USERS = ("user_001", "user_002")
ARTIFACTS = (
    "checkpoint_preflight.json",
    "latency-samples.jsonl",
    "failures.jsonl",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_FAILURE = re.compile(r"^[a-z0-9_]{1,64}$")


class QualityRuntimeError(RuntimeError):
    """Reject an unsafe or incomplete latency checkpoint."""


@dataclass(frozen=True)
class QualityRuntimeConfig:
    quality_evaluation_version: str
    runtime_version: str
    runtime_release: str
    index_version: str
    planner_version: str
    ranking_config_version: str
    baselines: tuple[str, ...]
    query_count: int
    result_count: int
    warmups_per_result: int
    latency_trials_per_result: int
    latency_sample_count: int
    timer: str
    trial_order: str
    result_digest: str
    relevance_grades: Mapping[str, int]
    cutoffs: tuple[int, ...]
    quality_decimal_places: int
    latency_decimal_places: int
    quality_rounding: str
    latency_percentile: str
    model_usage: str
    sha256: str


@dataclass(frozen=True)
class LatencySample:
    case_id: str
    query_id: str
    user_id: str
    baseline_id: str
    trial: int
    elapsed_ns: int
    result_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id.startswith("baseline_case_"):
            raise QualityRuntimeError("sample case ID is invalid")
        if not self.query_id.startswith("baseline_query_"):
            raise QualityRuntimeError("sample query ID is invalid")
        if self.user_id not in ALLOWED_USERS or self.baseline_id not in BASELINES:
            raise QualityRuntimeError("sample ownership is invalid")
        if type(self.trial) is not int or not 1 <= self.trial <= 10:
            raise QualityRuntimeError("sample trial is invalid")
        if type(self.elapsed_ns) is not int or self.elapsed_ns <= 0:
            raise QualityRuntimeError("sample duration is invalid")
        if SHA256.fullmatch(self.result_sha256) is None:
            raise QualityRuntimeError("sample result hash is invalid")


@dataclass(frozen=True)
class RuntimeFailure:
    failure_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        if SHA256.fullmatch(self.failure_id) is None:
            raise QualityRuntimeError("failure ID is invalid")
        if SAFE_FAILURE.fullmatch(self.code) is None or SAFE_FAILURE.fullmatch(self.location) is None:
            raise QualityRuntimeError("failure fields are not sanitized")


def load_quality_runtime_config(
    path: str | Path = CONFIG_PATH,
) -> QualityRuntimeConfig:
    source = Path(path)
    value = _read_object(source)
    expected = {
        "quality_evaluation_version",
        "runtime_version",
        "runtime_release",
        "index_version",
        "planner_version",
        "ranking_config_version",
        "baselines",
        "query_count",
        "result_count",
        "warmups_per_result",
        "latency_trials_per_result",
        "latency_sample_count",
        "timer",
        "trial_order",
        "result_digest",
        "relevance_grades",
        "cutoffs",
        "quality_decimal_places",
        "latency_decimal_places",
        "quality_rounding",
        "latency_percentile",
        "model_usage",
    }
    if set(value) != expected:
        raise QualityRuntimeError("quality config fields changed")
    config = QualityRuntimeConfig(
        str(value["quality_evaluation_version"]),
        str(value["runtime_version"]),
        str(value["runtime_release"]),
        str(value["index_version"]),
        str(value["planner_version"]),
        str(value["ranking_config_version"]),
        _tuple(value["baselines"], "baselines"),
        _integer(value["query_count"], "query count"),
        _integer(value["result_count"], "result count"),
        _integer(value["warmups_per_result"], "warmup count"),
        _integer(value["latency_trials_per_result"], "trial count"),
        _integer(value["latency_sample_count"], "sample count"),
        str(value["timer"]),
        str(value["trial_order"]),
        str(value["result_digest"]),
        _grades(value["relevance_grades"]),
        tuple(_integer(item, "cutoff") for item in value["cutoffs"]),
        _integer(value["quality_decimal_places"], "quality precision"),
        _integer(value["latency_decimal_places"], "latency precision"),
        str(value["quality_rounding"]),
        str(value["latency_percentile"]),
        str(value["model_usage"]),
        _file_sha256(source),
    )
    if (
        config.quality_evaluation_version != EVALUATION_VERSION
        or config.runtime_version != RUNTIME_VERSION
        or config.runtime_release != "baseline_execution_development_v1"
        or config.index_version != "retrieval_index_v1"
        or config.planner_version != "retrieval_query_planner_v1"
        or config.ranking_config_version != "retrieval_baseline_v1"
        or config.baselines != BASELINES
        or (config.query_count, config.result_count) != (8, 24)
        or (config.warmups_per_result, config.latency_trials_per_result) != (1, 10)
        or config.latency_sample_count != 240
        or config.timer != "time.perf_counter_ns"
        or config.trial_order != "round_case_baseline"
        or config.result_digest != "canonical_json_sha256"
        or dict(config.relevance_grades) != {"irrelevant": 0, "supporting": 1, "direct": 2}
        or config.cutoffs != (5, 10)
        or (config.quality_decimal_places, config.latency_decimal_places) != (6, 3)
        or config.quality_rounding != "ROUND_HALF_EVEN"
        or config.latency_percentile != "nearest_rank"
        or config.model_usage != "deterministic_no_model"
    ):
        raise QualityRuntimeError("quality config values changed")
    return config


def execute_quality_runtime(
    connection_factory,
    output_dir: str | Path = OUTPUT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[LatencySample, ...]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    manifest = _load_runtime_manifest(root)
    config = load_quality_runtime_config(root / CONFIG_PATH)
    absent_paths = tuple(str(value) for value in manifest["required_absent_paths"])
    if any((root / value).exists() for value in absent_paths):
        raise QualityRuntimeError("checkpoint boundary is not clean")
    preflight = _preflight(root, absent_paths, config, manifest)
    try:
        expected = _load_expected_results(root / BASELINE_RESULTS_PATH)
        cases = load_development_queries(
            root / "data/retrieval/baseline-execution-development-v1/queries.jsonl"
        )
        if len(cases) != config.query_count:
            raise QualityRuntimeError("query count changed")
        with tempfile.TemporaryDirectory() as directory:
            execute_index_evaluation(
                connection_factory,
                Path(directory) / "index-release",
                repo_root=root,
            )
        connection = connection_factory()
        try:
            machine = _machine_metadata(connection)
            repository = RetrievalSearchRepository(connection)
            planner_config = load_query_planner_config(
                root / "configs/retrieval/query_planner_v1.json"
            )
            baseline_config = load_baseline_config(root / "configs/retrieval/baseline_v1.json")
            for case in cases:
                for baseline_id in BASELINES:
                    request = replace(
                        case.request,
                        enabled_record_kinds=BASELINE_RECORD_KINDS[baseline_id],
                    )
                    result = repository.retrieve(
                        request,
                        baseline_id,
                        planner_config=planner_config,
                        baseline_config=baseline_config,
                    )
                    _require_expected_result(case.case_id, baseline_id, result, expected)
            samples: list[LatencySample] = []
            for trial in range(1, config.latency_trials_per_result + 1):
                for case in cases:
                    for baseline_id in BASELINES:
                        request = replace(
                            case.request,
                            enabled_record_kinds=BASELINE_RECORD_KINDS[baseline_id],
                        )
                        started = time.perf_counter_ns()
                        result = repository.retrieve(
                            request,
                            baseline_id,
                            planner_config=planner_config,
                            baseline_config=baseline_config,
                        )
                        elapsed = time.perf_counter_ns() - started
                        digest = _require_expected_result(
                            case.case_id,
                            baseline_id,
                            result,
                            expected,
                        )
                        samples.append(
                            LatencySample(
                                case.case_id,
                                case.request.query_id,
                                case.request.user_id,
                                baseline_id,
                                trial,
                                elapsed,
                                digest,
                            )
                        )
        finally:
            connection.close()
        _validate_samples(samples, config, expected)
    except Exception as error:
        _write_failed_attempt(output, preflight, error)
        if isinstance(error, QualityRuntimeError):
            raise
        raise QualityRuntimeError("latency execution failed") from error

    payloads = {
        "checkpoint_preflight.json": canonical_json_bytes(preflight),
        "latency-samples.jsonl": _serialize(samples),
        "failures.jsonl": b"",
    }
    checkpoint = {
        "runtime_version": RUNTIME_VERSION,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "starting_commit": STARTING_COMMIT,
        "config_sha256": config.sha256,
        "runtime_manifest_sha256": _file_sha256(root / RUNTIME_MANIFEST_PATH),
        "runtime_module_sha256": _file_sha256(root / "src/retrieval/quality_runtime.py"),
        "runtime_bindings": manifest["runtime_bindings"],
        "artifacts": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
        "machine": machine,
        "query_count": 8,
        "result_count": 24,
        "warmup_count": 24,
        "latency_sample_count": len(samples),
        "result_digest_match_count": len(samples),
        "failure_count": 0,
        "trial_order": "round_case_baseline",
        "relevance_opened": False,
        "relevance_hashed": False,
        "retrieval_metrics_computed": False,
        "model_usage": {
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
            "historical_openai_spend_usd": "0.2314404",
        },
    }
    payloads["checkpoint_manifest.json"] = canonical_json_bytes(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    for name in (*ARTIFACTS, "checkpoint_manifest.json"):
        _write_exclusive(output / name, payloads[name])
    verify_quality_runtime_checkpoint(output, repo_root=root)
    return tuple(samples)


def verify_quality_runtime_checkpoint(
    output_dir: str | Path = OUTPUT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    expected_names = {*ARTIFACTS, "checkpoint_manifest.json"}
    if not output.is_dir() or {item.name for item in output.iterdir()} != expected_names:
        raise QualityRuntimeError("runtime checkpoint artifact set changed")
    checkpoint = _read_object(output / "checkpoint_manifest.json")
    if (
        checkpoint.get("runtime_version") != RUNTIME_VERSION
        or checkpoint.get("guidance_sha256") != GUIDANCE_SHA256
        or checkpoint.get("starting_commit") != STARTING_COMMIT
        or checkpoint.get("runtime_module_sha256")
        != _file_sha256(root / "src/retrieval/quality_runtime.py")
        or checkpoint.get("config_sha256") != _file_sha256(root / CONFIG_PATH)
        or checkpoint.get("runtime_manifest_sha256")
        != _file_sha256(root / RUNTIME_MANIFEST_PATH)
        or checkpoint.get("latency_sample_count") != 240
        or checkpoint.get("result_digest_match_count") != 240
        or checkpoint.get("failure_count") != 0
        or checkpoint.get("relevance_opened") is not False
        or checkpoint.get("relevance_hashed") is not False
        or checkpoint.get("retrieval_metrics_computed") is not False
    ):
        raise QualityRuntimeError("runtime checkpoint identity changed")
    for name in ARTIFACTS:
        if checkpoint["artifacts"].get(name) != _file_sha256(output / name):
            raise QualityRuntimeError("runtime checkpoint artifact changed")
    if set(checkpoint["artifacts"]) != set(ARTIFACTS):
        raise QualityRuntimeError("runtime checkpoint artifact map changed")
    if (output / "failures.jsonl").read_bytes():
        raise QualityRuntimeError("runtime checkpoint has failures")
    config = load_quality_runtime_config(root / CONFIG_PATH)
    samples = tuple(
        LatencySample(**value) for value in _read_jsonl(output / "latency-samples.jsonl")
    )
    expected = _load_expected_results(root / BASELINE_RESULTS_PATH)
    _validate_samples(samples, config, expected)


def _load_runtime_manifest(root: Path) -> Mapping[str, object]:
    value = _read_object(root / RUNTIME_MANIFEST_PATH)
    if (
        value.get("runtime_version") != RUNTIME_VERSION
        or value.get("guidance_version") != GUIDANCE_VERSION
        or value.get("guidance_sha256") != GUIDANCE_SHA256
        or value.get("starting_commit") != STARTING_COMMIT
        or value.get("split") != "development"
        or value.get("allowed_users") != list(ALLOWED_USERS)
        or value.get("query_count") != 8
        or value.get("result_count") != 24
        or value.get("latency_sample_count") != 240
        or value.get("runtime_boundary") != "latency_checkpoint_before_relevance"
        or value.get("model_usage") != "deterministic_no_model"
    ):
        raise QualityRuntimeError("runtime manifest identity changed")
    bindings = value.get("runtime_bindings")
    if not isinstance(bindings, dict):
        raise QualityRuntimeError("runtime bindings changed")
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise QualityRuntimeError("runtime binding changed")
        for key, path in binding.items():
            if not key.endswith("_path") and key != "path":
                continue
            hash_key = key.removesuffix("_path") + "_sha256" if key != "path" else "sha256"
            if hash_key not in binding or _file_sha256(root / str(path)) != binding[hash_key]:
                raise QualityRuntimeError("runtime binding hash changed")
    absent = value.get("required_absent_paths")
    if not isinstance(absent, list) or len(absent) != 6 or len(set(absent)) != 6:
        raise QualityRuntimeError("checkpoint absence contract changed")
    return value


def _preflight(
    root: Path,
    absent_paths: Sequence[str],
    config: QualityRuntimeConfig,
    manifest: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "runtime_version": RUNTIME_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "config_sha256": config.sha256,
        "runtime_manifest_sha256": _file_sha256(root / RUNTIME_MANIFEST_PATH),
        "runtime_module_sha256": _file_sha256(root / "src/retrieval/quality_runtime.py"),
        "absent_paths": list(absent_paths),
        "all_required_paths_absent": True,
        "relevance_opened": False,
        "relevance_hashed": False,
        "frozen_test_content_opened": False,
        "oracle_content_opened": False,
        "review_content_opened": False,
        "runtime_bindings": manifest["runtime_bindings"],
        "model_usage": {
            "provider_requests": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "incremental_cost_usd": 0,
        },
    }


def _load_expected_results(path: Path) -> Mapping[tuple[str, str], str]:
    values = _read_jsonl(path)
    expected: dict[tuple[str, str], str] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "case_id",
            "capability",
            "primary_label",
            "result",
        }:
            raise QualityRuntimeError("baseline result fields changed")
        result = value["result"]
        if not isinstance(result, dict):
            raise QualityRuntimeError("baseline result changed")
        key = (str(value["case_id"]), str(result.get("baseline_id")))
        if key in expected:
            raise QualityRuntimeError("baseline result is duplicated")
        expected[key] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    expected_keys = {
        (f"baseline_case_{number:03d}", baseline_id)
        for number in range(1, 9)
        for baseline_id in BASELINES
    }
    if set(expected) != expected_keys:
        raise QualityRuntimeError("baseline result accounting changed")
    return expected


def _require_expected_result(
    case_id: str,
    baseline_id: str,
    result: BaselineRetrievalResult,
    expected: Mapping[tuple[str, str], str],
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(asdict(result))).hexdigest()
    if expected.get((case_id, baseline_id)) != digest:
        raise QualityRuntimeError("retrieval result digest changed")
    if (
        result.baseline_id != baseline_id
        or result.user_id not in ALLOWED_USERS
        or any(
            not item.claim_ids
            or not item.claim_version_ids
            or not item.source_ids
            or not item.span_ids
            for item in result.accepted
        )
    ):
        raise QualityRuntimeError("retrieval result lineage changed")
    return digest


def _validate_samples(
    samples: Sequence[LatencySample],
    config: QualityRuntimeConfig,
    expected: Mapping[tuple[str, str], str],
) -> None:
    if len(samples) != config.latency_sample_count:
        raise QualityRuntimeError("latency sample count changed")
    expected_order = tuple(
        (f"baseline_case_{case_number:03d}", baseline_id, trial)
        for trial in range(1, 11)
        for case_number in range(1, 9)
        for baseline_id in BASELINES
    )
    actual_order = tuple(
        (sample.case_id, sample.baseline_id, sample.trial) for sample in samples
    )
    if actual_order != expected_order:
        raise QualityRuntimeError("latency sample order changed")
    if any(
        expected[(sample.case_id, sample.baseline_id)] != sample.result_sha256
        for sample in samples
    ):
        raise QualityRuntimeError("latency result digest changed")


def _machine_metadata(connection) -> Mapping[str, object]:
    postgres_version = str(connection.execute("SHOW server_version").fetchone()[0])
    vector_row = connection.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if vector_row is None:
        raise QualityRuntimeError("pgvector is unavailable")
    values = {
        "operating_system": platform.system().lower(),
        "machine_architecture": platform.machine().lower(),
        "python_version": platform.python_version(),
        "postgresql_version": postgres_version,
        "pgvector_version": str(vector_row[0]),
        "cpu_count": os.cpu_count(),
        "timer": "time.perf_counter_ns",
        "trial_order": "round_case_baseline",
    }
    if not values["operating_system"] or not values["machine_architecture"]:
        raise QualityRuntimeError("machine metadata is unavailable")
    return values


def _write_failed_attempt(
    output: Path,
    preflight: Mapping[str, object],
    error: Exception,
) -> None:
    code = getattr(error, "code", "runtime_failed")
    location = getattr(error, "location", "execution")
    if SAFE_FAILURE.fullmatch(str(code)) is None:
        code = "runtime_failed"
    if SAFE_FAILURE.fullmatch(str(location)) is None:
        location = "execution"
    failure = RuntimeFailure(
        hashlib.sha256(
            canonical_json_bytes({"code": code, "location": location})
        ).hexdigest(),
        str(code),
        str(location),
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output / "checkpoint_preflight.json", canonical_json_bytes(preflight))
    _write_exclusive(output / "latency-samples.jsonl", b"")
    _write_exclusive(output / "failures.jsonl", _serialize((failure,)))


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise QualityRuntimeError("runtime output must be empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise QualityRuntimeError("runtime artifact already exists") from error


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(value)) for value in values)


def _read_jsonl(path: Path) -> list[object]:
    values = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(json.loads(line))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityRuntimeError("JSONL input is invalid") from error
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityRuntimeError("JSON object is invalid") from error
    if not isinstance(value, dict):
        raise QualityRuntimeError("JSON object changed")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise QualityRuntimeError("bound runtime file is unavailable") from error


def _tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QualityRuntimeError(f"{name} changed")
    return tuple(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise QualityRuntimeError(f"{name} changed")
    return value


def _grades(value: object) -> Mapping[str, int]:
    if not isinstance(value, dict) or any(type(item) is not int for item in value.values()):
        raise QualityRuntimeError("relevance grades changed")
    return dict(value)
