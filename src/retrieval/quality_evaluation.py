"""Offline deterministic retrieval-quality scorer for the frozen Step 7.3 output."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .quality_contracts import (
    BASELINES,
    MetricValue,
    PerQueryQuality,
    QualityScorecard,
    RelevanceAnnotation,
    RelevanceCase,
    RetrievalQualityError,
)
from .quality_runtime import (
    CONFIG_PATH,
    OUTPUT_ROOT as RUNTIME_OUTPUT_ROOT,
    LatencySample,
    load_quality_runtime_config,
    verify_quality_runtime_checkpoint,
)
from .query_contracts import canonical_json_bytes


EVALUATION_VERSION = "retrieval_quality_development_v1"
STARTING_COMMIT = "d849ea1140f97066edb408acd8704268655c7abe"
GUIDANCE_VERSION = "step-7.4-guidance-v1"
GUIDANCE_SHA256 = "17d8609f5e0799661ea4a7d6b3a1de3493d267a6cff90efa4d21e17de4df96ae"
DATA_ROOT = Path("data/retrieval/retrieval-quality-development-v1")
DATA_MANIFEST = DATA_ROOT / "manifest.json"
RELEVANCE_PATH = DATA_ROOT / "gold/relevance.jsonl"
REVIEW_PATH = DATA_ROOT / "gold/review.json"
INDEX_RECORDS_PATH = Path("results/retrieval/index-development-v1/records.jsonl")
QUERIES_PATH = Path("data/retrieval/baseline-execution-development-v1/queries.jsonl")
BASELINE_RESULTS_PATH = Path("results/retrieval/baseline-execution-development-v1/results.jsonl")
SOURCES_PATH = Path("data/scaled-v1/runtime/sources.jsonl")
RESULT_ROOT = Path("results/retrieval/retrieval-quality-development-v1")
EXPECTED_PREFIX_SHA256 = "b0dc2b0dae985b51e2e25e441f98dcbe8d86045fa6a3beae2545f50de9e844d9"
EXPECTED_RECORDS_SHA256 = "e29531cd3bf72419c947a31b13c0b4adbd008f019dfbfc3e6a820ad541e23cfe"
EXPECTED_QUERIES_SHA256 = "e6e98f9b6de0747652d0d99b2379abe2d1e1a01ab012b1c6cae00cfab8e72bb4"
ARTIFACTS = (
    "per-query.jsonl",
    "failures.jsonl",
    "scorecard.json",
    "checks.json",
    "run.json",
    "findings.md",
)


def load_relevance_cases(
    relevance_path: str | Path = RELEVANCE_PATH,
    *,
    records_path: str | Path = INDEX_RECORDS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    sources_path: str | Path = SOURCES_PATH,
) -> tuple[RelevanceCase, ...]:
    records_source = Path(records_path)
    queries_source = Path(queries_path)
    if _sha256(records_source) != EXPECTED_RECORDS_SHA256:
        raise RetrievalQualityError("index records changed")
    if _sha256(queries_source) != EXPECTED_QUERIES_SHA256:
        raise RetrievalQualityError("runtime queries changed")
    sources, prefix = _read_prefix(Path(sources_path), 20)
    if hashlib.sha256(prefix).hexdigest() != EXPECTED_PREFIX_SHA256:
        raise RetrievalQualityError("source prefix changed")
    source_types = {str(row["source_id"]): str(row["source_type"]) for row in sources}
    if len(source_types) != 20 or any(row.get("user_id") not in ("user_001", "user_002") for row in sources):
        raise RetrievalQualityError("source prefix ownership changed")
    records = _read_jsonl(records_source)
    queries = _read_jsonl(queries_source)
    record_map = {str(row["index_record_id"]): row for row in records}
    if len(record_map) != 50 or len(records) != 50:
        raise RetrievalQualityError("candidate universe changed")
    query_map = {str(row["case_id"]): row for row in queries}
    values = _read_jsonl(Path(relevance_path))
    cases: list[RelevanceCase] = []
    for value in values:
        expected = {
            "case_id", "query_id", "user_id", "split", "query_type",
            "capability", "difficulty", "annotations",
        }
        if set(value) != expected or not isinstance(value["annotations"], list):
            raise RetrievalQualityError("relevance fields changed")
        annotations = tuple(_annotation(item) for item in value["annotations"])
        case = RelevanceCase(
            str(value["case_id"]), str(value["query_id"]), str(value["user_id"]),
            str(value["split"]), str(value["query_type"]), str(value["capability"]),
            str(value["difficulty"]), annotations,
        )
        query = query_map.get(case.case_id)
        if query is None or (
            query.get("query_id"), query.get("user_id"), query.get("capability")
        ) != (case.query_id, case.user_id, case.query_type):
            raise RetrievalQualityError("relevance query identity changed")
        expected_ids = sorted(
            record_id for record_id, row in record_map.items() if row.get("user_id") == case.user_id
        )
        if [item.index_record_id for item in annotations] != expected_ids:
            raise RetrievalQualityError("same-user annotation universe is incomplete")
        for annotation in annotations:
            record = record_map[annotation.index_record_id]
            expected_types = tuple(sorted({
                source_types[str(link["source_id"])] for link in record.get("source_lineage", [])
            }))
            if annotation.record_kind != record.get("record_kind") or annotation.source_types != expected_types:
                raise RetrievalQualityError("annotation lineage changed")
        cases.append(case)
    expected_case_ids = tuple(f"baseline_case_{value:03d}" for value in range(1, 9))
    if tuple(item.case_id for item in cases) != expected_case_ids:
        raise RetrievalQualityError("relevance case order changed")
    if sum(len(item.annotations) for item in cases) != 200:
        raise RetrievalQualityError("annotation count changed")
    return tuple(cases)


def score_retrieval_quality(
    result_rows: Sequence[Mapping[str, object]],
    relevance_cases: Sequence[RelevanceCase],
    latency_samples: Sequence[LatencySample],
) -> tuple[tuple[PerQueryQuality, ...], QualityScorecard]:
    if len(result_rows) != 24 or len(relevance_cases) != 8 or len(latency_samples) != 240:
        raise RetrievalQualityError("evaluation accounting is incomplete")
    relevance = {item.case_id: item for item in relevance_cases}
    seen: set[tuple[str, str]] = set()
    per_query: list[PerQueryQuality] = []
    cross_user = 0
    for row in result_rows:
        case_id = str(row.get("case_id"))
        case = relevance.get(case_id)
        result = row.get("result")
        if case is None or not isinstance(result, dict):
            raise RetrievalQualityError("ranked result identity changed")
        baseline = str(result.get("baseline_id"))
        key = (case_id, baseline)
        if baseline not in BASELINES or key in seen:
            raise RetrievalQualityError("ranked result is duplicated or unknown")
        seen.add(key)
        if result.get("query_id") != case.query_id or result.get("user_id") != case.user_id:
            raise RetrievalQualityError("ranked result ownership changed")
        accepted = result.get("accepted")
        if not isinstance(accepted, list):
            raise RetrievalQualityError("accepted results changed")
        ids: list[str] = []
        for expected_rank, item in enumerate(accepted, 1):
            if not isinstance(item, dict) or item.get("rank") != expected_rank:
                raise RetrievalQualityError("accepted ranks changed")
            record_id = str(item.get("index_record_id"))
            ids.append(record_id)
            if record_id not in {value.index_record_id for value in case.annotations}:
                cross_user += 1
            if not item.get("claim_ids") or not item.get("claim_version_ids") or not item.get("source_ids") or not item.get("span_ids"):
                raise RetrievalQualityError("accepted provenance is incomplete")
        if len(ids) != len(set(ids)):
            raise RetrievalQualityError("accepted IDs are duplicated")
        per_query.append(_score_one(case, baseline, ids))
    if seen != {(case.case_id, baseline) for case in relevance_cases for baseline in BASELINES}:
        raise RetrievalQualityError("ranked result matrix is incomplete")
    if cross_user:
        raise RetrievalQualityError("cross-user result reached scoring")
    quality_rows = _quality_aggregates(per_query)
    latency_rows = _latency_aggregates(latency_samples, relevance)
    scorecard = QualityScorecard(
        EVALUATION_VERSION, 8, 24, 200, 240,
        tuple(quality_rows), tuple(latency_rows), 0, 0,
        {
            "provider_requests": 0, "retries": 0, "input_tokens": 0,
            "output_tokens": 0, "incremental_cost_usd": "$0",
        },
        False, True,
    )
    return tuple(per_query), scorecard


def execute_quality_evaluation(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> tuple[tuple[PerQueryQuality, ...], QualityScorecard]:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    _require_empty(output)
    # The checkpoint and runtime artifacts are fully verified before scorer-only gold opens.
    verify_quality_runtime_checkpoint(root / RUNTIME_OUTPUT_ROOT, repo_root=root)
    checkpoint = _read_object(root / RUNTIME_OUTPUT_ROOT / "checkpoint_manifest.json")
    if checkpoint.get("relevance_opened") is not False or checkpoint.get("failure_count") != 0:
        raise RetrievalQualityError("runtime checkpoint is not publishable")
    result_rows = tuple(_read_jsonl(root / BASELINE_RESULTS_PATH))
    latency_samples = tuple(
        LatencySample(**value)
        for value in _read_jsonl(root / RUNTIME_OUTPUT_ROOT / "latency-samples.jsonl")
    )
    # Runtime inputs are closed in memory before this first scorer-only gold read.
    manifest = _load_data_manifest(root)
    cases = load_relevance_cases(
        root / RELEVANCE_PATH,
        records_path=root / INDEX_RECORDS_PATH,
        queries_path=root / QUERIES_PATH,
        sources_path=root / SOURCES_PATH,
    )
    review = _read_object(root / REVIEW_PATH)
    _validate_review(review, cases)
    per_query, scorecard = score_retrieval_quality(result_rows, cases, latency_samples)
    payloads: dict[str, bytes] = {
        "per-query.jsonl": _serialize(per_query),
        "failures.jsonl": b"",
        "scorecard.json": canonical_json_bytes(asdict(scorecard)),
        "checks.json": canonical_json_bytes(_checks(per_query, scorecard, manifest)),
        "run.json": canonical_json_bytes(_run(scorecard)),
        "findings.md": _findings(scorecard).encode("utf-8"),
    }
    result_manifest = _result_manifest(root, manifest, payloads)
    payloads["manifest.json"] = canonical_json_bytes(result_manifest)
    output.mkdir(parents=True, exist_ok=True)
    for name in (*ARTIFACTS, "manifest.json"):
        _write_exclusive(output / name, payloads[name])
    verify_quality_release(output, repo_root=root)
    return per_query, scorecard


def verify_quality_release(
    output_dir: str | Path = RESULT_ROOT,
    *,
    repo_root: str | Path = ".",
) -> None:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    # Preserve the checkpoint-before-gold boundary during later verification too.
    verify_quality_runtime_checkpoint(root / RUNTIME_OUTPUT_ROOT, repo_root=root)
    names = {*ARTIFACTS, "manifest.json"}
    if not output.is_dir() or {item.name for item in output.iterdir()} != names:
        raise RetrievalQualityError("result artifact set changed")
    manifest = _read_object(output / "manifest.json")
    if manifest.get("evaluation_version") != EVALUATION_VERSION or manifest.get("guidance_sha256") != GUIDANCE_SHA256:
        raise RetrievalQualityError("result manifest identity changed")
    if set(manifest.get("artifacts", {})) != set(ARTIFACTS):
        raise RetrievalQualityError("result artifact map changed")
    for name in ARTIFACTS:
        if manifest["artifacts"][name] != _sha256(output / name):
            raise RetrievalQualityError("result artifact hash changed")
    if (output / "failures.jsonl").read_bytes():
        raise RetrievalQualityError("result failures changed")
    scorecard = _read_object(output / "scorecard.json")
    if (scorecard.get("query_count"), scorecard.get("result_count"), scorecard.get("annotation_count"), scorecard.get("latency_sample_count")) != (8, 24, 200, 240):
        raise RetrievalQualityError("result accounting changed")
    data_manifest = _load_data_manifest(root)
    expected_implementation = {
        "src/retrieval/quality_contracts.py": _sha256(root / "src/retrieval/quality_contracts.py"),
        "src/retrieval/quality_evaluation.py": _sha256(root / "src/retrieval/quality_evaluation.py"),
        "src/retrieval/quality_runtime.py": _sha256(root / "src/retrieval/quality_runtime.py"),
    }
    if (
        manifest.get("dataset_manifest_sha256") != _sha256(root / DATA_MANIFEST)
        or manifest.get("inputs") != data_manifest.get("inputs")
        or manifest.get("implementation") != expected_implementation
    ):
        raise RetrievalQualityError("result manifest binding changed")
    cases = load_relevance_cases(
        root / RELEVANCE_PATH,
        records_path=root / INDEX_RECORDS_PATH,
        queries_path=root / QUERIES_PATH,
        sources_path=root / SOURCES_PATH,
    )
    _validate_review(_read_object(root / REVIEW_PATH), cases)
    result_rows = tuple(_read_jsonl(root / BASELINE_RESULTS_PATH))
    latency_samples = tuple(
        LatencySample(**value)
        for value in _read_jsonl(root / RUNTIME_OUTPUT_ROOT / "latency-samples.jsonl")
    )
    per_query, computed_scorecard = score_retrieval_quality(
        result_rows, cases, latency_samples
    )
    expected_payloads: dict[str, bytes] = {
        "per-query.jsonl": _serialize(per_query),
        "failures.jsonl": b"",
        "scorecard.json": canonical_json_bytes(asdict(computed_scorecard)),
        "checks.json": canonical_json_bytes(
            _checks(per_query, computed_scorecard, data_manifest)
        ),
        "run.json": canonical_json_bytes(_run(computed_scorecard)),
        "findings.md": _findings(computed_scorecard).encode("utf-8"),
    }
    if any((output / name).read_bytes() != payload for name, payload in expected_payloads.items()):
        raise RetrievalQualityError("result artifact does not match recomputed score")
    if manifest != _result_manifest(root, data_manifest, expected_payloads):
        raise RetrievalQualityError("result manifest does not match recomputed release")


def _score_one(case: RelevanceCase, baseline: str, accepted_ids: Sequence[str]) -> PerQueryQuality:
    kinds = {"B2": {"atomic"}, "B3": {"session"}, "B4": {"atomic", "session"}}[baseline]
    annotations = {item.index_record_id: item for item in case.annotations}
    if any(value not in annotations for value in accepted_ids):
        raise RetrievalQualityError("accepted result is outside the relevance universe")
    relevant = {item.index_record_id: item for item in case.annotations if item.record_kind in kinds and item.relevance_grade > 0}
    accepted = [annotations[value] for value in accepted_ids]
    recall5 = _fraction(sum(value in relevant for value in accepted_ids[:5]), len(relevant), "no_relevant_records")
    recall10 = _fraction(sum(value in relevant for value in accepted_ids[:10]), len(relevant), "no_relevant_records")
    dcg = sum(
        Decimal(2 ** annotations[value].relevance_grade - 1)
        / Decimal(str(math.log2(rank + 1)))
        for rank, value in enumerate(accepted_ids[:10], 1)
        if value in annotations
    )
    ideal_grades = sorted((item.relevance_grade for item in relevant.values()), reverse=True)[:10]
    idcg = sum(Decimal(2 ** grade - 1) / Decimal(str(math.log2(rank + 1))) for rank, grade in enumerate(ideal_grades, 1))
    ndcg = _decimal_metric(dcg, idcg, "no_relevant_records")
    first = next((rank for rank, value in enumerate(accepted_ids, 1) if value in relevant), None)
    mrr = _fraction(Decimal(1) / Decimal(first) if first else 0, 1 if relevant else 0, "no_relevant_records")
    relevant_sessions = {key for key, value in relevant.items() if value.record_kind == "session"}
    if baseline == "B2":
        session_recall = MetricValue(0, 0, None, "baseline_has_no_session_path")
    else:
        session_recall = _fraction(sum(value in relevant_sessions for value in accepted_ids[:10]), len(relevant_sessions), "no_relevant_sessions")
    stale_count = sum(item.stale_for_query for item in accepted[:10])
    stale_rate = _fraction(stale_count, len(accepted[:10]), "no_accepted_records")
    source_types = tuple(sorted({source_type for item in relevant.values() for source_type in item.source_types}))
    return PerQueryQuality(
        case.case_id, case.query_id, case.user_id, baseline, case.query_type,
        case.capability, case.difficulty, source_types, recall5, recall10, ndcg,
        mrr, session_recall, stale_rate, len(accepted_ids), len(relevant), stale_count,
    )


def _quality_aggregates(rows: Sequence[PerQueryQuality]) -> list[Mapping[str, object]]:
    output: list[Mapping[str, object]] = []
    dimensions = (
        ("overall", lambda row: ("all",)),
        ("query_type", lambda row: (row.query_type,)),
        ("capability", lambda row: (row.capability,)),
        ("source_type", lambda row: row.source_types),
        ("difficulty", lambda row: (row.difficulty,)),
        ("split", lambda row: ("development",)),
    )
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row.baseline_id == baseline]
        for dimension, values in dimensions:
            groups: dict[str, list[PerQueryQuality]] = {}
            for row in baseline_rows:
                for value in values(row):
                    groups.setdefault(value, []).append(row)
            for value in sorted(groups):
                group = groups[value]
                metrics = {}
                for field in (
                    "recall_at_5", "recall_at_10", "ndcg_at_10", "mrr",
                    "relevant_session_recall", "stale_memory_rate",
                ):
                    metrics[field] = _aggregate_metric(group, field, micro=(field == "stale_memory_rate"))
                output.append({
                    "baseline_id": baseline, "slice_dimension": dimension,
                    "slice_value": value, "case_count": len(group), "metrics": metrics,
                })
    return output


def _latency_aggregates(
    samples: Sequence[LatencySample],
    relevance: Mapping[str, RelevanceCase],
) -> list[Mapping[str, object]]:
    output: list[Mapping[str, object]] = []
    dimensions = ("overall", "query_type", "capability", "source_type", "difficulty", "split")
    for baseline in BASELINES:
        baseline_samples = [item for item in samples if item.baseline_id == baseline]
        for dimension in dimensions:
            groups: dict[str, list[int]] = {}
            for sample in baseline_samples:
                case = relevance[sample.case_id]
                if dimension == "overall": values = ("all",)
                elif dimension == "query_type": values = (case.query_type,)
                elif dimension == "capability": values = (case.capability,)
                elif dimension == "difficulty": values = (case.difficulty,)
                elif dimension == "split": values = ("development",)
                else:
                    values = tuple(sorted({source for item in case.annotations if item.relevance_grade > 0 for source in item.source_types}))
                for value in values:
                    groups.setdefault(value, []).append(sample.elapsed_ns)
            for value in sorted(groups):
                values = sorted(groups[value])
                output.append({
                    "baseline_id": baseline, "slice_dimension": dimension,
                    "slice_value": value, "sample_count": len(values),
                    "mean_ms": _ms(sum(values) / len(values)),
                    "p50_ms": _ms(values[max(0, math.ceil(0.50 * len(values)) - 1)]),
                    "p95_ms": _ms(values[max(0, math.ceil(0.95 * len(values)) - 1)]),
                    "max_ms": _ms(values[-1]),
                })
    return output


def _aggregate_metric(rows: Sequence[PerQueryQuality], field: str, *, micro: bool) -> Mapping[str, object]:
    values = [getattr(row, field) for row in rows]
    defined = [item for item in values if item.value is not None]
    if not defined:
        reasons = sorted({item.null_reason for item in values if item.null_reason})
        return {
            "value": None, "null_reason": reasons[0] if len(reasons) == 1 else "no_scored_cases",
            "scored_case_count": 0, "null_case_count": len(values),
            "total_numerator": "0.000000", "total_denominator": "0.000000",
        }
    total_num = sum(Decimal(str(item.numerator)) for item in defined)
    total_den = sum(Decimal(str(item.denominator)) for item in defined)
    if micro:
        value = total_num / total_den if total_den else Decimal(0)
    else:
        value = sum(Decimal(item.value) for item in defined) / Decimal(len(defined))
    return {
        "value": _quality(value), "null_reason": None,
        "scored_case_count": len(defined), "null_case_count": len(values) - len(defined),
        "total_numerator": _quality(total_num), "total_denominator": _quality(total_den),
    }


def _fraction(numerator: int | Decimal, denominator: int | Decimal, reason: str) -> MetricValue:
    if not denominator:
        return MetricValue(str(numerator) if isinstance(numerator, Decimal) else numerator, denominator, None, reason)
    value = Decimal(numerator) / Decimal(denominator)
    return MetricValue(str(numerator) if isinstance(numerator, Decimal) else numerator, denominator, _quality(value), None)


def _decimal_metric(numerator: Decimal, denominator: Decimal, reason: str) -> MetricValue:
    if not denominator:
        return MetricValue(_quality(numerator), _quality(denominator), None, reason)
    return MetricValue(_quality(numerator), _quality(denominator), _quality(numerator / denominator), None)


def _quality(value: Decimal | int | float) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    return str(decimal_value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def _ms(nanoseconds: int | float) -> str:
    value = Decimal(str(nanoseconds)) / Decimal(1_000_000)
    return str(value.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def _annotation(value: object) -> RelevanceAnnotation:
    if not isinstance(value, dict) or set(value) != {
        "index_record_id", "record_kind", "relevance_grade", "stale_for_query",
        "stale_reason", "source_types", "review_status",
    } or not isinstance(value["source_types"], list):
        raise RetrievalQualityError("annotation fields changed")
    return RelevanceAnnotation(
        str(value["index_record_id"]), str(value["record_kind"]), value["relevance_grade"],
        value["stale_for_query"], value["stale_reason"],
        tuple(str(item) for item in value["source_types"]), str(value["review_status"]),
    )


def _checks(
    per_query: Sequence[PerQueryQuality],
    scorecard: QualityScorecard,
    manifest: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "annotation_count": scorecard.annotation_count,
        "blind_evaluation": False,
        "candidate_universe_complete": True,
        "cross_user_count": scorecard.cross_user_count,
        "failure_count": scorecard.runtime_failure_count,
        "gold_opened_after_checkpoint": True,
        "latency_sample_count": scorecard.latency_sample_count,
        "manifest_version": manifest["dataset_version"],
        "model_calls": 0,
        "per_query_row_count": len(per_query),
        "prior_ranked_result_exposure_possible": True,
        "provenance_complete": True,
        "query_count": scorecard.query_count,
        "result_count": scorecard.result_count,
        "runtime_checkpoint_verified": True,
        "test_user_count": 0,
    }


def _run(scorecard: QualityScorecard) -> Mapping[str, object]:
    return {
        "evaluation_version": EVALUATION_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "query_count": scorecard.query_count,
        "result_count": scorecard.result_count,
        "annotation_count": scorecard.annotation_count,
        "latency_sample_count": scorecard.latency_sample_count,
        "split": "development",
        "blind_evaluation": False,
        "prior_ranked_result_exposure_possible": True,
        "model_usage": dict(scorecard.model_usage),
        "historical_openai_spend_usd": "0.2314404",
    }


def _findings(scorecard: QualityScorecard) -> str:
    overall = [row for row in scorecard.quality_rows if row["slice_dimension"] == "overall"]
    lines = [
        "# Step 7.4 retrieval quality findings",
        "",
        "This development evaluation scores the frozen B2, B3, and B4 rankings. It did not change retrieval behavior.",
        "",
        "The review covered all 200 same-user record/query pairs. Rankings were not used while assigning labels. This was not a blind evaluation, and prior exposure to the committed rankings was possible.",
        "",
    ]
    for row in overall:
        metrics = row["metrics"]
        lines.append(
            f"- {row['baseline_id']}: Recall@10 {metrics['recall_at_10']['value']}; "
            f"nDCG@10 {metrics['ndcg_at_10']['value']}; MRR {metrics['mrr']['value']}; "
            f"stale-memory rate {metrics['stale_memory_rate']['value']}."
        )
    lines.extend([
        "",
        "The dataset is small, development-only, candidate-heavy, and uses deterministic token-hash vectors rather than a semantic embedding model. Scores are diagnostics, not production-quality claims.",
        "",
        "No provider or model was called. Incremental cost was $0; historical OpenAI spend remains $0.2314404.",
    ])
    return "\n".join(lines) + "\n"


def _result_manifest(
    root: Path,
    data_manifest: Mapping[str, object],
    payloads: Mapping[str, bytes],
) -> Mapping[str, object]:
    return {
        "evaluation_version": EVALUATION_VERSION,
        "starting_commit": STARTING_COMMIT,
        "guidance_version": GUIDANCE_VERSION,
        "guidance_sha256": GUIDANCE_SHA256,
        "dataset_manifest_sha256": _sha256(root / DATA_MANIFEST),
        "runtime_checkpoint": {
            "path": str(RUNTIME_OUTPUT_ROOT / "checkpoint_manifest.json"),
            "sha256": _sha256(root / RUNTIME_OUTPUT_ROOT / "checkpoint_manifest.json"),
        },
        "inputs": data_manifest["inputs"],
        "implementation": {
            "src/retrieval/quality_contracts.py": _sha256(root / "src/retrieval/quality_contracts.py"),
            "src/retrieval/quality_evaluation.py": _sha256(root / "src/retrieval/quality_evaluation.py"),
            "src/retrieval/quality_runtime.py": _sha256(root / "src/retrieval/quality_runtime.py"),
        },
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        "counts": {"queries": 8, "results": 24, "annotations": 200, "latency_samples": 240, "failures": 0},
        "blind_evaluation": False,
        "prior_ranked_result_exposure_possible": True,
        "model_usage": {"provider_requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0, "incremental_cost_usd": "$0"},
        "composite_score": None,
    }


def _load_data_manifest(root: Path) -> Mapping[str, object]:
    value = _read_object(root / DATA_MANIFEST)
    if value.get("dataset_version") != EVALUATION_VERSION or value.get("annotation_count") != 200:
        raise RetrievalQualityError("dataset manifest identity changed")
    for binding in value.get("inputs", {}).values():
        if isinstance(binding, dict) and "path" in binding and "sha256" in binding:
            if _sha256(root / str(binding["path"])) != binding["sha256"]:
                raise RetrievalQualityError("dataset input hash changed")
    return value


def _validate_review(review: Mapping[str, object], cases: Sequence[RelevanceCase]) -> None:
    if (
        review.get("annotation_count") != 200
        or review.get("reviewed_annotation_count") != 200
        or review.get("query_count") != 8
        or review.get("review_status") != "implementation_reviewed"
        or review.get("rankings_used_during_annotation") is not False
        or review.get("blind_evaluation") is not False
        or review.get("prior_ranked_result_exposure_possible") is not True
        or review.get("cross_user_annotation_count") != 0
        or sum(len(item.annotations) for item in cases) != 200
    ):
        raise RetrievalQualityError("review attestation changed")


def _read_prefix(path: Path, count: int) -> tuple[list[Mapping[str, object]], bytes]:
    rows: list[Mapping[str, object]] = []
    payload = bytearray()
    try:
        with path.open("rb") as handle:
            for _ in range(count):
                line = handle.readline()
                if not line:
                    raise RetrievalQualityError("source prefix is incomplete")
                payload.extend(line)
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RetrievalQualityError("source prefix row is invalid")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalQualityError("source prefix is unreadable") from error
    return rows, bytes(payload)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    values: list[Mapping[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise RetrievalQualityError("JSONL row is invalid")
                    values.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalQualityError("JSONL input is invalid") from error
    return values


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalQualityError("JSON object is invalid") from error
    if not isinstance(value, dict):
        raise RetrievalQualityError("JSON object is invalid")
    return value


def _serialize(values: Sequence[object]) -> bytes:
    return b"".join(canonical_json_bytes(asdict(value)) for value in values)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RetrievalQualityError("bound file is unavailable") from error


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RetrievalQualityError("result directory must be empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RetrievalQualityError("result artifact already exists") from error
