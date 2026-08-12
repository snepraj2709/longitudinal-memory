"""Run and score the small Qwen3-8B scaled-v1 extraction quality gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from extraction.atomic import validate_atomic_response
from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format

from .openai_client import OpenAIResponseMetadata
from .qwen_benchmark import REGISTRY, RUNTIME
from .qwen_compatibility import _response_format
from .qwen_execution import (
    ExecutionJob,
    TransportRetryLedger,
    _client_factory,
    _normalize_qwen_extraction_response,
    execute_jobs,
)
from .qwen_serverless_pilot import _normalize_base_url, read_api_key


CONFIG_PATH = Path("configs/evaluation/qwen3_8b_extraction_gate_v1.json")
PAID_RUN_CONFIRMATION = "qwen3-8b-vllm-extraction-gate-paid-run-approved"
GOLD_CLAIMS_PATH = Path("data/scaled-v1/gold/claims.jsonl")
KNOWN_BAD_OVERLAP_PREDICATES = frozenset({
    "employment_start_date",
    "has_scheduled_event",
    "primary_role",
})


class QwenExtractionGateError(RuntimeError):
    """Raised when the Qwen extraction gate cannot run or score safely."""


def load_config(repo_root: Path, config_path: Path = CONFIG_PATH) -> Mapping[str, object]:
    config = json.loads((repo_root / config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "qwen3_8b_extraction_gate_config_v1":
        raise QwenExtractionGateError("unexpected extraction gate config schema")
    if config.get("series_id") != "qwen3-8b-vllm-extraction-gate-v1":
        raise QwenExtractionGateError("unexpected extraction gate series id")
    runtime = config.get("runtime")
    model = config.get("model")
    profile = config.get("extraction_profile")
    if not isinstance(runtime, Mapping) or not isinstance(model, Mapping) or not isinstance(profile, Mapping):
        raise QwenExtractionGateError("extraction gate config is malformed")
    if runtime.get("client_concurrency") != 1 or runtime.get("temperature") != 0:
        raise QwenExtractionGateError("extraction gate must use concurrency 1 and temperature 0")
    if model.get("model_alias") != "qwen3-8b-vllm" or model.get("context_length") != 8192:
        raise QwenExtractionGateError("Qwen3 extraction gate model alias or context length changed")
    if profile.get("base_prompt_version") != "atomic-extraction-v3" or profile.get("include_speaker_name") is not True:
        raise QwenExtractionGateError("scaled-v1 extraction profile changed")
    _gate_config(config, "primary_gate")
    _gate_config(config, "holdout_gate")
    return config


def build_gate_jobs(
    repo_root: Path,
    *,
    config: Mapping[str, object],
    gate_name: str = "primary_gate",
) -> tuple[ExecutionJob, ...]:
    root = repo_root.resolve()
    gate = _gate_config(config, gate_name)
    selected = _selected_sources(root, gate)
    users = _jsonl(root / RUNTIME / "users.jsonl")
    names = {str(row["user_id"]): str(row["display_name"]) for row in users}
    registry = load_predicate_registry(root / REGISTRY)
    boolean_predicates = {
        definition.predicate
        for definition in registry.definitions
        if definition.object_shape == "boolean"
    }
    system_prompt = _scaled_v1_system_prompt(registry)
    raw_format = atomic_extraction_text_format(registry)
    response_format = _response_format(str(raw_format["name"]), raw_format["schema"])
    jobs: list[ExecutionJob] = []
    for position, source in enumerate(selected, 1):
        adapted = _adapt_source(
            source,
            (str(source["user_id"]), str(source["source_id"])),
            names,
        )

        def validate(
            raw: str,
            metadata: OpenAIResponseMetadata,
            *,
            item=adapted,
        ) -> Mapping[str, object]:
            normalized = _normalize_qwen_extraction_response(raw, boolean_predicates)
            result = validate_atomic_response(item, normalized, metadata, registry=registry)
            return {"claims": [asdict(claim) for claim in result.claims]}

        jobs.append(ExecutionJob(
            request_id=f"extraction:{source['source_id']}",
            position=position,
            split=str(config["split"]),
            task="extraction",
            record_id=str(source["source_id"]),
            user_id=str(source["user_id"]),
            baseline_id=None,
            context_sha256=None,
            context_count=0,
            system_prompt=system_prompt,
            user_prompt=build_atomic_extraction_prompt(adapted, include_speaker_name=True),
            response_format=response_format,
            max_output_tokens=1200,
            validator=validate,
            local_metadata={
                "gate_name": gate_name,
                "extraction_profile": config["extraction_profile"],
            },
            series_id=str(config["series_id"]),
        ))
    return tuple(jobs)


def dry_run(
    *,
    repo_root: Path,
    config_path: Path = CONFIG_PATH,
    gate_name: str = "primary_gate",
) -> Mapping[str, object]:
    config = load_config(repo_root, config_path)
    jobs = build_gate_jobs(repo_root, config=config, gate_name=gate_name)
    return {
        "schema_version": "qwen_extraction_gate_dry_run_v1",
        "series_id": config["series_id"],
        "gate_name": gate_name,
        "planned_request_count": len(jobs),
        "source_ids": [job.record_id for job in jobs],
        "worker_count": config["runtime"]["client_concurrency"],
        "temperature": config["runtime"]["temperature"],
        "gold_opened": False,
        "oracle_opened": False,
        "review_opened": False,
        "request_sha256": sha256(
            "\n".join(job.request_sha256 for job in jobs).encode("utf-8")
        ).hexdigest(),
    }


def run_gate(
    *,
    repo_root: Path,
    output_root: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    paid_run_confirmation: str,
    config_path: Path = CONFIG_PATH,
    gate_name: str = "primary_gate",
) -> Mapping[str, object]:
    if paid_run_confirmation != PAID_RUN_CONFIRMATION:
        raise QwenExtractionGateError(
            f"pass --confirm-paid-run {PAID_RUN_CONFIRMATION!r} after capped Jarvis approval"
        )
    root = repo_root.resolve()
    config = load_config(root, config_path)
    if model != config["model"]["model_alias"]:
        raise QwenExtractionGateError("served model alias does not match the extraction gate config")
    output = output_root if output_root.is_absolute() else root / output_root
    gate_output = output / gate_name
    gate_output.mkdir(parents=True, exist_ok=True)
    retry_ledger = TransportRetryLedger(int(config["runtime"]["maximum_transport_retries"]))
    execution = execute_jobs(
        build_gate_jobs(root, config=config, gate_name=gate_name),
        output_dir=gate_output / "extraction",
        client_factory=_client_factory(
            base_url=_normalize_base_url(base_url),
            model=model,
            api_key=api_key,
            temperature=0.0,
        ),
        retry_ledger=retry_ledger,
        workers=1,
        retryable_http_statuses=tuple(config["runtime"]["retryable_http_statuses"]),
        retry_validation_failures=False,
    )
    scores = score_gate_output(
        repo_root=root,
        config=config,
        extraction_dir=gate_output / "extraction",
        output_dir=gate_output / "scores",
        gate_name=gate_name,
    )
    manifest = {
        "schema_version": "qwen_extraction_gate_run_v1",
        "series_id": config["series_id"],
        "gate_name": gate_name,
        "status": "completed",
        "provider": config["provider"],
        "model": config["model"],
        "runtime": {
            **config["runtime"],
            "base_url": _normalize_base_url(base_url),
        },
        "extraction": execution,
        "scores": scores["scorecard"],
        "config_sha256": _file_sha(root / config_path),
        "gold_opened_after_extraction_seal": True,
        "oracle_opened": False,
        "review_opened": False,
    }
    _write_json(gate_output / "run-manifest.json", manifest)
    return manifest


def score_gate_output(
    *,
    repo_root: Path,
    config: Mapping[str, object],
    extraction_dir: Path,
    output_dir: Path,
    gate_name: str,
) -> Mapping[str, object]:
    manifest_path = extraction_dir / "manifest.json"
    responses_path = extraction_dir / "responses.jsonl"
    if not manifest_path.exists() or not responses_path.exists():
        raise QwenExtractionGateError("extraction gate output is not sealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("series_id") != config["series_id"] or manifest.get("gold_opened") is not False:
        raise QwenExtractionGateError("extraction gate seal is invalid")
    records = _jsonl(responses_path)
    gate = _gate_config(config, gate_name)
    gold_claims = _load_gate_gold(repo_root, gate)
    metrics = score_gate_records(
        records=records,
        gold_claims=gold_claims,
        gate=gate,
        series_id=str(config["series_id"]),
        gate_name=gate_name,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metric_payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in metrics
    )
    (output_dir / "metrics.jsonl").write_bytes(metric_payload)
    scorecard = {
        "schema_version": "qwen_extraction_gate_scorecard_v1",
        "series_id": config["series_id"],
        "gate_name": gate_name,
        "status": "passed" if _passes(metrics, gate) else "failed_quality_gate",
        "metric_count": len(metrics),
        "metrics_sha256": sha256(metric_payload).hexdigest(),
        "gold_opened_after_extraction_seal": True,
    }
    (output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"scorecard": scorecard, "metrics": metrics}


def score_gate_records(
    *,
    records: Sequence[Mapping[str, object]],
    gold_claims: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    series_id: str,
    gate_name: str,
) -> tuple[dict[str, object], ...]:
    source_ids = set(_source_ids(gate))
    predicted = []
    valid_count = 0
    for row in records:
        if row.get("record_id") not in source_ids:
            raise QwenExtractionGateError("extraction response includes a source outside the gate")
        if row.get("status") != "succeeded" or not isinstance(row.get("output"), Mapping):
            continue
        claims = row["output"].get("claims")
        if not isinstance(claims, list):
            continue
        valid_count += 1
        for claim in claims:
            if isinstance(claim, Mapping):
                predicted.append({
                    **claim,
                    "user_id": row.get("user_id"),
                    "source_id": row.get("record_id"),
                })
    gold = [
        claim for claim in gold_claims
        if any(item.get("source_id") in source_ids for item in claim.get("evidence", []))
    ]
    pred_counter = Counter(_proposition_key(claim) for claim in predicted)
    gold_counter = Counter(_proposition_key(claim) for claim in gold)
    matched_props = sum((pred_counter & gold_counter).values())
    strict_matches = sum((Counter(_strict_key(claim) for claim in predicted) & Counter(_strict_key(claim) for claim in gold)).values())
    matched_pairs = _matched_pairs(predicted, gold)
    pred_evidence = Counter(_evidence_key(item) for claim in predicted for item in claim.get("evidence", []) if _evidence_key(item))
    gold_evidence = Counter(_evidence_key(item) for claim in gold for item in claim.get("evidence", []) if _evidence_key(item))
    exact_quote_matches = sum((pred_evidence & gold_evidence).values())
    wrong_person = sum(
        1 for claim in predicted
        if claim.get("source_id") == _wrong_person_source(gate)
        and claim.get("subject_id") == gate.get("user_id")
    )
    bad_overlap_uses = sum(
        1 for claim in predicted
        if claim.get("predicate") in KNOWN_BAD_OVERLAP_PREDICATES
    )
    rows = [
        _metric(series_id, gate_name, "structurally_valid_sources", valid_count, len(source_ids)),
        _metric(series_id, gate_name, "proposition_precision", matched_props, sum(pred_counter.values())),
        _metric(series_id, gate_name, "proposition_recall", matched_props, sum(gold_counter.values())),
        _f1_metric(series_id, gate_name, "proposition_f1", matched_props, sum(pred_counter.values()), sum(gold_counter.values())),
        _metric(series_id, gate_name, "strict_claim_precision", strict_matches, len(predicted)),
        _metric(series_id, gate_name, "strict_claim_recall", strict_matches, len(gold)),
        _metric(series_id, gate_name, "unsupported_memory_rate", len(predicted) - matched_props, len(predicted)),
        _metric(series_id, gate_name, "speaker_accuracy", _pair_matches(matched_pairs, "speaker_id"), len(matched_pairs)),
        _metric(series_id, gate_name, "epistemic_status_accuracy", _pair_matches(matched_pairs, "epistemic_status"), len(matched_pairs)),
        _metric(series_id, gate_name, "valid_time_accuracy", _valid_time_matches(matched_pairs), len(matched_pairs)),
        _metric(series_id, gate_name, "exact_quote_precision", exact_quote_matches, sum(pred_evidence.values())),
        _metric(series_id, gate_name, "exact_quote_recall", exact_quote_matches, sum(gold_evidence.values())),
        _count_metric(series_id, gate_name, "wrong_person_claim_count", wrong_person),
        _count_metric(series_id, gate_name, "known_bad_overlap_predicate_uses", bad_overlap_uses),
    ]
    return tuple(rows)


def _scaled_v1_system_prompt(registry) -> str:
    base = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    profile = """

Scaled-v1 extraction profile:
- Use accepted_role, not primary_role, when the source says the user accepted a role.
- Use job_start_date, not employment_start_date, for an employment start date.
- Use project_review_date, not has_scheduled_event, when a project review is scheduled for a date.
- Use career_goal for stated goals such as "I want to ..."; do not encode goals as relocation plans.
- Use work_preference for remote-work preference; remote work is not a location or lives_in claim.
- For "has not decided to move" or "made no plan", use has_relocation_plan with the city object, negative polarity, and hypothetical or denied/uncertain status as supported by the wording.
- Preserve reports from other speakers as reported_by_other, and preserve explicit corrections as corrected.
- Do not attach another person's fact to the benchmark user. If the source says the fact is about Kabir or Lucia, the subject is Kabir or Lucia.
"""
    return base + profile


def _selected_sources(repo_root: Path, gate: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    sources = {str(row["source_id"]): row for row in _jsonl(repo_root / RUNTIME / "sources.jsonl")}
    selected = []
    for source_id in _source_ids(gate):
        source = sources.get(source_id)
        if source is None:
            raise QwenExtractionGateError(f"gate source not found: {source_id}")
        if source.get("user_id") != gate.get("user_id"):
            raise QwenExtractionGateError("gate source user does not match gate user")
        selected.append(source)
    if len(selected) != 10:
        raise QwenExtractionGateError("extraction gate must contain exactly 10 sources")
    return tuple(selected)


def _load_gate_gold(repo_root: Path, gate: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    source_ids = set(_source_ids(gate))
    claims = []
    for row in _jsonl(repo_root / GOLD_CLAIMS_PATH):
        if row.get("user_id") != gate.get("user_id"):
            continue
        if any(item.get("source_id") in source_ids for item in row.get("evidence", [])):
            claims.append(row)
    return tuple(claims)


def _gate_config(config: Mapping[str, object], gate_name: str) -> Mapping[str, object]:
    gate = config.get(gate_name)
    if not isinstance(gate, Mapping):
        raise QwenExtractionGateError(f"{gate_name} config is missing")
    source_ids = gate.get("source_ids")
    if not isinstance(source_ids, list) or len(source_ids) != 10 or not all(isinstance(item, str) for item in source_ids):
        raise QwenExtractionGateError(f"{gate_name} must pin exactly 10 source IDs")
    if not isinstance(gate.get("user_id"), str):
        raise QwenExtractionGateError(f"{gate_name} must pin one user ID")
    return gate


def _source_ids(gate: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(item) for item in gate["source_ids"])


def _wrong_person_source(gate: Mapping[str, object]) -> str:
    user_id = str(gate.get("user_id"))
    return "scaled_user_001_chat_001" if user_id == "user_001" else "scaled_user_002_chat_001"


def _proposition_key(claim: Mapping[str, object]) -> tuple[object, ...]:
    return (
        claim.get("user_id"),
        claim.get("subject_id"),
        claim.get("predicate"),
        _normalized_object(claim.get("object")),
        claim.get("polarity"),
    )


def _strict_key(claim: Mapping[str, object]) -> tuple[object, ...]:
    return (
        *_proposition_key(claim),
        claim.get("speaker_id"),
        claim.get("epistemic_status"),
        claim.get("valid_from"),
        claim.get("valid_to"),
    )


def _normalized_object(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _matched_pairs(
    predicted: Sequence[Mapping[str, object]],
    gold: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    remaining: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for claim in gold:
        remaining.setdefault(_proposition_key(claim), []).append(claim)
    pairs = []
    for claim in predicted:
        candidates = remaining.get(_proposition_key(claim), [])
        if not candidates:
            continue
        pairs.append((claim, candidates.pop(0)))
    return tuple(pairs)


def _pair_matches(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    field: str,
) -> int:
    return sum(1 for predicted, gold in pairs if predicted.get(field) == gold.get(field))


def _valid_time_matches(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> int:
    return sum(
        1 for predicted, gold in pairs
        if predicted.get("valid_from") == gold.get("valid_from")
        and predicted.get("valid_to") == gold.get("valid_to")
    )


def _evidence_key(item: object) -> tuple[object, object, object] | None:
    if not isinstance(item, Mapping):
        return None
    source_id = item.get("source_id")
    message_id = item.get("message_id")
    quote = item.get("quote")
    if not isinstance(source_id, str) or not isinstance(quote, str):
        return None
    return source_id, message_id, quote


def _metric(
    series_id: str,
    gate_name: str,
    name: str,
    numerator: float,
    denominator: float,
) -> dict[str, object]:
    return {
        "series_id": series_id,
        "gate_name": gate_name,
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "null_reason": None if denominator else "zero_denominator",
    }


def _f1_metric(
    series_id: str,
    gate_name: str,
    name: str,
    matched: int,
    predicted: int,
    gold: int,
) -> dict[str, object]:
    precision = matched / predicted if predicted else 0
    recall = matched / gold if gold else 0
    denominator = 1 if precision + recall else 0
    return _metric(
        series_id,
        gate_name,
        name,
        2 * precision * recall / (precision + recall) if denominator else 0,
        denominator,
    )


def _count_metric(series_id: str, gate_name: str, name: str, value: int) -> dict[str, object]:
    return _metric(series_id, gate_name, name, value, 1)


def _passes(metrics: Sequence[Mapping[str, object]], gate: Mapping[str, object]) -> bool:
    values = {str(row["metric"]): row.get("value") for row in metrics}
    acceptance = gate.get("acceptance", {})
    if not isinstance(acceptance, Mapping):
        return False
    minimums = {
        "structurally_valid_sources": "structurally_valid_sources",
        "proposition_precision": "proposition_precision",
        "proposition_recall": "proposition_recall",
        "proposition_f1": "proposition_f1",
        "exact_quote_precision": "exact_quote_precision",
        "exact_quote_recall": "exact_quote_recall",
    }
    maximums = {
        "unsupported_memory_rate_max": "unsupported_memory_rate",
        "wrong_person_claim_count_max": "wrong_person_claim_count",
        "known_bad_overlap_predicate_uses_max": "known_bad_overlap_predicate_uses",
    }
    for threshold_name, metric_name in minimums.items():
        if threshold_name in acceptance and (
            values.get(metric_name) is None
            or float(values[metric_name]) < float(acceptance[threshold_name])
        ):
            return False
    for threshold_name, metric_name in maximums.items():
        if threshold_name in acceptance and (
            values.get(metric_name) is None
            or float(values[metric_name]) > float(acceptance[threshold_name])
        ):
            return False
    return True


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", choices=("primary_gate", "holdout_gate"), default="primary_gate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="qwen3-8b-vllm")
    parser.add_argument("--api-key-env", default="QWEN_VLLM_API_KEY")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--confirm-paid-run")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if args.dry_run:
        print(json.dumps(
            dry_run(repo_root=root, config_path=args.config, gate_name=args.gate),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
        return 0
    if not args.base_url:
        raise SystemExit("--base-url is required unless --dry-run is set")
    config = load_config(root, args.config)
    output = args.output
    if output is None:
        output = root / str(config["output"]["root"])
    result = run_gate(
        repo_root=root,
        output_root=output,
        base_url=args.base_url,
        model=args.model,
        api_key=read_api_key(env_name=args.api_key_env, env_file=args.env_file),
        paid_run_confirmation=args.confirm_paid_run or "",
        config_path=args.config,
        gate_name=args.gate,
    )
    print(json.dumps({
        "status": result["status"],
        "series_id": result["series_id"],
        "gate_name": result["gate_name"],
        "output": str(output),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
