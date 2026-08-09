"""Run the frozen GPT-4.1 fallback for Step 3.5."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence, TextIO

from evaluation.openai_client import (
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)
from evaluation.run_config import assert_no_secrets, canonical_sha256

from .atomic import (
    AtomicExtractionValidationError,
    sanitized_validation_diagnostics,
    validate_atomic_response,
)
from .phase4_input import (
    Phase4InputValidationError,
    build_phase4_source_claims,
    phase4_claim_record,
    validate_phase4_claim_file,
)
from .run_phase4_input import (
    FALLBACK_CONFIG_PATH,
    FULL_CONFIG_V2_PATH,
    QUALIFICATION_CONFIG_V2_PATH,
    QUALIFICATION_ROOT_V2,
    Step35RunError,
    Step35StagePlan,
    _atomic_record,
    _build_evidence_index,
    _file_sha256,
    _git_commit,
    _git_dirty,
    _git_status_text,
    _model_request_plan,
    _read_json,
    _read_jsonl,
    _runtime_input_manifest,
    _score_model,
    _utc_now,
    _write_json,
    _write_jsonl,
    prepare_step35_stage,
)
from .run_safety import additive_cost_text, step35_cost
from .scaled_scoring import load_scaled_development_gold, score_scaled_development


FALLBACK_OUTPUT_ROOT = Path(
    "results/phase3/phase4-input-development-gpt41-fallback-v1"
)
FALLBACK_GUIDANCE_VERSION = "step-3.5-gpt41-fallback-guidance-v1"
FALLBACK_ENVELOPE_SHA256 = (
    "016f94919f46184994849fc46dd63b3259d6054096cffde210f9324799090f86"
)
PRIOR_SPEND_USD = Decimal("0.0697964")
NEW_HARD_MAXIMUM_USD = Decimal("0.3141640")
CUMULATIVE_HARD_MAXIMUM_USD = Decimal("0.3839604")
MAX_OUTPUT_TOKENS = 1200
_V2_CONFIG_HASHES = {
    QUALIFICATION_CONFIG_V2_PATH.as_posix(): (
        "d66c069e66fbc6baf3c0cbce147cae1c9e47ebb54418545e901e8c40b4e7f00f"
    ),
    FULL_CONFIG_V2_PATH.as_posix(): (
        "12eba0287733a1ce5284a1d9ba3957627671e4e03e7fd6799bb0faccf40df22d"
    ),
}
_V2_CANONICAL_HASHES = {
    QUALIFICATION_CONFIG_V2_PATH.as_posix(): (
        "cf9bd36947d5c4946903ec0061be84f7616e53c8f1a825810ec8c0ddcfd4e7bd"
    ),
    FULL_CONFIG_V2_PATH.as_posix(): (
        "14e1fb3fb60dffc28d2055a71de173c6b58dfa99930e153e0621cf1ab96ab4d6"
    ),
}
_V2_ARTIFACT_HASHES = {
    "comparison.json": "c55869a81f433c9762bc9003c58c503cf5ac25228e1bf6acdebbf65494436844",
    "decision.json": "9f3cc0d6aa15b6173c2b318a2e8fa10fd7aeab36d88b06089fe209c208847ef6",
    "findings.md": "fd873f1c3c731c8693fb10e8999b2dba7a40ee4d860c8e2c1ca8010006bb2f07",
    "gpt41/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gpt41/predictions.jsonl": "96c30aaddf64c67c7cecc551322787cade020b94d35f041bf796eac3618714a6",
    "gpt41/run.json": "bbcb7d075e02613d26d82f6e65510519170d8046d1767262e7a9ab86b0eb1a5d",
    "gpt41-mini/failures.jsonl": "2971dca7880c93fa5baffe39faaee31f0ff0e121504e003a460c0d05f6722f3c",
    "gpt41-mini/predictions.jsonl": "0adcc4ca501daee32499f182e36e8ad285fd3cced009ed68d8e92fb04b5cb925",
    "gpt41-mini/run.json": "b92c5f8ae63147eaec9b154dbb370429d127d8a417ae410254cc0bc661371842",
    "manifest.json": "e1670f586131e0c285a70f17f0acf510f3a56b58dbf12c95581a5aff52b95cd7",
}
_REUSE = (
    (1, 1, "user_001", "scaled_user_001_conversation_001", "5a6cacb958f90fa089c03bf9dd1a092aed427c7a264baff6b135a3147812f75b"),
    (2, 6, "user_001", "scaled_user_001_email_002", "d4798abd56a1554e4cd73339c1f3063677d588d057c07611e5a9963ce25b1013"),
    (3, 7, "user_001", "scaled_user_001_chat_002", "aa50d026030797c34c64f98866a71d170e67270e66c3f4a513512672c370ab82"),
    (4, 4, "user_001", "scaled_user_001_calendar_001", "ca14acc2120e57be1270650e1f18856d76dbd69b90dca9e1c887e8b595e7009c"),
)


@dataclass
class FallbackBudget:
    plan: Step35StagePlan
    positions: tuple[int, ...]
    cursor: int = 0
    actual_spend_usd: Decimal = Decimal(0)

    @property
    def model(self):
        return self.plan.config.models[0]

    @property
    def remaining_hard_usd(self) -> Decimal:
        return sum(
            (
                step35_cost(
                    self.model,
                    self.plan.maximum_input_tokens[self.model.label][position - 1],
                    MAX_OUTPUT_TOKENS,
                )
                for position in self.positions[self.cursor :]
            ),
            Decimal(0),
        )

    @property
    def projected_cumulative_usd(self) -> Decimal:
        return PRIOR_SPEND_USD + self.actual_spend_usd + self.remaining_hard_usd

    def before_request(self, position: int) -> None:
        if self.cursor >= len(self.positions) or self.positions[self.cursor] != position:
            raise Step35RunError("fallback request order changed")
        if self.projected_cumulative_usd > self.plan.config.cumulative_authorization_usd:
            raise Step35RunError("the next fallback request could exceed the cumulative cap")

    def charge(self, position: int, input_tokens: int, output_tokens: int) -> None:
        self.before_request(position)
        maximum_input = self.plan.maximum_input_tokens[self.model.label][position - 1]
        if input_tokens > maximum_input or output_tokens > MAX_OUTPUT_TOKENS:
            raise Step35RunError("provider usage exceeded the fallback reservation")
        self.actual_spend_usd += step35_cost(self.model, input_tokens, output_tokens)
        self.cursor += 1
        if self.projected_cumulative_usd > self.plan.config.cumulative_authorization_usd:
            raise Step35RunError("actual usage leaves insufficient fallback budget")

    def record(self) -> dict[str, object]:
        return {
            "prior_spend_usd": additive_cost_text(PRIOR_SPEND_USD),
            "actual_new_spend_usd": additive_cost_text(self.actual_spend_usd),
            "cumulative_actual_spend_usd": additive_cost_text(
                PRIOR_SPEND_USD + self.actual_spend_usd
            ),
            "remaining_hard_maximum_usd": additive_cost_text(self.remaining_hard_usd),
            "projected_cumulative_hard_maximum_usd": additive_cost_text(
                self.projected_cumulative_usd
            ),
            "requests_charged": self.cursor,
            "requests_reserved": len(self.positions),
            "cost_cap_usd": additive_cost_text(
                self.plan.config.cumulative_authorization_usd
            ),
        }


def prepare_fallback(
    repo_root: str | Path = ".",
    config_path: str | Path = FALLBACK_CONFIG_PATH,
) -> tuple[Step35StagePlan, list[dict[str, object]], tuple[int, ...]]:
    """Validate every predecessor binding without opening development gold."""

    root = Path(repo_root).resolve()
    plan = prepare_step35_stage(config_path, root)
    if Path(config_path) != FALLBACK_CONFIG_PATH:
        raise Step35RunError("the fallback config path is not allowlisted")
    if (
        plan.config.configuration_version
        != "phase4-input-development-gpt41-fallback-v1"
        or plan.config.maximum_retry_requests != 0
        or plan.config.prior_spend_usd != PRIOR_SPEND_USD
        or plan.config.cumulative_authorization_usd != CUMULATIVE_HARD_MAXIMUM_USD
        or len(plan.config.models) != 1
        or plan.config.models[0].label != "gpt41"
        or plan.config.models[0].requested_model != "gpt-4.1-2025-04-14"
        or plan.config.models[0].resolved_model != "gpt-4.1-2025-04-14"
        or plan.config.models[0].output_directory != FALLBACK_OUTPUT_ROOT.as_posix()
    ):
        raise Step35RunError("fallback model, spend, retry, or output contract changed")

    predecessor = plan.config.predecessor
    if not isinstance(predecessor, dict):
        raise Step35RunError("fallback predecessor binding is missing")
    if (
        predecessor.get("fallback_guidance_version") != FALLBACK_GUIDANCE_VERSION
        or predecessor.get("decision_envelope_sha256") != FALLBACK_ENVELOPE_SHA256
        or predecessor.get("qualification_config_path")
        != QUALIFICATION_CONFIG_V2_PATH.as_posix()
        or predecessor.get("full_config_path") != FULL_CONFIG_V2_PATH.as_posix()
        or predecessor.get("qualification_root") != QUALIFICATION_ROOT_V2.as_posix()
        or predecessor.get("configuration_file_sha256") != _V2_CONFIG_HASHES
        or predecessor.get("configuration_canonical_sha256") != _V2_CANONICAL_HASHES
        or predecessor.get("artifact_sha256") != _V2_ARTIFACT_HASHES
        or predecessor.get("spend")
        != {
            "v1_charged_usd": "0.0196460",
            "v2_charged_usd": "0.0501504",
            "prior_spend_usd": "0.0697964",
        }
    ):
        raise Step35RunError("fallback predecessor metadata changed")
    for path, expected in _V2_CONFIG_HASHES.items():
        if _file_sha256(root / path) != expected:
            raise Step35RunError(f"frozen predecessor config changed: {path}")
    for path, expected in _V2_ARTIFACT_HASHES.items():
        if _file_sha256(root / QUALIFICATION_ROOT_V2 / path) != expected:
            raise Step35RunError(f"frozen predecessor artifact changed: {path}")

    qualification = prepare_step35_stage(QUALIFICATION_CONFIG_V2_PATH, root)
    prior_full = prepare_step35_stage(FULL_CONFIG_V2_PATH, root)
    if (
        qualification.config.configuration_sha256
        != _V2_CANONICAL_HASHES[QUALIFICATION_CONFIG_V2_PATH.as_posix()]
        or prior_full.config.configuration_sha256
        != _V2_CANONICAL_HASHES[FULL_CONFIG_V2_PATH.as_posix()]
        or plan.config.case_order != prior_full.config.case_order
        or plan.config.prompt_sha256 != prior_full.config.prompt_sha256
        or plan.config.schema_sha256 != prior_full.config.schema_sha256
        or plan.config.predicate_registry_sha256
        != prior_full.config.predicate_registry_sha256
        or plan.config.generation_settings != prior_full.config.generation_settings
    ):
        raise Step35RunError("fallback source, prompt, schema, registry, or settings changed")

    gpt_run = _read_json(root / QUALIFICATION_ROOT_V2 / "gpt41/run.json")
    mini_run = _read_json(root / QUALIFICATION_ROOT_V2 / "gpt41-mini/run.json")
    gpt_predictions = _read_jsonl(
        root / QUALIFICATION_ROOT_V2 / "gpt41/predictions.jsonl"
    )
    gpt_failures = _read_jsonl(root / QUALIFICATION_ROOT_V2 / "gpt41/failures.jsonl")
    mini_predictions = _read_jsonl(
        root / QUALIFICATION_ROOT_V2 / "gpt41-mini/predictions.jsonl"
    )
    mini_failures = _read_jsonl(
        root / QUALIFICATION_ROOT_V2 / "gpt41-mini/failures.jsonl"
    )
    if (
        gpt_run.get("status") != "generated"
        or gpt_run.get("requested_model") != "gpt-4.1-2025-04-14"
        or gpt_run.get("resolved_model") != "gpt-4.1-2025-04-14"
        or len(gpt_predictions) != 4
        or gpt_failures
        or len(mini_predictions) != 3
        or len(mini_failures) != 1
        or mini_run.get("status") != "failed"
    ):
        raise Step35RunError("v2 qualification is not eligible for fallback reuse")
    failure = mini_failures[0]
    expected_failure = predecessor.get("failure")
    if (
        not isinstance(expected_failure, dict)
        or failure.get("position") != expected_failure.get("position")
        or failure.get("source_id") != expected_failure.get("source_id")
        or failure.get("stage") != expected_failure.get("stage")
        or failure.get("provider_metadata", {}).get("returned_model")
        != "gpt-4.1-mini-2025-04-14"
    ):
        raise Step35RunError("v2 failure class or source changed")

    v2_cost = _exact_attempt_cost(gpt_predictions + mini_predictions + mini_failures)
    if v2_cost != Decimal("0.0501504"):
        raise Step35RunError("v2 charged spend changed")
    reuse_config = predecessor.get("reuse")
    expected_reuse = [
        {
            "qualification_position": q_position,
            "full_position": full_position,
            "user_id": user_id,
            "source_id": source_id,
            "request_body_sha256": request_hash,
        }
        for q_position, full_position, user_id, source_id, request_hash in _REUSE
    ]
    if reuse_config != expected_reuse:
        raise Step35RunError("fallback reuse map changed")

    reused: list[dict[str, object]] = []
    for q_position, full_position, user_id, source_id, request_hash in _REUSE:
        prediction = gpt_predictions[q_position - 1]
        source = plan.sources[full_position - 1]
        metadata = prediction.get("provider_metadata")
        if (
            prediction.get("position") != q_position
            or prediction.get("user_id") != user_id
            or prediction.get("source_id") != source_id
            or source.user_id != user_id
            or source.source_id != source_id
            or not isinstance(metadata, dict)
            or metadata.get("returned_model") != "gpt-4.1-2025-04-14"
            or qualification.request_body_sha256["gpt41"][q_position - 1]
            != request_hash
            or plan.request_body_sha256["gpt41"][full_position - 1]
            != request_hash
            or hashlib.sha256(
                qualification.prompts[q_position - 1].encode("utf-8")
            ).hexdigest()
            != hashlib.sha256(plan.prompts[full_position - 1].encode("utf-8")).hexdigest()
        ):
            raise Step35RunError("a reused GPT-4.1 result is incompatible")
        validate_phase4_claim_file(prediction.get("claims", []), (source,), plan.registry)
        reused.append(
            {
                "position": full_position,
                "user_id": user_id,
                "source_id": source_id,
                "origin": "reused_v2_gpt41",
                "request_body_sha256": request_hash,
                "claims": prediction["claims"],
                "provider_metadata": _seven_decimal_metadata(metadata, plan),
            }
        )
    reused.sort(key=lambda item: item["position"])
    reused_positions = {item["position"] for item in reused}
    remaining = tuple(
        position for position in range(1, len(plan.sources) + 1)
        if position not in reused_positions
    )
    if len(reused) != 4 or len(remaining) != 16:
        raise Step35RunError("fallback must reuse four results and request sixteen")
    return plan, reused, remaining


def dry_run_fallback(
    repo_root: str | Path = ".",
    config_path: str | Path = FALLBACK_CONFIG_PATH,
    stdout: TextIO | None = None,
) -> dict[str, object]:
    plan, reused, remaining = prepare_fallback(repo_root, config_path)
    model = plan.config.models[0]
    exact_inputs = sum(
        plan.estimated_input_tokens[model.label][position - 1]
        for position in remaining
    )
    maximum_inputs = sum(
        plan.maximum_input_tokens[model.label][position - 1]
        for position in remaining
    )
    expected_outputs = len(remaining) * plan.config.expected_output_tokens_per_request
    maximum_outputs = len(remaining) * MAX_OUTPUT_TOKENS
    current_expected = step35_cost(model, exact_inputs, expected_outputs)
    current_hard = step35_cost(model, maximum_inputs, maximum_outputs)
    cumulative_hard = PRIOR_SPEND_USD + current_hard
    if current_hard != NEW_HARD_MAXIMUM_USD or cumulative_hard != CUMULATIVE_HARD_MAXIMUM_USD:
        raise Step35RunError("fallback hard-maximum math changed")
    requests = _model_request_plan(plan, model)
    record = {
        "dry_run": True,
        "provider_calls": 0,
        "output_writes": 0,
        "configuration_version": plan.config.configuration_version,
        "configuration_path": plan.config_path.as_posix(),
        "configuration_file_sha256": _file_sha256(
            plan.repo_root / plan.config_path
        ),
        "configuration_canonical_sha256": plan.config.configuration_sha256,
        "fallback_guidance_version": FALLBACK_GUIDANCE_VERSION,
        "decision_envelope_sha256": FALLBACK_ENVELOPE_SHA256,
        "requested_model": model.requested_model,
        "resolved_model": model.resolved_model,
        "generation_settings": dict(plan.config.generation_settings),
        "retry_request_count": 0,
        "reused_prediction_count": len(reused),
        "new_request_attempt_count": len(remaining),
        "case_count": len(plan.sources),
        "reused_source_ids": [item["source_id"] for item in reused],
        "new_source_ids": [plan.sources[position - 1].source_id for position in remaining],
        "token_counter": {
            "version": plan.config.token_counter_version,
            "package": "tiktoken",
            "package_version": "0.13.0",
            "encoding": plan.config.token_encoding,
            "reserve_per_request": plan.config.input_token_reserve_per_request,
            "exact_input_tokens": exact_inputs,
            "maximum_input_tokens": maximum_inputs,
            "expected_output_tokens": expected_outputs,
            "maximum_output_tokens": maximum_outputs,
        },
        "requests": [requests[position - 1] for position in remaining],
        "prior_spend_usd": additive_cost_text(PRIOR_SPEND_USD),
        "current_expected_cost_usd": additive_cost_text(current_expected),
        "current_hard_maximum_cost_usd": additive_cost_text(current_hard),
        "cumulative_hard_maximum_cost_usd": additive_cost_text(cumulative_hard),
        "cost_cap_usd": additive_cost_text(plan.config.cumulative_authorization_usd),
        "fits_cost_cap": cumulative_hard <= plan.config.cumulative_authorization_usd,
        "output_root": FALLBACK_OUTPUT_ROOT.as_posix(),
        "checkpoint_after_each_response": True,
        "resume_policy": "Resume only the exact missing suffix after four verified reused results. Never replay a success or failure.",
        "gold_loading_policy": "Open development gold only after all 20 predictions are persisted.",
        "oracle_gold_test_review_in_provider_payload": False,
        "prospective_failure_diagnostics": ["code", "location"],
        "predecessor": plan.config.predecessor,
    }
    assert_no_secrets(record, "Step 3.5 GPT-4.1 fallback dry run")
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=stdout or sys.stdout)
    return record


def execute_fallback(
    *,
    repo_root: str | Path,
    client: object,
    resume: bool = False,
    config_path: str | Path = FALLBACK_CONFIG_PATH,
) -> dict[str, object]:
    plan, reused, remaining = prepare_fallback(repo_root, config_path)
    preflight = dry_run_fallback(repo_root, config_path, stdout=_NullWriter())
    if preflight["fits_cost_cap"] is not True:
        raise Step35RunError("fallback hard maximum exceeds the frozen cap")
    output = plan.repo_root / FALLBACK_OUTPUT_ROOT
    budget = FallbackBudget(plan, remaining)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise Step35RunError("refusing to overwrite non-empty fallback output")
        predictions, failures, started_at, completed_scores = _load_checkpoint(
            output, plan, reused, remaining, budget
        )
        if completed_scores is not None:
            return _result(predictions, failures, budget, True, completed_scores)
    else:
        output.mkdir(parents=True, exist_ok=True)
        predictions, failures, started_at = list(reused), [], _utc_now()
        _checkpoint(output, plan, predictions, failures, budget, started_at, "running")
    if failures:
        raise Step35RunError("a failed fallback request cannot resume")
    completed = {item["position"] for item in predictions}
    model = plan.config.models[0]
    for position in remaining:
        if position in completed:
            continue
        source = plan.sources[position - 1]
        maximum_input = plan.maximum_input_tokens[model.label][position - 1]
        budget.before_request(position)
        try:
            raw, metadata = getattr(client, "complete_with_metadata")(
                system_prompt=plan.system_prompt,
                user_prompt=plan.prompts[position - 1],
            )
        except Exception:
            metadata_record = _missing_metadata(plan, position)
            budget.charge(position, maximum_input, MAX_OUTPUT_TOKENS)
            failures.append(
                _failure(plan, position, "provider", "provider_no_output", "response", metadata_record)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        if not isinstance(metadata, OpenAIResponseMetadata):
            metadata_record = _missing_metadata(plan, position)
            budget.charge(position, maximum_input, MAX_OUTPUT_TOKENS)
            failures.append(
                _failure(plan, position, "model_mismatch", "metadata_incompatible", "response.metadata", metadata_record)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        charged_input = metadata.input_tokens if metadata.input_tokens is not None else maximum_input
        charged_output = metadata.output_tokens if metadata.output_tokens is not None else MAX_OUTPUT_TOKENS
        try:
            budget.charge(position, charged_input, charged_output)
        except Step35RunError:
            metadata_record = _metadata(plan, position, metadata, charged_input, charged_output)
            failures.append(
                _failure(plan, position, "budget", "usage_exceeded_reservation", "response.usage", metadata_record)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        metadata_record = _metadata(plan, position, metadata, charged_input, charged_output)
        if metadata.returned_model != model.resolved_model:
            failures.append(
                _failure(plan, position, "model_mismatch", "returned_model_mismatch", "response.model", metadata_record)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        try:
            atomic = validate_atomic_response(source, raw, metadata, registry=plan.registry)
            claims = build_phase4_source_claims(
                source, [_atomic_record(claim) for claim in atomic.claims], plan.registry
            )
        except AtomicExtractionValidationError as error:
            diagnostics = [
                {"code": item.code, "location": item.location}
                for item in sanitized_validation_diagnostics(error)
            ]
            failures.append(
                _failure(plan, position, "validation", None, None, metadata_record, diagnostics)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        except Phase4InputValidationError:
            failures.append(
                _failure(plan, position, "persistence", "claim_rejected", "claims", metadata_record)
            )
            _checkpoint(output, plan, predictions, failures, budget, started_at, "failed")
            return _result(predictions, failures, budget, False)
        predictions.append(
            {
                "position": position,
                "user_id": source.user_id,
                "source_id": source.source_id,
                "origin": "provider",
                "request_body_sha256": plan.request_body_sha256[model.label][position - 1],
                "claims": [phase4_claim_record(claim) for claim in claims],
                "provider_metadata": metadata_record,
            }
        )
        predictions.sort(key=lambda item: item["position"])
        _checkpoint(output, plan, predictions, failures, budget, started_at, "running")
    if len(predictions) != 20:
        raise Step35RunError("fallback generation ended before all 20 predictions")
    _checkpoint(output, plan, predictions, failures, budget, started_at, "generated")
    return _score_and_finalize(plan, predictions, failures, budget, started_at, output)


def _score_and_finalize(plan, predictions, failures, budget, started_at, output):
    claims = [claim for prediction in predictions for claim in prediction["claims"]]
    validated = validate_phase4_claim_file(claims, plan.sources, plan.registry)
    gold = load_scaled_development_gold(plan.repo_root, plan.all_sources, plan.registry)
    predictions_by_source = {
        item["source_id"]: validate_phase4_claim_file(
            item["claims"], (plan.sources[item["position"] - 1],), plan.registry
        )
        for item in predictions
    }
    scores = score_scaled_development(gold, predictions_by_source, plan.sources, plan.registry)
    _write_jsonl(output / "claims.jsonl", [phase4_claim_record(item) for item in validated])
    _write_jsonl(output / "case_scores.jsonl", scores["case_results"])
    _write_json(output / "scores.json", scores)
    limitations = {
        "items": [
            "This fallback covers two synthetic development users only.",
            "Four predictions came from the compatible v2 GPT-4.1 qualification run.",
            "Frozen test users were not opened or processed.",
        ]
    }
    _write_json(output / "known_limitations.json", limitations)
    evidence_index = _build_evidence_index(plan, validated)
    _write_json(output / "evidence_index.json", evidence_index)
    _checkpoint(output, plan, predictions, failures, budget, started_at, "completed")
    files = {
        name: _file_sha256(output / name)
        for name in (
            "claims.jsonl", "predictions.jsonl", "failures.jsonl", "case_scores.jsonl",
            "scores.json", "run.json", "known_limitations.json", "evidence_index.json"
        )
    }
    manifest = {
        "manifest_version": "phase4-input-gpt41-fallback-manifest-v1",
        "configuration": {
            "path": plan.config_path.as_posix(),
            "file_sha256": _file_sha256(plan.repo_root / plan.config_path),
            "canonical_sha256": plan.config.configuration_sha256,
        },
        "predecessor": plan.config.predecessor,
        "runtime_inputs": _runtime_input_manifest(plan),
        "case_order": [
            {"position": index, "user_id": user_id, "source_id": source_id}
            for index, (user_id, source_id) in enumerate(plan.config.case_order, 1)
        ],
        "request_plan": _model_request_plan(plan, plan.config.models[0]),
        "model": {
            "requested_model": plan.config.models[0].requested_model,
            "resolved_model": plan.config.models[0].resolved_model,
            "returned_models": sorted({
                item["provider_metadata"]["returned_model"] for item in predictions
            }),
            "reused_prediction_count": 4,
            "new_request_count": 16,
        },
        "token_counter": {
            "version": plan.config.token_counter_version,
            "package": "tiktoken",
            "package_version": "0.13.0",
            "encoding": plan.config.token_encoding,
            "reserve_per_request": plan.config.input_token_reserve_per_request,
        },
        "budget": budget.record(),
        "repository": {
            "commit": _git_commit(),
            "worktree_dirty": _git_dirty(),
            "worktree_status_sha256": hashlib.sha256(
                _git_status_text().encode("utf-8")
            ).hexdigest(),
        },
        "gold_access": {
            "opened_only_after_all_20_predictions": True,
            "sha256": plan.config.gold_claim_file_sha256,
        },
        "reverse_source_to_claim_evidence_index": evidence_index["validation"],
        "files": files,
    }
    _write_json(output / "manifest.json", manifest)
    return _result(predictions, failures, budget, True, scores)


def _load_checkpoint(output, plan, reused, remaining, budget):
    allowed = {
        "predictions.jsonl", "failures.jsonl", "run.json", "claims.jsonl",
        "case_scores.jsonl", "scores.json", "known_limitations.json",
        "evidence_index.json", "manifest.json",
    }
    if any(item.name not in allowed or not item.is_file() for item in output.iterdir()):
        raise Step35RunError("fallback checkpoint contains unexpected artifacts")
    predictions = _read_jsonl(output / "predictions.jsonl")
    failures = _read_jsonl(output / "failures.jsonl")
    run = _read_json(output / "run.json")
    hashes = run.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise Step35RunError("fallback checkpoint hashes are missing")
    for name, expected in hashes.items():
        if _file_sha256(output / name) != expected:
            raise Step35RunError(f"fallback checkpoint changed: {name}")
    if failures:
        raise Step35RunError("a failed fallback request cannot resume")
    by_position = {item.get("position"): item for item in predictions}
    if len(by_position) != len(predictions):
        raise Step35RunError("fallback checkpoint contains duplicate positions")
    for expected in reused:
        if by_position.get(expected["position"]) != expected:
            raise Step35RunError("a reused fallback prediction changed")
    provider_positions = sorted(
        item["position"] for item in predictions if item.get("origin") == "provider"
    )
    if provider_positions != list(remaining[: len(provider_positions)]):
        raise Step35RunError("fallback checkpoint provider order changed")
    for item in predictions:
        position = item["position"]
        source = plan.sources[position - 1]
        metadata = item.get("provider_metadata")
        expected_origin = (
            "reused_v2_gpt41" if position not in remaining else "provider"
        )
        if (
            item.get("source_id") != source.source_id
            or item.get("user_id") != source.user_id
            or item.get("origin") != expected_origin
            or item.get("request_body_sha256")
            != plan.request_body_sha256["gpt41"][position - 1]
            or not isinstance(metadata, dict)
            or metadata.get("returned_model") != plan.config.models[0].resolved_model
            or not isinstance(metadata.get("charged_input_tokens"), int)
            or isinstance(metadata.get("charged_input_tokens"), bool)
            or not isinstance(metadata.get("charged_output_tokens"), int)
            or isinstance(metadata.get("charged_output_tokens"), bool)
            or metadata.get("charged_input_tokens") < 0
            or metadata.get("charged_output_tokens") < 0
            or metadata.get("charged_input_tokens")
            > plan.maximum_input_tokens["gpt41"][position - 1]
            or metadata.get("charged_output_tokens") > MAX_OUTPUT_TOKENS
            or metadata.get("charged_cost_usd")
            != additive_cost_text(
                step35_cost(
                    plan.config.models[0],
                    metadata.get("charged_input_tokens", -1),
                    metadata.get("charged_output_tokens", -1),
                )
            )
        ):
            raise Step35RunError(
                "fallback checkpoint source, model, request, or cost changed"
            )
        validate_phase4_claim_file(item.get("claims", []), (source,), plan.registry)
    for position in provider_positions:
        metadata = by_position[position]["provider_metadata"]
        budget.charge(
            position,
            metadata["charged_input_tokens"],
            metadata["charged_output_tokens"],
        )
    started_at = run.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise Step35RunError("fallback checkpoint start time changed")
    if len(predictions) == 20:
        scores = _verify_completed_checkpoint(
            output, plan, predictions, failures, run, budget
        )
        return predictions, failures, started_at, scores
    if any(name in allowed - {"predictions.jsonl", "failures.jsonl", "run.json"} for name in {item.name for item in output.iterdir()}):
        raise Step35RunError("unfinished fallback checkpoint contains final artifacts")
    return predictions, failures, started_at, None


def _verify_completed_checkpoint(output, plan, predictions, failures, run, budget):
    artifact_names = {
        "claims.jsonl",
        "predictions.jsonl",
        "failures.jsonl",
        "case_scores.jsonl",
        "scores.json",
        "run.json",
        "known_limitations.json",
        "evidence_index.json",
    }
    if {item.name for item in output.iterdir()} != artifact_names | {"manifest.json"}:
        raise Step35RunError("completed fallback artifact set changed")
    expected_run_artifacts = {
        name: _file_sha256(output / name)
        for name in artifact_names - {"run.json"}
    }
    model = plan.config.models[0]
    if (
        failures
        or run.get("status") != "completed"
        or run.get("configuration_version") != plan.config.configuration_version
        or run.get("configuration_sha256") != plan.config.configuration_sha256
        or run.get("requested_model") != model.requested_model
        or run.get("resolved_model") != model.resolved_model
        or run.get("successful_source_ids")
        != [source.source_id for source in plan.sources]
        or run.get("failure_count") != 0
        or run.get("artifact_sha256") != expected_run_artifacts
        or run.get("budget") != budget.record()
    ):
        raise Step35RunError("completed fallback run contract changed")

    manifest = _read_json(output / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != artifact_names:
        raise Step35RunError("completed fallback manifest file set changed")
    for name, expected in files.items():
        if _file_sha256(output / name) != expected:
            raise Step35RunError(f"completed fallback artifact changed: {name}")

    expected_case_order = [
        {"position": index, "user_id": user_id, "source_id": source_id}
        for index, (user_id, source_id) in enumerate(plan.config.case_order, 1)
    ]
    expected_model = {
        "requested_model": model.requested_model,
        "resolved_model": model.resolved_model,
        "returned_models": [model.resolved_model],
        "reused_prediction_count": 4,
        "new_request_count": 16,
    }
    expected_counter = {
        "version": plan.config.token_counter_version,
        "package": "tiktoken",
        "package_version": "0.13.0",
        "encoding": plan.config.token_encoding,
        "reserve_per_request": plan.config.input_token_reserve_per_request,
    }
    if (
        manifest.get("manifest_version")
        != "phase4-input-gpt41-fallback-manifest-v1"
        or manifest.get("configuration")
        != {
            "path": plan.config_path.as_posix(),
            "file_sha256": _file_sha256(plan.repo_root / plan.config_path),
            "canonical_sha256": plan.config.configuration_sha256,
        }
        or manifest.get("predecessor") != plan.config.predecessor
        or manifest.get("runtime_inputs") != _runtime_input_manifest(plan)
        or manifest.get("case_order") != expected_case_order
        or manifest.get("request_plan") != _model_request_plan(plan, model)
        or manifest.get("model") != expected_model
        or manifest.get("token_counter") != expected_counter
        or manifest.get("budget") != budget.record()
        or manifest.get("gold_access")
        != {
            "opened_only_after_all_20_predictions": True,
            "sha256": plan.config.gold_claim_file_sha256,
        }
    ):
        raise Step35RunError("completed fallback manifest contract changed")

    claim_records = _read_jsonl(output / "claims.jsonl")
    expected_claim_records = [
        claim for prediction in predictions for claim in prediction["claims"]
    ]
    if claim_records != expected_claim_records:
        raise Step35RunError("completed fallback claims differ from predictions")
    validated = validate_phase4_claim_file(claim_records, plan.sources, plan.registry)
    evidence_index = _build_evidence_index(plan, validated)
    if (
        _read_json(output / "evidence_index.json") != evidence_index
        or manifest.get("reverse_source_to_claim_evidence_index")
        != evidence_index["validation"]
    ):
        raise Step35RunError("completed fallback provenance changed")

    gold = load_scaled_development_gold(plan.repo_root, plan.all_sources, plan.registry)
    predictions_by_source = {
        item["source_id"]: validate_phase4_claim_file(
            item["claims"], (plan.sources[item["position"] - 1],), plan.registry
        )
        for item in predictions
    }
    expected_scores = score_scaled_development(
        gold, predictions_by_source, plan.sources, plan.registry
    )
    if (
        _read_json(output / "scores.json") != expected_scores
        or _read_jsonl(output / "case_scores.jsonl")
        != expected_scores["case_results"]
    ):
        raise Step35RunError("completed fallback scores changed")
    return expected_scores


def _checkpoint(output, plan, predictions, failures, budget, started_at, status):
    _write_jsonl(output / "predictions.jsonl", sorted(predictions, key=lambda item: item["position"]))
    _write_jsonl(output / "failures.jsonl", failures)
    artifacts = {
        "predictions.jsonl": _file_sha256(output / "predictions.jsonl"),
        "failures.jsonl": _file_sha256(output / "failures.jsonl"),
    }
    if status == "completed":
        for name in ("claims.jsonl", "case_scores.jsonl", "scores.json", "known_limitations.json", "evidence_index.json"):
            artifacts[name] = _file_sha256(output / name)
    run = {
        "run_format_version": "phase4-input-gpt41-fallback-run-v1",
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "configuration_version": plan.config.configuration_version,
        "configuration_sha256": plan.config.configuration_sha256,
        "requested_model": plan.config.models[0].requested_model,
        "resolved_model": plan.config.models[0].resolved_model,
        "successful_source_ids": [item["source_id"] for item in sorted(predictions, key=lambda item: item["position"])],
        "failure_count": len(failures),
        "artifact_sha256": artifacts,
        "budget": budget.record(),
    }
    _write_json(output / "run.json", run)


def _metadata(plan, position, metadata, charged_input, charged_output):
    model = plan.config.models[0]
    return {
        "response_id": metadata.response_id,
        "returned_model": metadata.returned_model,
        "request_id": metadata.request_id,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "total_tokens": metadata.total_tokens,
        "charged_input_tokens": charged_input,
        "charged_output_tokens": charged_output,
        "charged_cost_usd": additive_cost_text(
            step35_cost(model, charged_input, charged_output)
        ),
    }


def _missing_metadata(plan, position):
    maximum = plan.maximum_input_tokens["gpt41"][position - 1]
    model = plan.config.models[0]
    return {
        "response_id": None, "returned_model": None, "request_id": None,
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "charged_input_tokens": maximum,
        "charged_output_tokens": MAX_OUTPUT_TOKENS,
        "charged_cost_usd": additive_cost_text(step35_cost(model, maximum, MAX_OUTPUT_TOKENS)),
    }


def _seven_decimal_metadata(metadata, plan):
    record = dict(metadata)
    record["charged_cost_usd"] = additive_cost_text(
        step35_cost(
            plan.config.models[0],
            record["charged_input_tokens"],
            record["charged_output_tokens"],
        )
    )
    return record


def _failure(plan, position, stage, code, location, metadata, diagnostics=None):
    source = plan.sources[position - 1]
    safe_diagnostics = diagnostics or [{"code": code, "location": location}]
    record = {
        "position": position,
        "user_id": source.user_id,
        "source_id": source.source_id,
        "stage": stage,
        "failure_categories": ["persistence" if stage == "persistence" else "execution"],
        "provider_metadata": metadata,
        "diagnostics": safe_diagnostics,
    }
    return {"failure_id": f"failure_{canonical_sha256(record)}", **record}


def _exact_attempt_cost(attempts):
    gpt41_input = gpt41_output = mini_input = mini_output = 0
    for item in attempts:
        metadata = item["provider_metadata"]
        if metadata["returned_model"] == "gpt-4.1-2025-04-14":
            gpt41_input += metadata["charged_input_tokens"]
            gpt41_output += metadata["charged_output_tokens"]
        else:
            mini_input += metadata["charged_input_tokens"]
            mini_output += metadata["charged_output_tokens"]
    return (
        Decimal(gpt41_input) * Decimal("2") / Decimal(1_000_000)
        + Decimal(gpt41_output) * Decimal("8") / Decimal(1_000_000)
        + Decimal(mini_input) * Decimal("0.4") / Decimal(1_000_000)
        + Decimal(mini_output) * Decimal("1.6") / Decimal(1_000_000)
    )


def _result(predictions, failures, budget, complete, scores=None):
    result = {
        "generation_complete": complete,
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "budget": budget.record(),
    }
    if scores is not None:
        result["scores"] = scores
    return result


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Step 3.5 GPT-4.1 fallback.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run == args.execute:
        raise SystemExit("choose exactly one of --dry-run or --execute")
    if args.resume and not args.execute:
        raise SystemExit("--resume requires --execute")
    if args.dry_run:
        dry_run_fallback(config_path=args.config)
        return 0
    key = load_env_value(args.env_file, "OPENAI_API_KEY")
    plan, _, _ = prepare_fallback(config_path=args.config)
    model = plan.config.models[0]
    client = OpenAIResponsesClient(
        api_key=key,
        model=model.requested_model,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text_format=plan.text_format,
    )
    execute_fallback(repo_root=".", client=client, resume=args.resume, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
