"""Run the bounded Step 3.5 qualification and Phase 4 input stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence, TextIO

from evaluation.openai_client import (
    OpenAIModelMismatchError,
    OpenAIResponseMetadata,
    OpenAIResponsesClient,
    load_env_value,
)
from evaluation.run_config import assert_no_secrets, canonical_sha256

from .atomic import AtomicExtractionValidationError, validate_atomic_response
from .phase4_input import (
    Phase4InputClaim,
    Phase4InputValidationError,
    build_phase4_source_claims,
    phase4_claim_record,
    validate_phase4_claim_file,
)
from .predicate_registry import PredicateRegistry, load_predicate_registry
from .prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from .run_safety import (
    AtomicRunConfigError,
    STEP35_TOKEN_ENCODING,
    Step35ModelPlan,
    Step35RunConfig,
    count_step35_request_tokens,
    cost_text,
    load_step35_run_config,
    reserve_step35_input_tokens,
    step35_request_sha256,
    step35_cost,
)
from .scaled_scoring import (
    load_scaled_development_gold,
    qualification_gate,
    score_scaled_development,
)
from .scaled_source import (
    DEVELOPMENT_SOURCE_REFS,
    QUALIFICATION_SOURCE_REFS,
    load_scaled_development_sources,
    select_scaled_sources,
)
from .schema import ATOMIC_EXTRACTION_SCHEMA_VERSION, atomic_extraction_text_format
from .source import ExtractionSource


QUALIFICATION_CONFIG_V1_PATH = Path(
    "configs/extraction/phase4_input_model_qualification_v1.json"
)
FULL_CONFIG_V1_PATH = Path("configs/extraction/phase4_input_development_v1.json")
QUALIFICATION_CONFIG_V2_PATH = Path(
    "configs/extraction/phase4_input_model_qualification_v2.json"
)
FULL_CONFIG_V2_PATH = Path("configs/extraction/phase4_input_development_v2.json")
FALLBACK_CONFIG_PATH = Path(
    "configs/extraction/phase4_input_development_gpt41_fallback_v1.json"
)
QUALIFICATION_ROOT_V1 = Path(
    "results/phase3/atomic-extraction-step35-model-qualification-v1"
)
FULL_OUTPUT_ROOT_V1 = Path("results/phase3/phase4-input-development-v1")
QUALIFICATION_ROOT_V2 = Path(
    "results/phase3/atomic-extraction-step35-model-qualification-v2"
)
FULL_OUTPUT_ROOT_V2 = Path("results/phase3/phase4-input-development-v2")
QUALIFICATION_CONFIG_PATH = QUALIFICATION_CONFIG_V2_PATH
FULL_CONFIG_PATH = FULL_CONFIG_V2_PATH
QUALIFICATION_ROOT = QUALIFICATION_ROOT_V2
FULL_OUTPUT_ROOT = FULL_OUTPUT_ROOT_V2
_RUNTIME_USER_PATH = Path("data/scaled-v1/runtime/users.jsonl")
_RUNTIME_SOURCE_PATH = Path("data/scaled-v1/runtime/sources.jsonl")
_GOLD_CLAIM_PATH = Path("data/scaled-v1/gold/claims.jsonl")
_MAX_OUTPUT_TOKENS = 1200
_MODEL_FILES = {"predictions.jsonl", "failures.jsonl", "run.json"}


class Step35RunError(RuntimeError):
    """Stop Step 3.5 before an unsafe call or artifact write."""


@dataclass(frozen=True)
class Step35RunFamily:
    release_version: str
    qualification_config_path: Path
    full_config_path: Path
    qualification_root: Path
    full_output_root: Path


_V1_RUN_FAMILY = Step35RunFamily(
    release_version="v1",
    qualification_config_path=QUALIFICATION_CONFIG_V1_PATH,
    full_config_path=FULL_CONFIG_V1_PATH,
    qualification_root=QUALIFICATION_ROOT_V1,
    full_output_root=FULL_OUTPUT_ROOT_V1,
)
_V2_RUN_FAMILY = Step35RunFamily(
    release_version="v2",
    qualification_config_path=QUALIFICATION_CONFIG_V2_PATH,
    full_config_path=FULL_CONFIG_V2_PATH,
    qualification_root=QUALIFICATION_ROOT_V2,
    full_output_root=FULL_OUTPUT_ROOT_V2,
)
_RUN_FAMILIES = (_V1_RUN_FAMILY, _V2_RUN_FAMILY)
_RECOVERY_GUIDANCE_VERSION = "step-3.5-recovery-guidance-v1"
_RECOVERY_ENVELOPE_SHA256 = (
    "bb84b4c8bef85be389675832b2c106f2fce2a10dc30e6e3f7e1aedd66d265501"
)
_RECOVERY_PRIOR_SPEND_USD = Decimal("0.019646")
_RECOVERY_CONFIG_SHA256 = {
    QUALIFICATION_CONFIG_V1_PATH.as_posix(): (
        "e2ba375f4c27878a9bd59cb0f1cc08ff71c80d9f0e4fd529759b4137c582ae7e"
    ),
    FULL_CONFIG_V1_PATH.as_posix(): (
        "896d9f0444f2bcdd28af72da315571cc2a37b4f8dd39306cddda50f6a911a85d"
    ),
}
_RECOVERY_ARTIFACT_SHA256 = {
    "comparison.json": "7118753f6306203fa63cb04d5584ea1f863d6e0c5b0d1e90d72676e4ede79c46",
    "decision.json": "e35f67f89306658f790330bdfa2a45087c995cf0f094c3c9f835cb3af3671fbd",
    "findings.md": "8e2e2ccbbac7bd59a3a3c3e1a5efbd3fa66d1f28805bea232a82c15635cce8f9",
    "gpt41/failures.jsonl": "431dc3f414a4828ed00055792c2815947ec5857bc564710f866c39c1ccf3f00f",
    "gpt41/predictions.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gpt41/run.json": "1887760f58c6b7db56c7ee98e2c67a9214d437a648625259120da27b71ac2589",
    "manifest.json": "a0d0dfba7016e6db2401f5ac075a414048c12c1f7636bf6242283a4688113d70",
}
_RECOVERY_FAILURE = {
    "failure_id": "failure_e1fb350019c4ce5713760a4adaa57293541ee47cdd022e18e74d1a190f2b510d",
    "model_label": "gpt41",
    "source_id": "scaled_user_001_conversation_001",
    "stage": "provider",
    "failure_class": "provider_no_output",
    "failure_categories": ["execution"],
    "provider_output_received": False,
    "charged_input_token_ceiling": 5023,
    "charged_output_token_ceiling": 1200,
    "charged_cost_usd": "0.019646",
}


@dataclass(frozen=True)
class Step35StagePlan:
    repo_root: Path
    config_path: Path
    config: Step35RunConfig
    registry: PredicateRegistry
    all_sources: tuple[ExtractionSource, ...]
    sources: tuple[ExtractionSource, ...]
    system_prompt: str
    prompts: tuple[str, ...]
    text_format: Mapping[str, object]
    estimated_input_tokens: Mapping[str, tuple[int, ...]]
    maximum_input_tokens: Mapping[str, tuple[int, ...]]
    request_body_sha256: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class Step35RequestReservation:
    model: Step35ModelPlan
    source_id: str
    maximum_input_tokens: int

    @property
    def maximum_cost_usd(self) -> Decimal:
        return step35_cost(
            self.model, self.maximum_input_tokens, _MAX_OUTPUT_TOKENS
        )


@dataclass
class Step35Budget:
    prior_spend_usd: Decimal
    authorization_usd: Decimal
    reservations: tuple[Step35RequestReservation, ...]
    cursor: int = 0
    actual_spend_usd: Decimal = Decimal(0)

    def before_request(
        self, model: Step35ModelPlan, source_id: str, maximum_input: int
    ) -> None:
        if self.cursor >= len(self.reservations):
            raise Step35RunError("the request ceiling was reached")
        expected = self.reservations[self.cursor]
        if (
            expected.model.label != model.label
            or expected.source_id != source_id
            or expected.maximum_input_tokens != maximum_input
        ):
            raise Step35RunError("the cumulative budget sequence changed")
        if self.projected_hard_maximum_usd > self.authorization_usd:
            raise Step35RunError("the next request could exceed the cumulative cap")

    def charge_response(self, metadata: Mapping[str, object]) -> None:
        if self.cursor >= len(self.reservations):
            raise Step35RunError("the response has no request reservation")
        expected = self.reservations[self.cursor]
        charged_input = metadata.get("charged_input_tokens")
        charged_output = metadata.get("charged_output_tokens")
        if (
            not isinstance(charged_input, int)
            or isinstance(charged_input, bool)
            or not isinstance(charged_output, int)
            or isinstance(charged_output, bool)
            or charged_input < 0
            or charged_output < 0
        ):
            raise Step35RunError("provider usage metadata is incompatible")
        cost = step35_cost(expected.model, charged_input, charged_output)
        self.actual_spend_usd += cost
        self.cursor += 1
        if (
            charged_input > expected.maximum_input_tokens
            or charged_output > _MAX_OUTPUT_TOKENS
        ):
            raise Step35RunError("provider usage exceeded the request reservation")
        if self.projected_hard_maximum_usd > self.authorization_usd:
            raise Step35RunError("actual usage leaves insufficient cumulative budget")

    @property
    def projected_hard_maximum_usd(self) -> Decimal:
        return (
            self.prior_spend_usd
            + self.actual_spend_usd
            + sum(
                (
                    item.maximum_cost_usd
                    for item in self.reservations[self.cursor :]
                ),
                Decimal(0),
            )
        )

    def record(self) -> dict[str, object]:
        return {
            "prior_spend_usd": cost_text(self.prior_spend_usd),
            "actual_spend_usd": cost_text(self.actual_spend_usd),
            "cumulative_actual_spend_usd": cost_text(
                self.prior_spend_usd + self.actual_spend_usd
            ),
            "requests_charged": self.cursor,
            "requests_reserved": len(self.reservations),
            "remaining_reserved_requests": len(self.reservations) - self.cursor,
            "projected_hard_maximum_usd": cost_text(
                self.projected_hard_maximum_usd
            ),
            "cost_cap_usd": cost_text(self.authorization_usd),
        }


def prepare_step35_stage(
    config_path: str | Path,
    repo_root: str | Path = ".",
) -> Step35StagePlan:
    """Prepare source-only prompts without opening scorer-side gold."""

    root = Path(repo_root).resolve()
    allowed_path = _allowed_config_path(root, config_path)
    try:
        config = load_step35_run_config(allowed_path, root)
    except AtomicRunConfigError as error:
        raise Step35RunError(str(error)) from error
    expected_refs = (
        QUALIFICATION_SOURCE_REFS
        if config.stage == "qualification"
        else DEVELOPMENT_SOURCE_REFS
    )
    if config.case_order != expected_refs:
        raise Step35RunError(f"{config.stage} case order changed")
    if config.dataset_version != "scaled_v1" or config.dataset_sha256 != (
        "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61"
    ):
        raise Step35RunError("scaled development dataset version or hash changed")
    expected_hashes = {
        _RUNTIME_USER_PATH: config.runtime_user_file_sha256,
        _RUNTIME_SOURCE_PATH: config.runtime_source_file_sha256,
    }
    for path, expected in expected_hashes.items():
        if _file_sha256(root / path) != expected:
            raise Step35RunError(f"protected input changed: {path.as_posix()}")
    registry_path = root / config.predicate_registry_path
    registry = load_predicate_registry(registry_path)
    if (
        registry.registry_version != config.predicate_registry_version
        or registry.content_sha256 != config.predicate_registry_sha256
    ):
        raise Step35RunError("predicate registry version or hash changed")
    if config.prompt_version != "atomic-extraction-v3":
        raise Step35RunError("Step 3.5 must retain the accepted v3 prompt")
    if config.schema_version != ATOMIC_EXTRACTION_SCHEMA_VERSION:
        raise Step35RunError("Step 3.5 schema version changed")

    all_sources = load_scaled_development_sources(root)
    sources = select_scaled_sources(all_sources, config.case_order)
    system_prompt = get_atomic_extraction_system_prompt(
        config.prompt_version, registry=registry
    )
    prompts = tuple(
        build_atomic_extraction_prompt(source, include_speaker_name=False)
        for source in sources
    )
    prompt_payload = {
        "system_prompt": system_prompt,
        "cases": [
            {"user_id": user_id, "source_id": source_id, "prompt": prompt}
            for (user_id, source_id), prompt in zip(config.case_order, prompts)
        ],
    }
    if canonical_sha256(prompt_payload) != config.prompt_sha256:
        raise Step35RunError("rendered Step 3.5 prompts changed")
    text_format = atomic_extraction_text_format(registry)
    if canonical_sha256(text_format) != config.schema_sha256:
        raise Step35RunError("Step 3.5 structured-output schema changed")
    _validate_provider_payloads(prompts)
    estimates = {
        model.label: tuple(
            count_step35_request_tokens(
                model=model.requested_model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                text_format=text_format,
                generation_settings=config.generation_settings,
            )
            for prompt in prompts
        )
        for model in config.models
    }
    maximums = {
        model.label: tuple(
            reserve_step35_input_tokens(
                estimate, config.input_token_reserve_per_request
            )
            for estimate in estimates[model.label]
        )
        for model in config.models
    }
    request_hashes = {
        model.label: tuple(
            step35_request_sha256(
                model=model.requested_model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                text_format=text_format,
                generation_settings=config.generation_settings,
            )
            for prompt in prompts
        )
        for model in config.models
    }
    return Step35StagePlan(
        repo_root=root,
        config_path=allowed_path,
        config=config,
        registry=registry,
        all_sources=all_sources,
        sources=sources,
        system_prompt=system_prompt,
        prompts=prompts,
        text_format=text_format,
        estimated_input_tokens=estimates,
        maximum_input_tokens=maximums,
        request_body_sha256=request_hashes,
    )


def build_step35_preflight(
    qualification: Step35StagePlan,
    full: Step35StagePlan,
) -> dict[str, object]:
    """Build the complete no-call budget and data-transmission decision."""

    _require_compatible_plans(qualification, full)
    stage_records = [
        _stage_preflight(qualification),
        _stage_preflight(full),
    ]
    prior = qualification.config.prior_spend_usd
    authorization = qualification.config.cumulative_authorization_usd
    current_expected = sum(
        (
            _planned_model_cost(plan, model, hard=False)
            for plan in (qualification, full)
            for model in plan.config.models
        ),
        Decimal(0),
    )
    current_hard = sum(
        (
            _planned_model_cost(plan, model, hard=True)
            for plan in (qualification, full)
            for model in plan.config.models
        ),
        Decimal(0),
    )
    cumulative_expected = prior + current_expected
    cumulative_hard = prior + current_hard
    new_request_attempts = sum(
        model_record["maximum_request_attempts"]
        for stage in stage_records
        for model_record in stage["models"]
    )
    record: dict[str, object] = {
        "dry_run": True,
        "provider_calls": 0,
        "output_writes": 0,
        "step": "3.5",
        "purpose": "Qualify the extraction model, then produce immutable Phase 4 input from two synthetic development users.",
        "dataset_version": qualification.config.dataset_version,
        "dataset_split": "development",
        "dataset_sha256": qualification.config.dataset_sha256,
        "development_user_ids": ["user_001", "user_002"],
        "transmitted_fields": {
            "source": ["source_type", "source_id", "known_entities", "observations"],
            "known_entity": ["entity_id", "display_name"],
            "observation": [
                "observed_at",
                "message_id",
                "speaker_id",
                "text",
                "title",
            ],
        },
        "excluded_fields": [
            "ingested_at",
            "profile_note",
            "split",
            "task",
            "oracle",
            "gold",
            "lifecycle",
            "review",
        ],
        "synthetic_data_only": True,
        "sensitive_or_restricted_fields_present": False,
        "prompt_version": qualification.config.prompt_version,
        "predicate_registry_version": qualification.config.predicate_registry_version,
        "predicate_registry_sha256": qualification.config.predicate_registry_sha256,
        "schema_version": qualification.config.schema_version,
        "schema_sha256": qualification.config.schema_sha256,
        "token_counter": {
            "method": "Exact tiktoken count of the JSON request body serialized by the client, plus a fixed provider-framing reserve.",
            "version": qualification.config.token_counter_version,
            "package": "tiktoken",
            "package_version": "0.13.0",
            "encoding": qualification.config.token_encoding,
            "input_token_reserve_per_request": qualification.config.input_token_reserve_per_request,
        },
        "stages": stage_records,
        "new_request_attempt_count": new_request_attempts,
        "new_retry_request_count": sum(
            plan.config.maximum_retry_requests
            for plan in (qualification, full)
            for _ in plan.config.models
        ),
        "prior_spend_usd": cost_text(prior),
        "current_run_expected_cost_usd": cost_text(current_expected),
        "current_run_hard_maximum_cost_usd": cost_text(current_hard),
        "cumulative_expected_cost_usd": cost_text(cumulative_expected),
        "cumulative_hard_maximum_cost_usd": cost_text(cumulative_hard),
        "cumulative_cost_cap_usd": cost_text(authorization),
        "spend_remaining_before_run_usd": cost_text(authorization - prior),
        "fits_cost_cap": cumulative_hard <= authorization,
        "price_verification_required_before_execution": False,
        "price_verification": {
            "verified_on": "2026-08-09",
            "source": "official OpenAI documentation",
        },
        "checkpoint_after_each_provider_response": True,
        "resume_policy": "Resume missing cases and approved provider failures without output. Never replay successful or validation-failed cases.",
        "missing_usage_policy": "Charge the request preflight maximum when provider usage metadata is missing.",
        "oracle_or_gold_fields_in_prompts": False,
        "gold_loading_policy": "Load development gold only after every prediction in the relevant stage exists.",
        "recovery": _recovery_preflight(qualification, full),
    }
    assert_no_secrets(record, "Step 3.5 dry run")
    return record


def dry_run_step35(
    repo_root: str | Path = ".",
    qualification_config: str | Path = QUALIFICATION_CONFIG_PATH,
    full_config: str | Path = FULL_CONFIG_PATH,
    stdout: TextIO | None = None,
) -> dict[str, object]:
    """Print the complete Step 3.5 preflight without calls or writes."""

    qualification = prepare_step35_stage(qualification_config, repo_root)
    full = prepare_step35_stage(full_config, repo_root)
    record = build_step35_preflight(qualification, full)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=stdout or sys.stdout)
    return record


def execute_qualification(
    *,
    repo_root: str | Path,
    clients: Mapping[str, object],
    resume: bool = False,
    qualification_config: str | Path = QUALIFICATION_CONFIG_PATH,
    full_config: str | Path = FULL_CONFIG_PATH,
) -> dict[str, object]:
    """Generate all eight source results before opening qualification gold."""

    root = Path(repo_root).resolve()
    family = _run_family(root, qualification_config, full_config)
    qualification = prepare_step35_stage(family.qualification_config_path, root)
    full = prepare_step35_stage(family.full_config_path, root)
    preflight = build_step35_preflight(qualification, full)
    _require_authorized(preflight)
    qualification_root = root / family.qualification_root
    if not resume:
        _require_new_output_dir(qualification_root)
    elif qualification_root.exists():
        _require_allowed_qualification_root(qualification_root, root, family)
        if (qualification_root / "manifest.json").exists():
            _, existing_decision = _verify_qualification_release(
                root, qualification, full
            )
            if existing_decision.get("passed") is not True:
                return existing_decision
    budget = _qualification_budget(qualification, full)
    results: dict[str, dict[str, object]] = {}
    for model in qualification.config.models:
        result = _execute_model(
            qualification,
            model,
            _required_client(clients, model.label),
            root / model.output_directory,
            resume=resume,
            budget=budget,
        )
        results[model.label] = result
        if not result["generation_complete"]:
            break

    if set(results) == {"gpt41", "gpt41-mini"} and all(
        result["generation_complete"] for result in results.values()
    ):
        gold = _load_stage_gold(qualification)
        for model in qualification.config.models:
            results[model.label] = _score_model(
                qualification,
                model,
                results[model.label],
                gold,
                root / model.output_directory,
            )
    decision = _build_qualification_decision(qualification, results, budget)
    _finalize_qualification(
        qualification, full, results, decision, budget, qualification_root
    )
    return decision


def execute_full_phase4_input(
    *,
    repo_root: str | Path,
    client: object,
    resume: bool = False,
    qualification_config: str | Path = QUALIFICATION_CONFIG_PATH,
    full_config: str | Path = FULL_CONFIG_PATH,
) -> dict[str, object]:
    """Run the complete development histories only after mini qualification."""

    root = Path(repo_root).resolve()
    family = _run_family(root, qualification_config, full_config)
    qualification = prepare_step35_stage(family.qualification_config_path, root)
    full = prepare_step35_stage(family.full_config_path, root)
    preflight = build_step35_preflight(qualification, full)
    _require_authorized(preflight)
    qualification_manifest, decision = _verify_qualification_release(
        root, qualification, full
    )
    _verify_qualification_contract(
        qualification_manifest, qualification, full
    )
    decision_path = root / family.qualification_root / "decision.json"
    if decision.get("passed") is not True:
        raise Step35RunError("the mini-model qualification gate did not pass")
    model = full.config.models[0]
    if decision.get("mini_model") != model.resolved_model:
        raise Step35RunError("the final model differs from the qualified mini snapshot")
    qualification_cost = _decimal_field(decision, "actual_cost_usd")
    budget = _full_budget(full, qualification_cost)
    result = _execute_model(
        full,
        model,
        client,
        root / model.output_directory,
        resume=resume,
        budget=budget,
    )
    if result["generation_complete"]:
        gold = _load_stage_gold(full)
        result = _score_model(
            full, model, result, gold, root / model.output_directory
        )
        _write_final_handoff(
            full,
            result,
            root,
            decision_path,
            root / family.qualification_root / "manifest.json",
            qualification_manifest,
            budget,
            qualification,
        )
    return result


def _required_client(clients: Mapping[str, object], label: str) -> object:
    try:
        return clients[label]
    except KeyError:
        raise Step35RunError(f"missing client for {label}") from None


def _execute_model(
    plan: Step35StagePlan,
    model: Step35ModelPlan,
    client: object,
    output_dir: Path,
    *,
    resume: bool,
    budget: Step35Budget | None = None,
) -> dict[str, object]:
    budget = budget or _model_budget(plan, model)
    if resume and output_dir.exists():
        predictions, failures, started_at = _load_checkpoint(
            output_dir, plan, model, allow_completed=True
        )
        _restore_budget(budget, model, predictions, failures)
    else:
        _require_new_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions, failures = [], []
        started_at = _utc_now()
        _checkpoint(
            output_dir,
            plan,
            model,
            predictions,
            failures,
            started_at,
            "running",
            budget,
        )
    successful = {item["source_id"] for item in predictions}
    blocked = [
        item
        for item in failures
        if item["stage"] != "provider" and item.get("resolved") is not True
    ]
    provider_failures = [item for item in failures if item["stage"] == "provider"]
    active_provider_failures = [
        item for item in provider_failures if item.get("resolved") is not True
    ]
    if blocked:
        raise Step35RunError("validation, persistence, or model failures cannot resume")
    if active_provider_failures and len(provider_failures) > plan.config.maximum_retry_requests:
        raise Step35RunError("provider retry allowance is zero or exhausted")
    if len(predictions) == len(plan.sources) and not active_provider_failures:
        return {
            "generation_complete": True,
            "predictions": predictions,
            "failures": failures,
            "failure_counts": _failure_counts(failures),
            "actual_cost_usd": cost_text(
                _attempt_cost(predictions, failures, model)
            ),
        }

    for position, (source, prompt, maximum_input) in enumerate(
        zip(plan.sources, plan.prompts, plan.maximum_input_tokens[model.label]), 1
    ):
        if source.source_id in successful:
            continue
        budget.before_request(model, source.source_id, maximum_input)
        try:
            raw, metadata = getattr(client, "complete_with_metadata")(
                system_prompt=plan.system_prompt,
                user_prompt=prompt,
            )
        except Exception as error:
            stage = "model_mismatch" if isinstance(error, OpenAIModelMismatchError) else "provider"
            metadata_record = _missing_metadata_record(maximum_input, model)
            budget.charge_response(metadata_record)
            failures.append(
                _failure_record(
                    source,
                    position,
                    stage,
                    ("execution",),
                    metadata_record,
                    (
                        "The provider returned a different model snapshot."
                        if stage == "model_mismatch"
                        else "The provider returned no usable model output."
                    ),
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        if not isinstance(metadata, OpenAIResponseMetadata):
            metadata_record = _missing_metadata_record(maximum_input, model)
            budget.charge_response(metadata_record)
            failures.append(
                _failure_record(
                    source, position, "model_mismatch", ("execution",),
                    metadata_record, "The provider returned incompatible model metadata."
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        request_cursor = budget.cursor
        metadata_record = None
        try:
            metadata_record = _metadata_record(
                metadata, maximum_input, model, enforce_limits=False
            )
            budget.charge_response(metadata_record)
        except Step35RunError:
            if isinstance(metadata_record, dict):
                safe_metadata = metadata_record
            else:
                safe_metadata = _missing_metadata_record(maximum_input, model)
            if budget.cursor == request_cursor:
                try:
                    budget.charge_response(safe_metadata)
                except Step35RunError:
                    pass
            failures.append(
                _failure_record(
                    source, position, "budget", ("execution",), safe_metadata,
                    "Provider usage exceeded the frozen request or cumulative budget."
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        if metadata.returned_model != model.resolved_model:
            failures.append(
                _failure_record(
                    source, position, "model_mismatch", ("execution",),
                    metadata_record, "The provider returned incompatible model metadata."
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        try:
            atomic = validate_atomic_response(
                source, raw, metadata, registry=plan.registry
            )
            claims = build_phase4_source_claims(
                source,
                [_atomic_record(claim) for claim in atomic.claims],
                plan.registry,
            )
        except AtomicExtractionValidationError as error:
            failures.append(
                _safe_failure(
                    source,
                    position,
                    "validation",
                    _validation_failure_categories(error),
                    metadata_record,
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        except Phase4InputValidationError as error:
            failures.append(
                _safe_failure(
                    source,
                    position,
                    "persistence",
                    _persistence_failure_categories(error),
                    metadata_record,
                )
            )
            _checkpoint(output_dir, plan, model, predictions, failures, started_at, "failed", budget)
            break
        for failure in active_provider_failures:
            if failure["source_id"] == source.source_id:
                failure["resolved"] = True
        predictions.append(
            {
                "position": position,
                "user_id": source.user_id,
                "source_id": source.source_id,
                "claims": [phase4_claim_record(claim) for claim in claims],
                "provider_metadata": metadata_record,
            }
        )
        _checkpoint(output_dir, plan, model, predictions, failures, started_at, "running", budget)

    if any(item.get("resolved") is not True for item in failures) or len(
        predictions
    ) != len(plan.sources):
        return {
            "generation_complete": False,
            "predictions": predictions,
            "failures": failures,
            "failure_counts": _failure_counts(failures),
            "actual_cost_usd": cost_text(_attempt_cost(predictions, failures, model)),
        }
    _checkpoint(output_dir, plan, model, predictions, failures, started_at, "generated", budget)
    return {
        "generation_complete": True,
        "predictions": predictions,
        "failures": failures,
        "failure_counts": _failure_counts(failures),
        "actual_cost_usd": cost_text(_attempt_cost(predictions, failures, model)),
    }


def _score_model(
    plan: Step35StagePlan,
    model: Step35ModelPlan,
    generated: Mapping[str, object],
    gold: Sequence[object],
    output_dir: Path,
) -> dict[str, object]:
    predictions = generated["predictions"]
    failures = generated["failures"]
    claims = [claim for item in predictions for claim in item["claims"]]
    validated = validate_phase4_claim_file(claims, plan.sources, plan.registry)
    predictions_by_source = {
        item["source_id"]: tuple(
            validate_phase4_claim_file(item["claims"], plan.sources, plan.registry)
        )
        for item in predictions
    }
    scores = score_scaled_development(
        gold,
        predictions_by_source,
        plan.sources,
        plan.registry,
    )
    claim_records = [phase4_claim_record(item) for item in validated]
    _write_or_verify_jsonl(output_dir / "claims.jsonl", claim_records)
    _write_or_verify_jsonl(output_dir / "case_scores.jsonl", scores["case_results"])
    _write_or_verify_json(output_dir / "scores.json", scores)
    run = _read_json(output_dir / "run.json")
    if run.get("status") == "completed":
        _verify_checkpoint_hashes(output_dir, run)
    else:
        _checkpoint(
            output_dir,
            plan,
            model,
            predictions,
            failures,
            run["started_at"],
            "completed",
            None,
        )
    return {
        "generation_complete": True,
        "scores": scores,
        "failure_counts": _failure_counts(failures),
        "actual_cost_usd": cost_text(_attempt_cost(predictions, failures, model)),
        "claims": validated,
        "predictions": predictions,
        "failures": failures,
    }


def _load_stage_gold(plan: Step35StagePlan) -> tuple[object, ...]:
    gold_path = plan.repo_root / _GOLD_CLAIM_PATH
    if _file_sha256(gold_path) != plan.config.gold_claim_file_sha256:
        raise Step35RunError("scaled development gold changed before scoring")
    gold = load_scaled_development_gold(
        plan.repo_root,
        plan.all_sources,
        plan.registry,
        selected_source_ids={source.source_id for source in plan.sources},
    )
    if _file_sha256(gold_path) != plan.config.gold_claim_file_sha256:
        raise Step35RunError("scaled development gold changed during scoring")
    return gold


def _reservations(
    plan: Step35StagePlan, models: Sequence[Step35ModelPlan]
) -> tuple[Step35RequestReservation, ...]:
    return tuple(
        Step35RequestReservation(model, source.source_id, maximum_input)
        for model in models
        for source, maximum_input in zip(
            plan.sources, plan.maximum_input_tokens[model.label]
        )
    )


def _qualification_budget(
    qualification: Step35StagePlan, full: Step35StagePlan
) -> Step35Budget:
    return Step35Budget(
        prior_spend_usd=qualification.config.prior_spend_usd,
        authorization_usd=qualification.config.cumulative_authorization_usd,
        reservations=(
            *_reservations(qualification, qualification.config.models),
            *_reservations(full, full.config.models),
        ),
    )


def _full_budget(plan: Step35StagePlan, qualification_cost: Decimal) -> Step35Budget:
    return Step35Budget(
        prior_spend_usd=plan.config.prior_spend_usd + qualification_cost,
        authorization_usd=plan.config.cumulative_authorization_usd,
        reservations=_reservations(plan, plan.config.models),
    )


def _model_budget(plan: Step35StagePlan, model: Step35ModelPlan) -> Step35Budget:
    return Step35Budget(
        prior_spend_usd=plan.config.prior_spend_usd,
        authorization_usd=plan.config.cumulative_authorization_usd,
        reservations=_reservations(plan, (model,)),
    )


def _decimal_field(record: Mapping[str, object], name: str) -> Decimal:
    try:
        value = Decimal(record[name])
    except (InvalidOperation, KeyError, TypeError):
        raise Step35RunError(f"{name} must be decimal text") from None
    if value < 0:
        raise Step35RunError(f"{name} cannot be negative")
    return value


def _build_qualification_decision(
    plan: Step35StagePlan,
    results: Mapping[str, Mapping[str, object]],
    budget: Step35Budget,
) -> dict[str, object]:
    full = results.get("gpt41")
    mini = results.get("gpt41-mini")
    scored = bool(full and mini and "scores" in full and "scores" in mini)
    if scored:
        gate = qualification_gate(
            full["scores"],
            mini["scores"],
            full_failures=full["failure_counts"],
            mini_failures=mini["failure_counts"],
        )
    else:
        reasons = ["all eight qualification predictions must exist before scoring"]
        for label, result in results.items():
            for category, count in result.get("failure_counts", {}).items():
                if count:
                    reasons.append(f"{label} {category} failures must be zero")
        gate = {
            "passed": False,
            "reasons": reasons,
            "full_exact_true_positives": None,
            "mini_exact_true_positives": None,
            "full_matched_provenance_spans": None,
            "mini_matched_provenance_spans": None,
        }
    attempts = [
        item
        for result in results.values()
        for item in (*result.get("predictions", ()), *result.get("failures", ()))
    ]
    return {
        "decision_version": (
            f"step35-model-qualification-{_family_for_plan(plan).release_version}"
        ),
        **gate,
        "all_predictions_complete": scored,
        "full_model": plan.config.models[0].resolved_model,
        "mini_model": plan.config.models[1].resolved_model,
        "provider_request_count": len(attempts),
        "actual_cost_usd": cost_text(
            sum(
                (
                    _attempt_cost(
                        result.get("predictions", ()),
                        result.get("failures", ()),
                        next(model for model in plan.config.models if model.label == label),
                    )
                    for label, result in results.items()
                ),
                Decimal(0),
            )
        ),
        "budget": budget.record(),
    }


def _qualification_comparison(
    plan: Step35StagePlan,
    results: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
) -> dict[str, object]:
    models: dict[str, object] = {}
    for label in ("gpt41", "gpt41-mini"):
        result = results.get(label)
        if result is None:
            models[label] = {"generation_complete": False, "not_started": True}
            continue
        models[label] = {
            "generation_complete": result.get("generation_complete") is True,
            "failure_counts": result.get("failure_counts", {}),
            "scores": result.get("scores"),
            "actual_cost_usd": result.get("actual_cost_usd"),
        }
    return {
        "comparison_version": (
            "step35-model-qualification-comparison-"
            f"{_family_for_plan(plan).release_version}"
        ),
        "passed": decision["passed"],
        "reasons": decision["reasons"],
        "gate": {
            name: decision[name]
            for name in (
                "full_exact_true_positives",
                "mini_exact_true_positives",
                "full_matched_provenance_spans",
                "mini_matched_provenance_spans",
            )
        },
        "models": models,
    }


def _qualification_findings(decision: Mapping[str, object]) -> str:
    if decision["passed"]:
        outcome = "Qualification passed. The mini snapshot is cleared for the 20-source development run."
    else:
        reasons = "; ".join(str(item) for item in decision["reasons"])
        outcome = f"Qualification stopped: {reasons}."
    return (
        "# Qualification findings\n\n"
        f"{outcome}\n\n"
        f"Full/mini exact true positives: {decision['full_exact_true_positives']} / "
        f"{decision['mini_exact_true_positives']}. Full/mini matched provenance spans: "
        f"{decision['full_matched_provenance_spans']} / "
        f"{decision['mini_matched_provenance_spans']}.\n"
    )


def _recovery_preflight(
    qualification: Step35StagePlan, full: Step35StagePlan
) -> dict[str, object]:
    family = _family_for_plans(qualification, full)
    if family.release_version == "v1":
        return {"enabled": False, "release_version": "v1"}
    return {
        "enabled": True,
        "release_version": "v2",
        "recovery_guidance_version": _RECOVERY_GUIDANCE_VERSION,
        "decision_envelope_sha256": _RECOVERY_ENVELOPE_SHA256,
        "predecessor_qualification_root": QUALIFICATION_ROOT_V1.as_posix(),
        "predecessor_charged_cost_usd": cost_text(_RECOVERY_PRIOR_SPEND_USD),
        "eligibility": "verified_no_provider_output",
        "prior_spend_counted_once": True,
        "qualification_root": family.qualification_root.as_posix(),
        "full_output_root": family.full_output_root.as_posix(),
    }


def _recovery_manifest(
    qualification: Step35StagePlan,
    full: Step35StagePlan,
    budget: Step35Budget,
) -> dict[str, object]:
    record = _recovery_preflight(qualification, full)
    record.update(
        {
            "predecessor": qualification.config.predecessor,
            "budget_prior_spend_usd": cost_text(budget.prior_spend_usd),
            "current_stage_actual_spend_usd": cost_text(budget.actual_spend_usd),
            "cumulative_actual_spend_usd": cost_text(
                budget.prior_spend_usd + budget.actual_spend_usd
            ),
        }
    )
    return record


def _final_run_preflight(
    full: Step35StagePlan, qualification_cost: Decimal
) -> dict[str, object]:
    full_stage = _stage_preflight(full)
    hard = sum(
        (
            _planned_model_cost(full, model, hard=True)
            for model in full.config.models
        ),
        Decimal(0),
    )
    projected = full.config.prior_spend_usd + qualification_cost + hard
    record = {
        "preflight_version": (
            f"step35-final-run-preflight-{_family_for_plan(full).release_version}"
        ),
        "provider_calls": 0,
        "qualification_actual_cost_usd": cost_text(qualification_cost),
        "prior_spend_usd": cost_text(full.config.prior_spend_usd),
        "full_hard_maximum_cost_usd": cost_text(hard),
        "projected_cumulative_hard_maximum_cost_usd": cost_text(projected),
        "spend_remaining_before_full_usd": cost_text(
            full.config.cumulative_authorization_usd
            - full.config.prior_spend_usd
            - qualification_cost
        ),
        "cumulative_cost_cap_usd": cost_text(
            full.config.cumulative_authorization_usd
        ),
        "fits_cost_cap": projected <= full.config.cumulative_authorization_usd,
        "stage": full_stage,
        "oracle_or_gold_fields_in_prompts": False,
    }
    assert_no_secrets(record, "Step 3.5 final preflight")
    if record["fits_cost_cap"] is not True:
        raise Step35RunError("the full run no longer fits the cumulative cap")
    return record


def _finalize_qualification(
    plan: Step35StagePlan,
    full: Step35StagePlan,
    results: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
    budget: Step35Budget,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(output / "decision.json", decision)
    comparison = _qualification_comparison(plan, results, decision)
    _write_or_verify_json(output / "comparison.json", comparison)
    _write_or_verify_text(output / "findings.md", _qualification_findings(decision))
    top_files = ["decision.json", "comparison.json", "findings.md"]
    if decision["passed"] is True:
        preflight = _final_run_preflight(
            full, _decimal_field(decision, "actual_cost_usd")
        )
        _write_or_verify_json(output / "final_preflight.json", preflight)
        top_files.append("final_preflight.json")
    elif (output / "final_preflight.json").exists():
        raise Step35RunError("a failed qualification cannot retain a final preflight")

    files = {name: _file_sha256(output / name) for name in top_files}
    model_records: dict[str, object] = {}
    for model in plan.config.models:
        model_output = plan.repo_root / model.output_directory
        if not model_output.exists():
            model_records[model.label] = {"not_started": True}
            continue
        run = _read_json(model_output / "run.json")
        _verify_checkpoint_hashes(model_output, run)
        names = sorted(item.name for item in model_output.iterdir() if item.is_file())
        for name in names:
            files[f"{model.label}/{name}"] = _file_sha256(model_output / name)
        predictions = _read_jsonl(model_output / "predictions.jsonl")
        failures = _read_jsonl(model_output / "failures.jsonl")
        model_records[model.label] = _model_manifest_record(model, predictions, failures)

    family = _family_for_plans(plan, full)
    manifest = {
        "manifest_version": (
            f"step35-model-qualification-manifest-{family.release_version}"
        ),
        "configurations": _configuration_manifest(plan, full),
        "runtime_inputs": _runtime_input_manifest(plan),
        "case_order": _case_order_manifest(plan),
        "rendered_prompts": _prompt_manifest(plan),
        "schema": _schema_manifest(plan),
        "predicate_registry": _registry_manifest(plan),
        "token_counter": _token_counter_manifest(plan),
        "repository": _repository_manifest(),
        "models": model_records,
        "budget": budget.record(),
        "recovery": _recovery_manifest(plan, full, budget),
        "gold_access": {
            "opened_only_after_all_eight_predictions": decision["all_predictions_complete"],
            "expected_sha256": plan.config.gold_claim_file_sha256,
        },
        "files": files,
    }
    _write_or_verify_json(output / "manifest.json", manifest)
    _verify_manifest_files(output, manifest)


def _verify_qualification_release(
    root: Path,
    qualification: Step35StagePlan,
    full: Step35StagePlan,
) -> tuple[dict[str, object], dict[str, object]]:
    family = _family_for_plans(qualification, full)
    output = root / family.qualification_root
    _require_allowed_qualification_root(output, root, family)
    manifest = _read_json(output / "manifest.json")
    if manifest.get("manifest_version") != (
        f"step35-model-qualification-manifest-{family.release_version}"
    ):
        raise Step35RunError("qualification manifest version is invalid")
    _verify_manifest_files(output, manifest)
    decision = _read_json(output / "decision.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get("decision.json") != _file_sha256(
        output / "decision.json"
    ):
        raise Step35RunError("qualification decision is not bound by its manifest")
    if decision.get("decision_version") != (
        f"step35-model-qualification-{family.release_version}"
    ):
        raise Step35RunError("qualification decision version is invalid")
    if decision.get("passed") is True:
        expected = {
            "decision.json",
            "comparison.json",
            "findings.md",
            "final_preflight.json",
            *{
                f"{label}/{name}"
                for label in ("gpt41", "gpt41-mini")
                for name in (
                    "predictions.jsonl",
                    "failures.jsonl",
                    "run.json",
                    "claims.jsonl",
                    "case_scores.jsonl",
                    "scores.json",
                )
            },
        }
        if set(files) != expected:
            raise Step35RunError("passing qualification artifacts are incomplete")
    return manifest, decision


def _verify_qualification_contract(
    manifest: Mapping[str, object],
    qualification: Step35StagePlan,
    full: Step35StagePlan,
) -> None:
    expected = {
        "configurations": _configuration_manifest(qualification, full),
        "runtime_inputs": _runtime_input_manifest(qualification),
        "case_order": _case_order_manifest(qualification),
        "rendered_prompts": _prompt_manifest(qualification),
        "schema": _schema_manifest(qualification),
        "predicate_registry": _registry_manifest(qualification),
        "token_counter": _token_counter_manifest(qualification),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise Step35RunError(f"qualification manifest changed on {name}")
    recovery = manifest.get("recovery")
    expected_recovery = _recovery_preflight(qualification, full)
    if not isinstance(recovery, dict) or any(
        recovery.get(name) != value for name, value in expected_recovery.items()
    ):
        raise Step35RunError("qualification manifest changed on recovery")
    if recovery.get("predecessor") != qualification.config.predecessor:
        raise Step35RunError("qualification predecessor binding changed")


def _write_final_handoff(
    plan: Step35StagePlan,
    result: Mapping[str, object],
    root: Path,
    decision_path: Path,
    qualification_manifest_path: Path,
    qualification_manifest: Mapping[str, object],
    budget: Step35Budget,
    qualification: Step35StagePlan,
) -> None:
    output = root / plan.config.models[0].output_directory
    limitations = {
        "items": [
            "This run covers two synthetic development users. Frozen test users were not processed.",
            "Extraction errors remain visible in the scores and case results.",
            "Phase 4 storage and temporal resolution are outside this step.",
        ]
    }
    _write_or_verify_json(output / "known_limitations.json", limitations)
    evidence_index = _build_evidence_index(plan, result["claims"])
    _write_or_verify_json(output / "evidence_index.json", evidence_index)
    files = {}
    for name in (
        "claims.jsonl",
        "predictions.jsonl",
        "failures.jsonl",
        "case_scores.jsonl",
        "scores.json",
        "run.json",
        "known_limitations.json",
        "evidence_index.json",
    ):
        files[name] = _file_sha256(output / name)
    model = plan.config.models[0]
    predictions = result["predictions"]
    failures = result["failures"]
    family = _family_for_plans(qualification, plan)
    manifest = {
        "manifest_version": f"phase4-input-manifest-{family.release_version}",
        "configurations": _configuration_manifest(qualification, plan),
        "runtime_inputs": _runtime_input_manifest(plan),
        "case_order": _case_order_manifest(plan),
        "rendered_prompts": _prompt_manifest(plan),
        "schema": _schema_manifest(plan),
        "predicate_registry": _registry_manifest(plan),
        "token_counter": _token_counter_manifest(plan),
        "repository": _repository_manifest(),
        "model": _model_manifest_record(model, predictions, failures),
        "budget": budget.record(),
        "recovery": _recovery_manifest(qualification, plan, budget),
        "qualification_decision_sha256": _file_sha256(decision_path),
        "qualification_manifest_sha256": _file_sha256(qualification_manifest_path),
        "qualification_manifest_version": qualification_manifest.get("manifest_version"),
        "reverse_source_to_claim_evidence_index": evidence_index["validation"],
        "files": files,
    }
    _write_or_verify_json(output / "manifest.json", manifest)
    _verify_manifest_files(output, manifest)


def _configuration_manifest(
    qualification: Step35StagePlan, full: Step35StagePlan
) -> dict[str, object]:
    return {
        "qualification": {
            "path": qualification.config_path.as_posix(),
            "file_sha256": _file_sha256(
                qualification.repo_root / qualification.config_path
            ),
            "canonical_sha256": qualification.config.configuration_sha256,
        },
        "full": {
            "path": full.config_path.as_posix(),
            "file_sha256": _file_sha256(full.repo_root / full.config_path),
            "canonical_sha256": full.config.configuration_sha256,
        },
    }


def _runtime_input_manifest(plan: Step35StagePlan) -> dict[str, object]:
    return {
        "dataset_version": plan.config.dataset_version,
        "dataset_sha256": plan.config.dataset_sha256,
        "users": {
            "path": _RUNTIME_USER_PATH.as_posix(),
            "sha256": plan.config.runtime_user_file_sha256,
        },
        "sources": {
            "path": _RUNTIME_SOURCE_PATH.as_posix(),
            "sha256": plan.config.runtime_source_file_sha256,
        },
    }


def _case_order_manifest(plan: Step35StagePlan) -> list[dict[str, object]]:
    return [
        {"position": position, "user_id": user_id, "source_id": source_id}
        for position, (user_id, source_id) in enumerate(plan.config.case_order, 1)
    ]


def _prompt_manifest(plan: Step35StagePlan) -> dict[str, object]:
    return {
        "version": plan.config.prompt_version,
        "aggregate_sha256": plan.config.prompt_sha256,
        "system_prompt_sha256": hashlib.sha256(
            plan.system_prompt.encode("utf-8")
        ).hexdigest(),
        "cases": [
            {
                "position": position,
                "user_id": user_id,
                "source_id": source_id,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            for position, ((user_id, source_id), prompt) in enumerate(
                zip(plan.config.case_order, plan.prompts), 1
            )
        ],
    }


def _schema_manifest(plan: Step35StagePlan) -> dict[str, object]:
    return {"version": plan.config.schema_version, "sha256": plan.config.schema_sha256}


def _registry_manifest(plan: Step35StagePlan) -> dict[str, object]:
    return {
        "path": plan.config.predicate_registry_path,
        "version": plan.config.predicate_registry_version,
        "sha256": plan.config.predicate_registry_sha256,
    }


def _token_counter_manifest(plan: Step35StagePlan) -> dict[str, object]:
    return {
        "method": "client-serialized JSON request tokens plus fixed framing reserve",
        "version": plan.config.token_counter_version,
        "package": "tiktoken",
        "package_version": "0.13.0",
        "encoding": plan.config.token_encoding,
        "reserve_per_request": plan.config.input_token_reserve_per_request,
        "models": {
            model.label: _model_request_plan(plan, model)
            for model in plan.config.models
        },
    }


def _model_request_plan(
    plan: Step35StagePlan, model: Step35ModelPlan
) -> list[dict[str, object]]:
    return [
        {
            "position": position,
            "user_id": user_id,
            "source_id": source_id,
            "request_body_sha256": request_hash,
            "exact_input_tokens": exact_tokens,
            "maximum_input_tokens": maximum_tokens,
        }
        for position, (
            (user_id, source_id),
            request_hash,
            exact_tokens,
            maximum_tokens,
        ) in enumerate(
            zip(
                plan.config.case_order,
                plan.request_body_sha256[model.label],
                plan.estimated_input_tokens[model.label],
                plan.maximum_input_tokens[model.label],
            ),
            1,
        )
    ]


def _repository_manifest() -> dict[str, object]:
    status = _git_status_text()
    return {
        "commit": _git_commit(),
        "worktree_dirty": bool(status.strip()),
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _model_manifest_record(
    model: Step35ModelPlan,
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    attempts = sorted([*predictions, *failures], key=lambda item: item["position"])
    returned = sorted(
        {
            item["provider_metadata"]["returned_model"]
            for item in attempts
            if item["provider_metadata"]["returned_model"] is not None
        }
    )
    return {
        "label": model.label,
        "requested_model": model.requested_model,
        "resolved_model": model.resolved_model,
        "returned_models": returned,
        "request_count": len(attempts),
        "input_tokens_charged": sum(
            item["provider_metadata"]["charged_input_tokens"] for item in attempts
        ),
        "output_tokens_charged": sum(
            item["provider_metadata"]["charged_output_tokens"] for item in attempts
        ),
        "actual_cost_usd": cost_text(_attempt_cost(predictions, failures, model)),
        "requests": [
            {
                "position": item["position"],
                "source_id": item["source_id"],
                "outcome": "prediction" if item in predictions else item["stage"],
                "provider_metadata": item["provider_metadata"],
            }
            for item in attempts
        ],
    }


def _build_evidence_index(
    plan: Step35StagePlan, claims: Sequence[Phase4InputClaim]
) -> dict[str, object]:
    source_map = {source.source_id: source for source in plan.sources}
    entries: dict[str, list[dict[str, object]]] = {
        source.source_id: [] for source in plan.sources
    }
    claim_ids: set[str] = set()
    evidence_count = 0
    for claim in claims:
        if claim.claim_id in claim_ids:
            raise Step35RunError("the evidence index contains duplicate claims")
        claim_ids.add(claim.claim_id)
        for position, evidence in enumerate(claim.evidence, 1):
            source = source_map.get(evidence.source_id)
            if source is None or source.user_id != claim.user_id:
                raise Step35RunError("the evidence index crosses a source boundary")
            entries[evidence.source_id].append(
                {
                    "claim_id": claim.claim_id,
                    "evidence_position": position,
                    "message_id": evidence.message_id,
                    "quote_sha256": hashlib.sha256(
                        evidence.quote.encode("utf-8")
                    ).hexdigest(),
                }
            )
            evidence_count += 1
    indexed_claims = {
        item["claim_id"] for values in entries.values() for item in values
    }
    if indexed_claims != claim_ids:
        raise Step35RunError("the reverse evidence index is incomplete")
    return {
        "index_version": "phase4-source-claim-evidence-index-v1",
        "sources": [
            {
                "position": position,
                "user_id": source.user_id,
                "source_id": source.source_id,
                "claim_ids": sorted({item["claim_id"] for item in entries[source.source_id]}),
                "evidence": entries[source.source_id],
            }
            for position, source in enumerate(plan.sources, 1)
        ],
        "validation": {
            "valid": True,
            "source_count": len(plan.sources),
            "claim_count": len(claim_ids),
            "indexed_claim_count": len(indexed_claims),
            "evidence_count": evidence_count,
        },
    }


def _planned_model_cost(
    plan: Step35StagePlan, model: Step35ModelPlan, *, hard: bool
) -> Decimal:
    if hard:
        retry_input = (
            max(plan.maximum_input_tokens[model.label])
            * plan.config.maximum_retry_requests
        )
        input_tokens = sum(plan.maximum_input_tokens[model.label]) + retry_input
        output_tokens = (
            len(plan.sources) + plan.config.maximum_retry_requests
        ) * _MAX_OUTPUT_TOKENS
    else:
        input_tokens = sum(plan.estimated_input_tokens[model.label])
        output_tokens = (
            len(plan.sources) * plan.config.expected_output_tokens_per_request
        )
    return step35_cost(model, input_tokens, output_tokens)


def _stage_preflight(plan: Step35StagePlan) -> dict[str, object]:
    models = []
    for model in plan.config.models:
        model_estimates = plan.estimated_input_tokens[model.label]
        model_maximums = plan.maximum_input_tokens[model.label]
        retry_input = max(model_maximums) * plan.config.maximum_retry_requests
        planned_attempts = len(plan.sources)
        maximum_attempts = planned_attempts + plan.config.maximum_retry_requests
        estimated_input = sum(model_estimates)
        maximum_input = sum(model_maximums) + retry_input
        expected_output = (
            planned_attempts * plan.config.expected_output_tokens_per_request
        )
        maximum_output = maximum_attempts * _MAX_OUTPUT_TOKENS
        models.append(
            {
                "label": model.label,
                "requested_model": model.requested_model,
                "resolved_model": model.resolved_model,
                "input_usd_per_million_tokens": str(model.input_usd_per_million_tokens),
                "output_usd_per_million_tokens": str(model.output_usd_per_million_tokens),
                "planned_request_count": planned_attempts,
                "maximum_retry_requests": plan.config.maximum_retry_requests,
                "maximum_request_attempts": maximum_attempts,
                "estimated_input_tokens": estimated_input,
                "maximum_input_tokens": maximum_input,
                "expected_output_tokens": expected_output,
                "maximum_output_tokens": maximum_output,
                "expected_cost_usd": cost_text(
                    step35_cost(model, estimated_input, expected_output)
                ),
                "hard_maximum_cost_usd": cost_text(
                    step35_cost(model, maximum_input, maximum_output)
                ),
                "requests": _model_request_plan(plan, model),
                "output_directory": model.output_directory,
            }
        )
    return {
        "stage": plan.config.stage,
        "configuration_version": plan.config.configuration_version,
        "configuration_sha256": plan.config.configuration_sha256,
        "prompt_sha256": plan.config.prompt_sha256,
        "token_counter_version": plan.config.token_counter_version,
        "token_encoding": plan.config.token_encoding,
        "input_token_reserve_per_request": plan.config.input_token_reserve_per_request,
        "cases": [
            {"position": index, "user_id": user_id, "source_id": source_id}
            for index, (user_id, source_id) in enumerate(plan.config.case_order, 1)
        ],
        "generation_settings": dict(plan.config.generation_settings),
        "models": models,
    }


def _allowed_config_path(root: Path, config_path: str | Path) -> Path:
    candidate = Path(config_path)
    allowed = {
        family.qualification_config_path
        for family in _RUN_FAMILIES
    } | {family.full_config_path for family in _RUN_FAMILIES} | {FALLBACK_CONFIG_PATH}
    if candidate.is_absolute():
        for relative in allowed:
            if candidate == root / relative:
                return relative
    elif candidate in allowed:
        return candidate
    raise Step35RunError("Step 3.5 config path is not allowlisted")


def _run_family(
    root: Path,
    qualification_config: str | Path,
    full_config: str | Path,
) -> Step35RunFamily:
    qualification_path = _allowed_config_path(root, qualification_config)
    full_path = _allowed_config_path(root, full_config)
    for family in _RUN_FAMILIES:
        if (
            qualification_path == family.qualification_config_path
            and full_path == family.full_config_path
        ):
            return family
    raise Step35RunError("Step 3.5 v1 and v2 config families cannot be mixed")


def _family_for_plans(
    qualification: Step35StagePlan,
    full: Step35StagePlan,
) -> Step35RunFamily:
    if qualification.repo_root != full.repo_root:
        raise Step35RunError("qualification and full plans use different repositories")
    family = _run_family(
        qualification.repo_root,
        qualification.config_path,
        full.config_path,
    )
    expected_versions = {
        "v1": (
            "phase4-input-model-qualification-v1",
            "phase4-input-development-v1",
        ),
        "v2": (
            "phase4-input-model-qualification-v2",
            "phase4-input-development-v2",
        ),
    }[family.release_version]
    if (
        qualification.config.configuration_version,
        full.config.configuration_version,
    ) != expected_versions:
        raise Step35RunError("Step 3.5 configuration versions changed")
    return family


def _family_for_plan(plan: Step35StagePlan) -> Step35RunFamily:
    for family in _RUN_FAMILIES:
        if plan.config_path in {
            family.qualification_config_path,
            family.full_config_path,
        }:
            return family
    raise Step35RunError("Step 3.5 plan is not in the v1/v2 allowlist")


def _expected_recovery_predecessor() -> dict[str, object]:
    return {
        "recovery_guidance_version": _RECOVERY_GUIDANCE_VERSION,
        "decision_envelope_sha256": _RECOVERY_ENVELOPE_SHA256,
        "qualification_config_path": QUALIFICATION_CONFIG_V1_PATH.as_posix(),
        "full_config_path": FULL_CONFIG_V1_PATH.as_posix(),
        "qualification_root": QUALIFICATION_ROOT_V1.as_posix(),
        "configuration_file_sha256": dict(_RECOVERY_CONFIG_SHA256),
        "artifact_sha256": dict(_RECOVERY_ARTIFACT_SHA256),
        "failure": dict(_RECOVERY_FAILURE),
    }


def _verify_recovery_predecessor(
    qualification: Step35StagePlan,
    full: Step35StagePlan,
    family: Step35RunFamily,
) -> None:
    if family.release_version == "v1":
        if (
            qualification.config.predecessor is not None
            or full.config.predecessor is not None
            or qualification.config.prior_spend_usd != 0
            or full.config.prior_spend_usd != 0
        ):
            raise Step35RunError("Step 3.5 v1 recovery metadata changed")
        return

    expected = _expected_recovery_predecessor()
    if (
        qualification.config.predecessor != expected
        or full.config.predecessor != expected
    ):
        raise Step35RunError("Step 3.5 recovery predecessor metadata changed")
    if (
        qualification.config.prior_spend_usd != _RECOVERY_PRIOR_SPEND_USD
        or full.config.prior_spend_usd != _RECOVERY_PRIOR_SPEND_USD
        or Decimal(_RECOVERY_FAILURE["charged_cost_usd"])
        != _RECOVERY_PRIOR_SPEND_USD
    ):
        raise Step35RunError("Step 3.5 predecessor spend is not counted exactly once")

    root = qualification.repo_root
    for name, digest in _RECOVERY_CONFIG_SHA256.items():
        if _file_sha256(root / name) != digest:
            raise Step35RunError(f"recovery predecessor config changed: {name}")
    predecessor_root = root / QUALIFICATION_ROOT_V1
    if not predecessor_root.is_dir():
        raise Step35RunError("recovery predecessor qualification is missing")
    actual_files = {
        path.relative_to(predecessor_root).as_posix()
        for path in predecessor_root.rglob("*")
        if path.is_file()
    }
    if actual_files != set(_RECOVERY_ARTIFACT_SHA256):
        raise Step35RunError("recovery predecessor artifact set changed")
    for name, digest in _RECOVERY_ARTIFACT_SHA256.items():
        if _file_sha256(predecessor_root / name) != digest:
            raise Step35RunError(f"recovery predecessor artifact changed: {name}")
    if (root / FULL_OUTPUT_ROOT_V1).exists():
        raise Step35RunError("recovery is allowed only when the v1 full run has no output")

    decision = _read_json(predecessor_root / "decision.json")
    manifest = _read_json(predecessor_root / "manifest.json")
    run = _read_json(predecessor_root / "gpt41/run.json")
    predictions = _read_jsonl(predecessor_root / "gpt41/predictions.jsonl")
    failures = _read_jsonl(predecessor_root / "gpt41/failures.jsonl")
    if (
        decision.get("passed") is not False
        or decision.get("all_predictions_complete") is not False
        or decision.get("provider_request_count") != 1
        or decision.get("actual_cost_usd") != "0.019646"
        or predictions
        or len(failures) != 1
    ):
        raise Step35RunError("recovery predecessor is not the approved failed run")
    failure = failures[0]
    metadata = failure.get("provider_metadata")
    if (
        failure.get("failure_id") != _RECOVERY_FAILURE["failure_id"]
        or failure.get("source_id") != _RECOVERY_FAILURE["source_id"]
        or failure.get("stage") != _RECOVERY_FAILURE["stage"]
        or failure.get("failure_categories") != _RECOVERY_FAILURE["failure_categories"]
        or not isinstance(metadata, dict)
        or any(
            metadata.get(name) is not None
            for name in (
                "response_id",
                "returned_model",
                "request_id",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        )
        or metadata.get("charged_input_tokens")
        != _RECOVERY_FAILURE["charged_input_token_ceiling"]
        or metadata.get("charged_output_tokens")
        != _RECOVERY_FAILURE["charged_output_token_ceiling"]
        or metadata.get("charged_cost_usd")
        != _RECOVERY_FAILURE["charged_cost_usd"]
    ):
        raise Step35RunError("recovery predecessor is not a no-output provider failure")
    if (
        run.get("status") != "failed"
        or run.get("model_label") != _RECOVERY_FAILURE["model_label"]
        or run.get("successful_source_ids") != []
        or run.get("attempted_source_ids") != [_RECOVERY_FAILURE["source_id"]]
        or run.get("input_tokens_charged")
        != _RECOVERY_FAILURE["charged_input_token_ceiling"]
        or run.get("output_tokens_charged")
        != _RECOVERY_FAILURE["charged_output_token_ceiling"]
        or run.get("cost_usd") != _RECOVERY_FAILURE["charged_cost_usd"]
        or manifest.get("manifest_version")
        != "step35-model-qualification-manifest-v1"
        or manifest.get("gold_access", {}).get(
            "opened_only_after_all_eight_predictions"
        )
        is not False
    ):
        raise Step35RunError("recovery predecessor accounting changed")


def _require_compatible_plans(
    qualification: Step35StagePlan,
    full: Step35StagePlan,
) -> None:
    family = _family_for_plans(qualification, full)
    for name in (
        "dataset_version",
        "dataset_sha256",
        "predicate_registry_version",
        "predicate_registry_sha256",
        "prompt_version",
        "schema_version",
        "schema_sha256",
        "token_counter_version",
        "token_encoding",
        "input_token_reserve_per_request",
        "generation_settings",
        "prior_spend_usd",
        "cumulative_authorization_usd",
    ):
        if getattr(qualification.config, name) != getattr(full.config, name):
            raise Step35RunError(f"qualification and full plans differ on {name}")
    labels = [model.label for model in qualification.config.models]
    if labels != ["gpt41", "gpt41-mini"] or len(full.config.models) != 1:
        raise Step35RunError("Step 3.5 model plan changed")
    actual_models = [
        (
            model.label,
            model.requested_model,
            model.resolved_model,
            model.output_directory,
        )
        for model in (*qualification.config.models, *full.config.models)
    ]
    expected_models = [
        (
            "gpt41",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-2025-04-14",
            (family.qualification_root / "gpt41").as_posix(),
        ),
        (
            "gpt41-mini",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            (family.qualification_root / "gpt41-mini").as_posix(),
        ),
        (
            "gpt41-mini",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            family.full_output_root.as_posix(),
        ),
    ]
    if actual_models != expected_models:
        raise Step35RunError("Step 3.5 model snapshots or output directories changed")
    mini = qualification.config.models[1]
    final = full.config.models[0]
    if mini.requested_model != final.requested_model or mini.resolved_model != final.resolved_model:
        raise Step35RunError("full run must use the exact qualified mini snapshot")
    _verify_recovery_predecessor(qualification, full, family)


def _validate_provider_payloads(prompts: Sequence[str]) -> None:
    allowed_source = {"source_type", "source_id", "known_entities", "observations"}
    allowed_entity = {"entity_id", "display_name"}
    allowed_observation = {
        "observed_at",
        "message_id",
        "speaker_id",
        "text",
        "title",
    }
    for prompt in prompts:
        try:
            payload = json.loads(prompt.split("\n", 1)[1])
        except (IndexError, json.JSONDecodeError):
            raise Step35RunError("provider payload is not canonical JSON") from None
        if set(payload) != allowed_source:
            raise Step35RunError("provider source payload fields changed")
        if any(set(item) != allowed_entity for item in payload["known_entities"]):
            raise Step35RunError("provider known-entity fields changed")
        if any(not set(item) <= allowed_observation for item in payload["observations"]):
            raise Step35RunError("provider observation fields changed")


def _require_authorized(preflight: Mapping[str, object]) -> None:
    if preflight.get("fits_cost_cap") is not True:
        raise Step35RunError("the cumulative hard maximum exceeds the authorization")


def _checkpoint(
    output: Path,
    plan: Step35StagePlan,
    model: Step35ModelPlan,
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    started_at: str,
    status: str,
    budget: Step35Budget | Mapping[str, object] | None,
) -> None:
    if status not in {"running", "failed", "generated", "completed"}:
        raise Step35RunError("checkpoint status is invalid")
    _write_jsonl(output / "predictions.jsonl", predictions)
    _write_jsonl(output / "failures.jsonl", failures)
    attempts = [*predictions, *failures]
    artifact_hashes = {
        "predictions.jsonl": _file_sha256(output / "predictions.jsonl"),
        "failures.jsonl": _file_sha256(output / "failures.jsonl"),
    }
    if status == "completed":
        for name in ("claims.jsonl", "case_scores.jsonl", "scores.json"):
            if not (output / name).is_file():
                raise Step35RunError("completed checkpoint is missing score artifacts")
            artifact_hashes[name] = _file_sha256(output / name)
    if isinstance(budget, Step35Budget):
        budget_record: Mapping[str, object] = budget.record()
    elif isinstance(budget, Mapping):
        budget_record = dict(budget)
    elif (output / "run.json").exists():
        previous = _read_json(output / "run.json")
        existing = previous.get("budget")
        if not isinstance(existing, dict):
            raise Step35RunError("checkpoint budget record is missing")
        budget_record = existing
    else:
        raise Step35RunError("checkpoint budget record is missing")
    run = {
        "run_format_version": "phase4-input-run-v1",
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "repository_commit": _git_commit(),
        "repository_dirty": _git_dirty(),
        "configuration_version": plan.config.configuration_version,
        "configuration_sha256": plan.config.configuration_sha256,
        "model_label": model.label,
        "requested_model": model.requested_model,
        "resolved_model": model.resolved_model,
        "request_plan_sha256": canonical_sha256(
            _model_request_plan(plan, model)
        ),
        "successful_source_ids": [item["source_id"] for item in predictions],
        "attempted_source_ids": [
            item["source_id"]
            for item in sorted(attempts, key=lambda item: item["position"])
        ],
        "failure_count": len(failures),
        "input_tokens_charged": sum(
            item["provider_metadata"]["charged_input_tokens"] for item in attempts
        ),
        "output_tokens_charged": sum(
            item["provider_metadata"]["charged_output_tokens"] for item in attempts
        ),
        "cost_usd": cost_text(_attempt_cost(predictions, failures, model)),
        "artifact_sha256": artifact_hashes,
        "budget": dict(budget_record),
    }
    _write_json(output / "run.json", run)


def _load_checkpoint(
    output: Path,
    plan: Step35StagePlan,
    model: Step35ModelPlan,
    *,
    allow_completed: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    if not output.is_dir() or not _MODEL_FILES <= {item.name for item in output.iterdir()}:
        raise Step35RunError("resume checkpoint is incomplete")
    try:
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        predictions = _read_jsonl(output / "predictions.jsonl")
        failures = _read_jsonl(output / "failures.jsonl")
    except (OSError, json.JSONDecodeError) as error:
        raise Step35RunError(f"resume checkpoint is invalid: {error}") from error
    if run.get("status") == "completed" and not allow_completed:
        raise Step35RunError("completed results cannot resume")
    expected_run_fields = {
        "run_format_version",
        "status",
        "started_at",
        "updated_at",
        "repository_commit",
        "repository_dirty",
        "configuration_version",
        "configuration_sha256",
        "model_label",
        "requested_model",
        "resolved_model",
        "request_plan_sha256",
        "successful_source_ids",
        "attempted_source_ids",
        "failure_count",
        "input_tokens_charged",
        "output_tokens_charged",
        "cost_usd",
        "artifact_sha256",
        "budget",
    }
    if not isinstance(run, dict) or set(run) != expected_run_fields:
        raise Step35RunError("resume run fields changed")
    if run.get("run_format_version") != "phase4-input-run-v1" or run.get(
        "status"
    ) not in {"running", "failed", "generated", "completed"}:
        raise Step35RunError("resume run status or format is invalid")
    if (
        run.get("configuration_sha256") != plan.config.configuration_sha256
        or run.get("configuration_version") != plan.config.configuration_version
        or run.get("model_label") != model.label
        or run.get("requested_model") != model.requested_model
        or run.get("resolved_model") != model.resolved_model
        or run.get("request_plan_sha256")
        != canonical_sha256(_model_request_plan(plan, model))
    ):
        raise Step35RunError("resume checkpoint does not match the frozen plan")
    _verify_checkpoint_hashes(output, run)
    allowed_files = {
        "predictions.jsonl",
        "failures.jsonl",
        "run.json",
        "claims.jsonl",
        "case_scores.jsonl",
        "scores.json",
        "known_limitations.json",
        "evidence_index.json",
        "manifest.json",
    }
    if any(
        item.name not in allowed_files or not item.is_file()
        for item in output.iterdir()
    ):
        raise Step35RunError("resume checkpoint contains unexpected artifacts")
    seen_predictions: set[str] = set()
    for index, item in enumerate(predictions, 1):
        if set(item) != {
            "position",
            "user_id",
            "source_id",
            "claims",
            "provider_metadata",
        }:
            raise Step35RunError("resume prediction fields changed")
        source = plan.sources[index - 1] if index <= len(plan.sources) else None
        source_id = item.get("source_id")
        if (
            source is None
            or item.get("position") != index
            or source_id != source.source_id
            or item.get("user_id") != source.user_id
            or source_id in seen_predictions
        ):
            raise Step35RunError(
                "resume predictions are reordered, duplicated, or cross-user"
            )
        seen_predictions.add(source_id)
        validate_phase4_claim_file(item.get("claims", []), (source,), plan.registry)
        _validate_saved_metadata(
            item.get("provider_metadata"),
            model,
            successful=True,
            maximum_input=plan.maximum_input_tokens[model.label][index - 1],
        )
    seen_failures: set[str] = set()
    if plan.config.maximum_retry_requests == 0 and len(failures) > 1:
        raise Step35RunError("zero-retry checkpoints cannot contain multiple failures")
    for item in failures:
        required = {
            "failure_id",
            "position",
            "user_id",
            "source_id",
            "stage",
            "error",
            "attempted",
            "resolved",
            "failure_categories",
            "provider_metadata",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise Step35RunError("resume failure fields changed")
        position = item.get("position")
        source = (
            plan.sources[position - 1]
            if isinstance(position, int)
            and not isinstance(position, bool)
            and 1 <= position <= len(plan.sources)
            else None
        )
        expected_id = _failure_id(
            {key: value for key, value in item.items() if key != "failure_id"}
        )
        if (
            source is None
            or item.get("source_id") != source.source_id
            or item.get("user_id") != source.user_id
            or item.get("failure_id") != expected_id
            or item["failure_id"] in seen_failures
            or position > len(predictions) + 1
        ):
            raise Step35RunError(
                "resume failures are reordered, duplicated, or cross-user"
            )
        if plan.config.maximum_retry_requests == 0 and item.get("resolved") is True:
            raise Step35RunError("zero-retry checkpoints cannot contain resolved failures")
        if plan.config.maximum_retry_requests == 0 and position != len(predictions) + 1:
            raise Step35RunError("zero-retry failure order changed")
        if item.get("stage") not in {
            "provider",
            "model_mismatch",
            "budget",
            "validation",
            "persistence",
        }:
            raise Step35RunError("resume failure stage is invalid")
        categories = item.get("failure_categories")
        if (
            item.get("attempted") is not True
            or not isinstance(item.get("resolved"), bool)
            or not isinstance(item.get("error"), str)
            or not item["error"]
            or not isinstance(categories, list)
            or not categories
            or len(categories) != len(set(categories))
            or not set(categories) <= {
                "execution",
                "persistence",
                "cross_user",
                "provenance",
            }
        ):
            raise Step35RunError("resume failure details are invalid")
        seen_failures.add(item["failure_id"])
        _validate_saved_metadata(
            item.get("provider_metadata"),
            model,
            successful=False,
            maximum_input=plan.maximum_input_tokens[model.label][position - 1],
        )
    if run.get("successful_source_ids") != [
        item["source_id"] for item in predictions
    ]:
        raise Step35RunError("resume checkpoint summary is inconsistent")
    attempts = sorted([*predictions, *failures], key=lambda item: item["position"])
    if run.get("attempted_source_ids") != [item["source_id"] for item in attempts]:
        raise Step35RunError("resume attempt order is inconsistent")
    if run.get("failure_count") != len(failures):
        raise Step35RunError("resume failure count is inconsistent")
    if (
        run.get("input_tokens_charged")
        != sum(item["provider_metadata"]["charged_input_tokens"] for item in attempts)
        or run.get("output_tokens_charged")
        != sum(item["provider_metadata"]["charged_output_tokens"] for item in attempts)
        or run.get("cost_usd")
        != cost_text(_attempt_cost(predictions, failures, model))
    ):
        raise Step35RunError("resume usage accounting is inconsistent")
    if run["status"] in {"generated", "completed"} and (
        len(predictions) != len(plan.sources)
        or any(item.get("resolved") is not True for item in failures)
    ):
        raise Step35RunError("completed generation contains unfinished cases")
    if run["status"] == "failed" and not failures:
        raise Step35RunError("failed checkpoint has no failure")
    started_at = run.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise Step35RunError("resume checkpoint start time is invalid")
    return predictions, failures, started_at


def _validate_saved_metadata(
    record: object,
    model: Step35ModelPlan,
    *,
    successful: bool,
    maximum_input: int,
) -> None:
    required = {
        "response_id",
        "returned_model",
        "request_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "charged_input_tokens",
        "charged_output_tokens",
        "charged_cost_usd",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise Step35RunError("resume provider metadata fields changed")
    if successful and record.get("returned_model") != model.resolved_model:
        raise Step35RunError("resume provider model changed")
    charged_input = record.get("charged_input_tokens")
    charged_output = record.get("charged_output_tokens")
    if (
        not isinstance(charged_input, int)
        or isinstance(charged_input, bool)
        or charged_input < 0
        or not isinstance(charged_output, int)
        or isinstance(charged_output, bool)
        or not 0 <= charged_output <= _MAX_OUTPUT_TOKENS
    ):
        raise Step35RunError("resume charged token counts are invalid")
    if successful and charged_input > maximum_input:
        raise Step35RunError("resume input usage exceeded its reservation")
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = record.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise Step35RunError("resume provider usage metadata is invalid")
    if record.get("charged_cost_usd") != cost_text(
        step35_cost(model, charged_input, charged_output)
    ):
        raise Step35RunError("resume charged cost is inconsistent")


def _metadata_record(
    metadata: OpenAIResponseMetadata,
    maximum_input: int,
    model: Step35ModelPlan,
    *,
    enforce_limits: bool = True,
) -> dict[str, object]:
    for name, value in (
        ("input_tokens", metadata.input_tokens),
        ("output_tokens", metadata.output_tokens),
        ("total_tokens", metadata.total_tokens),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise Step35RunError(f"provider {name} is incompatible")
    if metadata.input_tokens is None or metadata.output_tokens is None:
        charged_input = maximum_input
        charged_output = _MAX_OUTPUT_TOKENS
    else:
        charged_input = metadata.input_tokens
        charged_output = metadata.output_tokens
    if enforce_limits and (
        charged_input > maximum_input or charged_output > _MAX_OUTPUT_TOKENS
    ):
        raise Step35RunError("provider usage exceeded the request reservation")
    return {
        "response_id": metadata.response_id,
        "returned_model": metadata.returned_model,
        "request_id": metadata.request_id,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "total_tokens": metadata.total_tokens,
        "charged_input_tokens": charged_input,
        "charged_output_tokens": charged_output,
        "charged_cost_usd": cost_text(step35_cost(model, charged_input, charged_output)),
    }


def _missing_metadata_record(
    maximum_input: int,
    model: Step35ModelPlan,
) -> dict[str, object]:
    return {
        "response_id": None,
        "returned_model": None,
        "request_id": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "charged_input_tokens": maximum_input,
        "charged_output_tokens": _MAX_OUTPUT_TOKENS,
        "charged_cost_usd": cost_text(
            step35_cost(model, maximum_input, _MAX_OUTPUT_TOKENS)
        ),
    }


def _attempt_cost(
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    model: Step35ModelPlan,
) -> Decimal:
    attempts = [*predictions, *failures]
    return step35_cost(
        model,
        sum(item["provider_metadata"]["charged_input_tokens"] for item in attempts),
        sum(item["provider_metadata"]["charged_output_tokens"] for item in attempts),
    )


def _failure_counts(failures: Sequence[Mapping[str, object]]) -> dict[str, int]:
    result = {"execution": 0, "persistence": 0, "cross_user": 0, "provenance": 0}
    for failure in failures:
        if failure.get("resolved") is True:
            continue
        for category in failure.get("failure_categories", ("execution",)):
            if category in result:
                result[category] += 1
    return result


def _safe_failure(
    source: ExtractionSource,
    position: int,
    stage: str,
    categories: Sequence[str],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return _failure_record(
        source,
        position,
        stage,
        categories,
        metadata,
        f"The source result failed {stage} checks.",
    )


def _failure_record(
    source: ExtractionSource,
    position: int,
    stage: str,
    categories: Sequence[str],
    metadata: Mapping[str, object],
    error: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "position": position,
        "user_id": source.user_id,
        "source_id": source.source_id,
        "stage": stage,
        "error": error,
        "attempted": True,
        "resolved": False,
        "failure_categories": list(categories),
        "provider_metadata": dict(metadata),
    }
    return {"failure_id": _failure_id(record), **record}


def _failure_id(record: Mapping[str, object]) -> str:
    return f"failure_{canonical_sha256(record)}"


def _restore_budget(
    budget: Step35Budget,
    model: Step35ModelPlan,
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> None:
    attempts = sorted([*predictions, *failures], key=lambda item: item["position"])
    for item in attempts:
        if budget.cursor >= len(budget.reservations):
            raise Step35RunError("checkpoint usage exceeds the request ceiling")
        expected = budget.reservations[budget.cursor]
        if expected.model.label != model.label or expected.source_id != item["source_id"]:
            raise Step35RunError("checkpoint request order changed")
        budget.charge_response(item["provider_metadata"])


def _verify_checkpoint_hashes(output: Path, run: Mapping[str, object]) -> None:
    hashes = run.get("artifact_sha256")
    required = {"predictions.jsonl", "failures.jsonl"}
    if run.get("status") == "completed":
        required |= {"claims.jsonl", "case_scores.jsonl", "scores.json"}
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise Step35RunError("checkpoint artifact hashes are incomplete")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or _file_sha256(output / name) != expected:
            raise Step35RunError(f"checkpoint artifact changed: {name}")


def _validation_failure_categories(
    error: AtomicExtractionValidationError,
) -> tuple[str, ...]:
    categories = {"execution"}
    if any("evidence" in detail or ".quote" in detail for detail in error.errors):
        categories.add("provenance")
    return tuple(sorted(categories))


def _persistence_failure_categories(
    error: Phase4InputValidationError,
) -> tuple[str, ...]:
    detail = str(error)
    categories = {"persistence"}
    if "user boundary" in detail:
        categories.add("cross_user")
    if any(word in detail for word in ("evidence", "source", "quote")):
        categories.add("provenance")
    return tuple(sorted(categories))


def _atomic_record(claim: object) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "subject_id": claim.subject_id,
        "speaker_id": claim.speaker_id,
        "predicate": claim.predicate,
        "object": claim.object,
        "polarity": claim.polarity,
        "epistemic_status": claim.epistemic_status,
        "valid_from": claim.valid_from,
        "valid_to": claim.valid_to,
        "confidence": claim.confidence,
        "evidence": [
            {"source_id": item.source_id, "message_id": item.message_id, "quote": item.quote}
            for item in claim.evidence
        ],
    }


def _require_new_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise Step35RunError(f"refusing to overwrite non-empty output: {path}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise Step35RunError(f"{path.name} contains a non-object record")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for item in records
    )
    assert_no_secrets(content, str(path))
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_secrets(record, str(path))
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Step35RunError(f"could not read {path.name}: {error}") from error
    if not isinstance(record, dict):
        raise Step35RunError(f"{path.name} must contain one JSON object")
    return record


def _json_content(record: Mapping[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _jsonl_content(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in records
    )


def _write_or_verify_json(path: Path, record: Mapping[str, object]) -> None:
    _write_or_verify_text(path, _json_content(record))


def _write_or_verify_jsonl(
    path: Path, records: Sequence[Mapping[str, object]]
) -> None:
    _write_or_verify_text(path, _jsonl_content(records))


def _write_or_verify_text(path: Path, content: str) -> None:
    assert_no_secrets(content, str(path))
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as error:
            raise Step35RunError(f"could not verify {path}: {error}") from error
        if existing != content:
            raise Step35RunError(f"refusing to overwrite changed artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _verify_manifest_files(root: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise Step35RunError("manifest file hashes are missing")
    for name, expected in files.items():
        relative = Path(name) if isinstance(name, str) else Path("/")
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise Step35RunError("manifest file entry is invalid")
        if _file_sha256(root / relative) != expected:
            raise Step35RunError(f"manifest-bound artifact changed: {name}")


def _require_allowed_qualification_root(
    path: Path, root: Path, family: Step35RunFamily
) -> None:
    if path != root / family.qualification_root:
        raise Step35RunError("qualification output root is not allowlisted")
    allowed = {
        "gpt41",
        "gpt41-mini",
        "decision.json",
        "comparison.json",
        "findings.md",
        "final_preflight.json",
        "manifest.json",
    }
    if not path.is_dir() or any(item.name not in allowed for item in path.iterdir()):
        raise Step35RunError("qualification output contains unexpected artifacts")


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Step35RunError(f"could not hash {path}: {error}") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    return bool(_git_status_text().strip())


def _git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise Step35RunError("could not inspect the repository worktree")
    return result.stdout


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Step 3.5 pipeline.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-qualification", action="store_true")
    parser.add_argument("--execute-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--qualification-config", type=Path, default=QUALIFICATION_CONFIG_PATH
    )
    parser.add_argument("--full-config", type=Path, default=FULL_CONFIG_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    modes = sum((args.dry_run, args.execute_qualification, args.execute_final))
    if modes != 1:
        parser.error("choose exactly one run mode")
    if args.dry_run:
        if args.resume:
            parser.error("--resume requires an execute mode")
        dry_run_step35(
            qualification_config=args.qualification_config,
            full_config=args.full_config,
        )
        return 0
    key = load_env_value(args.env_file, "OPENAI_API_KEY")
    if args.execute_qualification:
        family = _run_family(
            Path(".").resolve(), args.qualification_config, args.full_config
        )
        plan = prepare_step35_stage(family.qualification_config_path)
        clients = {
            model.label: OpenAIResponsesClient(
                api_key=key,
                model=model.requested_model,
                temperature=0.0,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                text_format=plan.text_format,
            )
            for model in plan.config.models
        }
        execute_qualification(
            repo_root=".",
            clients=clients,
            resume=args.resume,
            qualification_config=args.qualification_config,
            full_config=args.full_config,
        )
    else:
        family = _run_family(
            Path(".").resolve(), args.qualification_config, args.full_config
        )
        plan = prepare_step35_stage(family.full_config_path)
        model = plan.config.models[0]
        client = OpenAIResponsesClient(
            api_key=key,
            model=model.requested_model,
            temperature=0.0,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            text_format=plan.text_format,
        )
        execute_full_phase4_input(
            repo_root=".",
            client=client,
            resume=args.resume,
            qualification_config=args.qualification_config,
            full_config=args.full_config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
