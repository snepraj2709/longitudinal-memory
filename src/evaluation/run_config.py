"""Freeze and validate reproducible full-history baseline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .history import (
    FULL_HISTORY_PROMPT_VERSION,
    FULL_HISTORY_SYSTEM_PROMPT,
    FULL_HISTORY_USER_PROMPT_TEMPLATE,
    SOURCE_ORDERING_RULE,
)
from .smoke import SMOKE_CASE_IDS
from .prediction import PredictionValidationError, validate_prediction


PILOT_DATASET_FILES = (
    "data/pilot/evaluation/eval_answer.jsonl",
    "data/pilot/evaluation/eval_questions.jsonl",
    "data/pilot/oracle-event.jsonl",
    "data/pilot/sources/calendar.jsonl",
    "data/pilot/sources/conversations.jsonl",
    "data/pilot/sources/emails.jsonl",
    "data/pilot/user.jsonl",
)

FULL_HISTORY_GENERATION_SETTINGS = {
    "api": "responses",
    "max_output_tokens": 1000,
    "store": False,
    "text_format": "json_object",
}

_CONFIG_FIELDS = frozenset(
    {
        "configuration_version",
        "baseline_id",
        "provider",
        "requested_model",
        "resolved_model",
        "temperature",
        "generation_settings",
        "prompt_version",
        "prompt_sha256",
        "dataset_sha256",
        "dataset_files",
        "source_ordering_rule",
        "configuration_sha256",
    }
)
_CONTROLLED_CONFIG_FIELDS = _CONFIG_FIELDS - {
    "configuration_version",
    "configuration_sha256",
}
_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "run_id",
        "run_date",
        "repository_commit",
        "prompt_files_dirty",
        "dataset_files_dirty",
        "frozen_configuration_path",
        "frozen_configuration_sha256",
        "output_artifacts",
        "step3_exit_status",
        "returned_model",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"predictions", "diagnostics", "api_metadata", "report"}
)
_ARTIFACT_FILENAMES = {
    "predictions": "predictions.jsonl",
    "diagnostics": "diagnostics.jsonl",
    "api_metadata": "api_metadata.jsonl",
    "report": "report.md",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|password|credential|secret)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERN = re.compile(r"^(?:Bearer\s+\S+|sk-[A-Za-z0-9_-]+)", re.IGNORECASE)


class RunConfigurationError(ValueError):
    """Report every validation error found in one config or manifest."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class FrozenBaselineConfig:
    """Static settings and content fingerprints for one baseline version."""

    configuration_version: str
    baseline_id: str
    provider: str
    requested_model: str
    resolved_model: str
    temperature: float
    generation_settings: Mapping[str, object]
    prompt_version: str
    prompt_sha256: str
    dataset_sha256: str
    dataset_files: tuple[str, ...]
    source_ordering_rule: str
    configuration_sha256: str


@dataclass(frozen=True)
class RunManifest:
    """Execution-specific evidence linked to one frozen configuration."""

    manifest_version: str
    run_id: str
    run_date: str
    repository_commit: str
    prompt_files_dirty: bool
    dataset_files_dirty: bool
    frozen_configuration_path: str
    frozen_configuration_sha256: str
    output_artifacts: Mapping[str, str]
    step3_exit_status: int
    returned_model: str


@dataclass(frozen=True)
class CompletedStep3Run:
    """Verified provider details recovered from completed Step 3 artifacts."""

    provider: str
    requested_model: str
    returned_model: str
    temperature: float
    generation_settings: Mapping[str, object]


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using deterministic encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_sha256(
    system_prompt: str = FULL_HISTORY_SYSTEM_PROMPT,
    user_prompt_template: str = FULL_HISTORY_USER_PROMPT_TEMPLATE,
) -> str:
    """Fingerprint static prompt text without case-specific values."""

    return canonical_sha256(
        {
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
        }
    )


def dataset_sha256(repo_root: str | Path, files: Sequence[str]) -> str:
    """Hash sorted repository-relative paths and their exact bytes."""

    repo_root = Path(repo_root).resolve()
    normalized = sorted(_normalize_repo_relative_path(path) for path in files)
    if len(normalized) != len(set(normalized)):
        raise RunConfigurationError(("dataset_files must not contain duplicates",))

    digest = hashlib.sha256()
    for relative_path in normalized:
        path = repo_root / relative_path
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RunConfigurationError(
                (f"could not read dataset file {relative_path!r}: {error}",)
            ) from error
        path_bytes = relative_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def discover_pilot_jsonl_files(repo_root: str | Path) -> tuple[str, ...]:
    """List every pilot JSONL path in stable repository-relative order."""

    repo_root = Path(repo_root).resolve()
    pilot_root = repo_root / "data" / "pilot"
    return tuple(
        path.relative_to(repo_root).as_posix()
        for path in sorted(pilot_root.rglob("*.jsonl"))
        if path.is_file()
    )


def configuration_sha256(config: FrozenBaselineConfig | Mapping[str, object]) -> str:
    """Hash every controlled configuration field except the hash itself."""

    record = _config_record(config)
    record.pop("configuration_sha256", None)
    return canonical_sha256(record)


def build_frozen_config(
    *,
    repo_root: str | Path,
    completed_run: CompletedStep3Run,
    configuration_version: str = "1",
    baseline_id: str = "B1_full_history",
) -> FrozenBaselineConfig:
    """Build and validate a frozen config from verified Step 3 evidence."""

    dataset_files = discover_pilot_jsonl_files(repo_root)
    if dataset_files != PILOT_DATASET_FILES:
        raise RunConfigurationError(
            ("pilot JSONL files do not match the expected B1 v1 dataset set",)
        )
    partial = {
        "configuration_version": configuration_version,
        "baseline_id": baseline_id,
        "provider": completed_run.provider,
        "requested_model": completed_run.requested_model,
        "resolved_model": completed_run.returned_model,
        "temperature": completed_run.temperature,
        "generation_settings": dict(completed_run.generation_settings),
        "prompt_version": FULL_HISTORY_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "dataset_sha256": dataset_sha256(repo_root, dataset_files),
        "dataset_files": list(dataset_files),
        "source_ordering_rule": SOURCE_ORDERING_RULE,
    }
    partial["configuration_sha256"] = canonical_sha256(partial)
    return validate_frozen_config(partial)


def validate_frozen_config(value: object) -> FrozenBaselineConfig:
    """Strictly validate a frozen baseline config and its self-hash."""

    errors: list[str] = []
    record = _require_exact_object(value, _CONFIG_FIELDS, "config", errors)
    _reject_secrets(value, "config", errors)

    strings = {
        field: _non_empty_string(record.get(field), f"config.{field}", errors)
        for field in (
            "configuration_version",
            "baseline_id",
            "provider",
            "requested_model",
            "resolved_model",
            "prompt_version",
            "prompt_sha256",
            "dataset_sha256",
            "source_ordering_rule",
            "configuration_sha256",
        )
    }
    temperature = _finite_number(record.get("temperature"), "config.temperature", errors)
    settings_value = record.get("generation_settings")
    if not isinstance(settings_value, dict):
        errors.append("config.generation_settings must be an object")
        settings: dict[str, object] = {}
    else:
        settings = dict(settings_value)
        if settings != FULL_HISTORY_GENERATION_SETTINGS:
            errors.append(
                "config.generation_settings must exactly match the full-history settings"
            )

    files_value = record.get("dataset_files")
    files: tuple[str, ...] = ()
    if not isinstance(files_value, list):
        errors.append("config.dataset_files must be a list")
    else:
        parsed_files: list[str] = []
        for index, item in enumerate(files_value):
            parsed = _non_empty_string(item, f"config.dataset_files[{index}]", errors)
            if parsed:
                try:
                    parsed_files.append(_normalize_repo_relative_path(parsed))
                except RunConfigurationError as error:
                    errors.extend(error.errors)
        files = tuple(parsed_files)
        if files != PILOT_DATASET_FILES:
            errors.append("config.dataset_files must contain every pilot JSONL in sorted order")

    for field in ("prompt_sha256", "dataset_sha256", "configuration_sha256"):
        field_value = strings[field]
        if field_value and not _SHA256_PATTERN.fullmatch(field_value):
            errors.append(f"config.{field} must be a lowercase SHA-256 hex digest")
    if strings["prompt_version"] and strings["prompt_version"] != FULL_HISTORY_PROMPT_VERSION:
        errors.append(
            f"config.prompt_version must be {FULL_HISTORY_PROMPT_VERSION!r}"
        )
    if strings["source_ordering_rule"] and strings["source_ordering_rule"] != SOURCE_ORDERING_RULE:
        errors.append("config.source_ordering_rule does not match the history builder")
    if temperature is not None and temperature != 0:
        errors.append("config.temperature must be 0")

    if not errors:
        expected_hash = canonical_sha256(
            {key: item for key, item in record.items() if key != "configuration_sha256"}
        )
        if strings["configuration_sha256"] != expected_hash:
            errors.append("config.configuration_sha256 does not match its contents")

    if errors:
        raise RunConfigurationError(errors)
    return FrozenBaselineConfig(
        configuration_version=strings["configuration_version"],
        baseline_id=strings["baseline_id"],
        provider=strings["provider"],
        requested_model=strings["requested_model"],
        resolved_model=strings["resolved_model"],
        temperature=float(temperature),
        generation_settings=settings,
        prompt_version=strings["prompt_version"],
        prompt_sha256=strings["prompt_sha256"],
        dataset_sha256=strings["dataset_sha256"],
        dataset_files=files,
        source_ordering_rule=strings["source_ordering_rule"],
        configuration_sha256=strings["configuration_sha256"],
    )


def load_frozen_config(path: str | Path) -> FrozenBaselineConfig:
    """Load JSON and apply the strict frozen-config contract."""

    return validate_frozen_config(_load_json_object(Path(path), "frozen config"))


def write_frozen_config(path: str | Path, config: FrozenBaselineConfig) -> None:
    """Write a validated frozen config as stable, readable JSON."""

    validated = validate_frozen_config(_config_record(config))
    _write_json(Path(path), _config_record(validated))


def assert_no_controlled_drift(
    frozen: FrozenBaselineConfig, candidate: FrozenBaselineConfig
) -> None:
    """Reject changed controlled fields unless the version also changes."""

    if frozen.configuration_version != candidate.configuration_version:
        return
    frozen_record = _config_record(frozen)
    candidate_record = _config_record(candidate)
    changed = sorted(
        key
        for key in _CONTROLLED_CONFIG_FIELDS
        if frozen_record[key] != candidate_record[key]
    )
    if changed:
        raise RunConfigurationError(
            (
                "controlled configuration drift requires a new configuration_version: "
                + ", ".join(changed),
            )
        )


def verify_frozen_content(
    config: FrozenBaselineConfig, repo_root: str | Path
) -> None:
    """Reject prompt or pilot-data drift from a frozen config."""

    errors: list[str] = []
    current_prompt_hash = prompt_sha256()
    current_dataset_files = discover_pilot_jsonl_files(repo_root)
    current_dataset_hash = dataset_sha256(repo_root, config.dataset_files)
    if current_prompt_hash != config.prompt_sha256:
        errors.append("current prompt text does not match config.prompt_sha256")
    if current_dataset_hash != config.dataset_sha256:
        errors.append("current pilot data does not match config.dataset_sha256")
    if current_dataset_files != config.dataset_files:
        errors.append("current pilot JSONL file list does not match config.dataset_files")
    if errors:
        raise RunConfigurationError(errors)


def inspect_completed_step3_run(run_dir: str | Path) -> CompletedStep3Run:
    """Verify that artifacts came from a successful five-call provider run."""

    run_dir = Path(run_dir)
    errors: list[str] = []
    predictions = _load_jsonl(run_dir / "predictions.jsonl", errors)
    diagnostics = _load_jsonl(run_dir / "diagnostics.jsonl", errors)
    metadata = _load_jsonl(run_dir / "api_metadata.jsonl", errors)

    expected_ids = list(SMOKE_CASE_IDS)
    if [item.get("case_id") for item in predictions] != expected_ids:
        errors.append("predictions.jsonl must contain the five smoke cases in order")
    for index, item in enumerate(predictions):
        try:
            prediction = validate_prediction(item)
        except PredictionValidationError as error:
            errors.append(
                f"predictions.jsonl record {index + 1} violates the contract: {error}"
            )
        else:
            if index < len(expected_ids) and prediction.case_id != expected_ids[index]:
                errors.append(
                    f"predictions.jsonl record {index + 1} has the wrong case_id"
                )
    if [item.get("case_id") for item in diagnostics] != expected_ids:
        errors.append("diagnostics.jsonl must contain the five smoke cases in order")
    if [item.get("case_id") for item in metadata] != expected_ids:
        errors.append("api_metadata.jsonl must contain the five smoke cases in order")

    for index, item in enumerate(diagnostics):
        for field in ("valid_json", "valid_contract", "case_id_matches", "exact_evidence"):
            if item.get(field) is not True:
                errors.append(f"diagnostics.jsonl record {index + 1} has {field} != true")
        if item.get("validation_error") is not None:
            errors.append(
                f"diagnostics.jsonl record {index + 1} has a validation error"
            )

    returned_models: list[str] = []
    for index, item in enumerate(metadata):
        response_id = item.get("response_id")
        returned_model = item.get("returned_model")
        if not isinstance(response_id, str) or not response_id.strip():
            errors.append(f"api_metadata.jsonl record {index + 1} lacks a response_id")
        elif not response_id.startswith("resp_") or any(
            marker in response_id.lower() for marker in ("fake", "dry-run", "dry_run")
        ):
            errors.append(f"api_metadata.jsonl record {index + 1} is not a live response")
        if not isinstance(returned_model, str) or not returned_model.strip():
            errors.append(f"api_metadata.jsonl record {index + 1} lacks returned_model")
        else:
            returned_models.append(returned_model)

    provider = ""
    requested_model = ""
    temperature: float | None = None
    settings: dict[str, object] = {}
    report_path = run_dir / "report.md"
    try:
        report_lines = report_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"could not read {report_path}: {error}")
        report_lines = []
    for line in report_lines:
        if line.startswith("Provider: "):
            provider = line[len("Provider: ") :].strip()
        elif line.startswith("Model: "):
            requested_model = line[len("Model: ") :].strip()
        elif line.startswith("Settings: `") and line.endswith("`"):
            try:
                parsed_settings = json.loads(line[len("Settings: `") : -1])
            except json.JSONDecodeError as error:
                errors.append(f"report settings are invalid JSON: {error.msg}")
            else:
                if isinstance(parsed_settings, dict):
                    temperature_value = parsed_settings.pop("temperature", None)
                    temperature = _finite_number(
                        temperature_value, "report temperature", errors
                    )
                    settings = parsed_settings
                else:
                    errors.append("report settings must be a JSON object")
    if not provider:
        errors.append("report does not identify the provider")
    if not requested_model:
        errors.append("report does not identify the requested model")
    unique_returned_models = set(returned_models)
    if len(unique_returned_models) != 1:
        errors.append("provider metadata must contain one exact returned model")
        returned_model = ""
    else:
        returned_model = next(iter(unique_returned_models))
    if requested_model and returned_model and requested_model != returned_model:
        errors.append("requested and returned model snapshots must match exactly")
    if temperature is None:
        errors.append("report does not contain a valid temperature")
    elif temperature != 0:
        errors.append("Step 3 temperature must be 0")
    if settings != FULL_HISTORY_GENERATION_SETTINGS:
        errors.append("Step 3 generation settings do not match the frozen settings")

    if errors:
        raise RunConfigurationError(errors)
    return CompletedStep3Run(
        provider=provider,
        requested_model=requested_model,
        returned_model=returned_model,
        temperature=float(temperature),
        generation_settings=settings,
    )


def build_run_manifest(
    *,
    run_id: str,
    run_date: str,
    repository_commit: str,
    prompt_files_dirty: bool,
    dataset_files_dirty: bool,
    frozen_configuration_path: str,
    config: FrozenBaselineConfig,
    output_artifacts: Mapping[str, str],
    step3_exit_status: int,
    returned_model: str,
) -> RunManifest:
    """Build and validate execution metadata for one completed run."""

    return validate_run_manifest(
        {
            "manifest_version": "1",
            "run_id": run_id,
            "run_date": run_date,
            "repository_commit": repository_commit,
            "prompt_files_dirty": prompt_files_dirty,
            "dataset_files_dirty": dataset_files_dirty,
            "frozen_configuration_path": frozen_configuration_path,
            "frozen_configuration_sha256": config.configuration_sha256,
            "output_artifacts": dict(output_artifacts),
            "step3_exit_status": step3_exit_status,
            "returned_model": returned_model,
        },
        config=config,
    )


def validate_run_manifest(
    value: object, *, config: FrozenBaselineConfig | None = None
) -> RunManifest:
    """Strictly validate one run manifest and its config linkage."""

    errors: list[str] = []
    record = _require_exact_object(value, _MANIFEST_FIELDS, "manifest", errors)
    _reject_secrets(value, "manifest", errors)
    strings = {
        field: _non_empty_string(record.get(field), f"manifest.{field}", errors)
        for field in (
            "manifest_version",
            "run_id",
            "run_date",
            "repository_commit",
            "frozen_configuration_path",
            "frozen_configuration_sha256",
            "returned_model",
        )
    }
    if strings["run_date"] and not _is_aware_timestamp(strings["run_date"]):
        errors.append("manifest.run_date must be an ISO 8601 timestamp with a UTC offset")
    if strings["repository_commit"] and not _COMMIT_PATTERN.fullmatch(
        strings["repository_commit"]
    ):
        errors.append("manifest.repository_commit must be a full lowercase Git commit")
    if strings["frozen_configuration_sha256"] and not _SHA256_PATTERN.fullmatch(
        strings["frozen_configuration_sha256"]
    ):
        errors.append("manifest.frozen_configuration_sha256 must be a SHA-256 digest")
    prompt_dirty = record.get("prompt_files_dirty")
    if not isinstance(prompt_dirty, bool):
        errors.append("manifest.prompt_files_dirty must be a boolean")
    dataset_dirty = record.get("dataset_files_dirty")
    if not isinstance(dataset_dirty, bool):
        errors.append("manifest.dataset_files_dirty must be a boolean")
    exit_status = record.get("step3_exit_status")
    if not isinstance(exit_status, int) or isinstance(exit_status, bool):
        errors.append("manifest.step3_exit_status must be an integer")
    elif exit_status != 0:
        errors.append("manifest.step3_exit_status must be 0 before freezing")

    artifacts_value = record.get("output_artifacts")
    artifacts: dict[str, str] = {}
    if not isinstance(artifacts_value, dict):
        errors.append("manifest.output_artifacts must be an object")
    else:
        unknown = sorted(set(artifacts_value) - _ARTIFACT_FIELDS)
        missing = sorted(_ARTIFACT_FIELDS - set(artifacts_value))
        if unknown:
            errors.append("manifest.output_artifacts has unknown fields: " + ", ".join(unknown))
        if missing:
            errors.append("manifest.output_artifacts is missing fields: " + ", ".join(missing))
        for key, item in artifacts_value.items():
            parsed = _non_empty_string(item, f"manifest.output_artifacts.{key}", errors)
            if parsed:
                try:
                    artifacts[key] = _normalize_repo_relative_path(parsed)
                except RunConfigurationError as error:
                    errors.extend(error.errors)
                if key in _ARTIFACT_FILENAMES and Path(parsed).name != _ARTIFACT_FILENAMES[key]:
                    errors.append(
                        f"manifest.output_artifacts.{key} must reference "
                        f"{_ARTIFACT_FILENAMES[key]!r}"
                    )

    if config is not None:
        if strings["frozen_configuration_sha256"] != config.configuration_sha256:
            errors.append("manifest frozen config hash does not match the supplied config")
        if strings["returned_model"] != config.resolved_model:
            errors.append("manifest returned model does not match config.resolved_model")

    if errors:
        raise RunConfigurationError(errors)
    return RunManifest(
        manifest_version=strings["manifest_version"],
        run_id=strings["run_id"],
        run_date=strings["run_date"],
        repository_commit=strings["repository_commit"],
        prompt_files_dirty=prompt_dirty,
        dataset_files_dirty=dataset_dirty,
        frozen_configuration_path=strings["frozen_configuration_path"],
        frozen_configuration_sha256=strings["frozen_configuration_sha256"],
        output_artifacts=artifacts,
        step3_exit_status=exit_status,
        returned_model=strings["returned_model"],
    )


def load_run_manifest(
    path: str | Path, *, config: FrozenBaselineConfig | None = None
) -> RunManifest:
    """Load JSON and apply the strict run-manifest contract."""

    return validate_run_manifest(_load_json_object(Path(path), "run manifest"), config=config)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Write a validated manifest as stable, readable JSON."""

    validated = validate_run_manifest(asdict(manifest))
    _write_json(Path(path), asdict(validated))


def verify_manifest_artifacts(manifest: RunManifest, repo_root: str | Path) -> None:
    """Require every frozen configuration and output reference to exist."""

    repo_root = Path(repo_root)
    paths = {
        "frozen_configuration_path": manifest.frozen_configuration_path,
        **{
            f"output_artifacts.{key}": value
            for key, value in manifest.output_artifacts.items()
        },
    }
    errors = [
        f"manifest.{name} does not exist: {relative_path}"
        for name, relative_path in paths.items()
        if not (repo_root / relative_path).is_file()
    ]
    if errors:
        raise RunConfigurationError(errors)


def assert_no_secrets(value: object, location: str = "artifact") -> None:
    """Reject secret-bearing field names or credential-shaped values."""

    errors: list[str] = []
    _reject_secrets(value, location, errors)
    if errors:
        raise RunConfigurationError(errors)


def _config_record(config: FrozenBaselineConfig | Mapping[str, object]) -> dict[str, object]:
    if isinstance(config, FrozenBaselineConfig):
        record = asdict(config)
        record["dataset_files"] = list(config.dataset_files)
        record["generation_settings"] = dict(config.generation_settings)
        return record
    return dict(config)


def _normalize_repo_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise RunConfigurationError((f"path must be repository-relative: {value!r}",))
    return path.as_posix()


def _require_exact_object(
    value: object, fields: frozenset[str], location: str, errors: list[str]
) -> dict[str, object]:
    if not isinstance(value, dict):
        errors.append(f"{location} must be an object")
        return {}
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        errors.append(f"{location} has unknown fields: " + ", ".join(unknown))
    if missing:
        errors.append(f"{location} is missing fields: " + ", ".join(missing))
    return value


def _non_empty_string(value: object, location: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location} must be a non-empty string")
        return ""
    return value


def _finite_number(value: object, location: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{location} must be a real finite number")
        return None
    if not math.isfinite(value):
        errors.append(f"{location} must be a real finite number")
        return None
    return float(value)


def _is_aware_timestamp(value: str) -> bool:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _reject_secrets(value: object, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_location = f"{location}.{key_text}"
            if _SECRET_KEY_PATTERN.search(key_text):
                errors.append(f"{item_location} is a secret-bearing field")
            _reject_secrets(item, item_location, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{location}[{index}]", errors)
    elif isinstance(value, str) and _SECRET_VALUE_PATTERN.match(value):
        errors.append(f"{location} appears to contain a secret")


def _load_json_object(path: Path, description: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunConfigurationError((f"could not read {description} {path}: {error}",)) from error
    except json.JSONDecodeError as error:
        raise RunConfigurationError((f"{description} {path} is invalid JSON: {error.msg}",)) from error


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"could not read {path}: {error}")
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{line_number} is invalid JSON: {error.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number} must be an object")
            continue
        records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
