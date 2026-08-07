"""Run the ten-case atomic-extraction pilot without mixing gold into prompts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, TextIO

from evaluation.openai_client import OpenAIResponsesClient, load_env_value
from evaluation.run_config import assert_no_secrets

from .atomic import AtomicExtractionResult, AtomicExtractionValidationError, extract_atomic_claims
from .gold import ATOMIC_GOLD_PATH, AtomicGoldCase, load_atomic_gold
from .prompt import ATOMIC_EXTRACTION_PROMPT_VERSION, build_atomic_extraction_prompt
from .scoring import score_atomic_extraction
from .source import PILOT_SOURCE_DIR, ExtractionSource, load_pilot_sources


DEFAULT_OUTPUT_DIR = Path("results/phase3/atomic-extraction")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4_000
FROZEN_ATOMIC_GOLD_SHA256 = (
    "e6cb100e27d1612d9b3502a3f792a9a4701bf74bc12876ad40254bd73b365cca"
)
ATOMIC_CASE_REFS = (
    ("atomic_cal_001", "cal_001"),
    ("atomic_conv_002", "conv_002"),
    ("atomic_conv_003", "conv_003"),
    ("atomic_cal_002", "cal_002"),
    ("atomic_conv_004", "conv_004"),
    ("atomic_cal_003", "cal_003"),
    ("atomic_email_003", "email_003"),
    ("atomic_email_005", "email_005"),
    ("atomic_conv_010", "conv_010"),
    ("atomic_cal_006", "cal_006"),
)
_SOURCE_FILENAMES = ("calendar.jsonl", "conversations.jsonl", "emails.jsonl")


class AtomicPipelineError(RuntimeError):
    """Raised when the pilot cannot run safely."""


@dataclass(frozen=True)
class AtomicRunPlan:
    """Source-only inputs prepared before extraction starts."""

    case_refs: tuple[tuple[str, str], ...]
    selected_sources: tuple[ExtractionSource, ...]
    all_sources: tuple[ExtractionSource, ...]
    prompts: tuple[str, ...]
    gold_file_sha256: str
    source_file_sha256: Mapping[str, str]


def prepare_atomic_run(
    *,
    repo_root: str | Path = ".",
    source_groups: Sequence[ExtractionSource] | None = None,
    validate_gold: bool = True,
) -> AtomicRunPlan:
    """Prepare ten source-only prompts, optionally validating gold after that work."""

    root = Path(repo_root).resolve()
    gold_path = root / ATOMIC_GOLD_PATH
    gold_hash = _file_sha256(gold_path)
    if gold_hash != FROZEN_ATOMIC_GOLD_SHA256:
        raise AtomicPipelineError("the atomic gold file does not match its frozen hash")

    all_sources = tuple(source_groups) if source_groups is not None else load_pilot_sources()
    sources_by_id = {source.source_id: source for source in all_sources}
    missing_sources = [
        source_id for _, source_id in ATOMIC_CASE_REFS if source_id not in sources_by_id
    ]
    if missing_sources:
        raise AtomicPipelineError(
            "missing atomic source IDs: " + ", ".join(missing_sources)
        )
    selected = tuple(sources_by_id[source_id] for _, source_id in ATOMIC_CASE_REFS)
    prompts = tuple(build_atomic_extraction_prompt(source) for source in selected)

    if validate_gold:
        gold_cases = load_atomic_gold(gold_path, all_sources)
        _require_frozen_case_order(gold_cases)

    return AtomicRunPlan(
        case_refs=ATOMIC_CASE_REFS,
        selected_sources=selected,
        all_sources=all_sources,
        prompts=prompts,
        gold_file_sha256=gold_hash,
        source_file_sha256=_source_hashes(root),
    )


def dry_run_atomic(
    *,
    repo_root: str | Path = ".",
    stdout: TextIO | None = None,
) -> AtomicRunPlan:
    """Validate inputs and print the ten calls without using a client or writing files."""

    stdout = stdout or sys.stdout
    plan = prepare_atomic_run(repo_root=repo_root, validate_gold=True)
    for case_id, source_id in plan.case_refs:
        print(f"{case_id}\t{source_id}", file=stdout)
    return plan


def execute_atomic_pipeline(
    *,
    client: object,
    requested_model: str,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    source_groups: Sequence[ExtractionSource] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Run ten sequential calls, then load gold, score, and write results."""

    if not isinstance(requested_model, str) or not requested_model.strip():
        raise AtomicPipelineError("execute requires a non-empty model")
    output_path = Path(output_dir)
    _require_empty_output_directory(output_path)
    plan = prepare_atomic_run(
        repo_root=repo_root,
        source_groups=source_groups,
        validate_gold=False,
    )
    clock = now or (lambda: datetime.now(timezone.utc))
    started_at = _utc_text(clock())
    completed: list[tuple[str, AtomicExtractionResult]] = []

    for (case_id, _), source in zip(plan.case_refs, plan.selected_sources):
        try:
            result = extract_atomic_claims(source, client)
        except AtomicExtractionValidationError:
            return _write_failed_run(
                output_path=output_path,
                plan=plan,
                requested_model=requested_model,
                completed=completed,
                case_id=case_id,
                source_id=source.source_id,
                failure_stage="validation",
                failure_message="The extracted claims failed validation.",
                started_at=started_at,
                completed_at=_utc_text(clock()),
            )
        except Exception:
            return _write_failed_run(
                output_path=output_path,
                plan=plan,
                requested_model=requested_model,
                completed=completed,
                case_id=case_id,
                source_id=source.source_id,
                failure_stage="provider",
                failure_message="The extraction request failed.",
                started_at=started_at,
                completed_at=_utc_text(clock()),
            )
        completed.append((case_id, result))

    gold_cases = load_atomic_gold(
        Path(repo_root).resolve() / ATOMIC_GOLD_PATH,
        plan.all_sources,
    )
    _require_frozen_case_order(gold_cases)
    predictions_by_case = {
        case_id: result.claims for case_id, result in completed
    }
    scores = score_atomic_extraction(gold_cases, predictions_by_case)
    predictions = [
        {
            "case_id": case_id,
            "source_id": result.source_id,
            "claims": [asdict(claim) for claim in result.claims],
        }
        for case_id, result in completed
    ]
    failures: list[dict[str, object]] = []
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "predictions.jsonl", predictions)
    _write_jsonl(output_path / "case_scores.jsonl", scores["case_results"])
    _write_json(output_path / "scores.json", scores)
    _write_jsonl(output_path / "failures.jsonl", failures)
    output_hashes = _output_hashes(
        output_path,
        ("predictions.jsonl", "case_scores.jsonl", "scores.json", "failures.jsonl"),
    )
    run = _run_record(
        status="completed",
        plan=plan,
        requested_model=requested_model,
        completed=completed,
        started_at=started_at,
        completed_at=_utc_text(clock()),
        failed_cases=0,
        output_hashes=output_hashes,
    )
    _write_json(output_path / "run.json", run)
    return run


def _write_failed_run(
    *,
    output_path: Path,
    plan: AtomicRunPlan,
    requested_model: str,
    completed: Sequence[tuple[str, AtomicExtractionResult]],
    case_id: str,
    source_id: str,
    failure_stage: str,
    failure_message: str,
    started_at: str,
    completed_at: str,
) -> dict[str, object]:
    predictions = [
        {
            "case_id": completed_case_id,
            "source_id": result.source_id,
            "claims": [asdict(claim) for claim in result.claims],
        }
        for completed_case_id, result in completed
    ]
    failures = [
        {
            "case_id": case_id,
            "source_id": source_id,
            "failure_stage": failure_stage,
            "error": failure_message,
        }
    ]
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "predictions.jsonl", predictions)
    _write_jsonl(output_path / "failures.jsonl", failures)
    output_hashes = _output_hashes(
        output_path, ("predictions.jsonl", "failures.jsonl")
    )
    run = _run_record(
        status="failed",
        plan=plan,
        requested_model=requested_model,
        completed=completed,
        started_at=started_at,
        completed_at=completed_at,
        failed_cases=1,
        output_hashes=output_hashes,
    )
    _write_json(output_path / "run.json", run)
    return run


def _run_record(
    *,
    status: str,
    plan: AtomicRunPlan,
    requested_model: str,
    completed: Sequence[tuple[str, AtomicExtractionResult]],
    started_at: str,
    completed_at: str,
    failed_cases: int,
    output_hashes: Mapping[str, str],
) -> dict[str, object]:
    returned_models = sorted(
        {result.response_metadata.returned_model for _, result in completed}
    )
    return {
        "run_status": status,
        "prompt_version": ATOMIC_EXTRACTION_PROMPT_VERSION,
        "requested_model": requested_model,
        "resolved_model": returned_models[0] if len(returned_models) == 1 else None,
        "temperature": TEMPERATURE,
        "gold_file_sha256": plan.gold_file_sha256,
        "source_file_sha256": dict(plan.source_file_sha256),
        "started_at": started_at,
        "completed_at": completed_at,
        "total_cases": len(plan.case_refs),
        "calls_attempted": len(completed) + failed_cases,
        "successful_cases": len(completed),
        "failed_cases": failed_cases,
        "output_file_sha256": dict(output_hashes),
    }


def _require_frozen_case_order(gold_cases: Sequence[AtomicGoldCase]) -> None:
    refs = tuple((case.case_id, case.source_id) for case in gold_cases)
    if refs != ATOMIC_CASE_REFS:
        raise AtomicPipelineError("atomic gold case IDs or source order changed")


def _require_empty_output_directory(output_path: Path) -> None:
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise AtomicPipelineError(f"refusing to overwrite non-empty output: {output_path}")


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        (PILOT_SOURCE_DIR / filename).as_posix(): _file_sha256(
            repo_root / PILOT_SOURCE_DIR / filename
        )
        for filename in _SOURCE_FILENAMES
    }


def _output_hashes(output_path: Path, filenames: Sequence[str]) -> dict[str, str]:
    return {filename: _file_sha256(output_path / filename) for filename in filenames}


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise AtomicPipelineError(f"could not hash {path}: {error}") from error


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AtomicPipelineError("run timestamps must include a UTC offset")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    assert_no_secrets(list(records), str(path))
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    _write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    assert_no_secrets(record, str(path))
    _write_text(
        path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ten-case atomic-extraction pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Validate the dry run or execute ten approved paid calls."""

    stdout = stdout or sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        dry_run_atomic(stdout=stdout)
        return 0
    if not isinstance(args.model, str) or not args.model.strip():
        parser.error("--execute requires --model")

    try:
        api_key = load_env_value(args.env_file, "OPENAI_API_KEY")
        client = OpenAIResponsesClient(
            api_key=api_key,
            model=args.model,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        run = execute_atomic_pipeline(
            client=client,
            requested_model=args.model,
            output_dir=DEFAULT_OUTPUT_DIR,
        )
    except Exception as error:
        message = str(error)
        try:
            assert_no_secrets(message, "error")
        except Exception:
            message = "The run failed; a sensitive-looking error was omitted."
        print(f"error={message}", file=sys.stderr)
        return 1

    print(f"run_status={run['run_status']}", file=stdout)
    print(f"calls_attempted={run['calls_attempted']}", file=stdout)
    print(f"successful_cases={run['successful_cases']}", file=stdout)
    print(f"failed_cases={run['failed_cases']}", file=stdout)
    print(f"output_dir={DEFAULT_OUTPUT_DIR}", file=stdout)
    return 0 if run["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
