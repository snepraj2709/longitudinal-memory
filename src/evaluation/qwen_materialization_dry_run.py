"""Run a zero-provider Qwen materialization dry run from sealed extraction rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .qwen_benchmark import select_runtime
from .qwen_execution import extraction_rows_from_records
from .qwen_materialization import materialize_qwen_contexts


CONFIG_PATH = Path("configs/evaluation/qwen3_8b_vllm_development_v1.json")
EXTRACTION_RESPONSES = Path("results/evaluation/qwen3-8b-vllm-dev-v1/extraction/responses.jsonl")
OUTPUT_DIR = Path("results/evaluation/qwen3-8b-vllm-dev-v1/materialization-dry-run")
SERIES_ID = "qwen3-8b-vllm-dev-v1"


class QwenMaterializationDryRunError(RuntimeError):
    """Raised when the dry run cannot be produced or verified safely."""


def run_dry_run(
    *,
    repo_root: Path,
    database_url: str,
    output_dir: Path = OUTPUT_DIR,
    extraction_responses: Path = EXTRACTION_RESPONSES,
    config_path: Path = CONFIG_PATH,
    allow_reset_db: bool = False,
) -> Mapping[str, object]:
    if not allow_reset_db:
        raise QwenMaterializationDryRunError("pass --allow-reset-db for the isolated materialization database")
    root = repo_root.resolve()
    responses_path = extraction_responses if extraction_responses.is_absolute() else root / extraction_responses
    config_file = config_path if config_path.is_absolute() else root / config_path
    output = output_dir if output_dir.is_absolute() else root / output_dir

    try:
        import psycopg
    except ModuleNotFoundError as error:
        raise QwenMaterializationDryRunError("psycopg is required for materialization dry run") from error

    responses = _jsonl(responses_path)
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        connection.execute("CREATE SCHEMA public")
        try:
            result = materialize_qwen_contexts(
                connection,
                repo_root=root,
                split="development",
                runtime=select_runtime(root, "development"),
                extraction_rows=extraction_rows_from_records(responses),
                series_id=SERIES_ID,
                config_path=config_path,
            )
        finally:
            connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
            connection.execute("CREATE SCHEMA public")

    b6 = [item for item in result.contexts if item.baseline_id == "B6"]
    b7 = [item for item in result.contexts if item.baseline_id == "B7"]
    manifest = {
        "schema_version": "qwen_materialization_dry_run_v1",
        "series_id": SERIES_ID,
        "split": "development",
        "status": "passed" if not result.failures else "blocked",
        "provider_request_count": 0,
        "database_reset": True,
        "context_count": len(result.contexts),
        "failure_count": len(result.failures),
        "failure_codes": sorted({item.code for item in result.failures}),
        "failure_stages": sorted({item.stage for item in result.failures}),
        "b6_b7_context_identity": len(b6) == len(b7) and all(
            left.context_sha256 == right.context_sha256
            and left.context_records == right.context_records
            for left, right in zip(b6, b7, strict=True)
        ),
        "extraction_responses_sha256": _file_sha(responses_path),
        "config_sha256": _file_sha(config_file),
    }
    _write_or_verify(output / "manifest.json", manifest)
    return manifest


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_or_verify(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise QwenMaterializationDryRunError(f"existing dry-run artifact differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--database-url", default=os.environ.get("STORAGE_DATABASE_URL"))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--extraction-responses", type=Path, default=EXTRACTION_RESPONSES)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--allow-reset-db", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or STORAGE_DATABASE_URL is required")
    result = run_dry_run(
        repo_root=args.repo_root,
        database_url=args.database_url,
        output_dir=args.output,
        extraction_responses=args.extraction_responses,
        config_path=args.config,
        allow_reset_db=args.allow_reset_db,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
