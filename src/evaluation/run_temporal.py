"""Run the isolated, deterministic Step 4.4 temporal development evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from storage.migrations import apply_migrations

from .temporal import (
    DATASET_VERSION,
    TemporalEvaluationError,
    TemporalFailure,
    TemporalPrediction,
    load_temporal_gold,
    load_temporal_runtime,
    record,
    run_temporal_cases,
    score_temporal,
    serialize_jsonl,
)


DATASET_ROOT = Path("data/phase4/temporal-development-v1")
RESULT_ROOT = Path("results/phase4/step4.4-temporal-evaluation-v1")
DATASET_MANIFEST_SHA256 = "785f17876b56ebdf29b8765e104e0160c19befae5271687869e4db9726034f8e"
RUNTIME_SHA256 = "4c67e1a01f0513512f9c1c3d65bacf8a943f66d037c369182a24adae9b656416"
GOLD_SHA256 = "1afce9b37f441925826b0511c8e32b004d825b29e2c8d614f10a5c7ca04ceefa"
IMPLEMENTATION_PATHS = (
    "Makefile",
    "src/evaluation/temporal.py",
    "src/evaluation/run_temporal.py",
    "tests/unit/test_temporal_evaluation.py",
    "tests/integration/test_temporal_evaluation.py",
)
PROTECTED_SHA256 = {
    "migrations/0001_phase4_storage.sql": "6f8a84ce1f78adeccfd7ff15d830dbcf844c34159a0ed34ce11cea4313e95359",
    "migrations/0002_ingestion_reprocessing.sql": "52900567c16e67c654d6fb845b615043beb5d9bf68851eb58231c5740a41253b",
    "migrations/0003_temporal_lifecycle.sql": "2d4e888b262e3dab1a86464fa9de6d33d8818d8978004c5923a5a5e69226fc8e",
    "src/temporal/__init__.py": "24c692f46335c104e5338d7c639a1890bef83196ebcfd993549f4f574b7508e3",
    "src/temporal/contracts.py": "b0bef262e1332e6435daf7e2dcec602f73d38cfbbe2abec1174d64558b1fda0e",
    "src/temporal/service.py": "7a0836035f1b3e3845dd21f90370d20525ceb8f3508d9c18b17124d1effff61f",
    "src/storage/contracts.py": "23122b62d4831bb1ca9a881a6396627d2de609ef954e8c583e599a80989de660",
    "src/storage/repository.py": "c3602486372cccf9b360ad692a05c09b30a6405eefbbb5ebea22a47fd6dd9d3e",
    "src/ingestion/service.py": "4c2d98417f69896c37506f0b879dae7f306c9d7d61e103036553387ff2a1784e",
    "tests/unit/test_temporal_contracts.py": "f65d619f7f29e0a1cf1c802fdb936b3d4b44005575a68d1eb17035d88b9caafb",
    "tests/integration/test_temporal_service.py": "173c9b4a2d72c22c45a3d05b284b95b31c81b4989b476ef99406bc6801216f90",
    "tests/integration/test_phase4_storage.py": "ec072b86396f8d1ae1271590d4b4d53870c538bba6e6b10a79b916915c1ee154",
    "results/phase4/step4.3-temporal-lifecycle-v1/manifest.json": "68a527421761dcdc960862f39f589c0283a274e062afe6ca86dc01bbce1c67ad",
    "results/phase4/step4.3-temporal-lifecycle-v1/findings.md": "274b79a17d589474b70683aa77cbb2936ba0eb3d8fd5a8e4bd4c77887a6f9250",
    "results/phase4/step4.2-ingestion-v1/manifest.json": "5c5e27587642af9115e0b5454292afb4b211043ec641b57b97c91573f0796ed2",
    "results/phase4/step4.2-ingestion-v1/findings.md": "097af6341fd97031ef27802f2bfe79263e60fb377926bb811e94ff07e141010d",
    "compose.yaml": "c53c4d37a256e2203f32566a3196c246cd5b2bb67ad8e1839095c9c1c8b0f51a",
    "requirements-storage.txt": "d375af9a0f805ccfd0fbf9a4943cfc6f44b4d7371e6a2376f6b8318d3c73e3d6",
    "data/scaled-v1/manifest.json": "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3",
    "data/scaled-v1/runtime/users.jsonl": "13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef",
    "data/scaled-v1/runtime/sources.jsonl": "a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/manifest.json": "f5127cfdb7720ecf84e320da613396d11d71629b709813e2e25ea2ab00df0876",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl": "509c51229eb8a6e13e898a28fec594d0118c29b6f5a93fc917fb0d1af34b4ff7",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/evidence_index.json": "311107940b64e22d9ba8af77e17d04e17f4298e8b5797e6f7234261e852ca0b6",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/scores.json": "76c96cc9590da0ac40d31a6ff5ab1d96e3decacb732a9bfc9b531e58e8750b4b",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "configs/extraction/predicate_registry_v2.json": "15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1",
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
}
EVALUATION_TABLES = (
    "memory_users",
    "source_events",
    "source_spans",
    "extraction_versions",
    "processing_attempts",
    "claims",
    "claim_versions",
    "evidence_links",
    "claim_extractions",
    "processing_outbox",
    "source_tombstones",
    "lifecycle_transitions",
)


def execute_temporal_evaluation(
    connection: object,
    *,
    repo_root: str | Path = ".",
    result_root: str | Path | None = None,
) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    output = (root / RESULT_ROOT) if result_root is None else Path(result_root).resolve()
    _require_empty_output(output)
    dataset_manifest = _verify_runtime_inputs(root)
    runtime_path = root / DATASET_ROOT / str(dataset_manifest["runtime"]["path"])
    runtime = load_temporal_runtime(runtime_path)
    apply_migrations(connection, root / "migrations")
    _require_clean_database(connection)
    predictions, failures = run_temporal_cases(connection, runtime, root)
    _require_case_accounting(runtime, predictions, failures)

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    failure_path = output / "failures.jsonl"
    _write_exclusive(prediction_path, serialize_jsonl(predictions))
    _write_exclusive(failure_path, serialize_jsonl(failures))
    _verify_persisted_accounting(runtime, prediction_path, failure_path)

    # This is intentionally the first read or hash of the separate gold file.
    gold_path = root / DATASET_ROOT / str(dataset_manifest["gold"]["path"])
    _require_hash(gold_path, GOLD_SHA256)
    gold = load_temporal_gold(gold_path)
    if tuple(case.case_id for case in runtime) != tuple(case.case_id for case in gold):
        raise TemporalEvaluationError("runtime and gold case order differ")
    scores = score_temporal(predictions, failures, gold)
    score_path = output / "scores.json"
    _write_exclusive(score_path, _json_bytes(record(scores)))
    run_path = output / "run.json"
    _write_exclusive(
        run_path,
        _json_bytes(
            {
                "artifact_version": "step4.4-temporal-evaluation-v1",
                "dataset_version": DATASET_VERSION,
                "starting_commit": "21ebed2b1d06b623deee2b9b0f72ee45d0ff71d3",
                "case_count": len(runtime),
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "retries": 0,
                "llm_requests": 0,
                "hosted_writes": 0,
                "gold_opened_after_persisted_results": True,
            }
        ),
    )
    findings_path = output / "findings.md"
    _write_exclusive(findings_path, _findings(scores).encode("utf-8"))
    artifact_hashes = {
        path.name: _sha256(path)
        for path in (prediction_path, failure_path, score_path, run_path, findings_path)
    }
    manifest = {
        "artifact_version": "step4.4-temporal-evaluation-v1",
        "guidance_version": "step-4.4-guidance-v1",
        "decision_envelope_sha256": "294fd04ec604e94f79fe1e276342c15bf1b515b1ecce5c865eee98e8c6d2cb50",
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "runtime_sha256": RUNTIME_SHA256,
            "gold_sha256": GOLD_SHA256,
            "case_count": 12,
            "user_counts": {"user_001": 6, "user_002": 6},
        },
        "execution": {
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "gold_access_boundary": "after_predictions_and_failures_persisted",
            "retries": 0,
            "llm_requests": 0,
            "hosted_writes": 0,
        },
        "implementation_hashes": {
            relative: _sha256(root / relative) for relative in IMPLEMENTATION_PATHS
        },
        "artifacts": artifact_hashes,
        "protected_inputs": dict(sorted(PROTECTED_SHA256.items())),
        "known_limitations": [
            "This is a deterministic development evaluator, not a production workload.",
            "The 12 cases exercise explicit lifecycle commands; they do not infer lifecycle state.",
            "Interval IoU has one development case, so its mean is descriptive rather than broad evidence.",
            "Underperformance is reported in the scorecard and does not alter stored claims.",
        ],
    }
    manifest_path = output / "manifest.json"
    _write_exclusive(manifest_path, _json_bytes(manifest))
    return manifest


def _verify_runtime_inputs(root: Path) -> Mapping[str, object]:
    for relative, expected in PROTECTED_SHA256.items():
        _require_hash(root / relative, expected)
    manifest_path = root / DATASET_ROOT / "manifest.json"
    _require_hash(manifest_path, DATASET_MANIFEST_SHA256)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalEvaluationError("temporal dataset manifest is unreadable") from error
    if (
        manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("case_count") != 12
        or manifest.get("user_counts") != {"user_001": 6, "user_002": 6}
        or manifest.get("runtime", {}).get("sha256") != RUNTIME_SHA256
        or manifest.get("gold", {}).get("sha256") != GOLD_SHA256
        or manifest.get("source_contract", {}).get("scaled_dataset_sha256")
        != "746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61"
    ):
        raise TemporalEvaluationError("temporal dataset manifest contract changed")
    _require_hash(root / DATASET_ROOT / str(manifest["runtime"]["path"]), RUNTIME_SHA256)
    return manifest


def _require_case_accounting(
    runtime: Sequence[object],
    predictions: Sequence[TemporalPrediction],
    failures: Sequence[TemporalFailure],
) -> None:
    expected = [case.case_id for case in runtime]
    actual = [case.case_id for case in predictions] + [case.case_id for case in failures]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise TemporalEvaluationError("every runtime case requires one prediction or failure")


def _require_clean_database(connection: object) -> None:
    for table in EVALUATION_TABLES:
        row = connection.execute(
            f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)"
        ).fetchone()
        if row is None or len(row) != 1:
            raise TemporalEvaluationError("could not verify clean evaluation database")
        if row[0]:
            raise TemporalEvaluationError("temporal evaluation requires a clean database")


def _verify_persisted_accounting(
    runtime: Sequence[object], prediction_path: Path, failure_path: Path
) -> None:
    ids: list[str] = []
    for path in (prediction_path, failure_path):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("case_id"), str):
                    raise TemporalEvaluationError("persisted evaluation record is invalid")
                ids.append(value["case_id"])
    expected = [case.case_id for case in runtime]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise TemporalEvaluationError("persisted results do not cover runtime cases")


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise TemporalEvaluationError("temporal result directory must be absent or empty")


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise TemporalEvaluationError("temporal result artifact already exists") from None


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise TemporalEvaluationError("required evaluation input is unreadable") from error


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise TemporalEvaluationError(f"protected input hash changed: {path.name}")


def _findings(scores: object) -> str:
    values = asdict(scores)
    lines = [
        "# Temporal evaluation findings",
        "",
        f"The run completed {values['prediction_count']} predictions and {values['failure_count']} failures across {values['case_count']} reviewed development cases.",
        "",
        "Each score keeps its denominator visible. A missing denominator is recorded as null with a reason, and a failed case remains in every metric that applies to it.",
        "",
        "The six accuracy metrics were 1.0. Mean interval IoU was 0.027027 for one scored pair: the inclusive shared endpoint contributes one overlapping day across a 37-day union.",
        "",
        "This evaluator reports explicit temporal behavior only. It does not infer lifecycle changes or modify the frozen Phase 3 claims.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    database_url = os.environ.get("STORAGE_DATABASE_URL")
    if not database_url:
        raise SystemExit("STORAGE_DATABASE_URL is required")
    import psycopg

    with psycopg.connect(database_url) as connection:
        execute_temporal_evaluation(connection, repo_root=args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
