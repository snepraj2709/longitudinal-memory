from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retrieval.index_evaluation as index_evaluation
from retrieval.contracts import (
    AtomicIndexInput,
    ClaimVersionLineage,
    SourceSpanLineage,
    TransactionTime,
    ValidTime,
    load_index_config,
)
from retrieval.embeddings import DeterministicTokenHashEmbedder
from retrieval.index_evaluation import (
    DATASET_MANIFEST,
    DATASET_VERSION,
    EXPECTED,
    IndexChecks,
    IndexFailure,
    RetrievalIndexEvaluationError,
    _validate_dataset_manifest,
    _write_release,
    load_index_runtime,
    serialize_records,
)
from retrieval.indexing import build_atomic_index_record


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
AS_OF = datetime(2026, 11, 14, 9, tzinfo=UTC)


def record():
    value = AtomicIndexInput(
        "user_001",
        "claim_001",
        "version_001",
        "user_001",
        "user_001",
        "work_preference",
        {"mode": "remote"},
        "positive",
        "asserted",
        None,
        "candidate",
        ValidTime("unknown"),
        TransactionTime(AS_OF),
        "standard",
        (ClaimVersionLineage("user_001", "claim_001", "version_001", "candidate", 0),),
        (SourceSpanLineage("user_001", "claim_001", "version_001", "source_001", "span_001", "supports", 0),),
    )
    result = build_atomic_index_record(
        "user_001",
        AS_OF,
        value,
        config=load_index_config(ROOT / "configs/retrieval/index_v1.json"),
        embedder=DeterministicTokenHashEmbedder(),
    )
    assert result is not None
    return result


def checks() -> IndexChecks:
    return IndexChecks(
        DATASET_VERSION,
        2,
        20,
        2,
        33,
        17,
        0,
        50,
        50,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        True,
        True,
        True,
        True,
        256,
        50,
        50,
        True,
        True,
    )


class RetrievalIndexEvaluationUnitTests(unittest.TestCase):
    def test_runtime_manifest_loads_exact_development_handoff(self) -> None:
        manifest, sources, definitions, claims, summaries = load_index_runtime(
            repo_root=ROOT
        )
        self.assertEqual(manifest["expected"], EXPECTED)
        self.assertEqual(len(sources), 20)
        self.assertEqual(len(definitions), 20)
        self.assertEqual(len(claims), 33)
        self.assertEqual(len(summaries), 17)
        self.assertEqual({item.user_id for item in sources}, {"user_001", "user_002"})

    def test_runtime_stops_before_mixed_file_test_records(self) -> None:
        limits = {
            (ROOT / "data/scaled-v1/runtime/users.jsonl").resolve(): 2,
            (ROOT / "data/scaled-v1/runtime/sources.jsonl").resolve(): 20,
        }
        original_open = Path.open

        class PrefixReadGuard:
            def __init__(self, handle, limit: int) -> None:
                self.handle = handle
                self.limit = limit
                self.lines = 0

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def readline(self, *args, **kwargs):
                if self.lines >= self.limit:
                    raise AssertionError("runtime read crossed the development prefix")
                self.lines += 1
                return self.handle.readline(*args, **kwargs)

            def read(self, *args, **kwargs):
                raise AssertionError("runtime attempted a full mixed-file read")

        def guarded_open(path: Path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            limit = limits.get(path.resolve())
            return PrefixReadGuard(handle, limit) if limit is not None else handle

        with patch.object(Path, "open", guarded_open):
            _, sources, _, _, _ = load_index_runtime(repo_root=ROOT)
        self.assertEqual(len(sources), 20)

    def test_runtime_imports_have_no_gold_scorer_or_review_dependency(self) -> None:
        path = ROOT / "src/retrieval/index_evaluation.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        joined = " ".join(imports)
        for forbidden in ("gold", "oracle", "review", "summary_quality"):
            self.assertNotIn(forbidden, joined)

    def test_runtime_predecessor_attestation_hashes_only_authority_manifests(self) -> None:
        expected = {
            index_evaluation.STEP63_MANIFEST_PATH: index_evaluation.STEP63_MANIFEST_SHA256,
            index_evaluation.STEP64_RESULT_MANIFEST_PATH: (
                index_evaluation.STEP64_RESULT_MANIFEST_SHA256
            ),
            index_evaluation.STEP64_CHECKPOINT_MANIFEST_PATH: (
                index_evaluation.STEP64_CHECKPOINT_MANIFEST_SHA256
            ),
            index_evaluation.STEP64_CHECKPOINT_PREFLIGHT_PATH: (
                index_evaluation.STEP64_CHECKPOINT_PREFLIGHT_SHA256
            ),
        }
        opened = []

        def fake_hash(path: Path) -> str:
            relative = path.resolve().relative_to(ROOT)
            opened.append(relative)
            return expected[relative]

        with patch.object(index_evaluation, "_file_sha256", fake_hash):
            value = index_evaluation._predecessor_attestation(ROOT)

        self.assertEqual(opened, list(expected))
        self.assertEqual(value["runtime_hashed_file_count"], 4)
        self.assertFalse(value["out_of_band_protection"]["runtime_verified"])
        serialized = json.dumps(value, sort_keys=True)
        for forbidden in ("/gold/", "scorer", "evaluator", "oracle", "review_queue"):
            self.assertNotIn(forbidden, serialized)

    def test_manifest_rejects_forbidden_dependency_and_identity_drift(self) -> None:
        value = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
        value["inputs"]["bad"] = {
            "gold_path": "data/scaled-v1/gold/claims.jsonl",
            "gold_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(RetrievalIndexEvaluationError, "forbidden"):
            _validate_dataset_manifest(value)
        value = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
        value["expected"]["record_count"] = 49
        with self.assertRaisesRegex(RetrievalIndexEvaluationError, "identity"):
            _validate_dataset_manifest(value)

    def test_record_serialization_is_canonical_and_byte_stable(self) -> None:
        first = serialize_records((record(),))
        second = serialize_records((record(),))
        self.assertEqual(first, second)
        value = json.loads(first)
        self.assertEqual(value["record_kind"], "atomic")
        self.assertEqual(len(value["embedding"]), 256)
        self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())

    def test_failure_fields_are_sanitized(self) -> None:
        IndexFailure("a" * 64, "user_001", "build_failed", "user")
        with self.assertRaisesRegex(RetrievalIndexEvaluationError, "sanitized"):
            IndexFailure("a" * 64, "user_001", "raw/value", "user")
        with self.assertRaisesRegex(RetrievalIndexEvaluationError, "user"):
            IndexFailure("a" * 64, "user_003", "build_failed", "user")

    def test_checks_forbid_model_usage_and_keep_zero_metrics_explicit(self) -> None:
        value = checks()
        self.assertEqual(value.durative_record_count, 0)
        self.assertEqual(value.failure_count, 0)
        with self.assertRaisesRegex(RetrievalIndexEvaluationError, "model use"):
            replace(value, model_calls=1)

    def test_release_writer_refuses_nonempty_or_second_write(self) -> None:
        manifest = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            _write_release(ROOT, output, manifest, (record(),), (), checks())
            with self.assertRaisesRegex(RetrievalIndexEvaluationError, "already exists"):
                _write_release(ROOT, output, manifest, (record(),), (), checks())


if __name__ == "__main__":
    unittest.main()
