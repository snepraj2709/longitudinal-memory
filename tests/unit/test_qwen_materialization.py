from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_materialization import (
    ContextPackage,
    MaterializationResult,
    QwenMaterializationError,
    _b7_contexts,
    _validate_runtime,
    write_materialization,
)


class QwenMaterializationUnitTests(unittest.TestCase):
    def test_runtime_rejects_cross_user_extraction_checkpoint(self) -> None:
        case = {"as_of": "2026-12-01T12:00:00+00:00", "case_id": "case_1", "user_id": "user_001"}
        runtime = {
            "users": [{"display_name": "A", "split": "development", "user_id": "user_001"}],
            "sources": [{"source_id": "source_1", "user_id": "user_001"}],
            "qa": [{**case, "question": "Question"}],
            "summary": [{**case, "case_id": "case_2", "instruction": "Summary"}],
            "interactive": [{**case, "case_id": "case_3", "initial_user_message": "Message"}],
        }
        extraction = [{"record_id": "source_1", "user_id": "user_002", "valid": False}]
        with self.assertRaisesRegex(QwenMaterializationError, "user boundary"):
            _validate_runtime("development", runtime, extraction)

    def test_b7_copies_b6_context_and_disables_provider_call(self) -> None:
        b6 = ContextPackage(
            "qwen35-27b-fp8-v2",
            "development",
            "qa",
            "case_1",
            "user_001",
            "2026-12-01T12:00:00+00:00",
            "B6",
            True,
            "persisted_conflict_resolution",
            ({"record_id": "memory_1"},),
            "a" * 64,
        )
        b7 = _b7_contexts((b6,))[0]
        self.assertEqual(b7.context_records, b6.context_records)
        self.assertEqual(b7.context_sha256, b6.context_sha256)
        self.assertFalse(b7.provider_call)

    def test_materialization_output_is_immutable(self) -> None:
        result = MaterializationResult("development", (), (), {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            write_materialization(result, output)
            with self.assertRaises(FileExistsError):
                write_materialization(result, output)
            self.assertEqual(asdict(result)["split"], "development")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("contexts_sha256", manifest)

    def test_materialization_manifest_accepts_replacement_series_id(self) -> None:
        result = MaterializationResult("development", (), (), {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            write_materialization(result, output, series_id="qwen3-8b-vllm-dev-v1")
            manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["series_id"], "qwen3-8b-vllm-dev-v1")


if __name__ == "__main__":
    unittest.main()
