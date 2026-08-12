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
    _citation_evidence_for_record,
    _citation_evidence_limit,
    _context_record_limit,
    _source_citation_index,
    _source_context_record,
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

    def test_context_record_limit_is_loaded_from_qwen3_config(self) -> None:
        self.assertEqual(
            _context_record_limit(
                Path.cwd(),
                Path("configs/evaluation/qwen3_8b_vllm_development_v1.json"),
            ),
            4,
        )

    def test_citation_evidence_limit_is_loaded_from_qwen3_config(self) -> None:
        self.assertEqual(
            _citation_evidence_limit(
                Path.cwd(),
                Path("configs/evaluation/qwen3_8b_vllm_development_v1.json"),
            ),
            12,
        )

    def test_source_context_record_carries_citation_evidence(self) -> None:
        source = {
            "source_id": "source_1",
            "user_id": "user_001",
            "source_type": "chat",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content": "speaker: Known fact.",
            "messages": [{
                "message_id": "message_1",
                "speaker_id": "speaker",
                "text": "Known fact.",
            }],
        }

        record = _source_context_record(source)

        self.assertEqual(record["citation_evidence"][0]["quote"], "Known fact.")
        self.assertEqual(record["citation_evidence"][0]["user_id"], "user_001")
        self.assertEqual(record["citation_evidence"][0]["support_types"], ["source_history"])

    def test_citation_evidence_hydrates_full_source_message_and_dedupes(self) -> None:
        source = {
            "source_id": "source_1",
            "user_id": "user_001",
            "source_type": "email",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content": "sender: Full sentence with two facts.",
            "messages": [{
                "message_id": "message_1",
                "speaker_id": "sender",
                "text": "Full sentence with two facts.",
            }],
        }
        source_citations = _source_citation_index((source,))
        rows = (
            ("claim_1", "version_1", "source_1", "message_1", "sender", "Full sentence", "supports"),
            ("claim_2", "version_2", "source_1", "message_1", "sender", "two facts", "supports"),
        )

        evidence = _citation_evidence_for_record(
            "user_001",
            3,
            rows,
            source_citations,
            None,
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["quote"], "Full sentence with two facts.")
        self.assertEqual(evidence[0]["linked_claim_ids"], ["claim_1", "claim_2"])
        self.assertEqual(evidence[0]["originating_record_rank"], 3)

    def test_citation_evidence_rejects_cross_user_source(self) -> None:
        source = {
            "source_id": "source_1",
            "user_id": "user_002",
            "source_type": "chat",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content": "speaker: Known fact.",
            "messages": [{
                "message_id": "message_1",
                "speaker_id": "speaker",
                "text": "Known fact.",
            }],
        }

        with self.assertRaisesRegex(QwenMaterializationError, "user boundary"):
            _citation_evidence_for_record(
                "user_001",
                1,
                (("claim_1", "version_1", "source_1", "message_1", "speaker", "Known fact.", "supports"),),
                _source_citation_index((source,)),
                None,
            )


if __name__ == "__main__":
    unittest.main()
