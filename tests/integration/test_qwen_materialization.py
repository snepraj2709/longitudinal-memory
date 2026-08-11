from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from evaluation.qwen_materialization import materialize_qwen_contexts, write_materialization


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
AS_OF = "2026-12-01T12:00:00+00:00"


def _source(user_id: str, number: int, when: str, text: str) -> dict[str, object]:
    source_id = f"{user_id}_conversation_{number:03d}"
    message_id = f"{source_id}_message_001"
    return {
        "content": f"{user_id}: {text}",
        "created_at": when,
        "ingested_at": when,
        "messages": [{"message_id": message_id, "speaker_id": user_id, "text": text}],
        "metadata": {"thread_id": f"{source_id}_thread", "title": None},
        "participants": [user_id],
        "source_id": source_id,
        "source_type": "conversation",
        "user_id": user_id,
    }


def _claim(source: dict[str, object], value: str, start: str, end: str) -> dict[str, object]:
    message = source["messages"][0]
    return {
        "claim_id": f"model_claim_{value.casefold()}",
        "subject_id": source["user_id"],
        "speaker_id": source["user_id"],
        "predicate": "lives_in",
        "object": value,
        "polarity": "positive",
        "epistemic_status": "asserted",
        "valid_from": start,
        "valid_to": end,
        "confidence": 0.95,
        "evidence": [{
            "source_id": source["source_id"],
            "message_id": message["message_id"],
            "quote": message["text"],
        }],
    }


def _runtime() -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    delhi = _source("user_001", 1, "2026-01-01T09:00:00+00:00", "I live in Delhi.")
    mumbai = _source("user_001", 2, "2026-04-01T09:00:00+00:00", "I live in Mumbai now.")
    boston = _source("user_002", 1, "2026-02-01T09:00:00+00:00", "I live in Boston.")
    cases = {
        "qa": [{
            "as_of": AS_OF,
            "case_id": "qwen_dev_qa_001",
            "question": "Where does user_001 live now?",
            "split": "development",
            "task": "qa",
            "user_id": "user_001",
        }],
        "summary": [{
            "as_of": AS_OF,
            "case_id": "qwen_dev_summary_001",
            "instruction": "Summarize where user_001 lived this year.",
            "split": "development",
            "task": "summarization",
            "user_id": "user_001",
        }],
        "interactive": [{
            "allowed_turns": 3,
            "as_of": AS_OF,
            "case_id": "qwen_dev_interactive_001",
            "initial_user_message": "Where do I live now?",
            "scenario": "Use current memory.",
            "split": "development",
            "task": "interactive",
            "user_id": "user_001",
        }],
    }
    runtime = {
        "users": [
            {"display_name": "Asha", "split": "development", "user_id": "user_001"},
            {"display_name": "Mateo", "split": "development", "user_id": "user_002"},
        ],
        "sources": [delhi, mumbai, boston],
        **cases,
    }
    extraction = [
        {"record_id": delhi["source_id"], "user_id": "user_001", "valid": True, "claims": [_claim(delhi, "Delhi", "2026-01-01", "2026-03-31")]},
        {"record_id": mumbai["source_id"], "user_id": "user_001", "valid": True, "claims": [_claim(mumbai, "Mumbai", "2026-04-01", "2026-12-31")]},
        {"record_id": boston["source_id"], "user_id": "user_002", "valid": True, "claims": [_claim(boston, "Boston", "2026-02-01", "2026-12-31")]},
    ]
    return runtime, extraction


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for Qwen materialization tests",
)
class QwenMaterializationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        self.connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def tearDown(self) -> None:
        self.connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        self.connection.close()

    def _run(self):
        runtime, extraction = _runtime()
        return materialize_qwen_contexts(
            self.connection,
            repo_root=ROOT,
            split="development",
            runtime=runtime,
            extraction_rows=extraction,
        )

    def test_true_baseline_layers_and_b6_b7_identity(self) -> None:
        result = self._run()
        self.assertEqual(len(result.contexts), 24)
        self.assertFalse(result.failures)
        by_baseline = {
            baseline: [item for item in result.contexts if item.baseline_id == baseline]
            for baseline in (f"B{index}" for index in range(8))
        }
        self.assertTrue(all(not item.context_records for item in by_baseline["B0"]))
        self.assertTrue(all(len(item.context_records) == 2 for item in by_baseline["B1"]))
        self.assertTrue(all(record["evidence"] for item in by_baseline["B1"] for record in item.context_records))
        self.assertNotIn("user_002", json.dumps([asdict(item) for item in by_baseline["B1"]]))
        self.assertTrue(all({row["record_kind"] for row in item.context_records} <= {"atomic"} for item in by_baseline["B2"]))
        self.assertTrue(all({row["record_kind"] for row in item.context_records} <= {"session"} for item in by_baseline["B3"]))
        self.assertTrue(all({row["record_kind"] for row in item.context_records} <= {"atomic", "session"} for item in by_baseline["B4"]))
        self.assertTrue(all("candidate" not in row["lifecycle_statuses"] for item in by_baseline["B5"] for row in item.context_records))
        self.assertTrue(any(row["relations"] for item in by_baseline["B6"] for row in item.context_records))
        for b6, b7 in zip(by_baseline["B6"], by_baseline["B7"], strict=True):
            self.assertEqual(b6.context_sha256, b7.context_sha256)
            self.assertEqual(b6.context_records, b7.context_records)
            self.assertFalse(b7.provider_call)

    def test_materialization_and_export_are_byte_stable(self) -> None:
        first = self._run()
        with tempfile.TemporaryDirectory() as directory:
            first_output = Path(directory) / "first"
            write_materialization(first, first_output)
            first_bytes = {path.name: path.read_bytes() for path in first_output.iterdir()}
            first_manifest = json.loads((first_output / "manifest.json").read_text())
            self.assertEqual(
                first_manifest["contexts_sha256"],
                hashlib.sha256((first_output / "contexts.jsonl").read_bytes()).hexdigest(),
            )
            self.connection.execute("DROP SCHEMA public CASCADE")
            self.connection.execute("CREATE SCHEMA public")
            second = self._run()
            second_output = Path(directory) / "second"
            write_materialization(second, second_output)
            second_bytes = {path.name: path.read_bytes() for path in second_output.iterdir()}
            self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
