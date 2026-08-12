from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.data_foundation_readiness import (
    REPORT_JSON,
    REPORT_MD,
    build_readiness_audit,
    write_audit_outputs,
)


ROOT = Path(__file__).resolve().parents[2]


class DataFoundationReadinessTests(unittest.TestCase):
    def test_audit_flags_scaled_false_review_approval(self) -> None:
        audit = build_readiness_audit(ROOT)

        self.assertEqual(audit["schema_version"], "data_foundation_readiness_audit_v1")
        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["overall_score"], 5.5)
        scaled = audit["scaled_v1"]
        self.assertFalse(scaled["validator_passed"])
        self.assertTrue(scaled["validator_errors"])
        self.assertTrue(scaled["manifest_row_review_mismatch"])
        self.assertEqual(scaled["pending_gold_rows"], 730)
        self.assertEqual(scaled["pending_review_queue_rows"], 930)
        self.assertEqual(
            scaled["row_level_gold_status_counts"]["qa"],
            {"pending_human_review": 500},
        )
        flag_ids = {flag["id"] for flag in audit["flags"]}
        self.assertIn("scaled_manifest_false_review_approval", flag_ids)

    def test_audit_reports_evidence_reference_and_qwen_risk_counts(self) -> None:
        audit = build_readiness_audit(ROOT)

        scaled = audit["scaled_v1"]
        evidence = scaled["evidence_integrity"]["totals"]
        references = scaled["reference_integrity"]["totals"]
        self.assertEqual(evidence["evidence_items"], 996)
        self.assertEqual(evidence["missing_source_message_refs"], 0)
        self.assertEqual(evidence["bad_exact_quotes"], 0)
        self.assertEqual(evidence["cross_user_evidence"], 0)
        self.assertEqual(evidence["empty_or_malformed"], 0)
        self.assertEqual(references["claim_refs"], 800)
        self.assertEqual(references["missing_claim_refs"], 0)
        self.assertEqual(references["cross_user_claim_refs"], 0)
        self.assertEqual(references["event_refs"], 530)
        self.assertEqual(references["missing_event_refs"], 0)
        self.assertEqual(references["cross_user_event_refs"], 0)
        qwen = audit["qwen_downstream_risk"]
        self.assertEqual(qwen["context_materialization_failures"], 13)
        self.assertEqual(qwen["risk_level"], "high")

    def test_audit_writes_compact_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = write_audit_outputs(ROOT, directory)
            json_path = Path(directory) / REPORT_JSON
            markdown_path = Path(directory) / REPORT_MD

            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            written = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], audit["schema_version"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# Data foundation readiness", markdown)
            self.assertIn("Status: **blocked**", markdown)
            self.assertIn("scaled_manifest_false_review_approval", markdown)
            self.assertIn("data/scaled-v1/review_queues/evidence.jsonl", markdown)


if __name__ == "__main__":
    unittest.main()
