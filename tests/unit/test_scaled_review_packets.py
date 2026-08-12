from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.scaled_review_packets import build_review_packets, write_review_packets


ROOT = Path(__file__).resolve().parents[2]


class ScaledReviewPacketTests(unittest.TestCase):
    def test_builds_all_scaled_review_packet_counts(self) -> None:
        packets = build_review_packets(ROOT)

        self.assertEqual(len(packets["gold_claims"]), 160)
        self.assertEqual(len(packets["gold_qa"]), 500)
        self.assertEqual(len(packets["gold_summaries"]), 50)
        self.assertEqual(len(packets["gold_interactive"]), 20)
        self.assertEqual(len(packets["oracle_events"]), 120)
        self.assertEqual(len(packets["review_evidence"]), 730)
        self.assertEqual(sum(len(rows) for rows in packets.values()), 1780)

    def test_gold_claim_packet_hydrates_exact_source_quote(self) -> None:
        packet = build_review_packets(ROOT)["gold_claims"][0]

        self.assertEqual(packet["target_id"], "scaled_user_001_claim_001")
        self.assertEqual(packet["current_status"], "approved")
        self.assertNotIn("pending_review", packet["risk_tags"])
        self.assertEqual(packet["expected"]["predicate"], "accepted_role")
        self.assertEqual(packet["expected"]["review_status"], "approved")
        evidence = packet["source_evidence"][0]
        self.assertEqual(evidence["source_id"], "scaled_user_001_conversation_001")
        self.assertEqual(evidence["speaker_id"], "user_001")
        self.assertTrue(evidence["exact_quote_match"])
        self.assertIn("product engineer role", evidence["source_text"])

    def test_review_queue_packet_includes_target_without_auto_approval(self) -> None:
        packet = build_review_packets(ROOT)["review_evidence"][0]

        self.assertEqual(packet["packet_type"], "review_queue_evidence")
        self.assertEqual(packet["current_status"], "approved")
        self.assertEqual(packet["review_action"], "resolve_review_queue_item_then_patch_source_row")
        self.assertIn("requires_patch", packet["allowed_statuses_after_review"])
        self.assertEqual(packet["target_records"][0]["claim_id"], "scaled_user_001_claim_001")
        self.assertEqual(packet["source_evidence"][0]["source_id"], "scaled_user_001_conversation_001")

    def test_write_packets_outputs_jsonl_files_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = write_review_packets(ROOT, directory)
            output_root = Path(directory)

            self.assertEqual(index["total_packets"], 1780)
            self.assertEqual(index["status"], "approved")
            self.assertEqual(index["approved_by"], "Sneha")
            self.assertTrue((output_root / "index.json").is_file())
            claims_path = output_root / "gold_claims.jsonl"
            self.assertTrue(claims_path.is_file())
            first = json.loads(claims_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["schema_version"], "scaled_review_packet_v1")
            self.assertEqual(first["current_status"], "approved")


if __name__ == "__main__":
    unittest.main()
