from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.qwen_compatibility import build_requests


ROOT = Path(__file__).resolve().parents[2]


class QwenCompatibilityTests(unittest.TestCase):
    def test_pack_has_three_requests_for_each_task_and_no_context_reuse(self) -> None:
        rows = build_requests(ROOT)
        self.assertEqual(len(rows), 12)
        self.assertEqual({task: sum(row["task"] == task for row in rows) for task in {
            "extraction", "qa", "summary", "interactive",
        }}, {"extraction": 3, "qa": 3, "summary": 3, "interactive": 3})
        for row in rows[3:]:
            self.assertEqual(row["baseline_id"], "B0")
            self.assertIn('"context_records":[]', row["messages"][1]["content"])

    def test_every_request_uses_strict_schema_and_non_thinking_sampling(self) -> None:
        for row in build_requests(ROOT):
            response_format = row["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            self.assertEqual(row["sampling"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
