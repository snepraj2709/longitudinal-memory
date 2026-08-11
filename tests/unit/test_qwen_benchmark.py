from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.qwen_benchmark import select_runtime


ROOT = Path(__file__).resolve().parents[2]


class QwenBenchmarkTests(unittest.TestCase):
    def test_development_slice_has_932_requests(self) -> None:
        selected = select_runtime(ROOT, "development")
        count = len(selected["sources"]) + 8 * sum(len(selected[task]) for task in ("qa", "summary", "interactive"))
        self.assertEqual(count, 932)
        self.assertEqual({row["user_id"] for row in selected["users"]}, {"user_001", "user_002"})

    def test_test_slice_has_3728_requests(self) -> None:
        selected = select_runtime(ROOT, "test")
        count = len(selected["sources"]) + 8 * sum(len(selected[task]) for task in ("qa", "summary", "interactive"))
        self.assertEqual(count, 3728)
        self.assertTrue(all(row["split"] == "test" for row in selected["users"]))


if __name__ == "__main__":
    unittest.main()
