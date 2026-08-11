from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_series import (
    BudgetGate,
    BudgetStop,
    append_checkpoint,
    stage_three_is_allowed,
    write_immutable_json,
)


class QwenSeriesTests(unittest.TestCase):
    def test_budget_gate_stops_before_stage_cap(self) -> None:
        gate = BudgetGate(1, 200.0, 1500.0, 195.0, 195.0)
        gate.assert_can_spend(5.0)
        with self.assertRaisesRegex(BudgetStop, "stage 1"):
            gate.assert_can_spend(5.01)

    def test_budget_gate_stops_before_cumulative_cap(self) -> None:
        gate = BudgetGate(3, 1000.0, 1500.0, 900.0, 1490.0)
        with self.assertRaisesRegex(BudgetStop, "cumulative"):
            gate.assert_can_spend(10.01)

    def test_stage_three_requires_quality_and_reserved_budget(self) -> None:
        self.assertTrue(stage_three_is_allowed(
            valid_responses=886, total_responses=932,
            projected_run_cost_inr=800.0, remaining_budget_inr=1000.0,
        ))
        self.assertFalse(stage_three_is_allowed(
            valid_responses=885, total_responses=932,
            projected_run_cost_inr=800.0, remaining_budget_inr=1000.0,
        ))
        self.assertFalse(stage_three_is_allowed(
            valid_responses=932, total_responses=932,
            projected_run_cost_inr=840.0, remaining_budget_inr=1000.0,
        ))

    def test_checkpoint_appends_each_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            append_checkpoint(path, {"request_id": "2", "valid": False})
            append_checkpoint(path, {"request_id": "1", "valid": True})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["request_id"] for row in rows], ["2", "1"])

    def test_immutable_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_immutable_json(path, {"series_id": "qwen"})
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                write_immutable_json(path, {"series_id": "changed"})


if __name__ == "__main__":
    unittest.main()
