from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_pipeline import (
    QwenPipelineError,
    _historical_gpu_cost,
    _manager_cost,
    _next_attempt_id,
    build_compatibility_jobs,
)


ROOT = Path(__file__).resolve().parents[2]


class QwenPipelineTests(unittest.TestCase):
    def test_compatibility_pack_has_exact_stage_one_composition(self) -> None:
        jobs = build_compatibility_jobs(ROOT)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(job.task == "extraction" for job in jobs), 3)
        self.assertEqual(sum(job.baseline_id == "B0" for job in jobs), 3)
        self.assertEqual(sum(job.baseline_id == "B6" for job in jobs), 3)
        self.assertEqual(sum(job.task == "judge" for job in jobs), 3)
        self.assertEqual([job.position for job in jobs], list(range(1, 13)))

    def test_compatibility_prompts_have_no_scorer_references_or_model_identity(self) -> None:
        jobs = build_compatibility_jobs(ROOT)
        payload = json.dumps([job.user_prompt for job in jobs]).casefold()
        self.assertNotIn("reference_answer", payload)
        self.assertNotIn("acceptable_answers", payload)
        self.assertNotIn("qwen35-27b-fp8-v2", payload)

    def test_server_command_matches_pinned_vllm_cli(self) -> None:
        script = (ROOT / "scripts/run_qwen_vllm.sh").read_text(encoding="utf-8")
        self.assertIn("/home/qwen-v2-env/bin/vllm serve", script)
        self.assertIn("/home/qwen35-27b-fp8-v2-model", script)
        self.assertIn("--max-model-len 16384", script)
        self.assertIn("--reasoning-parser qwen3", script)
        self.assertIn("--language-model-only", script)
        self.assertNotIn("--task generate", script)
        self.assertNotIn("--api-key", script)

    def test_attempt_ids_and_historical_cost_preserve_failed_spend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "lifecycle/attempt-001/lifecycle.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({
                "machine_id": 468754,
                "gpu_cost_inr": 3.049554,
                "observed_account_spend_inr": 3.17,
                "destruction_verified": True,
            }))
            (root / "attempts/attempt-001").mkdir(parents=True)
            self.assertEqual(_historical_gpu_cost(root), 3.17)
            self.assertEqual(_next_attempt_id(root), "attempt-002")

    def test_unverified_historical_cleanup_stops_next_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "lifecycle/attempt-001/lifecycle.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({
                "machine_id": 1, "gpu_cost_inr": 1.0,
                "destruction_verified": False,
            }))
            with self.assertRaisesRegex(QwenPipelineError, "cleanup is unverified"):
                _historical_gpu_cost(Path(directory))

    def test_manager_cost_prefers_observed_billing(self) -> None:
        class Manager:
            @staticmethod
            def receipt():
                return {"gpu_cost_inr": 3.96, "observed_account_spend_inr": 4.77}

        self.assertEqual(_manager_cost(Manager()), 4.77)


if __name__ == "__main__":
    unittest.main()
