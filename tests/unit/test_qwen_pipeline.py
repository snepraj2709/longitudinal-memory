from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_pipeline import (
    QwenPipelineError,
    PAID_EXECUTION_CONFIRMATION,
    _historical_gpu_cost,
    _manager_cost,
    _next_attempt_id,
    _require_paid_execution_confirmation,
    build_compatibility_jobs,
)
from evaluation.qwen_benchmark import _answer_system_prompt, _render_answer_prompt


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

    def test_compatibility_answer_prompts_include_abstention_and_allowed_citations(self) -> None:
        jobs = build_compatibility_jobs(ROOT)
        b0_qa = next(job for job in jobs if job.request_id == "compat:B0:qa")
        b6_qa = next(job for job in jobs if job.request_id == "compat:B6:qa")
        b6_summary = next(job for job in jobs if job.request_id == "compat:B6:summary")

        b0_prompt = json.loads(b0_qa.user_prompt)
        self.assertEqual(b0_prompt["context_records"], [])
        self.assertEqual(b0_prompt["output_contract"]["abstention_template"], {
            "status": "abstained",
            "answer": "Not enough reliable memory.",
            "confidence": 0,
            "statements": [],
            "citations": [],
            "unresolved_parts": [],
            "abstention_reason": "insufficient_evidence",
        })
        self.assertEqual(b0_prompt["output_contract"]["allowed_citations"], [])
        b0_schema = b0_qa.response_format["json_schema"]["schema"]["properties"]
        self.assertEqual(b0_schema["status"]["enum"], ["abstained"])
        self.assertEqual(b0_schema["citations"]["maxItems"], 0)

        b6_prompt = json.loads(b6_qa.user_prompt)
        self.assertGreater(len(b6_prompt["output_contract"]["allowed_citations"]), 0)
        self.assertIn("grounded_example", b6_prompt["output_contract"])
        self.assertIn("minimal sufficient citation set", b6_prompt["output_contract"]["rules"])
        self.assertIn("Do not abstain merely", b6_qa.system_prompt)
        citation = b6_prompt["output_contract"]["allowed_citations"][0]
        self.assertEqual(set(citation), {"source_id", "message_id", "quote"})
        b6_schema = b6_qa.response_format["json_schema"]["schema"]["properties"]
        self.assertEqual(b6_schema["status"]["enum"], ["answered", "abstained", "disputed", "partially_answered"])

        summary_prompt = json.loads(b6_summary.user_prompt)
        self.assertNotIn("grounded_example", summary_prompt["output_contract"])
        summary_schema = b6_summary.response_format["json_schema"]["schema"]["properties"]
        self.assertEqual(summary_schema["status"]["enum"], ["abstained"])
        self.assertEqual(summary_schema["unresolved_parts"]["maxItems"], 0)

    def test_qwen_answer_prompt_uses_sufficient_evidence_before_abstaining(self) -> None:
        case = {
            "case_id": "case_1",
            "user_id": "user_001",
            "as_of": "2026-12-01T12:00:00+00:00",
            "question": "What role did Asha accept?",
        }
        citation = {
            "source_id": "source_1",
            "message_id": "message_1",
            "quote": "I accepted the product engineer role at Riverstone Labs.",
        }
        prompt = json.loads(_render_answer_prompt("qa", case, ({
            "record_kind": "atomic",
            "citation_evidence": [citation],
        },)))

        self.assertEqual(prompt["allowed_citations"], [citation])
        policy = prompt["answer_policy"]
        self.assertIn("Missing perfect evidence", policy["sufficiency_standard"])
        self.assertIn("not a reason to abstain", policy["sufficiency_standard"])
        self.assertIn("Use answered when the core request is supported", policy["status_policy"])
        self.assertIn("Apply answer_policy before abstaining", prompt["output_contract"]["rules"])
        self.assertIn("do not refuse because evidence is not perfect", prompt["output_contract"]["rules"])
        self.assertNotIn("reference_answer", json.dumps(prompt).casefold())

    def test_qwen_system_prompt_separates_citation_validation_from_answerability(self) -> None:
        tuned = _answer_system_prompt("Base prompt.")

        self.assertIn("exact citations are a validation requirement", tuned)
        self.assertIn("not an answerability threshold", tuned)
        self.assertIn("Do not abstain merely", tuned)

    def test_server_command_matches_pinned_vllm_cli(self) -> None:
        script = (ROOT / "scripts/run_qwen_vllm.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL_ID=\"${MODEL_ID:-Qwen/Qwen3-8B}\"", script)
        self.assertIn("MODEL_ALIAS=\"${MODEL_ALIAS:-qwen3-8b-vllm}\"", script)
        self.assertIn("MAX_MODEL_LEN=\"${MAX_MODEL_LEN:-8192}\"", script)
        self.assertIn("VLLM_API_KEY=\"${VLLM_API_KEY:-${QWEN_VLLM_API_KEY:-}}\"", script)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=\"${VLLM_USE_V2_MODEL_RUNNER:-0}\"", script)
        self.assertIn("exec \"${VLLM_BIN}\" serve \"${MODEL_ID}\"", script)
        self.assertNotIn("--api-key", script)
        self.assertIn("--enforce-eager", script)
        self.assertIn("--reasoning-parser qwen3", script)
        self.assertIn("--language-model-only", script)
        self.assertNotIn("--task generate", script)
        self.assertNotIn("qwen35-27b-fp8-v2", script)

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

    def test_paid_execution_requires_exact_fresh_confirmation(self) -> None:
        with self.assertRaisesRegex(QwenPipelineError, "paid GPU execution is locked"):
            _require_paid_execution_confirmation("")
        _require_paid_execution_confirmation(PAID_EXECUTION_CONFIRMATION)


if __name__ == "__main__":
    unittest.main()
