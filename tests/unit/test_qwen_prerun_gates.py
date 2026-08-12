from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_prerun_gates import (
    QwenPreRunGateError,
    assert_qwen_pre_run_gates,
    build_pre_run_gate_report,
)


ROOT = Path(__file__).resolve().parents[2]


class QwenPreRunGateTests(unittest.TestCase):
    def test_current_repo_blocks_on_missing_qwen_gates_not_scaled_data(self) -> None:
        report = build_pre_run_gate_report(ROOT)

        self.assertEqual(report["schema_version"], "qwen_pre_run_gates_v1")
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["provider_request_count"], 0)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["scaled_validator"]["status"], "passed")
        self.assertEqual(checks["scaled_review_status"]["status"], "passed")
        self.assertEqual(checks["scaled_evidence_integrity"]["status"], "passed")
        self.assertEqual(checks["qwen_extraction_primary_gate"]["status"], "blocked")
        self.assertEqual(checks["qwen_extraction_holdout_gate"]["status"], "blocked")
        self.assertEqual(checks["qwen_context_evidence_audit"]["status"], "blocked")
        self.assertEqual(checks["qwen_materialization_clean"]["status"], "blocked")

    def test_assertion_raises_for_blockers(self) -> None:
        with self.assertRaisesRegex(QwenPreRunGateError, "qwen_extraction_primary_gate"):
            assert_qwen_pre_run_gates(ROOT)

    def test_synthetic_qwen_artifacts_can_pass_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            extraction_root = tmp / "extraction-gate"
            for gate_name in ("primary_gate", "holdout_gate"):
                _write_json(
                    extraction_root / gate_name / "run-manifest.json",
                    {
                        "schema_version": "qwen_extraction_gate_run_v1",
                        "gate_name": gate_name,
                        "extraction": {"successful_count": 10},
                        "scores": {"status": "passed"},
                    },
                )

            context_audit_root = tmp / "context-audit"
            _write_json(
                context_audit_root / "audit.json",
                {
                    "schema_version": "qwen_context_evidence_audit_v1",
                    "provider_request_count": 0,
                    "gold_opened_after_context_seal": True,
                },
            )
            _write_jsonl(
                context_audit_root / "metrics.jsonl",
                [{
                    "metric_group": "context_evidence",
                    "metric": "cross_user_citation_evidence",
                    "value": 0,
                }],
            )

            contexts_path = tmp / "contexts" / "contexts.jsonl"
            _write_jsonl(contexts_path, [])
            _write_json(
                contexts_path.parent / "manifest.json",
                {
                    "schema_version": "qwen_context_materialization_v2",
                    "status": "completed",
                    "b6_b7_context_identity": True,
                    "failure_count": 0,
                },
            )
            _write_jsonl(contexts_path.parent / "failures.jsonl", [])

            report = build_pre_run_gate_report(
                ROOT,
                extraction_gate_root=extraction_root,
                context_audit_root=context_audit_root,
                contexts_path=contexts_path,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["blocker_count"], 0)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
