from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.qwen_extraction_quality_gate import (
    QwenExtractionGateError,
    build_gate_jobs,
    dry_run,
    load_config,
    score_gate_output,
    score_gate_records,
)


ROOT = Path(__file__).resolve().parents[2]


class QwenExtractionQualityGateTests(unittest.TestCase):
    def test_config_pins_primary_and_holdout_gates(self) -> None:
        config = load_config(ROOT)

        self.assertEqual(config["series_id"], "qwen3-8b-vllm-extraction-gate-v1")
        self.assertEqual(config["model"]["model_alias"], "qwen3-8b-vllm")
        self.assertEqual(config["runtime"]["temperature"], 0)
        self.assertEqual(config["runtime"]["client_concurrency"], 1)
        self.assertEqual(len(config["primary_gate"]["source_ids"]), 10)
        self.assertEqual(len(config["holdout_gate"]["source_ids"]), 10)

    def test_jobs_select_user_001_sources_and_include_scaled_profile(self) -> None:
        config = load_config(ROOT)

        jobs = build_gate_jobs(ROOT, config=config)

        self.assertEqual(len(jobs), 10)
        self.assertEqual(jobs[0].record_id, "scaled_user_001_conversation_001")
        self.assertTrue(all(job.user_id == "user_001" for job in jobs))
        self.assertTrue(all(job.series_id == "qwen3-8b-vllm-extraction-gate-v1" for job in jobs))
        self.assertIn('"speaker_name":"Asha"', jobs[0].user_prompt)
        self.assertIn("Use accepted_role, not primary_role", jobs[0].system_prompt)
        self.assertIn("Use job_start_date, not employment_start_date", jobs[0].system_prompt)
        self.assertIn("Use project_review_date, not has_scheduled_event", jobs[0].system_prompt)

    def test_dry_run_builds_ten_requests_without_gold(self) -> None:
        with patch(
            "evaluation.qwen_extraction_quality_gate._load_gate_gold",
            side_effect=AssertionError("gold must not be opened during dry run"),
        ):
            result = dry_run(repo_root=ROOT)

        self.assertEqual(result["planned_request_count"], 10)
        self.assertFalse(result["gold_opened"])
        self.assertFalse(result["oracle_opened"])
        self.assertFalse(result["review_opened"])
        self.assertEqual(result["source_ids"][0], "scaled_user_001_conversation_001")

    def test_score_separates_proposition_match_from_metadata_errors(self) -> None:
        gate = {
            "user_id": "user_001",
            "source_ids": ["scaled_user_001_email_001"] * 10,
        }
        gold = [{
            "user_id": "user_001",
            "subject_id": "user_001",
            "speaker_id": "recruiting",
            "predicate": "job_start_date",
            "object": "2026-02-03",
            "polarity": "positive",
            "epistemic_status": "reported_by_other",
            "valid_from": "2026-02-03T00:00:00+00:00",
            "valid_to": None,
            "evidence": [{
                "source_id": "scaled_user_001_email_001",
                "message_id": "scaled_user_001_email_001_message_001",
                "quote": "Our onboarding sheet records Asha's start date as 2026-02-03.",
            }],
        }]
        records = [{
            "status": "succeeded",
            "record_id": "scaled_user_001_email_001",
            "user_id": "user_001",
            "output": {"claims": [{
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "job_start_date",
                "object": "2026-02-03",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": "2026-02-04T00:00:00+00:00",
                "valid_to": None,
                "evidence": [{
                    "source_id": "scaled_user_001_email_001",
                    "message_id": "scaled_user_001_email_001_message_001",
                    "quote": "Our onboarding sheet records Asha's start date as 2026-02-03.",
                }],
            }]},
        }]

        metrics = {
            row["metric"]: row
            for row in score_gate_records(
                records=records,
                gold_claims=gold,
                gate=gate,
                series_id="series",
                gate_name="primary_gate",
            )
        }

        self.assertEqual(metrics["proposition_precision"]["value"], 1)
        self.assertEqual(metrics["proposition_recall"]["value"], 1)
        self.assertEqual(metrics["strict_claim_precision"]["value"], 0)
        self.assertEqual(metrics["speaker_accuracy"]["value"], 0)
        self.assertEqual(metrics["epistemic_status_accuracy"]["value"], 0)
        self.assertEqual(metrics["valid_time_accuracy"]["value"], 0)
        self.assertEqual(metrics["exact_quote_precision"]["value"], 1)

    def test_score_tracks_wrong_person_and_bad_overlap_predicates(self) -> None:
        gate = {
            "user_id": "user_001",
            "source_ids": ["scaled_user_001_chat_001"] * 10,
        }
        records = [{
            "status": "succeeded",
            "record_id": "scaled_user_001_chat_001",
            "user_id": "user_001",
            "output": {"claims": [{
                "subject_id": "user_001",
                "speaker_id": "kabir",
                "predicate": "primary_role",
                "object": "friend",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": None,
                "valid_to": None,
                "evidence": [],
            }]},
        }]

        metrics = {
            row["metric"]: row["value"]
            for row in score_gate_records(
                records=records,
                gold_claims=[],
                gate=gate,
                series_id="series",
                gate_name="primary_gate",
            )
        }

        self.assertEqual(metrics["wrong_person_claim_count"], 1)
        self.assertEqual(metrics["known_bad_overlap_predicate_uses"], 1)

    def test_score_output_refuses_to_open_gold_before_sealed_manifest(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            extraction = Path(directory) / "extraction"
            extraction.mkdir()
            (extraction / "responses.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(QwenExtractionGateError, "not sealed"):
                score_gate_output(
                    repo_root=ROOT,
                    config=config,
                    extraction_dir=extraction,
                    output_dir=Path(directory) / "scores",
                    gate_name="primary_gate",
                )

    def test_score_output_opens_gold_only_after_seal(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            extraction = Path(directory) / "extraction"
            extraction.mkdir()
            manifest = {
                "series_id": "qwen3-8b-vllm-extraction-gate-v1",
                "gold_opened": False,
            }
            (extraction / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (extraction / "responses.jsonl").write_text("", encoding="utf-8")
            with patch(
                "evaluation.qwen_extraction_quality_gate._load_gate_gold",
                return_value=(),
            ) as load_gold:
                score_gate_output(
                    repo_root=ROOT,
                    config=config,
                    extraction_dir=extraction,
                    output_dir=Path(directory) / "scores",
                    gate_name="primary_gate",
                )

        load_gold.assert_called_once()


if __name__ == "__main__":
    unittest.main()
