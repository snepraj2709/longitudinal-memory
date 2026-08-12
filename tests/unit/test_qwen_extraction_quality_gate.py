from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extraction.scaled_source import _adapt_source

from evaluation.qwen_extraction_quality_gate import (
    QwenExtractionGateError,
    _canonicalize_scaled_v1_claims,
    _gate_config,
    _selected_sources,
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
        self.assertEqual(
            config["extraction_profile"]["profile_id"],
            "scaled-v1-qwen-extraction-profile-v2",
        )
        self.assertEqual(
            config["extraction_profile"]["canonicalization_version"],
            "scaled-v1-qwen-canonicalization-v1",
        )
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
        self.assertIn("full source message text", jobs[0].system_prompt)
        self.assertIn('object "remote"', jobs[0].system_prompt)

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

    def test_scaled_v1_canonicalizer_fixes_qwen_gate_failures(self) -> None:
        source = _source("scaled_user_001_conversation_001")

        claims = _canonicalize_scaled_v1_claims(source, [
            {
                "claim_id": "claim_001",
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "work_preference",
                "object": "remote work",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": "2026-01-08T09:00:00+00:00",
                "valid_to": None,
                "confidence": 1,
                "evidence": [{
                    "source_id": source.source_id,
                    "message_id": "scaled_user_001_conversation_001_message_001",
                    "quote": "I do my best focused work remotely.",
                }],
            },
            {
                "claim_id": "claim_002",
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "work_preference",
                "object": "deepen work in data products",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": "2026-01-08T09:00:00+00:00",
                "valid_to": None,
                "confidence": 1,
                "evidence": [{
                    "source_id": source.source_id,
                    "message_id": "scaled_user_001_conversation_001_message_001",
                    "quote": "For now, I want to deepen my work in data products.",
                }],
            },
        ])

        self.assertEqual(claims[0]["object"], "remote")
        self.assertEqual(claims[1]["predicate"], "career_goal")
        self.assertEqual(claims[1]["object"], "data products")
        self.assertEqual(
            claims[0]["evidence"][0]["quote"],
            "I accepted the product engineer role at Riverstone Labs. For now, I want to deepen my work in data products. I do my best focused work remotely.",
        )

    def test_scaled_v1_canonicalizer_removes_overlap_predicates(self) -> None:
        email = _source("scaled_user_001_email_001")
        calendar = _source("scaled_user_001_calendar_002")

        email_claim = _canonicalize_scaled_v1_claims(email, [{
            "claim_id": "claim_001",
            "subject_id": "user_001",
            "speaker_id": "recruiting",
            "predicate": "employment_start_date",
            "object": "2026-02-03",
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": "2026-02-03",
            "valid_to": None,
            "confidence": 1,
            "evidence": [{
                "source_id": email.source_id,
                "message_id": "scaled_user_001_email_001_message_001",
                "quote": "Our onboarding sheet records Asha's start date as 2026-02-03.",
            }],
        }])[0]
        calendar_claim = _canonicalize_scaled_v1_claims(calendar, [{
            "claim_id": "claim_001",
            "subject_id": "user_001",
            "speaker_id": "user_001",
            "predicate": "has_scheduled_event",
            "object": {"title": "Care Map review", "location": None},
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": "2026-08-11T00:00:00+00:00",
            "valid_to": None,
            "confidence": 1,
            "evidence": [{
                "source_id": calendar.source_id,
                "message_id": None,
                "quote": "Care Map review is scheduled for 2026-08-11.",
            }],
        }])[0]

        self.assertEqual(email_claim["predicate"], "job_start_date")
        self.assertEqual(email_claim["epistemic_status"], "reported_by_other")
        self.assertEqual(email_claim["valid_from"], "2026-02-03T00:00:00+00:00")
        self.assertEqual(calendar_claim["predicate"], "project_review_date")
        self.assertEqual(calendar_claim["object"], "2026-08-11")
        self.assertEqual(calendar_claim["valid_to"], "2026-08-11T23:59:59+00:00")

    def test_scaled_v1_canonicalizer_drops_unscored_restatements(self) -> None:
        repeated_goal = _source("scaled_user_001_conversation_003")
        speculative_move = _source("scaled_user_001_conversation_004")

        self.assertEqual(
            _canonicalize_scaled_v1_claims(repeated_goal, [{
                "claim_id": "claim_001",
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "career_goal",
                "object": "build reliable health tools",
                "polarity": "positive",
                "epistemic_status": "asserted",
                "valid_from": None,
                "valid_to": None,
                "confidence": 1,
                "evidence": [],
            }]),
            [],
        )
        self.assertEqual(
            _canonicalize_scaled_v1_claims(speculative_move, [{
                "claim_id": "claim_001",
                "subject_id": "user_001",
                "speaker_id": "user_001",
                "predicate": "has_relocation_plan",
                "object": "Mumbai",
                "polarity": "negative",
                "epistemic_status": "hypothetical",
                "valid_from": None,
                "valid_to": None,
                "confidence": 1,
                "evidence": [],
            }]),
            [],
        )

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


def _source(source_id: str):
    config = load_config(ROOT)
    gate = _gate_config(config, "primary_gate")
    source_record = {
        str(source["source_id"]): source
        for source in _selected_sources(ROOT, gate)
    }[source_id]
    return _adapt_source(
        source_record,
        ("user_001", source_id),
        {"user_001": "Asha"},
    )


if __name__ == "__main__":
    unittest.main()
