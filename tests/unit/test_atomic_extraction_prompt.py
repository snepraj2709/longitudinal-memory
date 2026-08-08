from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest

from evaluation.history import HistoryObservation
from extraction.contracts import (
    ALLOWED_EPISTEMIC_STATUSES,
    ALLOWED_POLARITIES,
    ALLOWED_PREDICATES,
)
from extraction.prompt import (
    ATOMIC_EXTRACTION_CANDIDATE_PROMPT_VERSION,
    ATOMIC_EXTRACTION_PROMPT_VERSION,
    ATOMIC_EXTRACTION_V4_PROMPT_VERSION,
    ATOMIC_EXTRACTION_V5_PROMPT_VERSION,
    ATOMIC_EXTRACTION_V6_PROMPT_VERSION,
    ATOMIC_EXTRACTION_V7_PROMPT_VERSION,
    ATOMIC_EXTRACTION_V8_PROMPT_VERSION,
    ATOMIC_EXTRACTION_SYSTEM_PROMPT,
    build_atomic_extraction_prompt,
    get_atomic_extraction_system_prompt,
)
from extraction.run_atomic import prepare_atomic_run
from extraction.schema import atomic_extraction_text_format
from evaluation.run_config import canonical_sha256
from extraction.source import ExtractionSource, KnownEntity


class AtomicExtractionPromptTests(unittest.TestCase):
    def observation(
        self,
        *,
        observed_at: str,
        source_type: str,
        source_id: str,
        message_id: str | None,
        author_id: str,
        author_name: str,
        text: str,
        **optional: object,
    ) -> HistoryObservation:
        return HistoryObservation(
            observed_at=datetime.fromisoformat(observed_at),
            source_type=source_type,
            source_id=source_id,
            message_id=message_id,
            author_id=author_id,
            author_name=author_name,
            text=text,
            **optional,
        )

    def source_json(self, prompt: str) -> dict[str, object]:
        return json.loads(prompt.split("\n", 1)[1])

    def source(self, source_id: str, source_type: str, *observations: HistoryObservation) -> ExtractionSource:
        entities = tuple(
            KnownEntity(author_id, author_name)
            for author_id, author_name in sorted(
                {(item.author_id, item.author_name) for item in observations}
            )
        )
        return ExtractionSource(source_id, source_type, observations, entities)

    def test_prompt_version_and_output_are_deterministic(self) -> None:
        observation = self.observation(
            observed_at="2026-04-05T19:30:38+05:30",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
            author_id="i_am_maya",
            author_name="Maya",
            text='She said "wait".\nThen we left.',
        )
        source = self.source("conv_001", "conversation", observation)

        first = build_atomic_extraction_prompt(source)
        second = build_atomic_extraction_prompt(source)

        self.assertEqual(ATOMIC_EXTRACTION_PROMPT_VERSION, "atomic-extraction-v3")
        self.assertEqual(first, second)
        self.assertEqual(
            self.source_json(first)["observations"][0]["text"],
            observation.text,
        )
        self.assertEqual(
            self.source_json(first)["known_entities"],
            [{"entity_id": "i_am_maya", "display_name": "Maya"}],
        )

    def test_preserves_source_and_observation_order(self) -> None:
        first = self.observation(
            observed_at="2026-04-05T19:30:38+05:30",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_002",
            author_id="person_001",
            author_name="Asha",
            text="First supplied observation.",
        )
        second = self.observation(
            observed_at="2026-04-05T19:20:00+05:30",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
            author_id="person_002",
            author_name="Maya",
            text="Second supplied observation.",
        )

        payload = self.source_json(
            build_atomic_extraction_prompt(
                self.source("conv_001", "conversation", first, second)
            )
        )

        self.assertEqual(payload["source_type"], "conversation")
        self.assertEqual(payload["source_id"], "conv_001")
        self.assertEqual(
            [item["message_id"] for item in payload["observations"]],
            ["msg_002", "msg_001"],
        )

    def test_includes_conversation_and_email_details(self) -> None:
        cases = (
            ("conversation", "conv_001", "speaker_001", "Maya", None),
            ("email", "email_001", "sender_001", "Kavya", "Role update"),
        )
        for source_type, source_id, speaker_id, speaker_name, subject in cases:
            with self.subTest(source_type=source_type):
                observation = self.observation(
                    observed_at="2026-04-12T16:20:00+05:30",
                    source_type=source_type,
                    source_id=source_id,
                    message_id=f"msg_{source_id}_001",
                    author_id=speaker_id,
                    author_name=speaker_name,
                    text="The exact source text.",
                    subject=subject,
                )

                payload = self.source_json(
                    build_atomic_extraction_prompt(
                        self.source(source_id, source_type, observation)
                    )
                )
                item = payload["observations"][0]

                self.assertEqual(item["observed_at"], observation.observed_at.isoformat())
                self.assertEqual(item["message_id"], observation.message_id)
                self.assertEqual(item["speaker_id"], speaker_id)
                self.assertEqual(item["speaker_name"], speaker_name)
                self.assertEqual(item["text"], observation.text)
                if subject is not None:
                    self.assertEqual(item["subject"], subject)

    def test_calendar_uses_null_message_id_and_keeps_event_times(self) -> None:
        observation = self.observation(
            observed_at="2026-04-01T09:00:00+05:30",
            source_type="calendar",
            source_id="cal_001",
            message_id=None,
            author_id="i_am_maya",
            author_name="Maya",
            text="Final semester examinations.",
            title="Final semester examinations",
            start_at=datetime.fromisoformat("2026-04-20T09:00:00+05:30"),
            end_at=datetime.fromisoformat("2026-04-29T12:00:00+05:30"),
            timezone="Asia/Kolkata",
            location="College examination hall",
        )

        payload = self.source_json(
            build_atomic_extraction_prompt(
                self.source("cal_001", "calendar", observation)
            )
        )
        item = payload["observations"][0]

        self.assertIsNone(item["message_id"])
        self.assertEqual(item["speaker_id"], "i_am_maya")
        self.assertEqual(item["start_at"], "2026-04-20T09:00:00+05:30")
        self.assertEqual(item["end_at"], "2026-04-29T12:00:00+05:30")

    def test_system_prompt_contains_complete_output_contract(self) -> None:
        required_fields = (
            "claim_id",
            "subject_id",
            "speaker_id",
            "predicate",
            "object",
            "polarity",
            "epistemic_status",
            "valid_from",
            "valid_to",
            "confidence",
            "evidence",
            "source_id",
            "message_id",
            "quote",
        )
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, ATOMIC_EXTRACTION_SYSTEM_PROMPT)
        for value in ALLOWED_POLARITIES | ALLOWED_EPISTEMIC_STATUSES:
            with self.subTest(value=value):
                self.assertIn(value, ATOMIC_EXTRACTION_SYSTEM_PROMPT)
        for predicate in ALLOWED_PREDICATES:
            with self.subTest(predicate=predicate):
                self.assertIn(predicate, ATOMIC_EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("exactly one JSON object", ATOMIC_EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("claims must be a list", ATOMIC_EXTRACTION_SYSTEM_PROMPT)
        self.assertIn(
            "message_id must be a non-empty string except for calendar evidence",
            ATOMIC_EXTRACTION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "must not repeat a source_id and message_id pair",
            ATOMIC_EXTRACTION_SYSTEM_PROMPT,
        )

    def test_system_prompt_requires_exact_quotes_and_matching_provenance(self) -> None:
        required_text = (
            "Copy each evidence quote exactly",
            "source_id and message_id must exactly match that observation",
            "Calendar evidence must use message_id: null",
            "speaker_id is who made the statement",
            "subject_id is who or what the claim describes",
            "subject_id must be an entity_id from known_entities",
            "Record only what the source supports",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, ATOMIC_EXTRACTION_SYSTEM_PROMPT)

    def test_v4_intervention_is_clause_level_and_keeps_fields_separate(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V4_PROMPT_VERSION
        )
        required_text = (
            "Scan every independent clause",
            "shortest contiguous quote",
            "do not flip a boolean object",
            "Use denied when a speaker explicitly rejects",
            "resolve it to ISO values in valid_from and valid_to",
            "same date for both boundaries of a point event",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, candidate_prompt)
                self.assertNotIn(text, ATOMIC_EXTRACTION_SYSTEM_PROMPT)

    def test_v4_prompt_snapshot_matches_full_and_smoke_suites(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v4_snapshot.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            snapshot["prompt_version"],
            ATOMIC_EXTRACTION_V4_PROMPT_VERSION,
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V4_PROMPT_VERSION
        )
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_v5_rechecks_source_bound_identifiers_and_quotes(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V5_PROMPT_VERSION
        )

        self.assertIn("speaker_id and subject_id appear in the supplied source", candidate_prompt)
        self.assertIn("every evidence message_id exists in that source", candidate_prompt)
        self.assertIn("every evidence quote is copied verbatim", candidate_prompt)

    def test_v5_prompt_and_schema_snapshot_matches_all_suites(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v5_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V5_PROMPT_VERSION
        )

        self.assertEqual(snapshot["prompt_version"], "atomic-extraction-v5")
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["text_schema_sha256"],
            canonical_sha256(atomic_extraction_text_format()),
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_v6_targets_v5_quality_regressions_without_changing_v5(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V6_PROMPT_VERSION
        )
        v5_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V5_PROMPT_VERSION
        )
        required_text = (
            "map directly to one active registry predicate",
            "not automatically the speaker, recipient, or user",
            "object true with negative polarity",
            "do not know if",
            "Use reported_by_other",
            "mark the rejected value denied and the replacement corrected",
            "a deadline sets valid_to",
            "short reply depends on a preceding question",
            "remove duplicate or partially overlapping claims",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, candidate_prompt)
                self.assertNotIn(text, v5_prompt)
        self.assertEqual(
            ATOMIC_EXTRACTION_V6_PROMPT_VERSION,
            "atomic-extraction-v6",
        )

    def test_v6_prompt_and_schema_snapshot_matches_smoke_and_full_suites(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v6_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V6_PROMPT_VERSION
        )

        self.assertEqual(snapshot["prompt_version"], "atomic-extraction-v6")
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["text_schema_sha256"],
            canonical_sha256(atomic_extraction_text_format()),
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_v7_uses_registry_semantics_and_keeps_nonasserted_claims(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V7_PROMPT_VERSION
        )
        required_text = (
            "including temporal_behavior",
            "use the scheduled predicates will_work_for and will_have_job_role",
            "Use job_start_date",
            "Use assigned_task",
            "offered_help",
            "Use requested_leave",
            "introduced in phase3_atomic_v2",
            "attributed reports, hypotheticals, explicit denials, corrections, and offers",
            "immediately preceding question as additional evidence",
            "positive polarity with uncertain status",
            "A deadline sets valid_to",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, candidate_prompt)
        self.assertIn(
            "every evidence quote is copied verbatim",
            candidate_prompt,
        )
        self.assertNotIn("Scan every independent clause", candidate_prompt)
        self.assertEqual(
            ATOMIC_EXTRACTION_V7_PROMPT_VERSION,
            "atomic-extraction-v7",
        )

    def test_v7_prompt_and_schema_snapshot_matches_smoke_and_full_suites(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v7_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V7_PROMPT_VERSION
        )

        self.assertEqual(snapshot["prompt_version"], "atomic-extraction-v7")
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["text_schema_sha256"],
            canonical_sha256(atomic_extraction_text_format()),
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_v8_adds_final_predicate_and_boolean_checks(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V8_PROMPT_VERSION
        )
        v7_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V7_PROMPT_VERSION
        )
        required_text = (
            "does not name an employer",
            "request to prepare a deliverable is assigned_task",
            "Use job_start_date, not employment_start_date",
            "object is true and use polarity to carry direct negation",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, candidate_prompt)
                self.assertNotIn(text, v7_prompt)
        self.assertEqual(
            ATOMIC_EXTRACTION_V8_PROMPT_VERSION,
            "atomic-extraction-v8",
        )

    def test_v8_prompt_schema_and_normalization_snapshot_matches_suites(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v8_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V8_PROMPT_VERSION
        )

        self.assertEqual(snapshot["prompt_version"], "atomic-extraction-v8")
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["text_schema_sha256"],
            canonical_sha256(atomic_extraction_text_format()),
        )
        self.assertEqual(
            snapshot["normalization_version"],
            "source_span_boolean_polarity_v2",
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                plan.config.generation_settings["normalization_version"],
                snapshot["normalization_version"],
            )
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_v9_targets_helper_subject_task_and_employer_errors(self) -> None:
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_CANDIDATE_PROMPT_VERSION
        )
        v8_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_V8_PROMPT_VERSION
        )
        required_text = (
            "subject_id is the person who can or will help",
            "named person as subject_id",
            "action verbs such as prepare, create, send, or review as assigned_task",
            "does not create a separate assigned_project claim",
            "require an explicitly named employer or organisation",
            "city job or Bengaluru job names a location-qualified job",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, candidate_prompt)
                self.assertNotIn(text, v8_prompt)
        self.assertEqual(
            ATOMIC_EXTRACTION_CANDIDATE_PROMPT_VERSION,
            "atomic-extraction-v9",
        )

    def test_v9_snapshot_uses_targeted_smoke_and_full_suite(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot = json.loads(
            (
                repo_root
                / "configs/extraction/atomic_extraction_prompt_v9_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        candidate_prompt = get_atomic_extraction_system_prompt(
            ATOMIC_EXTRACTION_CANDIDATE_PROMPT_VERSION
        )

        self.assertEqual(snapshot["prompt_version"], "atomic-extraction-v9")
        self.assertEqual(
            snapshot["system_prompt_sha256"],
            hashlib.sha256(candidate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["suites"]["smoke"]["case_ids"],
            ["atomic_conv_003", "atomic_email_003", "atomic_email_005"],
        )
        for suite in snapshot["suites"].values():
            plan = prepare_atomic_run(
                repo_root=repo_root,
                config_path=repo_root / suite["config_path"],
            )
            self.assertEqual(plan.config.prompt_sha256, suite["prompt_sha256"])
            self.assertEqual(
                [case_id for case_id, _ in plan.case_refs], suite["case_ids"]
            )

    def test_system_prompt_defines_time_and_epistemic_rules(self) -> None:
        required_text = (
            "Use asserted for a direct statement",
            "reported_by_other for a report of someone else's statement",
            "hypothetical for a condition or imagined case",
            "uncertain for qualified or doubtful language",
            "denied for an explicit denial",
            "corrected for an explicit correction",
            "The observation timestamp alone does not prove",
            "Otherwise use null",
            "Do not turn a question, suggestion, plan, hypothetical, or another "
            "speaker's statement into a confirmed fact about the user",
            "negative when it negates the predicate",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, ATOMIC_EXTRACTION_SYSTEM_PROMPT)

    def test_system_prompt_defines_empty_claim_behavior(self) -> None:
        self.assertIn(
            'Return {"claims":[]} when the source supports no claim.',
            ATOMIC_EXTRACTION_SYSTEM_PROMPT,
        )

    def test_prompt_contains_no_evaluation_side_content(self) -> None:
        observation = self.observation(
            observed_at="2026-04-05T19:30:38+05:30",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
            author_id="i_am_maya",
            author_name="Maya",
            text="I started a new role.",
        )
        prompt = ATOMIC_EXTRACTION_SYSTEM_PROMPT + build_atomic_extraction_prompt(
            self.source("conv_001", "conversation", observation)
        )

        for forbidden in ("oracle", "gold", "evaluation", "b1"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prompt.lower())


if __name__ == "__main__":
    unittest.main()
