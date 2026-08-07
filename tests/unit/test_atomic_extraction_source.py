from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation.history import (
    HistoryDataError,
    HistoryObservation,
    load_history_observations,
)
from extraction.source import (
    PILOT_SOURCE_DIR,
    ExtractionSource,
    KnownEntity,
    group_source_observations,
    load_pilot_sources,
)


class AtomicExtractionSourceTests(unittest.TestCase):
    def observation(
        self,
        *,
        observed_at: str,
        source_type: str,
        source_id: str,
        message_id: str | None,
        author_id: str = "person_001",
        author_name: str = "Maya",
        text: str = "Source text",
    ) -> HistoryObservation:
        return HistoryObservation(
            observed_at=datetime.fromisoformat(observed_at),
            source_type=source_type,
            source_id=source_id,
            message_id=message_id,
            author_id=author_id,
            author_name=author_name,
            text=text,
        )

    def test_groups_multiple_messages_and_is_immutable(self) -> None:
        first = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
        )
        second = self.observation(
            observed_at="2026-01-01T09:05:00+00:00",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_002",
        )

        sources = group_source_observations((first, second))

        self.assertEqual(
            sources,
            (
                ExtractionSource(
                    "conv_001",
                    "conversation",
                    (first, second),
                    (KnownEntity("person_001", "Maya"),),
                ),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            sources[0].source_id = "conv_002"

    def test_orders_groups_by_earliest_observation_then_source_id(self) -> None:
        later_in_source = self.observation(
            observed_at="2026-01-03T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_b",
            message_id="msg_b2",
        )
        earliest_in_source = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_b",
            message_id="msg_b1",
        )
        same_time_a = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_a",
            message_id="msg_a1",
        )
        later_source = self.observation(
            observed_at="2026-01-02T09:00:00+00:00",
            source_type="email",
            source_id="email_001",
            message_id="msg_e1",
        )

        sources = group_source_observations(
            (later_in_source, later_source, earliest_in_source, same_time_a)
        )

        self.assertEqual(
            [source.source_id for source in sources],
            ["conv_a", "conv_b", "email_001"],
        )
        self.assertEqual(
            sources[1].observations,
            (later_in_source, earliest_in_source),
        )

    def test_preserves_conversation_and_email_fields_exactly(self) -> None:
        conversation = self.observation(
            observed_at="2026-02-01T10:00:00+05:30",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_conv_001_001",
            author_id="maya_001",
            author_name="Maya Rao",
            text="I am considering a new role.",
        )
        email = self.observation(
            observed_at="2026-02-02T11:30:00+05:30",
            source_type="email",
            source_id="email_001",
            message_id="msg_email_001_001",
            author_id="recruiter_001",
            author_name="Anita Shah",
            text="Would you like to discuss the role?",
        )

        sources = group_source_observations((email, conversation))

        self.assertIs(sources[0].observations[0], conversation)
        self.assertIs(sources[1].observations[0], email)
        self.assertEqual(conversation.message_id, "msg_conv_001_001")
        self.assertEqual(conversation.author_id, "maya_001")
        self.assertEqual(conversation.author_name, "Maya Rao")
        self.assertEqual(
            conversation.observed_at,
            datetime.fromisoformat("2026-02-01T10:00:00+05:30"),
        )
        self.assertEqual(conversation.text, "I am considering a new role.")
        self.assertEqual(email.message_id, "msg_email_001_001")
        self.assertEqual(email.author_id, "recruiter_001")
        self.assertEqual(email.author_name, "Anita Shah")
        self.assertEqual(email.text, "Would you like to discuss the role?")

    def test_preserves_calendar_null_message_id(self) -> None:
        calendar = self.observation(
            observed_at="2026-03-01T08:00:00+05:30",
            source_type="calendar",
            source_id="cal_001",
            message_id=None,
            text="Final semester examinations.",
        )

        source = group_source_observations((calendar,))[0]

        self.assertIsNone(source.observations[0].message_id)

    def test_rejects_mixed_source_types(self) -> None:
        conversation = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="source_001",
            message_id="msg_001",
        )
        email = self.observation(
            observed_at="2026-01-01T10:00:00+00:00",
            source_type="email",
            source_id="source_001",
            message_id="msg_002",
        )

        with self.assertRaisesRegex(HistoryDataError, "mixed source types"):
            group_source_observations((conversation, email))

    def test_rejects_duplicate_evidence_references(self) -> None:
        first = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
        )
        duplicate = self.observation(
            observed_at="2026-01-01T10:00:00+00:00",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
        )

        with self.assertRaisesRegex(HistoryDataError, "duplicate evidence reference"):
            group_source_observations((first, duplicate))

    def test_shares_known_entities_and_rejects_conflicting_names(self) -> None:
        maya = self.observation(
            observed_at="2026-01-01T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_001",
            message_id="msg_001",
            author_id="i_am_maya",
            author_name="Maya",
        )
        asha = self.observation(
            observed_at="2026-01-02T09:00:00+00:00",
            source_type="email",
            source_id="email_001",
            message_id="msg_002",
            author_id="person_asha",
            author_name="Asha",
        )

        sources = group_source_observations((asha, maya))

        expected = (
            KnownEntity("i_am_maya", "Maya"),
            KnownEntity("person_asha", "Asha"),
        )
        self.assertTrue(all(source.known_entities == expected for source in sources))

        conflicting = self.observation(
            observed_at="2026-01-03T09:00:00+00:00",
            source_type="conversation",
            source_id="conv_002",
            message_id="msg_003",
            author_id="i_am_maya",
            author_name="Maya Rao",
        )
        with self.assertRaisesRegex(HistoryDataError, "conflicting names"):
            group_source_observations((maya, conflicting))

    def test_loads_only_current_pilot_source_directory(self) -> None:
        with patch(
            "extraction.source.load_history_observations",
            wraps=load_history_observations,
        ) as loader:
            sources = load_pilot_sources()

        loader.assert_called_once_with(Path("data/pilot/sources"))
        self.assertEqual(PILOT_SOURCE_DIR, Path("data/pilot/sources"))
        self.assertTrue(sources)
        self.assertEqual(
            {source.source_type for source in sources},
            {"calendar", "conversation", "email"},
        )


if __name__ == "__main__":
    unittest.main()
