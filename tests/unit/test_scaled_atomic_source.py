from __future__ import annotations

import json
import unittest

from extraction.prompt import build_atomic_extraction_prompt
from extraction.scaled_source import (
    DEVELOPMENT_SOURCE_REFS,
    load_scaled_development_sources,
)


class ScaledAtomicSourceTests(unittest.TestCase):
    def test_loads_only_two_development_users_in_frozen_order(self) -> None:
        sources = load_scaled_development_sources()

        self.assertEqual(len(sources), 20)
        self.assertEqual(
            tuple((source.user_id, source.source_id) for source in sources),
            DEVELOPMENT_SOURCE_REFS,
        )
        self.assertEqual(
            tuple(source.source_type for source in sources[:10]),
            (
                "conversation",
                "email",
                "chat",
                "calendar",
                "conversation",
                "email",
                "chat",
                "calendar",
                "conversation",
                "conversation",
            ),
        )

    def test_provider_payload_contains_only_allowlisted_adapted_fields(self) -> None:
        for source in load_scaled_development_sources():
            prompt = build_atomic_extraction_prompt(
                source, include_speaker_name=False
            )
            payload = json.loads(prompt.split("\n", 1)[1])
            self.assertEqual(
                set(payload),
                {"source_type", "source_id", "known_entities", "observations"},
            )
            self.assertEqual(
                {item["entity_id"] for item in payload["known_entities"]},
                {item.entity_id for item in source.known_entities},
            )
            self.assertNotIn("user_id", payload)
            self.assertFalse(
                {"ingested_at", "profile_note", "split", "task", "gold", "oracle"}
                & set(payload)
            )

    def test_calendar_is_one_user_observation_with_null_message(self) -> None:
        calendar = next(
            source
            for source in load_scaled_development_sources()
            if source.source_id == "scaled_user_001_calendar_001"
        )

        self.assertEqual(len(calendar.observations), 1)
        observation = calendar.observations[0]
        self.assertIsNone(observation.message_id)
        self.assertEqual(observation.author_id, "user_001")
        self.assertIsNotNone(observation.title)
        self.assertIsNone(observation.start_at)
        self.assertIsNone(observation.end_at)


if __name__ == "__main__":
    unittest.main()
