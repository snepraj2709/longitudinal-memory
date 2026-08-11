from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from summaries.contracts import (
    BOUNDARY_VERSION,
    SessionBoundaryError,
    SessionSource,
    SessionizationRequest,
    load_session_boundary_config,
)
from summaries.repository import SessionSourceRepository, VISIBLE_SOURCES_SQL
from summaries.sessions import SessionizationService, SessionizationServiceError


CONFIG_PATH = Path("configs/summaries/session_boundaries_v1.json")
UTC = timezone.utc
BASE = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)


def source(
    source_id: str,
    source_type: str,
    offset_seconds: int,
    *,
    user_id: str = "user_001",
    session_id: str | None = None,
    metadata: dict[str, object] | None = None,
    ingested_offset: int = 0,
) -> SessionSource:
    produced = BASE + timedelta(seconds=offset_seconds)
    return SessionSource(
        source_id=source_id,
        user_id=user_id,
        source_type=source_type,
        session_id=session_id,
        produced_at=produced,
        ingested_at=produced + timedelta(seconds=ingested_offset),
        metadata=metadata or {},
    )


class FakeRepository:
    def __init__(self, sources: tuple[SessionSource, ...]) -> None:
        self.sources = sources
        self.requests: list[SessionizationRequest] = []

    def list_visible_sources(
        self, request: SessionizationRequest
    ) -> tuple[SessionSource, ...]:
        self.requests.append(request)
        return self.sources


def service_for(*sources: SessionSource) -> SessionizationService:
    ordered = tuple(sorted(sources, key=lambda item: (item.produced_at, item.source_id)))
    return SessionizationService(FakeRepository(ordered), config_path=CONFIG_PATH)


def request(*, as_of: datetime | None = None, user_id: str = "user_001") -> SessionizationRequest:
    return SessionizationRequest(
        user_id=user_id,
        transaction_as_of=as_of or BASE + timedelta(days=2),
        boundary_version=BOUNDARY_VERSION,
    )


class ConfigAndContractTests(unittest.TestCase):
    def test_frozen_config_and_hash(self) -> None:
        config = load_session_boundary_config(CONFIG_PATH)
        self.assertEqual(config.boundary_version, BOUNDARY_VERSION)
        self.assertEqual(config.chat_unthreaded_gap_seconds, 1800)
        self.assertEqual(
            config.chat_declared_key_precedence,
            ("session_id", "metadata.thread_id"),
        )
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "d2af3cbfd35e24d1f0b3a10acd148fb46a7fc2b4cd96268182bd12dd8570f20b",
        )

    def test_request_requires_aware_time_and_frozen_version(self) -> None:
        with self.assertRaisesRegex(SessionBoundaryError, "timezone-aware"):
            SessionizationRequest("user_001", datetime(2026, 1, 1), BOUNDARY_VERSION)
        with self.assertRaisesRegex(SessionBoundaryError, "unsupported"):
            SessionizationRequest("user_001", BASE, "session_boundaries_v2")

    def test_source_metadata_is_json_safe_and_detached(self) -> None:
        metadata = {"thread_id": "日本語", "items": [1, True, None]}
        record = source("chat_001", "chat", 0, metadata=metadata)
        metadata["thread_id"] = "changed"
        self.assertEqual(record.metadata["thread_id"], "日本語")
        with self.assertRaisesRegex(SessionBoundaryError, "JSON-safe"):
            source("chat_bad", "chat", 0, metadata={"bad": float("nan")})
        with self.assertRaisesRegex(SessionBoundaryError, "JSON-safe"):
            source("chat_bad", "chat", 0, metadata={1: "bad"})  # type: ignore[dict-item]


class BoundaryRuleTests(unittest.TestCase):
    def test_conversation_and_calendar_are_one_source_each(self) -> None:
        result = service_for(
            source("conversation_001", "conversation", 0, session_id="ignored"),
            source("conversation_002", "conversation", 1),
            source("calendar_001", "calendar", 2, metadata={"thread_id": "ignored"}),
        ).define_sessions(request())
        self.assertEqual(
            [(item.source_type, item.boundary_kind, item.source_ids) for item in result.definitions],
            [
                ("conversation", "source", ("conversation_001",)),
                ("conversation", "source", ("conversation_002",)),
                ("calendar", "source", ("calendar_001",)),
            ],
        )

    def test_email_precedence_grouping_and_own_source_fallback(self) -> None:
        result = service_for(
            source("email_001", "email", 0, session_id="thread-a"),
            source("email_002", "email", 1, metadata={"thread_id": "thread-a"}),
            source("email_003", "email", 2, metadata={"subject": "same subject"}),
            source("email_004", "email", 3, metadata={"subject": "same subject"}),
        ).define_sessions(request())
        self.assertEqual(
            [item.source_ids for item in result.definitions],
            [("email_001", "email_002"), ("email_003",), ("email_004",)],
        )
        serialized = json.dumps(
            [item.__dict__ for item in result.definitions], default=str, sort_keys=True
        )
        self.assertNotIn("thread-a", serialized)
        self.assertNotIn("same subject", serialized)

    def test_email_conflicting_declared_keys_fail_without_values(self) -> None:
        candidate = service_for(
            source(
                "email_001",
                "email",
                0,
                session_id="private-a",
                metadata={"thread_id": "private-b"},
            )
        )
        with self.assertRaises(SessionizationServiceError) as caught:
            candidate.define_sessions(request())
        self.assertEqual(caught.exception.code, "email_thread_conflict")
        self.assertNotIn("private", str(caught.exception))

    def test_chat_declared_key_precedence_uses_session_id(self) -> None:
        result = service_for(
            source(
                "chat_001",
                "chat",
                0,
                session_id="session-key",
                metadata={"thread_id": "metadata-key"},
            ),
            source("chat_002", "chat", 1, session_id="session-key"),
            source("chat_003", "chat", 2, metadata={"thread_id": "metadata-key"}),
        ).define_sessions(request())
        self.assertEqual(
            [item.source_ids for item in result.definitions],
            [("chat_001", "chat_002"), ("chat_003",)],
        )

    def test_unthreaded_chat_gap_is_inclusive_at_1800_and_splits_at_1801(self) -> None:
        result = service_for(
            source("chat_001", "chat", 0),
            source("chat_002", "chat", 1800),
            source("chat_003", "chat", 3601),
        ).define_sessions(request())
        self.assertEqual(
            [item.source_ids for item in result.definitions],
            [("chat_001", "chat_002"), ("chat_003",)],
        )

    def test_declared_and_unthreaded_chats_never_mix(self) -> None:
        result = service_for(
            source("chat_001", "chat", 0),
            source("chat_002", "chat", 10, session_id="declared"),
            source("chat_003", "chat", 20),
            source("chat_004", "chat", 30, session_id="declared"),
        ).define_sessions(request())
        self.assertEqual(
            [item.source_ids for item in result.definitions],
            [("chat_001",), ("chat_002", "chat_004"), ("chat_003",)],
        )

    def test_late_ingestion_and_cross_user_fail_before_feature_work(self) -> None:
        cutoff = BASE + timedelta(hours=1)
        late = source("chat_late", "chat", 0, ingested_offset=3601)
        with patch("summaries.sessions._metadata_thread_id") as feature:
            with self.assertRaisesRegex(SessionizationServiceError, "source_not_visible"):
                service_for(late).define_sessions(request(as_of=cutoff))
            feature.assert_not_called()
        other = source("email_other", "email", 0, user_id="user_002")
        with patch("summaries.sessions._metadata_thread_id") as feature:
            with self.assertRaisesRegex(SessionizationServiceError, "source_user_mismatch"):
                service_for(other).define_sessions(request())
            feature.assert_not_called()

    def test_out_of_order_repository_result_fails(self) -> None:
        first = source("chat_001", "chat", 0)
        second = source("chat_002", "chat", 1)
        candidate = SessionizationService(
            FakeRepository((second, first)), config_path=CONFIG_PATH
        )
        with self.assertRaisesRegex(SessionizationServiceError, "source_order_invalid"):
            candidate.define_sessions(request())

    def test_deleted_source_disappears_without_stored_session_state(self) -> None:
        first = source("chat_001", "chat", 0)
        deleted = source("chat_002", "chat", 1)
        before = service_for(first, deleted).define_sessions(request())
        after = service_for(first).define_sessions(request())
        self.assertEqual(before.definitions[0].source_ids, ("chat_001", "chat_002"))
        self.assertEqual(after.definitions[0].source_ids, ("chat_001",))
        self.assertNotEqual(
            before.definitions[0].membership_sha256,
            after.definitions[0].membership_sha256,
        )


class IdentityAndRepositoryTests(unittest.TestCase):
    def test_definition_and_membership_hashes_bind_exact_fields(self) -> None:
        as_of = BASE + timedelta(days=1)
        result = service_for(
            source("chat_001", "chat", 0),
            source("chat_002", "chat", 1800),
        ).define_sessions(request(as_of=as_of))
        definition = result.definitions[0]
        identifier_payload = {
            "boundary_key": "inactivity:chat_001",
            "boundary_kind": "inactivity",
            "boundary_version": BOUNDARY_VERSION,
            "source_type": "chat",
            "user_id": "user_001",
        }
        expected_id = hashlib.sha256(
            json.dumps(identifier_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(definition.definition_id, expected_id)
        membership_payload = {
            "boundary_kind": "inactivity",
            "boundary_version": BOUNDARY_VERSION,
            "definition_id": expected_id,
            "end_at": "2026-01-02T09:30:00Z",
            "source_ids": ("chat_001", "chat_002"),
            "source_type": "chat",
            "start_at": "2026-01-02T09:00:00Z",
            "transaction_as_of": "2026-01-03T09:00:00Z",
            "user_id": "user_001",
        }
        self.assertEqual(
            definition.membership_sha256,
            hashlib.sha256(
                json.dumps(membership_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def test_identity_is_user_local_and_output_is_stably_ordered(self) -> None:
        sources = (
            source("calendar_001", "calendar", 20),
            source("chat_001", "chat", 10),
            source("email_001", "email", 10),
        )
        result = service_for(*sources).define_sessions(request())
        self.assertEqual(
            [(item.start_at, item.source_type, item.source_ids[0]) for item in result.definitions],
            [
                (BASE + timedelta(seconds=10), "chat", "chat_001"),
                (BASE + timedelta(seconds=10), "email", "email_001"),
                (BASE + timedelta(seconds=20), "calendar", "calendar_001"),
            ],
        )
        other_result = service_for(
            source("chat_001", "chat", 10, user_id="user_002")
        ).define_sessions(request(user_id="user_002"))
        self.assertNotEqual(
            result.definitions[0].definition_id,
            other_result.definitions[0].definition_id,
        )

    def test_hash_collision_is_rejected(self) -> None:
        with patch("summaries.sessions._sha256", return_value="0" * 64):
            with self.assertRaisesRegex(SessionizationServiceError, "definition_id_collision"):
                service_for(
                    source("conversation_001", "conversation", 0),
                    source("conversation_002", "conversation", 1),
                ).define_sessions(request())

    def test_repository_sql_is_user_scoped_cutoff_inclusive_and_ordered(self) -> None:
        cutoff = BASE + timedelta(hours=1)
        rows = [
            (
                "chat_001",
                "user_001",
                "chat",
                None,
                BASE,
                cutoff,
                {"thread_id": "thread"},
            )
        ]

        class Cursor:
            def fetchall(self):
                return rows

        class Connection:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[object, ...]]] = []

            def execute(self, query, params):
                self.calls.append((query, params))
                return Cursor()

        connection = Connection()
        repository = SessionSourceRepository(connection)
        records = repository.list_visible_sources(request(as_of=cutoff))
        self.assertEqual(records[0].ingested_at, cutoff)
        self.assertEqual(connection.calls, [(VISIBLE_SOURCES_SQL, ("user_001", cutoff))])
        self.assertIn("WHERE user_id = %s AND ingested_at <= %s", VISIBLE_SOURCES_SQL)
        self.assertTrue(VISIBLE_SOURCES_SQL.endswith("ORDER BY produced_at, source_id"))

    def test_repository_rejects_cross_user_duplicate_late_and_unordered_rows(self) -> None:
        cutoff = BASE + timedelta(hours=1)

        class Cursor:
            def __init__(self, rows):
                self.rows = rows

            def fetchall(self):
                return self.rows

        class Connection:
            def __init__(self, rows):
                self.rows = rows

            def execute(self, query, params):
                return Cursor(self.rows)

        base_row = ("a", "user_001", "chat", None, BASE, BASE, {})
        cases = (
            [("a", "user_002", "chat", None, BASE, BASE, {})],
            [base_row, base_row],
            [("a", "user_001", "chat", None, BASE, cutoff + timedelta(seconds=1), {})],
            [
                ("b", "user_001", "chat", None, BASE + timedelta(seconds=1), BASE, {}),
                base_row,
            ],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                repository = SessionSourceRepository(Connection(rows))
                with self.assertRaises(SessionBoundaryError):
                    repository.list_visible_sources(request(as_of=cutoff))


if __name__ == "__main__":
    unittest.main()
