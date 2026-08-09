from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from retrieval.index_evaluation import execute_index_evaluation
from retrieval.query_contracts import (
    INDEX_VERSION,
    RequestedValidTime,
    RetrievalQueryRequest,
)
from retrieval.query_planner import build_query_plan, load_query_planner_config
from retrieval.query_repository import (
    RetrievalQueryRepository,
    RetrievalQueryRepositoryError,
)


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
SHA = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for query-planning integration tests",
)
class RetrievalQueryPlanningIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        cls.config = load_query_planner_config(
            ROOT / "configs/retrieval/query_planner_v1.json"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        self.temp = tempfile.TemporaryDirectory()
        execute_index_evaluation(
            lambda: psycopg.connect(DATABASE_URL, autocommit=True),
            Path(self.temp.name) / "release",
            repo_root=ROOT,
        )
        self.repository = RetrievalQueryRepository(self.connection)
        self.cutoff = self.connection.execute(
            """
            SELECT max(transaction_as_of) FROM retrieval_index_runs
            WHERE user_id = 'user_001' AND status = 'succeeded'
            """
        ).fetchone()[0]

    def tearDown(self) -> None:
        self.temp.cleanup()
        self.connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def _plan(
        self,
        text: str = "Tell me something useful",
        *,
        user_id: str = "user_001",
        as_of: datetime | None = None,
        kinds: tuple[str, ...] = ("atomic", "session"),
        valid_time: RequestedValidTime | None = None,
        speakers: tuple[str, ...] = (),
        entities: tuple[str, ...] = (),
        sensitivity: str = "sensitive",
        allow_unclassified: bool = True,
    ):
        request = RetrievalQueryRequest(
            digest(f"{user_id}:{text}:{as_of}:{kinds}:{valid_time}:{speakers}:{entities}"),
            user_id,
            text,
            as_of or self.cutoff,
            INDEX_VERSION,
            kinds,
            valid_time,
            speakers,
            entities,
            sensitivity,
            allow_unclassified,
        )
        return build_query_plan(request, config=self.config)

    def _decision(self, result, record_id: str):
        return next(item for item in result.decisions if item.index_record_id == record_id)

    def _record(self, kind: str = "atomic", *, offset: int = 0):
        return self.connection.execute(
            """
            SELECT index_record_id FROM retrieval_index_records
            WHERE user_id = 'user_001' AND record_kind = %s
            ORDER BY index_record_id OFFSET %s
            """,
            (kind, offset),
        ).fetchone()[0]

    def test_user_first_latest_safe_snapshot_and_cross_user_invisibility(self) -> None:
        result = self.repository.filter(self._plan())
        expected = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT index_record_id FROM retrieval_index_records
                WHERE user_id = 'user_001' AND run_id = %s
                """,
                (result.snapshot_run_id,),
            ).fetchall()
        }
        foreign = {
            row[0]
            for row in self.connection.execute(
                "SELECT index_record_id FROM retrieval_index_records WHERE user_id = 'user_002'"
            ).fetchall()
        }
        self.assertEqual({item.index_record_id for item in result.decisions}, expected)
        self.assertTrue(expected.isdisjoint(foreign))

        newer = self.cutoff + timedelta(hours=1)
        run_id = digest("newer-safe-empty-run")
        self.connection.execute(
            """
            INSERT INTO retrieval_index_runs (
                run_id, user_id, index_version, content_renderer_version,
                embedding_version, embedding_dimension, config_sha256,
                idempotency_key, transaction_as_of, input_snapshot_sha256,
                records_snapshot_sha256, status, atomic_count, session_count,
                record_count, started_at, completed_at
            ) VALUES (
                %s, 'user_001', 'retrieval_index_v1', 'retrieval_content_v1',
                'deterministic_token_hash_v1', 256, %s, 'query:newer', %s,
                %s, %s, 'succeeded', 0, 0, 0, %s, %s
            )
            """,
            (run_id, SHA, newer, SHA, digest("empty"), newer, newer),
        )
        newest = self.repository.filter(self._plan(as_of=newer))
        self.assertEqual(newest.snapshot_run_id, run_id)
        self.assertEqual(newest.decisions, ())
        with self.assertRaisesRegex(RetrievalQueryRepositoryError, "no_safe_index_snapshot"):
            self.repository.filter(self._plan(as_of=self.cutoff - timedelta(days=3650)))

    def test_transaction_source_and_future_relation_fail_closed(self) -> None:
        record_id = self._record()
        source_id = self.connection.execute(
            """
            SELECT source_id FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            ORDER BY source_order LIMIT 1
            """,
            (record_id,),
        ).fetchone()[0]
        future = self.cutoff + timedelta(days=1)
        self.connection.execute(
            """
            UPDATE retrieval_index_records
            SET transaction_from = %s, transaction_to = %s
            WHERE index_record_id = %s
            """,
            (self.cutoff - timedelta(days=1), self.cutoff, record_id),
        )
        self.connection.execute(
            "UPDATE source_events SET ingested_at = %s WHERE user_id = 'user_001' AND source_id = %s",
            (future, source_id),
        )
        anchor = self.connection.execute(
            """
            SELECT atomic_claim_id, atomic_claim_version_id
            FROM retrieval_index_records WHERE index_record_id = %s
            """,
            (record_id,),
        ).fetchone()
        other = self.connection.execute(
            """
            SELECT claim_id, version_id FROM claim_versions
            WHERE user_id = 'user_001' AND claim_id <> %s
            ORDER BY claim_id LIMIT 1
            """,
            (anchor[0],),
        ).fetchone()
        claims = sorted(((anchor[0], anchor[1]), (other[0], other[1])))
        self.connection.execute(
            """
            INSERT INTO conflict_decisions (
                decision_id, user_id, classifier_version, rule_version,
                pair_id, left_claim_id, right_claim_id, left_version_id,
                right_version_id, input_snapshot_sha256, transaction_as_of,
                matched_rule, label, classified_at
            ) VALUES (
                'query_future_decision', 'user_001', 'relation_classifier_v1',
                'conflict_relation_rules_v1', 'query_future_pair', %s, %s,
                %s, %s, %s, %s, 'explicit_correction',
                'explicit_correction', %s
            )
            """,
            (claims[0][0], claims[1][0], claims[0][1], claims[1][1], SHA, future, future),
        )
        self.connection.execute(
            """
            INSERT INTO claim_relations (
                relation_id, user_id, decision_id, classifier_version,
                source_claim_id, target_claim_id, relation_type, confidence,
                input_snapshot_sha256, created_at
            ) VALUES (
                'query_future_relation', 'user_001', 'query_future_decision',
                'relation_classifier_v1', %s, %s, 'corrects', 1, %s, %s
            )
            """,
            (claims[0][0], claims[1][0], SHA, future),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_relation_links (
                user_id, index_record_id, relation_id, source_claim_id,
                target_claim_id, relation_type, direction, relation_order
            ) VALUES (
                'user_001', %s, 'query_future_relation', %s, %s,
                'corrects', 'outgoing', 99
            )
            """,
            (record_id, claims[0][0], claims[1][0]),
        )
        decision = self._decision(self.repository.filter(self._plan()), record_id)
        self.assertEqual(
            set(decision.rejection_reasons),
            {"relation_after_as_of", "source_after_as_of", "transaction_hidden"},
        )

    def test_inclusive_date_timestamp_and_unknown_current_time(self) -> None:
        date_row = self.connection.execute(
            """
            SELECT index_record_id, valid_from_date
            FROM retrieval_index_records
            WHERE user_id = 'user_001' AND record_kind = 'atomic'
              AND valid_from_date IS NOT NULL
            ORDER BY index_record_id LIMIT 1
            """
        ).fetchone()
        exact_date = RequestedValidTime(kind="point", point_date=date_row[1])
        exact = self._decision(
            self.repository.filter(self._plan(kinds=("atomic",), valid_time=exact_date)),
            date_row[0],
        )
        self.assertNotIn("valid_time_mismatch", exact.rejection_reasons)
        touching_date_range = RequestedValidTime(
            kind="range",
            range_start_date=date_row[1] - timedelta(days=1),
            range_end_date=date_row[1],
        )
        touching_date = self._decision(
            self.repository.filter(
                self._plan(kinds=("atomic",), valid_time=touching_date_range)
            ),
            date_row[0],
        )
        self.assertNotIn("valid_time_mismatch", touching_date.rejection_reasons)
        before = RequestedValidTime(
            kind="point", point_date=date_row[1] - timedelta(days=1)
        )
        mismatched = self._decision(
            self.repository.filter(self._plan(kinds=("atomic",), valid_time=before)),
            date_row[0],
        )
        self.assertIn("valid_time_mismatch", mismatched.rejection_reasons)

        timestamp_id = self._record("atomic", offset=1)
        self.connection.execute(
            """
            UPDATE retrieval_index_records
            SET time_precision = 'timestamp', valid_from_date = NULL,
                valid_to_date = NULL, valid_from_timestamp = %s,
                valid_to_timestamp = %s
            WHERE index_record_id = %s
            """,
            (self.cutoff, self.cutoff, timestamp_id),
        )
        exact_timestamp = RequestedValidTime(
            kind="point", point_timestamp=self.cutoff
        )
        timestamp_decision = self._decision(
            self.repository.filter(
                self._plan(kinds=("atomic",), valid_time=exact_timestamp)
            ),
            timestamp_id,
        )
        self.assertNotIn("valid_time_mismatch", timestamp_decision.rejection_reasons)
        touching_timestamp_range = RequestedValidTime(
            kind="range",
            range_start_timestamp=self.cutoff - timedelta(seconds=1),
            range_end_timestamp=self.cutoff,
        )
        touching_timestamp = self._decision(
            self.repository.filter(
                self._plan(
                    kinds=("atomic",), valid_time=touching_timestamp_range
                )
            ),
            timestamp_id,
        )
        self.assertNotIn("valid_time_mismatch", touching_timestamp.rejection_reasons)

        current_sessions = self.repository.filter(
            self._plan("What is current now?", kinds=("session",))
        )
        unknown_session_ids = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT index_record_id FROM retrieval_index_records
                WHERE user_id = 'user_001' AND record_kind = 'session'
                  AND time_precision IN ('unknown', 'mixed')
                """
            ).fetchall()
        }
        self.assertTrue(unknown_session_ids)
        self.assertTrue(
            all(
                "unknown_valid_time" in item.rejection_reasons
                for item in current_sessions.decisions
                if item.index_record_id in unknown_session_ids
            )
        )

    def test_atomic_and_session_speaker_and_subject_filters_are_exact(self) -> None:
        atomic = self.connection.execute(
            """
            SELECT index_record_id, speaker_id, subject_id
            FROM retrieval_index_records
            WHERE user_id = 'user_001' AND record_kind = 'atomic'
            ORDER BY index_record_id LIMIT 1
            """
        ).fetchone()
        matched = self._decision(
            self.repository.filter(
                self._plan(
                    kinds=("atomic",),
                    speakers=(atomic[1],),
                    entities=(atomic[2],),
                )
            ),
            atomic[0],
        )
        self.assertTrue(matched.eligible)
        wrong = self._decision(
            self.repository.filter(
                self._plan(
                    kinds=("atomic",),
                    speakers=("speaker_missing",),
                    entities=("entity_missing",),
                )
            ),
            atomic[0],
        )
        self.assertIn("speaker_mismatch", wrong.rejection_reasons)
        self.assertIn("entity_mismatch", wrong.rejection_reasons)

        session = self.connection.execute(
            """
            SELECT record.index_record_id, claim.speaker_id, claim.subject_id
            FROM retrieval_index_records AS record
            JOIN retrieval_index_claim_links AS link
              ON link.user_id = record.user_id
             AND link.index_record_id = record.index_record_id
            JOIN claims AS claim
              ON claim.user_id = link.user_id AND claim.claim_id = link.claim_id
            WHERE record.user_id = 'user_001' AND record.record_kind = 'session'
            ORDER BY record.index_record_id, link.claim_order LIMIT 1
            """
        ).fetchone()
        session_match = self._decision(
            self.repository.filter(
                self._plan(
                    kinds=("session",),
                    speakers=(session[1],),
                    entities=(session[2],),
                )
            ),
            session[0],
        )
        self.assertTrue(session_match.eligible)

    def test_lifecycle_and_sensitivity_policies_remain_visible(self) -> None:
        historical_id = self._record("atomic", offset=0)
        sensitive_id = self._record("atomic", offset=1)
        unclassified_id = self._record("atomic", offset=2)
        self.connection.execute(
            "UPDATE retrieval_index_records SET lifecycle_statuses = '[\"historical\"]'::jsonb WHERE index_record_id = %s",
            (historical_id,),
        )
        self.connection.execute(
            "UPDATE retrieval_index_records SET sensitivity = 'sensitive', contains_sensitive = true WHERE index_record_id = %s",
            (sensitive_id,),
        )
        self.connection.execute(
            "UPDATE retrieval_index_records SET sensitivity = NULL, contains_sensitive = false WHERE index_record_id = %s",
            (unclassified_id,),
        )
        current = self._decision(
            self.repository.filter(
                self._plan("What is current now?", kinds=("atomic",))
            ),
            historical_id,
        )
        self.assertIn("lifecycle_blocked", current.rejection_reasons)
        standard = self.repository.filter(
            self._plan(
                kinds=("atomic",),
                sensitivity="standard",
                allow_unclassified=False,
            )
        )
        self.assertIn(
            "sensitive_not_authorized",
            self._decision(standard, sensitive_id).rejection_reasons,
        )
        self.assertIn(
            "unclassified_sensitivity",
            self._decision(standard, unclassified_id).rejection_reasons,
        )
        authorized = self.repository.filter(self._plan(kinds=("atomic",)))
        self.assertTrue(self._decision(authorized, sensitive_id).eligible)
        self.assertTrue(self._decision(authorized, unclassified_id).eligible)
        self.assertTrue(self._decision(authorized, unclassified_id).unclassified_sensitivity)
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.connection.execute(
                "UPDATE retrieval_index_records SET sensitivity = 'restricted' WHERE index_record_id = %s",
                (sensitive_id,),
            )

    def test_deletion_removes_record_and_partial_lineage_fails_closed(self) -> None:
        deleted_id = self._record("atomic", offset=0)
        evidence = self.connection.execute(
            """
            SELECT claim_id, span_id, support_type
            FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            ORDER BY source_order LIMIT 1
            """,
            (deleted_id,),
        ).fetchone()
        self.connection.execute(
            """
            DELETE FROM evidence_links
            WHERE user_id = 'user_001' AND claim_id = %s
              AND span_id = %s AND support_type = %s
            """,
            evidence,
        )
        after_delete = self.repository.filter(self._plan(kinds=("atomic",)))
        self.assertNotIn(deleted_id, {item.index_record_id for item in after_delete.decisions})

        partial_id = self._record("atomic", offset=0)
        self.connection.execute(
            "DELETE FROM retrieval_index_source_links WHERE user_id = 'user_001' AND index_record_id = %s",
            (partial_id,),
        )
        partial = self._decision(
            self.repository.filter(self._plan(kinds=("atomic",))), partial_id
        )
        self.assertIn("partial_lineage", partial.rejection_reasons)

    def test_deterministic_replay_and_sql_spy_prove_no_search_or_ranking(self) -> None:
        class Spy:
            def __init__(self, connection) -> None:
                self.connection = connection
                self.statements = []

            def execute(self, query, params=None):
                self.statements.append(str(query))
                return self.connection.execute(query, params)

        spy = Spy(self.connection)
        repository = RetrievalQueryRepository(spy)
        plan = self._plan()
        first = repository.filter(plan)
        second = repository.filter(plan)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.index_record_id for item in first.decisions),
            tuple(sorted(item.index_record_id for item in first.decisions)),
        )
        sql = "\n".join(spy.statements).casefold()
        user_at = sql.index("where user_id = %s")
        index_at = sql.index("and index_version = %s", user_at)
        self.assertLess(user_at, index_at)
        for forbidden in (
            "@@",
            "<=>",
            "<->",
            "search_document",
            "embedding",
            "ts_rank",
            "score",
            "fusion",
            "rerank",
        ):
            self.assertNotIn(forbidden, sql)
        self.assertIsNone(re.search(r"\blimit\s+\d+\b", sql))


if __name__ == "__main__":
    unittest.main()
