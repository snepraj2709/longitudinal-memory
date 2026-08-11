from __future__ import annotations

from datetime import timedelta
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from retrieval.baselines import build_baseline_request, load_baseline_config
from retrieval.embeddings import DeterministicTokenHashEmbedder
from retrieval.index_evaluation import execute_index_evaluation
from retrieval.query_contracts import RetrievalQueryRequest
from retrieval.query_planner import build_query_plan, load_query_planner_config
from retrieval.query_repository import RetrievalQueryRepository
from retrieval.search_repository import (
    RetrievalSearchRepository,
    RetrievalSearchRepositoryError,
)
from ingestion.service import IngestionService


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
PLANNER_CONFIG = ROOT / "configs/retrieval/query_planner_v1.json"
BASELINE_CONFIG = ROOT / "configs/retrieval/baseline_v1.json"
SHA = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for baseline integration tests",
)
class RetrievalBaselineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        cls.planner_config = load_query_planner_config(PLANNER_CONFIG)
        cls.baseline_config = load_baseline_config(BASELINE_CONFIG)

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
        self.repository = RetrievalSearchRepository(self.connection)
        self.cutoff = self.connection.execute(
            """
            SELECT max(transaction_as_of)
            FROM retrieval_index_runs
            WHERE status = 'succeeded'
            """
        ).fetchone()[0]

    def tearDown(self) -> None:
        self.temp.cleanup()
        self.connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.connection.execute("CREATE SCHEMA public")

    def _request(
        self,
        baseline_id: str,
        text: str = "Show candidate evidence",
        *,
        user_id: str = "user_001",
        speaker_ids: tuple[str, ...] = (),
        entity_ids: tuple[str, ...] = (),
        sensitivity_scope: str = "sensitive",
        allow_unclassified: bool = True,
    ) -> RetrievalQueryRequest:
        kinds = {
            "B2": ("atomic",),
            "B3": ("session",),
            "B4": ("atomic", "session"),
        }[baseline_id]
        return RetrievalQueryRequest(
            digest(
                repr(
                    (
                        baseline_id,
                        text,
                        user_id,
                        speaker_ids,
                        entity_ids,
                        sensitivity_scope,
                        allow_unclassified,
                    )
                )
            ),
            user_id,
            text,
            self.cutoff,
            "retrieval_index_v1",
            kinds,
            None,
            speaker_ids,
            entity_ids,
            sensitivity_scope,
            allow_unclassified,
        )

    def _retrieve(self, baseline_id: str, text: str = "Show candidate evidence", **kwargs):
        return self.repository.retrieve(
            self._request(baseline_id, text, **kwargs),
            baseline_id,
            planner_config=self.planner_config,
            baseline_config=self.baseline_config,
        )

    def _execution(self, baseline_id: str, text: str = "What changed over time?"):
        request = self._request(baseline_id, text)
        plan = build_query_plan(request, config=self.planner_config)
        eligibility = RetrievalQueryRepository(self.connection).filter(plan)
        return build_baseline_request(
            request,
            plan,
            eligibility,
            baseline_id,
            config=self.baseline_config,
        )

    def test_b2_b3_b4_kind_channels_user_binding_and_read_only_snapshot(self) -> None:
        observed: dict[str, str] = {}

        class ProbeRepository(RetrievalSearchRepository):
            def _bind_snapshot(probe_self, execution):
                observed["isolation"] = probe_self.connection.execute(
                    "SELECT current_setting('transaction_isolation')"
                ).fetchone()[0]
                observed["read_only"] = probe_self.connection.execute(
                    "SELECT current_setting('transaction_read_only')"
                ).fetchone()[0]
                return super()._bind_snapshot(execution)

        repository = ProbeRepository(self.connection)
        results = {
            baseline: repository.retrieve(
                self._request(baseline),
                baseline,
                planner_config=self.planner_config,
                baseline_config=self.baseline_config,
            )
            for baseline in ("B2", "B3", "B4")
        }
        self.assertEqual(observed, {"isolation": "repeatable read", "read_only": "on"})
        self.assertTrue(results["B2"].accepted)
        self.assertTrue(results["B3"].accepted)
        self.assertTrue(results["B4"].accepted)
        self.assertEqual({item.record_kind for item in results["B2"].accepted}, {"atomic"})
        self.assertEqual({item.record_kind for item in results["B3"].accepted}, {"session"})
        self.assertEqual(
            {item.record_kind for item in results["B4"].accepted},
            {"atomic", "session"},
        )
        b4_channels = {
            hit.channel_name
            for item in results["B4"].accepted
            for hit in item.component_hits
        }
        self.assertTrue({"atomic_vector", "session_vector"}.issubset(b4_channels))
        self.assertIn("WITH eligible_records AS MATERIALIZED", self.repository.LEXICAL_SQL)
        self.assertIn("record.user_id = %s", self.repository.LEXICAL_SQL)
        self.assertIn("record.index_record_id = ANY(%s::text[])", self.repository.VECTOR_SQL)
        self.assertIn("ts_rank_cd(record.search_document, query.value, 32)", self.repository.LEXICAL_SQL)
        self.assertIn("record.embedding <=> %s::vector", self.repository.VECTOR_SQL)

    def test_cross_user_and_prefiltered_records_never_reach_search_channels(self) -> None:
        result = self._retrieve(
            "B4",
            speaker_ids=("speaker_absent",),
            entity_ids=("subject_absent",),
            allow_unclassified=False,
        )
        foreign = {
            row[0]
            for row in self.connection.execute(
                "SELECT index_record_id FROM retrieval_index_records WHERE user_id = 'user_002'"
            ).fetchall()
        }
        visible = {
            *(item.index_record_id for item in result.accepted),
            *(item.index_record_id for item in result.rejected),
        }
        self.assertTrue(visible.isdisjoint(foreign))
        self.assertFalse(result.accepted)
        self.assertTrue(result.rejected)
        self.assertTrue(all(item.stage == "pre_filter" for item in result.rejected))
        self.assertTrue(
            all(
                set(item.reasons).intersection(
                    {"speaker_mismatch", "entity_mismatch", "unclassified_sensitivity"}
                )
                for item in result.rejected
            )
        )

    def test_vector_and_lexical_ties_use_record_id_and_punctuation_skips_channels(self) -> None:
        ids = tuple(
            row[0]
            for row in self.connection.execute(
                """
                SELECT index_record_id
                FROM retrieval_index_records
                WHERE user_id = 'user_001' AND record_kind = 'atomic'
                ORDER BY index_record_id LIMIT 2
                """
            ).fetchall()
        )
        vector = DeterministicTokenHashEmbedder().embed("uniquevectorword")
        vector_value = "[" + ",".join(str(value) for value in vector) + "]"
        self.connection.execute(
            """
            UPDATE retrieval_index_records
            SET content_text = 'uniquevectorword', embedding = %s::vector
            WHERE index_record_id = ANY(%s::text[])
            """,
            (vector_value, list(ids)),
        )
        tied = self._retrieve("B2", "uniquevectorword")
        lexical = sorted(
            (
                hit.rank,
                item.index_record_id,
            )
            for item in tied.accepted
            for hit in item.component_hits
            if hit.channel_name == "atomic_lexical" and item.index_record_id in ids
        )
        vector_hits = sorted(
            (
                hit.rank,
                item.index_record_id,
            )
            for item in tied.accepted
            for hit in item.component_hits
            if hit.channel_name == "atomic_vector" and item.index_record_id in ids
        )
        self.assertEqual([item[1] for item in lexical[:2]], list(ids))
        self.assertEqual([item[1] for item in vector_hits[:2]], list(ids))

        empty = self._retrieve("B3", "!!!")
        self.assertEqual(
            empty.channel_notices,
            ("empty_fts_query", "empty_query_vector"),
        )
        self.assertFalse(empty.accepted)
        self.assertTrue(all(item.reasons == ("no_channel_match",) for item in empty.rejected))

    def test_deleted_or_changed_lineage_fails_precomputed_execution_without_partial_result(self) -> None:
        execution = self._execution("B2")
        record_id = execution.eligibility.eligible_record_ids[0]
        self.connection.execute(
            """
            DELETE FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            """,
            (record_id,),
        )
        with self.assertRaisesRegex(RetrievalSearchRepositoryError, "lineage_incomplete"):
            self.repository.execute(execution)

        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        execute_index_evaluation(
            lambda: psycopg.connect(DATABASE_URL, autocommit=True),
            Path(self.temp.name) / "second-release",
            repo_root=ROOT,
        )
        self.cutoff = self.connection.execute(
            "SELECT max(transaction_as_of) FROM retrieval_index_runs WHERE status = 'succeeded'"
        ).fetchone()[0]
        execution = self._execution("B2")
        record_id = execution.eligibility.eligible_record_ids[0]
        source_id = self.connection.execute(
            """
            SELECT source_id FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            ORDER BY source_order LIMIT 1
            """,
            (record_id,),
        ).fetchone()[0]
        IngestionService(self.connection).delete_source(
            "user_001",
            source_id,
            self.cutoff + timedelta(seconds=1),
        )
        with self.assertRaisesRegex(RetrievalSearchRepositoryError, "stale_snapshot"):
            self.repository.execute(execution)
        rebuilt = self._retrieve("B2")
        self.assertNotIn(record_id, {item.index_record_id for item in rebuilt.accepted})
        self.assertNotIn(record_id, {item.index_record_id for item in rebuilt.rejected})

    def test_unrelated_source_lineage_fails_closed(self) -> None:
        execution = self._execution("B2")
        record_id = execution.eligibility.eligible_record_ids[0]
        extra = self.connection.execute(
            """
            SELECT claim_id, claim_version_id, source_id, span_id, support_type
            FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id <> %s
              AND (claim_id, claim_version_id) NOT IN (
                  SELECT claim_id, claim_version_id
                  FROM retrieval_index_claim_links
                  WHERE user_id = 'user_001' AND index_record_id = %s
              )
            ORDER BY index_record_id, source_order
            LIMIT 1
            """,
            (record_id, record_id),
        ).fetchone()
        source_order = self.connection.execute(
            """
            SELECT max(source_order) + 1
            FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            """,
            (record_id,),
        ).fetchone()[0]
        self.assertIsNotNone(extra)
        self.connection.execute(
            """
            INSERT INTO retrieval_index_source_links (
                user_id, index_record_id, claim_id, claim_version_id,
                source_id, span_id, support_type, source_order
            ) VALUES ('user_001', %s, %s, %s, %s, %s, %s, %s)
            """,
            (record_id, *extra, source_order),
        )

        with self.assertRaisesRegex(RetrievalSearchRepositoryError, "lineage_incomplete"):
            self.repository.execute(execution)

    def test_repeatable_read_hides_a_mid_execution_database_change(self) -> None:
        target_id = self.connection.execute(
            """
            SELECT index_record_id FROM retrieval_index_records
            WHERE user_id = 'user_001' AND record_kind = 'atomic'
            ORDER BY index_record_id LIMIT 1
            """
        ).fetchone()[0]

        class MutatingRepository(RetrievalSearchRepository):
            def _bind_snapshot(mutating_self, execution):
                bound = super()._bind_snapshot(execution)
                other = psycopg.connect(DATABASE_URL, autocommit=True)
                try:
                    other.execute(
                        """
                        UPDATE retrieval_index_records
                        SET content_text = content_text || ' snapshotmutationword'
                        WHERE index_record_id = %s
                        """,
                        (target_id,),
                    )
                finally:
                    other.close()
                return bound

        repository = MutatingRepository(self.connection)
        first = repository.retrieve(
            self._request("B2", "snapshotmutationword"),
            "B2",
            planner_config=self.planner_config,
            baseline_config=self.baseline_config,
        )
        first_target = next(item for item in first.accepted if item.index_record_id == target_id)
        self.assertNotIn(
            "atomic_lexical",
            {hit.channel_name for hit in first_target.component_hits},
        )
        second = self._retrieve("B2", "snapshotmutationword")
        second_target = next(item for item in second.accepted if item.index_record_id == target_id)
        self.assertIn(
            "atomic_lexical",
            {hit.channel_name for hit in second_target.component_hits},
        )

    def test_old_version_and_checked_relation_expansion_stay_inside_eligible_snapshot(self) -> None:
        seed = self.connection.execute(
            """
            SELECT record.index_record_id, record.atomic_claim_id,
                   record.atomic_claim_version_id, record.transaction_from,
                   record.run_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = 'user_001' AND record.record_kind = 'atomic'
            ORDER BY record.index_record_id LIMIT 1
            """
        ).fetchone()
        neighbor = self.connection.execute(
            """
            SELECT record.index_record_id, record.atomic_claim_id,
                   record.atomic_claim_version_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = 'user_001' AND record.record_kind = 'atomic'
              AND record.atomic_claim_id <> %s
            ORDER BY record.index_record_id LIMIT 1
            """,
            (seed[1],),
        ).fetchone()
        clone_id = digest("baseline-old-version-record")
        clone_version = "baseline_old_version"
        self.connection.execute(
            """
            INSERT INTO claim_versions (
                version_id, user_id, claim_id, lifecycle_status,
                transaction_from, transaction_to, belief_confidence,
                valid_from_date, valid_from_timestamp, valid_to_date,
                valid_to_timestamp, time_precision
            )
            SELECT
                %s, user_id, claim_id, 'historical',
                transaction_from - interval '1 day', transaction_from,
                belief_confidence, valid_from_date, valid_from_timestamp,
                valid_to_date, valid_to_timestamp, time_precision
            FROM claim_versions
            WHERE user_id = 'user_001' AND claim_id = %s AND version_id = %s
            """,
            (clone_version, seed[1], seed[2]),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_records (
                index_record_id, user_id, run_id, index_version, record_kind,
                atomic_claim_id, atomic_claim_version_id, subject_id,
                speaker_id, predicate, content_text, content_sha256,
                embedding_version, embedding_dimension, embedding,
                lifecycle_statuses, memory_kind, epistemic_status,
                time_precision, valid_from_date, valid_from_timestamp,
                valid_to_date, valid_to_timestamp, transaction_from,
                transaction_to, sensitivity, contains_sensitive,
                input_snapshot_sha256
            )
            SELECT
                %s, user_id, run_id, index_version, record_kind,
                atomic_claim_id, %s, subject_id, speaker_id, predicate,
                content_text || ' historical', %s, embedding_version,
                embedding_dimension, embedding, '["historical"]'::jsonb,
                memory_kind, epistemic_status, time_precision,
                valid_from_date, valid_from_timestamp, valid_to_date,
                valid_to_timestamp, transaction_from, transaction_to,
                sensitivity, contains_sensitive, input_snapshot_sha256
            FROM retrieval_index_records WHERE index_record_id = %s
            """,
            (clone_id, clone_version, digest("baseline-old-content"), seed[0]),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_claim_links (
                user_id, index_record_id, claim_id, claim_version_id,
                lifecycle_status, claim_order
            ) VALUES ('user_001', %s, %s, %s, 'historical', 0)
            """,
            (clone_id, seed[1], clone_version),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_source_links (
                user_id, index_record_id, claim_id, claim_version_id,
                source_id, span_id, support_type, source_order
            )
            SELECT user_id, %s, claim_id, %s, source_id, span_id,
                   support_type, source_order
            FROM retrieval_index_source_links
            WHERE user_id = 'user_001' AND index_record_id = %s
            """,
            (clone_id, clone_version, seed[0]),
        )
        relation_id = "baseline_relation_001"
        pair = sorted(((seed[1], seed[2]), (neighbor[1], neighbor[2])))
        self.connection.execute(
            """
            INSERT INTO conflict_decisions (
                decision_id, user_id, classifier_version, rule_version,
                pair_id, left_claim_id, right_claim_id, left_version_id,
                right_version_id, input_snapshot_sha256, transaction_as_of,
                matched_rule, label, classified_at
            ) VALUES (
                'baseline_decision_001', 'user_001', 'relation_classifier_v1',
                'conflict_relation_rules_v1', 'baseline_pair_001', %s, %s,
                %s, %s, %s, %s, 'explicit_correction',
                'explicit_correction', %s
            )
            """,
            (pair[0][0], pair[1][0], pair[0][1], pair[1][1], SHA, self.cutoff, self.cutoff),
        )
        self.connection.execute(
            """
            INSERT INTO claim_relations (
                relation_id, user_id, decision_id, classifier_version,
                source_claim_id, target_claim_id, relation_type, confidence,
                input_snapshot_sha256, created_at
            ) VALUES (
                %s, 'user_001', 'baseline_decision_001',
                'relation_classifier_v1', %s, %s, 'corrects', 1, %s, %s
            )
            """,
            (relation_id, seed[1], neighbor[1], SHA, self.cutoff),
        )
        self.connection.execute(
            """
            INSERT INTO retrieval_index_relation_links (
                user_id, index_record_id, relation_id, source_claim_id,
                target_claim_id, relation_type, direction, relation_order
            ) VALUES (
                'user_001', %s, %s, %s, %s, 'corrects', 'outgoing', 0
            )
            """,
            (seed[0], relation_id, seed[1], neighbor[1]),
        )
        content = self.connection.execute(
            "SELECT content_text FROM retrieval_index_records WHERE index_record_id = %s",
            (seed[0],),
        ).fetchone()[0]
        result = self._retrieve("B2", f"What changed over time? {content}")
        paths = [path for item in result.accepted for path in item.expansion_paths]
        self.assertIn(clone_id, {path.index_record_id for path in paths if path.channel_name == "old_version"})
        relation_paths = [path for path in paths if path.channel_name == "checked_relation"]
        self.assertTrue(any(path.relation_id == relation_id for path in relation_paths))
        self.assertTrue(
            all(
                path.index_record_id
                in {decision.index_record_id for decision in result.rejected if decision.stage != "pre_filter"}
                | {item.index_record_id for item in result.accepted}
                for path in paths
            )
        )

        self.connection.execute(
            """
            UPDATE retrieval_index_relation_links
            SET direction = 'symmetric'
            WHERE user_id = 'user_001' AND relation_id = %s
            """,
            (relation_id,),
        )
        symmetric = self._retrieve("B2", f"What changed over time? {content}")
        symmetric_path = next(
            path
            for item in symmetric.accepted
            for path in item.expansion_paths
            if path.relation_id == relation_id
        )
        self.assertEqual(symmetric_path.direction, "symmetric")

        self.connection.execute(
            """
            UPDATE retrieval_index_records
            SET sensitivity = 'sensitive', contains_sensitive = true
            WHERE index_record_id = %s
            """,
            (neighbor[0],),
        )
        filtered = self._retrieve(
            "B2",
            f"What changed over time? {content}",
            sensitivity_scope="standard",
        )
        neighbor_rejection = next(
            item for item in filtered.rejected if item.index_record_id == neighbor[0]
        )
        self.assertEqual(neighbor_rejection.stage, "pre_filter")
        self.assertIn("sensitive_not_authorized", neighbor_rejection.reasons)
        self.assertNotIn(
            neighbor[0],
            {
                path.index_record_id
                for item in filtered.accepted
                for path in item.expansion_paths
            },
        )

    def test_b3_marks_atomic_relation_expansion_not_applicable(self) -> None:
        b3 = self._retrieve("B3", "What changed over time?")
        self.assertIn("relation_expansion_not_applicable", b3.channel_notices)


if __name__ == "__main__":
    unittest.main()
