from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

import conflicts.candidates as candidate_module
from conflicts.candidates import (
    CandidateRequest,
    ConflictCandidateError,
    ConflictCandidateService,
)
from conflicts.evaluation import ConflictEvaluationError, execute_candidate_evaluation
from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
)
from storage.migrations import apply_migrations

if psycopg is not None:
    from ingestion.service import IngestionService
    from storage.repository import StorageRepository


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


@unittest.skipUnless(
    psycopg is not None and DATABASE_URL,
    "psycopg and STORAGE_DATABASE_URL are required for candidate integration tests",
)
class ConflictCandidateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.reset_database()
        self.repository = StorageRepository(self.connection)
        for user_id in ("user_001", "user_002"):
            self.repository.insert_user(MemoryUser(user_id, BASE))
        self.repository.insert_extraction_version(
            ExtractionVersionRecord(
                "candidate_test_v1", "deterministic", "none", "0" * 64,
                "candidate_test", "0" * 64, "predicate_registry_v2",
                "0" * 64, "0" * 64, BASE,
            )
        )

    def reset_database(self) -> None:
        self.connection.execute("DROP SCHEMA public CASCADE")
        self.connection.execute("CREATE SCHEMA public")
        apply_migrations(self.connection, ROOT / "migrations")

    def _source(
        self,
        source_id: str,
        *,
        user_id: str = "user_001",
        ingested_at: datetime = BASE,
    ) -> str:
        span_id = f"span_{source_id}"
        content = f"source-backed evidence for {source_id}"
        self.repository.insert_source_event(
            SourceEventRecord(
                source_id, user_id, "conversation", None, f"key_{user_id}_{source_id}",
                ingested_at, ingested_at, content, [user_id], {},
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
        self.repository.insert_source_span(
            SourceSpanRecord(span_id, user_id, source_id, "message_1", user_id, content)
        )
        return span_id

    def _claim(
        self,
        claim_id: str,
        span_ids: tuple[str, ...],
        *,
        user_id: str = "user_001",
        status: str = "candidate",
        version_from: datetime = BASE,
        predicate: str = "employer",
        object_json: object = "acme",
    ) -> None:
        self.repository.insert_claim(
            ClaimRecord(
                claim_id, user_id, user_id, user_id, predicate,
                "predicate_registry_v2", object_json, "positive", "asserted",
                date(2026, 1, 1), None, date(2026, 12, 31), None, "day",
                1, "durative", "standard", "candidate_test_v1",
            )
        )
        self.repository.insert_claim_version(
            ClaimVersionRecord(
                f"version_{claim_id}", user_id, claim_id, status, version_from,
                belief_confidence=None, valid_from_date=date(2026, 1, 1),
                valid_to_date=date(2026, 12, 31), time_precision="day",
            )
        )
        for span_id in span_ids:
            self.repository.insert_evidence_link(
                EvidenceLinkRecord(user_id, claim_id, span_id, "supports", 1)
            )

    def test_visibility_applies_source_ingestion_and_transaction_as_of(self) -> None:
        incoming_span = self._source("incoming")
        late_span = self._source(
            "late", ingested_at=datetime(2026, 3, 1, tzinfo=UTC)
        )
        future_span = self._source("future")
        self._claim("claim_incoming", (incoming_span,))
        self._claim("claim_late_source", (late_span,))
        self._claim(
            "claim_future_version",
            (future_span,),
            version_from=datetime(2026, 3, 1, tzinfo=UTC),
        )
        service = ConflictCandidateService(self.connection, repo_root=ROOT)
        early = service.generate(
            CandidateRequest(
                "user_001", datetime(2026, 2, 1, tzinfo=UTC), ("claim_incoming",)
            )
        )
        self.assertEqual(early, ())
        later = service.generate(
            CandidateRequest(
                "user_001", datetime(2026, 4, 1, tzinfo=UTC), ("claim_incoming",)
            )
        )
        self.assertEqual(
            {(pair.left_claim_id, pair.right_claim_id) for pair in later},
            {
                ("claim_future_version", "claim_incoming"),
                ("claim_incoming", "claim_late_source"),
            },
        )

    def test_all_eligible_statuses_and_user_filters_run_before_features(self) -> None:
        incoming_span = self._source("incoming")
        self._claim("claim_incoming", (incoming_span,))
        eligible_ids: set[str] = set()
        for status in ("candidate", "current", "historical", "disputed", "superseded"):
            claim_id = f"claim_{status}"
            eligible_ids.add(claim_id)
            self._claim(claim_id, (self._source(status),), status=status)
        self._claim(
            "claim_excluded", (self._source("excluded"),), status="excluded"
        )
        self._claim(
            "claim_cross_user",
            (self._source("cross", user_id="user_002"),),
            user_id="user_002",
        )
        service = ConflictCandidateService(self.connection, repo_root=ROOT)
        with mock.patch.object(
            candidate_module,
            "_candidate_signals",
            wraps=candidate_module._candidate_signals,
        ) as signals:
            pairs = service.generate(
                CandidateRequest(
                    "user_001", datetime(2026, 6, 1, tzinfo=UTC), ("claim_incoming",)
                )
            )
        paired = {
            next(iter({pair.left_claim_id, pair.right_claim_id} - {"claim_incoming"}))
            for pair in pairs
        }
        self.assertEqual(paired, eligible_ids)
        self.assertEqual(signals.call_count, len(eligible_ids))
        self.assertTrue(
            all(
                left.claim.user_id == right.claim.user_id == "user_001"
                for _, left, right, _ in (call.args for call in signals.call_args_list)
            )
        )

    def test_cross_user_incoming_fails_without_feature_work(self) -> None:
        self._claim(
            "claim_cross", (self._source("cross", user_id="user_002"),),
            user_id="user_002",
        )
        service = ConflictCandidateService(self.connection, repo_root=ROOT)
        with mock.patch.object(candidate_module, "_candidate_signals") as signals:
            with self.assertRaisesRegex(
                ConflictCandidateError, "not visible and supported"
            ):
                service.generate(
                    CandidateRequest(
                        "user_001", datetime(2026, 6, 1, tzinfo=UTC),
                        ("claim_cross",),
                    )
                )
        signals.assert_not_called()

    def test_deletion_preserves_remaining_evidence_then_removes_unsupported_pair(self) -> None:
        incoming_span = self._source("incoming")
        evidence_a = self._source("evidence_a")
        evidence_b = self._source("evidence_b")
        self._claim("claim_incoming", (incoming_span,))
        self._claim("claim_other", (evidence_a, evidence_b))
        service = ConflictCandidateService(self.connection, repo_root=ROOT)
        request = CandidateRequest(
            "user_001", datetime(2026, 6, 1, tzinfo=UTC), ("claim_incoming",)
        )
        self.assertEqual(
            service.generate(request)[0].source_ids,
            ("evidence_a", "evidence_b", "incoming"),
        )
        ingestion = IngestionService(self.connection)
        ingestion.delete_source(
            "user_001", "evidence_a", datetime(2026, 6, 2, tzinfo=UTC)
        )
        self.assertEqual(
            service.generate(request)[0].source_ids,
            ("evidence_b", "incoming"),
        )
        ingestion.delete_source(
            "user_001", "evidence_b", datetime(2026, 6, 3, tzinfo=UTC)
        )
        self.assertEqual(service.generate(request), ())

    def test_two_clean_runs_are_byte_identical_and_have_no_relation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self.reset_database()
            first_manifest = execute_candidate_evaluation(
                self.connection, repo_root=ROOT, result_root=first
            )
            first_artifacts = {
                name: (first / name).read_bytes()
                for name in (
                    "predictions.jsonl", "failures.jsonl", "scores.json",
                    "run.json", "findings.md", "manifest.json",
                )
            }
            predictions = [
                json.loads(line)
                for line in (first / "predictions.jsonl").read_text().splitlines()
            ]
            scores = json.loads((first / "scores.json").read_text())
            self.assertEqual(len(predictions), 8)
            self.assertEqual((first / "failures.jsonl").read_text(), "")
            self.assertEqual(scores["candidate_recall"]["value"], 1.0)
            self.assertEqual(scores["required_pair_count"], 8)
            self.assertEqual(scores["cross_user_candidate_count"], 0)
            self.assertTrue(
                all(
                    pair["source_ids"] and pair["signals"]
                    for prediction in predictions
                    for pair in prediction["pairs"]
                )
            )
            self.assertFalse(
                any(
                    "relation_label" in pair or "conflict_type" in pair
                    for prediction in predictions
                    for pair in prediction["pairs"]
                )
            )
            self.assertEqual(
                self.connection.execute("SELECT count(*) FROM claims").fetchone()[0], 0
            )
            self.reset_database()
            second_manifest = execute_candidate_evaluation(
                self.connection, repo_root=ROOT, result_root=second
            )
            self.assertEqual(first_manifest, second_manifest)
            for name, payload in first_artifacts.items():
                self.assertEqual((second / name).read_bytes(), payload, name)

    def test_result_directory_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            output.mkdir()
            (output / "marker").write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ConflictEvaluationError, "must be empty"):
                execute_candidate_evaluation(
                    self.connection, repo_root=ROOT, result_root=output
                )


if __name__ == "__main__":
    unittest.main()
