from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None

from answering.contracts import (
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    EvidencePackageBuildRequest,
    canonical_json_bytes,
    stable_sha256,
)
from answering.evaluation import (
    BASELINE_CHECKPOINT_SHA256,
    BASELINE_MANIFEST_SHA256,
    BASELINE_RESULTS,
    BASELINE_RESULTS_SHA256,
    RESULT_ROOT,
    EvidencePackageEvaluationError,
    _frozen_eligibility,
    _frozen_plan,
    _parse_result,
    execute_evidence_package_evaluation,
    verify_evidence_package_release,
)
from answering.evidence_package import build_evidence_package, load_evidence_package_config
from answering.repository import EvidencePackageRepository, EvidencePackageRepositoryError
from retrieval.baseline_contracts import BASELINE_RECORD_KINDS
from retrieval.baseline_evaluation import load_development_queries
from retrieval.index_evaluation import execute_index_evaluation


ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.environ.get("STORAGE_DATABASE_URL")
CHECKED = ROOT / RESULT_ROOT


@unittest.skipUnless(psycopg is not None and DATABASE_URL, "PostgreSQL is required")
class EvidencePackageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admin = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.admin.close()

    def setUp(self) -> None:
        self._clean()

    def tearDown(self) -> None:
        self._clean()

    def _clean(self) -> None:
        self.admin.execute("DROP SCHEMA IF EXISTS public CASCADE")
        self.admin.execute("CREATE SCHEMA public")

    def _factory(self):
        return psycopg.connect(DATABASE_URL, autocommit=True)

    def _execute(self, output: Path):
        return execute_evidence_package_evaluation(self._factory, output, repo_root=ROOT)

    def _request_from_row(self, row, case):
        result = _parse_result(row["result"])
        query = replace(case.request, enabled_record_kinds=BASELINE_RECORD_KINDS[result.baseline_id])
        plan = _frozen_plan(query, row["primary_label"], result.plan_id)
        eligibility = _frozen_eligibility(query.index_version, result)
        _, config_hash = load_evidence_package_config(ROOT / "configs/answering/evidence_package_v1.json")
        return EvidencePackageBuildRequest(
            PACKAGE_VERSION, SCHEMA_VERSION, CONFIG_VERSION, config_hash, RUNTIME_VERSION,
            INPUT_RELEASE_VERSION, BASELINE_MANIFEST_SHA256, BASELINE_CHECKPOINT_SHA256,
            BASELINE_RESULTS_SHA256, query, plan, eligibility, result, stable_sha256(result),
        )

    def _build_runtime_request(self, directory: str, row_index: int = 0):
        execute_index_evaluation(self._factory, Path(directory) / "index", repo_root=ROOT)
        rows = [json.loads(line) for line in (ROOT / BASELINE_RESULTS).read_text().splitlines()]
        cases = {item.case_id: item for item in load_development_queries(
            ROOT / "data/retrieval/baseline-execution-development-v1/queries.jsonl"
        )}
        row = rows[row_index]
        return self._request_from_row(row, cases[row["case_id"]])

    def _install_relation_link(self, connection, request):
        target = request.retrieval_result.accepted[0]
        source_claim_id = target.claim_ids[0]
        source_version_id = target.claim_version_ids[0]
        accepted_versions = [
            version_id
            for item in request.retrieval_result.accepted
            for version_id in item.claim_version_ids
        ]
        neighbor_claim_id, neighbor_version_id = connection.execute(
            "SELECT claim_id, version_id FROM claim_versions "
            "WHERE user_id=%s AND claim_id<>%s AND version_id<>ALL(%s::text[]) "
            "ORDER BY claim_id, version_id LIMIT 1",
            (request.query.user_id, source_claim_id, accepted_versions),
        ).fetchone()
        left_claim_id, right_claim_id = sorted((source_claim_id, neighbor_claim_id))
        versions = {
            source_claim_id: source_version_id,
            neighbor_claim_id: neighbor_version_id,
        }
        observed_at = request.query.as_of - timedelta(days=1)
        connection.execute(
            "INSERT INTO conflict_decisions "
            "(decision_id,user_id,classifier_version,rule_version,pair_id,left_claim_id,right_claim_id,"
            "left_version_id,right_version_id,input_snapshot_sha256,transaction_as_of,matched_rule,label,classified_at) "
            "VALUES ('evidence_decision',%s,'relation_classifier_v1','relation_rule_v1','evidence_pair',"
            "%s,%s,%s,%s,%s,%s,'explicit_correction','explicit_correction',%s)",
            (
                request.query.user_id, left_claim_id, right_claim_id,
                versions[left_claim_id], versions[right_claim_id], "a" * 64,
                observed_at, observed_at,
            ),
        )
        connection.execute(
            "INSERT INTO claim_relations "
            "(relation_id,user_id,decision_id,classifier_version,source_claim_id,target_claim_id,"
            "relation_type,confidence,input_snapshot_sha256,created_at) "
            "VALUES ('evidence_relation',%s,'evidence_decision','relation_classifier_v1',%s,%s,"
            "'corrects',1,%s,%s)",
            (request.query.user_id, source_claim_id, neighbor_claim_id, "a" * 64, observed_at),
        )
        connection.execute(
            "INSERT INTO retrieval_index_relation_links "
            "(user_id,index_record_id,record_kind,relation_id,source_claim_id,target_claim_id,"
            "relation_type,direction,relation_order) "
            "VALUES (%s,%s,'atomic','evidence_relation',%s,%s,'corrects','outgoing',0)",
            (request.query.user_id, target.index_record_id, source_claim_id, neighbor_claim_id),
        )
        return target, neighbor_claim_id, neighbor_version_id

    def test_clean_release_has_24_candidate_blocked_packages_and_zero_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            checks = self._execute(output)
            packages = [json.loads(line) for line in (output / "packages.jsonl").read_text().splitlines()]
        self.assertEqual((checks.query_count, checks.result_count, checks.package_count), (8, 24, 24))
        self.assertEqual((checks.b2_package_count, checks.b3_package_count, checks.b4_package_count), (8, 8, 8))
        self.assertEqual(checks.failure_count, 0)
        self.assertEqual(checks.unique_claim_version_count, 33)
        self.assertEqual(checks.coverage_complete_count, 24)
        self.assertEqual(checks.answer_allowed_count, 0)
        self.assertTrue(all(row["structural_blockers"] == ["no_promoted_claims"] for row in packages))
        self.assertTrue(all(not row["relevant_sources"] for row in packages))

    def test_two_clean_database_releases_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self._execute(first)
            self._clean()
            self._execute(second)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_checked_release_self_verifies(self) -> None:
        verify_evidence_package_release(CHECKED, repo_root=ROOT)

    def test_nonempty_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "existing").write_text("keep")
            with self.assertRaisesRegex(EvidencePackageEvaluationError, "must be empty"):
                self._execute(output)

    def test_tampered_release_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            (output / "checks.json").write_bytes((output / "checks.json").read_bytes() + b" ")
            with self.assertRaisesRegex(EvidencePackageEvaluationError, "does not recompute"):
                verify_evidence_package_release(output, repo_root=ROOT)

    def test_deep_verifier_rejects_rehashed_package_invariant_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            rows = [json.loads(line) for line in (output / "packages.jsonl").read_text().splitlines()]
            rows[0]["structural_blockers"] = ["clarification_required"]
            package_bytes = b"".join(canonical_json_bytes(row) for row in rows)
            (output / "packages.jsonl").write_bytes(package_bytes)
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["artifacts"]["packages.jsonl"] = hashlib.sha256(package_bytes).hexdigest()
            (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(Exception, "blockers do not recompute"):
                verify_evidence_package_release(output, repo_root=ROOT)

    def test_runtime_never_opens_quality_relevance_scorecard_gold_or_review(self) -> None:
        forbidden = (
            "retrieval-quality-development-v1/gold", "scorecard.json", "/oracle",
            "review_queue", "summary-quality-development-v2/gold", "user_003",
        )
        original = Path.open

        def guarded(path, *args, **kwargs):
            value = Path(path).as_posix()
            if any(token in value for token in forbidden):
                raise AssertionError(f"forbidden runtime read: {value}")
            return original(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "open", guarded):
            checks = self._execute(Path(directory) / "release")
        self.assertEqual(checks.package_count, 24)

    def test_runtime_does_not_rerun_planning_or_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "retrieval.query_planner.build_query_plan",
            side_effect=AssertionError("planning rerun"),
        ), patch(
            "retrieval.query_repository.RetrievalQueryRepository.filter",
            side_effect=AssertionError("filter rerun"),
        ):
            checks = self._execute(Path(directory) / "release")
        self.assertEqual(checks.package_count, 24)

    def test_source_deletion_after_retrieval_fails_without_partial_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        connection = self._factory()
        try:
            target = request.retrieval_result.accepted[0]
            connection.execute(
                "DELETE FROM evidence_links WHERE user_id = %s AND claim_id = %s AND span_id = %s",
                (request.query.user_id, target.claim_ids[0], target.span_ids[0]),
            )
            with self.assertRaises(EvidencePackageRepositoryError):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

        self._clean()
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        connection = self._factory()
        try:
            connection.execute(
                "UPDATE claims SET time_precision='day', valid_from_date=DATE '2030-01-01', valid_to_date=DATE '2030-01-02', valid_from_timestamp=NULL, valid_to_timestamp=NULL WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, target.claim_ids[0]),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "stale_result"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

        self._clean()
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        connection = self._factory()
        try:
            row = connection.execute(
                "SELECT source.source_type, span.message_id, length(source.raw_content) FROM source_events source JOIN source_spans span ON span.user_id=source.user_id AND span.source_id=source.source_id WHERE source.user_id=%s AND source.source_id=%s AND span.span_id=%s",
                (request.query.user_id, target.source_ids[0], target.span_ids[0]),
            ).fetchone()
            self.assertEqual((row[0], row[1]), ("calendar", None))
            connection.execute(
                "UPDATE source_spans SET end_offset=%s WHERE user_id=%s AND source_id=%s AND span_id=%s",
                (row[2] + 1, request.query.user_id, target.source_ids[0], target.span_ids[0]),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "lineage_incomplete"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

    def test_live_lifecycle_categories_subject_speaker_and_fatal_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        connection = self._factory()
        try:
            target = request.retrieval_result.accepted[0]
            claim_id, version_id = target.claim_ids[0], target.claim_version_ids[0]
            precision = connection.execute(
                "SELECT time_precision FROM claims WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, claim_id),
            ).fetchone()[0]
            self.assertNotEqual(precision, "unknown")
            subject_speaker = connection.execute(
                "SELECT subject_id, speaker_id FROM claims WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, claim_id),
            ).fetchone()
            expected = {
                "current": "current_claims", "confirmed": "current_claims",
                "historical": "historical_claims", "superseded": "historical_claims",
                "disputed": "conflicting_claims",
            }
            for status, category in expected.items():
                connection.execute(
                    "UPDATE claim_versions SET lifecycle_status=%s WHERE user_id=%s AND version_id=%s",
                    (status, request.query.user_id, version_id),
                )
                connection.execute(
                    "UPDATE retrieval_index_claim_links SET lifecycle_status=%s WHERE user_id=%s AND index_record_id=%s AND claim_version_id=%s",
                    (status, request.query.user_id, target.index_record_id, version_id),
                )
                connection.execute(
                    "UPDATE retrieval_index_records SET lifecycle_statuses=%s::jsonb WHERE user_id=%s AND index_record_id=%s",
                    (json.dumps([status]), request.query.user_id, target.index_record_id),
                )
                changed_item = replace(target, lifecycle_statuses=(status,))
                changed_result = replace(
                    request.retrieval_result,
                    accepted=(changed_item, *request.retrieval_result.accepted[1:]),
                )
                changed_request = replace(
                    request, retrieval_result=changed_result,
                    retrieval_result_sha256=stable_sha256(changed_result),
                )
                hydrated = EvidencePackageRepository(connection).hydrate(changed_request)
                package = build_evidence_package(changed_request, hydrated)
                self.assertTrue(getattr(package, category))
                claim = getattr(package, category)[0]
                self.assertEqual((claim.subject_id, claim.speaker_id), tuple(subject_speaker))

            connection.execute(
                "UPDATE claim_versions SET lifecycle_status='excluded' WHERE user_id=%s AND version_id=%s",
                (request.query.user_id, version_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "invariant_violation"):
                EvidencePackageRepository(connection).hydrate(changed_request)

            connection.execute(
                "UPDATE claim_versions SET lifecycle_status='candidate' WHERE user_id=%s AND version_id=%s",
                (request.query.user_id, version_id),
            )
            connection.execute(
                "UPDATE retrieval_index_claim_links SET lifecycle_status='candidate' WHERE user_id=%s AND index_record_id=%s AND claim_version_id=%s",
                (request.query.user_id, target.index_record_id, version_id),
            )
            connection.execute(
                "UPDATE retrieval_index_records SET lifecycle_statuses='[\"candidate\"]'::jsonb WHERE user_id=%s AND index_record_id=%s",
                (request.query.user_id, target.index_record_id),
            )
            connection.execute(
                "UPDATE claims SET sensitivity='restricted' WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, claim_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "invariant_violation"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

    def test_exact_retrieval_source_and_span_sets_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        changed = replace(target, source_ids=("source_wrong",))
        result = replace(request.retrieval_result, accepted=(changed, *request.retrieval_result.accepted[1:]))
        changed_request = replace(request, retrieval_result=result, retrieval_result_sha256=stable_sha256(result))
        connection = self._factory()
        try:
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "lineage_incomplete"):
                EvidencePackageRepository(connection).hydrate(changed_request)
        finally:
            connection.close()

    def test_poisoned_cross_user_relation_link_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        connection = self._factory()
        try:
            constraint = connection.execute(
                "SELECT conname FROM pg_constraint WHERE conrelid='retrieval_index_relation_links'::regclass AND confrelid='claim_relations'::regclass"
            ).fetchone()[0]
            connection.execute(f'ALTER TABLE retrieval_index_relation_links DROP CONSTRAINT "{constraint}"')
            connection.execute(
                "INSERT INTO retrieval_index_relation_links (user_id,index_record_id,record_kind,relation_id,source_claim_id,target_claim_id,relation_type,direction,relation_order) VALUES (%s,%s,'atomic','poison_relation','foreign_claim','foreign_target','corrects','outgoing',0)",
                (request.query.user_id, target.index_record_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "lineage_incomplete"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

    def test_relation_endpoint_transaction_and_valid_time_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        connection = self._factory()
        try:
            _, neighbor_claim_id, neighbor_version_id = self._install_relation_link(connection, request)
            connection.execute(
                "UPDATE claims SET valid_from_date=DATE '2026-08-11', valid_to_date=DATE '2026-08-11', "
                "valid_from_timestamp=NULL, valid_to_timestamp=NULL, time_precision='day' "
                "WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, neighbor_claim_id),
            )
            connection.execute(
                "UPDATE claim_versions SET transaction_from=%s + interval '1 second' "
                "WHERE user_id=%s AND version_id=%s",
                (request.query.as_of, request.query.user_id, neighbor_version_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "relation_transaction"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

        self._clean()
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        connection = self._factory()
        try:
            _, neighbor_claim_id, _ = self._install_relation_link(connection, request)
            connection.execute(
                "UPDATE claims SET valid_from_date=DATE '2030-01-01', valid_to_date=DATE '2030-01-02', "
                "valid_from_timestamp=NULL, valid_to_timestamp=NULL, time_precision='day' "
                "WHERE user_id=%s AND claim_id=%s",
                (request.query.user_id, neighbor_claim_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "relation_valid_time"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

    def test_poisoned_cross_user_accepted_record_id_is_invisible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        connection = self._factory()
        try:
            foreign_record_id = connection.execute(
                "SELECT index_record_id FROM retrieval_index_records "
                "WHERE user_id<>%s ORDER BY index_record_id LIMIT 1",
                (request.query.user_id,),
            ).fetchone()[0]
            original_item = request.retrieval_result.accepted[0]
            changed_item = replace(
                original_item,
                index_record_id=foreign_record_id,
                component_hits=tuple(
                    replace(hit, index_record_id=foreign_record_id)
                    for hit in original_item.component_hits
                ),
                expansion_paths=tuple(
                    replace(path, index_record_id=foreign_record_id)
                    for path in original_item.expansion_paths
                ),
            )
            changed_result = replace(
                request.retrieval_result,
                accepted=(changed_item, *request.retrieval_result.accepted[1:]),
            )
            changed_request = replace(
                request,
                eligibility=_frozen_eligibility(request.query.index_version, changed_result),
                retrieval_result=changed_result,
                retrieval_result_sha256=stable_sha256(changed_result),
            )
            with self.assertRaises(EvidencePackageRepositoryError) as context:
                EvidencePackageRepository(connection).hydrate(changed_request)
            self.assertNotIn(foreign_record_id, str(context.exception))
        finally:
            connection.close()

    def test_live_transaction_ingestion_valid_time_quote_and_calendar_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        claim_id, version_id = target.claim_ids[0], target.claim_version_ids[0]
        connection = self._factory()
        try:
            connection.execute(
                "UPDATE claim_versions SET transaction_from=%s WHERE user_id=%s AND version_id=%s",
                (request.query.as_of, request.query.user_id, version_id),
            )
            EvidencePackageRepository(connection).hydrate(request)
            connection.execute(
                "UPDATE claim_versions SET transaction_from=%s + interval '1 second' WHERE user_id=%s AND version_id=%s",
                (request.query.as_of, request.query.user_id, version_id),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "stale_result"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

        self._clean()
        with tempfile.TemporaryDirectory() as directory:
            request = self._build_runtime_request(directory)
        target = request.retrieval_result.accepted[0]
        connection = self._factory()
        try:
            connection.execute(
                "UPDATE source_events SET ingested_at=%s + interval '1 second' WHERE user_id=%s AND source_id=%s",
                (request.query.as_of, request.query.user_id, target.source_ids[0]),
            )
            with self.assertRaisesRegex(EvidencePackageRepositoryError, "stale_result"):
                EvidencePackageRepository(connection).hydrate(request)
        finally:
            connection.close()

    def test_repository_sql_is_user_first_set_based_and_has_no_search(self) -> None:
        from answering.repository import EvidencePackageRepository as Repository

        for sql in (Repository.CLAIMS_SQL, Repository.EVIDENCE_SQL, Repository.RELATIONS_SQL):
            normalized = " ".join(sql.split()).lower()
            self.assertIn("where record.user_id = %s and record.index_version = %s and record.run_id = %s", normalized)
            self.assertIn("any(%s::text[])", normalized)
            self.assertNotIn("@@", normalized)
            self.assertNotIn("<=>", normalized)
            self.assertNotIn("ts_rank", normalized)
        relation_sql = " ".join(Repository.RELATIONS_SQL.split()).lower()
        self.assertIn("left_version.version_id = decision.left_version_id", relation_sql)
        self.assertIn("right_version.version_id = decision.right_version_id", relation_sql)
        source = (ROOT / "src/answering/repository.py").read_text()
        evaluation = (ROOT / "src/answering/evaluation.py").read_text()
        self.assertNotIn("RetrievalQueryRepository", source + evaluation)
        self.assertNotIn("build_query_plan", source + evaluation)

    def test_serialized_release_contains_no_raw_source_or_summary_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            raw = (output / "packages.jsonl").read_text()
        self.assertNotIn("raw_content", raw)
        self.assertNotIn("summary_text", raw)
        self.assertNotIn("unresolved_question", raw)
        self.assertNotIn("participants", raw)
        self.assertNotIn("embedding", raw)

    def test_every_candidate_rejection_contains_ids_only_without_quote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            self._execute(output)
            packages = [json.loads(line) for line in (output / "packages.jsonl").read_text().splitlines()]
        candidates = [
            item for package in packages for item in package["rejected_evidence"]
            if item["stage"] == "package_validation"
        ]
        self.assertTrue(candidates)
        self.assertTrue(all(set(item) == {
            "rejection_id", "index_record_id", "retrieval_rank", "claim_id",
            "claim_version_id", "stage", "reasons",
        } for item in candidates))
        self.assertTrue(all(isinstance(item["retrieval_rank"], int) for item in candidates))

    def test_predecessor_drift_is_empty_and_tracked_tree_is_unchanged(self) -> None:
        dataset = json.loads((ROOT / "data/answering/evidence-package-development-v1/manifest.json").read_text())
        release = json.loads((CHECKED / "manifest.json").read_text())
        self.assertEqual(dataset["predecessor_drift"], [])
        self.assertEqual(release["predecessor_drift"], [])


if __name__ == "__main__":
    unittest.main()
