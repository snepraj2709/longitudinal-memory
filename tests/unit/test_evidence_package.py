from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from answering.contracts import (
    CONFIG_VERSION,
    INPUT_RELEASE_VERSION,
    PACKAGE_VERSION,
    RUNTIME_VERSION,
    SCHEMA_VERSION,
    EvidenceAnchor,
    EvidenceCoverage,
    EvidencePackageBuildRequest,
    EvidencePackageError,
    EvidencePackageFailure,
    EvidenceSpan,
    FrozenEligibilitySnapshot,
    canonical_json_bytes,
    stable_sha256,
)
from answering.evidence_package import build_evidence_package, load_evidence_package_config
from answering.repository import HydratedClaim, HydratedPackageInput, HydratedSource
from retrieval.baseline_contracts import (
    BaselineRetrievalResult,
    RejectedRetrievalItem,
    RetrievedItem,
    RRFContribution,
    SearchChannelHit,
)
from retrieval.query_contracts import QueryPlan, RetrievalQueryRequest


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
SHA = "a" * 64
INPUT_MANIFEST = "b" * 64
INPUT_CHECKPOINT = "c" * 64
INPUT_RESULTS = "d" * 64


def _request(*, clarification: bool = False, dual: bool = False, rejected: bool = False):
    query = RetrievalQueryRequest(
        "evidence_query", "user_001", "What is recorded?",
        datetime(2026, 12, 31, tzinfo=UTC), "retrieval_index_v1",
        ("atomic", "session") if dual else ("atomic",), None, (), (), "sensitive", True,
    )
    plan = QueryPlan(
        "1" * 64, query.query_id, query.user_id, "retrieval_query_planner_v1", "2" * 64,
        query.index_version, "unknown", query.as_of, query.enabled_record_kinds, None, (), (),
        ("candidate", "confirmed", "current", "historical", "disputed", "superseded"),
        True, True, False, False, (), False, clarification, clarification,
    )
    item = _item("record_atomic", "atomic", 1)
    items = (item, _item("record_session", "session", 2)) if dual else (item,)
    rejections = (RejectedRetrievalItem("record_rejected", "pre_filter", ("speaker_mismatch",)),) if rejected else ()
    result = BaselineRetrievalResult(
        "3" * 64, "B4" if dual else "B2", query.query_id, query.user_id,
        plan.plan_id, "snapshot_run", items, rejections, (),
    )
    eligible_ids = tuple(sorted(item.index_record_id for item in items))
    snapshot_value = {
        "plan_id": plan.plan_id,
        "user_id": query.user_id,
        "index_version": query.index_version,
        "snapshot_run_id": result.snapshot_run_id,
        "eligible_record_ids": eligible_ids,
        "pre_filter_rejections": rejections,
    }
    eligibility = FrozenEligibilitySnapshot(
        plan.plan_id, query.user_id, query.index_version, result.snapshot_run_id,
        eligible_ids, rejections, stable_sha256(snapshot_value),
    )
    _, config_hash = load_evidence_package_config(ROOT / "configs/answering/evidence_package_v1.json")
    return EvidencePackageBuildRequest(
        PACKAGE_VERSION, SCHEMA_VERSION, CONFIG_VERSION, config_hash, RUNTIME_VERSION,
        INPUT_RELEASE_VERSION, INPUT_MANIFEST, INPUT_CHECKPOINT, INPUT_RESULTS,
        query, plan, eligibility, result, stable_sha256(result),
    )


def _item(record_id: str, kind: str, rank: int) -> RetrievedItem:
    channel = f"{kind}_vector"
    version = "claim_version_001"
    hit = SearchChannelHit(record_id, kind, channel, rank, "0.500000000000")
    contribution = RRFContribution(channel, rank, "0.016393442623" if rank == 1 else "0.016129032258")
    return RetrievedItem(
        record_id, kind, version if kind == "atomic" else None,
        "summary_001" if kind == "session" else None, SHA, ("candidate",),
        ("claim_001",), (version,), ("source_001",), ("span_001",),
        (hit,), (contribution,), (), contribution.value, rank,
    )


def _hydrated(
    status: str = "candidate",
    *,
    evidence: bool = True,
    dual: bool = False,
    time_precision: str = "unknown",
):
    span = EvidenceSpan(
        "4" * 64, "claim_001", "claim_version_001", "source_001", "span_001",
        "message_001", "speaker_001", "exact quote", 0, 11, "supports", 0.9,
    )
    anchors = (EvidenceAnchor("record_atomic", "atomic", 1, "claim_version_001", None),)
    if dual:
        anchors += (EvidenceAnchor("record_session", "session", 2, None, "summary_001"),)
    date_boundary = date(2026, 1, 1) if time_precision != "unknown" else None
    claim = HydratedClaim(
        "user_001", "claim_001", "claim_version_001", "subject_001", "speaker_001",
        "works_at", {"organization": "Example"}, "positive", "asserted", "episodic",
        status, 0.9, None, date_boundary, None, date_boundary, None, time_precision,
        datetime(2026, 1, 1, tzinfo=UTC), None, "standard", anchors,
        (span,) if evidence else (), (),
    )
    source = HydratedSource(
        "user_001", "source_001", "conversation", "session_001",
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC),
        "5" * 64, "exact quote",
    )
    return HydratedPackageInput((claim,), (source,))


class EvidencePackageUnitTests(unittest.TestCase):
    def test_config_and_all_version_metadata_are_frozen(self) -> None:
        config, digest = load_evidence_package_config(ROOT / "configs/answering/evidence_package_v1.json")
        self.assertEqual(config["schema_version"], SCHEMA_VERSION)
        package = build_evidence_package(_request(), _hydrated())
        self.assertEqual(
            (package.package_version, package.schema_version, package.config_version,
             package.runtime_version, package.input_release_version),
            (PACKAGE_VERSION, SCHEMA_VERSION, CONFIG_VERSION, RUNTIME_VERSION, INPUT_RELEASE_VERSION),
        )
        self.assertEqual(package.config_sha256, digest)

    def test_candidate_has_complete_coverage_but_is_not_promoted(self) -> None:
        package = build_evidence_package(_request(), _hydrated())
        self.assertEqual(package.evidence_coverage.ratio, "1.000000")
        self.assertTrue(package.evidence_coverage.complete)
        self.assertFalse(package.answer_allowed)
        self.assertEqual(package.structural_blockers, ("no_promoted_claims",))
        self.assertEqual(package.relevant_sources, ())
        rejection = package.rejected_evidence[0]
        self.assertEqual((rejection.retrieval_rank, rejection.stage, rejection.reasons), (1, "package_validation", ("candidate_not_promoted",)))

    def test_lifecycle_categories_are_exact_disjoint_and_ordered(self) -> None:
        for status, field in (
            ("current", "current_claims"), ("confirmed", "current_claims"),
            ("historical", "historical_claims"), ("superseded", "historical_claims"),
            ("disputed", "conflicting_claims"),
        ):
            with self.subTest(status=status):
                package = build_evidence_package(_request(), _hydrated(status))
                self.assertEqual(len(getattr(package, field)), 1)
                self.assertTrue(package.answer_allowed)
                self.assertEqual(len(package.relevant_sources), 1)

    def test_atomic_and_session_anchors_deduplicate_one_claim(self) -> None:
        package = build_evidence_package(_request(dual=True), _hydrated("current", dual=True))
        self.assertEqual(len(package.current_claims), 1)
        self.assertEqual(len(package.current_claims[0].retrieval_anchors), 2)

    def test_zero_denominator_has_exact_blocker_order(self) -> None:
        request = _request()
        result = replace(request.retrieval_result, accepted=())
        eligibility_value = {
            "plan_id": request.plan.plan_id, "user_id": request.query.user_id,
            "index_version": request.query.index_version, "snapshot_run_id": result.snapshot_run_id,
            "eligible_record_ids": (), "pre_filter_rejections": (),
        }
        eligibility = FrozenEligibilitySnapshot(
            request.plan.plan_id, request.query.user_id, request.query.index_version,
            result.snapshot_run_id, (), (), stable_sha256(eligibility_value),
        )
        request = replace(request, eligibility=eligibility, retrieval_result=result, retrieval_result_sha256=stable_sha256(result))
        package = build_evidence_package(request, HydratedPackageInput((), ()))
        self.assertIsNone(package.evidence_coverage.ratio)
        self.assertEqual(package.structural_blockers, ("no_retrieved_claims", "no_promoted_claims"))

    def test_promoted_missing_evidence_is_incomplete(self) -> None:
        package = build_evidence_package(_request(), _hydrated("current", evidence=False))
        self.assertEqual(package.evidence_coverage.ratio, "0.000000")
        self.assertEqual(package.structural_blockers, ("incomplete_evidence",))

    def test_candidate_missing_evidence_is_rejected_fail_closed(self) -> None:
        with self.assertRaisesRegex(EvidencePackageError, "candidate evidence is incomplete"):
            build_evidence_package(_request(), _hydrated(evidence=False))

    def test_clarification_is_last_blocker(self) -> None:
        package = build_evidence_package(_request(clarification=True), _hydrated("current", evidence=False))
        self.assertEqual(package.structural_blockers, ("incomplete_evidence", "clarification_required"))

    def test_serialization_ids_and_rejections_are_byte_stable(self) -> None:
        first = build_evidence_package(_request(rejected=True), _hydrated())
        second = build_evidence_package(_request(rejected=True), _hydrated())
        self.assertEqual(first.package_id, second.package_id)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        carried = first.rejected_evidence[0]
        self.assertEqual((carried.retrieval_rank, carried.stage), (None, "pre_filter"))
        self.assertNotIn(b"raw_content", canonical_json_bytes(first))

    def test_metadata_and_result_drift_fail(self) -> None:
        with self.assertRaises(EvidencePackageError):
            replace(_request(), runtime_version="changed")
        with self.assertRaises(EvidencePackageError):
            replace(_request(), retrieval_result_sha256="e" * 64)

    def test_time_and_epistemic_enums_are_exact(self) -> None:
        build_evidence_package(_request(), _hydrated("current", time_precision="day"))
        claim = _hydrated("current").claims[0]
        with self.assertRaises(EvidencePackageError):
            build_evidence_package(_request(), replace(_hydrated("current"), claims=(replace(claim, epistemic_status="maybe"),)))
        with self.assertRaises(EvidencePackageError):
            build_evidence_package(_request(), replace(_hydrated("current"), claims=(replace(claim, time_precision="day"),)))

    def test_coverage_recomputes_exact_ratio(self) -> None:
        with self.assertRaises(EvidencePackageError):
            EvidenceCoverage(2, 1, 0, 0, "0.500001", None, False)

    def test_failures_are_sanitized(self) -> None:
        EvidencePackageFailure("f" * 64, "user_001", "query_001", "B2", "stale_result", "source")
        with self.assertRaises(EvidencePackageError):
            EvidencePackageFailure("f" * 64, "user_001", "query_001", "B2", "raw query", "source")

    def test_unknown_config_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = json.loads((ROOT / "configs/answering/evidence_package_v1.json").read_text())
            value["extra"] = True
            path.write_text(json.dumps(value))
            with self.assertRaises(EvidencePackageError):
                load_evidence_package_config(path)


if __name__ == "__main__":
    unittest.main()
