from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from retrieval.baseline_contracts import (
    BaselineRetrievalError,
    BaselineRetrievalFailure,
    BaselineRetrievalResult,
    ExpansionPath,
    RejectedRetrievalItem,
    RetrievedItem,
    RRFContribution,
    SearchChannelHit,
)
from retrieval.baselines import (
    AtomicExpansionSeed,
    FusedCandidate,
    OldVersionNeighbor,
    RelationNeighbor,
    RetrievalRecordMetadata,
    build_baseline_request,
    build_old_version_paths,
    build_relation_paths,
    build_rejections,
    build_result,
    fixed_decimal,
    fuse_channels,
    load_baseline_config,
    rerank_candidates,
    rrf_contribution,
    serialize_result,
)
from retrieval.query_contracts import (
    EligibilityDecision,
    EligibilityResult,
    RetrievalQueryRequest,
)
from retrieval.query_planner import build_query_plan, load_query_planner_config
from retrieval.search_repository import RetrievalSearchRepository


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
AS_OF = datetime(2026, 8, 10, 12, tzinfo=UTC)
SHA = "a" * 64


def execution(baseline_id: str = "B4", text: str = "What changed over time?"):
    config = load_baseline_config(ROOT / "configs/retrieval/baseline_v1.json")
    planner = load_query_planner_config(ROOT / "configs/retrieval/query_planner_v1.json")
    kinds = {"B2": ("atomic",), "B3": ("session",), "B4": ("atomic", "session")}[baseline_id]
    request = RetrievalQueryRequest(
        "query_baseline_001",
        "user_001",
        text,
        AS_OF,
        "retrieval_index_v1",
        kinds,
        None,
        (),
        (),
        "sensitive",
        True,
    )
    plan = build_query_plan(request, config=planner)
    decisions = (
        EligibilityDecision("record_atomic", True, (), ("candidate",), None, True),
        EligibilityDecision(
            "record_rejected",
            False,
            ("unknown_valid_time",),
            ("candidate",),
            None,
            True,
        ),
    )
    eligibility = EligibilityResult(
        plan.plan_id,
        request.user_id,
        request.index_version,
        "run_001",
        decisions,
    )
    return build_baseline_request(request, plan, eligibility, baseline_id, config=config)


def candidate(
    record_id: str,
    kind: str,
    lifecycle: tuple[str, ...],
    score: str = "0.016393442623",
) -> FusedCandidate:
    channel = f"{kind}_vector"
    hit = SearchChannelHit(record_id, kind, channel, 1, "0.500000000000")
    return FusedCandidate(
        record_id,
        kind,
        lifecycle,
        (hit,),
        (),
        (RRFContribution(channel, 1, "0.016393442623"),),
        score,
    )


def metadata(record_id: str = "record_atomic", kind: str = "atomic") -> RetrievalRecordMetadata:
    return RetrievalRecordMetadata(
        record_id,
        kind,
        "version_001" if kind == "atomic" else None,
        "summary_001" if kind == "session" else None,
        SHA,
        ("candidate",),
        ("claim_001",),
        ("version_001",),
        ("source_001",),
        ("span_001",),
    )


class BaselineConfigAndRequestTests(unittest.TestCase):
    def test_frozen_config_hash_depth_channels_and_reranker(self) -> None:
        config = load_baseline_config(ROOT / "configs/retrieval/baseline_v1.json")
        self.assertEqual(config.sha256, "6d49b6d9302b32eb5446ceeb36eb14642091a73ff264c4d7cfeeb4569858cea1")
        self.assertEqual(config.k, 10)
        self.assertEqual(config.candidate_pool_size, 40)
        self.assertEqual(config.rrf_constant, 60)
        self.assertEqual(config.score_decimal_places, 12)
        self.assertEqual(config.reranker_version, "ontology_tie_break_v1")

    def test_config_rejects_unknown_fields_and_frozen_value_drift(self) -> None:
        source = (ROOT / "configs/retrieval/baseline_v1.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(source.replace('"k": 10', '"k": 9'), encoding="utf-8")
            with self.assertRaisesRegex(BaselineRetrievalError, "changed"):
                load_baseline_config(path)
            path.write_text(source[:-2] + ',"extra":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(BaselineRetrievalError, "fields"):
                load_baseline_config(path)

    def test_b2_b3_b4_require_exact_enabled_kinds(self) -> None:
        self.assertEqual(execution("B2").request.enabled_record_kinds, ("atomic",))
        self.assertEqual(execution("B3").request.enabled_record_kinds, ("session",))
        self.assertEqual(execution("B4").request.enabled_record_kinds, ("atomic", "session"))
        with self.assertRaisesRegex(BaselineRetrievalError, "record kinds"):
            replace(execution("B2"), baseline_id="B3")

    def test_request_rejects_plan_eligibility_version_user_and_depth_drift(self) -> None:
        value = execution("B2")
        with self.assertRaisesRegex(BaselineRetrievalError, "depth"):
            replace(value, k=9)
        with self.assertRaisesRegex(BaselineRetrievalError, "users differ"):
            replace(value, eligibility=replace(value.eligibility, user_id="user_002"))
        with self.assertRaisesRegex(BaselineRetrievalError, "plan and eligibility"):
            replace(value, eligibility=replace(value.eligibility, plan_id="b" * 64))
        with self.assertRaisesRegex(BaselineRetrievalError, "ranking version"):
            replace(value, ranking_config_version="retrieval_baseline_v2")

    def test_execution_id_is_stable_and_binds_baseline_and_eligibility(self) -> None:
        first = execution("B2")
        self.assertEqual(first.execution_id, execution("B2").execution_id)
        self.assertNotEqual(first.execution_id, execution("B4").execution_id)
        changed = replace(
            first.eligibility,
            decisions=(
                replace(first.eligibility.decisions[0], eligible=False, rejection_reasons=("speaker_mismatch",)),
                first.eligibility.decisions[1],
            ),
        )
        config = load_baseline_config(ROOT / "configs/retrieval/baseline_v1.json")
        rebuilt = build_baseline_request(first.request, first.plan, changed, "B2", config=config)
        self.assertNotEqual(first.execution_id, rebuilt.execution_id)


class ScoreAndFusionTests(unittest.TestCase):
    def test_fixed_decimal_rounds_half_even_to_twelve_places(self) -> None:
        self.assertEqual(fixed_decimal(Decimal("1.2345678901234")), "1.234567890123")
        self.assertEqual(fixed_decimal(Decimal("1.2345678901236")), "1.234567890124")
        self.assertEqual(fixed_decimal(-0.25), "-0.250000000000")

    def test_nonfinite_scores_and_invalid_score_strings_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(BaselineRetrievalError, "finite"):
                fixed_decimal(value)
        with self.assertRaisesRegex(BaselineRetrievalError, "fixed decimal"):
            SearchChannelHit("record_1", "atomic", "atomic_vector", 1, "0.5")

    def test_rrf_rank_one_and_two_channel_sum_are_exact(self) -> None:
        self.assertEqual(rrf_contribution(1), "0.016393442623")
        hits = (
            SearchChannelHit("record_1", "atomic", "atomic_lexical", 1, "1.000000000000"),
            SearchChannelHit("record_1", "atomic", "atomic_vector", 1, "0.500000000000"),
        )
        fused = fuse_channels(hits, (), {"record_1": ("atomic", ("candidate",))})
        self.assertEqual(fused[0].final_score, "0.032786885246")
        self.assertEqual(len(fused[0].contributions), 2)

    def test_vector_score_may_be_negative_but_lexical_cannot(self) -> None:
        vector = SearchChannelHit("record_1", "atomic", "atomic_vector", 1, "-0.250000000000")
        self.assertEqual(
            fuse_channels((vector,), (), {"record_1": ("atomic", ("candidate",))})[0].final_score,
            "0.016393442623",
        )
        lexical = SearchChannelHit("record_1", "atomic", "atomic_lexical", 1, "-0.250000000000")
        with self.assertRaisesRegex(BaselineRetrievalError, "negative"):
            fuse_channels((lexical,), (), {"record_1": ("atomic", ("candidate",))})

    def test_duplicate_channel_rank_or_record_is_rejected(self) -> None:
        first = SearchChannelHit("record_1", "atomic", "atomic_vector", 1, "0.500000000000")
        duplicate_rank = SearchChannelHit("record_2", "atomic", "atomic_vector", 1, "0.400000000000")
        with self.assertRaisesRegex(BaselineRetrievalError, "duplicated"):
            fuse_channels(
                (first, duplicate_rank),
                (),
                {"record_1": ("atomic", ("candidate",)), "record_2": ("atomic", ("candidate",))},
            )
        duplicate_record = replace(first, rank=2)
        with self.assertRaisesRegex(BaselineRetrievalError, "duplicated"):
            fuse_channels((first, duplicate_record), (), {"record_1": ("atomic", ("candidate",))})

    def test_missing_record_attributes_fail_closed(self) -> None:
        hit = SearchChannelHit("record_1", "atomic", "atomic_vector", 1, "0.500000000000")
        with self.assertRaisesRegex(BaselineRetrievalError, "metadata"):
            fuse_channels((hit,), (), {})


class TieBreakerTests(unittest.TestCase):
    def test_score_precedes_every_tie_rule(self) -> None:
        lower = candidate("record_atomic", "atomic", ("current",), "0.010000000000")
        higher = candidate("record_session", "session", ("candidate",), "0.020000000000")
        self.assertEqual(rerank_candidates((lower, higher), "current_state")[0], higher)

    def test_kind_preferences_cover_all_query_labels(self) -> None:
        atomic = candidate("record_atomic", "atomic", ("candidate",))
        session = candidate("record_session", "session", ("candidate",))
        for label in ("historical_state", "change_over_time", "relationship"):
            self.assertEqual(rerank_candidates((atomic, session), label)[0].record_kind, "session")
        for label in ("current_state", "specific_event", "commitment", "evidence_request", "unknown"):
            self.assertEqual(rerank_candidates((session, atomic), label)[0].record_kind, "atomic")

    def test_current_historical_and_default_lifecycle_preferences(self) -> None:
        current = candidate("record_a", "atomic", ("current",))
        historical = candidate("record_b", "atomic", ("historical",))
        confirmed = candidate("record_c", "atomic", ("confirmed",))
        self.assertEqual(rerank_candidates((historical, current), "current_state")[0], current)
        self.assertEqual(rerank_candidates((current, historical), "historical_state")[0], historical)
        self.assertEqual(rerank_candidates((current, confirmed), "specific_event")[0], confirmed)

    def test_session_uses_best_lifecycle_and_record_id_is_final_tie_break(self) -> None:
        mixed = candidate("record_b", "session", ("candidate", "confirmed"))
        candidate_only = candidate("record_a", "session", ("candidate",))
        self.assertEqual(rerank_candidates((candidate_only, mixed), "unknown")[0], mixed)
        first = candidate("record_a", "atomic", ("candidate",))
        second = candidate("record_b", "atomic", ("candidate",))
        self.assertEqual(rerank_candidates((second, first), "unknown"), (first, second))


class ExpansionTests(unittest.TestCase):
    def test_atomic_seed_limit_is_applied_after_kind_filtering(self) -> None:
        sessions = tuple(
            candidate(f"session_{number:02d}", "session", ("historical",))
            for number in range(10)
        )
        atomics = tuple(
            candidate(f"atomic_{number:02d}", "atomic", ("historical",))
            for number in range(2)
        )
        bound = {
            item.index_record_id: SimpleNamespace(
                metadata=SimpleNamespace(
                    record_kind=item.record_kind,
                    claim_ids=(f"claim_{item.index_record_id}",),
                ),
                atomic_claim_id=(
                    f"claim_{item.index_record_id}"
                    if item.record_kind == "atomic"
                    else None
                ),
            )
            for item in (*sessions, *atomics)
        }

        seeds = RetrievalSearchRepository._atomic_seeds(
            object.__new__(RetrievalSearchRepository),
            (*sessions, *atomics),
            bound,
        )

        self.assertEqual(
            tuple((item.index_record_id, item.seed_rank) for item in seeds),
            (("atomic_00", 1), ("atomic_01", 2)),
        )

    def test_old_version_orders_by_seed_then_newest_neighbor_and_deduplicates(self) -> None:
        seeds = (
            AtomicExpansionSeed("seed_1", 1, "claim_1"),
            AtomicExpansionSeed("seed_2", 2, "claim_1"),
        )
        neighbors = (
            OldVersionNeighbor("old_1", "claim_1", AS_OF - timedelta(days=2)),
            OldVersionNeighbor("old_2", "claim_1", AS_OF - timedelta(days=1)),
            OldVersionNeighbor("other", "claim_2", AS_OF),
        )
        paths = build_old_version_paths(seeds, neighbors)
        self.assertEqual(tuple(item.index_record_id for item in paths), ("old_2", "old_1"))
        self.assertEqual(tuple(item.rank for item in paths), (1, 2))
        self.assertTrue(all(item.seed_record_id == "seed_1" for item in paths))

    def test_old_version_rejects_naive_transaction_time(self) -> None:
        with self.assertRaisesRegex(BaselineRetrievalError, "naive"):
            build_old_version_paths(
                (AtomicExpansionSeed("seed_1", 1, "claim_1"),),
                (OldVersionNeighbor("old_1", "claim_1", datetime(2026, 1, 1)),),
            )

    def test_relation_order_uses_seed_type_relation_neighbor_and_deduplicates(self) -> None:
        paths = build_relation_paths(
            (
                RelationNeighbor("seed_2", 2, "neighbor_1", "relation_3", "corrects", "incoming"),
                RelationNeighbor("seed_1", 1, "neighbor_2", "relation_2", "refines", "outgoing"),
                RelationNeighbor("seed_1", 1, "neighbor_1", "relation_1", "corrects", "outgoing"),
            )
        )
        self.assertEqual(tuple(item.index_record_id for item in paths), ("neighbor_1", "neighbor_2"))
        self.assertEqual(paths[0].relation_type, "corrects")
        self.assertEqual(paths[0].rank, 1)

    def test_symmetric_relation_direction_is_preserved(self) -> None:
        paths = build_relation_paths(
            (
                RelationNeighbor(
                    "seed_1", 1, "neighbor_1", "relation_1",
                    "contradicts", "symmetric",
                ),
            )
        )

        self.assertEqual(paths[0].direction, "symmetric")

    def test_relation_paths_reject_unknown_type_direction_and_self_edge(self) -> None:
        for value in (
            RelationNeighbor("seed", 1, "neighbor", "relation", "unknown", "outgoing"),
            RelationNeighbor("seed", 1, "neighbor", "relation", "corrects", "sideways"),
            RelationNeighbor("seed", 1, "seed", "relation", "corrects", "outgoing"),
        ):
            with self.assertRaises(BaselineRetrievalError):
                build_relation_paths((value,))


class ResultAndTraceTests(unittest.TestCase):
    def test_pre_filter_reasons_are_preserved_and_post_rank_reasons_are_exact(self) -> None:
        value = execution("B2")
        rejections = build_rejections(
            value.eligibility,
            ("record_atomic",),
            (),
        )
        by_id = {item.index_record_id: item for item in rejections}
        self.assertEqual(by_id["record_rejected"].stage, "pre_filter")
        self.assertEqual(by_id["record_rejected"].reasons, ("unknown_valid_time",))
        self.assertEqual(by_id["record_atomic"].reasons, ("outside_top_k",))
        unmatched = build_rejections(value.eligibility, (), ())
        self.assertEqual(unmatched[0].reasons, ("no_channel_match",))

    def test_build_result_keeps_lineage_scores_and_contiguous_rank(self) -> None:
        value = execution("B2")
        fused = (candidate("record_atomic", "atomic", ("candidate",)),)
        result = build_result(value, fused, {"record_atomic": metadata()})
        self.assertEqual(result.accepted[0].rank, 1)
        self.assertEqual(result.accepted[0].claim_ids, ("claim_001",))
        self.assertEqual(result.accepted[0].source_ids, ("source_001",))
        self.assertEqual(result.accepted[0].final_score, "0.016393442623")
        self.assertEqual(result.rejected[0].stage, "pre_filter")

    def test_result_rejects_duplicate_noncontiguous_and_wrong_kind_acceptance(self) -> None:
        item = RetrievedItem(
            "record_atomic", "atomic", "version_001", None, SHA, ("candidate",),
            ("claim_001",), ("version_001",), ("source_001",), ("span_001",),
            (SearchChannelHit("record_atomic", "atomic", "atomic_vector", 1, "0.500000000000"),),
            (RRFContribution("atomic_vector", 1, "0.016393442623"),), (),
            "0.016393442623", 1,
        )
        value = execution("B2")
        result_fields = dict(
            execution_id=value.execution_id,
            baseline_id="B2",
            query_id=value.request.query_id,
            user_id=value.request.user_id,
            plan_id=value.plan.plan_id,
            snapshot_run_id=value.eligibility.snapshot_run_id,
            accepted=(item,),
            rejected=(),
            channel_notices=(),
        )
        BaselineRetrievalResult(**result_fields)
        with self.assertRaisesRegex(BaselineRetrievalError, "duplicated"):
            BaselineRetrievalResult(**{**result_fields, "accepted": (item, item)})
        with self.assertRaisesRegex(BaselineRetrievalError, "contiguous"):
            BaselineRetrievalResult(**{**result_fields, "accepted": (replace(item, rank=2),)})
        with self.assertRaisesRegex(BaselineRetrievalError, "outside"):
            BaselineRetrievalResult(**{**result_fields, "accepted": (replace(item, record_kind="session", atomic_claim_version_id=None, session_summary_id="summary_1"),)})

    def test_retrieved_item_requires_complete_lineage_and_fixed_score(self) -> None:
        with self.assertRaisesRegex(BaselineRetrievalError, "lineage"):
            RetrievedItem(
                "record_1", "atomic", "version_1", None, SHA, ("candidate",),
                (), (), (), (), (), (), (), "0.000000000000", 1,
            )
        with self.assertRaisesRegex(BaselineRetrievalError, "fixed decimal"):
            replace(
                RetrievedItem(
                    "record_1", "atomic", "version_1", None, SHA, ("candidate",),
                    ("claim_1",), ("version_1",), ("source_1",), ("span_1",),
                    (), (), (), "0.000000000000", 1,
                ),
                final_score="0",
            )
        multiple_spans = RetrievedItem(
            "record_1", "atomic", "version_1", None, SHA, ("candidate",),
            ("claim_1",), ("version_1",), ("source_1",),
            ("span_1", "span_2"), (), (), (), "0.000000000000", 1,
        )
        self.assertEqual(multiple_spans.source_ids, ("source_1",))
        self.assertEqual(multiple_spans.span_ids, ("span_1", "span_2"))

    def test_failure_is_sanitized_and_has_no_raw_query_field(self) -> None:
        value = execution("B2")
        failure = BaselineRetrievalFailure(
            "b" * 64, value.execution_id, "B2", value.request.query_id,
            value.request.user_id, "search_failed", "search",
        )
        self.assertNotIn("query_text", failure.__dataclass_fields__)
        with self.assertRaisesRegex(BaselineRetrievalError, "sanitized"):
            replace(failure, location="search/raw")

    def test_serialization_is_stable_and_contains_no_raw_source_text(self) -> None:
        value = execution("B2")
        result = build_result(
            value,
            (candidate("record_atomic", "atomic", ("candidate",)),),
            {"record_atomic": metadata()},
        )
        first = serialize_result(result)
        self.assertEqual(first, serialize_result(result))
        self.assertNotIn(b"raw_content", first)
        self.assertNotIn(b"query_text", first)

    def test_source_has_no_relevance_metrics_model_network_or_search_execution(self) -> None:
        sources = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "src/retrieval/baseline_contracts.py",
                "src/retrieval/baselines.py",
            )
        )
        for forbidden in (
            "Recall@",
            "nDCG",
            "MRR",
            "latency",
            "openai",
            "httpx",
            "requests.",
            "socket",
            "plainto_tsquery",
            "<=>",
        ):
            self.assertNotIn(forbidden, sources)


if __name__ == "__main__":
    unittest.main()
