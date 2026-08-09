from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import ast
import json
from pathlib import Path
import tempfile
import unittest

from retrieval.query_contracts import (
    INDEX_VERSION,
    RELATION_EXPANSION_TYPES,
    EligibilityDecision,
    EligibilityResult,
    RequestedValidTime,
    RetrievalQueryError,
    RetrievalQueryFailure,
    RetrievalQueryRequest,
    parse_retrieval_query_request,
)
from retrieval.query_planner import (
    build_query_plan,
    classify_query,
    load_query_planner_config,
    normalized_query_tokens,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/retrieval/query_planner_v1.json"
UTC = timezone.utc
AS_OF = datetime(2026, 8, 10, 9, 30, tzinfo=UTC)
CONFIG_SHA256 = "538af5ceb41f50c752dc086c9f6ef39ee6b42b4ec0616948b3ac38192f66c654"


def request(
    text: str = "Where do I work now?",
    *,
    valid_time: RequestedValidTime | None = None,
    kinds: tuple[str, ...] = ("atomic", "session"),
    speakers: tuple[str, ...] = (),
    entities: tuple[str, ...] = (),
    sensitivity: str = "standard",
    allow_unclassified: bool = False,
) -> RetrievalQueryRequest:
    return RetrievalQueryRequest(
        query_id="query_001",
        user_id="user_001",
        query_text=text,
        as_of=AS_OF,
        index_version=INDEX_VERSION,
        enabled_record_kinds=kinds,
        requested_valid_time=valid_time,
        speaker_ids=speakers,
        entity_ids=entities,
        sensitivity_scope=sensitivity,
        allow_unclassified_sensitivity=allow_unclassified,
    )


def request_value() -> dict[str, object]:
    return {
        "query_id": "query_001",
        "user_id": "user_001",
        "query_text": "Where do I work now?",
        "as_of": "2026-08-10T09:30:00Z",
        "index_version": INDEX_VERSION,
        "enabled_record_kinds": ["atomic", "session"],
        "requested_valid_time": None,
        "speaker_ids": [],
        "entity_ids": [],
        "sensitivity_scope": "standard",
        "allow_unclassified_sensitivity": False,
    }


class QueryPlannerConfigAndClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_query_planner_config(CONFIG_PATH)

    def test_frozen_config_identity_hash_and_order(self) -> None:
        self.assertEqual(self.config.sha256, CONFIG_SHA256)
        self.assertEqual(
            self.config.classification_precedence,
            (
                "evidence_request",
                "change_over_time",
                "historical_state",
                "current_state",
                "specific_event",
                "relationship",
                "commitment",
                "unknown",
            ),
        )
        self.assertEqual(self.config.relation_expansion_types, RELATION_EXPANSION_TYPES)

    def test_all_eight_primary_labels_are_emitted(self) -> None:
        cases = {
            "current_state": "Where am I working currently?",
            "historical_state": "Where did I work previously?",
            "change_over_time": "How has my work changed over time?",
            "specific_event": "When did the specific event happen?",
            "relationship": "What is my relationship with Alex?",
            "commitment": "What deadline did I commit to?",
            "evidence_request": "What evidence supports my office location?",
            "unknown": "Tell me something useful",
        }
        self.assertEqual(
            {
                label: classify_query(text, config=self.config)
                for label, text in cases.items()
            },
            {label: label for label in cases},
        )

    def test_evidence_precedes_change_and_change_precedes_state(self) -> None:
        self.assertEqual(
            classify_query(
                "What evidence shows how my role changed over time?",
                config=self.config,
            ),
            "evidence_request",
        )
        self.assertEqual(
            classify_query(
                "How has my current role changed over time?",
                config=self.config,
            ),
            "change_over_time",
        )
        self.assertEqual(
            classify_query("What did I do in the past and now?", config=self.config),
            "historical_state",
        )

    def test_normalization_is_nfkc_casefolded_alphanumeric_tokens(self) -> None:
        self.assertEqual(normalized_query_tokens("ＣＵＲＲＥＮＴＬＹ—At_Work"), ("currently", "at", "work"))
        self.assertEqual(
            classify_query("ＲＩＧＨＴ　ＮＯＷ", config=self.config),
            "current_state",
        )

    def test_config_rejects_unknown_fields_and_rule_drift(self) -> None:
        value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value["extra"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalQueryError, "fields"):
                load_query_planner_config(path)
            value.pop("extra")
            value["classification_precedence"][0] = "current_state"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalQueryError, "changed"):
                load_query_planner_config(path)


class RetrievalQueryContractTests(unittest.TestCase):
    def test_parser_requires_exact_fields_and_aware_time(self) -> None:
        parsed = parse_retrieval_query_request(request_value())
        self.assertEqual(parsed.as_of, AS_OF)
        value = request_value()
        value["extra"] = "not allowed"
        with self.assertRaisesRegex(RetrievalQueryError, "fields"):
            parse_retrieval_query_request(value)
        value = request_value()
        value["as_of"] = "2026-08-10T09:30:00"
        with self.assertRaisesRegex(RetrievalQueryError, "timezone-aware"):
            parse_retrieval_query_request(value)

    def test_request_rejects_empty_version_duplicates_and_unsorted_filters(self) -> None:
        with self.assertRaisesRegex(RetrievalQueryError, "empty"):
            replace(request(), query_text="  ")
        with self.assertRaisesRegex(RetrievalQueryError, "index version"):
            replace(request(), index_version="retrieval_index_v2")
        with self.assertRaisesRegex(RetrievalQueryError, "record kinds"):
            replace(request(), enabled_record_kinds=("session", "atomic"))
        with self.assertRaisesRegex(RetrievalQueryError, "speaker IDs"):
            replace(request(), speaker_ids=("speaker_1", "speaker_1"))
        with self.assertRaisesRegex(RetrievalQueryError, "entity IDs"):
            replace(request(), entity_ids=("z", "a"))

    def test_requested_valid_time_rejects_mixed_naive_and_reversed_values(self) -> None:
        with self.assertRaisesRegex(RetrievalQueryError, "mix"):
            RequestedValidTime(
                kind="range",
                range_start_date=date(2026, 1, 1),
                range_end_timestamp=AS_OF,
            )
        with self.assertRaisesRegex(RetrievalQueryError, "timezone-aware"):
            RequestedValidTime(
                kind="point",
                point_timestamp=datetime(2026, 1, 1),
            )
        with self.assertRaisesRegex(RetrievalQueryError, "reversed"):
            RequestedValidTime(
                kind="range",
                range_start_date=date(2026, 2, 1),
                range_end_date=date(2026, 1, 1),
            )

    def test_failure_is_sanitized_and_has_no_query_text_field(self) -> None:
        failure = RetrievalQueryFailure(
            failure_id="a" * 64,
            query_id="query_001",
            user_id="user_001",
            code="invalid_request",
            location="request",
        )
        self.assertNotIn("query_text", failure.__dataclass_fields__)
        with self.assertRaisesRegex(RetrievalQueryError, "sanitized"):
            replace(failure, location="request/raw text")
        with self.assertRaisesRegex(RetrievalQueryError, "failure code"):
            replace(failure, code="provider_error")

    def test_eligibility_contracts_preserve_markers_and_stable_order(self) -> None:
        first = EligibilityDecision(
            "record_001",
            True,
            (),
            ("candidate",),
            "standard",
            False,
        )
        second = EligibilityDecision(
            "record_002",
            False,
            ("sensitive_not_authorized",),
            ("disputed",),
            "sensitive",
            False,
        )
        result = EligibilityResult(
            "a" * 64,
            "user_001",
            INDEX_VERSION,
            "run_001",
            (first, second),
        )
        self.assertEqual(result.eligible_record_ids, ("record_001",))
        with self.assertRaisesRegex(RetrievalQueryError, "sorted"):
            replace(result, decisions=(second, first))


class QueryPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_query_planner_config(CONFIG_PATH)

    def test_current_query_resolves_as_of_and_uses_current_policy(self) -> None:
        plan = build_query_plan(request(), config=self.config)
        self.assertEqual(plan.primary_label, "current_state")
        self.assertEqual(plan.requested_valid_time.point_timestamp, AS_OF)
        self.assertEqual(
            plan.allowed_lifecycle_statuses,
            ("candidate", "confirmed", "current", "disputed"),
        )
        self.assertFalse(plan.include_previous_versions)

    def test_structured_fields_override_text_hints_without_reordering(self) -> None:
        valid_time = RequestedValidTime(kind="point", point_date=date(2024, 5, 1))
        value = request(
            "What is current now?",
            valid_time=valid_time,
            kinds=("atomic",),
            speakers=("speaker_001", "speaker_002"),
            entities=("entity_001",),
        )
        plan = build_query_plan(value, config=self.config)
        self.assertIs(plan.requested_valid_time, valid_time)
        self.assertEqual(plan.enabled_record_kinds, ("atomic",))
        self.assertEqual(plan.speaker_ids, value.speaker_ids)
        self.assertEqual(plan.entity_ids, value.entity_ids)

    def test_historical_and_change_plans_include_previous_versions(self) -> None:
        historical = build_query_plan(request("Where did I work previously?"), config=self.config)
        changed = build_query_plan(request("How has my role changed over time?"), config=self.config)
        self.assertTrue(historical.include_previous_versions)
        self.assertFalse(historical.checked_relation_expansion_intent)
        self.assertTrue(changed.include_previous_versions)
        self.assertTrue(changed.checked_relation_expansion_intent)
        self.assertEqual(changed.relation_expansion_types, RELATION_EXPANSION_TYPES)

    def test_evidence_precedence_keeps_change_and_source_intents(self) -> None:
        plan = build_query_plan(
            request("What evidence supports the corrected office value?"),
            config=self.config,
        )
        self.assertEqual(plan.primary_label, "evidence_request")
        self.assertTrue(plan.source_evidence_intent)
        self.assertTrue(plan.checked_relation_expansion_intent)
        self.assertFalse(plan.include_previous_versions)

    def test_unstructured_time_is_flagged_but_never_parsed(self) -> None:
        for text in (
            "What happened last year?",
            "What happened on 2025-03-14?",
            "What happened at 09:45?",
        ):
            plan = build_query_plan(request(text), config=self.config)
            self.assertTrue(plan.unresolved_time)
            self.assertTrue(plan.clarification_required)
            self.assertIsNone(plan.requested_valid_time)

    def test_stable_plan_id_binds_request_config_and_resolved_policy(self) -> None:
        first = build_query_plan(request(), config=self.config)
        second = build_query_plan(request(), config=self.config)
        changed = build_query_plan(replace(request(), query_id="query_002"), config=self.config)
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertNotEqual(first.plan_id, changed.plan_id)
        self.assertEqual(first.planner_config_sha256, CONFIG_SHA256)

    def test_sensitivity_flags_and_unknown_policy_remain_explicit(self) -> None:
        plan = build_query_plan(
            request(
                "Tell me something useful",
                sensitivity="sensitive",
                allow_unclassified=True,
            ),
            config=self.config,
        )
        self.assertEqual(plan.primary_label, "unknown")
        self.assertTrue(plan.allow_sensitive)
        self.assertTrue(plan.allow_unclassified_sensitivity)
        self.assertEqual(
            plan.allowed_lifecycle_statuses,
            ("candidate", "confirmed", "current", "historical", "disputed", "superseded"),
        )

    def test_planner_source_has_no_model_network_or_search_execution(self) -> None:
        path = ROOT / "src/retrieval/query_planner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        self.assertTrue(imports.isdisjoint({"openai", "requests", "httpx", "urllib"}))
        plan_fields = set(build_query_plan(request(), config=self.config).__dataclass_fields__)
        self.assertTrue(plan_fields.isdisjoint({"score", "rank", "k", "query_embedding", "fusion"}))
        source = path.read_text(encoding="utf-8").casefold()
        for forbidden in ("vector distance", "ts_rank", "rerank", "recall@", "ndcg", "mrr"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
