from __future__ import annotations

import builtins
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from retrieval.contracts import (
    AtomicIndexInput,
    ClaimVersionLineage,
    RelationLineage,
    RetrievalIndexError,
    SessionIndexInput,
    SessionStatementInput,
    SourceSpanLineage,
    TransactionTime,
    ValidTime,
    canonical_json,
    load_index_config,
)
from retrieval.embeddings import DeterministicTokenHashEmbedder, normalized_tokens
from retrieval.indexing import (
    build_atomic_index_record,
    build_session_index_record,
    render_atomic_content,
    render_session_content,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/retrieval/index_v1.json"
UTC = timezone.utc
BASE = datetime(2026, 8, 1, 9, tzinfo=UTC)


def claim_lineage(
    *,
    user_id: str = "user_001",
    claim_id: str = "claim_001",
    claim_version_id: str = "claim_version_001",
    lifecycle_status: str = "candidate",
    order: int = 0,
) -> ClaimVersionLineage:
    return ClaimVersionLineage(
        user_id, claim_id, claim_version_id, lifecycle_status, order
    )


def source_lineage(
    *,
    user_id: str = "user_001",
    claim_id: str = "claim_001",
    claim_version_id: str = "claim_version_001",
    source_id: str = "source_001",
    span_id: str = "span_001",
    support_type: str = "supports",
    order: int = 0,
) -> SourceSpanLineage:
    return SourceSpanLineage(
        user_id,
        claim_id,
        claim_version_id,
        source_id,
        span_id,
        support_type,
        order,
    )


def atomic(**changes: object) -> AtomicIndexInput:
    values: dict[str, object] = {
        "user_id": "user_001",
        "claim_id": "claim_001",
        "claim_version_id": "claim_version_001",
        "subject_id": "user_001",
        "speaker_id": "user_001",
        "predicate": "work_preference",
        "object_json": {"mode": "café", "days": ["Mon", "Fri"]},
        "polarity": "positive",
        "epistemic_status": "asserted",
        "memory_kind": None,
        "lifecycle_status": "candidate",
        "valid_time": ValidTime("unknown"),
        "transaction_time": TransactionTime(BASE),
        "sensitivity": "standard",
        "claim_lineage": (claim_lineage(),),
        "source_lineage": (source_lineage(),),
        "relation_lineage": (),
    }
    values.update(changes)
    return AtomicIndexInput(**values)  # type: ignore[arg-type]


def statement(
    *,
    statement_id: str = "statement_001",
    statement_kind: str = "observed_fact",
    lifecycle_view: str = "candidate",
    text: str = "Candidate: user_001 works remotely.",
    claim_id: str = "claim_001",
    claim_version_id: str = "claim_version_001",
    lifecycle_status: str = "candidate",
    source_id: str = "source_001",
    span_id: str = "span_001",
    order: int = 0,
) -> SessionStatementInput:
    return SessionStatementInput(
        statement_id,
        statement_kind,
        lifecycle_view,
        text,
        (
            claim_lineage(
                claim_id=claim_id,
                claim_version_id=claim_version_id,
                lifecycle_status=lifecycle_status,
            ),
        ),
        (
            source_lineage(
                claim_id=claim_id,
                claim_version_id=claim_version_id,
                source_id=source_id,
                span_id=span_id,
            ),
        ),
        order,
    )


def session(**changes: object) -> SessionIndexInput:
    values: dict[str, object] = {
        "user_id": "user_001",
        "session_summary_id": "summary_001",
        "session_definition_id": "session_001",
        "renderer_version": "session_summary_renderer_v1",
        "summary_text": (
            "Observed facts\n- Candidate: user_001 works remotely.\n\n"
            "Unresolved questions\n- None."
        ),
        "statements": (statement(),),
        "session_source_ids": ("source_001", "source_without_claim"),
        "valid_time": ValidTime("unknown"),
        "transaction_time": TransactionTime(BASE),
        "sensitivity": "standard",
        "contains_sensitive": False,
    }
    values.update(changes)
    return SessionIndexInput(**values)  # type: ignore[arg-type]


class BombEmbedder:
    version = "deterministic_token_hash_v1"
    dimension = 256

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        raise AssertionError("feature work ran")


class RetrievalConfigAndContractTests(unittest.TestCase):
    def test_frozen_config_identity_field_order_and_hash(self) -> None:
        config = load_index_config(CONFIG)
        self.assertEqual(config.index_version, "retrieval_index_v1")
        self.assertEqual(config.embedding_dimension, 256)
        self.assertEqual(config.distance, "cosine")
        self.assertEqual(config.fts_configuration, "simple")
        self.assertEqual(
            config.eligible_lifecycle_statuses,
            (
                "candidate",
                "confirmed",
                "current",
                "historical",
                "disputed",
                "superseded",
            ),
        )
        self.assertEqual(
            config.atomic_content_fields,
            (
                "subject_id",
                "speaker_id",
                "predicate",
                "object_json",
                "polarity",
                "epistemic_status",
                "memory_kind",
                "lifecycle_status",
                "valid_time",
                "transaction_time",
                "checked_relations",
            ),
        )
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "4579b9fe671985a35f605ad9f0dcf256d4672b95329e597d0876754bc5c84a48",
        )

    def test_config_rejects_field_or_frozen_value_drift(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        value["embedding_dimension"] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalIndexError, "configuration changed"):
                load_index_config(path)
            value["extra"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalIndexError, "fields changed"):
                load_index_config(path)

    def test_json_is_detached_unicode_safe_and_canonical(self) -> None:
        original = {"z": ["café", True], "a": 1}
        value = atomic(object_json=original)
        original["a"] = 2
        self.assertEqual(value.object_json, {"a": 1, "z": ["café", True]})
        self.assertEqual(
            canonical_json(value.object_json), '{"a":1,"z":["café",true]}'
        )
        with self.assertRaisesRegex(RetrievalIndexError, "JSON-safe"):
            atomic(object_json={"bad": float("nan")})
        with self.assertRaisesRegex(RetrievalIndexError, "strings"):
            atomic(object_json={1: "bad"})  # type: ignore[dict-item]

    def test_valid_and_transaction_time_semantics_are_exact(self) -> None:
        day = ValidTime(
            "day", valid_from_date=date(2026, 8, 1), valid_to_date=date(2026, 8, 1)
        )
        self.assertEqual(day.valid_from_date, day.valid_to_date)
        with self.assertRaisesRegex(RetrievalIndexError, "cannot mix"):
            ValidTime(
                "day", valid_from_date=date(2026, 8, 1), valid_from_timestamp=BASE
            )
        mixed = ValidTime("mixed")
        self.assertEqual(mixed.time_precision, "mixed")
        with self.assertRaisesRegex(RetrievalIndexError, "cannot be mixed"):
            atomic(valid_time=mixed)
        with self.assertRaisesRegex(RetrievalIndexError, "timezone-aware"):
            ValidTime("timestamp", valid_from_timestamp=datetime(2026, 8, 1))
        with self.assertRaisesRegex(RetrievalIndexError, "known valid time"):
            atomic(
                lifecycle_status="current",
                claim_lineage=(claim_lineage(lifecycle_status="current"),),
            )
        interval = TransactionTime(BASE, BASE + timedelta(days=1))
        self.assertTrue(interval.contains(BASE))
        self.assertFalse(interval.contains(BASE + timedelta(days=1)))

    def test_lineage_requires_exact_order_anchor_and_user(self) -> None:
        with self.assertRaisesRegex(RetrievalIndexError, "consecutive order"):
            atomic(source_lineage=(source_lineage(order=1),))
        with self.assertRaisesRegex(RetrievalIndexError, "crosses its anchor"):
            atomic(source_lineage=(source_lineage(user_id="user_002"),))
        with self.assertRaisesRegex(RetrievalIndexError, "crosses users"):
            session(
                statements=(
                    replace(
                        statement(),
                        claim_lineage=(claim_lineage(user_id="user_002"),),
                        source_lineage=(source_lineage(user_id="user_002"),),
                    ),
                )
            )

    def test_session_preserves_no_claim_members_and_sensitive_flag(self) -> None:
        value = session()
        self.assertEqual(
            value.session_source_ids, ("source_001", "source_without_claim")
        )
        with self.assertRaisesRegex(RetrievalIndexError, "sensitivity flag"):
            session(sensitivity="sensitive", contains_sensitive=False)
        with self.assertRaisesRegex(RetrievalIndexError, "outside the session"):
            session(session_source_ids=("source_without_claim",))


class DeterministicEmbeddingTests(unittest.TestCase):
    def test_normalization_is_nfkc_casefolded_alphanumeric_runs(self) -> None:
        self.assertEqual(
            normalized_tokens("ＷＯＲＫ café_CAFÉ 2026!"),
            ("work", "café", "café", "2026"),
        )

    def test_embedding_is_stable_fixed_dimension_and_l2_normalized(self) -> None:
        embedder = DeterministicTokenHashEmbedder()
        first = embedder.embed("Remote remote café")
        second = embedder.embed("Remote remote café")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertTrue(math.isclose(math.sqrt(sum(x * x for x in first)), 1.0))
        self.assertTrue(any(value < 0 for value in first) or any(value > 0 for value in first))

    def test_empty_tokens_return_the_exact_zero_vector(self) -> None:
        vector = DeterministicTokenHashEmbedder().embed(" _ -- \n")
        self.assertEqual(vector, (0.0,) * 256)

    def test_embedding_never_uses_python_hash_or_random_state(self) -> None:
        embedder = DeterministicTokenHashEmbedder()
        with patch.object(builtins, "hash", side_effect=AssertionError("hash called")):
            vector = embedder.embed("deterministic token hash")
        self.assertEqual(len(vector), 256)


class RetrievalRenderingAndPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_index_config(CONFIG)
        self.embedder = DeterministicTokenHashEmbedder()

    def test_atomic_render_is_canonical_and_contains_checked_relations(self) -> None:
        relation = RelationLineage(
            "relation_001",
            "user_001",
            "claim_001",
            "claim_old",
            "corrects",
            "outgoing",
            0,
        )
        value = atomic(relation_lineage=(relation,))
        rendered = render_atomic_content(value, config=self.config)
        self.assertEqual(
            rendered,
            "\n".join(
                (
                    'subject_id:"user_001"',
                    'speaker_id:"user_001"',
                    'predicate:"work_preference"',
                    'object_json:{"days":["Mon","Fri"],"mode":"café"}',
                    'polarity:"positive"',
                    'epistemic_status:"asserted"',
                    "memory_kind:null",
                    'lifecycle_status:"candidate"',
                    'valid_time:{"time_precision":"unknown","valid_from_date":null,"valid_from_timestamp":null,"valid_to_date":null,"valid_to_timestamp":null}',
                    'transaction_time:{"transaction_from":"2026-08-01T09:00:00Z","transaction_to":null}',
                    'checked_relations:[{"direction":"outgoing","order":0,"relation_id":"relation_001","relation_type":"corrects","source_claim_id":"claim_001","target_claim_id":"claim_old"}]',
                )
            ),
        )

    def test_session_render_reuses_exact_text_and_separates_questions(self) -> None:
        question = statement(
            statement_id="statement_002",
            statement_kind="unresolved_question",
            lifecycle_view="disputed",
            text="Is the office Pune or Mumbai?",
            claim_id="claim_002",
            claim_version_id="claim_version_002",
            lifecycle_status="disputed",
            source_id="source_002",
            span_id="span_002",
            order=1,
        )
        value = session(
            statements=(statement(), question),
            session_source_ids=("source_001", "source_002", "source_without_claim"),
        )
        rendered = render_session_content(value, config=self.config)
        self.assertIn(canonical_json(value.summary_text), rendered)
        self.assertIn('"text":"Candidate: user_001 works remotely."', rendered)
        self.assertIn('unresolved_questions:[{"claim_ids":["claim_002"]', rendered)
        self.assertEqual(rendered.count("Is the office Pune or Mumbai?"), 1)

    def test_atomic_build_is_stable_and_content_drift_changes_snapshot(self) -> None:
        value = atomic()
        first = build_atomic_index_record(
            "user_001", BASE, value, config=self.config, embedder=self.embedder
        )
        second = build_atomic_index_record(
            "user_001", BASE, value, config=self.config, embedder=self.embedder
        )
        self.assertEqual(first, second)
        assert first is not None
        changed = build_atomic_index_record(
            "user_001",
            BASE,
            atomic(object_json={"mode": "office"}),
            config=self.config,
            embedder=self.embedder,
        )
        assert changed is not None
        self.assertNotEqual(first.index_record_id, changed.index_record_id)
        self.assertNotEqual(first.input_snapshot_sha256, changed.input_snapshot_sha256)
        self.assertEqual(first.lifecycle_statuses, ("candidate",))
        self.assertIsNone(first.memory_kind)

    def test_visible_historical_and_superseded_lifecycle_are_preserved(self) -> None:
        for status in ("confirmed", "historical", "disputed", "superseded"):
            with self.subTest(status=status):
                value = atomic(
                    lifecycle_status=status,
                    claim_lineage=(claim_lineage(lifecycle_status=status),),
                )
                record = build_atomic_index_record(
                    "user_001", BASE, value, config=self.config, embedder=self.embedder
                )
                assert record is not None
                self.assertEqual(record.lifecycle_statuses, (status,))

    def test_excluded_and_restricted_records_stop_before_embedding(self) -> None:
        cases = (
            atomic(
                lifecycle_status="excluded",
                claim_lineage=(claim_lineage(lifecycle_status="excluded"),),
            ),
            atomic(sensitivity="restricted"),
        )
        for value in cases:
            embedder = BombEmbedder()
            with self.subTest(value=value.lifecycle_status):
                self.assertIsNone(
                    build_atomic_index_record(
                        "user_001", BASE, value, config=self.config, embedder=embedder
                    )
                )
                self.assertEqual(embedder.calls, 0)
        embedder = BombEmbedder()
        self.assertIsNone(
            build_session_index_record(
                "user_001",
                BASE,
                session(sensitivity="restricted"),
                config=self.config,
                embedder=embedder,
            )
        )
        self.assertEqual(embedder.calls, 0)

    def test_user_and_transaction_visibility_stop_before_embedding(self) -> None:
        embedder = BombEmbedder()
        with self.assertRaisesRegex(RetrievalIndexError, "crosses users"):
            build_atomic_index_record(
                "user_002", BASE, atomic(), config=self.config, embedder=embedder
            )
        self.assertEqual(embedder.calls, 0)
        self.assertIsNone(
            build_atomic_index_record(
                "user_001",
                BASE - timedelta(microseconds=1),
                atomic(),
                config=self.config,
                embedder=embedder,
            )
        )
        self.assertEqual(embedder.calls, 0)

    def test_session_build_preserves_lifecycle_lineage_and_sensitive_flag(self) -> None:
        facts = (
            statement(),
            statement(
                statement_id="statement_002",
                text="Historical: user_001 worked in Pune.",
                lifecycle_view="historical",
                claim_id="claim_002",
                claim_version_id="claim_version_002",
                lifecycle_status="historical",
                source_id="source_002",
                span_id="span_002",
                order=1,
            ),
        )
        value = session(
            statements=facts,
            session_source_ids=("source_001", "source_002"),
            sensitivity="sensitive",
            contains_sensitive=True,
        )
        record = build_session_index_record(
            "user_001", BASE, value, config=self.config, embedder=self.embedder
        )
        assert record is not None
        self.assertEqual(record.lifecycle_statuses, ("candidate", "historical"))
        self.assertTrue(record.contains_sensitive)
        self.assertEqual(record.sensitivity, "sensitive")
        self.assertEqual(len(record.claim_lineage), 2)
        self.assertEqual(len(record.source_lineage), 2)

    def test_embedder_contract_mismatch_is_rejected(self) -> None:
        embedder = BombEmbedder()
        embedder.dimension = 255
        with self.assertRaisesRegex(RetrievalIndexError, "embedder contract"):
            build_atomic_index_record(
                "user_001", BASE, atomic(), config=self.config, embedder=embedder
            )


if __name__ == "__main__":
    unittest.main()
