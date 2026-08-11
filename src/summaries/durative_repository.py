"""PostgreSQL persistence and event recompute for durative claim plans."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime
import hashlib
from pathlib import Path
from typing import Iterable

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from extraction.predicate_registry import load_predicate_registry
from storage.contracts import ProcessingOutboxRecord

from .contracts import BOUNDARY_VERSION, SessionizationRequest
from .durative import DEFAULT_DURATIVE_RULES_PATH, infer_durative_claim
from .durative_contracts import (
    ALLOWED_FAMILIES,
    REGISTRY_PATH,
    RULES_VERSION,
    DurativeClaimPlan,
    DurativeEpisode,
    DurativeInferenceRequest,
    DurativeInferenceResult,
    DurativePropositionPlan,
    DurativePropositionRequest,
    canonical_json,
    load_durative_rules_config,
    stable_id,
)
from .repository import SessionSourceRepository
from .sessions import SessionizationService


REPO_ROOT = Path(__file__).resolve().parents[2]
DURATIVE_RECOMPUTE_EVENT_TYPES = frozenset(
    {
        "source_ingested",
        "claims_changed",
        "claim_recompute_required",
        "source_deleted",
        "claim_lifecycle_changed",
        "conflict_recompute_required",
        "belief_resolved",
    }
)


class DurativePersistenceError(RuntimeError):
    """A sanitized durative persistence failure."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


class DurativeClaimRepository:
    """Load exact episodes and persist deterministic inference results."""

    def __init__(
        self,
        connection: object,
        *,
        config_path: str | Path = DEFAULT_DURATIVE_RULES_PATH,
    ) -> None:
        self.connection = connection
        self.config_path = Path(config_path)
        self.config = load_durative_rules_config(self.config_path)
        self.session_service = SessionizationService(SessionSourceRepository(connection))

    def load_visible_episodes(
        self,
        user_id: str,
        transaction_as_of: datetime,
    ) -> tuple[DurativeEpisode, ...]:
        sessions = self.session_service.define_sessions(
            SessionizationRequest(user_id, transaction_as_of, BOUNDARY_VERSION)
        )
        source_sessions = {
            source_id: definition.definition_id
            for definition in sessions.definitions
            for source_id in definition.source_ids
        }
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT claim.claim_id, version.version_id AS claim_version_id,
                       claim.subject_id, claim.speaker_id, claim.predicate,
                       claim.predicate_registry_version, claim.object_json,
                       claim.polarity, claim.epistemic_status,
                       version.lifecycle_status, claim.memory_kind,
                       claim.sensitivity, claim.extraction_confidence,
                       version.belief_confidence, evidence.support_type,
                       version.time_precision, version.valid_from_date,
                       version.valid_from_timestamp, version.valid_to_date,
                       version.valid_to_timestamp, version.transaction_from,
                       version.transaction_to, source.source_id, span.span_id,
                       ARRAY(
                           SELECT DISTINCT conflict.label
                           FROM conflict_decisions AS conflict
                           WHERE conflict.user_id = claim.user_id
                             AND conflict.transaction_as_of <= %s
                             AND conflict.classified_at <= %s
                             AND NOT EXISTS (
                                 SELECT 1 FROM belief_resolutions AS resolution
                                 WHERE resolution.user_id = conflict.user_id
                                   AND resolution.decision_id = conflict.decision_id
                                   AND resolution.resolved_at <= %s
                             )
                             AND (
                                 (conflict.left_claim_id = claim.claim_id
                                  AND conflict.left_version_id = version.version_id)
                                 OR
                                 (conflict.right_claim_id = claim.claim_id
                                  AND conflict.right_version_id = version.version_id)
                             )
                           ORDER BY conflict.label
                       ) AS conflict_labels,
                       ARRAY(
                           SELECT DISTINCT relation.relation_type
                           FROM claim_relations AS relation
                           JOIN conflict_decisions AS conflict
                             ON conflict.user_id = relation.user_id
                            AND conflict.decision_id = relation.decision_id
                           WHERE relation.user_id = claim.user_id
                             AND conflict.transaction_as_of <= %s
                             AND conflict.classified_at <= %s
                             AND NOT EXISTS (
                                 SELECT 1 FROM belief_resolutions AS resolution
                                 WHERE resolution.user_id = conflict.user_id
                                   AND resolution.decision_id = conflict.decision_id
                                   AND resolution.resolved_at <= %s
                             )
                             AND (
                                 relation.source_claim_id = claim.claim_id
                                 OR relation.target_claim_id = claim.claim_id
                             )
                             AND (
                                 (conflict.left_claim_id = claim.claim_id
                                  AND conflict.left_version_id = version.version_id)
                                 OR
                                 (conflict.right_claim_id = claim.claim_id
                                  AND conflict.right_version_id = version.version_id)
                             )
                           ORDER BY relation.relation_type
                       ) AS relation_types
                FROM claims AS claim
                JOIN claim_versions AS version
                  ON version.user_id = claim.user_id
                 AND version.claim_id = claim.claim_id
                JOIN evidence_links AS evidence
                  ON evidence.user_id = claim.user_id
                 AND evidence.claim_id = claim.claim_id
                JOIN source_spans AS span
                  ON span.user_id = evidence.user_id
                 AND span.span_id = evidence.span_id
                JOIN source_events AS source
                  ON source.user_id = span.user_id
                 AND source.source_id = span.source_id
                WHERE claim.user_id = %s
                  AND claim.memory_kind IS DISTINCT FROM 'durative'
                  AND source.ingested_at <= %s
                  AND version.transaction_from <= %s
                  AND (version.transaction_to IS NULL OR %s < version.transaction_to)
                ORDER BY claim.claim_id, version.version_id,
                         source.source_id, span.span_id, evidence.support_type
                """,
                (
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                    user_id,
                    transaction_as_of,
                    transaction_as_of,
                    transaction_as_of,
                ),
            )
            rows = tuple(cursor.fetchall())
        episodes: list[DurativeEpisode] = []
        for row in rows:
            session_id = source_sessions.get(row["source_id"])
            if session_id is None:
                raise DurativePersistenceError("stale_episode", "session")
            episode_at = _episode_time(row)
            episode_id = stable_id(
                "durative_episode",
                user_id,
                row["claim_id"],
                row["claim_version_id"],
                session_id,
                row["source_id"],
                row["span_id"],
                row["support_type"],
            )
            episodes.append(
                DurativeEpisode(
                    episode_id=episode_id,
                    user_id=user_id,
                    claim_id=row["claim_id"],
                    claim_version_id=row["claim_version_id"],
                    session_definition_id=session_id,
                    source_id=row["source_id"],
                    span_id=row["span_id"],
                    subject_id=row["subject_id"],
                    speaker_id=row["speaker_id"],
                    predicate=row["predicate"],
                    predicate_registry_version=row["predicate_registry_version"],
                    object_json=row["object_json"],
                    polarity=row["polarity"],
                    epistemic_status=row["epistemic_status"],
                    lifecycle_status=row["lifecycle_status"],
                    memory_kind=row["memory_kind"],
                    sensitivity=row["sensitivity"],
                    extraction_confidence=row["extraction_confidence"],
                    belief_confidence=row["belief_confidence"],
                    support_type=row["support_type"],
                    time_precision=row["time_precision"],
                    valid_from_date=row["valid_from_date"],
                    valid_from_timestamp=row["valid_from_timestamp"],
                    valid_to_date=row["valid_to_date"],
                    valid_to_timestamp=row["valid_to_timestamp"],
                    episode_at=episode_at,
                    transaction_from=row["transaction_from"],
                    transaction_to=row["transaction_to"],
                    conflict_labels=tuple(row["conflict_labels"] or ()),
                    relation_types=tuple(row["relation_types"] or ()),
                )
            )
        return tuple(sorted(episodes, key=lambda item: item.episode_id))

    def persist(
        self,
        request: DurativeInferenceRequest,
        plans: tuple[DurativePropositionPlan, ...],
        episodes: tuple[DurativeEpisode, ...],
    ) -> DurativeInferenceResult:
        with self.connection.transaction():
            return self.persist_in_transaction(request, plans, episodes)

    def persist_in_transaction(
        self,
        request: DurativeInferenceRequest,
        plans: tuple[DurativePropositionPlan, ...],
        episodes: tuple[DurativeEpisode, ...],
    ) -> DurativeInferenceResult:
        self._lock(request.user_id)
        ordered_plans = tuple(sorted(plans, key=lambda item: item.decision.decision_id))
        if len({item.decision.decision_id for item in ordered_plans}) != len(ordered_plans):
            raise DurativePersistenceError("duplicate_decision", "run")
        if any(item.request.user_id != request.user_id for item in ordered_plans):
            raise DurativePersistenceError("cross_user", "run")
        visible = self._revalidate(request, ordered_plans, episodes)
        run_snapshot = _run_snapshot(request, visible, ordered_plans)
        run_id = stable_id(
            "durative_inference_run", request.user_id,
            request.rule_version, request.idempotency_key,
        )
        existing = self.connection.execute(
            """
            SELECT run_id FROM durative_inference_runs
            WHERE user_id = %s AND (run_id = %s OR idempotency_key = %s)
            FOR UPDATE
            """,
            (
                request.user_id,
                run_id,
                request.idempotency_key,
            ),
        ).fetchall()
        if existing:
            if len(existing) != 1 or existing[0][0] != run_id:
                raise DurativePersistenceError("idempotency_drift", "run")
            stored = self._run_matches(
                request, run_id, run_snapshot, ordered_plans, visible,
            )
            if stored is None:
                raise DurativePersistenceError("idempotency_drift", "run")
            return DurativeInferenceResult(
                run_id, request.user_id, request.rule_version, run_snapshot,
                stored, 0, len(stored),
            )

        self._ensure_extraction_version(
            ordered_plans[0] if ordered_plans else None,
            request.transaction_as_of,
        )
        persisted: list[tuple[DurativePropositionPlan, str | None, str | None]] = []
        changed_claims: list[str] = []
        for planned in ordered_plans:
            claim_id: str | None = None
            version_id: str | None = None
            changed = False
            selected = tuple(
                item for item in visible
                if item.episode_id in planned.decision.input_episode_ids
            )
            if planned.claim is not None:
                claim_id, version_id, changed = self._ensure_claim(planned, selected)
            persisted.append((planned, claim_id, version_id))
            if changed and claim_id is not None:
                changed_claims.append(claim_id)
        self._insert_run(
            request, run_id, run_snapshot, tuple(persisted), visible,
        )
        for claim_id in sorted(set(changed_claims)):
            self._insert_claims_changed(request, run_id, claim_id)
        decisions = tuple(
            replace(plan.decision, claim_id=claim_id)
            for plan, claim_id, _version_id in persisted
        )
        return DurativeInferenceResult(
            run_id, request.user_id, request.rule_version, run_snapshot,
            decisions, len(decisions), 0,
        )

    def purge_unmatched(
        self,
        user_id: str,
        active_claim_ids: Iterable[str],
    ) -> int:
        active = tuple(sorted(set(active_claim_ids)))
        self._lock(user_id)
        rows = self.connection.execute(
            """
            SELECT DISTINCT decision.derived_claim_id
            FROM durative_inference_decisions AS decision
            JOIN durative_inference_runs AS run
              ON run.user_id = decision.user_id AND run.run_id = decision.run_id
            WHERE run.user_id = %s AND run.rules_version = %s
              AND decision.derived_claim_id IS NOT NULL
              AND NOT (decision.derived_claim_id = ANY(%s))
            ORDER BY decision.derived_claim_id
            """,
            (user_id, self.config.rules_version, list(active)),
        ).fetchall()
        for row in rows:
            self._delete_derived_claim(user_id, row[0])
        return len(rows)

    def _revalidate(
        self,
        request: DurativeInferenceRequest,
        plans: tuple[DurativePropositionPlan, ...],
        episodes: tuple[DurativeEpisode, ...],
    ) -> tuple[DurativeEpisode, ...]:
        visible = self.load_visible_episodes(
            request.user_id, request.transaction_as_of,
        )
        if visible != tuple(sorted(episodes, key=lambda item: item.episode_id)):
            raise DurativePersistenceError("stale_episode", "provenance")
        by_id = {item.episode_id: item for item in visible}
        for planned in plans:
            try:
                selected = tuple(by_id[item] for item in planned.decision.input_episode_ids)
            except KeyError as error:
                raise DurativePersistenceError("stale_episode", "provenance") from error
            replay = infer_durative_claim(
                planned.request, selected, config_path=self.config_path,
            )
            if replay != planned:
                raise DurativePersistenceError("stale_plan", "snapshot")
        return visible

    def _ensure_extraction_version(
        self,
        planned: DurativePropositionPlan | None,
        created_at: datetime,
    ) -> None:
        metadata = (
            planned.claim.extraction
            if planned is not None and planned.claim is not None
            else None
        )
        if metadata is None:
            extraction_version_id = stable_id(
                "durative_extraction_version",
                RULES_VERSION,
                _file_sha256(self.config_path),
                self.config.predicate_registry_version,
                self.config.predicate_registry_sha256,
            )
            rules_sha256 = _file_sha256(self.config_path)
        else:
            extraction_version_id = metadata.extraction_version_id
            rules_sha256 = metadata.rules_sha256
        schema_hash = stable_id("durative_claim_schema", "v1")
        expected = (
            "deterministic",
            self.config.rules_version,
            rules_sha256,
            "durative_claim_semantic_v1",
            schema_hash,
            self.config.predicate_registry_version,
            self.config.predicate_registry_sha256,
            rules_sha256,
        )
        row = self.connection.execute(
            """
            SELECT model_version, prompt_version, prompt_hash,
                   schema_version, schema_hash, registry_version,
                   registry_hash, input_manifest_hash
            FROM extraction_versions WHERE version_id = %s
            """,
            (extraction_version_id,),
        ).fetchone()
        if row is not None:
            if row != expected:
                raise DurativePersistenceError("stable_id_conflict", "extraction_version")
            return
        self.connection.execute(
            """
            INSERT INTO extraction_versions (
                version_id, model_version, prompt_version, prompt_hash,
                schema_version, schema_hash, registry_version, registry_hash,
                input_manifest_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                extraction_version_id,
                *expected,
                created_at,
            ),
        )

    def _ensure_claim(
        self,
        planned: DurativePropositionPlan,
        episodes: tuple[DurativeEpisode, ...],
    ) -> tuple[str, str, bool]:
        claim = planned.claim
        assert claim is not None
        expected = _claim_row(claim)
        claim_id = self._resolve_claim_id(claim, expected)
        row = self.connection.execute(
            """
            SELECT subject_id, speaker_id, predicate,
                   predicate_registry_version, object_json, polarity,
                   epistemic_status, valid_from_date, valid_from_timestamp,
                   valid_to_date, valid_to_timestamp, time_precision,
                   extraction_confidence, memory_kind, sensitivity,
                   extraction_version_id
            FROM claims WHERE user_id = %s AND claim_id = %s FOR UPDATE
            """,
            (claim.user_id, claim_id),
        ).fetchone()
        created = row is None
        if created:
            self.connection.execute(
                """
                INSERT INTO claims (
                    claim_id, user_id, subject_id, speaker_id, predicate,
                    predicate_registry_version, object_json, polarity,
                    epistemic_status, valid_from_date, valid_from_timestamp,
                    valid_to_date, valid_to_timestamp, time_precision,
                    extraction_confidence, memory_kind, sensitivity,
                    extraction_version_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (claim_id, claim.user_id, *expected[:4], Jsonb(expected[4]), *expected[5:]),
            )
        elif (
            _claim_semantic(row) != _claim_semantic(expected)
            or _claim_time(row) != _claim_time(expected)
        ):
            raise DurativePersistenceError("stable_id_conflict", "claim")

        episode_by_id = {item.episode_id: item for item in episodes}
        evidence_by_span: dict[str, float] = {}
        for item in claim.evidence:
            source = episode_by_id[item.episode_id]
            evidence_by_span[item.span_id] = min(
                evidence_by_span.get(item.span_id, source.extraction_confidence),
                source.extraction_confidence,
            )
        stored_evidence = dict(
            self.connection.execute(
                """
                SELECT span_id, extraction_confidence FROM evidence_links
                WHERE user_id = %s AND claim_id = %s
                  AND support_type = 'supports'
                """,
                (claim.user_id, claim_id),
            ).fetchall()
        )
        for span_id, confidence in stored_evidence.items():
            if span_id in evidence_by_span and evidence_by_span[span_id] != confidence:
                raise DurativePersistenceError("stable_id_conflict", "evidence")
        added_spans = tuple(sorted(set(evidence_by_span) - set(stored_evidence)))

        open_version = self.connection.execute(
            """
            SELECT version_id, lifecycle_status, transaction_from,
                   belief_confidence, valid_from_date, valid_from_timestamp,
                   valid_to_date, valid_to_timestamp, time_precision
            FROM claim_versions
            WHERE user_id = %s AND claim_id = %s AND transaction_to IS NULL
            FOR UPDATE
            """,
            (claim.user_id, claim_id),
        ).fetchone()
        version_shape = (
            "candidate", claim.belief_confidence,
            claim.valid_from_date, claim.valid_from_timestamp,
            claim.valid_to_date, claim.valid_to_timestamp, claim.time_precision,
        )
        changed = created
        if open_version is None or added_spans:
            if open_version is not None:
                if open_version[1] != "candidate" or open_version[3:] != version_shape[1:]:
                    raise DurativePersistenceError("stable_id_conflict", "claim_version")
                if planned.request.transaction_as_of <= open_version[2]:
                    raise DurativePersistenceError("stale_plan", "claim_version")
                updated = self.connection.execute(
                    """
                    UPDATE claim_versions SET transaction_to = %s
                    WHERE user_id = %s AND claim_id = %s
                      AND version_id = %s AND transaction_to IS NULL
                    """,
                    (
                        planned.request.transaction_as_of, claim.user_id,
                        claim_id, open_version[0],
                    ),
                ).rowcount
                if updated != 1:
                    raise DurativePersistenceError("concurrent_write", "claim_version")
            version_id = stable_id(
                "durative_candidate_version", claim_id,
                planned.input_snapshot_sha256,
            )
            self.connection.execute(
                """
                INSERT INTO claim_versions (
                    version_id, user_id, claim_id, lifecycle_status,
                    transaction_from, transaction_to, belief_confidence,
                    valid_from_date, valid_from_timestamp, valid_to_date,
                    valid_to_timestamp, time_precision
                ) VALUES (%s, %s, %s, 'candidate', %s, NULL, %s, %s, %s, %s, %s, %s)
                """,
                (
                    version_id, claim.user_id, claim_id,
                    planned.request.transaction_as_of, claim.belief_confidence,
                    claim.valid_from_date, claim.valid_from_timestamp,
                    claim.valid_to_date, claim.valid_to_timestamp,
                    claim.time_precision,
                ),
            )
            changed = True
        else:
            if open_version[1] != "candidate" or open_version[3:] != version_shape[1:]:
                raise DurativePersistenceError("stable_id_conflict", "claim_version")
            if open_version[2] > planned.request.transaction_as_of:
                raise DurativePersistenceError("stale_plan", "claim_version")
            version_id = open_version[0]

        for span_id in added_spans:
            self.connection.execute(
                """
                INSERT INTO evidence_links (
                    user_id, claim_id, span_id, support_type,
                    extraction_confidence
                ) VALUES (%s, %s, %s, 'supports', %s)
                """,
                (claim.user_id, claim_id, span_id, evidence_by_span[span_id]),
            )
        return claim_id, version_id, changed

    def _resolve_claim_id(
        self,
        claim: DurativeClaimPlan,
        expected: tuple[object, ...],
    ) -> str:
        root = self.connection.execute(
            """
            SELECT subject_id, speaker_id, predicate,
                   predicate_registry_version, object_json, polarity,
                   epistemic_status, valid_from_date, valid_from_timestamp,
                   valid_to_date, valid_to_timestamp, time_precision,
                   extraction_confidence, memory_kind, sensitivity,
                   extraction_version_id
            FROM claims WHERE user_id = %s AND claim_id = %s FOR UPDATE
            """,
            (claim.user_id, claim.claim_id),
        ).fetchone()
        if root is None or _claim_time(root) == _claim_time(expected):
            return claim.claim_id
        if _claim_semantic(root) != _claim_semantic(expected):
            raise DurativePersistenceError("stable_id_conflict", "claim")
        return stable_id(
            "durative_replacement_claim", claim.claim_id,
            canonical_json(_claim_time(expected)),
        )

    def _insert_run(
        self,
        request: DurativeInferenceRequest,
        run_id: str,
        run_snapshot: str,
        persisted: tuple[
            tuple[DurativePropositionPlan, str | None, str | None], ...
        ],
        episodes: tuple[DurativeEpisode, ...],
    ) -> None:
        extraction_version_id = _inference_extraction_id(self.config_path, self.config)
        accepted_count = sum(
            plan.decision.status == "accepted" for plan, _claim, _version in persisted
        )
        self.connection.execute(
            """
            INSERT INTO durative_inference_runs (
                run_id, user_id, idempotency_key, rules_version,
                input_snapshot_sha256, extraction_version_id,
                transaction_as_of, created_at, decision_count,
                accepted_count, rejected_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id, request.user_id, request.idempotency_key,
                request.rule_version, run_snapshot, extraction_version_id,
                request.transaction_as_of, request.transaction_as_of,
                len(persisted), accepted_count, len(persisted) - accepted_count,
            ),
        )
        by_id = {item.episode_id: item for item in episodes}
        for planned, claim_id, version_id in persisted:
            self.connection.execute(
                """
                INSERT INTO durative_inference_decisions (
                    decision_id, user_id, run_id, decision_status,
                    decision_reason, derived_claim_id, derived_version_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    planned.decision.decision_id, request.user_id, run_id,
                    planned.decision.status, planned.decision.reason,
                    claim_id, version_id,
                ),
            )
            evidence_roles = (
                *((item, "supports") for item in planned.decision.support_episode_ids),
                *((item, "counter_evidence") for item in planned.decision.counter_episode_ids),
            )
            for position, (episode_id, role) in enumerate(sorted(evidence_roles)):
                item = by_id[episode_id]
                self.connection.execute(
                    """
                    INSERT INTO durative_inference_evidence (
                        user_id, decision_id, episode_id, role,
                        evidence_order, supporting_claim_id,
                        supporting_version_id, session_definition_id,
                        source_id, span_id, support_type
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        request.user_id, planned.decision.decision_id,
                        item.episode_id, role, position, item.claim_id,
                        item.claim_version_id, item.session_definition_id,
                        item.source_id, item.span_id, item.support_type,
                    ),
                )

    def _run_matches(
        self,
        request: DurativeInferenceRequest,
        run_id: str,
        run_snapshot: str,
        plans: tuple[DurativePropositionPlan, ...],
        episodes: tuple[DurativeEpisode, ...],
    ) -> tuple[object, ...] | None:
        row = self.connection.execute(
            """
            SELECT idempotency_key, rules_version, input_snapshot_sha256,
                   extraction_version_id, transaction_as_of, decision_count,
                   accepted_count, rejected_count
            FROM durative_inference_runs
            WHERE user_id = %s AND run_id = %s
            """,
            (request.user_id, run_id),
        ).fetchone()
        if row is None:
            return None
        accepted_count = sum(plan.decision.status == "accepted" for plan in plans)
        if row != (
            request.idempotency_key, request.rule_version, run_snapshot,
            _inference_extraction_id(self.config_path, self.config),
            request.transaction_as_of, len(plans), accepted_count,
            len(plans) - accepted_count,
        ):
            return None
        stored_decisions = tuple(
            self.connection.execute(
                """
                SELECT decision_id, decision_status, decision_reason,
                       derived_claim_id, derived_version_id
                FROM durative_inference_decisions
                WHERE user_id = %s AND run_id = %s ORDER BY decision_id
                """,
                (request.user_id, run_id),
            ).fetchall()
        )
        if len(stored_decisions) != len(plans):
            return None
        by_id = {item.episode_id: item for item in episodes}
        result: list[object] = []
        for planned, stored in zip(plans, stored_decisions, strict=True):
            if stored[:3] != (
                planned.decision.decision_id, planned.decision.status,
                planned.decision.reason,
            ):
                return None
            claim_id, version_id = stored[3:]
            expected_evidence = []
            for episode_id in planned.decision.support_episode_ids:
                expected_evidence.append((episode_id, "supports"))
            for episode_id in planned.decision.counter_episode_ids:
                expected_evidence.append((episode_id, "counter_evidence"))
            expected_rows = tuple(
                (
                    item.episode_id, role, item.claim_id,
                    item.claim_version_id, item.session_definition_id,
                    item.source_id, item.span_id, item.support_type,
                )
                for episode_id, role in sorted(expected_evidence)
                for item in (by_id[episode_id],)
            )
            evidence_rows = tuple(self.connection.execute(
                """
                SELECT episode_id, role, supporting_claim_id,
                       supporting_version_id, session_definition_id,
                       source_id, span_id, support_type
                FROM durative_inference_evidence
                WHERE user_id = %s AND decision_id = %s ORDER BY evidence_order
                """,
                (request.user_id, planned.decision.decision_id),
            ).fetchall())
            if evidence_rows != expected_rows:
                return None
            if planned.claim is None:
                if claim_id is not None or version_id is not None:
                    return None
            elif not self._stored_claim_matches(planned, claim_id, version_id):
                return None
            result.append(replace(planned.decision, claim_id=claim_id))
        return tuple(result)

    def _stored_claim_matches(
        self,
        planned: DurativePropositionPlan,
        claim_id: str | None,
        version_id: str | None,
    ) -> bool:
        if claim_id is None or version_id is None or planned.claim is None:
            return False
        persisted_claim = self.connection.execute(
            """
            SELECT subject_id, speaker_id, predicate,
                   predicate_registry_version, object_json, polarity,
                   epistemic_status, valid_from_date, valid_from_timestamp,
                   valid_to_date, valid_to_timestamp, time_precision,
                   extraction_confidence, memory_kind, sensitivity,
                   extraction_version_id
            FROM claims WHERE user_id = %s AND claim_id = %s
            """,
            (planned.claim.user_id, claim_id),
        ).fetchone()
        return bool(
            persisted_claim is not None
            and _claim_semantic(persisted_claim) == _claim_semantic(_claim_row(planned.claim))
            and _claim_time(persisted_claim) == _claim_time(_claim_row(planned.claim))
            and self.connection.execute(
                """
                SELECT 1 FROM claim_versions
                WHERE user_id = %s AND claim_id = %s AND version_id = %s
                """,
                (planned.claim.user_id, claim_id, version_id),
            ).fetchone() == (1,)
        )

    def _insert_claims_changed(
        self,
        request: DurativeInferenceRequest,
        run_id: str,
        claim_id: str,
    ) -> None:
        event_id = stable_id("durative_claims_changed", run_id, claim_id)
        self.connection.execute(
            """
            INSERT INTO processing_outbox (
                event_id, user_id, event_type, aggregate_id, dedupe_key,
                payload, state, created_at, published_at
            ) VALUES (%s, %s, 'claims_changed', %s, %s, %s, 'pending', %s, NULL)
            ON CONFLICT (user_id, dedupe_key) DO NOTHING
            """,
            (
                event_id,
                request.user_id,
                claim_id,
                f"durative_claims_changed:{run_id}:{claim_id}",
                Jsonb({"claim_id": claim_id, "rules_version": RULES_VERSION}),
                request.transaction_as_of,
            ),
        )

    def _delete_derived_claim(self, user_id: str, claim_id: str) -> None:
        self.connection.execute(
            """
            DELETE FROM durative_inference_runs AS run
            USING durative_inference_decisions AS decision
            WHERE run.user_id = %s
              AND decision.user_id = run.user_id
              AND decision.run_id = run.run_id
              AND decision.derived_claim_id = %s
            """,
            (user_id, claim_id),
        )
        self.connection.execute(
            """
            DELETE FROM processing_outbox
            WHERE user_id = %s AND aggregate_id = %s
              AND event_type IN ('claims_changed', 'claim_lifecycle_changed')
            """,
            (user_id, claim_id),
        )
        self.connection.execute(
            "DELETE FROM evidence_links WHERE user_id = %s AND claim_id = %s",
            (user_id, claim_id),
        )
        self.connection.execute(
            "DELETE FROM claim_versions WHERE user_id = %s AND claim_id = %s",
            (user_id, claim_id),
        )
        self.connection.execute(
            "DELETE FROM claims WHERE user_id = %s AND claim_id = %s",
            (user_id, claim_id),
        )

    def _lock(self, user_id: str) -> None:
        self.connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (f"durative:{self.config.rules_version}:{user_id}",),
        )


class DurativeClaimCoordinator:
    """Recompute one user's durative candidates from an existing outbox event."""

    def __init__(
        self,
        connection: object,
        *,
        config_path: str | Path = DEFAULT_DURATIVE_RULES_PATH,
    ) -> None:
        self.connection = connection
        self.repository = DurativeClaimRepository(connection, config_path=config_path)
        self.config_path = Path(config_path)
        self.registry = load_predicate_registry(REPO_ROOT / REGISTRY_PATH)

    def infer(self, request: DurativeInferenceRequest) -> DurativeInferenceResult:
        """Persist one atomic, user-scoped inference run."""

        with self.connection.transaction():
            episodes = self.repository.load_visible_episodes(
                request.user_id, request.transaction_as_of,
            )
            propositions = _candidate_propositions(episodes, self.registry)
            plans = tuple(
                infer_durative_claim(
                    DurativePropositionRequest(
                        user_id=request.user_id,
                        subject_id=proposition[0],
                        predicate=proposition[1],
                        predicate_registry_version=proposition[2],
                        object_json=proposition[3],
                        polarity=proposition[4],
                        transaction_as_of=request.transaction_as_of,
                        idempotency_key=(
                            f"{request.idempotency_key}:"
                            f"{stable_id('durative_proposition', *proposition)}"
                        ),
                    ),
                    tuple(
                        item for item in episodes
                        if item.subject_id == proposition[0]
                        and item.predicate == proposition[1]
                        and item.predicate_registry_version == proposition[2]
                    ),
                    config_path=self.config_path,
                )
                for proposition in propositions
            )
            result = self.repository.persist_in_transaction(
                request, plans, episodes,
            )
            self.repository.purge_unmatched(
                request.user_id,
                tuple(
                    item.claim_id for item in result.decisions
                    if item.claim_id is not None
                ),
            )
        return result

    def process(self, event: ProcessingOutboxRecord) -> DurativeInferenceResult:
        if event.event_type not in DURATIVE_RECOMPUTE_EVENT_TYPES:
            raise DurativePersistenceError("unsupported_event", "event_type")
        return self.infer(
            DurativeInferenceRequest(
                user_id=event.user_id,
                transaction_as_of=event.created_at,
                idempotency_key=f"durative:{event.event_id}",
                rule_version=RULES_VERSION,
            )
        )


def _candidate_propositions(
    episodes: tuple[DurativeEpisode, ...],
    registry: object,
) -> tuple[tuple[str, str, str, object, str], ...]:
    by_key: dict[str, tuple[str, str, str, object, str]] = {}
    for item in episodes:
        definition = registry.by_predicate.get(item.predicate)
        if (
            item.support_type != "supports"
            or definition is None
            or definition.temporal_behavior != "interval"
            or definition.family not in ALLOWED_FAMILIES
        ):
            continue
        value = (
            item.subject_id,
            item.predicate,
            item.predicate_registry_version,
            item.object_json,
            item.polarity,
        )
        by_key[canonical_json(value)] = value
    return tuple(by_key[key] for key in sorted(by_key))


def _episode_time(row: dict[str, object]) -> date | datetime | None:
    if row["time_precision"] == "timestamp":
        return row["valid_from_timestamp"] or row["valid_to_timestamp"]
    if row["time_precision"] != "unknown":
        return row["valid_from_date"] or row["valid_to_date"]
    return None


def _claim_row(claim: DurativeClaimPlan) -> tuple[object, ...]:
    return (
        claim.subject_id,
        claim.speaker_id,
        claim.predicate,
        claim.predicate_registry_version,
        claim.object_json,
        claim.polarity,
        claim.epistemic_status,
        claim.valid_from_date,
        claim.valid_from_timestamp,
        claim.valid_to_date,
        claim.valid_to_timestamp,
        claim.time_precision,
        claim.extraction_confidence,
        claim.memory_kind,
        claim.sensitivity,
        claim.extraction.extraction_version_id,
    )


def _claim_semantic(row: tuple[object, ...]) -> tuple[object, ...]:
    return (*row[:7], row[13])


def _claim_time(row: tuple[object, ...]) -> tuple[object, ...]:
    return row[7:12]


def _run_snapshot(
    request: DurativeInferenceRequest,
    episodes: tuple[DurativeEpisode, ...],
    plans: tuple[DurativePropositionPlan, ...],
) -> str:
    payload = {
        "schema_version": "durative_inference_run_snapshot_v1",
        "request": asdict(request),
        "episodes": [asdict(item) for item in episodes],
        "decisions": [asdict(item.decision) for item in plans],
        "proposition_snapshots": [item.input_snapshot_sha256 for item in plans],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _inference_extraction_id(config_path: Path, config: object) -> str:
    return stable_id(
        "durative_extraction_version", RULES_VERSION,
        _file_sha256(config_path),
        config.predicate_registry_version,
        config.predicate_registry_sha256,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
