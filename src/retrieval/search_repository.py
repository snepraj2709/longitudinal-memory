"""User-first PostgreSQL search for deterministic B2, B3, and B4 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import re
from typing import Mapping, Sequence

from .baseline_contracts import (
    BaselineRetrievalError,
    BaselineRetrievalRequest,
    BaselineRetrievalResult,
    SearchChannelHit,
)
from .baselines import (
    AtomicExpansionSeed,
    BaselineConfig,
    OldVersionNeighbor,
    RelationNeighbor,
    RetrievalRecordMetadata,
    build_baseline_request,
    build_old_version_paths,
    build_relation_paths,
    build_result,
    fixed_decimal,
    fuse_channels,
    load_baseline_config,
    rerank_candidates,
)
from .embeddings import DeterministicTokenHashEmbedder, Embedder
from .query_contracts import RetrievalQueryRequest
from .query_planner import (
    QueryPlannerConfig,
    build_query_plan,
    load_query_planner_config,
)
from .query_repository import RetrievalQueryRepository, RetrievalQueryRepositoryError


SAFE_FAILURE = re.compile(r"^[a-z0-9_]{1,64}$")
ALLOWED_RELATIONS = ("corrects", "supersedes", "contradicts", "refines", "same_event_as")


class RetrievalSearchRepositoryError(RuntimeError):
    """A sanitized, fail-closed baseline-search error."""

    def __init__(self, code: str, location: str) -> None:
        if SAFE_FAILURE.fullmatch(code) is None or SAFE_FAILURE.fullmatch(location) is None:
            raise ValueError("search error fields must be sanitized")
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


@dataclass(frozen=True)
class _BoundRecord:
    metadata: RetrievalRecordMetadata
    atomic_claim_id: str | None
    transaction_from: datetime


class RetrievalSearchRepository:
    """Execute retrieval inside one read-only repeatable-read snapshot."""

    RUN_SQL = """
        SELECT transaction_as_of, status
        FROM retrieval_index_runs
        WHERE user_id = %s
          AND index_version = %s
          AND run_id = %s
    """

    BINDING_SQL = """
        SELECT
            record.index_record_id,
            record.record_kind,
            record.lifecycle_statuses,
            record.sensitivity,
            record.atomic_claim_id,
            record.atomic_claim_version_id,
            record.session_summary_id,
            record.content_sha256,
            record.transaction_from
        FROM retrieval_index_records AS record
        WHERE record.user_id = %s
          AND record.index_version = %s
          AND record.run_id = %s
          AND record.record_kind = ANY(%s::text[])
        ORDER BY record.index_record_id
    """

    CLAIM_LINEAGE_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = ANY(%s::text[])
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT link.index_record_id, link.claim_id, link.claim_version_id
        FROM eligible_records AS record
        JOIN retrieval_index_claim_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN claim_versions AS version
          ON version.user_id = link.user_id
         AND version.claim_id = link.claim_id
         AND version.version_id = link.claim_version_id
        ORDER BY link.index_record_id, link.claim_order
    """

    SOURCE_LINEAGE_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = ANY(%s::text[])
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT
            link.index_record_id,
            link.claim_id,
            link.claim_version_id,
            link.source_id,
            link.span_id,
            source.ingested_at
        FROM eligible_records AS record
        JOIN retrieval_index_source_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN source_events AS source
          ON source.user_id = link.user_id
         AND source.source_id = link.source_id
        JOIN source_spans AS span
          ON span.user_id = link.user_id
         AND span.source_id = link.source_id
         AND span.span_id = link.span_id
        ORDER BY link.index_record_id, link.source_order
    """

    RELATION_SAFETY_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = ANY(%s::text[])
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT
            link.index_record_id,
            relation.created_at,
            decision.transaction_as_of,
            decision.classified_at
        FROM eligible_records AS record
        JOIN retrieval_index_relation_links AS link
          ON link.user_id = %s
         AND link.index_record_id = record.index_record_id
        JOIN claim_relations AS relation
          ON relation.user_id = link.user_id
         AND relation.relation_id = link.relation_id
         AND relation.source_claim_id = link.source_claim_id
         AND relation.target_claim_id = link.target_claim_id
         AND relation.relation_type = link.relation_type
        JOIN conflict_decisions AS decision
          ON decision.user_id = relation.user_id
         AND decision.decision_id = relation.decision_id
        ORDER BY link.index_record_id, link.relation_order
    """

    FTS_SHAPE_SQL = "SELECT numnode(plainto_tsquery('simple', %s))"

    LEXICAL_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.*
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = %s
              AND record.index_record_id = ANY(%s::text[])
        ), query AS (
            SELECT plainto_tsquery('simple', %s) AS value
        )
        SELECT
            record.index_record_id,
            ts_rank_cd(record.search_document, query.value, 32)::double precision AS score
        FROM eligible_records AS record
        CROSS JOIN query
        WHERE record.search_document @@ query.value
        ORDER BY score DESC, record.index_record_id ASC
        LIMIT 40
    """

    VECTOR_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.*
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = %s
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT
            record.index_record_id,
            (record.embedding <=> %s::vector)::double precision AS distance
        FROM eligible_records AS record
        ORDER BY distance ASC, record.index_record_id ASC
        LIMIT 40
    """

    RELATIONS_SQL = """
        WITH eligible_records AS MATERIALIZED (
            SELECT record.index_record_id
            FROM retrieval_index_records AS record
            WHERE record.user_id = %s
              AND record.index_version = %s
              AND record.run_id = %s
              AND record.record_kind = 'atomic'
              AND record.index_record_id = ANY(%s::text[])
        )
        SELECT
            link.index_record_id,
            link.relation_id,
            link.source_claim_id,
            link.target_claim_id,
            link.relation_type,
            link.direction
        FROM eligible_records AS seed
        JOIN retrieval_index_relation_links AS link
          ON link.user_id = %s
         AND link.index_record_id = seed.index_record_id
        WHERE link.index_record_id = ANY(%s::text[])
          AND link.relation_type = ANY(%s::text[])
        ORDER BY link.index_record_id, link.relation_order
    """

    def __init__(
        self,
        connection: object,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.connection = connection
        self.embedder = embedder or DeterministicTokenHashEmbedder()

    def retrieve(
        self,
        request: RetrievalQueryRequest,
        baseline_id: str,
        *,
        planner_config: QueryPlannerConfig | None = None,
        baseline_config: BaselineConfig | None = None,
    ) -> BaselineRetrievalResult:
        """Plan, filter, and search in the same database transaction."""

        planner = planner_config or load_query_planner_config()
        ranking = baseline_config or load_baseline_config()
        try:
            with self.connection.transaction():
                self.connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                plan = build_query_plan(request, config=planner)
                eligibility = RetrievalQueryRepository(self.connection).filter(plan)
                execution = build_baseline_request(
                    request,
                    plan,
                    eligibility,
                    baseline_id,
                    config=ranking,
                )
                return self._execute_bound(execution)
        except RetrievalSearchRepositoryError:
            raise
        except RetrievalQueryRepositoryError as error:
            raise RetrievalSearchRepositoryError("stale_snapshot", "eligibility") from error
        except BaselineRetrievalError as error:
            raise RetrievalSearchRepositoryError("invalid_request", "request") from error
        except Exception as error:
            raise RetrievalSearchRepositoryError("search_failed", "database") from error

    def execute(self, execution: BaselineRetrievalRequest) -> BaselineRetrievalResult:
        """Search a precomputed request after revalidating every snapshot binding."""

        try:
            with self.connection.transaction():
                self.connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                return self._execute_bound(execution)
        except RetrievalSearchRepositoryError:
            raise
        except BaselineRetrievalError as error:
            raise RetrievalSearchRepositoryError("invalid_request", "request") from error
        except Exception as error:
            raise RetrievalSearchRepositoryError("search_failed", "database") from error

    def _execute_bound(self, execution: BaselineRetrievalRequest) -> BaselineRetrievalResult:
        bound = self._bind_snapshot(execution)
        eligible_ids = execution.eligibility.eligible_record_ids
        hits, notices = self._search_channels(execution, eligible_ids)
        attributes = {
            record_id: (
                item.metadata.record_kind,
                item.metadata.lifecycle_statuses,
            )
            for record_id, item in bound.items()
        }
        initial = rerank_candidates(
            fuse_channels(hits, (), attributes), execution.plan.primary_label
        )
        seeds = self._atomic_seeds(initial, bound)
        expansions = []
        if execution.plan.include_previous_versions:
            neighbors = tuple(
                OldVersionNeighbor(
                    record_id,
                    item.atomic_claim_id,
                    item.transaction_from,
                )
                for record_id, item in bound.items()
                if item.metadata.record_kind == "atomic"
                and item.atomic_claim_id is not None
            )
            expansions.extend(build_old_version_paths(seeds, neighbors))
        if execution.plan.checked_relation_expansion_intent:
            if "atomic" not in execution.plan.enabled_record_kinds:
                notices.add("relation_expansion_not_applicable")
            elif seeds:
                expansions.extend(
                    build_relation_paths(
                        self._relation_neighbors(execution, seeds, bound)
                    )
                )
        final = rerank_candidates(
            fuse_channels(hits, tuple(expansions), attributes),
            execution.plan.primary_label,
        )
        return build_result(
            execution,
            final,
            {key: value.metadata for key, value in bound.items()},
            tuple(notices),
        )

    def _bind_snapshot(
        self, execution: BaselineRetrievalRequest
    ) -> Mapping[str, _BoundRecord]:
        run = self.connection.execute(
            self.RUN_SQL,
            (
                execution.request.user_id,
                execution.request.index_version,
                execution.eligibility.snapshot_run_id,
            ),
        ).fetchone()
        if run is None or run[1] != "succeeded" or run[0] > execution.request.as_of:
            raise RetrievalSearchRepositoryError("stale_snapshot", "snapshot")

        rows = self.connection.execute(
            self.BINDING_SQL,
            (
                execution.request.user_id,
                execution.request.index_version,
                execution.eligibility.snapshot_run_id,
                list(execution.request.enabled_record_kinds),
            ),
        ).fetchall()
        decision_by_id = {
            item.index_record_id: item for item in execution.eligibility.decisions
        }
        row_ids = tuple(str(row[0]) for row in rows)
        if row_ids != tuple(decision_by_id):
            raise RetrievalSearchRepositoryError("stale_snapshot", "eligibility")
        for row in rows:
            decision = decision_by_id[str(row[0])]
            lifecycle = _string_tuple(row[2], "lifecycle")
            if (
                lifecycle != decision.lifecycle_statuses
                or row[3] != decision.sensitivity
                or (row[3] is None) != decision.unclassified_sensitivity
            ):
                raise RetrievalSearchRepositoryError("stale_snapshot", "eligibility")

        eligible_ids = execution.eligibility.eligible_record_ids
        if not eligible_ids:
            return {}
        common = (
            execution.request.user_id,
            execution.request.index_version,
            execution.eligibility.snapshot_run_id,
            list(execution.request.enabled_record_kinds),
            list(eligible_ids),
        )
        claim_rows = self.connection.execute(
            self.CLAIM_LINEAGE_SQL,
            (*common, execution.request.user_id),
        ).fetchall()
        source_rows = self.connection.execute(
            self.SOURCE_LINEAGE_SQL,
            (*common, execution.request.user_id),
        ).fetchall()
        relation_rows = self.connection.execute(
            self.RELATION_SAFETY_SQL,
            (*common, execution.request.user_id),
        ).fetchall()

        claims: dict[str, list[tuple[str, str]]] = {}
        for record_id, claim_id, version_id in claim_rows:
            claims.setdefault(str(record_id), []).append((str(claim_id), str(version_id)))
        sources: dict[str, list[tuple[str, str]]] = {}
        source_claim_versions: dict[str, set[tuple[str, str]]] = {}
        for record_id, claim_id, version_id, source_id, span_id, ingested_at in source_rows:
            if ingested_at > execution.request.as_of:
                raise RetrievalSearchRepositoryError("stale_snapshot", "source")
            key = str(record_id)
            sources.setdefault(key, []).append((str(source_id), str(span_id)))
            source_claim_versions.setdefault(key, set()).add((str(claim_id), str(version_id)))
        for _, created_at, transaction_as_of, classified_at in relation_rows:
            if any(
                value > execution.request.as_of
                for value in (created_at, transaction_as_of, classified_at)
            ):
                raise RetrievalSearchRepositoryError("stale_snapshot", "relation")

        bound: dict[str, _BoundRecord] = {}
        row_by_id = {str(row[0]): row for row in rows}
        for record_id in eligible_ids:
            row = row_by_id.get(record_id)
            claim_values = claims.get(record_id, [])
            source_values = sources.get(record_id, [])
            if (
                row is None
                or not claim_values
                or not source_values
                or set(claim_values) != source_claim_versions.get(record_id, set())
                or len({value[0] for value in claim_values}) != len(claim_values)
                or len({value[1] for value in claim_values}) != len(claim_values)
            ):
                raise RetrievalSearchRepositoryError("lineage_incomplete", "lineage")
            source_ids = tuple(dict.fromkeys(value[0] for value in source_values))
            span_ids = tuple(dict.fromkeys(value[1] for value in source_values))
            metadata = RetrievalRecordMetadata(
                record_id,
                str(row[1]),
                str(row[5]) if row[5] is not None else None,
                str(row[6]) if row[6] is not None else None,
                str(row[7]),
                _string_tuple(row[2], "lifecycle"),
                tuple(value[0] for value in claim_values),
                tuple(value[1] for value in claim_values),
                source_ids,
                span_ids,
            )
            bound[record_id] = _BoundRecord(
                metadata,
                str(row[4]) if row[4] is not None else None,
                row[8],
            )
        return bound

    def _search_channels(
        self,
        execution: BaselineRetrievalRequest,
        eligible_ids: Sequence[str],
    ) -> tuple[tuple[SearchChannelHit, ...], set[str]]:
        notices: set[str] = set()
        hits: list[SearchChannelHit] = []
        fts_nodes = self.connection.execute(
            self.FTS_SHAPE_SQL, (execution.request.query_text,)
        ).fetchone()[0]
        if fts_nodes == 0:
            notices.add("empty_fts_query")
        vector = self.embedder.embed(execution.request.query_text)
        if len(vector) != 256:
            raise RetrievalSearchRepositoryError("search_failed", "embedding")
        zero_vector = all(value == 0.0 for value in vector)
        if zero_vector:
            notices.add("empty_query_vector")
        vector_value = "[" + ",".join(str(value) for value in vector) + "]"
        for kind in execution.request.enabled_record_kinds:
            kind_ids = [
                item.index_record_id
                for item in execution.eligibility.decisions
                if item.eligible
            ]
            if not kind_ids:
                continue
            if fts_nodes:
                rows = self.connection.execute(
                    self.LEXICAL_SQL,
                    (
                        execution.request.user_id,
                        execution.request.index_version,
                        execution.eligibility.snapshot_run_id,
                        kind,
                        kind_ids,
                        execution.request.query_text,
                    ),
                ).fetchall()
                hits.extend(
                    SearchChannelHit(
                        str(record_id),
                        kind,
                        f"{kind}_lexical",
                        rank,
                        fixed_decimal(score),
                    )
                    for rank, (record_id, score) in enumerate(rows, start=1)
                )
            if not zero_vector:
                rows = self.connection.execute(
                    self.VECTOR_SQL,
                    (
                        execution.request.user_id,
                        execution.request.index_version,
                        execution.eligibility.snapshot_run_id,
                        kind,
                        kind_ids,
                        vector_value,
                    ),
                ).fetchall()
                hits.extend(
                    SearchChannelHit(
                        str(record_id),
                        kind,
                        f"{kind}_vector",
                        rank,
                        fixed_decimal(Decimal(1) - Decimal(str(distance))),
                    )
                    for rank, (record_id, distance) in enumerate(rows, start=1)
                )
        return tuple(hits), notices

    def _atomic_seeds(
        self,
        ranked: Sequence[object],
        bound: Mapping[str, _BoundRecord],
    ) -> tuple[AtomicExpansionSeed, ...]:
        seeds: list[AtomicExpansionSeed] = []
        atomic_candidates = tuple(
            candidate
            for candidate in ranked
            if bound[candidate.index_record_id].metadata.record_kind == "atomic"
        )
        for rank, candidate in enumerate(atomic_candidates[:10], start=1):
            item = bound[candidate.index_record_id]
            if item.atomic_claim_id is None or item.metadata.claim_ids != (item.atomic_claim_id,):
                raise RetrievalSearchRepositoryError("lineage_incomplete", "claim")
            seeds.append(
                AtomicExpansionSeed(candidate.index_record_id, rank, item.atomic_claim_id)
            )
        return tuple(seeds)

    def _relation_neighbors(
        self,
        execution: BaselineRetrievalRequest,
        seeds: Sequence[AtomicExpansionSeed],
        bound: Mapping[str, _BoundRecord],
    ) -> tuple[RelationNeighbor, ...]:
        seed_by_id = {item.index_record_id: item for item in seeds}
        rows = self.connection.execute(
            self.RELATIONS_SQL,
            (
                execution.request.user_id,
                execution.request.index_version,
                execution.eligibility.snapshot_run_id,
                list(execution.eligibility.eligible_record_ids),
                execution.request.user_id,
                list(seed_by_id),
                list(execution.plan.relation_expansion_types),
            ),
        ).fetchall()
        by_claim: dict[str, list[str]] = {}
        for record_id, item in bound.items():
            if item.metadata.record_kind == "atomic" and item.atomic_claim_id is not None:
                by_claim.setdefault(item.atomic_claim_id, []).append(record_id)
        neighbors: list[RelationNeighbor] = []
        for seed_id, relation_id, source_claim, target_claim, relation_type, stored_direction in rows:
            seed = seed_by_id[str(seed_id)]
            if relation_type not in ALLOWED_RELATIONS:
                continue
            if seed.claim_id == source_claim:
                neighbor_claim = str(target_claim)
                direction = "outgoing"
            elif seed.claim_id == target_claim:
                neighbor_claim = str(source_claim)
                direction = "incoming"
            else:
                raise RetrievalSearchRepositoryError("lineage_incomplete", "relation")
            if stored_direction not in {direction, "symmetric"}:
                raise RetrievalSearchRepositoryError("lineage_incomplete", "relation")
            trace_direction = "symmetric" if stored_direction == "symmetric" else direction
            for neighbor_id in sorted(by_claim.get(neighbor_claim, ())):
                if neighbor_id != seed.index_record_id:
                    neighbors.append(
                        RelationNeighbor(
                            seed.index_record_id,
                            seed.seed_rank,
                            neighbor_id,
                            str(relation_id),
                            str(relation_type),
                            trace_direction,
                        )
                    )
        return tuple(neighbors)


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RetrievalSearchRepositoryError("lineage_incomplete", location)
    return tuple(value)
