from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import unittest

import conflicts.resolver as resolver_module
from conflicts.resolver import (
    RESOLVER_VERSION,
    AuthorityEvidence,
    BeliefResolver,
    BeliefResolverError,
    PersistedResolutionInput,
    ResolutionRequest,
    load_resolver_config,
    resolve_belief,
)
from storage.contracts import (
    ClaimRecord,
    ClaimRelationRecord,
    ClaimVersionRecord,
    ConflictDecisionEvidenceRecord,
    ConflictDecisionRecord,
    StorageValidationError,
)
from temporal.contracts import TemporalClaim, VisibleEvidence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/conflicts/belief_resolver_v1.json"
UTC = timezone.utc
TX_FROM = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
RESOLVED_AT = datetime(2026, 6, 1, 0, 1, tzinfo=UTC)
SNAPSHOT = "a" * 64


def _claim(
    claim_id: str,
    *,
    user_id: str = "user_001",
    subject_id: str = "user_001",
    speaker_id: str = "user_001",
    predicate: str = "lives_in",
    object_json: object = "Delhi",
    epistemic_status: str = "asserted",
    status: str = "candidate",
    precision: str = "day",
    valid_from: date | datetime | None = date(2026, 1, 1),
    valid_to: date | datetime | None = date(2026, 12, 31),
    transaction_from: datetime = TX_FROM,
    transaction_to: datetime | None = None,
    sensitivity: str | None = "standard",
    extraction_confidence: float = 0.25,
    source_id: str | None = None,
    evidence: bool = True,
) -> TemporalClaim:
    source_id = source_id or f"source_{claim_id[-1]}"
    date_from = valid_from if type(valid_from) is date else None
    date_to = valid_to if type(valid_to) is date else None
    timestamp_from = valid_from if isinstance(valid_from, datetime) else None
    timestamp_to = valid_to if isinstance(valid_to, datetime) else None
    claim = ClaimRecord(
        claim_id=claim_id,
        user_id=user_id,
        subject_id=subject_id,
        speaker_id=speaker_id,
        predicate=predicate,
        predicate_registry_version="predicate_registry_v2",
        object_json=object_json,
        polarity="positive",
        epistemic_status=epistemic_status,
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=precision,
        extraction_confidence=extraction_confidence,
        memory_kind="durative",
        sensitivity=sensitivity,
        extraction_version_id="resolver_test_v1",
    )
    version = ClaimVersionRecord(
        version_id=f"version_{claim_id}",
        user_id=user_id,
        claim_id=claim_id,
        lifecycle_status=status,
        transaction_from=transaction_from,
        transaction_to=transaction_to,
        belief_confidence=None,
        valid_from_date=date_from,
        valid_from_timestamp=timestamp_from,
        valid_to_date=date_to,
        valid_to_timestamp=timestamp_to,
        time_precision=precision,
    )
    visible = (
        VisibleEvidence(
            source_id=source_id,
            span_id=f"span_{claim_id[-1]}",
            message_id=f"message_{claim_id[-1]}",
            speaker_id=speaker_id,
            quote="source-backed assertion",
        ),
    ) if evidence else ()
    return TemporalClaim(claim, version, visible)


def _persisted(
    label: str,
    *,
    left: TemporalClaim | None = None,
    right: TemporalClaim | None = None,
    authority: tuple[AuthorityEvidence, ...] = (),
    deleted_source_ids: tuple[str, ...] = (),
) -> PersistedResolutionInput:
    left = left or _claim("claim_a")
    right = right or _claim("claim_b", object_json="Mumbai")
    decision = ConflictDecisionRecord(
        decision_id="decision_1",
        user_id="user_001",
        classifier_version="relation_classifier_v1",
        rule_version="relation_rules_v1",
        pair_id="pair_1",
        left_claim_id=left.claim.claim_id,
        right_claim_id=right.claim.claim_id,
        left_version_id=left.version.version_id,
        right_version_id=right.version.version_id,
        input_snapshot_sha256=SNAPSHOT,
        transaction_as_of=AS_OF,
        matched_rule=label,
        label=label,
        classified_at=AS_OF,
    )
    relation_type = {
        "hard_contradiction": "contradicts",
        "source_disagreement": "contradicts",
        "temporal_change": "same_topic_as",
        "explicit_correction": "corrects",
        "retraction": "corrects",
        "refinement": "refines",
    }.get(label)
    relations: tuple[ClaimRelationRecord, ...] = ()
    if relation_type is not None:
        if relation_type in {"contradicts", "same_topic_as"}:
            source_id, target_id = left.claim.claim_id, right.claim.claim_id
        else:
            source_id, target_id = right.claim.claim_id, left.claim.claim_id
        relations = (
            ClaimRelationRecord(
                relation_id="relation_1",
                user_id="user_001",
                decision_id=decision.decision_id,
                classifier_version=decision.classifier_version,
                source_claim_id=source_id,
                target_claim_id=target_id,
                relation_type=relation_type,
                confidence=1,
                input_snapshot_sha256=SNAPSHOT,
                created_at=AS_OF,
            ),
        )
    decision_evidence = tuple(
        sorted(
            (
                ConflictDecisionEvidenceRecord(
                    decision_evidence_id=f"decision_evidence_{item.claim.claim_id}_{evidence.span_id}",
                    user_id="user_001",
                    decision_id=decision.decision_id,
                    claim_id=item.claim.claim_id,
                    span_id=evidence.span_id,
                    support_type="supports",
                    input_snapshot_sha256=SNAPSHOT,
                )
                for item in (left, right)
                for evidence in item.evidence
            ),
            key=lambda value: (
                value.claim_id,
                value.span_id,
                value.support_type,
                value.decision_evidence_id,
            ),
        )
    )
    sources = tuple(
        (source_id, datetime(2026, 5, 1, tzinfo=UTC))
        for source_id in sorted(
            {
                evidence.source_id
                for item in (left, right)
                for evidence in item.evidence
            }
        )
    )
    return PersistedResolutionInput(
        decision=decision,
        left=left,
        right=right,
        relations=relations,
        decision_evidence=decision_evidence,
        source_ingested_at=sources,
        authority=authority,
        deleted_source_ids=deleted_source_ids,
    )


def _request(
    *,
    valid_at: date | datetime | None = date(2026, 6, 1),
    idempotency_key: str = "resolve_1",
) -> ResolutionRequest:
    return ResolutionRequest(
        user_id="user_001",
        decision_id="decision_1",
        transaction_as_of=AS_OF,
        valid_at=valid_at,
        resolved_at=RESOLVED_AT,
        idempotency_key=idempotency_key,
        resolver_version=RESOLVER_VERSION,
    )


def _official(item: TemporalClaim, **changes: object) -> AuthorityEvidence:
    evidence = item.evidence[0]
    values = {
        "user_id": item.claim.user_id,
        "claim_id": item.claim.claim_id,
        "authority_kind": "official_record",
        "speaker_id": item.claim.speaker_id,
        "subject_id": item.claim.subject_id,
        "predicate": item.claim.predicate,
        "source_type": "email",
        "source_id": evidence.source_id,
        "span_ids": (evidence.span_id,),
        "source_ingested_at": datetime(2026, 5, 1, tzinfo=UTC),
    }
    values.update(changes)
    return AuthorityEvidence(**values)


class BeliefResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_resolver_config(CONFIG_PATH, repo_root=ROOT)
        cls.resolver = BeliefResolver(repo_root=ROOT)

    def plan(
        self,
        label: str,
        *,
        request: ResolutionRequest | None = None,
        persisted: PersistedResolutionInput | None = None,
    ):
        return resolve_belief(
            self.config,
            request or _request(),
            persisted or _persisted(label),
        )

    def test_frozen_config_request_shape_and_dependency_bindings(self) -> None:
        self.assertEqual(
            {value.name for value in fields(ResolutionRequest)},
            {
                "user_id", "decision_id", "transaction_as_of", "valid_at",
                "resolved_at", "idempotency_key", "resolver_version",
            },
        )
        self.assertEqual(self.config.resolver_version, RESOLVER_VERSION)
        self.assertEqual(self.config.labels, resolver_module.CONFLICT_LABELS)
        self.assertEqual(self.config.belief_confidence, "null_only")
        self.assertEqual(
            hashlib.sha256(
                (ROOT / self.config.classifier_config_path).read_bytes()
            ).hexdigest(),
            self.config.classifier_config_sha256,
        )
        self.assertEqual(self.resolver.config, self.config)

    def test_unrelated_is_no_change_and_preserves_provenance(self) -> None:
        plan = self.plan("unrelated")
        self.assertEqual(plan.outcome, "no_change")
        self.assertEqual(plan.actions, ())
        self.assertEqual(plan.relations, ())
        self.assertEqual(plan.selected_current_claim_id, None)
        self.assertEqual(plan.belief_confidence, None)
        self.assertEqual(plan.decision_evidence_ids, tuple(sorted(plan.decision_evidence_ids)))
        self.assertEqual(plan.source_ids, ("source_a", "source_b"))

    def test_temporal_change_plans_candidate_historical_then_current(self) -> None:
        earlier = _claim(
            "claim_a", valid_from=date(2026, 1, 1), valid_to=date(2026, 3, 31)
        )
        later = _claim(
            "claim_b", object_json="Mumbai",
            valid_from=date(2026, 4, 1), valid_to=date(2026, 12, 31),
        )
        plan = self.plan(
            "temporal_change",
            persisted=_persisted("temporal_change", left=earlier, right=later),
        )
        self.assertEqual(
            [(value.claim_id, value.from_status, value.target_status) for value in plan.actions],
            [
                ("claim_a", "candidate", "historical"),
                ("claim_b", "candidate", "current"),
            ],
        )
        self.assertEqual(plan.selected_current_claim_id, "claim_b")
        self.assertEqual(plan.relations, ())

    def test_correction_orders_replacement_and_plans_supersedes(self) -> None:
        plan = self.plan("explicit_correction")
        self.assertEqual(
            [(value.claim_id, value.target_status) for value in plan.actions],
            [("claim_a", "superseded"), ("claim_b", "current")],
        )
        self.assertEqual(plan.actions[0].replacement_claim_id, "claim_b")
        self.assertEqual(plan.selected_current_claim_id, "claim_b")
        self.assertEqual(
            (
                plan.relations[0].source_claim_id,
                plan.relations[0].target_claim_id,
                plan.relations[0].relation_type,
            ),
            ("claim_b", "claim_a", "supersedes"),
        )

    def test_refinement_supersedes_broad_with_specific(self) -> None:
        plan = self.plan("refinement")
        self.assertEqual(plan.outcome, "refinement_resolved")
        self.assertEqual(plan.actions[0].claim_id, "claim_a")
        self.assertEqual(plan.actions[0].target_status, "superseded")
        self.assertEqual(plan.actions[1].claim_id, "claim_b")
        self.assertEqual(plan.relations[0].source_claim_id, "claim_b")

    def test_retraction_supersedes_proposition_and_confirms_event(self) -> None:
        plan = self.plan("retraction")
        self.assertEqual(plan.outcome, "retraction_resolved")
        self.assertEqual(
            [(value.claim_id, value.target_status) for value in plan.actions],
            [("claim_a", "superseded"), ("claim_b", "confirmed")],
        )
        self.assertEqual(plan.selected_current_claim_id, None)
        self.assertEqual(plan.relations, ())

    def test_hard_contradiction_without_decisive_authority_disputes_both(self) -> None:
        plan = self.plan("hard_contradiction")
        self.assertEqual(plan.outcome, "disputed")
        self.assertEqual(
            [(value.claim_id, value.target_status) for value in plan.actions],
            [("claim_a", "disputed"), ("claim_b", "disputed")],
        )
        self.assertEqual(plan.authority_reason, "no_exact_decisive_official_record")

    def test_source_disagreement_exact_official_record_selects_one(self) -> None:
        left = _claim("claim_a")
        right = _claim("claim_b", object_json="Mumbai", speaker_id="manager")
        persisted = _persisted(
            "source_disagreement",
            left=left,
            right=right,
            authority=(_official(left),),
        )
        plan = self.plan("source_disagreement", persisted=persisted)
        self.assertEqual(plan.outcome, "authority_resolved")
        self.assertEqual(plan.selected_current_claim_id, "claim_a")
        self.assertEqual(plan.actions[0].claim_id, "claim_b")
        self.assertEqual(plan.actions[0].target_status, "disputed")
        self.assertNotIn("superseded", {value.target_status for value in plan.actions})

    def test_unresolved_ambiguity_ignores_official_authority(self) -> None:
        left = _claim("claim_a", precision="unknown", valid_from=None, valid_to=None)
        right = _claim(
            "claim_b", object_json="Mumbai", precision="unknown",
            valid_from=None, valid_to=None,
        )
        authority = _official(left)
        persisted = _persisted(
            "unresolved_ambiguity", left=left, right=right,
            authority=(authority,),
        )
        plan = self.plan("unresolved_ambiguity", persisted=persisted)
        self.assertEqual(plan.outcome, "disputed")
        self.assertEqual(plan.selected_current_claim_id, None)
        self.assertEqual(
            plan.authority_reason, "authority_cannot_override_ambiguity"
        )

    def test_exclusions_run_first_in_frozen_precedence(self) -> None:
        cases = {
            "restricted": _claim("claim_a", sensitivity="restricted", subject_id="other"),
            "hypothetical": _claim("claim_a", epistemic_status="hypothetical"),
            "wrong_subject": _claim("claim_a", subject_id="other"),
            "deleted": _claim("claim_a"),
            "unsupported": _claim("claim_a", evidence=False),
        }
        for reason, left in cases.items():
            with self.subTest(reason=reason):
                deleted = ("source_a",) if reason == "deleted" else ()
                persisted = _persisted(
                    "hard_contradiction", left=left,
                    deleted_source_ids=deleted,
                )
                plan = self.plan("hard_contradiction", persisted=persisted)
                self.assertEqual(plan.outcome, "excluded")
                self.assertEqual(len(plan.actions), 1)
                self.assertEqual(plan.actions[0].target_status, "excluded")
                self.assertEqual(plan.actions[0].reason, f"exclusion:{reason}")
                self.assertEqual(plan.selected_current_claim_id, None)

    def test_authority_requires_exact_scope_time_source_and_spans(self) -> None:
        left = _claim("claim_a")
        right = _claim("claim_b", object_json="Mumbai", speaker_id="manager")
        changes = (
            {"speaker_id": "other"},
            {"subject_id": "other"},
            {"predicate": "office_base"},
            {"source_type": "conversation"},
            {"span_ids": ("wrong_span",)},
        )
        for values in changes:
            with self.subTest(values=values):
                persisted = _persisted(
                    "source_disagreement", left=left, right=right,
                    authority=(_official(left, **values),),
                )
                plan = self.plan("source_disagreement", persisted=persisted)
                self.assertEqual(plan.outcome, "disputed")
        stale = self.plan(
            "source_disagreement",
            request=_request(valid_at=date(2027, 1, 1)),
            persisted=_persisted(
                "source_disagreement", left=left, right=right,
                authority=(_official(left),),
            ),
        )
        self.assertEqual(stale.outcome, "disputed")

    def test_two_official_records_leave_dispute(self) -> None:
        left = _claim("claim_a")
        right = _claim("claim_b", object_json="Mumbai", speaker_id="manager")
        authority = tuple(
            sorted(
                (_official(left), _official(right)),
                key=lambda value: (
                    value.claim_id, value.source_id, value.span_ids, value.authority_kind
                ),
            )
        )
        plan = self.plan(
            "source_disagreement",
            persisted=_persisted(
                "source_disagreement", left=left, right=right, authority=authority
            ),
        )
        self.assertEqual(plan.outcome, "disputed")

    def test_null_mixed_and_approximate_time_never_become_current(self) -> None:
        scenarios = (
            (_request(valid_at=None), _claim("claim_b", object_json="Mumbai")),
            (
                _request(valid_at=datetime(2026, 6, 1, tzinfo=UTC)),
                _claim("claim_b", object_json="Mumbai"),
            ),
            (
                _request(),
                _claim(
                    "claim_b", object_json="Mumbai", precision="approximate",
                    valid_from=date(2026, 1, 1), valid_to=date(2026, 12, 31),
                ),
            ),
        )
        for request, replacement in scenarios:
            with self.subTest(valid_at=request.valid_at, precision=replacement.version.time_precision):
                plan = self.plan(
                    "explicit_correction",
                    request=request,
                    persisted=_persisted("explicit_correction", right=replacement),
                )
                self.assertEqual(plan.selected_current_claim_id, None)
                self.assertEqual(plan.actions[-1].target_status, "confirmed")

    def test_existing_current_status_is_preserved_without_valid_at(self) -> None:
        replacement = _claim("claim_b", object_json="Mumbai", status="current")

        plan = self.plan(
            "explicit_correction",
            request=_request(valid_at=None),
            persisted=_persisted("explicit_correction", right=replacement),
        )

        self.assertNotIn("claim_b", [value.claim_id for value in plan.actions])
        self.assertEqual(plan.selected_current_claim_id, "claim_b")

    def test_terminal_claims_are_never_reopened(self) -> None:
        left = _claim("claim_a", status="superseded")
        plan = self.plan(
            "explicit_correction",
            persisted=_persisted("explicit_correction", left=left),
        )
        self.assertEqual([value.claim_id for value in plan.actions], ["claim_b"])
        self.assertNotIn("claim_a", [value.claim_id for value in plan.actions])

    def test_terminal_replacement_is_never_selected_or_linked_as_active(self) -> None:
        replacement = _claim(
            "claim_b", object_json="Mumbai", status="superseded"
        )
        plan = self.plan(
            "explicit_correction",
            persisted=_persisted(
                "explicit_correction",
                right=replacement,
            ),
        )
        self.assertIsNone(plan.selected_current_claim_id)
        self.assertEqual(plan.relations, ())
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].claim_id, "claim_a")
        self.assertIsNone(plan.actions[0].replacement_claim_id)

    def test_cutoff_user_and_provenance_are_validated(self) -> None:
        persisted = _persisted("unrelated")
        with self.assertRaisesRegex(BeliefResolverError, "decision user"):
            resolve_belief(
                self.config,
                replace(_request(), user_id="user_002"),
                persisted,
            )
        closed = replace(
            persisted.left,
            version=replace(persisted.left.version, transaction_to=AS_OF),
        )
        with self.assertRaisesRegex(BeliefResolverError, "not visible"):
            self.plan(
                "unrelated",
                persisted=_persisted("unrelated", left=closed),
            )
        future = replace(
            persisted,
            source_ingested_at=tuple(
                (key, AS_OF.replace(day=2)) for key, _ in persisted.source_ingested_at
            ),
        )
        with self.assertRaisesRegex(BeliefResolverError, "future evidence"):
            self.plan("unrelated", persisted=future)

    def test_deterministic_snapshot_ids_and_idempotency_intent(self) -> None:
        persisted = _persisted("explicit_correction")
        first = self.plan("explicit_correction", persisted=persisted)
        second = self.plan("explicit_correction", persisted=persisted)
        changed = self.plan(
            "explicit_correction",
            request=_request(idempotency_key="resolve_2"),
            persisted=persisted,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first.resolution_id, changed.resolution_id)
        self.assertNotEqual(first.input_snapshot_sha256, changed.input_snapshot_sha256)
        self.assertEqual([value.action_order for value in first.actions], [1, 2])

    def test_confidence_is_not_derived_from_extraction_confidence(self) -> None:
        plan = self.plan(
            "unrelated",
            persisted=_persisted(
                "unrelated",
                left=_claim("claim_a", extraction_confidence=0.01),
                right=_claim("claim_b", object_json="Mumbai", extraction_confidence=0.99),
            ),
        )
        self.assertIsNone(plan.belief_confidence)

    def test_request_cannot_supply_label_authority_state_or_evidence(self) -> None:
        request_fields = {value.name for value in fields(ResolutionRequest)}
        self.assertTrue(
            request_fields.isdisjoint(
                {"label", "authority", "left", "right", "relations", "evidence"}
            )
        )

    def test_records_reject_naive_time_and_non_json_snapshot_values(self) -> None:
        with self.assertRaisesRegex(BeliefResolverError, "timezone-aware"):
            replace(_request(), resolved_at=datetime(2026, 6, 1))
        with self.assertRaisesRegex(BeliefResolverError, "strings"):
            resolver_module._canonical_json({1: "bad"})
        with self.assertRaisesRegex(BeliefResolverError, "finite"):
            resolver_module._canonical_json({"bad": float("nan")})
        with self.assertRaises(StorageValidationError):
            _claim("claim_a", object_json=float("nan"))

    def test_relation_and_evidence_drift_fail_closed(self) -> None:
        persisted = _persisted("explicit_correction")
        wrong_relation = replace(
            persisted,
            relations=(replace(persisted.relations[0], relation_type="refines"),),
        )
        with self.assertRaisesRegex(BeliefResolverError, "decision label"):
            self.plan("explicit_correction", persisted=wrong_relation)
        wrong_evidence = replace(
            persisted,
            decision_evidence=persisted.decision_evidence[:-1],
        )
        with self.assertRaisesRegex(BeliefResolverError, "visible evidence"):
            self.plan("explicit_correction", persisted=wrong_evidence)


if __name__ == "__main__":
    unittest.main()
