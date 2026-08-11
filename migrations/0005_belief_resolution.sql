CREATE TABLE belief_resolutions (
    resolution_id text PRIMARY KEY CHECK (btrim(resolution_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    resolver_version text NOT NULL CHECK (btrim(resolver_version) <> ''),
    policy_version text NOT NULL CHECK (btrim(policy_version) <> ''),
    decision_id text NOT NULL,
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    transaction_as_of timestamptz NOT NULL,
    valid_at_date date,
    valid_at_timestamp timestamptz,
    resolved_at timestamptz NOT NULL,
    outcome text NOT NULL CHECK (
        outcome IN (
            'no_change', 'excluded', 'temporal_change_resolved',
            'correction_resolved', 'refinement_resolved',
            'retraction_resolved', 'authority_resolved', 'disputed'
        )
    ),
    selected_current_claim_id text,
    authority_reason text NOT NULL CHECK (btrim(authority_reason) <> ''),
    belief_confidence double precision CHECK (belief_confidence IS NULL),
    UNIQUE (user_id, resolution_id),
    UNIQUE (user_id, idempotency_key),
    FOREIGN KEY (user_id, decision_id)
        REFERENCES conflict_decisions (user_id, decision_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, selected_current_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    CHECK (resolved_at >= transaction_as_of),
    CHECK (valid_at_date IS NULL OR valid_at_timestamp IS NULL),
    CHECK (selected_current_claim_id IS NULL
        OR valid_at_date IS NOT NULL OR valid_at_timestamp IS NOT NULL)
);

CREATE TABLE belief_resolution_actions (
    action_id text PRIMARY KEY CHECK (btrim(action_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    resolution_id text NOT NULL,
    action_order integer NOT NULL CHECK (action_order > 0),
    claim_id text NOT NULL,
    from_status text NOT NULL CHECK (
        from_status IN (
            'candidate', 'confirmed', 'current', 'historical',
            'disputed', 'superseded', 'excluded'
        )
    ),
    target_status text NOT NULL CHECK (
        target_status IN (
            'confirmed', 'current', 'historical', 'disputed',
            'superseded', 'excluded'
        )
    ),
    replacement_claim_id text,
    reason text NOT NULL CHECK (btrim(reason) <> ''),
    from_version_id text NOT NULL,
    to_version_id text NOT NULL,
    transition_id text NOT NULL,
    UNIQUE (user_id, action_id),
    UNIQUE (user_id, resolution_id, action_order),
    UNIQUE (user_id, resolution_id, claim_id),
    FOREIGN KEY (user_id, resolution_id)
        REFERENCES belief_resolutions (user_id, resolution_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, replacement_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, from_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, to_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, transition_id)
        REFERENCES lifecycle_transitions (user_id, transition_id) ON DELETE RESTRICT,
    CHECK (from_status <> target_status),
    CHECK (from_version_id <> to_version_id),
    CHECK (replacement_claim_id IS NULL OR replacement_claim_id <> claim_id)
);

ALTER TABLE conflict_decision_evidence
    ADD CONSTRAINT conflict_decision_evidence_owned_lineage UNIQUE (
        user_id, decision_id, decision_evidence_id
    );

CREATE TABLE belief_resolution_evidence (
    user_id text NOT NULL,
    resolution_id text NOT NULL,
    decision_id text NOT NULL,
    decision_evidence_id text NOT NULL,
    PRIMARY KEY (user_id, resolution_id, decision_evidence_id),
    FOREIGN KEY (user_id, resolution_id)
        REFERENCES belief_resolutions (user_id, resolution_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, decision_id, decision_evidence_id)
        REFERENCES conflict_decision_evidence (
            user_id, decision_id, decision_evidence_id
        ) ON DELETE RESTRICT
);

ALTER TABLE claim_relations
    ADD COLUMN resolver_version text,
    ADD COLUMN resolution_id text,
    ADD CONSTRAINT claim_relations_resolver_provenance_shape CHECK (
        (resolver_version IS NULL AND resolution_id IS NULL)
        OR
        (btrim(resolver_version) <> '' AND resolution_id IS NOT NULL)
    ),
    ADD CONSTRAINT claim_relations_owned_resolution FOREIGN KEY (
        user_id, resolution_id
    ) REFERENCES belief_resolutions (user_id, resolution_id) ON DELETE RESTRICT;

ALTER TABLE processing_outbox
    DROP CONSTRAINT processing_outbox_event_type_check,
    ADD CONSTRAINT processing_outbox_event_type_check CHECK (
        event_type IN (
            'source_ingested', 'claims_changed', 'claim_recompute_required',
            'source_deleted', 'claim_lifecycle_changed',
            'conflict_recompute_required', 'belief_resolved'
        )
    );

CREATE INDEX belief_resolutions_user_decision_idx
    ON belief_resolutions (user_id, decision_id, resolved_at, resolution_id);
CREATE INDEX belief_resolution_actions_user_claim_idx
    ON belief_resolution_actions (user_id, claim_id, action_order, resolution_id);
CREATE INDEX belief_resolution_evidence_decision_idx
    ON belief_resolution_evidence (user_id, decision_id, decision_evidence_id);
