CREATE TABLE conflict_decisions (
    decision_id text PRIMARY KEY CHECK (btrim(decision_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    classifier_version text NOT NULL CHECK (btrim(classifier_version) <> ''),
    rule_version text NOT NULL CHECK (btrim(rule_version) <> ''),
    pair_id text NOT NULL CHECK (btrim(pair_id) <> ''),
    left_claim_id text NOT NULL,
    right_claim_id text NOT NULL,
    left_version_id text NOT NULL,
    right_version_id text NOT NULL,
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    transaction_as_of timestamptz NOT NULL,
    matched_rule text NOT NULL CHECK (
        matched_rule IN (
            'hard_contradiction', 'temporal_change', 'explicit_correction',
            'refinement', 'source_disagreement', 'retraction',
            'unresolved_ambiguity', 'unrelated'
        )
    ),
    label text NOT NULL CHECK (
        label IN (
            'hard_contradiction', 'temporal_change', 'explicit_correction',
            'refinement', 'source_disagreement', 'retraction',
            'unresolved_ambiguity', 'unrelated'
        )
    ),
    classified_at timestamptz NOT NULL,
    UNIQUE (user_id, decision_id),
    UNIQUE (
        user_id, pair_id, classifier_version, left_version_id,
        right_version_id, input_snapshot_sha256
    ),
    FOREIGN KEY (user_id, left_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, right_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, left_claim_id, left_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, right_claim_id, right_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    CHECK (left_claim_id < right_claim_id),
    CHECK (matched_rule = label)
);

CREATE TABLE claim_relations (
    relation_id text PRIMARY KEY CHECK (btrim(relation_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    decision_id text NOT NULL,
    classifier_version text NOT NULL CHECK (btrim(classifier_version) <> ''),
    source_claim_id text NOT NULL,
    target_claim_id text NOT NULL,
    relation_type text NOT NULL CHECK (
        relation_type IN (
            'supports', 'contradicts', 'corrects', 'supersedes', 'refines',
            'same_event_as', 'caused_by', 'hindered_by', 'same_topic_as'
        )
    ),
    confidence double precision NOT NULL CHECK (confidence = 1),
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    created_at timestamptz NOT NULL,
    UNIQUE (user_id, relation_id),
    FOREIGN KEY (user_id, decision_id)
        REFERENCES conflict_decisions (user_id, decision_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, source_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, target_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    CHECK (source_claim_id <> target_claim_id),
    CHECK (
        relation_type NOT IN ('contradicts', 'same_event_as', 'same_topic_as')
        OR source_claim_id < target_claim_id
    )
);

CREATE TABLE conflict_decision_evidence (
    decision_evidence_id text PRIMARY KEY CHECK (
        btrim(decision_evidence_id) <> ''
    ),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    decision_id text NOT NULL,
    claim_id text NOT NULL,
    span_id text NOT NULL,
    support_type text NOT NULL CHECK (
        support_type IN ('supports', 'contradicts', 'corrects')
    ),
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    UNIQUE (user_id, decision_evidence_id),
    UNIQUE (user_id, decision_id, claim_id, span_id, support_type),
    FOREIGN KEY (user_id, decision_id)
        REFERENCES conflict_decisions (user_id, decision_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, span_id)
        REFERENCES source_spans (user_id, span_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, span_id, support_type)
        REFERENCES evidence_links (user_id, claim_id, span_id, support_type)
        ON DELETE RESTRICT
);

ALTER TABLE processing_outbox
    DROP CONSTRAINT processing_outbox_event_type_check,
    ADD CONSTRAINT processing_outbox_event_type_check CHECK (
        event_type IN (
            'source_ingested', 'claims_changed', 'claim_recompute_required',
            'source_deleted', 'claim_lifecycle_changed',
            'conflict_recompute_required'
        )
    );

CREATE INDEX conflict_decisions_user_pair_as_of_idx
    ON conflict_decisions (user_id, pair_id, transaction_as_of, decision_id);
CREATE INDEX claim_relations_user_source_target_idx
    ON claim_relations (user_id, source_claim_id, target_claim_id, relation_type);
CREATE INDEX conflict_decision_evidence_span_idx
    ON conflict_decision_evidence (user_id, span_id, decision_id);
