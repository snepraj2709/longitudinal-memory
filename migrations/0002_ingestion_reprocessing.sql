ALTER TABLE processing_attempts
    ADD COLUMN lease_owner text,
    ADD COLUMN lease_expires_at timestamptz,
    ADD COLUMN retryable boolean;

UPDATE processing_attempts
SET retryable = state IN ('pending', 'running', 'failed');

ALTER TABLE processing_attempts
    ALTER COLUMN retryable SET NOT NULL,
    ADD CONSTRAINT processing_attempts_lease_owner_nonempty CHECK (
        lease_owner IS NULL OR btrim(lease_owner) <> ''
    ),
    ADD CONSTRAINT processing_attempts_state_shape CHECK (
        (state = 'pending'
            AND completed_at IS NULL
            AND sanitized_error_code IS NULL
            AND sanitized_error_metadata IS NULL
            AND lease_owner IS NULL
            AND lease_expires_at IS NULL
            AND retryable)
        OR
        (state = 'running'
            AND completed_at IS NULL
            AND sanitized_error_code IS NULL
            AND sanitized_error_metadata IS NULL
            AND lease_owner IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND retryable)
        OR
        (state = 'succeeded'
            AND completed_at IS NOT NULL
            AND sanitized_error_code IS NULL
            AND sanitized_error_metadata IS NULL
            AND lease_owner IS NULL
            AND lease_expires_at IS NULL
            AND NOT retryable)
        OR
        (state = 'failed'
            AND completed_at IS NOT NULL
            AND sanitized_error_code IS NOT NULL
            AND lease_owner IS NULL
            AND lease_expires_at IS NULL)
    ),
    ADD CONSTRAINT processing_attempts_owned_identity UNIQUE (
        user_id, source_id, extraction_version_id, attempt_id
    );

CREATE TABLE claim_extractions (
    user_id text NOT NULL,
    claim_id text NOT NULL,
    source_id text NOT NULL,
    extraction_version_id text NOT NULL,
    attempt_id text NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, claim_id, source_id, extraction_version_id),
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE RESTRICT,
    FOREIGN KEY (
        user_id, source_id, extraction_version_id, attempt_id
    ) REFERENCES processing_attempts (
        user_id, source_id, extraction_version_id, attempt_id
    ) ON DELETE RESTRICT
);

CREATE TABLE processing_outbox (
    event_id text PRIMARY KEY CHECK (btrim(event_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    event_type text NOT NULL CHECK (
        event_type IN (
            'source_ingested', 'claims_changed',
            'claim_recompute_required', 'source_deleted'
        )
    ),
    aggregate_id text NOT NULL CHECK (btrim(aggregate_id) <> ''),
    dedupe_key text NOT NULL CHECK (btrim(dedupe_key) <> ''),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    state text NOT NULL CHECK (state IN ('pending', 'published')),
    created_at timestamptz NOT NULL,
    published_at timestamptz,
    UNIQUE (user_id, event_id),
    UNIQUE (user_id, dedupe_key),
    CHECK (
        (state = 'pending' AND published_at IS NULL)
        OR (state = 'published' AND published_at IS NOT NULL)
    ),
    CHECK (published_at IS NULL OR published_at >= created_at)
);

CREATE TABLE source_tombstones (
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    source_id text NOT NULL CHECK (btrim(source_id) <> ''),
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    deleted_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, source_id),
    UNIQUE (user_id, idempotency_key)
);

CREATE INDEX processing_attempts_pending_idx
    ON processing_attempts (started_at, user_id, source_id, extraction_version_id)
    WHERE state = 'pending';
CREATE INDEX processing_attempts_expired_lease_idx
    ON processing_attempts (lease_expires_at, user_id, source_id)
    WHERE state = 'running';
CREATE INDEX source_spans_source_evidence_idx
    ON source_spans (user_id, source_id, span_id, speaker_id);
CREATE INDEX evidence_links_span_provenance_idx
    ON evidence_links (user_id, span_id, claim_id);
CREATE INDEX claim_extractions_source_idx
    ON claim_extractions (user_id, source_id, claim_id, extraction_version_id);
CREATE INDEX claim_extractions_claim_idx
    ON claim_extractions (user_id, claim_id, source_id);
CREATE INDEX processing_outbox_pending_idx
    ON processing_outbox (created_at, event_id)
    WHERE state = 'pending';
