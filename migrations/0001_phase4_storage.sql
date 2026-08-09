CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE memory_users (
    user_id text PRIMARY KEY CHECK (btrim(user_id) <> ''),
    created_at timestamptz NOT NULL
);

CREATE TABLE source_events (
    source_id text PRIMARY KEY CHECK (btrim(source_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    source_type text NOT NULL CHECK (
        source_type IN ('conversation', 'email', 'calendar', 'chat')
    ),
    session_id text CHECK (session_id IS NULL OR btrim(session_id) <> ''),
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    produced_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL,
    raw_content text NOT NULL CHECK (btrim(raw_content) <> ''),
    participants jsonb NOT NULL CHECK (
        jsonb_typeof(participants) IS NOT NULL
        AND jsonb_typeof(participants) = 'array'
    ),
    metadata jsonb NOT NULL CHECK (
        jsonb_typeof(metadata) IS NOT NULL
        AND jsonb_typeof(metadata) = 'object'
    ),
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    UNIQUE (user_id, source_id),
    UNIQUE (user_id, idempotency_key)
);

CREATE TABLE source_spans (
    span_id text PRIMARY KEY CHECK (btrim(span_id) <> ''),
    user_id text NOT NULL,
    source_id text NOT NULL,
    message_id text CHECK (message_id IS NULL OR btrim(message_id) <> ''),
    speaker_id text NOT NULL CHECK (btrim(speaker_id) <> ''),
    verbatim_quote text NOT NULL CHECK (verbatim_quote <> ''),
    start_offset integer,
    end_offset integer,
    UNIQUE (user_id, span_id),
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE RESTRICT,
    CHECK (
        (start_offset IS NULL AND end_offset IS NULL)
        OR (
            start_offset IS NOT NULL
            AND end_offset IS NOT NULL
            AND start_offset >= 0
            AND start_offset < end_offset
        )
    )
);

CREATE TABLE extraction_versions (
    version_id text PRIMARY KEY CHECK (btrim(version_id) <> ''),
    model_version text NOT NULL CHECK (btrim(model_version) <> ''),
    prompt_version text NOT NULL CHECK (btrim(prompt_version) <> ''),
    prompt_hash text NOT NULL CHECK (prompt_hash ~ '^[0-9a-f]{64}$'),
    schema_version text NOT NULL CHECK (btrim(schema_version) <> ''),
    schema_hash text NOT NULL CHECK (schema_hash ~ '^[0-9a-f]{64}$'),
    registry_version text NOT NULL CHECK (btrim(registry_version) <> ''),
    registry_hash text NOT NULL CHECK (registry_hash ~ '^[0-9a-f]{64}$'),
    input_manifest_hash text NOT NULL CHECK (
        input_manifest_hash ~ '^[0-9a-f]{64}$'
    ),
    created_at timestamptz NOT NULL
);

CREATE TABLE processing_attempts (
    attempt_id text PRIMARY KEY CHECK (btrim(attempt_id) <> ''),
    user_id text NOT NULL,
    source_id text NOT NULL,
    extraction_version_id text NOT NULL REFERENCES extraction_versions (version_id)
        ON DELETE RESTRICT,
    attempt_number integer NOT NULL CHECK (attempt_number > 0),
    state text NOT NULL CHECK (
        state IN ('pending', 'running', 'succeeded', 'failed')
    ),
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    sanitized_error_code text CHECK (
        sanitized_error_code IS NULL OR btrim(sanitized_error_code) <> ''
    ),
    sanitized_error_metadata jsonb CHECK (
        sanitized_error_metadata IS NULL
        OR (
            jsonb_typeof(sanitized_error_metadata) IS NOT NULL
            AND jsonb_typeof(sanitized_error_metadata) = 'object'
        )
    ),
    UNIQUE (user_id, attempt_id),
    UNIQUE (user_id, source_id, extraction_version_id, attempt_number),
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE RESTRICT,
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE claims (
    claim_id text PRIMARY KEY CHECK (btrim(claim_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    subject_id text NOT NULL CHECK (btrim(subject_id) <> ''),
    speaker_id text NOT NULL CHECK (btrim(speaker_id) <> ''),
    predicate text NOT NULL CHECK (btrim(predicate) <> ''),
    predicate_registry_version text NOT NULL CHECK (
        btrim(predicate_registry_version) <> ''
    ),
    object_json jsonb NOT NULL CHECK (
        jsonb_typeof(object_json) IN ('array', 'object', 'string', 'number', 'boolean')
    ),
    polarity text NOT NULL CHECK (polarity IN ('positive', 'negative')),
    epistemic_status text NOT NULL CHECK (
        epistemic_status IN (
            'asserted', 'inferred', 'reported_by_other', 'hypothetical',
            'uncertain', 'denied', 'corrected'
        )
    ),
    valid_from_date date,
    valid_from_timestamp timestamptz,
    valid_to_date date,
    valid_to_timestamp timestamptz,
    time_precision text NOT NULL CHECK (
        time_precision IN ('timestamp', 'day', 'month', 'year', 'approximate', 'unknown')
    ),
    extraction_confidence double precision NOT NULL CHECK (
        extraction_confidence >= 0 AND extraction_confidence <= 1
    ),
    memory_kind text CHECK (memory_kind IN ('episodic', 'durative')),
    sensitivity text CHECK (
        sensitivity IN ('standard', 'sensitive', 'restricted')
    ),
    extraction_version_id text NOT NULL REFERENCES extraction_versions (version_id)
        ON DELETE RESTRICT,
    UNIQUE (user_id, claim_id),
    CHECK (NOT (valid_from_date IS NOT NULL AND valid_from_timestamp IS NOT NULL)),
    CHECK (NOT (valid_to_date IS NOT NULL AND valid_to_timestamp IS NOT NULL)),
    CHECK (
        NOT (
            (valid_from_date IS NOT NULL OR valid_to_date IS NOT NULL)
            AND (valid_from_timestamp IS NOT NULL OR valid_to_timestamp IS NOT NULL)
        )
    ),
    CHECK (
        (time_precision = 'unknown'
            AND valid_from_date IS NULL AND valid_from_timestamp IS NULL
            AND valid_to_date IS NULL AND valid_to_timestamp IS NULL)
        OR
        (time_precision = 'timestamp'
            AND (valid_from_timestamp IS NOT NULL OR valid_to_timestamp IS NOT NULL)
            AND valid_from_date IS NULL AND valid_to_date IS NULL)
        OR
        (time_precision IN ('day', 'month', 'year', 'approximate')
            AND (valid_from_date IS NOT NULL OR valid_to_date IS NOT NULL)
            AND valid_from_timestamp IS NULL AND valid_to_timestamp IS NULL)
    ),
    CHECK (
        valid_from_date IS NULL OR valid_to_date IS NULL
        OR valid_from_date <= valid_to_date
    ),
    CHECK (
        valid_from_timestamp IS NULL OR valid_to_timestamp IS NULL
        OR valid_from_timestamp <= valid_to_timestamp
    )
);

CREATE TABLE claim_versions (
    version_id text PRIMARY KEY CHECK (btrim(version_id) <> ''),
    user_id text NOT NULL,
    claim_id text NOT NULL,
    lifecycle_status text NOT NULL CHECK (
        lifecycle_status IN (
            'candidate', 'confirmed', 'current', 'historical',
            'disputed', 'superseded', 'excluded'
        )
    ),
    transaction_from timestamptz NOT NULL,
    transaction_to timestamptz,
    belief_confidence double precision CHECK (
        belief_confidence IS NULL
        OR (belief_confidence >= 0 AND belief_confidence <= 1)
    ),
    UNIQUE (user_id, version_id),
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    CHECK (transaction_to IS NULL OR transaction_to > transaction_from)
);

CREATE UNIQUE INDEX claim_versions_one_open_per_claim
    ON claim_versions (user_id, claim_id)
    WHERE transaction_to IS NULL;

CREATE TABLE evidence_links (
    user_id text NOT NULL,
    claim_id text NOT NULL,
    span_id text NOT NULL,
    support_type text NOT NULL CHECK (
        support_type IN ('supports', 'contradicts', 'corrects')
    ),
    extraction_confidence double precision NOT NULL CHECK (
        extraction_confidence >= 0 AND extraction_confidence <= 1
    ),
    PRIMARY KEY (user_id, claim_id, span_id, support_type),
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, span_id)
        REFERENCES source_spans (user_id, span_id) ON DELETE RESTRICT
);

CREATE INDEX source_events_user_produced_idx
    ON source_events (user_id, produced_at, source_id);
CREATE INDEX source_spans_user_source_idx
    ON source_spans (user_id, source_id, span_id);
CREATE INDEX claims_user_predicate_idx
    ON claims (user_id, predicate, claim_id);
CREATE INDEX claim_versions_user_claim_transaction_idx
    ON claim_versions (user_id, claim_id, transaction_from, transaction_to);
