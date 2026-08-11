ALTER TABLE claim_relations
    ADD CONSTRAINT claim_relations_retrieval_lineage UNIQUE (
        user_id, relation_id, source_claim_id, target_claim_id, relation_type
    );

CREATE TABLE retrieval_index_runs (
    run_id text PRIMARY KEY CHECK (run_id ~ '^[0-9a-f]{64}$'),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE CASCADE,
    index_version text NOT NULL CHECK (btrim(index_version) <> ''),
    content_renderer_version text NOT NULL CHECK (
        btrim(content_renderer_version) <> ''
    ),
    embedding_version text NOT NULL CHECK (btrim(embedding_version) <> ''),
    embedding_dimension integer NOT NULL CHECK (embedding_dimension = 256),
    config_sha256 text NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    transaction_as_of timestamptz NOT NULL,
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    records_snapshot_sha256 text NOT NULL CHECK (
        records_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    atomic_count integer NOT NULL CHECK (atomic_count >= 0),
    session_count integer NOT NULL CHECK (session_count >= 0),
    record_count integer NOT NULL CHECK (record_count >= 0),
    sanitized_failure_code text,
    started_at timestamptz NOT NULL,
    completed_at timestamptz NOT NULL,
    UNIQUE (user_id, run_id),
    UNIQUE (user_id, index_version, idempotency_key),
    CHECK (completed_at >= started_at),
    CHECK (record_count = atomic_count + session_count),
    CHECK (
        (status = 'succeeded'
            AND sanitized_failure_code IS NULL)
        OR
        (status = 'failed'
            AND sanitized_failure_code ~ '^[a-z0-9_]{1,64}$'
            AND record_count = 0)
    )
);

CREATE TABLE retrieval_index_records (
    index_record_id text PRIMARY KEY CHECK (
        index_record_id ~ '^[0-9a-f]{64}$'
    ),
    user_id text NOT NULL,
    run_id text NOT NULL,
    index_version text NOT NULL CHECK (btrim(index_version) <> ''),
    record_kind text NOT NULL CHECK (record_kind IN ('atomic', 'session')),
    atomic_claim_id text,
    atomic_claim_version_id text,
    session_summary_id text,
    subject_id text,
    speaker_id text,
    predicate text,
    content_text text NOT NULL CHECK (btrim(content_text) <> ''),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    embedding_version text NOT NULL CHECK (btrim(embedding_version) <> ''),
    embedding_dimension integer NOT NULL CHECK (embedding_dimension = 256),
    embedding vector(256) NOT NULL CHECK (vector_dims(embedding) = 256),
    search_document tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, content_text)
    ) STORED,
    lifecycle_statuses jsonb NOT NULL CHECK (
        jsonb_typeof(lifecycle_statuses) = 'array'
        AND jsonb_array_length(lifecycle_statuses) > 0
        AND lifecycle_statuses <@ '[
            "candidate", "confirmed", "current", "historical",
            "disputed", "superseded"
        ]'::jsonb
    ),
    memory_kind text CHECK (memory_kind IN ('episodic', 'durative')),
    epistemic_status text CHECK (
        epistemic_status IN (
            'asserted', 'inferred', 'reported_by_other', 'hypothetical',
            'uncertain', 'denied', 'corrected'
        )
    ),
    time_precision text NOT NULL CHECK (
        time_precision IN (
            'timestamp', 'day', 'month', 'year', 'approximate',
            'unknown', 'mixed'
        )
    ),
    valid_from_date date,
    valid_from_timestamp timestamptz,
    valid_to_date date,
    valid_to_timestamp timestamptz,
    transaction_from timestamptz NOT NULL,
    transaction_to timestamptz,
    sensitivity text CHECK (
        sensitivity IN ('standard', 'sensitive')
    ),
    contains_sensitive boolean NOT NULL,
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    UNIQUE (user_id, index_record_id),
    UNIQUE (user_id, index_record_id, record_kind),
    FOREIGN KEY (user_id, run_id)
        REFERENCES retrieval_index_runs (user_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, atomic_claim_id, atomic_claim_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, session_summary_id)
        REFERENCES session_summaries (user_id, summary_id) ON DELETE CASCADE,
    CHECK (
        (record_kind = 'atomic'
            AND atomic_claim_id IS NOT NULL
            AND atomic_claim_version_id IS NOT NULL
            AND session_summary_id IS NULL
            AND subject_id IS NOT NULL AND btrim(subject_id) <> ''
            AND speaker_id IS NOT NULL AND btrim(speaker_id) <> ''
            AND predicate IS NOT NULL AND btrim(predicate) <> ''
            AND epistemic_status IS NOT NULL)
        OR
        (record_kind = 'session'
            AND atomic_claim_id IS NULL
            AND atomic_claim_version_id IS NULL
            AND session_summary_id IS NOT NULL
            AND subject_id IS NULL
            AND speaker_id IS NULL
            AND predicate IS NULL
            AND memory_kind IS NULL
            AND epistemic_status IS NULL)
    ),
    CHECK (transaction_to IS NULL OR transaction_to > transaction_from),
    CHECK (
        (time_precision = 'unknown'
            AND valid_from_date IS NULL AND valid_to_date IS NULL
            AND valid_from_timestamp IS NULL AND valid_to_timestamp IS NULL)
        OR
        (time_precision = 'mixed'
            AND record_kind = 'session'
            AND valid_from_date IS NULL AND valid_to_date IS NULL
            AND valid_from_timestamp IS NULL AND valid_to_timestamp IS NULL)
        OR
        (time_precision = 'timestamp'
            AND (valid_from_timestamp IS NOT NULL
                 OR valid_to_timestamp IS NOT NULL)
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
    ),
    CHECK (
        record_kind <> 'atomic'
        OR time_precision <> 'unknown'
        OR NOT lifecycle_statuses @> '["current"]'::jsonb
    ),
    CHECK (
        contains_sensitive = COALESCE(sensitivity = 'sensitive', false)
    )
);

CREATE TABLE retrieval_index_claim_links (
    user_id text NOT NULL,
    index_record_id text NOT NULL,
    claim_id text NOT NULL,
    claim_version_id text NOT NULL,
    lifecycle_status text NOT NULL CHECK (
        lifecycle_status IN (
            'candidate', 'confirmed', 'current', 'historical',
            'disputed', 'superseded'
        )
    ),
    claim_order integer NOT NULL CHECK (claim_order >= 0),
    PRIMARY KEY (user_id, index_record_id, claim_id, claim_version_id),
    UNIQUE (user_id, index_record_id, claim_order),
    FOREIGN KEY (user_id, index_record_id)
        REFERENCES retrieval_index_records (user_id, index_record_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, claim_id, claim_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id)
        ON DELETE CASCADE
);

CREATE TABLE retrieval_index_source_links (
    user_id text NOT NULL,
    index_record_id text NOT NULL,
    claim_id text NOT NULL,
    claim_version_id text NOT NULL,
    source_id text NOT NULL,
    span_id text NOT NULL,
    support_type text NOT NULL CHECK (
        support_type IN ('supports', 'contradicts', 'corrects')
    ),
    source_order integer NOT NULL CHECK (source_order >= 0),
    PRIMARY KEY (
        user_id, index_record_id, claim_id, claim_version_id,
        source_id, span_id, support_type
    ),
    UNIQUE (user_id, index_record_id, source_order),
    FOREIGN KEY (user_id, index_record_id)
        REFERENCES retrieval_index_records (user_id, index_record_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, claim_id, claim_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id, span_id)
        REFERENCES source_spans (user_id, source_id, span_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, claim_id, span_id, support_type)
        REFERENCES evidence_links (user_id, claim_id, span_id, support_type)
        ON DELETE CASCADE
);

CREATE TABLE retrieval_index_relation_links (
    user_id text NOT NULL,
    index_record_id text NOT NULL,
    record_kind text NOT NULL DEFAULT 'atomic' CHECK (record_kind = 'atomic'),
    relation_id text NOT NULL,
    source_claim_id text NOT NULL,
    target_claim_id text NOT NULL,
    relation_type text NOT NULL CHECK (
        relation_type IN (
            'supports', 'contradicts', 'corrects', 'supersedes', 'refines',
            'same_event_as', 'caused_by', 'hindered_by', 'same_topic_as'
        )
    ),
    direction text NOT NULL CHECK (
        direction IN ('incoming', 'outgoing', 'symmetric')
    ),
    relation_order integer NOT NULL CHECK (relation_order >= 0),
    PRIMARY KEY (user_id, index_record_id, relation_id),
    UNIQUE (user_id, index_record_id, relation_order),
    FOREIGN KEY (user_id, index_record_id, record_kind)
        REFERENCES retrieval_index_records (
            user_id, index_record_id, record_kind
        )
        ON DELETE CASCADE,
    FOREIGN KEY (
        user_id, relation_id, source_claim_id, target_claim_id, relation_type
    ) REFERENCES claim_relations (
        user_id, relation_id, source_claim_id, target_claim_id, relation_type
    ) ON DELETE CASCADE
);

CREATE INDEX retrieval_runs_user_version_status_idx
    ON retrieval_index_runs (
        user_id, index_version, status, transaction_as_of, run_id
    );
CREATE INDEX retrieval_records_user_atomic_filter_idx
    ON retrieval_index_records (
        user_id, index_version, record_kind, lifecycle_statuses,
        transaction_from, subject_id, speaker_id, predicate
    ) WHERE record_kind = 'atomic';
CREATE INDEX retrieval_records_user_atomic_date_idx
    ON retrieval_index_records (
        user_id, index_version, record_kind, valid_from_date, valid_to_date
    ) WHERE record_kind = 'atomic';
CREATE INDEX retrieval_records_user_atomic_timestamp_idx
    ON retrieval_index_records (
        user_id, index_version, record_kind,
        valid_from_timestamp, valid_to_timestamp
    ) WHERE record_kind = 'atomic';
CREATE INDEX retrieval_records_user_session_filter_idx
    ON retrieval_index_records (
        user_id, index_version, record_kind, lifecycle_statuses,
        transaction_from, session_summary_id
    ) WHERE record_kind = 'session';
CREATE INDEX retrieval_claim_links_user_claim_idx
    ON retrieval_index_claim_links (
        user_id, claim_id, claim_version_id, index_record_id
    );
CREATE INDEX retrieval_source_links_user_source_idx
    ON retrieval_index_source_links (
        user_id, source_id, span_id, index_record_id
    );
CREATE INDEX retrieval_relation_links_user_relation_idx
    ON retrieval_index_relation_links (
        user_id, relation_id, index_record_id
    );
CREATE INDEX retrieval_records_atomic_fts_gin
    ON retrieval_index_records USING gin (search_document)
    WHERE record_kind = 'atomic';
CREATE INDEX retrieval_records_session_fts_gin
    ON retrieval_index_records USING gin (search_document)
    WHERE record_kind = 'session';
CREATE INDEX retrieval_records_atomic_vector_hnsw
    ON retrieval_index_records USING hnsw (embedding vector_cosine_ops)
    WHERE record_kind = 'atomic';
CREATE INDEX retrieval_records_session_vector_hnsw
    ON retrieval_index_records USING hnsw (embedding vector_cosine_ops)
    WHERE record_kind = 'session';

CREATE FUNCTION purge_retrieval_records_for_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM retrieval_index_records AS record
    USING retrieval_index_source_links AS link
    WHERE link.user_id = OLD.user_id
      AND link.source_id = OLD.source_id
      AND record.user_id = link.user_id
      AND record.index_record_id = link.index_record_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER source_events_purge_retrieval_records
    BEFORE DELETE ON source_events
    FOR EACH ROW EXECUTE FUNCTION purge_retrieval_records_for_source();

CREATE FUNCTION purge_retrieval_records_for_span()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM retrieval_index_records AS record
    USING retrieval_index_source_links AS link
    WHERE link.user_id = OLD.user_id
      AND link.span_id = OLD.span_id
      AND record.user_id = link.user_id
      AND record.index_record_id = link.index_record_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER source_spans_purge_retrieval_records
    BEFORE DELETE ON source_spans
    FOR EACH ROW EXECUTE FUNCTION purge_retrieval_records_for_span();

CREATE FUNCTION purge_retrieval_records_for_evidence()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM retrieval_index_records AS record
    USING retrieval_index_source_links AS link
    WHERE link.user_id = OLD.user_id
      AND link.claim_id = OLD.claim_id
      AND link.span_id = OLD.span_id
      AND link.support_type = OLD.support_type
      AND record.user_id = link.user_id
      AND record.index_record_id = link.index_record_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER evidence_links_purge_retrieval_records
    BEFORE DELETE ON evidence_links
    FOR EACH ROW EXECUTE FUNCTION purge_retrieval_records_for_evidence();

CREATE FUNCTION purge_retrieval_records_for_relation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM retrieval_index_records AS record
    USING retrieval_index_relation_links AS link
    WHERE link.user_id = OLD.user_id
      AND link.relation_id = OLD.relation_id
      AND record.user_id = link.user_id
      AND record.index_record_id = link.index_record_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER claim_relations_purge_retrieval_records
    BEFORE DELETE ON claim_relations
    FOR EACH ROW EXECUTE FUNCTION purge_retrieval_records_for_relation();
