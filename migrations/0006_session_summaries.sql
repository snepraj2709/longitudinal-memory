ALTER TABLE source_spans
    ADD CONSTRAINT source_spans_owned_source_span UNIQUE (
        user_id, source_id, span_id
    );

CREATE TABLE session_summaries (
    summary_id text PRIMARY KEY CHECK (summary_id ~ '^[0-9a-f]{64}$'),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    session_definition_id text NOT NULL CHECK (
        session_definition_id ~ '^[0-9a-f]{64}$'
    ),
    session_membership_sha256 text NOT NULL CHECK (
        session_membership_sha256 ~ '^[0-9a-f]{64}$'
    ),
    renderer_version text NOT NULL CHECK (btrim(renderer_version) <> ''),
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    summary_text text NOT NULL CHECK (btrim(summary_text) <> ''),
    valid_time_kind text NOT NULL CHECK (
        valid_time_kind IN ('date', 'timestamp', 'unknown', 'mixed')
    ),
    valid_time_start_date date,
    valid_time_end_date date,
    valid_time_start_timestamp timestamptz,
    valid_time_end_timestamp timestamptz,
    contains_sensitive boolean NOT NULL,
    transaction_from timestamptz NOT NULL,
    transaction_to timestamptz,
    UNIQUE (user_id, summary_id),
    UNIQUE (user_id, idempotency_key),
    UNIQUE (
        user_id, session_definition_id, renderer_version,
        input_snapshot_sha256
    ),
    CHECK (transaction_to IS NULL OR transaction_to > transaction_from),
    CHECK (
        (valid_time_kind = 'date'
            AND (valid_time_start_date IS NOT NULL
                 OR valid_time_end_date IS NOT NULL)
            AND valid_time_start_timestamp IS NULL
            AND valid_time_end_timestamp IS NULL)
        OR
        (valid_time_kind = 'timestamp'
            AND (valid_time_start_timestamp IS NOT NULL
                 OR valid_time_end_timestamp IS NOT NULL)
            AND valid_time_start_date IS NULL
            AND valid_time_end_date IS NULL)
        OR
        (valid_time_kind IN ('unknown', 'mixed')
            AND valid_time_start_date IS NULL
            AND valid_time_end_date IS NULL
            AND valid_time_start_timestamp IS NULL
            AND valid_time_end_timestamp IS NULL)
    ),
    CHECK (
        valid_time_start_date IS NULL OR valid_time_end_date IS NULL
        OR valid_time_start_date <= valid_time_end_date
    ),
    CHECK (
        valid_time_start_timestamp IS NULL
        OR valid_time_end_timestamp IS NULL
        OR valid_time_start_timestamp <= valid_time_end_timestamp
    )
);

CREATE UNIQUE INDEX session_summaries_one_open_per_renderer
    ON session_summaries (user_id, session_definition_id, renderer_version)
    WHERE transaction_to IS NULL;

ALTER TABLE session_summaries
    ADD CONSTRAINT session_summaries_transaction_nonoverlap EXCLUDE USING gist (
        user_id WITH =,
        session_definition_id WITH =,
        renderer_version WITH =,
        tstzrange(transaction_from, transaction_to, '[)') WITH &&
    );

CREATE TABLE session_summary_sources (
    user_id text NOT NULL,
    summary_id text NOT NULL,
    source_id text NOT NULL,
    source_order integer NOT NULL CHECK (source_order >= 0),
    PRIMARY KEY (user_id, summary_id, source_id),
    UNIQUE (user_id, summary_id, source_order),
    FOREIGN KEY (user_id, summary_id)
        REFERENCES session_summaries (user_id, summary_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE RESTRICT
);

CREATE TABLE session_summary_statements (
    statement_id text PRIMARY KEY CHECK (statement_id ~ '^[0-9a-f]{64}$'),
    user_id text NOT NULL,
    summary_id text NOT NULL,
    statement_kind text NOT NULL CHECK (
        statement_kind IN ('observed_fact', 'unresolved_question')
    ),
    lifecycle_view text NOT NULL CHECK (
        lifecycle_view IN ('accepted', 'candidate', 'historical', 'disputed')
    ),
    statement_text text NOT NULL CHECK (btrim(statement_text) <> ''),
    claim_ids jsonb NOT NULL CHECK (
        jsonb_typeof(claim_ids) = 'array'
        AND jsonb_array_length(claim_ids) > 0
    ),
    statement_order integer NOT NULL CHECK (statement_order >= 0),
    contains_sensitive boolean NOT NULL,
    UNIQUE (user_id, statement_id),
    UNIQUE (user_id, summary_id, statement_id),
    UNIQUE (user_id, summary_id, statement_order),
    FOREIGN KEY (user_id, summary_id)
        REFERENCES session_summaries (user_id, summary_id) ON DELETE CASCADE
);

CREATE TABLE session_summary_statement_evidence (
    user_id text NOT NULL,
    summary_id text NOT NULL,
    statement_id text NOT NULL,
    evidence_id text NOT NULL CHECK (evidence_id ~ '^[0-9a-f]{64}$'),
    claim_id text NOT NULL,
    claim_version_id text NOT NULL,
    source_id text NOT NULL,
    span_id text NOT NULL,
    support_type text NOT NULL CHECK (
        support_type IN ('supports', 'contradicts', 'corrects')
    ),
    evidence_order integer NOT NULL CHECK (evidence_order >= 0),
    PRIMARY KEY (user_id, summary_id, statement_id, evidence_id),
    UNIQUE (user_id, summary_id, statement_id, evidence_order),
    FOREIGN KEY (user_id, summary_id, statement_id)
        REFERENCES session_summary_statements (
            user_id, summary_id, statement_id
        ) ON DELETE CASCADE,
    FOREIGN KEY (user_id, summary_id, source_id)
        REFERENCES session_summary_sources (
            user_id, summary_id, source_id
        ) ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id, span_id)
        REFERENCES source_spans (user_id, source_id, span_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, claim_version_id)
        REFERENCES claim_versions (
            user_id, claim_id, version_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, span_id, support_type)
        REFERENCES evidence_links (
            user_id, claim_id, span_id, support_type
        ) ON DELETE RESTRICT
);

CREATE INDEX session_summaries_user_open_idx
    ON session_summaries (
        user_id, renderer_version, session_definition_id, transaction_from
    ) WHERE transaction_to IS NULL;
CREATE INDEX session_summary_sources_source_idx
    ON session_summary_sources (user_id, source_id, summary_id);
CREATE INDEX session_summary_evidence_claim_idx
    ON session_summary_statement_evidence (
        user_id, claim_id, claim_version_id, summary_id
    );
CREATE INDEX session_summary_evidence_span_idx
    ON session_summary_statement_evidence (user_id, span_id, summary_id);

CREATE FUNCTION purge_session_summaries_for_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM session_summaries AS summary
    USING session_summary_sources AS mapped
    WHERE mapped.user_id = OLD.user_id
      AND mapped.source_id = OLD.source_id
      AND summary.user_id = mapped.user_id
      AND summary.summary_id = mapped.summary_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER source_events_purge_session_summaries
    BEFORE DELETE ON source_events
    FOR EACH ROW EXECUTE FUNCTION purge_session_summaries_for_source();

CREATE TRIGGER source_spans_purge_session_summaries
    BEFORE DELETE ON source_spans
    FOR EACH ROW EXECUTE FUNCTION purge_session_summaries_for_source();

CREATE FUNCTION purge_session_summaries_for_evidence()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM session_summaries AS summary
    USING session_summary_sources AS mapped, source_spans AS span
    WHERE span.user_id = OLD.user_id
      AND span.span_id = OLD.span_id
      AND mapped.user_id = span.user_id
      AND mapped.source_id = span.source_id
      AND summary.user_id = mapped.user_id
      AND summary.summary_id = mapped.summary_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER evidence_links_purge_session_summaries
    BEFORE DELETE ON evidence_links
    FOR EACH ROW EXECUTE FUNCTION purge_session_summaries_for_evidence();
