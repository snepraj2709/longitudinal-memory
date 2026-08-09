CREATE EXTENSION IF NOT EXISTS btree_gist;

ALTER TABLE claim_versions
    ADD COLUMN valid_from_date date,
    ADD COLUMN valid_from_timestamp timestamptz,
    ADD COLUMN valid_to_date date,
    ADD COLUMN valid_to_timestamp timestamptz,
    ADD COLUMN time_precision text;

UPDATE claim_versions AS version
SET valid_from_date = claim.valid_from_date,
    valid_from_timestamp = claim.valid_from_timestamp,
    valid_to_date = claim.valid_to_date,
    valid_to_timestamp = claim.valid_to_timestamp,
    time_precision = claim.time_precision
FROM claims AS claim
WHERE claim.user_id = version.user_id
  AND claim.claim_id = version.claim_id;

ALTER TABLE claim_versions
    ALTER COLUMN time_precision SET NOT NULL,
    ADD CONSTRAINT claim_versions_time_precision_check CHECK (
        time_precision IN ('timestamp', 'day', 'month', 'year', 'approximate', 'unknown')
    ),
    ADD CONSTRAINT claim_versions_valid_representation_check CHECK (
        NOT (
            (valid_from_date IS NOT NULL OR valid_to_date IS NOT NULL)
            AND (valid_from_timestamp IS NOT NULL OR valid_to_timestamp IS NOT NULL)
        )
    ),
    ADD CONSTRAINT claim_versions_valid_precision_shape CHECK (
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
    ADD CONSTRAINT claim_versions_valid_date_order CHECK (
        valid_from_date IS NULL OR valid_to_date IS NULL
        OR valid_from_date <= valid_to_date
    ),
    ADD CONSTRAINT claim_versions_valid_timestamp_order CHECK (
        valid_from_timestamp IS NULL OR valid_to_timestamp IS NULL
        OR valid_from_timestamp <= valid_to_timestamp
    ),
    ADD CONSTRAINT claim_versions_current_has_known_valid_time CHECK (
        lifecycle_status <> 'current' OR time_precision <> 'unknown'
    ),
    ADD CONSTRAINT claim_versions_owned_claim_version UNIQUE (
        user_id, claim_id, version_id
    ),
    ADD CONSTRAINT claim_versions_transaction_nonoverlap EXCLUDE USING gist (
        user_id WITH =,
        claim_id WITH =,
        tstzrange(transaction_from, transaction_to, '[)') WITH &&
    );

CREATE TABLE lifecycle_transitions (
    transition_id text PRIMARY KEY CHECK (btrim(transition_id) <> ''),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE RESTRICT,
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    claim_id text NOT NULL,
    from_version_id text NOT NULL,
    to_version_id text NOT NULL,
    target_status text NOT NULL CHECK (
        target_status IN (
            'confirmed', 'current', 'historical', 'disputed',
            'superseded', 'excluded'
        )
    ),
    reason text NOT NULL CHECK (btrim(reason) <> '' AND char_length(reason) <= 500),
    replacement_claim_id text,
    transitioned_at timestamptz NOT NULL,
    UNIQUE (user_id, transition_id),
    UNIQUE (user_id, idempotency_key),
    FOREIGN KEY (user_id, claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, from_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, claim_id, to_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id) ON DELETE RESTRICT,
    FOREIGN KEY (user_id, replacement_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE RESTRICT,
    CHECK (from_version_id <> to_version_id),
    CHECK (replacement_claim_id IS NULL OR replacement_claim_id <> claim_id)
);

ALTER TABLE processing_outbox
    DROP CONSTRAINT processing_outbox_event_type_check,
    ADD CONSTRAINT processing_outbox_event_type_check CHECK (
        event_type IN (
            'source_ingested', 'claims_changed', 'claim_recompute_required',
            'source_deleted', 'claim_lifecycle_changed'
        )
    );

CREATE INDEX claim_versions_user_status_transaction_idx
    ON claim_versions (
        user_id, lifecycle_status, transaction_from, transaction_to, claim_id
    );
CREATE INDEX lifecycle_transitions_user_claim_time_idx
    ON lifecycle_transitions (user_id, claim_id, transitioned_at, transition_id);
