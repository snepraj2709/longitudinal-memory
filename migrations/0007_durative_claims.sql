CREATE TABLE durative_inference_runs (
    run_id text PRIMARY KEY CHECK (run_id ~ '^[0-9a-f]{64}$'),
    user_id text NOT NULL REFERENCES memory_users (user_id) ON DELETE CASCADE,
    idempotency_key text NOT NULL CHECK (btrim(idempotency_key) <> ''),
    rules_version text NOT NULL CHECK (btrim(rules_version) <> ''),
    input_snapshot_sha256 text NOT NULL CHECK (
        input_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    extraction_version_id text NOT NULL REFERENCES extraction_versions (version_id)
        ON DELETE RESTRICT,
    transaction_as_of timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    decision_count integer NOT NULL CHECK (decision_count >= 0),
    accepted_count integer NOT NULL CHECK (accepted_count >= 0),
    rejected_count integer NOT NULL CHECK (rejected_count >= 0),
    UNIQUE (user_id, run_id),
    UNIQUE (user_id, idempotency_key),
    CHECK (accepted_count + rejected_count = decision_count)
);

CREATE TABLE durative_inference_decisions (
    decision_id text PRIMARY KEY CHECK (decision_id ~ '^[0-9a-f]{64}$'),
    user_id text NOT NULL,
    run_id text NOT NULL,
    decision_status text NOT NULL CHECK (
        decision_status IN ('accepted', 'rejected')
    ),
    decision_reason text NOT NULL CHECK (btrim(decision_reason) <> ''),
    derived_claim_id text,
    derived_version_id text,
    UNIQUE (user_id, decision_id),
    UNIQUE (user_id, run_id, decision_id),
    FOREIGN KEY (user_id, run_id)
        REFERENCES durative_inference_runs (user_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, derived_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, derived_claim_id, derived_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id)
        ON DELETE CASCADE,
    CHECK (
        (decision_status = 'accepted'
            AND derived_claim_id IS NOT NULL
            AND derived_version_id IS NOT NULL)
        OR
        (decision_status = 'rejected'
            AND derived_claim_id IS NULL
            AND derived_version_id IS NULL)
    )
);

CREATE TABLE durative_inference_evidence (
    user_id text NOT NULL,
    decision_id text NOT NULL,
    episode_id text NOT NULL CHECK (episode_id ~ '^[0-9a-f]{64}$'),
    role text NOT NULL CHECK (
        role IN ('supports', 'counter_evidence')
    ),
    evidence_order integer NOT NULL CHECK (evidence_order >= 0),
    supporting_claim_id text NOT NULL,
    supporting_version_id text NOT NULL,
    session_definition_id text NOT NULL CHECK (
        session_definition_id ~ '^[0-9a-f]{64}$'
    ),
    source_id text NOT NULL,
    span_id text NOT NULL,
    support_type text NOT NULL CHECK (
        support_type IN ('supports', 'contradicts', 'corrects')
    ),
    PRIMARY KEY (user_id, decision_id, episode_id),
    UNIQUE (user_id, decision_id, evidence_order),
    FOREIGN KEY (user_id, decision_id)
        REFERENCES durative_inference_decisions (user_id, decision_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, supporting_claim_id)
        REFERENCES claims (user_id, claim_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, supporting_claim_id, supporting_version_id)
        REFERENCES claim_versions (user_id, claim_id, version_id)
        ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id)
        REFERENCES source_events (user_id, source_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, source_id, span_id)
        REFERENCES source_spans (user_id, source_id, span_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id, supporting_claim_id, span_id, support_type)
        REFERENCES evidence_links (user_id, claim_id, span_id, support_type)
        ON DELETE CASCADE
);

CREATE INDEX durative_runs_user_rules_time_idx
    ON durative_inference_runs (
        user_id, rules_version, transaction_as_of, run_id
    );
CREATE INDEX durative_decisions_derived_claim_idx
    ON durative_inference_decisions (user_id, derived_claim_id)
    WHERE derived_claim_id IS NOT NULL;
CREATE INDEX durative_evidence_source_idx
    ON durative_inference_evidence (user_id, source_id, decision_id);
CREATE INDEX durative_evidence_supporting_claim_idx
    ON durative_inference_evidence (
        user_id, supporting_claim_id, supporting_version_id, decision_id
    );
CREATE INDEX durative_evidence_span_idx
    ON durative_inference_evidence (user_id, span_id, decision_id);

CREATE FUNCTION purge_durative_derivations_for_evidence()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    run_value text;
    derived_id text;
BEGIN
    FOR run_value, derived_id IN
        SELECT DISTINCT decision.run_id, decision.derived_claim_id
        FROM durative_inference_evidence AS evidence
        JOIN durative_inference_decisions AS decision
          ON decision.user_id = evidence.user_id
         AND decision.decision_id = evidence.decision_id
        WHERE evidence.user_id = OLD.user_id
          AND evidence.supporting_claim_id = OLD.claim_id
          AND evidence.span_id = OLD.span_id
          AND evidence.support_type = OLD.support_type
        ORDER BY decision.run_id, decision.derived_claim_id
    LOOP
        DELETE FROM durative_inference_runs
        WHERE user_id = OLD.user_id AND run_id = run_value;
        IF derived_id IS NOT NULL THEN
            DELETE FROM lifecycle_transitions
            WHERE user_id = OLD.user_id
              AND (claim_id = derived_id OR replacement_claim_id = derived_id);
            DELETE FROM processing_outbox
            WHERE user_id = OLD.user_id
              AND aggregate_id = derived_id
              AND event_type IN ('claims_changed', 'claim_lifecycle_changed');
            DELETE FROM evidence_links
            WHERE user_id = OLD.user_id AND claim_id = derived_id;
            DELETE FROM claim_versions
            WHERE user_id = OLD.user_id AND claim_id = derived_id;
            DELETE FROM claims
            WHERE user_id = OLD.user_id AND claim_id = derived_id;
        END IF;
    END LOOP;
    RETURN OLD;
END;
$$;

CREATE TRIGGER evidence_links_purge_durative_derivations
    BEFORE DELETE ON evidence_links
    FOR EACH ROW EXECUTE FUNCTION purge_durative_derivations_for_evidence();

CREATE FUNCTION purge_empty_durative_inference_run()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM durative_inference_runs
    WHERE user_id = OLD.user_id AND run_id = OLD.run_id;
    RETURN OLD;
END;
$$;

CREATE TRIGGER durative_decision_purge_empty_run
    AFTER DELETE ON durative_inference_decisions
    FOR EACH ROW EXECUTE FUNCTION purge_empty_durative_inference_run();
