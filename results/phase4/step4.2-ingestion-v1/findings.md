# Step 4.2 ingestion findings

Source ingestion now treats an exact replay as a no-op. Two concurrent copies create one source, attempt, and outbox event. A changed source, reused key, or deleted source returns a short error code without exposing stored content.

Workers lease pending attempts in PostgreSQL. Expired leases become failed attempts before one retry is queued. Validation, provenance, user-isolation, and stable-ID failures do not retry. Claim writes, evidence, attempt state, and outbox records share one transaction, so a failed write leaves no partial memory.

Deleting a source removes its spans and evidence. Claims with other evidence remain and receive a recompute event. Claims with no evidence are removed, and the source leaves a content-free tombstone. Repeating the delete changes nothing.

The Docker test replayed the 20 development sources and 33 candidate claims twice without adding duplicate records. It did not read evaluation gold or oracle data.

This step does not run an extractor, publish outbox events, update an index, or decide temporal lifecycle states. Step 4.3 will own lifecycle transitions and temporal queries.
