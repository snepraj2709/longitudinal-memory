# Step 4.3 findings

The migration copies each existing claim's valid time into its version row without changing the saved status or confidence. After that, the service keeps every lifecycle change as a new version. Earlier rows retain their status, confidence, valid-time snapshot, and transaction interval.

Transitions are explicit. The caller supplies the target status, reason, idempotency key, and aware timestamp. Corrections name both claims and update them in one transaction. The service does not infer relations or choose a replacement.

As-of reads require a user and transaction cutoff. They use half-open transaction intervals, inclusive valid-time bounds, and only evidence from sources visible by that cutoff. Deleted sources stay hidden, even for an earlier cutoff.

The Docker tests covered an incremental upgrade, concurrent requests, idempotent replay, competing transitions, correction rollback, late sources, and tombstone dominance. PostgreSQL also rejects an audit row if its versions belong to another claim, even within the same user. The existing 20-source replay still stores all 33 development claims as candidates because their memory kind is unset.

Step 4.4 still needs reviewed temporal cases and a deterministic scorer. No model calls, gold reads, hosted writes, or lifecycle inference were used here.
