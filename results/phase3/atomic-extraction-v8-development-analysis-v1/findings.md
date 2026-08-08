# Step 3.4 v8 development result

V8 completed all ten Maya cases. Claim F1 increased from 0.658824 in v7 to 0.704545; v2 scored 0.642857.

## Promotion check

V8 beat v2 on precision, recall, F1, predicate accuracy, valid-time accuracy, and exact evidence scores. Its unsupported-memory rate fell to 0.100000, but v2 had no unsupported claims.

The four remaining unsupported claims occur in two cases. One infers an employer where the source names only a job and location. Two assign help offers to the recipient instead of the helper. One treats a deliverable as a project assignment instead of a task.

## Decision

Tune one more candidate. Keep v3 as the accepted runtime default and do not begin Step 3.5. V9 should change only these subject and predicate checks, then use a smoke suite containing both affected cases.

`metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v7.
