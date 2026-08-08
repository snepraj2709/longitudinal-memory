# Step 3.4 v9 development result

V9 completed all ten Maya cases. Claim F1 fell from 0.704545 in v8 to 0.620690; v2 scored 0.642857.

## Promotion check

V9 failed both quality gates. Its F1 was below v2, and its unsupported-memory rate rose to 0.153846. V2 had no unsupported claims.

The six unsupported outputs affect four cases. They include a duplicate calendar date, a calendar event assigned to its organizer instead of Maya, three overlapping employment predicates, and a location-qualified job mistaken for a role. Four are source-grounded facts that do not match the narrow expected predicate set. This run also shows that prompt-only tuning varies on cases outside the targeted smoke suite.

## Decision

Reject v9. Keep v3 as the accepted runtime default and do not begin Step 3.5. The next candidate should make narrow, source-based canonicalization deterministic instead of adding another prompt-only exception.

`metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v8.
