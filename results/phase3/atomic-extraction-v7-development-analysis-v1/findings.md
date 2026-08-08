# Step 3.4 v7 development result

V7 completed all ten Maya cases and improved claim quality over v6. Its F1 was 0.658824, compared with 0.619048 for v6 and 0.642857 for v2.

## Promotion check

V7 cleared the claim-F1 gate. Precision and recall also exceeded v2. It did not clear the unsupported-memory gate: the scorer found five unsupported claims, compared with none in v2.

Two unsupported claims encoded a negated boolean as `false` with positive polarity. Three used a predicate or representation that did not align with the source-backed gold claim. These are narrow enough for one more candidate rather than accepting the regression.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not begin Step 3.5. V8 should preserve v7's claim coverage, normalize boolean polarity deterministically, and tighten the remaining predicate choices.

`metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v6.
