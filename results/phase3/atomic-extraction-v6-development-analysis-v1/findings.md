# Step 3.4 v6 development result

V6 completed all ten Maya cases. It improved several v5 metrics but still fails the promotion gates against v2.

## Full comparison

Claim F1 increased from 0.545455 in v5 to 0.619048 in v6. V2 remains higher at 0.642857. V6 precision was 0.722222, recall was 0.541667, and its unsupported-memory rate was 0.166667. V2 had no unsupported claims.

V6 had better valid-time accuracy than both earlier full runs. Its exact evidence scores remained above v2 but fell below v5. `metric_comparison.jsonl` records all thirteen metrics against full v2, the comparable v4 smoke subset, and full v5.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not begin Step 3.5. The next prompt should retain v6's recovered precision and field accuracy while removing unsupported claims and recovering the remaining missed claims.

## Limits

This result covers Maya development data only. The v4 full run is incomplete, so v4 comparisons use its completed three-case smoke. A single run per prompt version does not measure provider variance.
