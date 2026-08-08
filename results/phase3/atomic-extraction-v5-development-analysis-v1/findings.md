# Step 3.4 v5 development result

V5 completed all ten Maya cases without a structural failure, fixing the blocker that stopped v4. It is not ready to replace v3: compared with v2, it found fewer gold claims and introduced unsupported claims.

## Full comparison

The main trade-off is clear. Claim F1 moved from 0.642857 to 0.545455; precision moved from 0.750000 to 0.600000, and recall from 0.562500 to 0.500000. The unsupported-memory rate moved from 0.000000 to 0.275000.

Evidence handling improved: span precision moved from 0.305556 to 0.625000, and span recall from 0.220000 to 0.500000. Valid-time accuracy improved as well. `metric_comparison.jsonl` records all thirteen metrics and classifies each change.

## Structural result

The v4 full run stopped on its third case. V5 completed both its three-case smoke and the full ten-case run. One evidence quote needed the source-span fallback; the run metadata records that repair and its field location.

## Decision

Tune another candidate. Keep v3 as the accepted runtime default and do not start Step 3.5 yet. The next iteration should keep strict structured output and source-backed evidence repair, while recovering claim coverage and preventing unsupported claims.
