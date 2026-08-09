# Temporal evaluation findings

The run completed 12 predictions and 0 failures across 12 reviewed development cases.

Each score keeps its denominator visible. A missing denominator is recorded as null with a reason, and a failed case remains in every metric that applies to it.

The six accuracy metrics were 1.0. Mean interval IoU was 0.027027 for one scored pair: the inclusive shared endpoint contributes one overlapping day across a 37-day union.

This evaluator reports explicit temporal behavior only. It does not infer lifecycle changes or modify the frozen Phase 3 claims.
