# Step 3.4 extraction development result

The v4 candidate is blocked. Its full Maya run stopped on `atomic_conv_003` because the model response failed deterministic validation. Two of ten cases completed before the stop. The protocol does not allow a resume or a score for this failure type, so the accepted v3 prompt remains the runtime default.

## What changed

The candidate adds one prompt paragraph. It asks the model to cover independent clauses, use shorter exact evidence spans, keep boolean objects separate from polarity, mark explicit denials, and fill valid-time boundaries when the source gives a date.

## Smoke result

The three-case smoke completed and produced 18 claims for 21 gold claims. Against the same three v2 cases, claim F1 moved from 0.540541 to 0.512821. Evidence precision moved from 0.187500 to 0.722222, and evidence recall moved from 0.136364 to 0.590909. Unsupported-memory rate moved from 0.000000 to 0.222222.

This smoke is not a release comparison. It covers only three cases and cannot replace the incomplete full run.

## Failure breakdown

For the valid smoke, issues by source type were `conversation` (24 issues), `email` (18 issues). Predicate-family issues were `belief` (3 issues), `commitment` (12 issues), `event` (3 issues), `goal` (1 issue), `role` (11 issues), `schedule` (2 issues), `state` (4 issues), `task` (6 issues). The active failure categories were `missed_gold_claim` (7), `extra_source_backed_claim` (4), `unsupported_claim` (4), `predicate_mismatch` (4), `object_mismatch` (6), `polarity_mismatch` (1), `epistemic_status_mismatch` (2), `missing_valid_time` (7), `evidence_span_mismatch` (7).

The full-run failure record is intentionally sanitized. It identifies the case and validation stage but does not keep the invalid response, so this analysis cannot name the rejected field. `summary.json` contains every smoke metric, the v2 deltas, and the source-type, predicate-family, and category breakdowns. `case_failures.jsonl` contains the claim-level smoke audit trail.

## Decision

Do not promote v4 and do not start Step 3.5. A future, separately authorized development iteration would need a new prompt version and a fresh bounded run; this failed full run must remain unchanged.
