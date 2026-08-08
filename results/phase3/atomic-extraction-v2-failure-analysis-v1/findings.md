# Phase 3 atomic extraction v2: failure analysis

This report analyses the frozen ten-case v2 pilot. It does not rerun extraction or change the published scores.

## What the run got right

All 10 cases completed. The run produced 36 claims against 48 gold claims, with 27 exact matches. Precision was 0.750000, recall was 0.562500, and F1 was 0.642857. No prediction was classified as unsupported by the frozen scorer.

## Where it failed

The exact-match totals contain 9 false positives and 21 false negatives. The affected cases were `atomic_email_005` (4 FP, 5 FN), `atomic_email_003` (3 FP, 4 FN), `atomic_conv_002` (1 FP, 3 FN), `atomic_conv_004` (0 FP, 4 FN), `atomic_conv_010` (1 FP, 3 FN), `atomic_conv_003` (0 FP, 2 FN).

The classified issues were `evidence_span_mismatch` (25), `missed_gold_claim` (12), `missing_valid_time` (12), `extra_source_backed_claim` (9), `object_mismatch` (9), `epistemic_status_mismatch` (3), `predicate_mismatch` (1). Source-backed representation differences are kept separate from unsupported output: there were 9 representation pairs and 0 unsupported claims.

Valid-time accuracy was 0.666667. All 12 observed time errors were missing boundaries. The taxonomy can also record wrong or over-broad ranges when they occur.

Evidence was the weakest part of the run. Only 11 of 36 predicted spans exactly matched a gold span, while the gold set contained 50 spans. The case records show the missing and extra exact spans for every aligned evidence mismatch.

## Reading the artifacts

`case_failures.jsonl` is the claim-level audit trail. `summary.json` groups the same issues by case, source type, predicate family, and category. Execution failures have their own field and are not mixed into semantic errors. The manifest records the frozen inputs, protected B1 hashes, output hashes, denominator rules, and review state.
