# Step 3.4 selection

Sneha approved keeping `atomic-extraction-v3` as the runtime prompt. The measured safety evidence remains the v2 run, which scored 0.642857 F1 with no unsupported claims.

V8 reached 0.704545 F1 but produced four unsupported claims. V9 fell to 0.620690 F1 and produced six. Neither candidate is promoted.

Step 3.4 is complete. Step 3.5 may begin after its own paid-run preflight and approval. It has not started yet.

## Known limits

- The v2 result missed 21 of 48 gold claims.
- V3 has contract-test coverage but no separate paid full-development result.
- Candidate tuning used Maya development cases only. Frozen test users were not inspected.
- Each candidate was run once, so the results do not measure provider variance.
