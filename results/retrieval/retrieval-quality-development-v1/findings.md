# Step 7.4 retrieval quality findings

This development evaluation scores the frozen B2, B3, and B4 rankings. It did not change retrieval behavior.

The review covered all 200 same-user record/query pairs. Rankings were not used while assigning labels. This was not a blind evaluation, and prior exposure to the committed rankings was possible.

- B2: Recall@10 1.000000; nDCG@10 0.983456; MRR 1.000000; stale-memory rate 0.014085.
- B3: Recall@10 1.000000; nDCG@10 0.836945; MRR 0.854167; stale-memory rate 0.016393.
- B4: Recall@10 0.975000; nDCG@10 0.910682; MRR 0.937500; stale-memory rate 0.027778.

The dataset is small, development-only, candidate-heavy, and uses deterministic token-hash vectors rather than a semantic embedding model. Scores are diagnostics, not production-quality claims.

No provider or model was called. Incremental cost was $0; historical OpenAI spend remains $0.2314404.
