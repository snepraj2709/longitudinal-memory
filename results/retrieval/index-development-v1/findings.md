# Retrieval index development findings

The deterministic build stored 33 atomic records and 17 session records for the two development users. Every record kept its exact claim-version and source-span lineage. The imported claims remain candidates, and the durative handoff remains empty.

The 256-dimensional vectors are signed token hashes for local storage and index testing. They are not semantic embeddings, and this step does not report retrieval-quality or latency metrics. No model was called.
