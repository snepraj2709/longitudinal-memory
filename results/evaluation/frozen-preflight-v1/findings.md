# Step 10.2 findings

The deterministic preflight covers 100 synthetic sources and 570 cases across all eight baselines. It creates 25 resumable batches and plans 4,660 requests with no retries.

The expected incremental cost is $32.8869328. The hard maximum is $91.2131728. These are estimates, not provider usage.

The local runtime rows were read only to validate schemas, list opaque IDs, and count tokens. Their text was not logged. Gold, oracle, review queues, predictions, and credentials were not opened by the builder.

The existing key can be reused, but data transmission and paid execution are still unapproved. Step 10.3 has not started.
