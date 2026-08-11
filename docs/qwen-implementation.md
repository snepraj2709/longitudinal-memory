# Qwen B0-B7 implementation and execution contract

## 1. Scope and authority

This document controls the `qwen35-27b-fp8-v2` implementation, JarvisLabs execution, scoring, demo publication, and Railway deployment. Follow [benchmark.md](benchmark.md) for evaluation semantics and [memory-ontology.md](memory-ontology.md) for claims, evidence, time, lifecycle, and provenance.

The approved run is bounded by all of these conditions:

- Model: `Qwen/Qwen3.5-27B-FP8` at Hugging Face revision `97f5941bf617e31c5e237364a8602ce3f03a551a`.
- Runtime: vLLM revision `65b7662d3fcb773afaf751ab29ac6960a0cf011d`.
- Resource: one H100 80 GB spot container in `IN2`, or one RTX-PRO6000 96 GB spot container in `IN1` as the only fallback.
- Price ceiling: H100 spot at or below INR 133.33/hour; RTX-PRO6000 spot at or below INR 100/hour.
- Total JarvisLabs cap: INR 1,500, including setup, model download, idle time, inference, judging, transfer, failures, and cleanup time.
- Branch and deployment: push `testing`, then deploy that exact commit to Railway project `longitudinal-memory`, service `demo`.
- Public domain: `longitudinal-memory-benchmark.up.railway.app`.

Stop before creating an instance if authentication fails, neither approved resource is available below its ceiling, or any model, revision, provider, region, resource class, budget, dataset, branch, Railway plan, or deployment target differs. A change to any of those terms needs a new explicit approval.

Execution is paused after eight setup attempts on 2026-08-11. They spent INR 39.69 in total and produced zero provider responses. Every instance was destroyed. The first seven attempts failed during setup; the eighth was stopped during model download after the user withdrew approval for further paid debugging. Do not start another instance from the earlier approval.

A retry now needs fresh approval and the exact `--confirm-paid-gpu qwen35-27b-fp8-v2-paid-gpu-approved` argument. The confirmation is an operator safety lock, not a substitute for approval in the conversation.

## 2. Historical series

Do not modify or combine these series:

| Series | Status | Rule |
| --- | --- | --- |
| `openai-gpt41-v1` | `interrupted_not_scored` | Preserve all configurations, checkpoints, failures, hashes, empty prediction files, and the recorded `$0.4399284` spend. Do not resume it. |
| `qwen35-27b-fp8-v1` | `superseded_not_run` | Preserve the configuration, compatibility pack, and empty result scaffold as historical planning evidence. Do not execute it. |
| `qwen35-27b-fp8-v2` | next execution series | Regenerate Qwen extraction, contexts, predictions, judge diagnostics, scores, and demo artifacts under new immutable paths. |

The v2 root is `results/evaluation/qwen35-27b-fp8-v2/`. A command must refuse to write into an existing sealed run or another series root.

## 3. Required local state

Before implementation or execution:

```bash
git switch testing
git status --short
git rev-parse HEAD
git rev-parse origin/testing
docker --version
jl --version
jl status --json
jl gpus --json
jl resources --json
railway status
```

Requirements:

- The worktree contains no unrelated staged change.
- Docker is available for the ephemeral PostgreSQL integration gate.
- `jl status --json` succeeds without printing a credential.
- Hugging Face and vLLM credentials exist only in local environment or ignored local files.
- No credential is passed as a command-line literal, written to a result, uploaded in the repository archive, printed in logs, or configured on Railway.
- The frozen manifest and all protected source hashes match the v2 configuration.

If Jarvis authentication needs repair, the owner must enter or configure the token locally. Never request that a token be pasted into chat. Re-run `jl status --json` after repair.

## 4. Frozen v2 configuration

Create `configs/evaluation/qwen35_27b_fp8_v2.json` with these immutable values:

| Field | Value |
| --- | --- |
| Series and model alias | `qwen35-27b-fp8-v2` |
| Model revision | `97f5941bf617e31c5e237364a8602ce3f03a551a` |
| vLLM revision | `65b7662d3fcb773afaf751ab29ac6960a0cf011d` |
| Context | 16,384 tokens, text only |
| Thinking | disabled in every request |
| Server generation config | `vllm` |
| Tensor parallel size | 1 |
| GPU memory utilization | 0.90 |
| Maximum server sequences | 16 |
| Client concurrency | 8 |
| Sampling seed | 42 |
| Runtime generation sampling | frozen per task in the config |
| Judge temperature | 0 |
| Invalid-output retries | 0 |
| Transport retries | one per eligible request, 25 globally |

The configuration must pin hashes for prompts, JSON schemas, tokenizer inputs, predicate registry, split manifest, context builder, scorer, answerability policy, and database migrations. Verification must fail on coordinated configuration or artifact rebinding.

## 5. Data-access boundaries

Use separate processes and path allowlists for prediction and scoring.

### Prediction process may read

- Runtime source records for its current split.
- Runtime QA, summary, and interactive case prompts for its current split.
- The frozen comparison definition and runtime-only configuration.
- Its own Qwen extraction checkpoints and materialized Phase 4-7 runtime artifacts.
- Its own completed prediction checkpoints when resuming.

### Prediction process must not read

- Gold answers, gold claims, gold evidence, scorer labels, or expected outputs.
- Oracle files or review queues.
- Another split's source or case records.
- OpenAI predictions or extraction output.
- Qwen v1 results.
- Another user's source, memory, context, or checkpoint while processing a case.
- `.env`, provider credentials, or Railway configuration through runtime path traversal.

### Scoring process

Gold may open only after the corresponding prediction checkpoint set is complete, ordered predictions have been composed, and a sealed prediction manifest records every expected request as valid or failed. Scoring writes to a sibling score directory and cannot alter predictions.

Development and frozen-test processes use different allowlists, manifests, seals, and score roots. The frozen-test process must not open development gold as an implicit fallback.

## 6. Correct baseline construction

Every case uses only records for the same user with source time visible before the case `as_of` boundary.

| Baseline | Context supplied to Qwen |
| --- | --- |
| B0 | Query or task instruction only. |
| B1 | Complete same-user source history visible before `as_of`, in deterministic chronological order. |
| B2 | Top-k Qwen-extracted atomic memories from the Phase 7 atomic index. |
| B3 | Top-k grounded session summaries produced by the Phase 6 pipeline and retrieved through the Phase 7 session index. |
| B4 | Fused Phase 7 atomic and session retrieval. |
| B5 | B4 with persisted Phase 4 valid-time, transaction-time, lifecycle, and `as_of` filtering. |
| B6 | B5 with persisted Phase 5 relation classification and deterministic current-belief resolution. |
| B7 | The exact B6 context and exact underlying B6 Qwen response, followed by the deterministic Phase 9 answerability policy. No additional model request. |

Do not use `src/evaluation/frozen_contexts.py` for v2. Materialize Qwen claims into an ephemeral local PostgreSQL database through the production Phase 4 ingestion and temporal services. Run the Phase 5 relation and resolver path, Phase 6 session and summary path, and Phase 7 indexes and retrieval path.

Persist for each logical prediction:

- Dataset, split, user, task, case, baseline, and `as_of` identity.
- Ordered source, claim, session, relation, and evidence identifiers.
- Context payload hash and canonical context hash.
- Prompt, schema, model, sampling, and response hashes.
- Provider request identity when a provider call exists.
- Structural validity, failure type, latency, token counts, and retry count.
- For B7, the paired B6 prediction ID, shared context hash, shared raw response hash, and deterministic gate output.

The verifier must prove that B6 and B7 share the same context and underlying response hashes. It must also prove that B7 caused no provider call and that the number of B0-B6 provider calls matches the planned denominator.

## 7. PostgreSQL materialization

Use the repository's PostgreSQL/pgvector service only on the local machine. Railway remains artifact-only.

Execution order for one split:

1. Start a clean ephemeral database using the pinned Compose service.
2. Apply all repository migrations and record their hashes.
3. Load only the split's runtime sources.
4. Run Qwen extraction and checkpoint each source response.
5. Validate extraction structure without scorer gold.
6. Ingest valid extracted claims through the Phase 4 service with source provenance and transaction time.
7. Build and persist temporal versions and lifecycle transitions.
8. Generate, classify, and persist Phase 5 relation candidates; resolve beliefs deterministically.
9. Build Phase 6 sessions, grounded summaries, and durative claims from Qwen outputs where the contract requires model output.
10. Build Phase 7 atomic and session indexes, then materialize B2-B6 context packages.
11. Export canonical runtime artifacts and database row-count/hash receipts.
12. Drop the database after artifacts and receipts pass verification.

The materializer must be idempotent on a clean database and byte-stable after canonical export. Tests must include corrections, simultaneous conflicts, changing preferences, valid-time boundaries, transaction-time replay, supersession, deletion/exclusion, abstention, cross-user isolation, and failed extraction.

## 8. Request and checkpoint protocol

Use eight concurrent workers. Concurrency changes scheduling only; final JSONL order is the frozen request order.

For every provider request:

1. Check the wall-clock budget before scheduling.
2. Build the task payload from allowlisted runtime data.
3. Send the pinned model alias, non-thinking controls, task response schema, and sampling settings.
4. Write the raw response, sanitized error metadata, usage, latency, and validity to a temporary file.
5. `fsync` the file and parent directory, then atomically rename it to the request checkpoint path.
6. Update append-only cost and event ledgers without storing credentials.
7. Do not schedule more work once a stop marker exists.

Checkpoint layout:

```text
results/evaluation/qwen35-27b-fp8-v2/
  stage1/compatibility-run-001/
  stage2/development-run-001/
    extraction/checkpoints/<request_id>.json
    contexts/<baseline>/<task>/<case_id>.json
    predictions/checkpoints/<request_id>.json
    predictions/predictions.jsonl
    predictions/failures.jsonl
    judge/checkpoints/<request_id>.json
    scores/
    manifests/
  stage3/frozen-run-001/
    ...
  published/
```

A checkpoint is terminal when it contains a valid response, invalid structured output, non-retryable provider error, or exhausted eligible transport retry. Resume skips all terminal checkpoints.

Retry only when the transport failed before any response body was received. Retry that request once, and stop retries globally after 25. Never retry invalid JSON, schema mismatch, wrong model identity, truncated body, refusal, or any response that contains a body.

## 9. Server launch contract

Create one spot container and keep it running through accepted stages to avoid a second model download. Use 100 GB storage and the PyTorch template.

Selection order:

1. Re-read `jl gpus --json` and `jl resources --json` immediately before creation.
2. Select H100 container spot in `IN2` only when a free spot device exists and the price is at or below INR 133.33/hour.
3. Otherwise select RTX-PRO6000 container spot in `IN1` only when a free spot device exists and the price is at or below INR 100/hour.
4. Stop without spending if neither condition is true.

The rates observed on 2026-08-11 were INR 112.59/hour for H100 spot and INR 93.96/hour for RTX-PRO6000 spot. These observations are not execution approval by themselves; the lifecycle command must parse and record fresh availability and price data immediately before creation.

The lifecycle wrapper, not an operator's memory, owns cleanup. It must install `EXIT`, `INT`, and `TERM` traps immediately after recording the machine ID. The trap stops local workers, downloads all completed checkpoints and logs, destroys the instance, and verifies through both `jl list --json` and `jl get <id> --json` that the instance no longer exists. Pausing is not accepted as cleanup.

The PyTorch template uses Python 3.10. The pinned vLLM build installs FlashInfer `0.6.16.post3`, whose `array.array[int]` annotation fails under that interpreter. Create an isolated Python 3.12 environment from conda-forge and install the exact vLLM wheel there. Before downloading model weights, require all of these checks:

1. Import the pinned vLLM version and `flashinfer.comm.fd_exchange` under Python 3.12.
2. Download only the pinned model config and tokenizer files. Weight files must remain excluded.
3. Start the complete vLLM engine with `--load-format dummy`, a 1,024-token context, and one sequence.
4. Confirm that `/v1/models` reports only `qwen35-27b-fp8-v2`.
5. Stop the whole dummy-server process group before downloading the 30.9 GB model snapshot.

The production server must bind to `0.0.0.0:6006` and use:

```text
--served-model-name qwen35-27b-fp8-v2
--max-model-len 16384
--tensor-parallel-size 1
--max-num-seqs 16
--gpu-memory-utilization 0.90
--seed 42
--generation-config vllm
--reasoning-parser qwen3
--language-model-only
```

The model revision is enforced during the separate Hugging Face snapshot download. The server loads that sealed local directory. Do not pass the removed `--task generate` flag.

Store the API key only in process environment. Confirm `/v1/models` reports exactly the expected alias before the first request. Record server startup time, model revision, vLLM revision, GPU name, driver, CUDA version, and `nvidia-smi` output with credential fields removed.

## 10. Budget accounting and stop logic

Cost begins at successful instance creation and ends only after destruction is confirmed. Use provider timestamps when available and a monotonic local timer as the conservative fallback.

Before scheduling a request, calculate:

```text
current_cost = elapsed_billable_hours * observed_hourly_rate
reserved_cleanup_cost = conservative_minutes_for_checkpoint_flush_download_destroy
projected_next_cost = recent_seconds_per_request * observed_hourly_rate / 3600
```

Stop scheduling when `current_cost + reserved_cleanup_cost + projected_next_cost` could cross the current stage cap or INR 1,500 cumulative cap. The cleanup reserve is never available for inference.

### Stage 1: compatibility

- Cap: INR 200; cumulative cap: INR 200.
- Provider requests: 12 non-scored requests.
- Composition: three extraction, three B0 task outputs, three B6-context task outputs, and three judge batches. The task set must cover QA, summary, and interactive output contracts.
- Pass gate: 12/12 structurally valid; exact alias and revisions; no OOM; no context overflow; no credential or prohibited-path read; projected remaining run plus 20% reserve fits INR 1,300.

Any failed Stage 1 request fails the gate. Download artifacts and destroy the instance; do not continue to Stage 2.

### Stage 2: development

- Incremental cap: INR 300; cumulative cap: INR 500.
- Users: the two development users only.
- Extraction calls: 20.
- B0-B6 answer calls: 798.
- Logical B0-B7 predictions: 912 (114 cases across eight baselines).
- Judge calls: 112.
- Total provider requests: 930.
- Pass gate: 20/20 extraction structurally valid; at least 884/930 provider responses structurally valid; every task/baseline provider batch at least 90% valid; no integrity, leakage, user-isolation, OOM, or B6/B7 identity failure; projected Stage 3 cost plus 20% reserve fits INR 1,000.

Seal development predictions before opening development gold. Then run deterministic development scoring and the approved judge diagnostics while the same instance is available. Do not make commits while it is billing. Continue to Stage 3 only after the complete Stage 2 gate passes.

### Stage 3: frozen test

- Incremental cap: INR 1,000; cumulative cap: INR 1,500.
- Users: the eight frozen-test users only.
- Extraction calls: 80.
- B0-B6 answer calls: 3,192.
- Logical B0-B7 predictions: 3,648 (456 cases across eight baselines).
- Judge calls: 448.
- Total provider requests: 3,720.

Stage 3 starts only after the Stage 2 gate passes. Predictions must seal before frozen gold opens. An incomplete budget-limited Stage 3 remains visible as incomplete and is not scored as though missing cases were incorrect.

Maximum planned provider calls are 4,662 including Stage 1. The absolute ceiling is 4,687 if all 25 eligible transport retries are consumed.

## 11. Semantic judge diagnostic

The semantic judge is diagnostic and runs only after the corresponding predictions are sealed. It uses the same pinned Qwen model, temperature 0, non-thinking mode, and a versioned structured schema.

- A judge batch contains no more than ten cases.
- Every batch contains one user and one task.
- Inputs hide baseline name, model name, series identity, and provider identity.
- Randomized presentation order is deterministic from the frozen seed and recorded.
- QA labels cover semantic correctness against the scorer-only reference.
- Summary labels cover faithfulness and material omission.
- Interactive labels cover required behaviour and unsafe unsupported behaviour.

Judge output must never determine evidence correctness, provenance, leakage, execution validity, structural validity, or whether a deterministic metric passes. Do not use it to create a composite score. Preserve invalid judge output as a judge failure without changing the underlying prediction.

## 12. Authoritative scoring

Publish development and frozen-test scorecards separately. Every metric record includes:

- `series_id`, split, task, baseline, capability, and metric group.
- Numerator, denominator, value, and explicit null reason.
- Completion and structural-validity status.
- Extraction, retrieval, temporal, conflict, answer/evidence, abstention, summary, or interactive metric identity.
- Requests, failures, input/output tokens, latency, throughput, GPU, hourly rate, elapsed billable time, and INR cost where applicable.
- Configuration, input, prediction, scorer, and output hashes.

Deterministic metrics are authoritative. Report regressions and failed cases. Do not collapse tasks or components into one score. Do not convert an incomplete batch, execution failure, or unavailable denominator into zero.

Required comparisons include B2/B3/B4 retrieval, B4/B5 temporal and lifecycle effects, B5/B6 conflict effects, and B6/B7 coverage-risk effects. Report extraction quality as an upstream bound on later metrics.

Two clean scoring runs over sealed inputs must produce byte-identical JSON and JSONL artifacts.

## 13. Demo publication

The public application remains artifact-only and keeps these API paths unchanged:

- `GET /api/demo/cases`
- `GET /api/demo/cases/{case_id}`
- `GET /api/runs`
- `GET /api/scorecards`
- `GET /healthz`

Extend the sanitized bundle with series, split, task, metric group, numerator, denominator, value, null reason, GPU, cost, validity, and completion fields. Include B0 and use these labels:

- B5: temporal and lifecycle aware.
- B6: conflict aware.
- B7: answerability gated.

Public guided cases may include sanitized development-only source timelines and B0-B7 `baseline_results`. Never include frozen-test case content, gold, oracle data, review queues, prompts, raw invalid model output, credentials, provider URLs, or private checkpoint metadata. Frozen-test publication is aggregate only.

The demo must retain:

- OpenAI as `interrupted_not_scored` with its recorded historical cost.
- Qwen v1 as `superseded_not_run` with zero execution cost.
- Qwen v2 as its measured completion state, including budget-limited or failed states.

Bundle generation must be byte-stable. Runtime file-access tests must prove that the API and image cannot read gold, oracle, review, secret, or provider-execution paths.

## 14. Verification gates

Every implementation commit requires focused tests, deterministic reconstruction where applicable, `git diff --check`, staged and unstaged diff inspection, secret scanning, prohibited-path scanning, and explicit-path staging.

Before Stage 1:

```bash
make test
# Run the repository's PostgreSQL integration targets for Phases 4-9.
# Run the v2 dry-run, request-count, checkpoint, retry, budget, and cleanup tests.
npm --prefix web run build
```

Required automated coverage:

- Qwen payload, authentication, model/revision mismatch, non-thinking controls, and task schemas.
- Atomic checkpointing, deterministic JSONL composition, resume, transport retry ceiling, invalid-output no-retry, and concurrency ordering.
- Stage and cumulative budget stops, cleanup reserve, signal handling, interrupted downloads, destruction, and absence verification.
- PostgreSQL-backed B0-B7 semantics and B6/B7 context-response identity.
- Gold-open ordering and runtime rejection of gold, oracle, review, secrets, cross-user data, and wrong-split data.
- Deterministic scoring, judge blinding and batching, failed/incomplete denominators, and immutable result roots.
- API contracts and unsupported methods.
- Desktop and mobile replay, evidence inspection, baseline switching, filters, empty states, failures, and incomplete-budget display.
- Clean `make demo` startup and zero provider calls.

## 15. Commit sequence

Make these commits in order. Do not combine layers or stage unrelated paths.

1. `docs: refresh implementation handoff`
2. `docs: define Qwen end-to-end execution`
3. `evaluation: freeze corrected Qwen comparison series`
4. `evaluation: materialize true B0 B7 contexts`
5. `evaluation: add concurrent Qwen execution`
6. `evaluation: add Qwen judge and scorecard`
7. `evaluation: harden Jarvis budget lifecycle`
8. `evaluation: record Qwen compatibility run`
9. `evaluation: record Qwen development run`
10. `evaluation: record Qwen frozen run`
11. `evaluation: publish Qwen scorecard`
12. `demo: publish completed Qwen benchmark`
13. `docs: close Qwen benchmark handoff`

Do not spend GPU time making commits. During live execution, write and download checkpoints, complete or stop the approved stages, destroy the instance, verify destruction, and only then inspect and commit stage artifacts.

## 16. Final push and deployment

After all applicable gates and commits pass:

```bash
git push origin testing
git rev-parse HEAD
git rev-parse origin/testing
railway status
railway up --service demo
```

Verify that `origin/testing` equals local `HEAD` and Railway deploys that exact commit. Stop if Railway requests payment or a plan change.

Smoke-test the retained public domain:

```bash
curl --fail --silent --show-error https://longitudinal-memory-benchmark.up.railway.app/healthz
curl --fail --silent --show-error https://longitudinal-memory-benchmark.up.railway.app/api/runs
curl --fail --silent --show-error https://longitudinal-memory-benchmark.up.railway.app/api/scorecards
curl --fail --silent --show-error https://longitudinal-memory-benchmark.up.railway.app/api/demo/cases
```

Confirm `X-Robots-Tag` noindex headers, immutable assets, read-only method rejection, sanitized public content, no Railway provider credentials, and zero provider calls from the hosted service.

## 17. Completion record

After scoring and deployment, update this document and [Implementation-handoff.md](Implementation-handoff.md) with:

- Final commit and deployed commit.
- Observed GPU, region, spot rate, instance creation and destruction timestamps, and verified destruction evidence.
- Stage request counts, validity, retries, tokens, throughput, elapsed billable time, and INR cost.
- Complete, incomplete, failed, and unscored batches.
- Development and frozen-test artifact paths and hashes.
- Scorecard and demo bundle hashes.
- Test commands and results.
- Railway deployment ID, URL, endpoint smoke tests, and confirmation of zero hosted provider calls.

The benchmark is complete only when all 100 source extraction attempts and all 4,560 logical B0-B7 predictions exist or have explicit execution failures, B6/B7 pairing is proven, scorecards reproduce byte-for-byte, the public bundle is sanitized, Railway serves the exact pushed commit, and the Jarvis instance is confirmed destroyed.
