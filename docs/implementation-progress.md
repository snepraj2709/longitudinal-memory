# Implementation progress

## Phase 3, Step 3.5: Produce Phase 4 input

Status: complete

### Repository state

- Starting commit: `f9cd3cd5e6dd60981aed57b73afd4ab52b03d7e5`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Commit message: `memory: produce Phase 4 development claim input`
- The worktree was clean after the final commit check.

### Inputs and frozen contracts

- Dataset: `scaled_v1`, development split, SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`
- Runtime users: `data/scaled-v1/runtime/users.jsonl`, SHA-256 `13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef`
- Runtime sources: `data/scaled-v1/runtime/sources.jsonl`, SHA-256 `a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de`
- Development gold: SHA-256 `a74aa1e2f07b238ee837dfea80f0562c1a2c9fb81f02fa6b168dd0e3290cfff3`. The runner opens it only after all 20 development predictions exist.
- Qualification prompt: SHA-256 `57ec5307bf3cdf691e303ccd65602a0d3856f95b442c8408e44fd74e9d05ac5b`
- Full prompt: SHA-256 `1a6a371a73b043807249a4c130b5b309b8a29fbcbea34f6c2e912f6eded38e59`
- Schema: SHA-256 `f694616f02143cc4e5595fe0658b100f29562c9c4d4df716af362fa10989aac1`
- Predicate registry v2: 70 predicates, canonical SHA-256 `5cb9ba2f2b7a81aa2ce61d52d9425fbf1f2d39964a3a66286d5679ebb2e5175d`, file SHA-256 `15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1`
- The legacy v1 registry and the frozen v3 extraction prompt kept their canonical hashes, `64991ee0b9e52e61d5238b2c404447d22d633e3adf0fb858757dd2fa5c574b4d` and `e399c3c6108cf2103e844f762c8b34339f5a096a16bd5c91c5a5500c6482bff2` respectively.
- Guidance: `step-3.5-guidance-v1`, envelope SHA-256 `f0625309517640cf48611a88cf126bb9f00e88dc6a186f846d73cb522a19ca75`
- Recovery guidance: `step-3.5-recovery-guidance-v1`, envelope SHA-256 `bb84b4c8bef85be389675832b2c106f2fce2a10dc30e6e3f7e1aedd66d265501`
- Fallback guidance: `step-3.5-gpt41-fallback-guidance-v1`, envelope SHA-256 `016f94919f46184994849fc46dd63b3259d6054096cffde210f9324799090f86`

### Implementation

- Added a development-only source loader for `user_001` and `user_002`. It stops before parsing test-user records and sends only participant entities and adapted source observations to the provider.
- Added the exact Phase 4 claim contract, canonical claim IDs, strict source-result rejection, exact quote and evidence checks, time-precision checks, and user-scoped provenance validation.
- Added deterministic development scoring, including user, source-type, and predicate slices with explicit null denominators.
- Added frozen qualification, recovery, and GPT-4.1 fallback runners. They enforce exact request order, requested and returned model snapshots, zero retries, cumulative cost ceilings, checkpoint integrity, and immutable completed resume.
- Added the pinned `tiktoken==0.13.0` request-body counter with a 256-token reserve for every request.
- Extended the existing prompt, schema, source, atomic validation, and run-safety modules through parameterised paths. Legacy prompt and registry behaviour remains covered by its original tests.
- Added focused tests for the source adapter, claim boundary, scorer, qualification and recovery runner, fallback runner, and immutable resume.

### Paid execution

The original qualification stopped after one GPT-4.1 request returned no usable output. It charged 5,023 input tokens and 1,200 reserved output tokens, or exactly `$0.0196460`. It did not open gold and was not retried.

The recovery qualification made eight requests. GPT-4.1 used 17,130 input and 930 output tokens for exactly `$0.0417000`. GPT-4.1 mini used 17,130 input and 999 output tokens for exactly `$0.0084504`. The mini calendar result at qualification position 4 failed the frozen output contract after a provider response. The failure was preserved as validation, was not retried, and prevented qualification scoring and the original final run. Recovery cost was exactly `$0.0501504`.

The approved fallback reused the four compatible GPT-4.1 recovery predictions and made the remaining 16 GPT-4.1 requests in original source order. Those new requests used 68,398 input and 3,106 output tokens, costing exactly `$0.1616440`.

Across the three attempts, 25 provider requests were charged with zero retries. GPT-4.1 used 90,551 input and 5,236 output tokens. GPT-4.1 mini used 17,130 input and 999 output tokens. Exact cumulative spend was `$0.2314404`, below the fallback authorization ceiling of `$0.3839604`.

### Tests and contract checks

- Focused Step 3.5 suite: 32 tests passed. After the final completed-resume integrity change, the fallback and runner subset passed 23 tests in 7.617 seconds.
- `make validate-scaled-benchmark`: passed with 10 users, 100 sources, 160 claims, 500 QA cases, 50 summaries, and 20 interactive cases.
- `make test`: 339 tests passed in 51.788 seconds.
- Python compilation, `git diff --check`, and `git diff --cached --check`: passed.
- Completed fallback resume: verified all nine artifact hashes, returned 20 predictions, made zero provider calls, and changed zero bytes.
- The runtime source order was exactly 20 sources: ten for `user_001`, followed by ten for `user_002`. Predictions retained positions 1 through 20 in that order.
- The final release has 20 successful source results, 33 canonical claims, 33 exact evidence links, and zero execution or persistence failures. All claims and the reverse evidence index revalidated.
- Runtime prompts and provider payloads contain no gold, oracle, review, lifecycle, profile, task, or split fields. Frozen test users were not opened or transmitted.
- Requested, resolved, and returned final models are all `gpt-4.1-2025-04-14`.
- `preference.md`, `docs/memory-evaluation-steps.md`, `data/scaled-v1`, the v1 predicate registry, Phase 3 source and gold data, the protected Phase 3 v2 result, and B1 remained unchanged from the starting commit.
- The changed-file and staged secret scans found no API key, private key, `.env`, or secret-shaped token.

### Scorecard

The final development scorecard compares 33 predicted claims with 32 reviewed claims.

| Measure | Result |
| --- | ---: |
| Claim precision | 0.303030 |
| Claim recall | 0.312500 |
| Claim F1 | 0.307692 |
| Provenance-span precision | 0.272727 |
| Provenance-span recall | 0.281250 |
| Subject accuracy | 1.000000 |
| Speaker accuracy | 1.000000 |
| Predicate accuracy | 0.620690 |
| Object accuracy | 0.482759 |
| Polarity accuracy | 0.896552 |
| Epistemic-status accuracy | 0.655172 |
| Valid-time accuracy | 0.000000 |
| Unsupported-memory rate | 0.454545 |

### Failures and limitations

- The qualification failures remain in their immutable v1 and v2 result directories. A provider no-output failure is kept separate from the later mini validation failure.
- Extraction quality is weak. Only 10 of 32 reviewed claims matched, 15 of 33 predictions were unsupported, provenance recall was 0.281250, and no aligned claim received a matching valid-time value.
- The result covers two synthetic development users. It does not measure frozen test performance and was not tuned against test users.
- Four final predictions are byte-compatible outputs reused from the recovery qualification. The manifest records their source positions, request hashes, model, and predecessor hashes.
- Phase 4 must treat this as measured development input, not as evidence that extraction quality meets a research target. The structural handoff is valid, but downstream work must preserve the scorecard and allow claims to be replayed or replaced under a later extractor version.

### Artifacts and hashes

- Original qualification: `results/phase3/atomic-extraction-step35-model-qualification-v1`, manifest SHA-256 `a0d0dfba7016e6db2401f5ac075a414048c12c1f7636bf6242283a4688113d70`
- Recovery qualification: `results/phase3/atomic-extraction-step35-model-qualification-v2`, manifest SHA-256 `e1670f586131e0c285a70f17f0acf510f3a56b58dbf12c95581a5aff52b95cd7`
- Phase 4 development input: `results/phase3/phase4-input-development-gpt41-fallback-v1`, manifest SHA-256 `f5127cfdb7720ecf84e320da613396d11d71629b709813e2e25ea2ab00df0876`
- Claims: `claims.jsonl`, SHA-256 `509c51229eb8a6e13e898a28fec594d0118c29b6f5a93fc917fb0d1af34b4ff7`
- Scores: `scores.json`, SHA-256 `76c96cc9590da0ac40d31a6ff5ab1d96e3decacb732a9bfc9b531e58e8750b4b`
- Evidence index: `evidence_index.json`, SHA-256 `311107940b64e22d9ba8af77e17d04e17f4298e8b5797e6f7234261e852ca0b6`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

### Next-step input

Phase 4 receives the immutable `claims.jsonl`, predicate registry v2, manifest, reverse evidence index, scorecard, and known-limitations file from `phase4-input-development-gpt41-fallback-v1`.

Step 4.1 has not started. It still requires separate approval.

## Phase 4, Step 4.1: Add storage and migrations

Status: complete

### Repository state

- Starting commit: `a768f6292d0c1404876ba1273ecbbd25829f9ed8`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-4.1-guidance-v1`, envelope SHA-256 `104d11223157154be066b61357cf7e75d62fbccc415f034722750a46cfed0af0`
- Commit message: `storage: add Phase 4 PostgreSQL schema`
- The worktree was clean after the final commit check.

### Implementation

- Added one checksum-bound SQL migration and a small advisory-lock migration runner. Reapplying the same migration is a no-op; changing an applied migration's bytes is an error.
- Added eight domain tables: `memory_users`, `source_events`, `source_spans`, `extraction_versions`, `processing_attempts`, `claims`, `claim_versions`, and `evidence_links`. The runner also owns `schema_migrations`.
- Added typed, frozen Python records and a parameter-bound repository. The repository inserts and reads records without committing the caller's transaction.
- Added composite `(user_id, id)` foreign keys for sources, spans, claims, versions, and evidence. Cross-user evidence cannot satisfy these keys.
- Added database checks for enums, confidence ranges, JSON shapes, stable IDs, exact span offsets, valid-time representation, transaction interval order, idempotency keys, and one open version per claim.
- Source and claim deletion uses `ON DELETE RESTRICT`. Step 4.2 still owns deletion propagation and reprocessing behaviour.
- The reviewer changed unknown valid-time membership so null boundaries do not match a date. Date and timestamp query representations can no longer cross. The reviewer also moved the Docker cleanup trap ahead of container startup.

### Schema and dependency pins

- Migration: `migrations/0001_phase4_storage.sql`, SHA-256 `6f8a84ce1f78adeccfd7ff15d830dbcf844c34159a0ed34ce11cea4313e95359`
- PostgreSQL image: `pgvector/pgvector:0.8.6-pg16@sha256:84a355869251af1a3379cfc9fa7b4dbf962c03f642a4bb7b339a203925071c43`
- Python driver: `psycopg[binary]==3.3.4`
- Requirements file SHA-256: `d375af9a0f805ccfd0fbf9a4943cfc6f44b4d7371e6a2376f6b8318d3c73e3d6`
- Compose file SHA-256: `c53c4d37a256e2203f32566a3196c246cd5b2bb67ad8e1839095c9c1c8b0f51a`
- PostgreSQL reported pgvector `0.8.6`. The schema enables the extension but has no embedding column, vector dimension, or vector index.

### Phase 3 handoff check

The PostgreSQL integration test loaded all 20 development source records and all 33 Phase 3 claims in one transaction. It preserved the canonical claim IDs, created 33 candidate versions and 33 evidence links, and left `memory_kind`, `sensitivity`, and `belief_confidence` null. The test then rolled back the transaction and confirmed that every loaded domain table was empty.

The weak Phase 3 scorecard remains unchanged: precision `0.303030`, recall `0.312500`, F1 `0.307692`, unsupported-memory rate `15/33`, and valid-time accuracy `0.000000`. Step 4.1 did not filter, promote, repair, or rescore a claim.

### Tests and contract checks

- `make test-storage PYTHON=.venv-storage/bin/python`: 16 tests passed, including seven tests against disposable PostgreSQL 16 with pgvector 0.8.6.
- `make validate-scaled-benchmark`: passed with the frozen `scaled_v1` counts and dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test`: 355 tests passed in 23.166 seconds. The seven Docker-only tests were skipped in this command and passed in `make test-storage`.
- Clean and repeated migration, checksum drift, typed round trips, stable-ID conflicts, duplicate idempotency, cross-user references, unknown references, invalid JSON, offset and enum checks, interval order, one-open-version uniqueness, and transaction rollback all passed.
- Valid time is inclusive. Transaction time includes its start and excludes its end. Unknown valid time does not imply current membership.
- The protected Phase 3 manifest, claims, evidence index, scores, failures, and predicate registry kept their recorded hashes.
- Runtime storage code reads no gold or oracle files and makes no model calls. Step 4.1 used zero OpenAI requests, tokens, and cost.
- The changed-file secret scan found no API key, private key, `.env`, or secret-shaped token.
- The disposable Docker container, network, and volumes were absent after the storage test.

### Limitations

- This step provides schema and record access only. It does not implement ingestion acknowledgement, retries, reprocessing, lifecycle transitions, `as_of` queries, conflict resolution, deletion propagation, retrieval, or embeddings.
- Lifecycle values beyond `candidate` are schema vocabulary for later steps. This step does not promote claims or decide what is current.
- The integration load is a rollback proof, not a retained development database or a performance measurement.
- Phase 3 extraction quality remains the limiting input quality and must stay visible in later temporal results.

### Next-step input

Step 4.2 receives the versioned SQL migration, typed storage records, checksum migration runner, transaction-scoped repository, pinned disposable PostgreSQL configuration, and the 20-source/33-claim rollback test.

Step 4.2 has not started. It still requires separate implementation and review.

## Phase 4, Step 4.2: Add idempotent ingestion and reprocessing

Status: complete

### Repository state

- Starting commit: `5e1d042360d4e084324f8ca258c601ef90dfd293`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-4.2-guidance-v1`, envelope SHA-256 `0fefea8634648673692bd65a39a9f71248bb6c302bd7ee8828e77360a34c8894`
- Commit message: `ingestion: add idempotent source reprocessing`
- The worktree was clean after the final commit check.

### Implementation

- Added migration `0002` for worker leases, retry state, claim-to-extraction records, transactional outbox events, and content-free source tombstones. Composite keys keep every source, claim, attempt, and evidence reference within one user.
- Added typed ingestion requests and results, deterministic attempt and outbox IDs, and short error codes that do not include source content.
- Added one-transaction source ingestion. An exact replay is a no-op. Changed content or a reused identity is rejected, while the same idempotency key remains valid for a different user.
- Added PostgreSQL `SKIP LOCKED` worker leasing, expired-lease recovery, retry classification, reprocessing under a new extraction version, and atomic claim, evidence, attempt, and outbox writes. Existing canonical claims are reused without overwriting them.
- Added evidence-aware source deletion. Claims with remaining evidence are retained and queued for recomputation. Claims with no evidence are removed. Repeated deletion is a no-op.
- Updated storage reads so runtime source, span, attempt, claim, version, extraction, outbox, and tombstone lookups require the owning user where applicable.
- Added unit and disposable PostgreSQL tests. During review, the concurrent duplicate-ingest case was made explicit and the result manifest was extended to bind the findings file.

### Tests and contract checks

- `make test-ingestion PYTHON=.venv-storage/bin/python`: 5 tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 tests passed, including 16 tests against disposable PostgreSQL 16 with pgvector 0.8.6.
- `make validate-scaled-benchmark`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test`: 369 tests passed in 24.214 seconds. The 16 database tests skipped in this command passed in `make test-storage`.
- Two concurrent copies of one ingest request produced one source, one attempt, and one outbox event.
- The development replay loaded 20 sources and 33 unchanged candidate claims twice. Counts stayed at 20 sources, 20 attempts, 33 claims, 33 candidate versions, 33 claim-extraction records, and 33 evidence links, with no duplicate derived records.
- Worker rollback, expired-lease recovery, non-retryable validation failures, same-extractor no-op, new-extractor reuse, out-of-order timestamps, cross-user access, sole-evidence deletion, shared-evidence recomputation, and repeat deletion passed against PostgreSQL.
- `git diff --check`, the unstaged and staged diff checks, the changed-file secret scan, all 22 manifest-bound hashes, and Docker cleanup passed.

### Protected inputs and costs

- Migration `0001` kept SHA-256 `6f8a84ce1f78adeccfd7ff15d830dbcf844c34159a0ed34ce11cea4313e95359`.
- `compose.yaml` and `requirements-storage.txt` kept SHA-256 `c53c4d37a256e2203f32566a3196c246cd5b2bb67ad8e1839095c9c1c8b0f51a` and `d375af9a0f805ccfd0fbf9a4943cfc6f44b4d7371e6a2376f6b8318d3c73e3d6`.
- The protected Phase 3 manifest, claims, evidence index, scores, failures, predicate registry, and scaled runtime inputs kept their recorded hashes.
- Step 4.2 loaded no gold or oracle data. It made zero OpenAI requests, used zero tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Migration `migrations/0002_ingestion_reprocessing.sql`: SHA-256 `52900567c16e67c654d6fb845b615043beb5d9bf68851eb58231c5740a41253b`.
- Result manifest `results/phase4/step4.2-ingestion-v1/manifest.json`: SHA-256 `5c5e27587642af9115e0b5454292afb4b211043ec641b57b97c91573f0796ed2`.
- Findings `results/phase4/step4.2-ingestion-v1/findings.md`: SHA-256 `097af6341fd97031ef27802f2bfe79263e60fb377926bb811e94ff07e141010d`.
- This step does not run an extractor, publish outbox events, update an index, resolve conflicts, assign lifecycle states, or answer temporal queries.
- The 33 development claims retain the weak Phase 3 scorecard. This step does not filter, repair, promote, or rescore them.

### Next-step input

Step 4.3 receives user-scoped source and claim records, immutable extraction mappings, retryable processing attempts, pending outbox events, content-free tombstones, and deterministic deletion-recompute events. Step 4.3 has not started and still requires a separate implementation and review.

## Phase 4, Step 4.3: Add temporal version transitions

Status: complete

### Repository state

- Starting commit: `c50103c878c2c5a1d98d287d352b6ffca3a61719`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-4.3-guidance-v1`, envelope SHA-256 `9a9687396f0a77f9ab3939e0202b89376290118493c62731a6dd6f0e80e2be16`
- Commit message: `temporal: add deterministic lifecycle queries`
- The worktree was clean after the final commit check.

### Implementation

- Added migration `0003` with valid-time snapshots on claim versions, non-overlapping transaction intervals, lifecycle transition audit rows, and `claim_lifecycle_changed` outbox events. The incremental migration keeps an existing version's status and confidence while copying its valid time from the claim.
- Added explicit lifecycle requests for the approved transition matrix. The caller supplies the target status, reason, idempotency key, confidence, and aware timestamp. The service does not infer a status or replacement.
- Each change closes one open transaction interval and inserts an immutable successor. Repeating the same request returns the saved result. Reusing its key with different inputs fails.
- Added atomic corrections that supersede one claim and promote an explicit same-user candidate replacement to `confirmed` or `current`.
- Added user-scoped bitemporal reads. Transaction intervals are start-inclusive and end-exclusive. Valid-time bounds are inclusive, and date values never mix with timestamp values.
- Queries hide versions and evidence recorded after the cutoff. Source deletion and tombstones dominate historical reconstruction, while claims with other visible evidence remain available.
- Null `memory_kind` development claims cannot be promoted. All 33 Phase 3 claims remain candidates.

### Review corrections

- Lifecycle audit foreign keys now prove that both version IDs belong to the audit row's user and claim. PostgreSQL rejects a version from another claim owned by the same user.
- The SQL and typed boundaries reject `candidate` as a transition target.
- Added an incremental `0002` to `0003` regression that preserves a previously promoted version and its valid-time snapshot.
- Updated the 33-claim rollback load to preserve each claim's valid-time snapshot on its candidate version.
- Moved shared valid-time validation and interval membership into one storage helper instead of keeping separate claim and version copies.
- Expanded the immutable result manifest to bind every implementation and test file, the findings, and every required protected input.

### Files changed

`Makefile`, `migrations/0003_temporal_lifecycle.sql`, `src/temporal`, the narrow storage and ingestion extensions, temporal unit and PostgreSQL integration tests, the Phase 4 storage regression, the Step 4.3 manifest and findings, and this ledger entry.

### Tests and contract checks

- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 tests passed against disposable PostgreSQL 16.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 storage and ingestion tests passed against disposable PostgreSQL 16.
- `make validate-scaled-benchmark`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test`: 384 tests were discovered in 24.909 seconds; 359 passed and 25 database tests skipped. Those 25 passed in the Docker targets.
- The transition matrix, terminal states, deterministic IDs, replay and drift handling, aware timestamps, valid-time boundaries, transaction boundaries, null time, and date/timestamp separation passed.
- PostgreSQL covered clean and repeated migration, incremental backfill, non-overlap, exact boundary queries, concurrent identical and competing requests, stale rollback, normal endings, corrections, cross-user rejection, late-source visibility, deletion dominance, and atomic outbox/audit writes.
- `git diff --check`, the staged and unstaged checks, the secret scan, all 26 manifest-bound hashes, and Docker cleanup passed.

### Protected inputs and costs

- Migrations `0001` and `0002`, `compose.yaml`, `requirements-storage.txt`, the Step 4.2 manifest and findings, the Phase 3 manifest, claims, evidence index, scores and failures, predicate registry v2, `preference.md`, and the roadmap kept their recorded hashes.
- Step 4.3 opened no benchmark gold, oracle data, or frozen test-user input. It made zero OpenAI requests, used zero tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Migration `migrations/0003_temporal_lifecycle.sql`: SHA-256 `2d4e888b262e3dab1a86464fa9de6d33d8818d8978004c5923a5a5e69226fc8e`.
- Result manifest `results/phase4/step4.3-temporal-lifecycle-v1/manifest.json`: SHA-256 `68a527421761dcdc960862f39f589c0283a274e062afe6ca86dc01bbce1c67ad`.
- Findings `results/phase4/step4.3-temporal-lifecycle-v1/findings.md`: SHA-256 `274b79a17d589474b70683aa77cbb2936ba0eb3d8fd5a8e4bd4c77887a6f9250`.
- This step does not classify conflicts, infer lifecycle status, repair extraction, add embeddings, or score temporal accuracy. Step 4.4 owns reviewed temporal cases and scoring.
- Deleting the last source evidence removes the unsupported claim and its lifecycle audit rows. The content-free tombstone remains.

### Next-step input

Step 4.4 receives the checksum-bound temporal migration, explicit lifecycle service, immutable transition audit, transactional outbox events, and user-scoped bitemporal query interface. Step 4.4 has not started and still requires a separate implementation and review.

## Phase 4, Step 4.4: Evaluate temporal behavior

Status: complete

### Repository state

- Starting commit: `21ebed2b1d06b623deee2b9b0f72ee45d0ff71d3`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-4.4-guidance-v1`, envelope SHA-256 `294fd04ec604e94f79fe1e276342c15bf1b515b1ecce5c865eee98e8c6d2cb50`
- Commit message: `eval: add temporal behavior scorecard`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added 12 reviewed development cases, six each for `user_001` and `user_002`. Every source reference resolves through the development-only loader, and every evidence quote and speaker matches its source observation.
- Kept runtime cases and gold expectations in separate files. The runner writes one prediction or sanitized failure for every runtime case before it hashes or opens gold.
- Covered correction visibility before, at, and after the transaction cutoff; repeated evidence; a normal change to historical status; approximate dates; time-zone normalization; out-of-order ingestion; separate valid periods; inclusive valid endpoints; shared-endpoint overlap; and half-open transaction boundaries.
- Added strict runtime, gold, prediction, failure, and scorecard records. Runtime imports have no path to scaled gold, oracle data, or review queues.
- Added deterministic event-ordering, date-normalization and precision, interval-relation, interval-IoU, current-state, historical-state, and correction-visibility scores. Every score records its denominator; a zero denominator returns null with a reason.
- Added a clean-database precondition, immutable result-directory checks, protected-input hashes, and byte-stable JSON output. The reviewer also corrected current, historical, and correction scoring to compare exact sets without depending on database row order.
- Added unit and disposable PostgreSQL tests, including two clean runs that compare all six emitted artifact files byte for byte and a regression that rejects an already populated database.

### Results

The evaluator produced 12 predictions and no failures. Event ordering, date normalization and precision, interval relation, current-state selection, historical-state selection, and correction visibility each scored `1.000000`.

Mean interval IoU was `0.027027` with a denominator of one. This is the exact inclusive overlap for the reviewed shared-endpoint pair: one overlapping day across a 37-day union. The small value is preserved as measured rather than treated as a failed or omitted case.

### Tests and contract checks

- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 tests passed against disposable PostgreSQL 16.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed against disposable PostgreSQL 16.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed against disposable PostgreSQL 16.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with the frozen `scaled_v1` counts and dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 404 tests were discovered in 24.843 seconds; 374 passed and 30 database tests skipped. Those 30 passed in the Docker targets above.
- The 12-case run repeated byte-identically from a second clean database. The existing 20-source, 33-candidate Phase 3 handoff also replayed twice without changing counts.
- Gold sequencing, exact source ownership, source order, cross-user rejection, time-zone equivalence, inclusive valid time, half-open transaction time, failure denominators, immutable output refusal, protected hashes, secret scanning, diff checks, and Docker cleanup passed.
- Migrations `0001` through `0003`, production storage, ingestion, extraction, and temporal code, the Step 4.3 release, the scaled release, the Phase 3 handoff, the predicate registry, `preference.md`, and the roadmap kept their recorded hashes.
- Step 4.4 made zero model requests, used zero tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Dataset manifest: `data/phase4/temporal-development-v1/manifest.json`, SHA-256 `785f17876b56ebdf29b8765e104e0160c19befae5271687869e4db9726034f8e`
- Runtime cases: `runtime/cases.jsonl`, SHA-256 `4c67e1a01f0513512f9c1c3d65bacf8a943f66d037c369182a24adae9b656416`
- Reviewed gold: `gold/cases.jsonl`, SHA-256 `1afce9b37f441925826b0511c8e32b004d825b29e2c8d614f10a5c7ca04ceefa`
- Result manifest: `results/phase4/step4.4-temporal-evaluation-v1/manifest.json`, SHA-256 `8f7cc49fbe5620094c618eaaaa98c27ce7a337fdb2747ca9918d8bcfe6d4d644`
- Predictions: `predictions.jsonl`, SHA-256 `641e2b1221f6b0123c7521b95b997fa7d4a321faf7aa49f4fc8ab1713726c8b1`
- Scores: `scores.json`, SHA-256 `23ce6b37522e1367d91e72959b3acfb1a6558597a2667f53c0da9bd970a53d9e`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Run metadata: `run.json`, SHA-256 `1613ed40d8e9a73c2263aa651400e2240fda9a3ca46174e76d903d49e44cb285`
- Findings: `findings.md`, SHA-256 `1317ef934f24f8b3f7eb08b49703bde0bc23855ae1cde4e419d1d223d047159f`
- This is a 12-case deterministic development evaluation, not a production workload or frozen-test result. Interval IoU has one case, so its mean is descriptive rather than broad evidence.
- The fixtures exercise explicit lifecycle commands. They do not infer lifecycle changes, classify conflicts, repair the weak Phase 3 claims, or modify the stored development claim set.

### Next-step input

Phase 5 receives the protected relational store and temporal query interface together with the immutable Step 4.4 runtime cases, reviewed gold, predictions, scorecard, manifest, and documented limitations. Step 5 has not started and still requires separate approval.

## Phase 5, Step 5.1: Generate conflict candidates

Status: complete

### Repository state

- Starting commit: `f0f658a356179f363f31a21bab146b446b144f72`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-5.1-guidance-v1`, envelope SHA-256 `e14a27ee4e7f5b2ddb58ca986aa61f4fbe523a974dbc0c0d1adddd1d5c3b92ec`
- Commit message: `conflicts: add deterministic candidate linking`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added eight reviewed development cases, four each for `user_001` and `user_002`. Every claim points to evidence owned by a protected Step 4.4 runtime claim. The runtime file contains no required pairs, relation labels, scaled gold, oracle data, review queues, or test users.
- Added a frozen candidate-linker configuration bound to predicate registry v2. It uses complete normalized string leaves, subject IDs, inclusive finite-time overlap or a gap of at most 90 days, and token Jaccard over the predicate, predicate family, and canonical object.
- Added deterministic candidate generation with the three approved rules. Every result involves an incoming claim, uses lexically ordered claim IDs, records all visible source IDs, and has a stable SHA-256 pair ID. The linker neither ranks candidates nor assigns a relation type.
- Added user-scoped PostgreSQL reads through the existing temporal service. Source visibility, transaction visibility, lifecycle eligibility, evidence support, and deletion effects are applied before features are computed.
- Added separate runtime and scorer-only gold loaders. The evaluator writes one prediction or sanitized failure for every case before it hashes or opens gold. It refuses a nonempty output path and produces byte-stable artifacts from clean database runs.
- During review, the pure generator was corrected to reject mixed-user claim or version inputs before feature computation instead of silently dropping them. A regression covers both ownership mismatches.
- During review, the immutable result manifest was expanded to bind all 42 protected inputs. The runner checks the 41 non-gold inputs before execution and defers the prior scorer-only gold hash until after this step's outcomes have been persisted and scored.

### Results

The evaluator produced eight predictions and no failures. It returned all eight reviewed required pairs from ten possible same-user pairs, for candidate recall `1.000000` and pair reduction `0.200000`. It generated no cross-user pair.

The signal counts were six same-subject pairs, six same-family pairs, eight shared-entity pairs, three temporal overlaps, two pairs within the 90-day gap, one approximate-time pair, and four pairs at or above the lexical threshold. These are descriptive counts, not precision estimates.

### Tests and contract checks

- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 tests passed, including six against disposable PostgreSQL 16.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected temporal-evaluation tests passed against disposable PostgreSQL 16.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed against disposable PostgreSQL 16.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed against disposable PostgreSQL 16.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 433 tests were discovered in 25.144 seconds; 397 passed and 36 database tests skipped. Those database paths passed in the Docker targets above.
- Exact lexical and 90-day thresholds, inclusive endpoints, mixed and unknown time, repeated evidence, source-ID completeness, user isolation before feature work, deletion visibility, failure denominators, gold sequencing, immutable output refusal, and two clean byte-identical runs passed.
- `git diff --check`, the staged and unstaged checks, the changed-file secret scan, all 42 protected hashes, all seven implementation hashes, all five artifact hashes, and Docker cleanup passed.

### Protected inputs and costs

- Migrations `0001` through `0003`, storage, ingestion and temporal services, the complete Step 4.4 dataset and result, the Step 4.2 and Step 4.3 handoffs, the Phase 3 claim input, the scaled runtime identity, predicate registry v2, `preference.md`, and the roadmap kept their recorded hashes.
- Step 5.1 made zero model requests, used zero tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Candidate configuration: `configs/conflicts/candidate_linker_v1.json`, SHA-256 `c21fe89467f64d31098f42940cb0b7a62d8a213bd93d7ac04d1d91f2280d9f7f`
- Dataset manifest: `data/conflicts/candidate-development-v1/manifest.json`, SHA-256 `9dd85b3bf49d04ebdd3e0f3a6ea05da4b01fff9901be0d5d9019272bc728f2bb`
- Runtime cases: `runtime/cases.jsonl`, SHA-256 `19aa0171b3ce655f06725e4c555a8e0a5d4b3c9c3c25bc9cc3d7de35e4368a33`
- Reviewed required-pair gold: `gold/required_pairs.jsonl`, SHA-256 `e22bff6a26c34ede3b57076d9bddf74849cbd7f62b3a39457ecf30976559c924`
- Result manifest: `results/conflicts/candidate-generation-development-v1/manifest.json`, SHA-256 `e089dd87b4361982988cd6df37a150e678f3b9245c1a14a007f90202de6b6c18`
- Predictions: `predictions.jsonl`, SHA-256 `2d8d0c790c3aa136735b2ac8bb1f2fcca5eeeb9732acca2af5b4dd5dd35870ae`
- Scores: `scores.json`, SHA-256 `57dc7c12e6e3665978862158fd8d592c86481479778d00546c6f742fe05214fa`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- This is an eight-pair deterministic development check, not a production workload or frozen-test result. Candidate recall measures whether required pairs survive linking. It does not estimate precision.
- This step proposes pairs only. It does not persist a relation, classify a conflict, resolve a belief, change lifecycle state, or use embeddings or a model.

### Next-step input

Step 5.2 receives the frozen candidate-linker configuration, `CandidateRequest`, `CandidatePair` and signal contracts, the deterministic development dataset, and the immutable candidate-generation scorecard. Step 5.2 has not started and still requires separate implementation and review.

## Phase 5, Step 5.2: Classify and persist checked relations

Status: complete

### Repository state

- Starting commit: `4780c85a05d6397d24abe15de96f6c3979867f33`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-5.2-guidance-v1`, envelope SHA-256 `212dd69b414fab36987c917cc1c6a05bf2828b8abf9c7fa657afb7fc0838767b`
- Commit message: `conflicts: classify and persist checked relations`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added a frozen deterministic classifier for eight labels: hard contradiction, temporal change, explicit correction, refinement, source disagreement, retraction, unresolved ambiguity, and unrelated. The precedence order is explicit, and every decision records the exact candidate pair, claim versions, evidence snapshot, rule version, and transaction cutoff.
- The classifier accepts only canonical Step 5.1 pairs for the same user. It checks lifecycle and transaction visibility, source ingestion time, exact evidence ownership, predicate registry v2 compatibility, object shape, valid-time representation, and structured correction targets before applying a rule.
- Added migration `0004` and typed repository records for conflict decisions, claim relations, and decision evidence. Composite foreign keys enforce user ownership. Stable IDs make exact replay a no-op and changed input a conflict, while one transaction prevents partial decisions, relations, or evidence.
- Storage accepts the full nine-relation ontology. The v1 classifier emits only `contradicts`, `corrects`, `refines`, and `same_topic_as`; it never emits `supersedes`. Symmetric stored relations use canonical claim order.
- Source deletion removes decisions, relations, and cited evidence before deleting spans. It schedules one content-free, ID-only recomputation event only when both claims still have evidence. It does not change lifecycle state.
- Added eight reviewed development cases from the exact Step 5.1 candidate output, four per development user. Runtime cases contain no relation expectations. The evaluator persists all predictions and failures, closes runtime resources, and only then hashes and opens the separate gold file.
- Added deterministic scoring for overall labels, supported-label precision, recall and F1, exact relation sets, relation direction, failures, unresolved cases, execution mode, and cross-user relations. Missing denominators are null and carry a reason.

### Review corrections

- Expanded the storage and SQL relation vocabulary from the four v1 outputs to all nine ontology relations. Added coverage for future relation types and canonical `same_event_as` storage while keeping classifier output restricted to v1's four relations.
- Bound classifier input to `candidate_linker_v1` so a pair produced under another linker version cannot be reinterpreted under this frozen rule set.
- Regenerated only the affected dataset and result manifest bindings. A clean-database evaluator run reproduced predictions, failures, scores, run metadata, and findings byte for byte.

### Results

The evaluator produced eight predictions and no failures. Overall label accuracy, macro F1 across the three represented labels, and exact relation-set accuracy were each `1.000000`. One of eight cases was unresolved, for an unresolved rate of `0.125000`. The release contains no directed gold relation, so direction accuracy is null with reason `no_directed_gold_relations`.

The represented labels are six unrelated cases, one temporal change, and one unresolved ambiguity. The other five labels remain explicitly not evaluated. This step made no model call or lifecycle decision.

### Tests and contract checks

- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 tests passed, including eight tests against disposable PostgreSQL 16.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 protected Step 5.1 tests passed, including six against disposable PostgreSQL 16.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected Step 4.4 tests passed against disposable PostgreSQL 16.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed against disposable PostgreSQL 16.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed against disposable PostgreSQL 16.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 462 tests were discovered in 24.613 seconds; 418 passed and 44 database tests skipped. Those database paths passed in the Docker targets above.
- Rule precedence, all eight labels, registry compatibility, inclusive valid time, half-open transaction time, unknown and mixed time, explicit target direction, exact evidence snapshots, cross-user rejection, stable IDs, replay and drift handling, rollback, deletion recomputation, gold sequencing, failure denominators, and immutable outputs passed.
- The Step 5.1 release replayed through the current migrations with the same predictions, failures, and scores. The protected Step 4.4 release also kept its recorded prediction, failure, score, run, findings, and manifest hashes.
- `git diff --check`, staged and unstaged checks, manifest self-verification, predecessor drift checks, the changed-file secret scan, and Docker cleanup passed.

### Protected inputs and costs

- The Step 5.1 manifest kept SHA-256 `e089dd87b4361982988cd6df37a150e678f3b9245c1a14a007f90202de6b6c18`. Its predictions, empty failures, and scores kept SHA-256 values `2d8d0c790c3aa136735b2ac8bb1f2fcca5eeeb9732acca2af5b4dd5dd35870ae`, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, and `57dc7c12e6e3665978862158fd8d592c86481479778d00546c6f742fe05214fa`.
- The protected Step 4.4 manifest, predictions, failures, scores, run, and findings kept their recorded hashes. Migrations `0001` through `0003`, the Step 4.3 and Step 4.2 releases, Phase 3 claims and evidence, scaled runtime identity, predicate registry v2, `preference.md`, and the roadmap also remained unchanged.
- Step 5.2 made zero OpenAI requests, used zero tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Classifier configuration: `configs/conflicts/relation_classifier_v1.json`, SHA-256 `fab3171a55ad21e25e303e89f44e57b7d0b926eb8f82283ce1dcb910d8960659`
- Migration: `migrations/0004_conflict_relations.sql`, SHA-256 `d48a4c3902b59f9fc83f3497cedc65c964de25c4edb3b1fd53a04e3afaa1e5c3`
- Dataset manifest: `data/conflicts/relation-development-v1/manifest.json`, SHA-256 `690694179f910a93fd536d082e7abd4c48cf76d9216959211189833af9f72d5a`
- Runtime cases: `runtime/cases.jsonl`, SHA-256 `9a298284bc25a155954be6e20e7807541f9638fbfd50640060878b7a73b5fe0e`
- Reviewed gold: `gold/cases.jsonl`, SHA-256 `0e245546cab7b85cffa83fa3ef05fe860ea6a230891718f36dfe44851fa42d74`
- Result manifest: `results/conflicts/relation-classification-development-v1/manifest.json`, SHA-256 `2f29edd580957192cc7808e2e4a454b7fa7c0b89bfc9ea3851efbf9e1c76179e`
- Predictions: `predictions.jsonl`, SHA-256 `f41a15fa83df9602c0536976aaebbc7aef9d7e33d1353f057dbeb52a6b3cf201`
- Scores: `scores.json`, SHA-256 `4efc0bbc243e9bca92898da6aeabf7d0693ef479b7cfeef39afb647c2ecd49bd`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- This is an eight-pair deterministic development check. Five labels and directed relation accuracy have no development denominator, so the perfect represented-label scores do not estimate production accuracy.
- This step records checked classification and relation facts only. It does not select a current belief, mutate lifecycle state, rank source authority, embed or retrieve memory, call a model, or implement Step 5.3.

### Next-step input

Step 5.3 receives canonical candidate pairs, immutable conflict decisions, full-vocabulary relation storage, exact evidence snapshots, user-scoped repository reads, and the deterministic Step 5.2 development scorecard. Step 5.3 has not started and still requires separate implementation and review.

## Phase 5, Step 5.3: Resolve temporal beliefs deterministically

Status: complete

### Repository state

- Starting commit: `56ce3125886af28625808b31c6b412aa62ae1c9f`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-5.3-guidance-v1`, envelope SHA-256 `8bbaf5c53358858027db6f5a6eda383c6c8cc47dae80f36a5228fe1b8ed6bea8`
- Commit message: `conflicts: resolve temporal beliefs deterministically`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added a frozen resolver configuration and a deterministic planner for every Step 5.2 label. The request carries only user, decision, time, idempotency, and resolver identifiers; the service loads the decision, claim versions, relations, authority inputs, and exact evidence from PostgreSQL.
- Exclusion runs before conflict policy. Restricted, hypothetical, wrong-subject, deleted, and unsupported claims cannot become current. Authority must come from an exact, current, source-backed official record with matching speaker, subject, predicate, spans, and time scope. Belief confidence remains null.
- Added migration `0005` and typed repository records for resolutions, ordered lifecycle actions, evidence lineage, and resolver provenance on new `supersedes` relations. Composite foreign keys retain user ownership, stable IDs support exact replay, and lifecycle, audit, relation, and outbox writes share one transaction.
- Added the single approved lifecycle edge, `candidate -> historical`, and a transaction-scoped temporal transition seam. No other Step 4.3 transition changed.
- Source deletion now invalidates directly affected resolutions, rewinds the complete resolver-owned suffix to its baseline, removes the deleted decision lineage, and replays eligible surviving decisions in stable order. Any rewind or replay failure rolls back the whole deletion.
- Added eight development cases derived only from the protected Step 5.2 runtime and predictions, four per development user. Predictions and sanitized failures are written and runtime resources are closed before the separate reviewed gold file is hashed or opened.
- During review, the resolver was corrected so an already terminal `superseded` or `excluded` replacement cannot be selected as current or recorded as an active replacement. A focused regression covers the selection, relation, and replacement-action fields.

### Results

The evaluator produced eight deterministic predictions and no failures. It recorded four no-change outcomes, two exclusions for non-user subjects, one temporal resolution that preserved the existing historical/current state without a redundant action, and one unresolved ambiguity that left both claims disputed.

Exact outcome, exact action, current selection, historical preservation, dispute handling, no-change handling, and evidence-trace coverage each scored `1.000000`. Supersession is not evaluated because the fixed input has no correction, refinement, or retraction case. All eight predictions were deterministic, no cross-user action was recorded, and belief confidence remained null.

### Tests and contract checks

- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 tests passed, including 13 against disposable PostgreSQL 16.
- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 protected Step 5.2 tests passed.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 protected Step 5.1 tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected Step 4.4 tests passed.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 508 tests were discovered in 25.487 seconds; 451 passed and 57 database tests skipped. Every skipped database group passed in the Docker targets above.
- The review covered all policy branches, exact relation direction, authority scope, terminal-state handling, null and mixed time, user and transaction visibility, deterministic IDs, exact replay and drift rejection, rollback, three deletion rewind/replay paths, gold sequencing, failure denominators, immutable outputs, and two clean byte-identical evaluator runs.
- `git diff --check`, staged and unstaged checks, the changed-file secret and leakage scans, 23 implementation hashes, 58 predecessor hashes with exactly 13 authorized changes, result-manifest self-verification, and Docker cleanup passed.

### Protected inputs and costs

- Migrations `0001` through `0004`, the Step 5.2 configuration, dataset and release, the Step 5.1 and Phase 4 releases, storage and temporal predecessor interfaces outside the authorized seams, Phase 3 claims and evidence, the scaled runtime identity, predicate registry v2, `preference.md`, and the roadmap kept their recorded hashes.
- Step 5.3 made zero OpenAI requests, used zero input and output tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Resolver configuration: `configs/conflicts/belief_resolver_v1.json`, SHA-256 `9cd5ac711da1ec11f0528852845f42e9da7f2050d088bd51ad7e12cf8e9356c6`
- Migration: `migrations/0005_belief_resolution.sql`, SHA-256 `641588e4a05a6c20bc5513fe1c0a41825ba9ed0379bf2de50a64c339732f59cb`
- Dataset manifest: `data/conflicts/belief-resolution-development-v1/manifest.json`, SHA-256 `0c0903606fe25793c9c76f4eb699f5a59c75f294075e4547bb69812e809768d7`
- Runtime cases: `runtime/cases.jsonl`, SHA-256 `f2d92261abd6a7bf49616068e7738ea4929cf8bcfaf0a85f1ec5252558421692`
- Reviewed gold: `gold/cases.jsonl`, SHA-256 `f76b7bafd0a1796a0a7474cc222384c3f81716c44d5b5db71e29472b9a659276`
- Result manifest: `results/conflicts/belief-resolution-development-v1/manifest.json`, SHA-256 `df01c8fbf9494e3f2eb0898e6f2c18aae3ee1cc57e8b2e3fe39ab0006e9f887d`
- Predictions: `predictions.jsonl`, SHA-256 `f9250461eac63627f564f8d811b4964e1912da439b5a5e055c5d062b243f5596`
- Scores: `scores.json`, SHA-256 `030f3db4a64be821b1e5d45a538e896bbf5ee04bab571536a836812a7c72ec29`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Run metadata: `run.json`, SHA-256 `c85ce29f65785f50eab16cd8ee0607c7668c7a3bc618d9bc86f30315474e752e`
- Findings: `findings.md`, SHA-256 `630d9fe87380fe271a231f9588542d9a0a7035791a52f1dcb90ef42cd86371fb`
- This is an eight-case deterministic development check. Five conflict labels and supersession have no development denominator, so the exact represented-case scores do not estimate production accuracy.
- Unsupported lifecycle transitions remain fail-closed. In particular, the resolver does not add `confirmed -> historical`; Step 5.3 changes only the approved `candidate -> historical` edge.

### Next-step input

Step 5.4 receives the frozen resolver policy, checksum-bound resolution schema, user-scoped atomic service, deletion rewind/replay semantics, and immutable belief-resolution scorecard. Step 5.4 has not started and still requires separate implementation and review.

## Phase 5, Step 5.4: Evaluate conflicts

Status: complete

### Repository state

- Starting commit: `5dd1eed3fd61a99eb1c234b08651b38fe9e34067`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-5.4-guidance-v1`, envelope SHA-256 `512dcbffc4d6b8dab30ba9c55045bb0b22eb8fea16942f9b9c50ea61896043e3`
- Commit message: `eval: freeze Phase 5 conflict scorecard`
- The worktree was clean after the final commit check.

### Dataset and evaluator

- Added one eight-case Phase 5 development view, with four cases for each development user. It binds only the protected Step 5.1 candidate runtime, Step 5.2 relation runtime, and Step 5.3 resolution runtime.
- Runtime loading checks the exact case order, users, pairs, decision snapshots, and cross-stage continuity without opening or hashing gold. The scorer opens only the three corresponding Step 5 gold files after every fresh prediction or sanitized failure has been written and the database and runtime resources are closed.
- The integration runner starts from a clean PostgreSQL database, applies migrations `0001` through `0005`, and runs the candidate linker, relation classifier, and belief resolver through their current production services. It does not load predecessor predictions. Every case stops at its first failed stage, and extra candidates or changed snapshots become visible failures.
- The scorecard reports each component separately. It includes candidate recall, conflict-pair precision, recall and F1, type accuracy, false contradiction rate, correction links, current and historical belief selection, superseded preservation, unresolved disputes, evidence lineage, failures, cross-user output, and execution mode. It has no composite score.
- The result path is immutable. A nonempty directory is rejected, and two clean database runs produced byte-identical artifacts.

### Review corrections

- Changed evidence coverage to require an exact source-ID set. The earlier subset check could have credited a prediction with extra provenance. A regression now proves that an unexpected source lowers coverage.
- Changed the two empty-denominator reasons to the frozen values `no_reviewed_correction_links` and `no_reviewed_superseded_claims`.
- Added explicit F1 accounting checks. The perfect fixed set is represented as numerator `4` over denominator `4`; adding one false positive changes it to `4/5`.

### Results

The fresh run produced eight deterministic predictions and no failures. Candidate recall was `8/8`. Conflict-pair precision, recall, and F1 were each `1.000000` over two reviewed positive pairs, and conflict-type accuracy was `2/2`. The false contradiction rate was `0/8`.

Current belief selection, historical preservation, unresolved-dispute handling, and exact evidence lineage each scored `1.000000`, with denominators `1`, `1`, `1`, and `8`. The dataset has no reviewed correction link or superseded claim, so those two metrics are null with their recorded reasons. The run produced no cross-user output, made no model call, and used no fallback.

### Tests and contract checks

- `make test-conflict-eval PYTHON=.venv-storage/bin/python`: 39 tests passed against disposable PostgreSQL 16 after the review corrections and artifact rebind.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 protected candidate tests passed.
- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 protected relation tests passed.
- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 protected resolver tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected temporal-evaluation tests passed.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 526 tests were discovered in 23.787 seconds; 464 passed and 62 database tests skipped. Every skipped database group passed in the sequential Docker targets above.
- Gold sequencing, fresh service execution, failure stop points, exact evidence lineage, F1 denominators, deletion and replay behavior, downstream temporal views, immutable output refusal, and two clean byte-identical runs passed.
- `git diff --check`, staged and unstaged inspection, result-manifest self-verification, the changed-file secret and leakage scans, all 68 effective protected hashes, and Docker cleanup passed. The Makefile test target was the only authorized predecessor-file change.

### Protected inputs and costs

- Production code, migrations `0001` through `0005`, configurations, the complete Step 5.1 through Step 5.3 datasets and releases, Phase 4 releases, Phase 3 claims and evidence, scaled runtime identity, predicate registry v2, `preference.md`, and the roadmap kept their recorded hashes.
- Step 5.4 made zero OpenAI requests, used zero input and output tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Dataset manifest: `data/conflicts/phase5-evaluation-development-v1/manifest.json`, SHA-256 `62be6e153e74e7263d14303dd09f09a2ca4820efa766ec44375c96487bccff77`
- Result manifest: `results/conflicts/phase5-conflict-evaluation-development-v1/manifest.json`, SHA-256 `35f37c3ef5f4a45739df200e64053a5d35552dec264336bb6bca3b701fb850ff`
- Predictions: `predictions.jsonl`, SHA-256 `0655f923dc99a592dec1731f7c04409304108def6d647c024bebecf53eb2021b`
- Scores: `scores.json`, SHA-256 `c1524091183d2fe5fb152523ae92f24600bdb4e743aecc1d3c9d76ffc186366c`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Run metadata: `run.json`, SHA-256 `43aacc222959492916499d43bc9110d61fd948de4bc7d4dd8e5d020e7654d880`
- Findings: `findings.md`, SHA-256 `e72c3643fe6ab7f9c702959f229951e0c45efe3ab16d55be59e0b6d16bfe6409`
- This is an eight-case deterministic development evaluation. It contains two conflict-positive pairs and no correction or supersession example, so the exact scores do not estimate production accuracy.
- The evaluator measures the frozen Phase 5 pipeline. It does not add conflict policy, change lifecycle state, call a model, or inspect frozen test users.

### Next-step input

Phase 6 receives the protected Phase 5 candidate, relation, resolution, and component scorecards, together with exact user-scoped provenance and current-belief state. Phase 6.1 has not started and still requires its own implementation and review.

## Phase 6, Step 6.1: Compute deterministic session boundaries

Status: complete

### Repository state

- Starting commit: `548b12750142eb07c8749f0a8f7834ba4c7b7c3b`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-6.1-guidance-v1`, envelope SHA-256 `bdbf7a891369104061b675a9c0754de573444e9eeca27972da6c98b6c863f09f`
- Commit message: `summary: add deterministic session boundaries`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added a frozen session-boundary configuration and typed contracts for user-scoped, transaction-cutoff reads. Sessions are computed from live source events; this step adds no migration or session table.
- Conversation and calendar sources form single-source sessions. Email uses `source_events.session_id`, then `metadata.thread_id`, and falls back to the source itself. Conflicting declared email keys fail without exposing either value, and subject text is never used.
- Chat uses the declared session ID before a metadata thread ID. Declared and unthreaded chats never mix. Unthreaded chats share a session when the gap is at most 1,800 seconds and split at 1,801 seconds.
- Stable session definition IDs bind the user, source type, boundary rule and internal key. Membership hashes bind the ordered sources, time range, user and as-of cutoff. Raw thread values are not written to output artifacts.
- PostgreSQL reads require a user and aware as-of time, apply `ingested_at <= as_of`, and order by produced time and source ID. Deletion is reflected on the next read, and earlier as-of views remain reproducible.
- Added a development loader that reads only the first 20 scaled source records and first two user records. The prefix parser stops before the next record. Runtime code does not load scaled gold, oracle data, review queues or test users.
- Added unit and live PostgreSQL coverage, an immutable development result, a focused Makefile target, and the narrow Step 5.4 replay adapter. The adapter permits exactly two predecessor drifts: the Makefile target and its own replay test.

### Results

The release contains 20 deterministic sessions from 20 development sources, split evenly between `user_001` and `user_002`. The type counts are eight conversation sessions and four each for email, chat and calendar. Source coverage is `20/20`; failures, invalid sessions, duplicate sources and cross-user outputs are all zero.

The frozen development sources happen to have unique declared thread IDs, so the release does not exercise multi-source grouping. The live fixtures cover the 1,800-second boundary, declared-key precedence, late ingestion, deletion, bridge recomputation, last-member removal and tombstones.

### Tests and contract checks

- `make test-sessionization PYTHON=.venv-storage/bin/python`: 33 tests passed against disposable PostgreSQL 16.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 protected candidate tests passed.
- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 protected relation tests passed.
- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 protected resolver tests passed.
- `make test-conflict-eval PYTHON=.venv-storage/bin/python`: 39 protected Phase 5 evaluator tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected temporal-evaluation tests passed.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 559 tests were discovered in 30.823 seconds; 492 passed and 67 database tests skipped. Every skipped database group passed in the sequential Docker targets above.
- Exact boundary inclusivity, declared-key precedence and conflicts, user and as-of filtering before grouping, stable IDs and hashes, deletion-derived recomputation, prefix isolation, immutable output refusal, and two clean byte-identical runs passed.
- `git diff --check`, staged and unstaged inspection, result self-verification, the changed-file secret and leakage scans, all 68 effective predecessor hashes, and Docker cleanup passed.

### Files, protected inputs and costs

- Added `configs/summaries/session_boundaries_v1.json`, `data/summaries/sessionization-development-v1/manifest.json`, the five files under `src/summaries`, three focused test files, and the six-file immutable result under `results/summaries/sessionization-development-v1`.
- Changed only `Makefile`, `tests/integration/test_phase5_conflict_evaluation.py`, and this ledger outside those new paths. Production migrations, storage, ingestion, temporal and conflict code remain unchanged.
- The Step 5.4 manifest kept SHA-256 `35f37c3ef5f4a45739df200e64053a5d35552dec264336bb6bca3b701fb850ff`. Its five non-manifest artifacts remained byte-identical, and its fresh replay still matched those artifacts under the current Makefile.
- Step 6.1 made zero OpenAI requests, used zero input and output tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Boundary configuration: `configs/summaries/session_boundaries_v1.json`, SHA-256 `d2af3cbfd35e24d1f0b3a10acd148fb46a7fc2b4cd96268182bd12dd8570f20b`
- Dataset manifest: `data/summaries/sessionization-development-v1/manifest.json`, SHA-256 `c139e2624cbec7904ca8edd67b4b1f42423dcfda8a47a1d8ca04cf8ee5507b69`
- Result manifest: `results/summaries/sessionization-development-v1/manifest.json`, SHA-256 `34611525b22cb9dcf8b5c9eb4affd2422d778b58b4b443c90913ce29c8f9365c`
- Predictions: `predictions.jsonl`, SHA-256 `2b7de7fc1b2d821867e8eeaa29182a1e22e15977387ce40e46301e2e56f86845`
- Scores: `scores.json`, SHA-256 `12fd9d01232fa5e1a418d699388c5bf333676d0827c477bcc85c4c5b66c02131`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- This step tests deterministic boundary mechanics and source coverage. It does not assess summary quality, create durative memories, persist sessions, call a model, or inspect frozen test users.

### Next-step input

Step 6.2 receives the frozen boundary configuration, typed session contracts, user-scoped repository, immutable 20-source development release, and deletion-aware as-of behavior. Step 6.2 has not started and still requires separate implementation and review.

## Phase 6, Step 6.2: Build grounded session summaries

Status: complete

### Repository state

- Starting commit: `97485bcc62ccc954c63fb1d8cd593d5f0d42533e`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-6.2-guidance-v1`, envelope SHA-256 `175f3a05789d633cefeab329816c38123de4f2226713a82154f10755bb123fd1`
- Commit message: `summary: add grounded session summaries`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added migration `0006` with user-owned summary definitions, immutable versions, complete source membership, and statement-to-claim-version-and-span evidence. Composite foreign keys enforce ownership, and transaction time is stored as a half-open interval.
- Source, span, and evidence deletion triggers physically purge every affected summary version before protected predecessor rows are removed. The coordinator also handles each existing source, claim, lifecycle, and recompute outbox event; it adds no new event type.
- Added a typed transactional repository with deterministic create, exact replay, drift rejection, successor, stale-input, concurrent first-write, and empty-summary behavior. It never mutates an older version.
- Rendering is structural and deterministic. It preserves status, time, attribution, and sensitivity qualifiers; omits restricted, hypothetical, and excluded claims; marks disputes as unresolved; and asks a question only for explicit uncertainty or dispute. Every statement must cite the exact same-user claim version and source span. No evidence is generated.
- The development loader binds the protected Step 6.1 sessions to the Phase 3 development claims and evidence. It reads no summary gold, oracle data, review queue, frozen test user, or model output.
- During review, exact-lineage validation was tightened. Statement claim IDs must now equal the claims represented by their evidence, and provenance scoring independently recomputes and verifies each span ID instead of accepting any span from the same source. Two focused regressions cover these cases.

### Results

The release evaluated 20 sessions. Seventeen produced persisted summaries and three had no eligible evidence and remained explicitly empty. The 17 summaries contain 34 statements and cover all 33 development claims with exact source, span, claim-version, and session-membership lineage. One uncertain claim produces a separate review question.

All structural coverage and consistency metrics scored `1.000000`; invalid, stale, cross-user, and unsupported-lineage counts were zero. The deletion check removed an affected summary before rebuilding the surviving sessions. Two clean PostgreSQL runs produced byte-identical artifacts.

### Tests and contract checks

- `make test-grounded-summaries PYTHON=.venv-storage/bin/python`: 45 tests passed against disposable PostgreSQL 16, including the two review regressions.
- `make test-sessionization PYTHON=.venv-storage/bin/python`: 33 protected Step 6.1 tests passed.
- `make test-conflict-eval PYTHON=.venv-storage/bin/python`: 39 protected Phase 5 evaluator tests passed.
- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 protected resolver tests passed.
- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 protected relation tests passed.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 protected candidate tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected temporal-evaluation tests passed.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 protected lifecycle tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 604 tests were discovered in 27.758 seconds; 528 passed and 76 database tests skipped. Every skipped database group passed in the sequential Docker targets above.
- Migration replay and checksum enforcement, four-table constraints, composite ownership, all-version deletion, exact replay and successor behavior, stale and concurrent writes, rendering rules, time normalization, source and claim lineage, immutable output refusal, and two clean byte-identical runs passed.
- `git diff --check`, staged and unstaged inspection, result-manifest self-verification, the changed-file secret and leakage scans, all 79 predecessor hashes with exactly seven authorized changes, and Docker cleanup passed.

### Files, protected inputs and costs

- Added `configs/summaries/session_summary_renderer_v1.json`, `migrations/0006_session_summaries.sql`, four implementation files under `src/summaries`, five focused test files, one dataset manifest, and the seven-file immutable release under `results/summaries/grounded-summary-development-v1`.
- Changed only `Makefile` and the six migration adapters named in the Step 6.2 contract outside those new paths. Core storage, ingestion, temporal, conflict, extraction, and sessionization implementations remain unchanged.
- The Step 6.1 and Phase 5 frozen releases kept their recorded manifest and artifact hashes. The adapter applies only migrations `0001` through `0005` when replaying Step 6.1, and its five non-manifest artifacts remain byte-identical.
- Step 6.2 made zero OpenAI requests, used zero input and output tokens, cost `$0`, and wrote to no hosted service.

### Artifacts and limitations

- Renderer configuration: `configs/summaries/session_summary_renderer_v1.json`, SHA-256 `1e15e3359c292095f7563de1f00f0d348d43030eebe34f68545b0f851e8797a5`
- Migration: `migrations/0006_session_summaries.sql`, SHA-256 `64181b9e87054bb4f206018a4576e31f197218762e8f684a95ab642df4bc0834`
- Dataset manifest: `data/summaries/grounded-summary-development-v1/manifest.json`, SHA-256 `4b4a48e62029a4e138f54dba4235b506eba8cc6236526ff7c2ede14243fa07b5`
- Result manifest: `results/summaries/grounded-summary-development-v1/manifest.json`, SHA-256 `ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761`
- Summaries: `summaries.jsonl`, SHA-256 `6ce2ddbdc3ade54154e4db0bd136ff6f12e1037ebf4b214d560a2247fb674b22`
- Empty sessions: `empty_sessions.jsonl`, SHA-256 `409d2d275cd4ea4e1f218b8b7fc3e6810bbaa886fb3c0128ce78b734ccf71a81`
- Checks: `checks.json`, SHA-256 `4d5b6a3f96ac2fa0f60ff9dd1177e51b44e6badd612cbb10d2ad8885aacfc7ea`
- Failures: `failures.jsonl`, empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Run metadata: `run.json`, SHA-256 `278c541668357b69d2a0884375537c69164bea65be76f9df5aec144cf83edf8e`
- Findings: `findings.md`, SHA-256 `4d535d9f1abad75c67734d7e5128f8ec734070f71bb744c674870cba0adf7eed`
- This development set contains only single-source sessions and candidate claims. The scorecard checks grounding, structure, persistence, and reproducibility; it does not rate editorial quality or estimate production summary quality.
- This step does not create durative memories, call a model, infer missing evidence, or implement Step 6.3.

### Next-step input

Step 6.3 receives the checksum-bound summary schema, deterministic renderer, exact lineage model, deletion-aware coordinator, and immutable grounded-summary development release. Step 6.3 has not started and still requires separate implementation and review.

## Phase 6, Step 6.3: Add deterministic durative claims

Status: complete

### Repository state

- Starting commit: `16f6c367ed7543aa05b3da00044f9fab1555dfb1`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-6.3-guidance-v1`, envelope SHA-256 `de50f5ce8955d85f617bfef15af7e39297704c4b6a056f32a467ad7473ce28e2`
- Commit message: `summary: add deterministic durative claims`
- The worktree was clean after the final commit check.

### Dataset and implementation

- Added a frozen rules file for interval predicates in the role, goal, preference, belief, relationship, and state families. Inference requires exact user, subject, predicate, polarity, and canonical JSON object identity. It does not paraphrase or merge predicates.
- The public request covers one user and transaction cutoff. Each atomic run stores every proposition decision for that user. Migration `0007` adds user-owned run, decision, and evidence tables with composite foreign keys and source-deletion cascades. Persisted evidence roles are exactly `supports` or `counter_evidence`; ignored inputs remain in the run snapshot and are not written as lineage.
- Eligible support must be episodic, visible at the transaction cutoff, asserted or corrected, and in a confirmed, current, or historical lifecycle state. One explicit closed interval is sufficient. Otherwise, support must span at least two sessions, sources, and episode times.
- Restricted, hypothetical, denied, disputed, unresolved, recursive, and otherwise ineligible evidence cannot create a durative claim. Incompatible values block only when their time overlaps or is unknown. A resolved temporal change across disjoint intervals remains historical context.
- Accepted results use the ordinary claim model with `memory_kind="durative"`, inferred epistemic status, candidate lifecycle, `speaker_id="memory_system"`, null belief confidence, minimum support confidence, highest eligible sensitivity, and exact span lineage. The extraction version records `model_version="deterministic"`.
- Claim identity is stable for the exact proposition. Additional evidence appends an immutable transaction-time successor. A changed valid interval creates a replacement claim instead of mutating the old claim. Replay, concurrent writes, source deletion, and survivor recomputation are transactional.
- The development loader binds only the frozen Step 6.2 release, Phase 3 development claims and evidence, session definitions, predicate registry, and rules file. It has no runtime path to summary gold, scaled gold, an oracle, a review queue, or frozen test users.

### Results

The release accounted for all 33 development claims. Seven use predicates outside the durative rules. The other 26 propositions were rejected as counterevidence because the imported claims still have null memory kind and candidate lifecycle. This is the conservative result required by the frozen handoff; no rule was weakened to create a positive example.

The database contains two user-level inference runs with 26 decisions, split 13 per user. The decisions contain 42 `counter_evidence` lineage rows and no ignored rows. Accepted claims, failures, duplicate decisions, stale references, cross-user references, unsupported decisions, model calls, retries, and cost are all zero. Decision accounting, input-claim accounting, and provenance integrity each scored `1.000000`; replay and deletion recompute checks passed.

During review, a wrong exclusion glob accidentally exposed frozen-test gold text to the reviewer. The reviewer stopped immediately. None of the exposed text was used in code, fixtures, expected values, metrics, or judgments. The implementation and release are derived only from the explicit development allowlist and bound manifests.

### Tests and contract checks

- `make test-durative-claims PYTHON=.venv-storage/bin/python`: 55 tests passed against disposable PostgreSQL 16. The first unpinned invocation stopped at import because the system Python lacked `psycopg`; no test body ran in that attempt.
- `make test-grounded-summaries PYTHON=.venv-storage/bin/python`: 45 protected Step 6.2 tests passed.
- `make test-sessionization PYTHON=.venv-storage/bin/python`: 33 protected sessionization tests passed.
- `make test-conflict-eval PYTHON=.venv-storage/bin/python`: 39 protected Phase 5 evaluator tests passed.
- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 protected resolver tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 protected temporal-evaluation tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 protected storage and ingestion tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 659 tests were discovered in 28.953 seconds; 564 passed and 95 database tests skipped. The guidance-required database groups passed in the sequential Docker targets above.
- Public run contracts, exact-proposition matching, family coverage, counterevidence precedence, complete date and timestamp boundaries, transaction visibility, stable claim identity, successor and replacement behavior, deterministic extraction metadata, exact lineage roles, replay, concurrency, deletion, immutable output refusal, and two clean byte-identical releases passed.
- `git diff --check`, compilation, result self-verification, the changed-file secret and leakage scans, the 91-file predecessor attestation with exactly nine authorized drifts, staged and unstaged inspection, and Docker cleanup passed.

### Files, protected inputs and costs

- Added the rules configuration, migration `0007`, four durative modules, four focused test files, the development manifest, and the seven-file immutable release under `results/summaries/durative-claim-development-v1`.
- Changed only the Makefile and the eight migration or replay adapters authorized by the Step 6.3 contract outside those new paths. Core storage, ingestion, temporal, conflict, sessionization, grounded-summary, and extraction production modules remain unchanged.
- The Step 6.2 manifest remains `ca38522d51e8568f49326789074d146dadcac3935687f21dc5fe0e937aba5761`. Migration `0006`, its four production modules, and all six Step 6.2 result artifacts kept their recorded hashes. The effective predecessor map covers 91 paths: 82 unchanged and exactly nine authorized drifts.
- Step 6.3 made zero OpenAI requests, used zero input and output tokens, cost `$0`, and wrote to no hosted service. Historical OpenAI spend remains `$0.2314404`.

### Artifacts and limitations

- Rules configuration: `configs/summaries/durative_claim_rules_v1.json`, SHA-256 `680de008e33de6824b8fded16f8e6fdc6877130d9a2caa35908be1c45c1a0b7b`
- Migration: `migrations/0007_durative_claims.sql`, SHA-256 `64e1fc2a9538d342b28003d5a1bf78b1532326fa5be0e6d8d448b8e94bc07325`
- Dataset manifest: `data/summaries/durative-claim-development-v1/manifest.json`, SHA-256 `51db96e51079303c8e6267c224e02ea117c4a9d810217b3310ccef31b4ac4d72`
- Result manifest: `results/summaries/durative-claim-development-v1/manifest.json`, SHA-256 `1d3f1c78d95bd96399224581bec21143c4b52562517d4779b74850e42d26fdbb`
- Rejections: `rejections.jsonl`, SHA-256 `7bbc228591b89e8049fbc42a3de8f2da060162498bbce7f631f0eadfb792ebc9`
- Claims and failures: both empty, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Checks: `checks.json`, SHA-256 `fe36f87449c02b5260f9be755c8406b64ca213d9375eb8ceb844f67f2d2de8a5`
- Run metadata: `run.json`, SHA-256 `863a1671a32de482112143cfdbb85f11eafa3570c49a41ce7ff9ecee1a9bbb02`
- Findings: `findings.md`, SHA-256 `85f0aca8d2fbf164dee91790f5c51b9b0ef63ce5c4637bcea6251087ef4c1bfc`
- The frozen development handoff cannot produce a positive durative claim because its claims are unresolved candidates with null memory kind. Positive inference, persistence, time, concurrency, and deletion behavior are covered by synthetic unit and live PostgreSQL fixtures instead.

### Next-step input

Step 6.4 receives the frozen rules, checksum-bound schema, user-run inference contract, exact support and counterevidence lineage, deletion-aware recompute behavior, and immutable development release. Step 6.4 has not started and still requires separate implementation and review.

## Phase 6, Step 6.4: Evaluate summary quality

Status: complete

### Repository state

- Starting commit: `d52b4a2a9a1178ab37354fc65fd1d202519185cc`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-6.4-guidance-v2`, envelope SHA-256 `3b5673d169a1ad58cffda7ff45bd83ffe0a541da9ed38e870be99503842955b9`
- Commit message: `eval: add Phase 6 summary scorecard`
- The worktree was clean after the final commit check.

### Evaluation boundary and sequencing

- The first Step 6.4 attempt under guidance v1 was invalid and never committed. Cached-diff review found a trailing blank line in the runtime module after development gold had already been opened. Even though the change was only whitespace, changing the frozen runtime byte invalidated that checkpoint.
- Guidance v2 froze the corrected runtime module at SHA-256 `5b8b991d7fe5ea6a723fe7ca18929351ade8559b6e3082aa38c43afc228ea093`. The v2 runtime contains no scorer, gold, oracle, review, reference-summary, or test-user import or path.
- The runtime checkpoint was generated for ten development cases while all v1 and v2 scorer configurations, scorer code, scorer tests, gold files, event maps, and final result paths were absent. The preflight records that state, the two byte-identical runtime trials, five cases per user, and zero model use. The complete checkpoint tree was frozen before any scorer file returned.
- The previously authorized v1 development-only gold was then copied byte-for-byte from the private backup into the v2 paths. Gold cases, claims, and event map retained SHA-256 values `66271b5cd113a126f3ed5c339a599e5ff35d342fa2be233bdeea39310174e8d6`, `88361358e7972753655a38107e3a8fcef2131a3e8b9e98000e771f808ef7001f`, and `83393611342afde3bd8606a78f377c2f41969bf2e41c18a4a5493b2029052c17`.
- This was not a blind evaluation: the implementing agent had already seen the authorized development gold during the invalid v1 attempt. The v2 scorer and mapping were carried forward without reinterpretation or threshold tuning. No frozen test-user, oracle, or review-queue content was opened, copied, or scored.

### Results

The release has one prediction and no failure for each of the ten cases belonging to `user_001` and `user_002`. The all-visible-session baseline contains 165 factual statements and five unresolved questions. The reviewed gold contains 22 events, 31 evidence instances, two correction events, and nine uncertainty events.

Exact Claim-and-evidence matching found no event match. Gold-event micro precision is `0/165`, recall is `0/22`, and F1 is `0.000000`; macro precision, recall, and F1 are each `0.000000` over ten cases. Supporting-evidence micro precision is `0/165`, recall is `0/31`, and F1 is `0.000000`. Current-versus-historical accuracy is `null` because there are no matched reviewed-state events. Correction preservation is `0/2`, and uncertainty preservation is `0/9`.

Case accounting is `10/10`, and exact statement-to-Claim-version-to-span provenance coverage is `170/170`. Cross-user predictions, stale references, unsupported statements, runtime failures, and sanitized failures are all zero. The result publishes no composite score.

These low scores are an honest outcome of the frozen baseline. It returns every visible session summary instead of retrieving by instruction, and the upstream weak extraction does not exactly match the reviewed gold Claims and evidence. Step 6.4 did not change production behavior or tune to the exposed development gold.

### Tests and contract checks

- Focused Step 6.4 unit and integration tests: 30 tests passed, including guarded runtime reads, exact checkpoint bytes, invalid-v1 rejection, carried-gold identity, one-to-one scoring, metric denominators, and two deterministic scorer runs.
- `make test-durative-claims PYTHON=.venv-storage/bin/python`: 55 tests passed against disposable PostgreSQL 16.
- `make test-grounded-summaries PYTHON=.venv-storage/bin/python`: 45 tests passed.
- `make test-sessionization PYTHON=.venv-storage/bin/python`: 33 tests passed.
- `make test-conflict-eval PYTHON=.venv-storage/bin/python`: 39 tests passed.
- `make test-belief-resolution PYTHON=.venv-storage/bin/python`: 46 tests passed.
- `make test-conflict-relations PYTHON=.venv-storage/bin/python`: 29 tests passed.
- `make test-conflict-candidates PYTHON=.venv-storage/bin/python`: 29 tests passed.
- `make test-temporal-eval PYTHON=.venv-storage/bin/python`: 20 tests passed.
- `make test-temporal PYTHON=.venv-storage/bin/python`: 15 tests passed.
- `make test-storage PYTHON=.venv-storage/bin/python`: 30 tests passed.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 689 tests were discovered in 26.246 seconds; 594 passed and 95 database tests skipped. Every guidance-required database group passed in the sequential Docker targets above.
- Compilation, `git diff --check`, staged and unstaged inspection, secret and leakage scans, manifest self-verification, exact checkpoint/final prediction and failure identity, the 103-path predecessor check with zero drift, and Docker cleanup passed.

### Artifacts and costs

- Scorer configuration: `configs/summaries/summary_quality_scorer_v2.json`, SHA-256 `28b02f194f2cfafb4c2ec82e876bfae6797a9f588465f7fcc3564da6d58bddf1`
- Runtime cases: `data/summaries/summary-quality-development-v2/runtime/cases.jsonl`, SHA-256 `3ec24d5abe216b125f088307822bd4e48a3b82f9a21f75cf91e3110640797b33`
- Runtime input manifest: `data/summaries/summary-quality-development-v2/runtime/manifest.json`, SHA-256 `4feba4d18e7ec6ad8932a683d58114d55a138a0b1f4a90d61547a8a0d9f7ac8f`
- Dataset manifest: `data/summaries/summary-quality-development-v2/manifest.json`, SHA-256 `d58646b3df446c467fabc0bf41099f80fc0a045f54a41c361ee92013702fbc17`
- Runtime preflight: `results/summaries/summary-quality-development-runtime-v2/checkpoint_preflight.json`, SHA-256 `4a2b8947c10259199ab2ca122f5d86ab8ce225f3819ec4469f98f723f5c23dd7`
- Runtime checkpoint manifest: `checkpoint_manifest.json`, SHA-256 `6b0a474e42962ac792516c0ed024fa78d9d52842974cf8e8f221f0297e04500e`
- Runtime and final predictions: SHA-256 `4bcd6c0c821936465f927e2f3196fcaccf108b997a24c45cc24191602942813a`
- Runtime and final failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Final result manifest: `results/summaries/summary-quality-development-v2/manifest.json`, SHA-256 `b9ed38d2bf68afc0532adda7a1b229cc989548eaabd1087e597b8787819b4385`
- Scores: `scores.json`, SHA-256 `c5b1590be212388b1daf5a81065e1ec6c972402b7d348b0ab47f08c2a087658c`
- Run metadata: `run.json`, SHA-256 `9ffd729551312accdfa40112b625c72ee24a5e51e41f5291e6d8710d1ec13d6d`
- Findings: `findings.md`, SHA-256 `26312cf69531aa36b5aee405feba88230fbf97ee83ded3bae16a8a23429a5bde`
- Step 6.4 made zero requests or retries, used zero input and output tokens, cost `$0`, and called no provider. Historical OpenAI spend remains `$0.2314404`.

### Next-step input

Step 7.1 receives the frozen session boundaries, grounded-summary and durative contracts, immutable durative and summary-quality scorecards, exact statement provenance, and versioned session-index input. Step 7.1 has not started and requires separate guidance and authorization.

## Phase 7, Step 7.1: Build atomic and session indexes

Status: complete

### Repository state

- Starting commit: `9e72921e930a336c8a3ef280165f5715cac0019a`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-7.1-guidance-v1`, envelope SHA-256 `41a3f67caa7a41f9039f24a3d78eff26c1e45568eda64a9be3b3753ee8c33418`
- Commit message: `retrieval: add atomic and session indexes`

### Schema and implementation

- Migration `0008` adds user-owned index runs, atomic and session records, and exact claim-version, source-span, and checked-relation lineage. Composite foreign keys enforce ownership. Separate partial GIN and HNSW indexes cover atomic and session records, while B-tree filters begin with user, index version, and record kind.
- Atomic records preserve the frozen Claim and ClaimVersion fields, lifecycle, valid and transaction time, sensitivity, evidence, and checked relation context. Session records reuse the existing summary text, ordered statements, unresolved questions, lifecycle set, sensitivity flag, and exact statement lineage. Restricted records are excluded; candidate, current, historical, disputed, and superseded records keep their original status.
- `deterministic_token_hash_v1` produces 256-dimensional, L2-normalized vectors with Unicode normalization and SHA-256 bucket and sign selection. It uses only the Python standard library, has no random or network path, and is explicitly a development storage vector rather than a semantic embedding model.
- Per-user builds use an advisory lock and one transaction. Exact replay is a no-op; an idempotency-key drift, ownership mismatch, stale lineage, or partial write fails closed. Source, span, relation, evidence, and summary deletion remove affected index rows before any stale content can remain, and survivors can be rebuilt under the same contract.

### Review corrections

- The development loader originally hashed the complete mixed runtime user and source files. Review replaced those bindings with the frozen two-user and 20-source prefix hashes from the sessionization manifest and added a read trap that fails on record 3 or source 21.
- Checked relation lineage originally lacked the build cutoff. It now requires `created_at <= transaction_as_of`; a live regression covers both the exact boundary and a future relation.
- A pre-commit release attempt used the 126-file protection map inside runtime generation and therefore hashed three prohibited Step 6.4 gold files. That release was rejected and never committed. The corrected runtime hashes only four approved authority manifests: the Step 6.3 result manifest, Step 6.4 result manifest, checkpoint manifest, and checkpoint preflight. A full-execution read trap rejects gold, scorer, evaluator, oracle, review-queue, and test-user paths.
- The final release states that its 126-path result is a `git_and_reviewer_gate` with `runtime_verified=false`. Reviewer-side recomputation found exactly the nine authorized compatibility changes and 117 unchanged protected paths. The private pre-correction backup remains outside the repository.

### Results

The final development build contains 33 atomic records and 17 session records for `user_001` and `user_002`, with no durative records. It created two successful user-scoped runs and 50 total records. Claim and source lineage each contain 66 rows; there are no checked relation rows because the frozen development handoff contains none.

Failures, duplicates, cross-user references, stale references, unsupported records, restricted records, partial writes, model calls, retries, tokens, and incremental cost are all zero. All 50 records have 256-dimensional vectors and full-text documents. Exact lineage, replay, deletion, and deterministic-release checks passed. The five payload artifacts remained byte-identical across the leakage correction; only the release manifest changed to record the narrower runtime authority contract.

### Tests and contract checks

- `make test-retrieval-index PYTHON=.venv-storage/bin/python`: 40 tests passed against disposable PostgreSQL 16, including the prefix trap, relation cutoff, full runtime-read trap, migration, index, replay, concurrency, rollback, deletion, and two-clean-build checks.
- Protected live gates passed: ingestion 5, storage 30, temporal lifecycle 15, temporal evaluation 20, conflict candidates 29, conflict relations 29, belief resolution 46, Phase 5 evaluation 39, sessionization 33, grounded summaries 45, durative claims 55, and Step 6.4 summary quality 30 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 729 tests were discovered in 30.385 seconds; 621 passed and 108 database tests skipped. The skipped database groups passed in the sequential live gates above.
- Release self-verification, in-memory compilation, `git diff --check`, staged and unstaged inspection, secret and leakage scans, out-of-band 126-path protection with exactly nine authorized changes, payload-byte comparison, and Docker cleanup passed.

### Artifacts, costs and limitations

- Index configuration: `configs/retrieval/index_v1.json`, SHA-256 `4579b9fe671985a35f605ad9f0dcf256d4672b95329e597d0876754bc5c84a48`
- Migration: `migrations/0008_retrieval_indexes.sql`, SHA-256 `57608bce946af98cd88c8ecb1741e4d2ce32520f7894bf81b9beaf103b639156`
- Dataset manifest: `data/retrieval/index-development-v1/manifest.json`, SHA-256 `b9c1afd7490d78d25d9bd37d34abb0b90b739908f6ab3da89f9e3b7da343f8fc`
- Result manifest: `results/retrieval/index-development-v1/manifest.json`, SHA-256 `5854d9389224128549df992a9fd7f60e857333ed273adb946b1c1c8c58dea0c6`
- Records: `records.jsonl`, SHA-256 `e29531cd3bf72419c947a31b13c0b4adbd008f019dfbfc3e6a820ad541e23cfe`
- Checks: `checks.json`, SHA-256 `513c869d0ed4c1247be650a065ac88ad1b4b8256f9c3c9c029d53aa70f59441d`
- Run metadata: `run.json`, SHA-256 `802c46fc63cefcb5d2e99a9a27f6ebf9b6173e4519da0bf4433fa17286168987`
- Findings: `findings.md`, SHA-256 `11a712bbd98e9340a3c0a06840ade6181c0ca6fd4acc8794b2b9950e2f4a2980`
- Failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Step 7.1 made zero provider requests, used zero tokens, and cost `$0`. Historical OpenAI spend remains `$0.2314404`.
- The development handoff contains only candidate atomic claims and no accepted durative claim. The token-hash vectors prove index mechanics and reproducibility; they do not establish semantic retrieval quality, relevance, ranking, latency, or production embedding quality.

### Next-step input

Step 7.2 receives the frozen index schema, deterministic renderer and embedder contract, exact record lineage, deletion-aware rebuild behavior, and immutable 50-record development release. Query classification, filters, ranking, fusion, retrieval scoring, and answers have not started and require separate guidance and authorization.

## Phase 7, Step 7.2: Add query planning and pre-search filters

Status: complete

### Repository state

- Starting commit: `cd8fc5856b682584aa979623b500fabaf8d5f902`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-7.2-guidance-v1`, envelope SHA-256 `d698cc1b53460bc6422fa9d25fb9f47a398425e7848cdf18f93a71e2e9103e70`
- Compatibility ruling: `step-7.2-guidance-v1-compatibility-ruling-1`, envelope SHA-256 `bb83c54b37ff2ff14c7e6655fded7a696de79e19c4dfa4b0f63086cd1dbc1e4b`
- Commit message: `retrieval: add deterministic query planning and filters`

### Implementation

- The request and plan contracts require an aware cutoff, a frozen index version, sorted record-kind, speaker, and subject filters, typed valid time, and explicit sensitivity permissions. Unknown fields, duplicate filters, mixed or reversed time, naive timestamps, and version drift fail before repository work.
- The planner classifies eight query types with frozen NFKC and casefolded phrase rules. Evidence requests take precedence over the underlying factual type, while change and correction intent takes precedence over simple current or historical wording. Structured time remains authoritative; unstructured time is flagged for clarification instead of parsed.
- The query repository chooses the latest successful user-owned index run at or before the request cutoff. It then applies transaction, source-ingestion, relation, inclusive valid-time, speaker, subject, lifecycle, and sensitivity checks. Cross-user rows never enter the result or its rejection counts. Restricted rows cannot be authorized, sensitive rows require permission, and null sensitivity requires a separate audit opt-in.
- The repository returns stable record-ID order for reproducibility only. It does not execute full-text or vector search, compute an embedding, score or rank a record, choose `k`, fuse results, rerank, or answer a question.
- Compatibility ruling 1 updates only the frozen Makefile hash in `tests/integration/test_phase5_conflict_evaluation.py`. The final Makefile SHA-256 is `347eab60fd4d62d3764bb4315f1831cc024c3696bd37524de8ffd8106252c6e1`; the ruled adapter SHA-256 is `580fd9b546996641a397f9ea1f57980c44e41066ceb92f91f0fea50f3d734a8f`. The Step 5.3 and Phase 5 manifests and payloads remain unchanged.

### Development release

The release contains 24 synthetic development requests, split evenly across `user_001` and `user_002`, with three requests for each query label. Runtime planning and filtering completed before the scorer opened the separate reviewed reference. The runtime checkpoint records that no reference, gold, oracle, review-queue, test-user, search, ranking, or model path ran.

All 24 labels, 24 plan expectations, and 24 eligibility expectations matched. Failures, cross-user rows, restricted rows, post-cutoff rows, duplicate decisions, stale rows, unsupported rows, model calls, retries, tokens, and incremental cost were all zero. Two clean PostgreSQL runs produced byte-identical runtime artifacts, and two scorer runs over the checkpoint produced byte-identical releases.

### Tests and review gates

- `make test-retrieval-planning PYTHON=.venv-storage/bin/python`: 81 tests passed, including the frozen Step 7.1 index tests and 41 new unit and PostgreSQL integration tests.
- Protected live gates passed: ingestion 5, storage 30, temporal lifecycle 15, temporal evaluation 20, conflict candidates 29, conflict relations 29, belief resolution 46, Phase 5 evaluation 39, sessionization 33, grounded summaries 45, durative claims 55, and Step 6.4 summary quality 30 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 770 tests were discovered in 32.473 seconds; 650 passed and 120 database tests skipped. The skipped database groups passed in the sequential live gates above.
- Independent release recomputation, in-memory compilation, manifest self-verification, `git diff --check`, staged and unstaged inspection, secret and leakage scans, Docker cleanup, and the no-search SQL spy passed.
- The out-of-band 126-path predecessor replay found the same nine authorized compatibility paths and 117 unchanged paths. The Step 7.1 result manifest remains `5854d9389224128549df992a9fd7f60e857333ed273adb946b1c1c8c58dea0c6`, and its 50-record payload remains byte-exact. Before the ledger entry, tracked Step 7.2 drift was limited to `Makefile`, `src/retrieval/__init__.py`, and the ruled Phase 5 adapter.

### Artifacts, costs, and limitations

- Planner configuration: `configs/retrieval/query_planner_v1.json`, SHA-256 `538af5ceb41f50c752dc086c9f6ef39ee6b42b4ec0616948b3ac38192f66c654`
- Dataset manifest: `data/retrieval/query-planning-development-v1/manifest.json`, SHA-256 `c93af3341693425611e75749d962d12fb185bb3ee9788d0eb906ad03bcc4f260`
- Runtime requests: `requests.jsonl`, SHA-256 `9763e0a723a00e9dce7ec2f031ba9863983c06b7a20fe42f97019f1e229a2b30`
- Reviewed reference: `reference.jsonl`, SHA-256 `5c2f6f9a2aa21f18a3050cf48d4d8954377ee1f58d013813481da211db1be36b`
- Runtime checkpoint: `runtime-checkpoint.json`, SHA-256 `2f7986346623f7af93115c1f74ea4640447975317f93a399cdfe155c372279eb`
- Predictions: `predictions.jsonl`, SHA-256 `b925bc80a1cb1e37832d096adc0256de8f7f42c85be275f6b316c164743080b9`
- Filter decisions: `filter-decisions.jsonl`, SHA-256 `1167c707272c308b8dca6b29aecd120d7796f36512210000b618f8a0fe5d07bc`
- Checks: `checks.json`, SHA-256 `c71ddba046d8ae6a1f7bc5d46179e4651d67a296dd37cc2c2df0010e08165cd1`
- Run metadata: `run.json`, SHA-256 `6a6f1aa3de6adc5162eee154189cddef99ac7f9c2ca86b8f991e4f593418d2fb`
- Findings: `findings.md`, SHA-256 `b2e0cb66b82f8eb463b432914793a25fbead6823877a02f53851761eb30bcb7a`
- Failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Result manifest: `results/retrieval/query-planning-development-v1/manifest.json`, SHA-256 `007b5c14c7e74a87c717150a5ab0ee454e8ed1e61dd493c014705cae5dd7383f`
- Step 7.2 made zero provider requests, used zero tokens, and cost `$0`. Historical OpenAI spend remains `$0.2314404`.
- This is a structural planner and filter release, not a retrieval-quality result. It does not reconstruct an older aggregate index snapshot when no safe snapshot exists. The frozen index remains candidate-heavy, and its deterministic token-hash vectors do not establish semantic retrieval quality.

### Next-step input

Step 7.3 receives the frozen request, plan, eligibility-result, planner configuration, repository boundary, and immutable Step 7.2 release. Search, ranking, fusion, reranking, connected-history expansion, retrieval metrics, evidence packages, and answers have not started and require separate guidance and authorization.

## Phase 7, Step 7.3: Add B2-B4 search and fusion

Status: complete

### Repository state

- Starting commit: `701a23b1af2749f83ba782206b1d29cffc83dc5d`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-7.3-guidance-v1`, envelope SHA-256 `4f9b8a1f9671eaa722ae71134d9714300d8a7302b952a8d896cee5f7cc060597`
- Commit message: `retrieval: add B2-B4 search and fusion`

### Implementation

- B2 searches atomic records, B3 searches session records, and B4 searches both. Each run rebinds the frozen Step 7.2 eligibility result by user, index version, successful snapshot run, record kind, and record ID before search.
- Lexical and vector queries start from a materialized, user-owned eligible set. They use PostgreSQL full-text search and exact pgvector cosine ordering with a pool of 40 records per kind. An empty text-search query or zero vector skips that channel instead of creating a match.
- Search, expansion, metadata loading, and result assembly run in one read-only repeatable-read transaction. Missing records, changed eligibility fields, future source or relation lineage, unrelated source lineage, non-finite scores, and incomplete provenance fail the whole execution.
- Fusion uses unweighted reciprocal rank fusion with `1 / (60 + rank)`. Scores are stored as fixed 12-place decimal strings. The reranker changes only equal-score order using the frozen query-label kind and lifecycle preferences, followed by stable record ID.
- Change queries can add eligible versions of the same claim and one-hop checked relation neighbours. Expansion stays inside the same user, index version, snapshot, and eligibility result. Symmetric relation direction is kept in the trace. The first ten atomic seeds are chosen after filtering out session results.
- Accepted items contain record anchors, component scores, RRF contributions, lifecycle state, claim-version lineage, source-span lineage, and expansion paths. Rejected items retain the Step 7.2 pre-filter reasons or record `outside_top_k` and `no_channel_match`. No raw source text is copied into the result.

### Review corrections

- Source-lineage validation originally allowed extra claim-version pairs that were not part of the indexed record's claim lineage. It now requires exact equality and has a PostgreSQL regression proving the corrupt record fails closed.
- Symmetric checked relations were accepted from storage but rewritten as incoming or outgoing in the result trace. The contract and repository now preserve `symmetric`, with unit and live coverage.
- B4 originally limited the combined atomic and session ranking to ten before selecting atomic expansion seeds. It now filters to atomic records first and then takes ten, as required by the expansion contract.
- The corrected release kept `results.jsonl`, `checks.json`, `failures.jsonl`, `run.json`, and `findings.md` byte-identical. Only the implementation-bound runtime checkpoint and manifest changed.

### Development release

The release contains eight handcrafted runtime queries, one per Step 7.2 query label and four per development user. Each query ran as B2, B3, and B4, producing 24 results: eight per baseline. B2 accepted only atomic records, B3 accepted only session records, and B4 used the atomic and session lexical and vector channels.

The 24 runs accepted 204 records, recorded 44 pre-filter and 152 post-rank rejections, and produced no failures. All accepted ranks are contiguous and at most ten. Cross-user, restricted, post-cutoff, stale, unsupported, duplicate, and partial-lineage counts are zero. The frozen development index has no checked relation links or multi-version claim chain, so its expansion count is zero; live fixtures cover both paths.

This is a runtime-only release. It created, opened, and hashed no relevance file and computed no retrieval-quality or timing metric. Provider requests, retries, tokens, and incremental cost are zero. Historical OpenAI spend remains `$0.2314404`.

### Tests and contract checks

- `make test-retrieval-baselines PYTHON=.venv-storage/bin/python`: 132 tests passed against disposable PostgreSQL 16. This includes the frozen index and planning suites, B2-B4 search, tie ordering, expansion, deletion, repeatable-read consistency, read traps, and two clean byte-identical releases.
- Protected live gates passed: conflict evaluation 39, belief resolution 46, conflict relations 29, conflict candidates 29, durative claims 55, grounded summaries 45, sessionization 33, temporal evaluation 20, temporal lifecycle 15, and storage with ingestion 30 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 821 tests were discovered in 29.409 seconds; 688 passed and 133 database tests skipped. Every required database group passed in the sequential live gates above.
- Release self-verification, compilation, `git diff --check`, staged and unstaged inspection, secret and leakage scans, exact three-path predecessor drift, all 15 protected hashes, and Docker cleanup passed.

### Artifacts, costs, and limitations

- Ranking configuration: `configs/retrieval/baseline_v1.json`, SHA-256 `6d49b6d9302b32eb5446ceeb36eb14642091a73ff264c4d7cfeeb4569858cea1`
- Dataset manifest: `data/retrieval/baseline-execution-development-v1/manifest.json`, SHA-256 `d32915d803cb1d2dcaeaf4f0269b1da33b3f9111a58222c4027b9a5e95446b48`
- Runtime queries: `queries.jsonl`, SHA-256 `e6e98f9b6de0747652d0d99b2379abe2d1e1a01ab012b1c6cae00cfab8e72bb4`
- Result manifest: `results/retrieval/baseline-execution-development-v1/manifest.json`, SHA-256 `ab45d4a51766d9d0edcc39c77f8b0ccd4abbcb1254437153f7759418cfe01703`
- Runtime checkpoint: `runtime-checkpoint.json`, SHA-256 `d99a2ee8720f16fb40867f92d1740582536ecced84c7eadfffc5f1907065e3b4`
- Results: `results.jsonl`, SHA-256 `e1e69fe8ac64a81f27e171cdd7082b7648e2a339659f11e6e4f2a1702a1dbb60`
- Checks: `checks.json`, SHA-256 `c5ec466fe24c7105931886632cb4751aca9211a816e7edd2ef8056052179bc0d`
- Run metadata: `run.json`, SHA-256 `03b1edb8079cd9601ec70aa13f71c3017170cdae20a3176fcee7e925ae564555`
- Findings: `findings.md`, SHA-256 `860e385b5d5a99742351d8bdbf33eb51710e3dbf3e9474b5b9b77c11dea2c4a9`
- Failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- The deterministic signed token-hash vector rewards shared tokens and hash collisions; it is not a semantic embedding. The index remains candidate-heavy. This step makes no relevance, latency, or production retrieval-quality claim.

### Next-step input

Step 7.4 receives the immutable runtime queries, ranking configuration, 24 B2-B4 results, exact lineage, and runtime checkpoint. Relevance annotations and retrieval metrics have not started and require separate guidance and authorization.

## Phase 7, Step 7.4: Evaluate B2-B4 retrieval quality and latency

Status: complete

### Repository state

- Starting commit: `d849ea1140f97066edb408acd8704268655c7abe`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-7.4-guidance-v1`, envelope SHA-256 `17d8609f5e0799661ea4a7d6b3a1de3493d267a6cff90efa4d21e17de4df96ae`
- Commit message: `retrieval: evaluate B2-B4 quality and latency`

### Evaluation boundary

- The runtime checkpoint was created before the relevance data, scorer contracts, scorer implementation, scorer tests, final data manifest, and final result directory existed. Its preflight records those paths as absent and records that relevance, gold, oracle, review-queue, frozen-test, and model content were not opened.
- The checkpoint contains 240 warm local latency samples: one warmup followed by ten measured repository calls for each of eight queries and three baselines. Every sample carries the digest of its frozen Step 7.3 result. Runtime generation used only the approved development releases and did not read scorer or relevance paths.
- After the checkpoint was frozen, all 200 same-user query/record pairs were reviewed in stable record-ID order: four queries over 24 records for `user_001` and four over 26 records for `user_002`. Ranked order was not used during labeling. This was a development diagnostic, not a blind evaluation; prior exposure to the committed ranking was possible.
- The reviewer checked every annotation against the approved 50-record index universe and the first 20 development source records. The two start-date correction records remain separate: the correcting record is direct evidence, while the corrected record remains useful historical evidence and is marked stale for that query. Calendar-title-only rows do not inherit relevance from a nearby project event.
- Final scoring is offline. It opens no database and performs no retrieval. The final result uses the frozen latency samples and Step 7.3 rankings without changing the index, planner, filters, search, fusion, reranking, `k`, or any predecessor artifact.

### Results

The release has eight queries, 24 baseline results, 200 reviewed annotations, 240 latency samples, and zero failures. Cross-user predictions, unsupported references, stale lineage, provider calls, retries, input tokens, output tokens, and incremental cost are all zero.

- B2: Recall@5 `16/17 = 0.958333`; Recall@10 `17/17 = 1.000000`; nDCG@10 `32.478399/33.071157 = 0.983456`; MRR `8/8 = 1.000000`; relevant-session recall is `null` for all eight cases with reason `baseline_has_no_session_path`; stale-memory rate `1/71 = 0.014085`.
- B3: Recall@5 `14/14 = 1.000000`; Recall@10 `14/14 = 1.000000`; nDCG@10 `24.821684/30.047438 = 0.836945`; MRR `6.833333/8 = 0.854167`; relevant-session recall `14/14 = 1.000000`; stale-memory rate `1/61 = 0.016393`.
- B4: Recall@5 `24/31 = 0.829167`; Recall@10 `30/31 = 0.975000`; nDCG@10 `45.124256/50.171080 = 0.910682`; MRR `7.5/8 = 0.937500`; relevant-session recall `14/14 = 1.000000`; stale-memory rate `2/72 = 0.027778`.
- Warm local latency: B2 used 80 samples with mean `10.521 ms`, p50 `9.624 ms`, p95 `15.607 ms`, and max `82.647 ms`; B3 used 80 with mean `8.863 ms`, p50 `8.408 ms`, p95 `12.289 ms`, and max `20.009 ms`; B4 used 80 with mean `13.439 ms`, p50 `13.222 ms`, p95 `18.966 ms`, and max `31.464 ms`.
- The scorecard contains 63 quality rows and 63 latency rows. It records the same numerator, denominator, value, scored-case count, null-case count, and null reason for every baseline slice by query type, benchmark capability, source type, difficulty, and development split. Source-type slices are intentionally multi-membership and are not additive. No composite score is published.

### Review corrections

- An accepted result ID outside the reviewed relevance universe originally raised an untyped key lookup error. The scorer now fails closed with `RetrievalQualityError`, and a regression covers a cross-user high scorer.
- A new regression distinguishes a report's speaker from its subject so an attributed claim is scored for the correct person.
- Release verification originally checked only artifact hashes and selected counts. It now verifies the checkpoint-before-gold boundary, every dataset and implementation binding, all relevance and review records, and byte-for-byte recomputation of per-query results, scorecard, checks, run metadata, findings, and the final manifest. A tampered bound input now fails verification.
- These corrections did not change the frozen runtime checkpoint, latency samples, relevance labels, per-query scores, scorecard, checks, run metadata, failures, or findings. Only the evaluator and its implementation-bound result manifest changed.

### Tests and contract checks

- Focused Step 7.4 unit and live PostgreSQL integration tests: 31 passed, including metric arithmetic, all five adversarial cases, immutable writes, checkpoint ordering, runtime read traps, 200-label completeness, 240 latency samples, two deterministic scorer runs, and tamper rejection.
- Protected live gates passed sequentially: retrieval baselines, planning, and index 132; storage and ingestion 30; temporal lifecycle 15; temporal evaluation 20; conflict candidates 29; conflict relations 29; belief resolution 46; Phase 5 conflict evaluation 39; sessionization 33; grounded summaries 45; and durative claims 55. The live total was 473 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 852 tests were discovered in 23.868 seconds; 717 passed and 135 database tests skipped. Every required database group passed in the sequential live gates above.
- Independent metric and nearest-rank latency recomputation matched all 24 per-query rows and all 63 aggregate and slice rows. Release self-verification, in-memory compilation, exact allowlist inspection, `git diff --check`, secret and leakage scans, and Docker cleanup passed.
- All protected Step 7.1, Step 7.2, and Step 7.3 hashes remained byte-exact. Existing-path drift from `d849ea1140f97066edb408acd8704268655c7abe` is limited to this ledger entry; the implementation adds exactly the 21 authorized Step 7.4 paths.

### Artifacts, costs, and limitations

- Evaluation configuration: `configs/retrieval/quality_evaluation_v1.json`, SHA-256 `45e0f0bd5057f92b1bdd20fb14313911b12202b79d072e8d05c6c0600489bb2a`
- Runtime manifest: `data/retrieval/retrieval-quality-development-v1/runtime/manifest.json`, SHA-256 `f148dc0df972f9bba3dd511014ba216047d206ccafcd84a0c1b107b57c5e9432`
- Dataset manifest: `data/retrieval/retrieval-quality-development-v1/manifest.json`, SHA-256 `c1cb67ce7fcec77ad8004fb8a19cabd52a1ffc26ef0dba080b13ea44bae0bb6c`
- Relevance annotations: `gold/relevance.jsonl`, SHA-256 `b1fd2024620ad8baf2165824f5715eadd1009c739dc01d11bb9aa288f8a75c83`
- Review record: `gold/review.json`, SHA-256 `68f7ee182a11e9d812e18e6eb51c1b524ad2c203ba3c3bce34c31364427600d5`
- Checkpoint preflight: `checkpoint_preflight.json`, SHA-256 `265a32d07bc85a3948bac6e8dc954a61aa53e12461444d2ce0880895e1a5ecab`
- Runtime checkpoint manifest: `checkpoint_manifest.json`, SHA-256 `82807fa0ab3ae99961aa1fef3ece65353361cb45a1ed28a9c443d901c1d688e4`
- Latency samples: `latency-samples.jsonl`, SHA-256 `4d947e64abd5d3fc0fb8d6d1f6e723a8cd77b123fa66d0e85d44971c66893305`
- Per-query scores: `per-query.jsonl`, SHA-256 `1ff5b724786cf8d1b12979484f2d156bc5fe4a9c0b7b18491e8c78295d87d1c6`
- Scorecard: `scorecard.json`, SHA-256 `0f6a9cf75e7b526ab893052a43c198a7d295da6efc8fb0d5cdabbb661469f0f8`
- Final result manifest: `manifest.json`, SHA-256 `6e89700beb6a483ce0b23c3033927122c2170897b81a366ffde31fc7785a63e4`
- Failures: both runtime and final files have empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Step 7.4 made zero provider requests, used zero tokens, and cost `$0`. Historical OpenAI spend remains `$0.2314404`.
- The evaluation covers a small development set with prior ranking exposure possible. The index remains candidate-heavy, and the deterministic token-hash vectors are not semantic embeddings. The reported scores and warm local timings are diagnostics, not production guarantees.

### Phase 8 handoff

Phase 8 receives the frozen session, index, planner, ranking, lineage, relevance, and retrieval-quality artifacts. Step 7.4 does not authorize an answerer, evidence-package assembly, benchmark test run, model call, or any Phase 8 implementation. Phase 8 has not started.

## Phase 8, Step 8.1: Build validated evidence packages

Status: complete

### Repository state

- Starting commit: `1df7e3c0846e9bebb873debe4cc2ae1f532a2cb1`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-8.1-guidance-v1`, envelope SHA-256 `6af1d3eb7d227c430a93b89abfce8fcc1853c5d29027f828650b5c8a58dec229`
- Commit message: `answering: build validated evidence packages`

### Implementation and review

- The builder hydrates the 24 frozen Step 7.3 results in a read-only, repeatable-read transaction. Queries bind the user, index version, snapshot, immutable claim version, source, and span before returning any row. Atomic and session paths are deduplicated by claim version; session summaries remain navigation records and never become evidence.
- Every accepted claim version must match the frozen retrieval item's exact claim, source, and span lineage. Quotes are checked against the source bytes, including offset bounds when offsets exist. Raw source content, participant payloads, arbitrary metadata, embeddings, summary prose, and unresolved-question prose are not serialized.
- Lifecycle handling stays conservative. Current and confirmed claims are current, historical and superseded claims are historical, disputed claims are conflicting, and candidates are rejected with `package_validation/candidate_not_promoted`. Excluded or restricted evidence fails the package instead of being exposed.
- Package identity binds the schema, configuration, runtime and input-release versions and hashes, user, query, baseline, execution, result, snapshot, index, and time cutoffs. Rejections retain stable identifiers, stage, rank where the predecessor supplies one, and exact reasons.
- The review tightened invariant checks, source and span equality, offset validation, blocker order, manifest verification, and recomputed counters. It also made relation hydration validate both decision endpoints by exact user, claim, and version ownership, half-open transaction visibility, inclusive requested valid time, and cutoff. Regressions cover a future endpoint, an ineligible endpoint, a poisoned cross-user relation, and a poisoned accepted cross-user record ID.
- The runtime does not rerun planning, filtering, search, ranking, or relevance scoring. It opens no Step 7.4 relevance gold or scorecard and makes no model or provider call.

### Development release

The release contains 24 packages and zero failures: eight each for B2, B3, and B4. It hydrates 204 accepted retrieval records into 33 unique candidate claim versions. All 24 packages have complete exact provenance, with zero cross-user, missing-lineage, post-cutoff, quote-mismatch, restricted, duplicate, or partial-package records.

All 33 development claims remain candidates. The release therefore has 295 candidate rejections, 196 carried retrieval rejections, zero categorized claim versions, zero serialized relevant sources or spans, and zero `answer_allowed=true` packages. This is the expected conservative result. Complete provenance does not promote a candidate or establish semantic answerability.

### Tests and contract checks

- Focused Step 8.1 unit and live PostgreSQL integration tests: 33 passed, including exact package execution and replay, quote and offset validation, lifecycle partitions, relation endpoint cutoffs, poisoned ownership, read traps, tamper rejection, and two clean byte-identical releases.
- Protected live gates passed sequentially: Step 7.4 retrieval quality 31, retrieval baselines 132, Phase 5 conflict evaluation 39, storage and ingestion 30, durative claims 55, grounded summaries 45, and sessionization 33 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 885 tests were discovered in 27.224 seconds; 731 passed and 154 database tests skipped. The required database groups passed in the live gates, with the remaining groups already covered by the coding gate.
- Release self-verification, in-memory compilation, exact allowlist inspection, protected-hash checks, `git diff --check`, secret and leakage scans, staged and unstaged inspection, and Docker cleanup passed. No predecessor file changed before this ledger entry; the implementation adds exactly the 15 authorized Step 8.1 paths.

### Artifacts, costs, and limitations

- Evidence-package configuration: `configs/answering/evidence_package_v1.json`, SHA-256 `64c85873389c291bcee89df6aed6dc3a72ea4cd396fd0593613b809acdf17fdd`
- Dataset manifest: `data/answering/evidence-package-development-v1/manifest.json`, SHA-256 `016b34eccba3260974e5c8eb2be58d6fb2023ad4399919634577b82c5c4bc7f4`
- Packages: `packages.jsonl`, SHA-256 `bb57898bea51417b2ecad1252b451669ae2c748033c9cef88360f16a825f8186`
- Checks: `checks.json`, SHA-256 `9e081b65e5bc8c8f59c8fb320a357e14240eedc06fd5febb67ddbcf588d8c766`
- Run metadata: `run.json`, SHA-256 `87c10978b82cf309e01982b580c41a9e48371dd262f9d8cc28aa7a4d5afd3a1d`
- Findings: `findings.md`, SHA-256 `3c9a43eb218073ace99aff998948c1f80a1b97927ca7f7d479ae14294325a211`
- Failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Result manifest: `manifest.json`, SHA-256 `8d0b3a44c5a7452827f5ead93fedc39ade92a6097cfa06641eecee0c996d8213`
- Step 8.1 made zero provider requests, used zero tokens, and cost `$0`. Historical OpenAI spend remains `$0.2314404`.
- This release verifies package structure and provenance only. The candidate-heavy upstream data yields no promoted evidence and no answer-allowed package; it does not measure relevance, truth, or answer quality.

### Next-step input

Step 8.2 receives only the frozen `evidence_package_development_v1` release and its manifest. Answer generation, citation rendering, abstention text, model use, and Step 8.2 implementation have not started and require separate guidance and authorization.

## Phase 8, Step 8.2: Add the memory answer contract

Status: complete

### Repository state

- Starting commit: `8fec075d754dff7f12821947919d5c01f867d949`
- Branch: `codex/implementation-handoff-3.5-11.4`
- Ending commit: the commit containing this entry
- Guidance: `step-8.2-guidance-v1`, envelope SHA-256 `91593dde89451249910fdbc00b0f048db4964d499b0c83d3c60deba5c3a2033d`
- Commit message: `answering: add grounded memory answer contract`

### Contract and runtime boundary

- The new immutable contracts cover `answered`, `abstained`, `disputed`, and `partially_answered` outputs. Answer and statement IDs are hashes of their canonical content without the ID field. Unknown fields, unsafe JSON, duplicate or unsorted provenance, invalid time data, non-finite confidence, and inconsistent status fields fail closed.
- Each factual statement names exact claim and immutable version IDs. Its citations must match the same package's evidence ID, source, span, nullable message ID, and quote. The validator rejects missing, changed, rejected, cross-user, wrong-version, wrong-source, wrong-span, wrong-message, and wrong-quote provenance.
- Non-abstained answer text is only the newline join of its grounded statement text. Answered output cannot cite conflicting claims. Disputed output needs at least two conflicting claim versions, and partial output needs a grounded statement plus a non-empty unresolved part.
- The input loader first runs the frozen Step 8.1 verifier, then checks the exact dataset, result, and package hashes. Every answer binds the Step 8.1 release, the canonical package record, user, query, baseline, plan, snapshot, time cutoffs, configuration, prompt, and runtime versions.
- The renderer treats package strings as untrusted JSON data. Its frozen schema spells out the exact status, statement, claim-reference, and citation fields, status rules, and provenance requirements. It omits rejected evidence, summary prose, raw source content, source metadata, embeddings, gold, and runtime-owned IDs.
- `answer_allowed=false` short-circuits before prompt rendering or candidate inspection. The provider model remains a dormant configuration value for a future validated candidate path. Step 8.2 has no provider adapter, environment lookup, database write, or model call.

### Review corrections

- The first renderer listed only top-level candidate and statement field names. It now includes the strict nested candidate schema, allowed status values, category rules, citation fields, and provenance requirements promised by the contract.
- The contract now rejects invalid valid-time representations and enforces the exact generation-mode and dormant-model binding for non-blocked candidates. Unsafe Unicode strings fail with the same sanitized contract error as other unsafe JSON.
- Dataset and result verification now bind all six Step 8.1 authorities: dataset manifest, result manifest, packages, checks, run metadata, and failures. A regression proves a rehashed authority mismatch fails closed.
- These changes alter the dormant prompt hash and therefore the answer IDs, `answers.jsonl`, and final manifest. The release was regenerated before any model, gold, oracle, review, or frozen-test access.

### Development release

All 24 frozen packages contain only candidate claims and carry `no_promoted_claims`. Each produced the fixed abstention `I cannot answer this from the available memory.` with the configured unconfirmed-claim reason, confidence zero, no statements or citations, and null requested and resolved model fields.

The release has 24 answers, 24 abstentions, and zero answered, disputed, partial, failed, duplicate, cross-user, or invalid-provenance records. It made no provider request and used no tokens. These counts test contract behavior only; they do not measure answer correctness or abstention accuracy.

### Tests and contract checks

- Focused Step 8.2 unit and integration tests: 27 passed, covering all four statuses, canonical IDs, strict prompt schema and escaping, blocked short-circuiting, time and model binding, exact citation closure, sanitized failures, all authority hashes, immutable output, byte-identical replay, and prohibited-read traps.
- Step 8.1 unit and live PostgreSQL integration tests: 33 passed.
- Protected live gates passed: retrieval baselines 132, Phase 5 conflict evaluation 39, grounded summaries 45, sessionization 33, temporal lifecycle 15, and storage with ingestion 30 tests.
- `make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`: passed with dataset SHA-256 `746756cb7d9aa76d3646d96b50ba74c0616780c7d015cb0f48f685ad03746b61`.
- `make test PYTHON=.venv-storage/bin/python`: 912 tests were discovered in 27.969 seconds; 758 passed and 154 database tests skipped. The required database groups passed in the live gates above.
- Release self-verification, two clean byte-identical runs, in-memory compilation, exact allowlist inspection, protected hashes, `git diff --check`, secret and prohibited-data scans, staged and unstaged inspection, and Docker cleanup passed. No predecessor file changed before this ledger entry; the implementation adds exactly the 14 authorized Step 8.2 paths.

### Artifacts, costs, and limitations

- Memory-answer configuration: `configs/answering/memory_answer_v1.json`, SHA-256 `98b743d593e17a88216ac74cf17693f08898537027e0d91672d45aaeb4dfcc10`
- Dataset manifest: `data/answering/memory-answer-contract-development-v1/manifest.json`, SHA-256 `015e0d5e16f9870e04728f3668ee1f21f8c1b1a9a383f274c3c6bcb23c6f455e`
- Prompt contract: `memory_answer_prompt_v1`, SHA-256 `69dd688430c55fc35a16369201ef8eea7d470d10f610edec254f5cbd59bf14c1`
- Answers: `answers.jsonl`, SHA-256 `d83fcac2eed3a4b2493c575cc49f593657669a690b9cfa61307afc8890940ba4`
- Checks: `checks.json`, SHA-256 `3642da6ae21165b1c2ddf6c4e64773e423615aa3a85b18834c8682ba26252164`
- Run metadata: `run.json`, SHA-256 `bcbe9eef40c0b126e106c7b88c61e2a5c74db6df5fd2be54cd1b8183cb1e80d2`
- Findings: `findings.md`, SHA-256 `63aefda40e96341de9e62181db8397be70aaa152d45a78a70279389a5726bd55`
- Failures: empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Result manifest: `manifest.json`, SHA-256 `d0d987ff126aca2c7b05a0966e6b797247c7123e252fb26fc59d9599374fb841`
- Step 8.2 made zero provider requests, used zero tokens, and cost `$0`. Historical OpenAI spend remains `$0.2314404`.
- All real development outputs abstain because upstream has no promoted claims. Positive statuses are covered only by invented unit fixtures and are not benchmark results.

### Next-step input

Step 8.3 receives the frozen answer schema, prompt contract, candidate validator, configuration, all-abstained development release, and its candidate-data limitation. Comparable answer runs, answer gold, paid-call preflight, provider execution, and Step 8.3 implementation have not started and require separate guidance and authorization.
