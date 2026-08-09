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
