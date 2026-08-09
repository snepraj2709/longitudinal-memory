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
