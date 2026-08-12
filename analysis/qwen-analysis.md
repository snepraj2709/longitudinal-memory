# Qwen3-8B run failure analysis

## Executive summary

The Qwen3 run succeeded as a paid JarvisLabs execution, but the benchmark quality was weak. It is better to call this a completed development scorecard with serious quality failures, not an infrastructure failure.

The final run, `qwen3-8b-vllm-dev-v1`, produced 912 sealed logical predictions for B0-B7. Provider execution, answer validation, prediction sealing, and judge diagnostics completed without provider failures. The run still exposed real benchmark problems: 13 context materialization failures, very low answer accuracy, weak source-message evidence recall, heavy over-abstention, and semantically failed extraction.

The main lesson for future LLM runs is simple: do not spend on a full B0-B7 pass until extraction quality, answer abstention behavior, and source-message citation coverage pass a small cheap gate.

## JarvisLabs run facts

This run used rented JarvisLabs hardware.

- Model: `Qwen/Qwen3-8B`
- Served alias: `qwen3-8b-vllm`
- Runtime: vLLM through an OpenAI-compatible API
- GPU: JarvisLabs L4 in IN2
- Context length: `8192`
- Temperature: `0`
- Client concurrency: `1`
- Final output root: `results/evaluation/qwen3-8b-vllm-dev-v1/`
- Final scorecard status: completed development scorecard, no composite score

Important artifacts:

- `results/evaluation/qwen3-8b-vllm-dev-v1/run-manifest.json`
- `results/evaluation/qwen3-8b-vllm-dev-v1/scores/metrics.jsonl`
- `results/evaluation/qwen3-8b-vllm-dev-v1/predictions/predictions.jsonl`
- `results/evaluation/qwen3-8b-vllm-dev-v1/contexts/failures.jsonl`

## Cost and GPU time

The final successful development run recorded:

- Provider requests: `930`
- Input tokens: `2,653,791`
- Output tokens: `193,672`
- GPU inference time: `3093.778641` seconds, about `51.56` minutes
- Recorded JarvisLabs GPU cost: `INR 97.5664`
- Transport retries: `0`

This cost is only the recorded GPU cost for the final run metadata. Earlier compatibility and failed attempts also consumed some rented GPU time, but the committed final manifest only records `INR 97.5664` for the successful development run. One earlier compatibility run recorded about `INR 3.32` for a 12-request pilot.

## Attempt history

These attempts should not all be treated as benchmark results. They are useful failure evidence.

| Attempt | Result | What happened | Keep or rerun? |
| --- | ---: | --- | --- |
| `qwen3-8b-vllm-pilot-v1-http404-attempt-001` | `0/12` succeeded | Endpoint path/config problem. No useful model quality signal. | Keep as setup lesson only. Do not rerun. |
| `qwen3-8b-vllm-pilot-v1-answer-validation-attempt-002` | `6/12` succeeded | Strict output validation failed on compatibility cases. | Keep as schema lesson. Do not score. |
| `qwen3-8b-vllm-pilot-v1` | `10/12` succeeded | Compatibility mostly passed, but two strict validation failures remained. | Keep as partial pilot evidence. |
| `qwen3-8b-vllm-pilot-v1-success-attempt-003` | `12/12` succeeded | Compatibility passed after local hardening. | Keep as successful gate evidence. |
| `qwen3-8b-vllm-pilot-v1-success-attempt-004` | `12/12` succeeded | Repeat compatibility pass. | Keep one compact reference; duplicate full artifacts are optional. |
| `qwen3-8b-vllm-dev-v1-answer-validation-attempt-010` | `906/912` sealed | Development pass still had answer/B7 validation failures. | Keep as failure lesson. Do not score. |
| `qwen3-8b-vllm-dev-v1` | `912/912` sealed | Final development B0-B7 run completed, with 13 context materialization failures recorded separately. | Keep as the scored development artifact. |

There are also local scratch attempt directories without complete manifests. Keep them out of benchmark claims unless a future cleanup pass extracts a compact ledger from them.

## Operational failures versus quality failures

The final run had a clean provider path:

- Final provider execution failures: `0`
- Final answer/schema failures: `0`
- Final prediction-seal failures: `0`
- B0-B6 answer calls: `798/798` succeeded
- B7 derived predictions: `114/114` succeeded
- Judge diagnostics: `112/112` succeeded, but they are diagnostic only and uncalibrated

The final run still had materialization failures:

- Context materialization failures: `13`
- Stage: `belief_resolution`
- Failure codes: `lifecycle_execution_failed`, `open_version_drift`
- Users affected: both `user_001` and `user_002`

These 13 failures did not stop the run from producing 912 contexts and 912 sealed predictions, but they matter. They show that the memory pipeline has lifecycle/version drift cases that should be fixed before another paid run. They are separate from Qwen's answer quality failures.

## Main failure modes

### Over-abstention

Qwen abstained too often. B0 abstaining on everything is expected because B0 has no memory context. The problem is that B1-B7 also abstained on many answerable QA cases.

The clearest example is B6/B7: they had 40 unnecessary QA abstentions out of 80 answerable QA cases. That alone caps answer accuracy.

### Flat answer accuracy

Strict QA accuracy stayed low across all memory baselines:

| Baseline | Strict QA accuracy |
| --- | ---: |
| B0 | `20%` |
| B1 | `24%` |
| B2 | `23%` |
| B3 | `23%` |
| B4 | `25%` |
| B5 | `25%` |
| B6 | `25%` |
| B7 | `25%` |

The memory system retrieved or supplied more evidence, but Qwen did not turn that evidence into reliably correct answers.

### Weak evidence recall

Evidence recall was weak even when retrieval found useful material.

- B2 retrieval Recall@10: `74.4%`
- B2 final source-message evidence recall: `40.1%`
- B4 retrieval Recall@10: `67.2%`
- B4 final source-message evidence recall: `34.7%`

This means retrieval is not the only bottleneck. Qwen often saw some useful context but cited too few of the required source messages.

### Citation precision was better than citation completeness

Source-message precision stayed around `74-80%` for B1-B7. Source-message recall stayed much lower, around `35-52%`.

When Qwen cited evidence, it was often valid. It just did not cite enough of the evidence trail. That is bad for this benchmark because memory quality depends on complete provenance, not only a plausible answer.

### Exact quote quality dropped after B1

B1 exact quote correctness was `78.4%`. B2-B7 were only around `25-29%`.

The likely cause is the context format. B1 uses source history. B2-B7 use transformed atomic/session memory records. Those records preserve some source IDs and quotes, but the answer prompt is not reliably copying exact source-message evidence from them.

### Extraction was structurally valid but semantically wrong

Qwen extraction looked valid at the schema level:

- Structurally valid sources: `20/20`

But semantic scoring failed:

- Claim precision: `0/36`
- Claim recall: `0/32`
- Claim F1: null because there were no matched claims

This is probably the biggest upstream problem. Schema-valid extraction is not enough. The extracted claims need to match the benchmark's ontology, subject scope, lifecycle rules, and evidence expectations.

## Failure modes by baseline

- B0: This is a no-memory baseline. It abstained on all 114 predictions, which is expected. It only scores on true abstention cases.
- B1: Best evidence recall at `52.1%`, but strict QA accuracy was only `24%`. Full history helped citation coverage, but Qwen still failed too many reasoning and answerability cases.
- B2: Atomic retrieval had decent Recall@10, but final evidence recall fell to `40.1%`. Many answerable cases became abstentions.
- B3: Session summaries behaved like B2. Abstention behavior was a little better, but answer accuracy stayed weak.
- B4: Hybrid retrieval underperformed B2/B3 on evidence recall. Mixing atomic and session records under a four-context cap seems to dilute useful evidence.
- B5: There was no real conflict-aware lift. Relation accuracy was not scored because no conflict relation reference was available.
- B6: Lifecycle improved temporal/lifecycle accuracy from B5 `16.3%` to `29.4%`, but final answer accuracy stayed at `25%`.
- B7: B7 did not add useful behavior in this run. The manifest confirms B6/B7 answer identity, so the gate did not materially change the answers.

## Root causes

1. Extraction quality is the first blocker. Qwen produced schema-valid extraction output, but the claims did not match the benchmark's semantic contract.
2. The answer prompt is too conservative. It often treats incomplete or transformed evidence as a reason to abstain, even when the case is answerable.
3. Memory contexts are too lossy for source-message scoring. B2-B7 contexts expose source IDs and quotes, but not in a way Qwen reliably turns into complete citations.
4. The four-record context cap is too tight for several longitudinal cases. Multi-hop history, corrections, and lifecycle changes need more evidence than a compact context allows.
5. B7 is not yet a real independent gate. It derived from B6 and preserved B6/B7 answer identity, so it did not fix over-answering or over-abstention.
6. The pipeline allowed 13 materialization failures to remain in a completed run. That is acceptable as recorded development evidence, but future paid runs should either fix those cases first or explicitly exclude them from the paid scope.

## Lessons for future LLM runs

- Do not run full B0-B7 until a cheap gate passes.
- Treat `12/12` compatibility as the minimum entry point, not as proof of benchmark readiness.
- Add a 20-50 case development sample before a full paid run.
- Stop early on schema or validation failures. Do not finish hundreds of calls after the first repeated validation failure.
- Track provider requests, tokens, GPU time, cost, machine ID, GPU type, region, output directory, and cleanup confirmation in every run note.
- Keep judge diagnostics separate until calibrated. The judge run completed, but it is not authoritative.
- Separate infrastructure failures from quality failures. The final Jarvis/vLLM path worked; the benchmark quality did not.
- Fix materialization failures before spending on another full run.

## Failed-artifact retention and backlog recommendation

Do not commit full failed attempt directories. They are large, noisy, and easy to confuse with scored results.

Keep a lightweight backlog instead. For each failed attempt, preserve:

- Attempt name
- Failure type
- Success/failure count
- Token or cost signal, if available
- Lesson learned
- Decision: rerun, archive, or ignore

Recommended decision for the current state:

- Keep the final successful artifacts committed.
- Keep failed directories local for now.
- Later replace the failed directories with a compact ledger, either in this file or in `analysis/qwen-failed-attempts-ledger.md`.
- Do not rerun full B0-B7 until extraction and abstention fixes pass a cheap 12-request or 20-request gate.

## Cost-saving plan before the next paid JarvisLabs run

1. Run a zero-cost local schema and materialization dry run.
2. Fix the 13 `belief_resolution` materialization failures or explicitly remove those records from the next paid scope.
3. Run only the 12-request compatibility pack.
4. Require `12/12` pass before any larger run.
5. Run a capped 20-50 case development sample.
6. Inspect extraction claim precision/recall, answer abstention rate, and source-message recall before approving a full run.
7. Only run full B0-B7 after the small sample shows that the model is not repeating the same failure modes.

Fix order:

1. Improve extraction prompt/schema semantics.
2. Add exact source-message evidence into memory contexts.
3. Tune abstention behavior on answerable QA.
4. Validate B7 separately so it changes B6 behavior when it should.
5. Then run a new development or hidden-test scorecard.
