# Longitudinal memory implementation handoff

## Purpose

This document is the implementation handoff for the rest of the project. It records what exists now, what must be built next, and what each phase must leave behind for the following phase.

Use these documents as the source of truth:

- [Memory evaluation steps](memory-evaluation-steps.md) is the phase roadmap.
- [Benchmark](benchmark.md) defines releases, cases, baselines, and scoring.
- [Memory ontology](memory-ontology.md) defines claims, evidence, time, lifecycle, and provenance.
- [Architecture design](architecture-design.md) defines the target system and debugging application.

If an example here drifts from one of those contracts, follow the contract. Update this handoff only after the contract has been corrected and reviewed.

The project evaluates memory from conversation transcripts, chat, email, and calendar records. Audio capture and transcription quality remain outside the benchmark.

## Current status

This snapshot describes commit `5002450`. It distinguishes implemented component work from completed benchmark evidence. A component marked development-tested has not necessarily been run as a full B0-B7 ablation.

| Phase | Status | What exists |
| --- | --- | --- |
| 1. Benchmark contract | Complete | The pilot and scaled contracts, ontology, releases, cases, and user-level split are frozen at their documented scopes. |
| 2. Full-history baseline | Scored at pilot scope | B1 answered all 25 Pilot v0 questions from source history and returned evidence. |
| 3. Atomic extraction | Scored at pilot scope | The reviewed 10-source v2 pilot extracted and scored atomic claims. The scaled OpenAI extraction is historical and unscored; the Qwen scaled extraction has not run. |
| 4. Bi-temporal versioning | Implemented and development-tested | PostgreSQL persistence, idempotent ingestion, valid-time and transaction-time transitions, lifecycle handling, and component scoring exist. They have not been exercised in a complete scaled ablation. |
| 5. Conflict detection | Implemented and development-tested | Candidate linking, checked relation persistence, deterministic belief resolution, and component evaluation exist. They have not been exercised in a complete scaled ablation. |
| 6. Session summaries | Implemented and development-tested | Sessionization, grounded summary contracts, durative claims, and component evaluation exist. No complete scaled Qwen summary release exists. |
| 7. Dual retrieval | Implemented and development-tested | Atomic and session indexes, query planning, pre-search filters, B2-B4 fusion, and component evaluation exist. No complete scaled B2-B4 Qwen release exists. |
| 8. Grounded answering | Partial | Evidence-package and answer contracts exist. The development release structurally abstained on all available B2-B4 cases; B5-B6 answer evidence is incomplete and no scaled Qwen answers exist. |
| 9. Abstention gate | Partial | The deterministic gate, thresholds, and matched four-case B6/B7 development comparison exist. Both baselines abstained on all four cases, so over-abstention remains unresolved and the 16 frozen cases remain deferred. |
| 10. Ablations | In progress | Steps 10.1 and 10.2 are complete. OpenAI Step 10.3 is `interrupted_not_scored`; Qwen v1 is configured but unrun; corrected Qwen v2 is next. Step 10.4 is pending measured Qwen results. |
| 11. UI and deployment | Read-only slice deployed | The deterministic FastAPI and React demo runs locally and is live on Railway. It replays sanitized artifacts and currently presents partial historical results, not a completed Qwen benchmark. |

### Phase 2 evidence

The authoritative files are the [B1 run manifest](../results/pilot/b1-full-history/run.json), [scores](../results/pilot/b1-full-history/scores.json), [predictions](../results/pilot/b1-full-history/predictions.jsonl), and [findings](../results/pilot/b1-full-history/baseline_findings.md).

The completed run used the pinned `gpt-4.1-2025-04-14` snapshot. It produced 25 successful predictions and no final execution failures.

| Measure | Result |
| --- | ---: |
| Strict answer accuracy | 0.92 |
| Lenient answer accuracy | 0.96 |
| User-modelling strict accuracy | 0.60 |
| Unsupported-claim rate | 0.03125 |
| Source evidence micro recall | 0.675 |
| Message evidence micro recall | 0.50 |

B1 is a useful ceiling for the small pilot, not proof that memory works. It sees the full source history for every question. Its weaker user modelling and evidence recall also show that a correct-looking answer can still use incomplete or imprecise support.

### Phase 3 evidence

The authoritative files are the [Phase 3 run manifest](../results/phase3/atomic-extraction-v2/run.json), [scores](../results/phase3/atomic-extraction-v2/scores.json), [predictions](../results/phase3/atomic-extraction-v2/predictions.jsonl), and [reviewed gold](../data/phase3/atomic_extraction_gold.jsonl).

The corrected v2 pilot processed ten sources successfully. Its gold set contains 48 claims; the extractor produced 36.

| Measure | Result |
| --- | ---: |
| Claim precision | 0.75 |
| Claim recall | 0.5625 |
| Claim F1 | 0.642857 |
| Valid-time accuracy | 0.666667 |
| Provenance-span precision | 0.305556 |
| Provenance-span recall | 0.22 |
| Unsupported-memory rate | 0 |

The zero unsupported-memory rate means the produced claims were grounded in their sources. It does not cancel the missed claims, object mismatches, time errors, or imprecise gold-span matches. Phase 4 must not assume that every relevant fact was extracted or that every time boundary is correct.

The current Phase 3 runner also stops on a failed call and refuses a non-empty result directory. Unlike B1, it has no per-case checkpoint, safe resume, preflight cost estimate, or hard cost cap. Fix those gaps before another paid extraction run.

### Next implementation

Run the corrected `qwen35-27b-fp8-v2` comparison under the execution contract in [Qwen implementation](qwen-implementation.md). It must materialize B2-B7 through the real PostgreSQL-backed Phase 4-7 services, preserve the OpenAI and Qwen v1 histories, publish reproducible scores, and then update the read-only demo.

## How the phases connect

```text
Reviewed source records
  -> Phase 3 atomic claims and exact evidence
  -> Phase 4 temporal versions and lifecycle
  -> Phase 5 relations and current-belief resolution
  -> Phase 6 session summaries and durative claims
  -> Phase 7 ranked atomic and session evidence
  -> Phase 8 grounded answers
  -> Phase 9 answerability decisions
  -> Phase 10 controlled B0-B7 comparisons
  -> Phase 11 inspectable local demonstration
```

Each phase adds one capability. It must not hide failures inherited from an earlier phase.

| Upstream limitation | Downstream effect |
| --- | --- |
| Extraction misses a claim. | Versioning, conflict resolution, and retrieval cannot recover it. |
| A valid-time boundary is wrong. | Current and historical state selection may be wrong. |
| A claim is linked to the wrong person. | The user model and every later answer can become unsafe. |
| A correction is classified as a contradiction. | The resolver may leave a needless dispute. |
| A summary drops uncertainty. | Session retrieval can make an inference look like a fact. |
| Retrieval omits the required claim or session. | The answerer cannot cite complete evidence. |
| Retrieval returns related but insufficient evidence. | The answerability gate must still abstain. |

This is why every phase needs an isolated score and a downstream contract. A better final answer score does not excuse worse provenance, stale retrieval, or unsafe coverage.

## Working protocol

An implementation agent must work on one numbered step at a time. After that step:

1. Run the focused tests for the changed component.
2. Run any deterministic scorer affected by the change.
3. Re-read the four governing documents listed at the start of this file.
4. Check leakage, user isolation, provenance, time semantics, deletion behavior, and compatibility with completed phases.
5. Record input hashes, output hashes, measured results, failures, and known limitations.
6. Describe the exact artifact or interface the next step will receive.
7. Stop for Sneha's review.

Do not begin the next phase, make an LLM call, commit, deploy, or change hosted infrastructure without Sneha's explicit approval. A phase approval does not grant permission for those other actions.

### Step report

Every completed step should leave a short report with these fields:

```text
phase and step
repository commit and worktree state
inputs and hashes
implementation changed
tests run and results
metrics run and results
failures and limitations
contract checks
artifacts and hashes
next-step input
approval still required
```

Store run evidence in a versioned result directory. Do not overwrite an earlier run. Execution failures and reasoning failures must remain separate.

### Phase exit review

Before a phase is complete, verify all of the following:

- Its isolated evaluator exists and runs deterministically where possible.
- Runtime prompts cannot read oracle events, reference answers, gold summaries, gold conflicts, or gold evidence labels.
- Every runtime entity and query is scoped by `user_id`.
- Every promoted claim and factual answer traces to an exact source span.
- The phase consumes only documented outputs from earlier phases.
- Reprocessing is idempotent and does not create duplicate derived records.
- Deleting a source removes its evidence and recomputes or retires unsupported derived data.
- New fields and labels match the ontology.
- Benchmark counts, splits, and metrics match the benchmark contract.
- Failed or regressed scores are present in the result, not omitted or rewritten.
- The full local test suite still passes.

Research targets are not phase-completion claims. A phase may finish below a target if its contract is sound, the result is measured honestly, the limitation is documented, and Sneha accepts the downstream risk. Structural invalidity, provenance leakage, cross-user access, or missing reproducibility controls block the phase regardless of score.

## LLM calls and cost approval

The ignored `.env` file contains `OPENAI_API_KEY`. Read only that named value through the existing safe loader. Never print, log, serialize, or commit it.

No LLM call is implicit. Before each bounded run, give Sneha one approval request that covers the complete run.

### Required preflight

The request must state:

```text
purpose and phase
runtime cases and source IDs sent to the provider
whether any sensitive or restricted fields are present
requested model and pinned snapshot
prompt version and SHA-256
dataset version, split, and SHA-256
generation settings
planned model requests
maximum retry requests
estimated input tokens
expected output tokens
hard output-token ceiling
current input and output prices
expected total cost
hard maximum cost
output directory
checkpoint and resume behavior
confirmation that oracle and gold fields are excluded
```

Treat credential reuse, paid execution, and transmitting the listed source records as separate facts inside the approval. Sneha's approval must cover all three.

The B0-B7 answer model remains `gpt-4.1-2025-04-14` so architecture changes can be compared against B1. The [official GPT-4.1 model page](https://developers.openai.com/api/docs/models/gpt-4.1) currently lists standard text rates of $2 per million input tokens and $8 per million output tokens. Recheck that page immediately before every approval request. Do not rely on the price copied into this document.

Calculate cost as:

```text
expected cost =
  expected uncached input tokens * current input rate
  + expected cached input tokens * current cached-input rate
  + expected output tokens * current output rate

hard maximum cost =
  maximum approved input tokens * current input rate
  + maximum approved output tokens * current output rate
```

Divide token counts by one million when applying the rates. Assume no cache discount unless the run can prove which tokens qualify. Use exact rendered prompts for the input estimate. Use measured development output for expected output and the configured token ceiling for the maximum.

Approval expires if the model, prompt, data, cases, settings, transmitted fields, request count, retry allowance, or cost ceiling changes. Ask again with a revised total.

### Failure and resume rules

- Complete schema, hash, leakage, and dry-run checks before the first paid request.
- Write a checkpoint after every provider response.
- Never regenerate a successful case during resume.
- Retry only a provider failure that returned no usable model output.
- Permit at most the retry count stated in the approval.
- Stop before a request that would exceed the approved request or cost ceiling.
- Do not retry a validation failure automatically. Preserve it for diagnosis.
- Do not rerun a stage merely because its score is disappointing.
- Reuse deterministic intermediate artifacts when their hashes still match.

A cheaper pinned model may be proposed for extraction, synthetic rendering, summaries, or judging. First compare it with the current model on a small development set, report the quality and cost difference, and request approval. Do not switch the B0-B7 answer model without rerunning B1 and freezing a new comparison series.

## Benchmark release work before Phase 4

The benchmark must stay ahead of the architecture. Build the release data before tuning temporal, conflict, summary, retrieval, or answerability behavior against it.

### Step R1: Expand Maya to Benchmark v1

Inputs:

- [Pilot user](../data/pilot/user.jsonl)
- [Pilot oracle](../data/pilot/oracle-event.jsonl)
- [Pilot sources](../data/pilot/sources/)
- [Pilot questions](../data/pilot/evaluation/eval_questions.jsonl)
- [Pilot answers](../data/pilot/evaluation/eval_answer.jsonl)

Implement:

- Expand QA from 25 to 50 cases, ten per capability.
- Add five longitudinal summary cases.
- Add two interactive scenarios.
- Add the common benchmark fields and keep stable `case_id` values.
- Add reviewed gold events, claims, evidence, answerability, and failure tags.
- Keep runtime source files separate from scorer-only gold.
- Add schema and consistency checks before any model-assisted rendering.

Outputs:

- A versioned Benchmark v1 dataset.
- A release manifest with file hashes, counts, review state, and split rules.
- Deterministic validators for IDs, references, timestamps, evidence, capability balance, and leakage.

Exit when the release has exactly 50 QA cases, five summaries, and two interactive scenarios; every gold reference resolves; all required cases have been reviewed; and runtime loaders cannot open its gold fields.

### Step R2: Build the scaled quality benchmark

Implement:

- Create ten complete synthetic users.
- Keep two whole users in development and eight whole users in the frozen test set.
- Do not split questions from one user across development and test.
- Create 500 QA cases, 100 per capability.
- Create 50 summaries: 25 temporal and 25 user-model summaries.
- Create 20 interactive scenarios, four per capability.
- Include fragmentation, corrections, temporal changes, disagreement, uncertainty, wrong-person distractors, hypotheticals, missing facts, and cross-user distractors.
- Review every correction, conflict, abstention case, timeline, summary event set, and evidence set.
- Freeze histories, cases, gold, schemas, splits, and hashes before architecture tuning.

Only the two development users may guide prompts, thresholds, or rules. The eight test users remain frozen. Runtime generation receives source observations or derived memory, never oracle truth or evaluation gold.

Exit when the scaled counts and user-level split match the benchmark contract and an independent validation command can reproduce the release hashes.

### Step R3: Build a separate load corpus

The architecture mentions approximately 500,000 events. Do not turn that number into the quality benchmark. The quality benchmark must remain reviewable.

Generate a separate non-evaluation corpus for ingestion, reprocessing, retrieval latency, and storage tests. It may reuse structural templates but must not contribute QA, summary, or interactive quality denominators. Its manifest must label it `load_test`, not `development` or `test`.

Exit when performance reports cannot accidentally combine load-corpus records with benchmark quality records.

## Phase 3 hardening

Phase 3 converts immutable source observations into candidate claims. It records what a source says, including reports, uncertainty, denials, corrections, and hypotheticals. It does not decide final truth.

Expected code areas:

```text
src/extraction/
configs/extraction/
data/phase3/
results/phase3/
tests/unit/
tests/integration/
```

### Step 3.1: Freeze and analyse the v2 pilot

Inputs are the current v2 run, predictions, scores, reviewed gold, prompt version, and source hashes.

Implement a deterministic failure report for:

- Missed gold claims.
- Extra or unsupported claims.
- Subject, speaker, predicate, and object mismatches.
- Polarity and epistemic-status mismatches.
- Missing, wrong, or over-broad time boundaries.
- Exact evidence-span mismatches.
- Differences by source type and predicate family.

Do not change or overwrite v2. The report must distinguish source-backed representation differences from unsupported model output.

Exit when every false positive and false negative has a stable category and a source-backed explanation.

### Step 3.2: Add run safety before another paid extraction

Bring the Phase 3 runner up to the B1 run standard:

- Freeze model, prompt, generation settings, case order, registry version, and input hashes.
- Add a dry run that renders prompts, validates inputs, counts requests, and estimates tokens and cost without writing results or calling a model.
- Add an approved hard cost cap.
- Checkpoint each case with provider metadata and token usage.
- Resume only missing cases or provider failures with no model output.
- Refuse incompatible checkpoints and completed result directories.
- Record repository commit, worktree state, requested and resolved model, prompt hash, input hashes, output hashes, and execution failures.
- Sanitize error text and exclude secrets.
- Load gold only after all runtime predictions are complete.

Exit when unit tests prove that interrupted work resumes without replaying successful cases and that no request can cross the approved limit.

### Step 3.3: Add the predicate registry

Replace the Maya-only closed vocabulary with a versioned registry that follows the ontology. Each entry defines:

```text
predicate
family
subject_scope
object_shape
temporal_behavior
conflict_compatibility
introduced_in
```

Keep predicates lowercase `snake_case`. Reject unknown predicates at the runtime boundary, but allow a reviewed registry version to add new ones. A change in meaning or object shape creates a new version.

Exit when existing v2 claims still validate and Benchmark v1 predicates no longer require editing a hard-coded Maya-only enum.

### Step 3.4: Improve on development data

Use the deterministic failure report to make small prompt or normalisation changes. Work only against Maya and the scaled development users.

For every change:

- Freeze a new prompt version.
- Run deterministic contract tests first.
- Request approval for a bounded development run.
- Compare all extraction metrics with v2 or the last frozen run.
- Record regressions as well as improvements.
- Preserve the previous result directory.

Do not select a prompt by inspecting the frozen test users.

### Step 3.5: Produce Phase 4 input

Run the hardened extractor over the complete development histories. Every accepted claim must have:

```text
claim_id
user_id
subject_id
speaker_id
predicate and registry version
object
polarity
epistemic_status
valid_from and valid_to
time_precision
extraction confidence
one or more exact source spans
```

The persistence boundary must reject an invalid claim, wrong-user evidence, an unknown source, or a quote that is not an exact normalised substring.

Phase 3 exits when its output is reproducible, resumable, user-scoped, provenance-valid, and leakage-free. The handoff to Phase 4 is an immutable claim file plus its registry, manifest, hashes, scorecard, and known limitations.

## Phase 4: Bi-temporal versioning

Use Docker PostgreSQL with pgvector. Keep temporal behavior in tested services rather than database triggers unless a trigger is needed for an invariant that cannot be enforced safely elsewhere.

Expected code areas:

```text
migrations/
src/ingestion/
src/storage/
src/temporal/
tests/integration/
```

### Step 4.1: Add storage and migrations

Create versioned migrations and repository interfaces for:

- Immutable `SourceEvent` records.
- Exact `SourceSpan` records.
- The single canonical `Claim` entity using `claim_id`.
- `EvidenceLink` records.
- Processing attempts and extraction versions.

Add `user_id`, `memory_kind`, lifecycle `status`, `time_precision`, `transaction_from`, `transaction_to`, `belief_confidence`, and `sensitivity` where the ontology assigns them. Transaction intervals include the start and exclude the end. Valid-time boundaries remain inclusive.

Enforce foreign keys, user-consistent evidence links, stable IDs, unique idempotency keys, and safe JSON object storage. Add migration upgrade tests against a clean database.

### Step 4.2: Add idempotent ingestion and reprocessing

Implement source ingestion so the same `user_id` and idempotency key cannot create two events. Preserve raw source content and source timestamps. Record ingestion time separately.

Support:

- Retrying failed extraction.
- Reprocessing a source under a new extractor or registry version.
- Recovering from partial storage or index failure.
- Rejecting cross-user references.
- Deleting a source and recomputing or retiring every unsupported derived record.

Test duplicate requests, failed workers, out-of-order delivery, source deletion, reprocessing, and user isolation.

### Step 4.3: Add temporal version transitions

Implement deterministic transitions for candidate, confirmed, current, historical, disputed, superseded, and excluded claims.

A normal state ending becomes historical. A correction closes the old transaction interval and marks the replaced claim superseded. Never edit the earlier version in place. A null valid-time boundary means unknown or open-ended; it never means current.

Support `as_of` queries over both transaction time and valid time. Hide sources and versions recorded after the evaluation cutoff even if they describe an earlier date.

### Step 4.4: Evaluate temporal behavior

Add reviewed gold and deterministic scorers for:

- Event ordering.
- Date normalisation.
- Interval relation and overlap.
- Current-state selection.
- Historical-state selection.
- Correction-time visibility.

Test explicit corrections, repeated evidence, normal change, approximate dates, time zones, out-of-order ingestion, and two values that apply in different periods.

Phase 4 exits when the Phase 3 claim set can be replayed idempotently and produces the same current and historical states from a clean database. Its handoff is a versioned relational store and temporal query interface with a frozen scorecard.

## Phase 5: Conflict detection and resolution

Expected code areas:

```text
src/conflicts/
data/conflicts/
results/conflicts/
tests/unit/
tests/integration/
```

### Step 5.1: Generate candidate pairs

Filter by `user_id` first. Generate candidates from shared subject, predicate family, entity overlap, temporal proximity or overlap, and lexical or semantic similarity. Similarity proposes a pair; it does not create a semantic relation.

Measure candidate recall separately. A classifier cannot recover a conflict that candidate generation never returns.

### Step 5.2: Classify relations

Classify each candidate as:

```text
hard_contradiction
temporal_change
explicit_correction
refinement
source_disagreement
retraction
unresolved_ambiguity
unrelated
```

Apply deterministic time-overlap and value-compatibility checks first. Use a model only for unresolved semantic cases and only after bounded-run approval.

Persist checked directed relations such as `supports`, `contradicts`, `corrects`, `supersedes`, `refines`, `same_event_as`, `caused_by`, `hindered_by`, and `same_topic_as`.

### Step 5.3: Resolve the current belief

Use time, speaker, source scope, explicit correction language, and evidence authority. Keep extraction confidence separate from belief confidence.

The resolver may select a current claim, preserve a historical or superseded claim, or leave a dispute. It must preserve both sides and all evidence. It must exclude hypothetical, wrong-person, unsupported, and restricted claims from normal factual personalization.

### Step 5.4: Evaluate conflicts

Report:

- Candidate-pair recall.
- Conflict-pair precision, recall, and F1.
- Conflict-type accuracy.
- False contradiction rate.
- Correction-link accuracy.
- Current-belief selection accuracy.
- Preservation of historical and superseded claims.
- Unresolved-dispute accuracy.

Phase 5 exits when every resolution traces to its claims and evidence and later phases can request current, historical, disputed, or superseded views without applying their own conflict rules.

## Phase 6: Session summaries and durative memory

Expected code areas:

```text
src/summaries/
data/summaries/
results/summaries/
tests/unit/
```

### Step 6.1: Define sessions

Use deterministic boundaries:

- One conversation record is a conversation session.
- One email thread is an email session.
- One calendar event is a calendar session.
- Chat records use their declared thread or a versioned inactivity rule.

Session membership and ordering must be reproducible from source records.

### Step 6.2: Build grounded summaries

A `SessionSummary` contains its summary text, observed facts, unresolved questions, valid-time range, source IDs, and claim IDs. Generate it from claims and source evidence after conflict state is available.

Every factual summary statement must map to claims and exact sources. Preserve reports, uncertainty, correction history, and unresolved disputes. Rebuild a summary when an underlying source or claim changes.

### Step 6.3: Build durative claims

Create durative claims for roles, goals, traits, relationships, states, and patterns only when several episodes or an explicit interval support them. Check counter-evidence. One isolated event cannot establish a stable trait.

Keep durative claims in the same Claim ontology. Do not create a second memory object.

### Step 6.4: Evaluate summaries

Activate the benchmark summary track. Report gold-event and evidence precision, recall, and F1; current-versus-historical accuracy; and correction and uncertainty preservation.

Use lexical or semantic similarity only as diagnostics. If an LLM judge scores faithfulness, user-model quality, or overstatement, manually score at least 20% of the same cases and report exact agreement, weighted Cohen's kappa, Spearman correlation, and mean absolute difference.

Phase 6 exits when every summary statement has a machine-checkable claim and source path and source deletion recomputes the affected summary. Its handoff is a versioned session-index input plus summary and durative-memory scorecards.

## Phase 7: Dual retrieval

Use PostgreSQL full-text search and pgvector. Keep one retrieval request and result contract so B2, B3, and B4 differ only in enabled retrieval paths.

Expected code areas:

```text
src/retrieval/
configs/retrieval/
data/retrieval/
results/retrieval/
tests/integration/
```

### Step 7.1: Build the indexes

Create:

- An atomic index for claims, versions, dates, people, predicates, and corrections.
- A session index for grounded summaries and their event context.

Index records carry `user_id`, lifecycle status, valid time, transaction time, sensitivity, source links, embedding version, and content hash. Index writes must be idempotent and recoverable.

### Step 7.2: Plan and filter retrieval

Classify the query as current state, historical state, change over time, specific event, relationship, commitment, evidence request, or unknown.

Filter by user before lexical or vector search. Apply `as_of`, valid-time, speaker, entity, lifecycle, and sensitivity rules before fusion. Retrieve old versions and conflict neighbours when the query asks about change or corrections.

### Step 7.3: Implement the baselines

- B2 returns atomic claims only.
- B3 returns session summaries only.
- B4 fuses both result types.

Use a frozen fusion method and reranker. Return rank, component scores, final score, index version, accepted items, and rejection reasons. Connected-history expansion may add checked relations, but it cannot bypass user or time filters.

### Step 7.4: Evaluate retrieval

Report Recall@5, Recall@10, nDCG@10, mean reciprocal rank, relevant-session recall, stale-memory rate, and latency. Slice results by query type, capability, source type, difficulty, and split.

Add adversarial tests for cross-user distractors, wrong speakers, semantically similar stale claims, misleading calendar titles, and summaries that omit qualifications.

Phase 7 exits when a frozen query against a frozen index returns reproducible ranked results with enough provenance to build an evidence package. No retrieval code may read gold relevance labels at runtime.

## Phase 8: Grounded answering

Expected code areas:

```text
src/answering/
configs/answering/
results/answering/
tests/integration/
```

### Step 8.1: Build the evidence package

Convert retrieval output into the ontology's `EvidencePackage`:

```text
query_type
current_claims
historical_claims
conflicting_claims
relevant_sources
evidence_coverage
answer_allowed
```

Keep each category separate. Validate claim status, requested time, source ownership, exact spans, and rejected evidence before generation.

### Step 8.2: Add the memory answer contract

Return one of `answered`, `abstained`, `disputed`, or `partially_answered`. Answered, disputed, and partial responses require claim IDs and exact source spans. Every factual statement must follow:

```text
Answer -> Claim -> SourceSpan -> SourceEvent
```

At this phase, `answer_allowed` records evidence coverage for inspection. Phase 9 owns the separate answerability decision.

### Step 8.3: Freeze comparable answer runs

Keep `gpt-4.1-2025-04-14`, the prompt version, generation settings, dataset, and scorer fixed across comparable B2-B6 runs. Complete deterministic evidence checks before an approved model run.

Report strict and lenient answer correctness, evidence precision and recall, citation correctness, unsupported-claim rate, and current-versus-outdated fact error rate.

Phase 8 exits when factual answers are fully traceable and invalid citations or unsupported claims remain visible failures. Its handoff is a validated evidence package and answer contract for Phase 9.

## Phase 9: Answerability and abstention

Expected code areas:

```text
src/abstention/
configs/abstention/
results/abstention/
tests/unit/
tests/integration/
```

### Step 9.1: Add the answerability decision

Run answerability before answer generation. Consider:

- Evidence coverage.
- Requested valid time and `as_of` cutoff.
- Speaker and source authority.
- Wrong-person risk.
- Staleness.
- Unresolved conflicts.
- Whether the evidence can support a stable trait or causal claim.
- Whether the requested information exists in the user's history.

Return a versioned decision, confidence, reasons, accepted evidence, and rejected evidence. Retrieval success alone does not make an answer safe.

### Step 9.2: Freeze thresholds

Tune rules or thresholds only on the two development users. Do not tune for maximum abstention accuracy alone. Measure usefulness at different coverage levels.

### Step 9.3: Activate interactive answering

Run the 20 scaled interactive scenarios. Test memory use, old-versus-current beliefs, correction handling, clarification, evidence, and abstention. Do not score emotional-support style.

### Step 9.4: Evaluate B7

Report abstention precision and recall, coverage, answer accuracy on answered cases, selective risk, false answers on unanswerable cases, and unnecessary abstentions on answerable cases.

Phase 9 exits when B7 can be compared with B6 using the same retrieval and answer model and the coverage-versus-risk trade-off is explicit.

## Phase 10: Frozen ablations and scaled evaluation

Do not run the frozen test set until the component gates pass on development users and Sneha approves the complete costed run.

| Baseline | Runtime input and enabled capability |
| --- | --- |
| B0 | Current query only. |
| B1 | Full source history. |
| B2 | Phase 3 atomic claims. |
| B3 | Phase 6 session summaries. |
| B4 | Phase 7 atomic and session retrieval. |
| B5 | B4 plus Phase 4 temporal versioning. |
| B6 | B5 plus Phase 5 conflict resolution. |
| B7 | B6 plus Phase 9 answerability. |

### Step 10.1: Freeze the comparison

Freeze:

- Benchmark version and user-level split.
- Source, case, and gold hashes.
- Answer model and resolved snapshot.
- Prompt versions and hashes.
- Generation settings.
- Predicate, embedding, index, resolver, and threshold versions.
- Source ordering and `as_of` rules.
- Scorers and denominator rules.

Only the memory available to the baseline may change.

### Step 10.2: Preflight and approve

Run every deterministic validator and development smoke test first. Produce one bounded-run approval request for each independently resumable batch. Include the combined expected and maximum cost for all batches so Sneha can see the complete test cost before approving the first one.

### Step 10.3: Run B0-B7

Use immutable directories and per-case checkpoints. Finish predictions before loading scorer-only gold. Preserve execution failures without inventing answers. If a batch stops, resume only approved unfinished or provider-failed cases.

#### Current execution state

The OpenAI series is `interrupted_not_scored`. Its 100-request `gpt-4.1-mini-2025-04-14` extraction completed, but the two `gpt-4.1-2025-04-14` B0 attempts produced no valid predictions and no score. Preserve every OpenAI configuration, checkpoint, failure, prediction file, hash, and cost record unchanged.

The recorded OpenAI spend is `$0.4399284` across 149 requests, 536,405 input tokens, and 26,921 output tokens. Execution was paused before the projected `$91.2131728` hard maximum, approximately `$100`; the project did not spend `$100`. OpenAI execution remains disabled. Any future restart requires a new series, output directory, corrected JSON contract, and separate cost approval.

`qwen35-27b-fp8-v1` is a configured but unrun scaffold. It must remain immutable and become `superseded_not_run`. The next execution series is `qwen35-27b-fp8-v2`, which corrects B3-B7 semantics, scoring, concurrency, and GPU lifecycle policy. Qwen v2 regenerates extraction and all downstream artifacts; it must not reuse the completed OpenAI extraction.

### Step 10.4: Publish the scorecard

Report by baseline, capability, task, source type, difficulty, split, and answerability. Do not publish one composite score.

Step 10.4 is pending. The current demo exposes completed pilot and development evidence plus explicit missing and failed states, but no Qwen series has been run or scored. Publish only complete Qwen batches and mark every incomplete or failed batch explicitly.

The final presentation must show:

- Extraction quality that bounds every later system.
- Temporal gains or regressions from B4 to B5.
- Conflict-resolution changes from B5 to B6.
- Retrieval differences between B2, B3, and B4.
- Answer and evidence quality for each baseline.
- Coverage and risk changes from B6 to B7.
- Summary and interactive results where applicable.
- Cost, token use, execution failures, and latency alongside quality.

If a later baseline scores worse, keep the result and analyse the failure. Do not alter outputs, discard valid cases, change the scorer, or claim an improvement that the frozen comparison does not show.

### Step 10.5: Run load tests separately

Use the load corpus for ingestion acknowledgement, replay, reprocessing, retrieval latency, throughput, and duplicate-derived-memory checks. Record the machine, database settings, corpus size, concurrency, and duration.

Do not mix these measurements with quality denominators. The architecture values remain research targets until a report measures them.

Phase 10 exits when all planned baselines have final or clearly failed manifests, all scorecard denominators are visible, and another developer can reproduce the run from its hashes and instructions.

## Phase 11: Local debugging application

Build a local deployable application with FastAPI, React, and TypeScript. Use the same services and result artifacts as the evaluation pipeline; the UI must not implement its own memory rules.

The current implementation is a narrower, read-only demonstration slice. It provides a three-panel artifact explorer, run and scorecard views, deterministic bundle generation, one-command local startup, and a single Docker image. It is deployed at [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app). The hosted service has no GPU, database, provider credentials, write endpoint, or live model execution. It currently presents partial historical evidence and must be regenerated after Qwen v2 scoring.

Expected code areas:

```text
src/api/
web/
docker-compose.yml
tests/api/
tests/e2e/
```

### Step 11.1: Expose read and replay APIs

Add typed endpoints for:

- Users and source history.
- Controlled replay and processing status.
- Current, historical, disputed, superseded, and excluded claims.
- Claim relations and conflict decisions.
- Session summaries and durative claims.
- Retrieval traces and rejected evidence.
- Evidence packages, answers, and answerability decisions.
- Run manifests, scorecards, and failure records.

Every endpoint enforces `user_id`. Keep oracle and gold routes out of the runtime API.

### Step 11.2: Build the three-panel view

The left panel replays source records and shows processing state. The middle panel shows evolving memory and opens the complete version, relation, and evidence history. The right panel runs a query and shows retrieved claims, sessions, filters, rejected evidence, the evidence package, answer, and abstention decision.

Use plain labels. An evaluator should not need to understand database or embedding jargon to see what changed and why.

### Step 11.3: Add the evaluation dashboard

Read versioned result artifacts. Filter by capability, task, baseline, source type, difficulty, split, failure type, and answerability. Show coverage versus risk and component metrics without inventing an overall score.

Every displayed number links to its manifest, denominator, and cases. Missing or failed results appear as missing or failed, not zero.

### Step 11.4: Package and test locally

Provide reproducible local startup for PostgreSQL/pgvector, FastAPI, and the React application. Add API contract tests and browser tests for replay, memory inspection, conflict history, retrieval traces, evaluation filters, loading, empty, and error states.

Hosted deployment is a separate external action. Do not choose a provider, create infrastructure, or deploy without Sneha's approval.

Phase 11 exits when a new evaluator can start the project locally, replay one user, inspect every memory transition, reproduce a frozen query, and understand why B0-B7 scores differ.

## Final presentation standard

The finished repository should answer these questions directly:

- What does the benchmark test across a changing user's history?
- Which sources and facts were available at each `as_of` time?
- What did extraction miss or misread?
- Which beliefs are current, historical, disputed, or superseded?
- Which correction or evidence changed a belief?
- What did each retrieval strategy return or miss?
- Which claims support each answer?
- Why did the system answer, partially answer, dispute, or abstain?
- Which phase changed each metric?
- What did each run cost, and which cases failed to execute?

Use the scorecard and UI to show the chain from sources to answers. Do not smooth over regressions or turn research targets into achieved results.

## Completion checklist

For every document or implementation change:

- Run focused tests.
- Run `make test`.
- Run `git diff --check`.
- Verify internal paths and external links.
- Confirm no secret or `.env` content entered the diff.
- Confirm oracle and gold fields remain outside runtime inputs.
- Inspect staged and unstaged changes separately.
- Report the exact files changed and current `git status`.
- Stop for Sneha's review before committing or moving to the next phase.
