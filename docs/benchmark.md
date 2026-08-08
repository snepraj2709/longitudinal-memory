# Longitudinal memory benchmark

## Purpose

This benchmark tests whether a personal AI can build reliable memory from noisy, changing information. The system must know what was said, who said it, when it was true, what changed, and which evidence supports an answer.

The benchmark is built before the full memory system. It gives each later architecture phase a fixed dataset and an isolated measure of success.

The [memory ontology](memory-ontology.md) defines the meaning of claims, evidence, valid time, lifecycle status and provenance used by this benchmark.

The benchmark must answer two kinds of question:

- What is true now?
- What was believed earlier, and why did it change?

An answer must cite its source evidence and state its confidence. If the available evidence is weak, missing, stale, or disputed, the system must abstain.

## Boundary

Evaluation starts from conversation transcripts, chat records, email, and calendar events. Audio capture and transcription quality are outside this benchmark.

The benchmark covers memory extraction, temporal reasoning, correction, retrieval, answering, and abstention. It does not score speech recognition, emotional-support quality, general helpfulness, or fluency by itself.

The design takes conceptual guidance from [ES-MemEval](https://arxiv.org/html/2602.01885v1) and its [reference repository](https://github.com/slptongji/ES-MemEval). This project reuses its capability-based evaluation, task separation, retrieval metrics, and ablation approach. It does not copy the paper's emotional-support objective, model-specific scripts, or implementation.

## What remains separate

The benchmark keeps five layers separate:

```text
Hidden oracle truth
        ↓
Source observations
        ↓
System memories
        ↓
Predictions
        ↓
Evaluation gold
```

- Hidden oracle truth records what happened in the synthetic user's life.
- Source observations are partial claims from conversations, chat, email, and calendar events.
- System memories are the claims and summaries produced by the memory system.
- Predictions are answers, summaries, retrieval results, or interactive responses produced during a run.
- Evaluation gold contains reference answers, expected memories, conflicts, timelines, and evidence labels.

Runtime systems may receive source observations or their own derived memories. They must never receive oracle events, reference answers, gold summaries, gold conflict labels, or gold evidence labels.

## Capabilities

The benchmark uses five capabilities.

| Capability | What it tests |
| --- | --- |
| Information extraction | Finds atomic facts, speakers, dates, epistemic status, and exact source spans. |
| Temporal reasoning | Orders events and distinguishes current, historical, corrected, and temporary states. |
| Conflict detection and correction | Finds related claims, classifies their relationship, preserves old beliefs, and selects the current belief when the evidence allows it. |
| Abstention | Refuses to answer when evidence is missing, weak, stale, about the wrong person, or still disputed. |
| User modelling | Distinguishes stable traits, current states, and patterns supported across time. |

Provenance is not a sixth capability. It is required and scored across every capability. A correct answer without correct evidence is incomplete.

## Evaluation tracks

### Targeted QA

QA tests all five capabilities with focused questions. Cases may require one source, several messages in one source, or evidence spread across sessions and source types.

Examples include:

- What role did the user accept?
- What was the user's goal before June?
- Which reported date is currently authoritative?
- What does the history establish about the user's work preference?
- What fact cannot be established from the available history?

QA is the first active track because it gives each capability a clear failure signal.

### Longitudinal summarization

This track asks the system to explain how a state, goal, relationship, or belief changed over time. Each summary must name the relevant events, preserve uncertainty, and cite supporting sources.

The scaled benchmark contains 25 temporal summaries and 25 user-model summaries. Correction and conflict tags may appear in either group.

Summary quality is scored against gold events and evidence first. Text similarity is only a supporting diagnostic.

### Interactive answering

This track gives the system a new query or situation after the history. It tests whether the response uses the right memories, distinguishes old and current beliefs, asks for clarification when needed, and avoids unsupported personal claims.

The scaled benchmark contains four scenarios for each capability. It does not score emotional-support style.

The summary and interactive tracks are defined now but are activated only when their required architecture phases exist. Their absence from an earlier release is not treated as a zero score.

## Releases and splits

| Release | Users | QA | Summaries | Interactive scenarios | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Pilot v0 | 1 | 25 | 0 | 0 | Implemented |
| Benchmark v1 | 1 | 50 | 5 | 2 | Planned |
| Scaled benchmark | 10 | 500 | 50 | 20 | Planned |

Pilot v0 uses Maya's current history:

- 29 oracle events
- 12 conversations
- 6 email threads
- 6 calendar events
- 25 QA cases, with five cases for each capability

Benchmark v1 expands QA to ten cases per capability. It adds five longitudinal summaries and two interactive scenarios when those tracks become active.

The scaled benchmark uses ten complete synthetic users. Two users form the development split and eight form the frozen test split. A user's history and questions must stay in one split. This prevents prompt or pipeline changes made for one part of a life history from leaking into its test questions.

The scaled QA set contains 100 cases per capability. The development split contains 100 QA cases, 10 summaries, and 4 interactive scenarios. The test split contains 400 QA cases, 40 summaries, and 16 interactive scenarios.

## Case contracts

JSONL data uses `snake_case`. Each line contains one complete JSON object. These are logical benchmark contracts; their runtime schemas are implemented separately.

### Common fields

Benchmark v1 and later store these fields in each case:

| Field | Meaning |
| --- | --- |
| `case_id` | Stable identifier that is never reused. |
| `benchmark_version` | Frozen dataset release. |
| `split` | `development` or `test`. |
| `user_id` | Synthetic user whose history is evaluated. |
| `task` | `qa`, `summarization`, or `interactive`. |
| `capability` | Primary capability used for reporting. |
| `as_of` | Evaluation cutoff. Sources or transactions after this time are hidden. |
| `difficulty` | Evidence and reasoning difficulty. |
| `failure_tags` | Known failure modes tested by the case. |
| `should_abstain` | Whether the available evidence permits an answer. |
| `evidence` | Gold source and message references. |

Cases may carry secondary capability tags, but each case has one primary capability for balanced reporting.

Pilot v0 predates this unified record shape. It keeps question fields in `eval_questions.jsonl` and gold fields in `eval_answer.jsonl`, joined by `case_id`. Its paired files remain valid for that frozen release.

### QA fields

QA adds `question`, `reference_answer`, `acceptable_answers`, and `abstention_reason`. Cases that evaluate a built memory system may also include `required_memory_ids`.

### Summary fields

Summarization adds `instruction`, `reference_summary`, `gold_event_ids`, and `required_claim_ids`. Gold events and claims are the primary scoring units.

### Interactive fields

Interactive cases add `scenario`, `initial_user_message`, `allowed_turns`, and `expected_behaviours`. Expected behaviours identify required memory use, corrections, clarification, evidence, and abstention without prescribing exact wording.

## Data construction and review

Build each synthetic history in this order:

```text
User profile
    ↓
Oracle event timeline
    ↓
Disclosure plan
    ↓
Conversation, chat, email, and calendar rendering
    ↓
Consistency checks
    ↓
Human review
```

The disclosure plan controls which facts are explicit, indirect, mistaken, hypothetical, corrected, or never disclosed. It must include fragmentation, changing states, speaker disagreement, uncertain dates, and facts that the system should not store.

Review every conflict, correction, temporal dependency, abstention case, and gold evidence set manually. Do not use one unreviewed model pass to generate the history, labels, reference answers, and judge scores.

At scale, include cross-user and wrong-person distractors. Retrieval and answering must enforce user isolation.

## Baselines and ablations

Run the benchmark against the following systems:

| Baseline | Memory available |
| --- | --- |
| B0 | Current query only. |
| B1 | Full source history. |
| B2 | Top-k atomic memories. |
| B3 | Top-k session summaries. |
| B4 | Atomic memories and session summaries. |
| B5 | Dual retrieval with temporal versioning. |
| B6 | Dual retrieval with temporal versioning and conflict resolution. |
| B7 | Full system with an answerability gate. |

Use the same answer model, resolved model version, prompt version, generation settings, dataset version, and scoring rules when comparing baselines. Change one architecture capability at a time.

Retrieval experiments must compare atomic and session granularity and record `k`. Later experiments may add turn-level results when they help explain a failure, but turn-level retrieval is not a required production index.

## Scorecard

Report a scorecard, not one composite number. A high answer score must not hide poor provenance, unsafe answering, stale retrieval, or failed correction handling.

Report each metric by task, capability, source type, difficulty, baseline, and dataset split when the denominator is large enough. A zero denominator returns `null` with a reason.

### Extraction

- Claim precision, recall, and F1
- Subject, predicate, object, polarity, speaker, and epistemic-status accuracy
- Valid-time accuracy
- Provenance-span precision and recall
- Unsupported-memory rate

### Temporal reasoning

- Event ordering accuracy
- Date-normalisation accuracy
- Interval relation accuracy and interval overlap
- Current-state and historical-state accuracy
- Temporal QA accuracy

### Conflict and correction

- Conflict-pair precision, recall, and F1
- Conflict-type accuracy
- False contradiction rate
- Correction-link accuracy
- Current-belief selection accuracy
- Preservation of superseded beliefs
- Unresolved-dispute accuracy

### Retrieval

- Recall@5 and Recall@10
- nDCG@10
- Mean reciprocal rank
- Relevant-session recall
- Stale-memory retrieval rate

### Answers and evidence

- Strict and lenient answer correctness
- Evidence precision and recall at source and message level
- Citation and exact-quote correctness
- Unsupported-claim rate
- Current-versus-outdated fact error rate

### Abstention

- Abstention precision and recall
- Answer accuracy on answered cases
- Coverage
- Selective risk
- False-answer rate on unanswerable cases
- Unnecessary-abstention rate on answerable cases

### Summaries

- Gold-event precision, recall, and F1
- Supporting-evidence precision and recall
- Current-versus-historical state accuracy
- Correction and uncertainty preservation

Lexical overlap and semantic similarity may be reported as supporting diagnostics. They do not replace event, claim, or evidence scoring.

## Judge policy

Use deterministic scoring whenever the expected value can be compared directly. An LLM judge may score semantic equivalence, summary faithfulness, user-model quality, and whether a response overstates its evidence.

Before publishing an LLM-judge score, manually score at least 20% of the same cases. Report:

- Exact agreement
- Weighted Cohen's kappa
- Spearman correlation
- Mean absolute difference

Record the judge model, prompt, settings, and calibration sample. Do not report an uncalibrated judge score.

## Run contract

Every run manifest must record:

- Baseline ID
- Benchmark version and split
- Dataset hash
- Requested and resolved model versions
- Prompt version and hash
- Generation settings
- Source ordering rule
- Start and completion times
- Successful cases and execution failures
- Output artifact hashes
- Repository commit and worktree state

Keep execution failures separate from reasoning failures. Preserve prior runs instead of overwriting them.

## Research targets

These are targets for the scaled system, not claims about current performance:

| Measure | Target |
| --- | ---: |
| Duplicate derived memories | 0 |
| Retrieval Recall@10 | Above 85% |
| Conflict-detection F1 | Above 80% |
| Unsupported-claim rate | Below 5% |
| Ingestion acknowledgement p95 | Below 250 ms |
| Retrieval p95 | Below 1.5 seconds |

Quality targets must be reported with their denominators and dataset version. Latency measurements must include the test environment and workload.

## Current artifacts

The contract does not copy current scores because measurements change as runs are added. Use the versioned artifacts instead:

- [Pilot data](../data/pilot/)
- [B1 baseline findings](../results/pilot/b1-full-history/baseline_findings.md)
- [B1 run manifest](../results/pilot/b1-full-history/run.json)
- [B1 scores](../results/pilot/b1-full-history/scores.json)
- [Phase 3 atomic-extraction v2 run](../results/phase3/atomic-extraction-v2/run.json)
- [Phase 3 atomic-extraction v2 scores](../results/phase3/atomic-extraction-v2/scores.json)

Pilot v0 has targeted QA, B1, and a Phase 3 atomic-extraction pilot. Summarization, interactive answering, retrieval, temporal versioning, conflict resolution, the answerability gate, and the scaled dataset remain planned work.

## References

- [Memory evaluation steps](memory-evaluation-steps.md)
- [Memory ontology](memory-ontology.md)
- [Project architecture](architecture-design.md)
- [ES-MemEval paper](https://arxiv.org/html/2602.01885v1)
- [ES-MemEval reference repository](https://github.com/slptongji/ES-MemEval)
