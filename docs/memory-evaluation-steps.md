# Task

Build a corrective longitudinal-memory system 

## Start with the benchmark, not the memory architecture

The important sequence:

```
Hidden structured truth
        ↓
Synthetic conversations, emails and calendar events
        ↓
Gold memories, conflicts, timelines and questions
        ↓
Naive baseline
        ↓
Memory architecture
        ↓
Ablations and evaluation
```

Follow a benchmark-first structure: user profiles expand to temporally and causally connected event timelines, which are then rendered in multi-session conversations. 

---

# 1. Hidden “oracle history”

This is the actual life of the synthetic user as structured data before generating any natural-language conversations.

```json
{
  "event_id": "event_017",
  "type": "goal_change",
  "valid_from": "2025-05-12",
  "valid_to": "2025-09-03",
  "entities": ["user", "product_management"],
  "facts": [
    {
      "subject": "user",
      "predicate": "career_goal",
      "object": "product_manager"
    }
  ],
  "caused_by": ["event_012"],
  "superseded_by": "event_031"
}
```

The oracle should contain:

- Stable facts
- Temporary states
- Changing goals
- Corrections
- Simultaneously conflicting reports
- Relationships between people
- Events with uncertain dates
- Facts never disclosed to the system
- Facts disclosed indirectly
- Hypothetical statements that must not become memories

The oracle is the source of truth. Conversations, emails and calendar events are only **partial observations** of that truth.

## Start small

Before generating 40–60 conversation, start with miniature version:

```
8 conversations
3 emails
3 calendar events
2 speakers besides the user
5 changing beliefs
5 explicit contradictions
25 evaluation questions
```

Get the full pipeline working on this first. Then scale it to the complete history.

This gives faster feedback and makes failures understandable.

---

# 2. Separate truth, observations and system beliefs

These are three different layers.

```
Truth
What actually happened

Observation
What a speaker, email or conversation claims happened

System belief
What your memory system currently believes happened
```

For example:

```
Truth:
User joined Acme on 15 January.

Conversation:
User says, “I joined around February.”

Later email:
Employment letter says 15 January.

System belief:
Current best-supported date is 15 January,
but an older February belief is preserved.
```

Do not directly transform source text into “truth.”

That distinction is essential because several speakers may:

- Be mistaken
- Disagree
- Speak hypothetically
- Report hearsay
- Correct themselves
- Describe an earlier belief rather than a current one

---

# 3. Use a relational bi-temporal model

Use PostgreSQL with `pgvector`  for this data stage

> Given the messy, incomplete and contradictory human information, can the system correctly determine what was said, who said it, when it was true, whether it changed, and what it should believe now?
> 

That is primarily a **reasoning and modelling problem**, not a graph-traversal problem.

A graph database would add unnecessary infrastructure complexity without resolving the underlying memory problem.

Use model graph-like relations with relational tables and explicit claim links.

## Core tables

### `sources`

```
source_id
source_type          conversation | email | calendar
session_id
speaker_id
created_at           when the source was produced
ingested_at          when the system received it
raw_content
metadata
```

### `source_spans`

```
span_id
source_id
start_offset
end_offset
verbatim_text
message_id
speaker_id
```

### `memory_claims`

```
memory_id
subject_id
predicate
object_json
polarity
epistemic_status
valid_from
valid_to
transaction_from
transaction_to
confidence
status
created_by_model
```

Useful `epistemic_status` values:

```
asserted
inferred
reported_by_other
hypothetical
uncertain
denied
corrected
```

### `memory_evidence`

```
memory_id
span_id
support_type         supports | contradicts | corrects
extraction_confidence
```

### `memory_relations`

```
from_memory_id
to_memory_id
relation_type
```

Possible relations:

```
contradicts
supersedes
corrects
refines
caused_by
same_event_as
supports
```

### `session_summaries`

```
session_id
summary
observed_facts
unresolved_questions
valid_time_start
valid_time_end
embedding
```

## Bi-temporal meaning

```
valid time:
When was the fact true in the user’s life?

transaction time:
When did the memory system hold this version as its belief?
```

When a correction arrives, do not edit the old row in place.

```
Old memory:
transaction_to = correction ingestion time
status = superseded

New memory:
transaction_from = correction ingestion time
status = current
```

`ingested_at` is the first transaction timestamp, but a proper bi-temporal model needs both `transaction_from` and `transaction_to`.

---

# 4. Treat contradiction detection as classification, not similarity

Two semantically different statements are not necessarily contradictions.

```
“I wanted to become a product manager in March.”

“I want to remain an engineer now.”
```

These are compatible if their valid-time intervals do not overlap.

The conflict classifier should distinguish:

| Type | Example |
| --- | --- |
| Hard contradiction | “I live in Delhi” vs “I live in Mumbai” at the same time |
| Temporal change | PM goal in March, engineering goal in August |
| Explicit correction | “I joined in February—actually, it was January” |
| Refinement | “I work in technology” → “I am a product engineer” |
| Source disagreement | User says January; colleague says February |
| Retraction | “Ignore what I said earlier” |
| Unresolved ambiguity | Two dates with no authoritative correction |

Use hybrid pipeline:

```
1. Candidate generation
   Find claims sharing subject + predicate or entity cluster

2. Temporal overlap check
   Determine whether the claims refer to the same period

3. Value compatibility
   Determine whether both values can be true simultaneously

4. Conflict classification
   contradiction | update | correction | refinement | unrelated

5. Belief resolution
   current | historical | disputed | superseded
```

The LLM can classify candidate pairs, but deterministic rules should enforce temporal overlap and preservation of prior versions.

---

# 5. Build dual-granularity retrieval

Session-level retrieval performed better than turn-level and round-level retrieval as per [ES Eval Paper](https://arxiv.org/html/2602.01885v1) for its dialogue domain because important information was sparsely distributed across multiple turns. However, it also found redundancy and imperfect ranking quality. ([arXiv](https://arxiv.org/html/2602.01885v1))

Use two indexes:

```
Atomic memory index
Precise claims, dates, goals, relationships and corrections

Session index
Conversation summaries retaining narrative and causal context
```

## Adaptive retrieval

```
Direct factual question
→ primarily atomic memories

“How has the user’s career direction changed?”
→ session summaries + goal claim history

“What is the user’s current goal?”
→ current claim + superseded claims + correcting evidence

“Why did the user become anxious about the promotion?”
→ sessions + causally connected events + atomic memories

“What did Alice say about the move?”
→ evidence filtered by speaker
```

A practical retrieval pipeline:

```
Query
  ↓
Query classifier
  ↓
Atomic retrieval
Session retrieval
Temporal/graph expansion
  ↓
Reranking
  ↓
Evidence bundle
```

Do not rely only on embedding similarity. Combine:

- Semantic similarity
- BM25 or keyword matching
- Time filters
- Speaker filters
- Claim status
- Entity overlap
- Graph-neighbour expansion
- Source authority

---

# 6. Make evidence a first-class output

The answerer should not return only prose.

```json
{
  "status": "answered",
  "answer": "The user's current goal is to remain an AI software engineer.",
  "confidence": 0.88,
  "belief_status": "current",
  "evidence": [
    {
      "source_id": "conversation_042",
      "message_id": "message_008",
      "speaker": "user",
      "quote": "I realised I want to remain on the engineering path.",
      "valid_time": "2026-06-10"
    }
  ],
  "historical_beliefs": [
    {
      "value": "product manager",
      "valid_from": "2026-03-02",
      "valid_to": "2026-06-10"
    }
  ]
}
```

Other possible statuses:

```
answered
abstained
disputed
partially_answered
```

Every answer should be traceable through:

```
Answer
  → memory claim
  → source span
  → original conversation/email/calendar item
```

---

# 7. Implement abstention as a separate decision

Do not merely prompt the final model with “say you don’t know.”

ES-MemEval found that retrieval could reduce abstention performance for some commercial models, apparently because retrieved but insufficient content encouraged confident answers. ([arXiv](https://arxiv.org/html/2602.01885v1))

Create an explicit answerability gate before generation.

## Abstain when

- No supporting evidence was retrieved
- Evidence does not answer the requested time period
- Only another speaker asserted the fact
- Two current claims remain unresolved
- Evidence is indirect and below the confidence threshold
- The question asks for a stable trait from one isolated incident
- The system lacks information after a specified date
- Retrieval found related information but not the requested fact

Example:

```json
{
  "status": "abstained",
  "reason": "insufficient_temporal_evidence",
  "answer": "The history does not establish where the user lived after June 2025.",
  "evidence": []
}
```

Build abstention questions deliberately:

```
Missing fact
Wrong speaker
Unknown current state
Unresolved contradiction
Future event
Unsupported causal claim
Over-specific date
Trait inference from insufficient observations
```

---

# 8. Evaluate components and the complete system separately

A single QA score will conceal the failing architecture.

## A. Extraction

Measure:

- Claim precision, recall and F1
- Subject/predicate/object accuracy
- Valid-time accuracy
- Speaker attribution accuracy
- Epistemic-status accuracy
- Provenance-span precision and recall
- Unsupported-memory rate

A memory should count as fully correct only when the claim **and its source evidence** are correct.

## B. Temporal reasoning

Measure:

- Event ordering accuracy
- Date-normalisation accuracy
- Interval relation classification
- Current-state accuracy
- Historical-state accuracy
- Temporal QA accuracy
- Valid-time interval overlap or IoU

Test questions such as:

```
What was the user’s goal in April?
What changed after the conference?
Which happened first?
How long did the belief remain active?
What did the system believe before the correction arrived?
```

## C. Conflict detection

Measure:

- Conflict-pair precision, recall and F1
- Conflict-type classification
- False contradiction rate
- Correction-link accuracy
- Current-belief selection accuracy
- Preservation of superseded beliefs
- Unresolved-dispute accuracy

## D. User modelling

Separate three categories:

```
Stable traits
Vegetarian, prefers remote work

Current states
Currently preparing for an interview

Longitudinal patterns
Repeatedly delays decisions when evidence is ambiguous
```

Measure:

- Profile slot precision/recall
- Current-state accuracy
- Trajectory-event F1
- Supporting-evidence precision
- Unsupported trait inference rate
- Counter-evidence sensitivity
- Stable-trait versus temporary-state confusion

## E. Abstention

Measure more than abstention accuracy:

- Abstention precision
- Abstention recall
- Answer accuracy on answered cases
- Coverage: percentage of questions answered
- Selective risk: error rate as coverage increases
- False-answer rate on unanswerable questions
- Incorrect-abstention rate on answerable questions

The best system is not the one that abstains most. It should maximise correctness at useful coverage.

---

# 9. Use three evaluation formats

The paper uses QA, summarization and dialogue generation because each reveals different memory failures. ([arXiv](https://arxiv.org/html/2602.01885v1))

Adapt that structure.

## Task 1: Targeted QA

Best for isolated capability measurement.

```
Extraction
“What company did the user join?”

Temporal
“What was the user’s goal before June?”

Conflict
“Which date is currently considered correct?”

Abstention
“What university did the user attend?”

User modelling
“How did the user’s career preferences evolve?”
```

## Task 2: Longitudinal summarization

Example:

```
Summarize how the user’s career goals changed over the year.
Include the key turning points and supporting sources.
```

Evaluate extracted events rather than only text similarity.

The paper uses event-level precision, recall and F1 alongside lexical and LLM-based summary evaluation. ([arXiv](https://arxiv.org/html/2602.01885v1))

## Task 3: Interactive answering

Give the agent a new query or situation and test whether it uses memory appropriately.

Example:

```
User: I am considering leaving engineering again. What patterns do you notice?
```

Check whether the response:

- Uses relevant history
- Distinguishes old and current beliefs
- Avoids unsupported psychological claims
- Provides evidence
- Asks for clarification when necessary

---

# 10. Build baselines before proposed architecture

Run every evaluation against these systems:

```
B0: No memory
Current query only

B1: Full history
All conversations in context

B2: Atomic retrieval only
Top-k extracted memories

B3: Session retrieval only
Top-k session summaries

B4: Dual retrieval
Atomic + session-level retrieval

B5: Dual retrieval + temporal versioning

B6: Dual retrieval + temporal versioning + conflict resolution

B7: Full system + answerability gate
```

Use the same answer model across all baselines so that architecture changes are isolated.

The ES Eval paper found retrieval improved factual consistency and summarization but was weak on temporal reasoning and user modelling. It found that longer context was not automatically better, especially for smaller models. ([arXiv](https://arxiv.org/html/2602.01885v1))

Primary research questions are:

```
Does extraction improve over full-history prompting?

Does session retrieval help user modelling more than atomic retrieval?

Does temporal filtering improve current-belief accuracy?

Does conflict resolution outperform asking the LLM to reconcile evidence?

Does the answerability gate improve reliability without destroying coverage?

Does adding more retrieved context eventually reduce performance?
```

---

# 11. Generate the dataset from controlled templates

Use structured generation pipeline:

```
User profile
     ↓
Oracle event timeline
     ↓
Disclosure plan
     ↓
Source rendering
     ↓
Consistency validator
     ↓
Human review
```

## Disclosure plan

For each oracle fact, specify how it appears:

```json
{
  "fact_id": "fact_019",
  "disclosures": [
    {
      "source": "conversation_014",
      "speaker": "user",
      "style": "implicit",
      "accuracy": "approximately_correct"
    },
    {
      "source": "email_007",
      "speaker": "manager",
      "style": "explicit",
      "accuracy": "correct"
    },
    {
      "source": "conversation_029",
      "speaker": "user",
      "style": "correction",
      "corrects": "conversation_014"
    }
  ]
}
```

This gives intentional control over:

- Fragmentation
- Contradictions
- Speaker reliability
- Temporal evolution
- Information density
- Evidence distribution

The paper generated synthetic timelines and conversations with model assistance but also used human review because synthetic conversations can drift from realistic dynamics. ([arXiv](https://arxiv.org/html/2602.01885v1))

At minimum, manually review every:

- Conflict
- Correction
- Temporal dependency
- Abstention question
- Gold evidence set

Do not use the same model once to generate the history, labels, reference answers and judge scores without review. That creates correlated errors.

---

# 12. Create the evaluation set manually before scaling

For the pilot dataset, create approximately:

| Capability | Questions |
| --- | --- |
| Extraction | 10 |
| Temporal reasoning | 10 |
| Conflict detection | 10 |
| User modelling | 10 |
| Abstention | 10 |

Each case should contain:

```json
{
  "case_id": "temporal_007",
  "capability": "temporal_reasoning",
  "question": "What was the user's career goal immediately before June 2026?",
  "reference_answer": "Product manager",
  "acceptable_answers": ["product management"],
  "evidence_source_ids": [
    "conversation_011",
    "conversation_022"
  ],
  "required_memory_ids": [
    "memory_031",
    "memory_044"
  ],
  "should_abstain": false,
  "difficulty": "multi_session",
  "failure_tags": [
    "goal_change",
    "historical_state"
  ]
}
```

Use evidence annotations as the principal ground truth, not only reference-answer text.

---

# 13. Calibrate LLM-as-judge evaluation

Use deterministic metrics wherever possible.

```
Dates → exact comparison
Claim extraction → structured matching
Retrieval → Recall@k and nDCG@k
Conflict labels → classification F1
Evidence → precision and recall
Abstention → coverage and selective risk
```

Use an LLM judge for:

- Semantic equivalence
- Summary faithfulness
- User-model quality
- Whether a generated answer overstates evidence

The paper evaluates retrieval with Recall@k and nDCG@k, QA with overlap, semantic and judge-based metrics, and validates its LLM judge against human ratings using agreement and correlation measures. ([arXiv](https://arxiv.org/html/2602.01885v1))

Manually score approximately 20% of evaluation set and compare the human and LLM judge using:

```
Exact agreement
Weighted Cohen’s kappa
Spearman correlation
Mean absolute difference
```

Do not report an LLM-judge score without showing judge calibration.

---

# 14. Build a debugging UI, not merely a chatbot

The UI should expose the system’s reasoning artifacts.

## Screens

### Timeline

```
January ── joined Acme
March ─── considered PM role
June ──── returned to engineering goal
August ── corrected joining date
```

### Belief inspector

```
Career goal

Current:
AI Software Engineer
Valid from: June 2026

Historical:
Product Manager
Valid: March–June 2026

Evidence:
Conversation 12
Conversation 31
```

### Conflict view

```
Claim A: Joined on 1 February
Claim B: Joined on 15 January

Classification: explicit correction
Current belief: 15 January
```

### Query inspector

Show:

- Atomic memories retrieved
- Sessions retrieved
- Reranker scores
- Temporal filters
- Rejected evidence
- Final answer
- Abstention decision

### Evaluation dashboard

Show:

```
Overall score
Score by capability
Score by source type
Score by retrieval strategy
Score by difficulty
Failure taxonomy
Coverage versus accuracy
```

This UI will improve both evaluator clarity and development speed.

---

# 15. Test the dangerous cases first

## Unit tests

- Relative-date normalisation
- Time-zone handling
- Interval overlap
- Negation detection
- Correction handling
- Current-belief resolution
- Multiple-speaker attribution
- Evidence-span preservation
- Abstention thresholds

## Integration tests

```
Conversation
→ extraction
→ temporal storage
→ conflict creation
→ retrieval
→ grounded answer
```

## Adversarial cases

- Assistant says something about the user that the user never confirmed
- User quotes another person
- User describes a hypothetical future
- User says “I used to…”
- Two speakers use the same name
- Calendar title is misleading
- Email summary omits a qualification
- Explicit correction occurs months later
- Two beliefs are both valid in different contexts
- Retrieval returns a semantically similar but temporally incorrect memory

Aim for at least one regression test per real bug you find.

---

# 16. Recommended implementation order

## Phase 1 — Benchmark contract

Deliver:

```
docs/benchmark.md
docs/memory-ontology.md
data/pilot/oracle.json
data/pilot/questions.json
```

Define all labels and metrics before writing the memory system.

## Phase 2 — Full-history baseline

Build the simplest end-to-end answerer:

```
Question + full history → answer + evidence
```

This gives you the first benchmark number.

## Phase 3 — Atomic extraction

Implement structured-output extraction with provenance and valid-time inference.

Do not add retrieval yet. Evaluate extraction alone.

## Phase 4 — Bi-temporal versioning

Add transaction time, corrections and historical belief preservation.

## Phase 5 — Conflict detection

Implement candidate generation, conflict type classification and current-belief resolution.

## Phase 6 — Session summaries

Generate summaries with explicit event lists, unresolved conflicts and source links.

## Phase 7 — Dual retrieval

Add atomic and session indexes, fusion and reranking.

## Phase 8 — Grounded answering

Require answers to cite memory IDs and source spans.

## Phase 9 — Abstention gate

Add answerability classification and coverage-versus-accuracy evaluation.

## Phase 10 — Ablations

Run all baselines with frozen data, prompts and models.

## Phase 11 — UI and deployment

Expose the timeline, conflicts, evidence and evaluation results.

---

# 17. Sample Repository structure

Treat this as an inspiration to keep the repository files organized and structured, its not set in stone, don’t treat it as hard and fast rule to just copy this.

Build an inspired folder struture grounded in the current project as it evolve and build over time. 

```
longitudinal-memory/
├── data/
│   ├── pilot/
│   │   ├── oracle.json
│   │   ├── conversations.json
│   │   ├── emails.json
│   │   ├── calendar.json
│   │   └── eval_cases.json
│   └── full/
├── docs/
│   ├── benchmark.md
│   ├── architecture.md
│   ├── memory-ontology.md
│   ├── evaluation.md
│   └── tradeoffs.md
├── src/
│   ├── generation/
│   ├── ingestion/
│   ├── extraction/
│   ├── temporal/
│   ├── conflicts/
│   ├── retrieval/
│   ├── answering/
│   └── evaluation/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── adversarial/
├── ui/
├── migrations/
└── scripts/
```

Use Python with Pydantic for the extraction and evaluation pipeline, PostgreSQL/pgvector for storage, and Next.js for the debugging UI.

The authors’ repository contains the EvoEmo dataset, execution scripts for full-history and RAG variants, and separate scripts for different retrieval granularities. Use it as an evaluation reference rather than copying its system architecture blindly. ([GitHub](https://github.com/slptongji/ES-MemEval))

---

# 18. Commit Guide

Make the commit history reflect the reasoning and logic:

```
docs: define memory capabilities and benchmark contract

data: add pilot oracle timeline and disclosure plan

data: add capability-tagged gold evaluation cases

baseline: implement full-history grounded QA

memory: extract atomic claims with source provenance

memory: add valid and transaction-time versioning

memory: classify corrections and conflicting claims

retrieval: add atomic memory index

retrieval: add session summary index and fusion

answering: require evidence-backed responses

answering: add explicit abstention policy

eval: add component metrics and end-to-end benchmark

eval: add architecture ablation runner

test: add temporal and contradiction adversarial cases

ui: add timeline and belief inspector

ui: add retrieval and evaluation dashboard

deploy: add production database and deployment config

docs: report results, failures and architecture trade-offs
```

Avoid one enormous “complete project or phase wise” commit.

---

# 19.  Trade-offs compilation

## Relational versus graph storage

Why PostgreSQL was sufficient for the benchmark and where graph traversal would become valuable.

## Atomic versus session retrieval

Atomic memories improve precision; sessions preserve context and causal meaning.

## LLM versus deterministic conflict resolution

LLMs understand semantic incompatibility; rules provide temporal consistency and reproducibility.

## Full history versus RAG

Full history preserves recall but adds noise and cost. RAG reduces context but can omit or mis-rank relevant evidence.

## Updating versus preserving beliefs

Overwriting is simpler but destroys historical reasoning. Versioning increases complexity but supports longitudinal queries.

## Aggressive answering versus abstention

Higher coverage may increase unsupported answers. Conservative abstention improves trust but may reduce usefulness.

## Synthetic realism versus controllability

Synthetic data provides exact truth and reproducibility but may not represent natural conversation distributions.

---

# First steps

First development session should have four things:

```
1. A 12–20 event oracle timeline
2. Eight manually written source artifacts
3. Fifty capability-tagged evaluation questions
4. A full-history baseline returning answers with evidence
```

Then freeze the pilot dataset.

Only after the baseline is measurable then implement extraction, temporal versioning and retrieval.

The key learning loop is:

```
Implement one capability
        ↓
Run its isolated evaluation
        ↓
Inspect failures manually
        ↓
Add one architectural intervention
        ↓
Run an ablation
        ↓
Document what changed and why
```