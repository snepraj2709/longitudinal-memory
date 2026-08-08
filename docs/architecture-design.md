
## Product description
a backend memory system for a personal AI that continuously receives noisy information from conversations, chats, email and calendar events.

## 
Start with the benchmark, not the memory architecture

Use the [benchmark contract](benchmark.md) for releases, tasks, cases, baselines and scoring. Use the [memory ontology](memory-ontology.md) for claims, evidence, time, lifecycle and provenance.

## Requirement
The system must determine:

What is worth remembering?
Which memories are still true?
What changed over time?
Which new statements correct or contradict older statements?
What evidence should be retrieved for a current question?
When is the available evidence too weak to answer?
How can millions of events be processed without duplicates, missing memories or inconsistent indexes?

The final system should answer both:

“What is the user’s current career goal?”

and:

“What did the user believe about their career in January, and what caused that belief to change?”

Every answer must contain supporting source events, temporal validity and confidence—or explicitly abstain

## Core user scenario

Generate a synthetic year-long history:

```
January:
“I want to remain at Zamp for another year.”

March:
“I am beginning to consider AI startups.”

April:
“I want to build Orion before considering another job.”

June:
“Orion is not retaining users. I may join a company working on personal AI.”

August:
“I want to join Thine and work on longitudinal memory.”

Later correction:
“I started considering AI startups in February, not March.”
```

The system must preserve all statements while producing:

```
{
  "current_goal": "Join a personal-AI company and work on longitudinal memory",
  "valid_from": "2026-08",
  "previous_states": [
    {
      "goal": "Remain at Castler",
      "valid_from": "2026-01",
      "valid_until": "2026-02"
    },
    {
      "goal": "Explore AI startups",
      "valid_from": "2026-02",
      "valid_until": "2026-03",
      "corrected_at": "2026-08"
    },
    {
      "goal": "Build Orion",
      "valid_from": "2026-04",
      "valid_until": "2026-06"
    }
  ],
  "supporting_events": ["event_17", "event_43", "event_91"],
  "confidence": 0.89
}
```

## Required system architecture

```
Conversations, email, calendar and chat
                  ↓
          Idempotent ingestion API
                  ↓
             Event queue
                  ↓
      Extraction and identity workers
                  ↓
       Candidate memory generation
                  ↓
 Correction and contradiction resolver
                  ↓
 ┌────────────────┼──────────────────┐
 │                │                  │
Raw events   Temporal facts     Vector index
 │                │                  │
 └────────────────┼──────────────────┘
                  ↓
         Hybrid retrieval planner
                  ↓
    Evidence validation and abstention
                  ↓
 Answer with provenance and confidence
 ```

## 1. Event ingestion layer

The system must accept:

```
type SourceEvent = {
  eventId: string;
  userId: string;
  idempotencyKey: string;

  source: "conversation" | "email" | "calendar" | "chat";
  occurredAt: string;
  ingestedAt: string;

  participants: string[];
  content: string;
  sourceUri?: string;
};
```

Required backend behaviour:

Duplicate requests do not create duplicate events.
Failed extraction jobs can be retried.
Poison jobs move to a dead-letter queue.
Events can be reprocessed after extraction logic changes.
One user’s data cannot be retrieved by another user.
Deleting an event removes or recalculates every derived memory.
Partial failure between the database and vector index is recoverable.

## 2. Corrective memory layer

Separate the original event from the memory inferred from it.

```
type MemoryClaim = {
  claimId: string;
  userId: string;

  subject: string;
  predicate: string;
  object: string;

  validFrom: string;
  validUntil?: string;

  recordedAt: string;
  supersededAt?: string;

  status:
    | "tentative"
    | "confirmed"
    | "disputed"
    | "superseded";

  confidence: number;
  sourceEventIds: string[];
};
```

The system must support:

New facts
Repeated evidence
Corrections
Direct contradictions
Gradual changes
Temporary states
Unresolved conflicts
User-confirmed facts
Deleted source evidence

A correction must not erase history. It must close or supersede the older memory while maintaining provenance.

### 1.1 candidate linking

Relation-aware candidate linking
When a new memory arrives, retrieve similar older memories and ask whether they have a relationship such as Changed, Cause, HinderedBy, or SameTopic. This could help propose graph edges—but those edges must be validated before persistence.

Connected-history expansion
After retrieving a relevant claim, expand through its connected component to find context that vector similarity alone might miss.

### 2. Human states exist over durations, not isolated moments

Suppose the system receives these memories:

January: Sneha starts learning backend development.
March: Sneha builds Orion.
May: Sneha starts studying agent memory.
July: Sneha begins building AI evaluation systems.
August: Sneha applies for AI Product Engineer roles.

These are individual episodic facts. But together they support a longer-lived state:

#### Durative memory:
Sneha has been moving toward AI Product Engineering
from approximately January 2026 to the present.

The paper separates memory into two layers:

Memory type	What it represents
Episodic memory	Individual events at specific times
Durative memory	Goals, interests, roles and patterns that persist across time

This matters because questions such as:

What are Sneha’s career goals?

cannot always be answered from one conversation. The answer must be inferred from evidence accumulated over months.

## 3. Hybrid retrieval layer

Implement at least four retrieval paths:

Semantic vector retrieval
Keyword/full-text retrieval
Temporal filtering
Person/entity relationship retrieval

The retrieval planner should determine whether the query asks for:

Current state
Historical state
Change over time
Specific event
Person relationship
Unresolved commitment
Supporting evidence

For a change-over-time question, the system should retrieve:

Current memory
Previous versions
Transition events
Contradictory evidence
Original sources

Do not send the top vector matches directly to the LLM.

First produce a structured evidence package:

```
type EvidencePackage = {
  queryType: string;
  currentClaims: MemoryClaim[];
  historicalClaims: MemoryClaim[];
  conflictingClaims: MemoryClaim[];
  relevantEpisodes: SourceEvent[];
  evidenceCoverage: number;
  answerAllowed: boolean;
};
```

## 4. Abstention layer
The system must decline to make personal claims when:

No relevant source exists.
Only one weak inference exists.
Contradictions remain unresolved.
Retrieved evidence concerns the wrong person.
The available evidence is outdated.
The query requires information outside the user’s history.

Example:

Question:
“Why does Sneha dislike her manager?”

Available evidence:
One conversation describing a disagreement.

Correct answer:
“I found one disagreement, but there is not enough evidence to
conclude that Sneha dislikes her manager.”

## 5. Evaluation harness

The scaled release contains 500 QA cases across five capabilities. Provenance is required and scored across every capability rather than treated as a sixth category. The [benchmark contract](benchmark.md) is authoritative for release sizes, splits, evaluation tracks and metrics.

| Capability | Example |
| --- | --- |
| Information extraction | What did the user promise during Friday’s meeting? |
| Temporal reasoning | When did the user begin considering AI startups? |
| Conflict resolution | Which resignation date is currently authoritative? |
| User modelling | Is entrepreneurship a stable interest or recent curiosity? |
| Abstention | Why does the user dislike their manager? |

Each case also identifies the evidence needed to support or abstain from an answer.

Measure:

### Retrieval metrics
Recall@5 and Recall@10
nDCG@10
Mean Reciprocal Rank
Relevant-session recall
Stale-memory retrieval rate

### Memory metrics
Current-state accuracy
Historical-state accuracy
Contradiction-detection F1
Correction-resolution accuracy
Temporal-order accuracy
Provenance completeness

### Answer metrics
Grounded-answer accuracy
Unsupported-claim rate
Abstention precision and recall
Citation/source correctness
Current-versus-outdated fact error rate

### User datasets:

Testing dataset
10 synthetic users
One year per user
10–20 events per day
Approximately 500,000 events
Manually inspectable ground truth

The following values are [research targets](benchmark.md#research-targets) for the scaled system. They are not claims about current performance:

Ingestion acknowledgement p95: below 250 ms
Retrieval p95: below 1.5 seconds
Duplicate derived memories: 0
Retrieval Recall@10: above 85%
Contradiction-detection F1: above 80%
Unsupported-claim rate: below 5%

----------------------------------

## The core flow

Three years of conversations into the system chronologically.

Data available	Extracted career state
Month 1	“Sneha wants to become a chemical engineer.”
Month 6	“Sneha is exploring product management through an EdTech internship.”
Year 1	“Sneha wants to become a marketing lead.”
Year 2	“Sneha is working toward becoming an SDE 2.”
Year 3	“Sneha is building AI products and wants to become a Product Engineer.”

The system should not store these as five competing answers. It should construct an evolving memory:

The system should not store these as five competing answers. It should construct an evolving memory:

```
Chemical Engineer
valid: Jan–May 2022
superseded by: Product Manager goal

Product Manager
valid: Jun–Dec 2022
superseded by: Marketing Lead goal

Marketing Lead
valid: Jan–Dec 2023
superseded by: SDE 2 goal

SDE 2
valid: Jan–Dec 2024
superseded by: Product Engineer goal

Product Engineer
valid: Jan 2025–present
status: current
```

The dates do not need to be manually supplied. The system should infer them from conversation timestamps and retain the source evidence.

## What the actual product should look like

The demonstration should be a three-panel application.

### Panel 1. Data replay

The left panel shows conversations entering the system over time:

Source
Conversation date
Participants
Extracted people
Candidate memories
Processing status

With controls such as:

Replay one month
Replay six months
Replay one year
Process all three years

This makes the evolution visible instead of presenting a pre-computed database.

### Panel 2. Evolving memory

The middle panel shows the memory produced from those conversations.

For example:

Career goal: Product Engineer
Status: Current
Valid from: January 2025
Confidence: 0.91
Previous state: SDE 2
Supported by: 7 conversations
Contradicting evidence: 1 conversation

Clicking the memory opens its history:

Each transition should answer:

What changed?
When did it change?
Which conversation caused the update?
Was the earlier memory incorrect, or merely outdated?
Was the new goal explicitly stated or inferred?

That last distinction matters. “I am working as an SDE” does not necessarily mean “I want to become SDE 2.”

### Panel 3. Query and evidence

The right panel lets an evaluator ask questions against the same memory store.

Current-state question

What are Sneha’s current career goals?

Response:

Sneha currently wants to grow as a Product Engineer building AI products, particularly around cognition and metacognition.

Underneath:

Current memory used
Relevant conversations
Retrieved historical memories
Confidence
Exact supporting excerpts
Any unresolved contradiction
Historical question

What were Sneha’s career goals in June 2022?

Response:

At that time, Sneha was interning at an EdTech startup and exploring product management. The evidence suggests this was an active direction, although not yet a settled long-term goal.

Longitudinal question

How have Sneha’s career goals changed over three years?

Response:

Sneha moved from a domain-led career goal toward role-led goals. She initially considered chemical engineering, explored product management and marketing, then moved into software engineering. Her current goal combines the product judgment developed in earlier roles with engineering execution: becoming a Product Engineer building AI products.

It proves that the system can synthesize a trajectory rather than retrieve isolated facts.