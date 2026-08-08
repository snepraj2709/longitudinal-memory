# Memory ontology

## Purpose

This document defines what a memory means in this project. It is the semantic contract shared by benchmark data, extraction, temporal reasoning, conflict resolution, retrieval, answering, and abstention.

It is not a database schema or runtime API. Current and planned fields are marked so later phases can extend the existing Phase 3 claim contract without changing its meaning.

The design takes limited guidance from the [ES-MemEval paper](https://arxiv.org/html/2602.01885v1) and its [reference repository](https://github.com/slptongji/ES-MemEval). ES-MemEval separates user profiles, event timelines, sessions, observations, evidence, and evaluation tasks. This project adapts that separation to conversations, chat, email, and calendar records. It does not adopt the paper's emotional-support ontology or copy its code.

Audio capture and transcription are outside this contract. The ontology starts after a transcript or structured source record exists.

## Separation of data layers

The project keeps five layers separate:

```text
Hidden oracle truth
        ↓
Source observations
        ↓
System memory claims
        ↓
Predictions
        ↓
Evaluation gold
```

The arrows show the evaluation boundary, not permission to pass every layer into the system.

| Layer | Meaning | Runtime access |
| --- | --- | --- |
| Hidden oracle truth | The synthetic user's actual history, including facts never disclosed. | Never available. |
| Source observations | What a conversation, email, calendar item, or chat record says. | Available according to the baseline. |
| System memory claims | Source-grounded claims produced by the memory system. | Available to memory-based baselines. |
| Predictions | Extracted claims, retrieved evidence, answers, summaries, or interactive responses produced during a run. | Produced by the system. |
| Evaluation gold | Reference claims, answers, timelines, conflict labels, summaries, and evidence annotations. | Scoring only. |

Runtime prompts, retrieval indexes, and memory stores must not receive oracle events, oracle facts, reference answers, gold summaries, gold conflict labels, or gold evidence labels. A system belief is not oracle truth, even when its evidence is strong.

## Core entities

```text
SourceEvent → SourceSpan → EvidenceLink → Claim
                                      Claim ↔ ClaimRelation ↔ Claim
                         SourceEvent + Claim → SessionSummary
             Claim + SessionSummary + SourceEvent → EvidencePackage → Answer
```

All runtime entities belong to one `user_id`. A query, relation, summary, or answer must not cross that user boundary.

### `SourceEvent`

A source event is the immutable record received from a conversation, chat, email, or calendar system. The current source records contain:

```text
source_id
source_type
user_id
created_at
ingested_at
participants
payload
metadata
content
```

`created_at` records when the source was produced. `ingested_at` records when this system received it. They are not substitutes for the time described inside the source.

### `SourceSpan`

A source span identifies the exact text that supports an extracted claim. In the current Phase 3 contract, a span contains:

```text
source_id
message_id
quote
```

`message_id` is null for a calendar record that has no message boundary. The quote must match the normalized source text exactly. Later storage may add character offsets, but the source and message reference remain authoritative.

### `Claim`

A claim is one atomic proposition attributed to a speaker and grounded in source evidence. The project uses one claim entity throughout the memory pipeline. `claim_id` is the canonical identifier. References to `memory_id` in earlier planning examples mean this same persistent claim and do not create a second entity.

Phase 3 currently requires:

```text
claim_id
subject_id
speaker_id
predicate
object
polarity
epistemic_status
valid_from
valid_to
confidence
evidence
```

Later phases add semantics for:

```text
user_id
memory_kind
status
time_precision
transaction_from
transaction_to
belief_confidence
sensitivity
```

These additions are part of the target ontology. They are not implemented fields yet.

`subject_id` names who or what the proposition concerns. `speaker_id` names who made the statement. They may differ. A claim about Aryan reported by Ankita must not become a direct statement by Aryan.

`object` remains a JSON value. The predicate registry defines whether a predicate expects a string, number, boolean, list, or structured object.

### `EvidenceLink`

An evidence link connects one claim to one source span. A claim may have several links. The current nested `evidence` records the source, message, and quote. Later conflict work adds:

```text
support_type       supports | contradicts | corrects
extraction_confidence
```

One claim may need several spans. One span may support several claims. Evidence must not be replaced with a generated explanation.

### `ClaimRelation`

A claim relation records a checked relationship between two claims. Similarity may propose a candidate pair, but it is not a persisted semantic relation by itself.

Each relation contains a source claim, target claim, relation type, confidence, and the resolver or rule version that created it. Relations are planned for Phase 5.

### `SessionSummary`

A session summary is a retrieval unit derived from source events, claims, and their evidence. It may contain:

```text
session_id
summary
observed_facts
unresolved_questions
valid_time_start
valid_time_end
source_ids
claim_ids
```

A summary is not independent truth. Every factual statement in it must trace back to claims and source evidence. If the underlying sources change or are deleted, the summary must be rebuilt.

### `EvidencePackage`

Retrieval produces a structured evidence package before answer generation:

```text
query_type
current_claims
historical_claims
conflicting_claims
relevant_sources
evidence_coverage
answer_allowed
```

The package keeps retrieval and answerability visible. It prevents the answer model from treating every retrieved item as equally relevant or current.

### `Answer`

An answer contains prose plus its decision and provenance. Its status is one of:

```text
answered
abstained
disputed
partially_answered
```

Answered, disputed, and partially answered responses require evidence. An abstained response states what the history does not establish and gives an abstention reason.

The required trace is:

```text
Answer
  → Claim
  → SourceSpan
  → SourceEvent
```

## Claim meaning

### Memory kinds

The ontology has two memory kinds.

| Kind | Meaning | Examples |
| --- | --- | --- |
| `episodic` | An event or observation tied to a particular occurrence. | Accepted an offer, attended a meeting, received project feedback. |
| `durative` | A state, goal, trait, role, relationship, or pattern supported over an interval. | Works in product, prefers remote work, has been moving toward product engineering. |

A stable trait is a durative claim with evidence across time. A current state may also be durative but short lived. A longitudinal pattern needs several supporting episodes and counter-evidence review. One isolated event is not enough to establish a trait or pattern.

### Predicate registry

Predicates are lowercase `snake_case`. The vocabulary is extensible, but it is not free form. Every benchmark release and runtime run records a registry version.

Each registry entry defines:

```text
predicate
family
subject_scope
object_shape
temporal_behavior
conflict_compatibility
introduced_in
```

Stable predicate families are:

```text
identity
role
relationship
goal
preference
belief
state
event
commitment
task
schedule
assessment
```

Current Maya predicates are examples, not a permanent closed list:

| Predicate | Family | Expected object |
| --- | --- | --- |
| `job_start_date` | event | ISO date string |
| `feels_exhausted` | state | boolean |
| `wants_to_talk_to` | goal | person ID |
| `has_scheduled_event` | schedule | title and location object |
| `explains_work_supportively` | assessment | boolean |

Two predicates that mean the same thing must not be added under different names. A changed object shape or meaning requires a new registry version.

### Polarity and epistemic status

Polarity and epistemic status answer different questions.

`polarity` says whether the proposition is positive or negative:

```text
positive
negative
```

`epistemic_status` says how the speaker presents it:

| Status | Meaning |
| --- | --- |
| `asserted` | Presented as a fact or commitment. |
| `inferred` | Derived by the system rather than stated directly. |
| `reported_by_other` | Attributed through another person. |
| `hypothetical` | A possibility, example, or imagined future. |
| `uncertain` | Expressed with doubt or incomplete confidence. |
| `denied` | Explicitly rejected or said not to be true. |
| `corrected` | Presented as a correction to an earlier claim. |

A negative claim is not automatically a denial, and a denial does not erase the earlier statement. Both must retain their evidence.

### Confidence

Phase 3 `confidence` means extraction confidence: how confident the extractor is that the claim matches the source. It does not measure truth, source authority, or the strength of the final system belief.

Later resolution adds `belief_confidence`. This score reflects the combined evidence after time, source attribution, corrections, and conflicts have been considered. Neither score may replace evidence.

## Time

### Valid time

Valid time describes when a claim applies in the user's life.

```text
valid_from
valid_to
time_precision
```

`valid_from` and `valid_to` are inclusive. A null `valid_from` means the start is unknown. A null `valid_to` means the end is unknown or open, depending on the evidence and lifecycle status. Null never means "true now."

Planned `time_precision` values are:

```text
timestamp
day
month
year
approximate
unknown
```

Relative dates must be resolved against the source timestamp and timezone. If the source only supports an approximate date, the system must preserve that uncertainty.

### Transaction time

Transaction time describes when the system held a claim version:

```text
transaction_from
transaction_to
```

The interval includes `transaction_from` and excludes `transaction_to`. `ingested_at` is the first possible transaction start, but it is not a complete transaction history.

When a correction arrives, the system closes the old transaction interval and creates the new version. It does not edit the old belief in place.

An `as_of` query applies both the evaluation cutoff and the requested valid time. Sources and claim versions recorded after the cutoff stay hidden even if they describe an earlier date.

## Lifecycle

One `status` field records the claim's place in the memory lifecycle.

| Status | Meaning |
| --- | --- |
| `candidate` | Extracted and source grounded, but not resolved for the user model. |
| `confirmed` | Accepted evidence about a non-versioned fact or completed event. |
| `current` | The active accepted version of a changing claim. |
| `historical` | An accepted claim whose valid period ended normally. |
| `disputed` | Competing evidence remains unresolved. |
| `superseded` | A correction or replacement closed this claim as the system's active belief. |
| `excluded` | The claim cannot influence the factual user model or normal personalization retrieval. |

Typical transitions are:

```text
candidate → confirmed
candidate → current
candidate → historical
candidate → disputed
candidate → excluded
current → historical
current → disputed
current → superseded
disputed → current
disputed → confirmed
disputed → superseded
disputed → excluded
```

`historical` and `superseded` are different. A historical state may have been correct for its interval. A superseded claim was replaced because later evidence corrected or refined what the system should believe.

## Conflicts and relations

### Conflict labels

| Label | Meaning |
| --- | --- |
| `hard_contradiction` | Both values cannot be true for the same subject, predicate, and overlapping time. |
| `temporal_change` | Both values can be true in different periods. |
| `explicit_correction` | A later statement directly corrects an earlier one. |
| `refinement` | A later claim adds detail without making the earlier claim false. |
| `source_disagreement` | Different speakers or records disagree without a decisive correction. |
| `retraction` | A speaker withdraws an earlier statement without necessarily supplying a replacement. |
| `unresolved_ambiguity` | The evidence supports more than one reading or value. |
| `unrelated` | The claims are similar enough to compare but have no semantic conflict. |

Conflict resolution follows this order:

```text
Candidate linking
  → temporal overlap
  → value compatibility
  → conflict classification
  → belief resolution
```

Embedding similarity may help with candidate linking. It must not decide contradiction or authority.

### Relation labels

The planned relation vocabulary is:

| Relation | Direction |
| --- | --- |
| `supports` | Supporting claim to supported claim. |
| `contradicts` | Symmetric. |
| `corrects` | Correcting claim to corrected claim. |
| `supersedes` | New active claim to replaced claim. |
| `refines` | More specific claim to broader claim. |
| `same_event_as` | Symmetric. |
| `caused_by` | Effect claim to cause claim. |
| `hindered_by` | Goal or event claim to the hindering claim. |
| `same_topic_as` | Symmetric. |

The resolver must preserve both sides of every conflict. Selecting a current belief must not delete the older claim or its evidence.

### Evidence authority

Source authority is an input to resolution, not a declaration of truth. The default order is:

```text
Relevant official record
Direct correction by the subject
Direct assertion by the subject
Firsthand assertion by a participant
Third-party report
Inference, speculation, or hypothetical statement
```

This order is defeasible. An official record may be stale, a direct statement may concern the wrong time, and two authoritative sources may still leave a dispute. The resolver must consider time, speaker, source scope, and the content of the correction.

## Promotion and data policy

Extraction records what the source says. Promotion decides whether a claim may influence the factual user model.

The system may extract denials, uncertainty, reports, corrections, and hypotheticals because they matter for later reasoning. It must exclude unsupported model output, wrong-person claims, policy-blocked claims, and hypotheticals presented as current facts from the factual user model.

An excluded observation may remain available for audit or a question about what someone said. It must not appear in normal personalization retrieval as a fact about the user.

### Sensitivity

Planned sensitivity levels are:

| Level | Meaning |
| --- | --- |
| `standard` | Ordinary facts such as work tasks, general preferences, and non-sensitive schedules. |
| `sensitive` | Personal states or details such as health, emotions, finances, relationships, and precise location. |
| `restricted` | Information that policy or user consent limits to narrow uses or blocks from general retrieval. |

Sensitivity does not change whether a claim is true. It controls retention, access, and retrieval. Restricted claims must not enter general personalization retrieval.

### Isolation and deletion

Every source, claim, relation, summary, index record, evidence package, and answer belongs to one user. Candidate linking and retrieval must filter by `user_id` before semantic or keyword search.

When a source is deleted:

1. Remove its source spans and evidence links.
2. Recompute every affected claim, relation, summary, and belief confidence.
3. Delete a derived claim that has no valid evidence left. An operational audit may keep a tombstone without the deleted content.
4. Rebuild or remove affected vector and keyword index entries.
5. Keep no generated summary or answer as substitute evidence.

If other valid evidence remains, the claim may remain after recalculation. Deletion must not leave an unsupported memory or stale index entry.

## Connection to implementation phases

| Phase | Ontology responsibility | Repository status |
| --- | --- | --- |
| Phase 1 | Define benchmark and ontology contracts. | This document and [benchmark.md](benchmark.md). |
| Phase 2 | Answer from full source history with evidence. | B1 pilot implemented. |
| Phase 3 | Extract atomic claims, epistemic status, valid time, and exact evidence. | Pilot implemented. |
| Phase 4 | Add lifecycle status and transaction-time versioning. | Planned. |
| Phase 5 | Add candidate linking, conflict labels, relations, and belief resolution. | Planned. |
| Phase 6 | Add source-grounded session summaries and durative memories. | Planned. |
| Phase 7 | Retrieve atomic claims and session summaries together. | Planned. |
| Phase 8 | Build evidence packages and grounded answers. | Planned. |
| Phase 9 | Apply a separate answerability and abstention decision. | Planned. |

Phase 3 provides the shared claim and evidence base. Phase 4 preserves changes over time. Phase 5 decides how related versions and disagreements connect. Phase 6 adds context without turning summaries into truth. Phase 7 retrieves both precise claims and broader sessions. Phases 8 and 9 use that evidence to answer, dispute, partially answer, or abstain.

Later work must extend the Phase 3 contract rather than create a parallel memory representation.

## Current references

- [Memory evaluation steps](memory-evaluation-steps.md)
- [Benchmark contract](benchmark.md)
- [Architecture design](architecture-design.md)
- [Pilot oracle](../data/pilot/oracle-event.jsonl)
- [Phase 3 atomic extraction gold](../data/phase3/atomic_extraction_gold.jsonl)
- [Phase 3 atomic extraction contract](../src/extraction/contracts.py)
- [ES-MemEval paper](https://arxiv.org/html/2602.01885v1)
- [ES-MemEval reference repository](https://github.com/slptongji/ES-MemEval)
