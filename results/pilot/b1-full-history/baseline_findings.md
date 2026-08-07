# B1 full-history baseline findings

## Run identity

- Provider: OpenAI
- Model: gpt-4.1-2025-04-14
- Prompt: full-history-v1
- Dataset hash: `82a9d32719704b1cf6ce44d66ae146b34589d89cd90a7e54b650e6d5d720439b`
- Configuration hash: `3228dd4eb6b1031232e0bba902a46db8f702e1dc2ae79a367d0b5673ea7f9906`

## Baseline result

- Total questions: 25
- Correct answers: 23
- Partially correct answers: 2
- Incorrect answers: 0
- Abstained predictions: 5
- Execution failures: 0
- Unresolved cases: 0

## Failure distribution

| Category | Cases |
| --- | ---: |
| `missed_evidence` | 12 |
| `wrong_date` | 1 |
| `used_stale_information` | 0 |
| `failed_to_apply_correction` | 0 |
| `treated_assumption_as_fact` | 0 |
| `unsupported_user_inference` | 0 |
| `should_have_abstained` | 0 |
| `abstained_despite_sufficient_evidence` | 0 |
| `invalid_citation` | 0 |

| Capability | Failed cases |
| --- | ---: |
| `extraction` | 1 |
| `temporal_reasoning` | 3 |
| `conflict_detection` | 3 |
| `user_modeling` | 5 |
| `abstention` | 0 |

Overlapping categories: user_modeling_001.

## Concrete findings

- Evidence coverage was the main failure. 12 cases omitted at least one gold-designated message. Conversation evidence was missed more often than email or calendar evidence. Some omitted messages were contextual rather than independent support.
- In `temporal_004`, the omitted conversation says "yesterday" on June 27, which points to June 26. The exact email says June 25. Its failure count therefore reflects exact gold evidence recall, not stronger date support.
- One answer used April for Maya's marketing start even though its cited offer states May 4. The `wrong_date` count is 1.
- The baseline applied the explicit Aryan and Kids Spark corrections. Stale-information and failed-correction counts are 0 and 0.
- It did not turn Maya's stated suspicions into confirmed facts, and it made no unsupported stable-trait inference. Those counts are 0 and 0.
- Abstention matched all five abstention cases. The two abstention-error counts are 0 and 0.
- Every citation passed the structural and verbatim-quote checks. The invalid-citation count is 0.
- Three correct answers cited additional non-gold evidence. Manual inspection found that the extra citations were valid and supported the answers, so they are not listed as failures.

## Failed cases

| Case | Capability | Result | Primary failure | Contributing failures | Explanation |
| --- | --- | --- | --- | --- | --- |
| `extraction_003` | `extraction` | correct | `missed_evidence` | none | The answer names Pravin correctly, but it omits the conversation that explicitly names him and relies on email text that does not name the sender. |
| `temporal_001` | `temporal_reasoning` | correct | `missed_evidence` | none | The calendar supports April 20, but the answer omits Maya’s separate statement that the examinations start on the 20th. |
| `temporal_003` | `temporal_reasoning` | correct | `missed_evidence` | none | The answer applies Aryan’s May 18 correction, but its citation trail omits the original May 11 report and the message that prompted the correction. |
| `temporal_004` | `temporal_reasoning` | correct | `missed_evidence` | none | The answer cites the exact June 25 email but omits a gold-designated conversation whose relative date points to June 26, making that omitted item less precise and one day inconsistent. |
| `conflict_002` | `conflict_detection` | correct | `missed_evidence` | none | The answer uses Pravin’s clarification correctly but omits Maya’s earlier statement that she experienced his response as a rejection. |
| `conflict_003` | `conflict_detection` | correct | `missed_evidence` | none | The answer uses the written 40/60 allocation but omits the message that framed the competing half-and-half claim. |
| `conflict_004` | `conflict_detection` | correct | `missed_evidence` | none | The answer cites Pravin’s campaign-coverage explanation but omits the exchange showing that Maya’s transfer-related explanation was only a feeling. |
| `user_modeling_001` | `user_modeling` | partial | `missed_evidence` | wrong_date | The answer captures the marketing-to-product direction, but it skips the intermediate changes and says Maya started in April even though the cited offer gives May 4. |
| `user_modeling_002` | `user_modeling` | correct | `missed_evidence` | none | The answer identifies product work correctly but omits Maya’s earlier direct statement that product was the work she wanted to grow into. |
| `user_modeling_003` | `user_modeling` | correct | `missed_evidence` | none | The answer describes the May-to-June adjustment correctly but omits the message that explicitly observes Maya seemed more settled. |
| `user_modeling_004` | `user_modeling` | correct | `missed_evidence` | none | The answer gives the broad relationship arc but omits Maya’s explicit early praise and the August exchange that separates her suspicion from Pravin’s stated reason. |
| `user_modeling_005` | `user_modeling` | partial | `missed_evidence` | none | The answer stops after Maya asked for feedback. It omits the feedback, revision work, second review, and acceptance for testing. |

## Baseline interpretation

This first B1 run is the starting measurement. It does not establish a target or say whether the score is good or bad. Later memory designs can be compared by whether they reduce the 12 missed-evidence cases and the 1 wrong-date case without introducing failures in the zero-count categories.
