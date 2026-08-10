# Frozen-run approval request

Step 10.2 is ready, but no model request has been sent.

The run has 25 resumable batches: one extraction batch for 100 synthetic sources and 24 answer batches for B0-B7 across QA, summary, and interactive tasks. It plans 4,660 requests with no retries.

The existing local `OPENAI_API_KEY` may be reused. Its value was not read into an artifact and will not be logged or committed.

## Data that would leave this machine

The extraction batch would send synthetic source text, timestamps, participant and speaker IDs, source metadata, and user display names. Answer batches would send the synthetic case prompt, `as_of`, user and case IDs, and only the memory context allowed for that baseline. Gold, oracle records, review queues, credentials, and scorer-only fields are excluded.

Users are `user_001` through `user_010`. The exact 100 source IDs and 570 case IDs are frozen in `data/evaluation/frozen-preflight-v1/runtime/transmission-plan.jsonl`.

## Models and settings

Extraction uses `gpt-4.1-mini-2025-04-14` with structured JSON, temperature 0, `store=false`, and a 1,200-token output limit. Answers use `gpt-4.1-2025-04-14` with JSON output, temperature 0, `store=false`, and a 1,000-token output limit.

The estimate uses standard uncached prices checked on 2026-08-10: GPT-4.1 input $2.00/M and output $8.00/M; GPT-4.1 mini input $0.40/M and output $1.60/M. Cached-input discounts are not assumed.

## Batch costs

| Batch | Requests | Model | Expected | Maximum |
| --- | ---: | --- | ---: | ---: |
| batch_01_extraction_all_sources | 100 | `gpt-4.1-mini-2025-04-14` | $0.2545328 | $0.3927728 |
| batch_02_B0_qa | 500 | `gpt-4.1-2025-04-14` | $1.3947160 | $4.3947160 |
| batch_03_B0_summary | 50 | `gpt-4.1-2025-04-14` | $0.2399400 | $0.4399400 |
| batch_04_B0_interactive | 20 | `gpt-4.1-2025-04-14` | $0.0646440 | $0.1766440 |
| batch_05_B1_qa | 500 | `gpt-4.1-2025-04-14` | $4.0947160 | $11.3947160 |
| batch_06_B1_summary | 50 | `gpt-4.1-2025-04-14` | $0.5099400 | $1.1399400 |
| batch_07_B1_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1726440 | $0.4566440 |
| batch_08_B2_qa | 500 | `gpt-4.1-2025-04-14` | $3.0947160 | $9.3947160 |
| batch_09_B2_summary | 50 | `gpt-4.1-2025-04-14` | $0.4099400 | $0.9399400 |
| batch_10_B2_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1326440 | $0.3766440 |
| batch_11_B3_qa | 500 | `gpt-4.1-2025-04-14` | $2.7947160 | $8.3947160 |
| batch_12_B3_summary | 50 | `gpt-4.1-2025-04-14` | $0.3799400 | $0.8399400 |
| batch_13_B3_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1206440 | $0.3366440 |
| batch_14_B4_qa | 500 | `gpt-4.1-2025-04-14` | $4.0947160 | $11.3947160 |
| batch_15_B4_summary | 50 | `gpt-4.1-2025-04-14` | $0.5099400 | $1.1399400 |
| batch_16_B4_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1726440 | $0.4566440 |
| batch_17_B5_qa | 500 | `gpt-4.1-2025-04-14` | $4.0947160 | $11.3947160 |
| batch_18_B5_summary | 50 | `gpt-4.1-2025-04-14` | $0.5099400 | $1.1399400 |
| batch_19_B5_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1726440 | $0.4566440 |
| batch_20_B6_qa | 500 | `gpt-4.1-2025-04-14` | $4.0947160 | $11.3947160 |
| batch_21_B6_summary | 50 | `gpt-4.1-2025-04-14` | $0.5099400 | $1.1399400 |
| batch_22_B6_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1726440 | $0.4566440 |
| batch_23_B7_qa | 500 | `gpt-4.1-2025-04-14` | $4.1947160 | $11.8947160 |
| batch_24_B7_summary | 50 | `gpt-4.1-2025-04-14` | $0.5199400 | $1.1899400 |
| batch_25_B7_interactive | 20 | `gpt-4.1-2025-04-14` | $0.1766440 | $0.4766440 |

Expected incremental cost: $32.8869328
Hard maximum incremental cost: $91.2131728
Historical spend: $0.2314404
Projected expected cumulative spend: $33.1183732
Projected hard maximum cumulative spend: $91.4446132

Each successful response is checkpointed immediately. A successful case is never replayed. Provider failures are preserved, and this plan allows no automatic retry.

## Approval needed

Please approve both remaining facts before Step 10.3 starts:

1. Send the listed synthetic runtime fields to OpenAI.
2. Allow paid execution up to the $91.2131728 incremental hard cap ($91.4446132 cumulative including historical spend).

Credential reuse is already approved. A provider call will not be made until the transmission and spend approvals are explicit.
