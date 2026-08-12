# Qwen Extraction Quality Handoff

Run inspected: `qwen3-8b-vllm-extraction-gate-v1` primary gate, archived under
`results/evaluation/_ignored_failed_runs/qwen3-8b-vllm-extraction-gate-v1-primary-quality-failed-2026-08-12/`.

The provider run itself was clean: 10 requests, 10 valid responses, no retries, no transport failures. The quality gate failed because the extraction profile still left too much room for loose mapping.

## Aggregate Failure

- Proposition precision: `0.50`
- Proposition recall: `0.5625`
- Proposition F1: `0.5294`
- Unsupported-memory rate: `0.50`
- Exact quote precision: `0.3333`
- Exact quote recall: `0.375`
- Valid-time accuracy on matched claims: `0.1111`
- Wrong-person claims: `0`
- Known bad overlap predicates: `3`

## Source-Level Findings

| Source | Main issue |
| --- | --- |
| `scaled_user_001_conversation_001` | Career goal became `work_preference`; remote preference object became `remote work`; quotes were snippets instead of the full source message. |
| `scaled_user_001_email_001` | Used `employment_start_date` instead of `job_start_date`. |
| `scaled_user_001_chat_001` | Person identity was correct, but a non-user speaker fact stayed `asserted` instead of `reported_by_other`. |
| `scaled_user_001_calendar_001` | Relocation denial was right, but valid time was missing. |
| `scaled_user_001_conversation_002` | Propositions matched, but month-level valid time was over-inferred and quotes were snippets. |
| `scaled_user_001_email_002` | `Leena` was lowercased; non-user reports were marked `asserted`; source time was missing. |
| `scaled_user_001_chat_002` | Propositions matched; valid time was missing. |
| `scaled_user_001_calendar_002` | Used `has_scheduled_event` instead of `project_review_date`. |
| `scaled_user_001_conversation_003` | Corrected start date used `employment_start_date`; the model also extracted a repeated `still` goal that gold does not score. |
| `scaled_user_001_conversation_004` | Remote object became `remote work`; the speculative move/no-plan sentence was promoted into a relocation claim; deadline valid time was missing. |

## Implementation Direction

Tighten only the Qwen gate path:

- Keep schema-constrained vLLM output.
- Add a `scaled-v1` profile note that asks Qwen to use full source-message quotes for this gate.
- Add deterministic `scaled-v1` canonicalization after provider output and before scoring:
  - map deprecated overlap predicates to the scored predicate;
  - normalize small scored objects like `remote work` to `remote`;
  - preserve `reported_by_other` for non-user speakers;
  - hydrate evidence quotes to the exact source message/calendar content;
  - fill valid time from explicit date objects or source time where the gate contract expects it;
  - drop repeated `still` goal restatements and speculative `might move someday / no plan` relocation claims for this gate.

This is not a general extraction rewrite. It is a narrow bridge between Qwen3-8B behavior and the current scaled-v1 gate contract. The longer benchmark still needs a separate design pass for durable restatements and speculative no-plan statements, because both are useful for long-horizon memory but are not scored consistently in this 10-source gate.
