# Thine founder brief

## What I am building

Pratyush,

I am building a longitudinal memory evaluation project for a Personal AI system that receives conversation, email, chat, and calendar history over time. The goal is not to make a chatbot sound fluent. The goal is to test whether a memory system knows what was said, who said it, when it was true, what changed, and which evidence supports an answer.

The repository starts with benchmark contracts and synthetic user histories instead of starting with a memory architecture. It separates hidden oracle truth, source observations, extracted system memories, predictions, and evaluation gold. That separation matters because a personal AI should not treat every remembered sentence as truth.

## What works today

The current repo has a working pilot benchmark, validated Benchmark v1 and scaled datasets, a full-history B1 QA baseline, atomic extraction runs, PostgreSQL storage contracts, temporal lifecycle logic, deterministic conflict evaluation, retrieval indexes, B2-B4 retrieval baselines, evidence packages, memory-answer contracts, and answerability evaluation mechanics.

I verified the deterministic data checks during this audit. `make validate-scaled-benchmark` passed with 10 users, 100 sources, 500 QA cases, 50 summaries, and 20 interactive cases. `make validate-benchmark-v1` passed with 50 QA cases, 5 summaries, and 2 interactive scenarios. A Docker-backed retrieval gate also passed 132 tests against disposable PostgreSQL.

## The longitudinal-memory failure it currently exposes

The clearest result is the B1 full-history pilot baseline in `results/pilot/b1-full-history/`. It answered 25 QA cases with 23 strict correct answers, 24 lenient correct answers, and zero execution failures. That sounds strong, but the manual failure analysis found 12 reasoning failures, all centered on missed evidence. One case also made a wrong-date claim.

For example, the model answered the corrected date for Aryan's Bengaluru job start as May 18, 2026. That answer was right, but the citation trail missed the older May 11 report that made the correction meaningful. In a personal AI system, that is not a small detail. The product needs to know the difference between "I found the latest answer" and "I can explain what changed and why."

## How I represent and evaluate the expected answer

Pilot expected answers live in `data/pilot/evaluation/eval_answer.jsonl`. Each case stores the reference answer, acceptable paraphrases, abstention expectation, abstention reason when needed, evidence source IDs, message IDs, quotes, oracle fact IDs, and review notes.

Later benchmark files add stronger runtime/gold separation under `data/benchmark-v1/` and `data/scaled-v1/`. The ontology in `docs/memory-ontology.md` defines the trace I am trying to preserve: Answer -> Claim -> SourceSpan -> SourceEvent.

## What the baseline has revealed

The B1 result shows that full-history prompting can often get the answer text right while still losing provenance. The scaled extraction handoff is weaker: it produced 33 claims against 32 reviewed claims, with precision `0.303030`, recall `0.312500`, F1 `0.307692`, unsupported-memory rate `0.454545`, and valid-time accuracy `0.0`. That is useful evidence because it shows where the memory architecture needs to improve before product claims are safe.

## Why this matters for an ambient Personal AI product like Thine

I am not assuming Thine's internal architecture. The overlap is at the product reliability layer. An ambient Personal AI has to build memory from partial, changing human information. It needs to track speakers, relationships, corrections, stale facts, current state, and missing evidence. This repo demonstrates how I think about that problem: benchmark first, preserve evidence, measure failure categories, and avoid presenting planned behavior as shipped behavior.

## What I would improve next

The next useful improvement is to promote a tiny, conservative subset of source-grounded claims so B2-B4 can answer instead of structurally abstaining. Today, the memory-answer development run produces 24 structural abstentions and no answered memory cases. B7 also has `0/4` coverage and unnecessary abstention rate `1.0`. I would make one small promotion rule, rerun the same cases, and measure whether coverage improves without increasing unsupported answers.

## How you can inspect or run it

The repo is not founder-ready yet. There is no README, no `.env.example`, no UI, and the full local test suite currently fails six release-boundary hash tests because of uncommitted Step 10.3 changes. The safe commands today are:

`make validate-scaled-benchmark PYTHON=.venv-storage/bin/python`

`make test-retrieval-baselines PYTHON=.venv-storage/bin/python`

The first command checks the frozen scaled benchmark. The second runs the retrieval baseline suite against disposable PostgreSQL, if Docker is available.

## Request for a 30-minute technical conversation

I would value a 30-minute technical conversation to walk through the benchmark, the B1 failure analysis, and the current retrieval/answerability gap. I can show the exact files and the limits of the implementation rather than presenting this as a finished product.

**If you find my profile a match for Thine’s vision, let’s build the Longitudinal Memory Eval for Thine.**
