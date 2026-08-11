# Longitudinal memory benchmark brief

A longitudinal memory benchmark for an ambient Personal AI that receives conversation, email, chat, and calendar history over time. It tests whether a system knows who said something, when it was true, what later changed, which account should be trusted, and whether the available evidence is strong enough to answer.

## The working demonstration

The read-only demo is available at [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app) and runs locally with `make demo`. The Railway service uses the same Docker image and needs no GPU or database.

The first screen is the benchmark itself, not a product landing page. It shows source history, memory state over time, retrieved evidence, the generated answer or abstention, and the human review. Four guided cases cover:

- a second-hand May 11 date corrected directly to May 18;
- Maya's experience of rejection versus Pravin's factual clarification;
- a career preference changing from uncertainty about marketing to a product transfer;
- a college subject that is absent from the source history and should not be guessed.

## What the benchmark has found

The full-history B1 pilot answered 23 of 25 cases strictly correctly and 24 leniently correctly, with zero execution failures. That headline is incomplete: message-level evidence recall was only 50%, and manual review found 12 missed-evidence failures. One partial answer also claimed Maya started in April while its own citation gave May 4.

That is the reliability problem I want the benchmark to expose. A Personal AI can produce plausible answer text while losing the evidence chain, historical state, or correction that makes the answer trustworthy.

The current retrieval development scorecard is also deliberately separate from answer quality. B2 and B3 achieved Recall@10 of 1.0, while B4 achieved 0.975. B6 and B7 abstained on all four cases in their small development evaluation: abstention recall was 1.0, but precision was 0.25. The system avoided false answers by refusing too much.

The Qwen3-8B development run is now complete for B0-B7. It used JarvisLabs L4, vLLM, an OpenAI-compatible endpoint, temperature 0, 930 provider requests, and recorded INR 97.5664 GPU cost. It produced 912 sealed logical predictions, zero execution failures, and a deterministic scorecard with no composite score. Its separate judge diagnostic completed 112 blinded requests but remains uncalibrated and non-authoritative.

## What is not complete

The frozen B0-B7 model comparison is not finished for every provider. The historical OpenAI Step 10.3 run is preserved as `interrupted_not_scored`. GPT-4.1-mini extraction completed, but GPT-4.1 B0 attempts produced no valid prediction or score. Recorded historical spend is `$0.4399284`; OpenAI execution remains disabled.

The old Qwen3.5 plans are historical planning evidence and must not run. Qwen3-8B has one completed development scorecard, not a final test-set claim.

## Why this matters

An ambient system sees partial, emotional, contradictory, and changing information. It needs to preserve speaker perspective without turning every statement into fact, replace stale values without erasing history, and show why an answer changed.

This project starts with those failure modes as evaluation contracts. The UI makes them inspectable; the benchmark code keeps runtime inputs, predictions, gold, oracle truth, and review records separate.

## How to inspect it

Open [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app) for the hosted demo.

For a local run:

```bash
make demo
```

Then open `http://127.0.0.1:8000`. A complete walkthrough of three cases, Qwen3-8B B0-B7 status, historical OpenAI failure evidence, and remaining benchmark work takes under ten minutes in either version.

The benchmark is ready for a focused technical review of its design, current evidence, and remaining product risks.
