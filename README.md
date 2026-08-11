# Longitudinal memory benchmark

This repository evaluates whether an AI memory system can preserve changing facts, conflicting accounts, corrections, evidence, and uncertainty across conversation, email, chat, and calendar history.

## Demo

The hosted read-only demo is available at [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app).

Start the complete read-only explorer with one command:

```bash
make demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The first run creates `.venv-demo`, installs pinned Python and web dependencies, builds the sanitized bundle, compiles React, and starts FastAPI. It does not need Docker, a database, an API key, or a model provider.

The explorer includes four guided cases: correction, conflict, changing user history, and abstention. The scorecard keeps completed Qwen3-8B development results, older development component scores, the interrupted OpenAI run, and the historical unrun Qwen3.5 series visibly separate.

Run the focused local checks with:

```bash
make test-demo
```

## Docker

Build and run the same single service:

```bash
docker compose up --build demo
```

The image contains only FastAPI, the compiled web app, and the sanitized demo bundle. It has no database or provider credentials.

## Qwen benchmark

The current Qwen path is `Qwen/Qwen3-8B` served by vLLM on JarvisLabs L4 through an OpenAI-compatible API. The development run `qwen3-8b-vllm-dev-v1` completed B0-B7 with 930 provider requests, zero execution failures, a sealed deterministic scorecard, and separate uncalibrated judge diagnostics. The old `qwen35-27b-fp8-v1` and `qwen35-27b-fp8-v2` plans are historical and should not run.

The 12-request Stage 1 pack is generated from runtime-only development inputs:

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_compatibility --repo-root .
```

No JarvisLabs resource, paid benchmark, OpenAI call, or Railway deployment is started by these commands. Those actions require separate approval. The latest successful Qwen run artifacts are committed under `results/evaluation/qwen3-8b-vllm-dev-v1/`. See `docs/qwen-implementation.md`, `docs/QWEN_JARVIS_RUNBOOK.md`, and `docs/DEMO_RUNBOOK.md`.

## Project contracts

- `docs/benchmark.md`: releases, tasks, baselines, and scoring.
- `docs/memory-ontology.md`: claims, evidence, time, lifecycle, and provenance.
- `docs/memory-evaluation-steps.md`: phased implementation plan.
- `docs/Implementation-handoff.md`: current handoff and boundaries.
