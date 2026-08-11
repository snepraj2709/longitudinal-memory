# Longitudinal memory benchmark

This repository evaluates whether an AI memory system can preserve changing facts, conflicting accounts, corrections, evidence, and uncertainty across conversation, email, chat, and calendar history.

## Demo

The hosted read-only demo is available at [longitudinal-memory-benchmark.up.railway.app](https://longitudinal-memory-benchmark.up.railway.app).

Start the complete read-only explorer with one command:

```bash
make demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The first run creates `.venv-demo`, installs pinned Python and web dependencies, builds the sanitized bundle, compiles React, and starts FastAPI. It does not need Docker, a database, an API key, or a model provider.

The explorer includes four guided cases: correction, conflict, changing user history, and abstention. The scorecard keeps completed development results, the interrupted OpenAI run, and the unrun Qwen series visibly separate.

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

The current Qwen path is a small `Qwen/Qwen3-8B` vLLM pilot on JarvisLabs L4, exposed through an OpenAI-compatible API. The old `qwen35-27b-fp8-v1` and `qwen35-27b-fp8-v2` plans are historical and should not run.

The 12-request Stage 1 pack is generated from runtime-only development inputs:

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_compatibility --repo-root .
```

No JarvisLabs resource, paid benchmark, OpenAI call, or Railway deployment is started by these commands. Those actions require separate approval. See `docs/qwen-implementation.md`, `docs/QWEN_JARVIS_RUNBOOK.md`, and `docs/DEMO_RUNBOOK.md`.

## Project contracts

- `docs/benchmark.md`: releases, tasks, baselines, and scoring.
- `docs/memory-ontology.md`: claims, evidence, time, lifecycle, and provenance.
- `docs/memory-evaluation-steps.md`: phased implementation plan.
- `docs/Implementation-handoff.md`: current handoff and boundaries.
