# Longitudinal memory benchmark

This repository evaluates whether an AI memory system can preserve changing facts, conflicting accounts, corrections, evidence, and uncertainty across conversation, email, chat, and calendar history.

## Demo

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

The new comparison series is `qwen35-27b-fp8-v1`. Its model, Hugging Face revision, vLLM revision, prompts, schemas, sampling, seed, GPU type, and budget gates are pinned in `configs/evaluation/qwen35_27b_fp8_v1.json`.

The 12-request Stage 1 pack is generated from runtime-only development inputs:

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_compatibility --repo-root .
```

No JarvisLabs resource, paid benchmark, OpenAI call, or Railway deployment is started by these commands. Those actions require separate approval. See `docs/QWEN_JARVIS_RUNBOOK.md` and `docs/DEMO_RUNBOOK.md`.

## Project contracts

- `docs/benchmark.md`: releases, tasks, baselines, and scoring.
- `docs/memory-ontology.md`: claims, evidence, time, lifecycle, and provenance.
- `docs/memory-evaluation-steps.md`: phased implementation plan.
- `docs/Implementation-handoff.md`: current handoff and boundaries.
