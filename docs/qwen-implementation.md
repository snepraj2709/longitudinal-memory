# Qwen3-8B vLLM execution plan

Updated on 2026-08-11. This document replaces the old Qwen3.5/H100 plan as the active Qwen execution path.

## Current direction

Run the benchmark against an OpenAI-compatible vLLM endpoint on JarvisLabs:

- Model: `Qwen/Qwen3-8B`.
- Served model alias: `qwen3-8b-vllm`.
- GPU: prefer L4 24GB.
- Fallback: A5000 24GB if it appears in current JarvisLabs inventory. On 2026-08-11, `jl gpus --json` showed L4 and A30 24GB containers in `IN2`, but no A5000 row.
- Runtime: vLLM serving `/v1/chat/completions`.
- Context: start at 8,192 tokens.
- Temperature: 0 for extraction, answers, and judge diagnostics.
- Concurrency: 1 for the first 12 compatibility requests. Increase only after measured validity and latency are acceptable.

Do not restart the old Qwen3.5/H100 private-container work. It spent INR 39.69, produced zero provider responses, and added setup complexity that does not help the benchmark right now.

## What stays

Keep these useful pieces:

- The benchmark contracts in [benchmark.md](benchmark.md) and [memory-ontology.md](memory-ontology.md).
- PostgreSQL materialization for B2-B7 contexts.
- Per-request checkpointing, ordered JSONL composition, failure records, and retry ledgers.
- The provider boundary in `src/evaluation/vllm_client.py`, which speaks the OpenAI-compatible chat completions API.
- The 12-request compatibility pack from `build_compatibility_jobs()`.

## What is inactive

These are historical or full-run-only pieces, not the current execution path:

- `qwen35-27b-fp8-v1`: preserve as `superseded_not_run`.
- `qwen35-27b-fp8-v2`: preserve old code and artifacts as historical planning work. Do not run it.
- H100/RTX-PRO6000/A100 resource policy.
- Python 3.12 conda bootstrapping for a pinned custom vLLM wheel.
- 30.9GB Qwen3.5 snapshot verification.
- Metadata-only dummy-weight engine boot.

## JarvisLabs launch

Check current inventory first:

```bash
jl status --json
jl gpus --json
jl resources --json
```

Start with L4 in `IN2` when available. Use A5000 only if `jl gpus --json` shows an available 24GB A5000 container. If A5000 is not offered, stop and choose between waiting or separately approving A30.

Exact L4 commands:

```bash
jl create --gpu L4 --spot --template pytorch --storage 60 --http-ports "6006" --region IN2 --name longitudinal-memory-qwen3-8b --yes --json
jl upload <machine_id> scripts/run_qwen_vllm.sh /home/run_qwen_vllm.sh
jl upload <machine_id> .qwen-vllm.env /home/.qwen-vllm.env
jl exec <machine_id> -- sh -lc 'python -m pip install --upgrade vllm'
jl exec <machine_id> -- sh -lc 'chmod 700 /home/run_qwen_vllm.sh && set -a && . /home/.qwen-vllm.env && set +a && nohup /home/run_qwen_vllm.sh >/home/qwen3-8b-vllm.log 2>&1 < /dev/null &'
```

Read the returned `machine_id`, then get the public endpoint:

```bash
jl get <machine_id> --json
```

Use the port 6006 HTTPS endpoint as the repo `--base-url`, with `/v1` appended if the endpoint does not already include it.

Cleanup is mandatory:

```bash
jl destroy <machine_id> --yes --json
```

## Repo pilot command

After the vLLM endpoint is reachable and paid requests are explicitly approved:

```bash
PYTHONPATH=src python -m evaluation.qwen_serverless_pilot \
  --env-file .qwen-vllm.env \
  --base-url https://<port-6006-endpoint>/v1 \
  --deployment-id <machine_id> \
  --model qwen3-8b-vllm \
  --gpu L4 \
  --storage-gb 60 \
  --confirm-paid-serverless qwen3-8b-vllm-pilot-paid-requests-approved
```

The env file must stay local and ignored:

```text
QWEN_VLLM_API_KEY=<same value as VLLM_API_KEY on the Jarvis instance>
```

The pilot writes to `results/evaluation/qwen3-8b-vllm-pilot-v1/`. It is compatibility evidence only. It is not a scored replacement for a full B0-B7 release.

## Provider boundary

The benchmark must stay provider-independent:

- Benchmark jobs produce prompts, schemas, contexts, and validators.
- Provider clients only translate those jobs into API calls.
- Qwen/Jarvis-specific values live in config, docs, scripts, and local env.
- Scoring reads sealed predictions and must not know whether the provider was JarvisLabs, OpenAI, or another OpenAI-compatible server.

Any future provider swap should require a new config and output root, not a rewrite of the benchmark contracts.

## First benchmark steps

1. Run the 12-request compatibility pilot against Qwen3-8B on L4.
2. If all 12 responses are structurally valid, run the development extraction subset and materialize B2-B7 contexts.
3. Run a small development answer batch at temperature 0, score only after predictions are sealed, then decide whether to scale.
