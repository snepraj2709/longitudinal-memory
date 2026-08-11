# Qwen JarvisLabs runbook

The decision-complete implementation and execution contract is [qwen-implementation.md](qwen-implementation.md). That document controls model and runtime revisions, B0-B7 semantics, PostgreSQL materialization, request counts, budget gates, checkpointing, cleanup, scoring, commits, and deployment.

## Series status

- `qwen35-27b-fp8-v1` is a historical, unrun scaffold. Preserve it as `superseded_not_run`; do not start its runner.
- `qwen35-27b-fp8-v2` is the only permitted execution series, but paid execution is paused.
- `openai-gpt41-v1` remains `interrupted_not_scored` and must not be resumed or used as Qwen input.

Eight setup attempts on 2026-08-11 spent INR 39.69 and produced no provider response. All instances were destroyed. A new GPU attempt requires fresh approval; the earlier execution approval is no longer valid.

## Resource policy

Use one H100 80 GB spot container in `IN2` when its current spot rate is at or below INR 133.33/hour. If it is unavailable or above that ceiling, use one RTX-PRO6000 96 GB spot container in `IN1` when its rate is at or below INR 100/hour.

Do not use H100 on-demand, H200, multiple GPUs, another region, or another model. Stop without spending when neither approved spot resource meets its ceiling.

## Operator entrypoint

Before creating a resource:

```bash
jl status --json
jl gpus --json
jl resources --json
```

Authentication must succeed without exposing a token. Then follow sections 3 through 17 of [qwen-implementation.md](qwen-implementation.md) in order. Do not improvise a direct `jl create` or invoke `scripts/run_qwen_vllm.sh` outside the v2 lifecycle wrapper.

The wrapper also requires this explicit command-line lock after fresh approval:

```bash
PYTHONPATH=src python -m evaluation.qwen_pipeline \
  --confirm-paid-gpu qwen35-27b-fp8-v2-paid-gpu-approved
```

Before the full model download, the wrapper must pass the Python 3.12 FlashInfer import check and a metadata-only dummy-weight engine boot. Failure at either gate destroys the instance without attempting the 30.9 GB download.

The wrapper must keep one accepted instance running across approved stages, checkpoint every response, download recoverable artifacts, destroy the instance on every exit path, and verify that it no longer exists. Pausing is not final cleanup.

## Hard limits

| Stage | Incremental cap | Cumulative cap | Planned provider requests |
| --- | ---: | ---: | ---: |
| Compatibility | INR 200 | INR 200 | 12 |
| Development | INR 300 | INR 500 | 930 |
| Frozen test | INR 1,000 | INR 1,500 | 3,720 |

The planned total is 4,662 provider requests. At most 25 additional requests may result from the single-retry transport policy. Invalid structured output is never retried. Stop before a stage or cumulative cap, retaining enough time to download artifacts and destroy the resource.

## Emergency cleanup

If the lifecycle wrapper is interrupted, identify the recorded machine ID, download the available run directory, and destroy the instance:

```bash
jl list --json
jl get <machine_id> --json
jl download <machine_id> <remote_result_directory> <local_recovery_directory> -r
jl destroy <machine_id> --yes --json
jl list --json
jl get <machine_id> --json
```

Record download or destruction failures locally and keep retrying cleanup. Do not leave an approved run in a paused or unknown state.
