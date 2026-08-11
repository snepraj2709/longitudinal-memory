# Qwen3-8B JarvisLabs runbook

Use this runbook with [qwen-implementation.md](qwen-implementation.md). It is for the current Qwen3-8B vLLM path, not the old Qwen3.5/H100 work.

## Current policy

- Current model: `Qwen/Qwen3-8B`.
- Served model alias: `qwen3-8b-vllm`.
- GPU: L4 24GB first.
- Fallback: A5000 24GB only if current JarvisLabs inventory offers it. On 2026-08-11, A5000 was not listed; L4 and A30 24GB containers were listed in `IN2`.
- Context: 8,192 tokens.
- Temperature: 0.
- API: OpenAI-compatible vLLM `/v1/chat/completions`.
- First repo run: 12 compatibility requests only.

The old `qwen35-27b-fp8-v1` and `qwen35-27b-fp8-v2` plans are historical. Preserve their artifacts, but do not run them.

## Before spending

Check auth and current inventory:

```bash
jl status --json
jl gpus --json
jl resources --json
```

Do not paste tokens into chat. Put the vLLM API key in ignored local `.qwen-vllm.env`:

```text
QWEN_VLLM_API_KEY=<local secret>
```

## Start vLLM on JarvisLabs

Use L4 in `IN2`:

```bash
jl create --gpu L4 --spot --template pytorch --storage 60 --http-ports "6006" --region IN2 --name longitudinal-memory-qwen3-8b --yes --json
jl upload <machine_id> scripts/run_qwen_vllm.sh /home/run_qwen_vllm.sh
jl upload <machine_id> .qwen-vllm.env /home/.qwen-vllm.env
jl exec <machine_id> -- sh -lc 'python -m pip install --upgrade vllm'
jl exec <machine_id> -- sh -lc 'chmod 700 /home/run_qwen_vllm.sh && set -a && . /home/.qwen-vllm.env && set +a && nohup /home/run_qwen_vllm.sh >/home/qwen3-8b-vllm.log 2>&1 < /dev/null &'
```

Get the HTTPS endpoint for port 6006:

```bash
jl get <machine_id> --json
```

Use that endpoint plus `/v1` as the repo `--base-url`.

## Verify from this repo

After the endpoint is reachable and paid requests are explicitly approved:

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

The receipt is written under `results/evaluation/qwen3-8b-vllm-pilot-v1/`.

## Cleanup

Destroy the instance after the pilot:

```bash
jl destroy <machine_id> --yes --json
jl get <machine_id> --json
```

The second command should fail or show that the machine no longer exists. Do not leave the instance paused unless Sneha explicitly asks for that.
