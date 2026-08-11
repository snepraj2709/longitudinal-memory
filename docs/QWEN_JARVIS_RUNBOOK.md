# Qwen JarvisLabs runbook

## Frozen series

- Series: `qwen35-27b-fp8-v1`
- Model: `Qwen/Qwen3.5-27B-FP8`
- Hugging Face revision: `97f5941bf617e31c5e237364a8602ce3f03a551a`
- vLLM revision: `65b7662d3fcb773afaf751ab29ac6960a0cf011d`
- Alias: `qwen35-27b-fp8-v1`
- Context: 16,384 tokens, text only, non-thinking
- Preferred GPU: one spot RTX-PRO6000 96 GB container in IN1, PyTorch template
- Fallback: H200 spot only after separate approval

The RTX-PRO6000 spot rate inspected on 2026-08-11 was INR 93.96/hour. Recheck it immediately before creation:

```bash
jl status --json
jl gpus --json
jl resources --json
```

Do not create an instance until the owner separately approves the current rate and Stage 1 spend.

## Stage 1: INR 200 cap

The committed compatibility pack contains three extraction, three QA, three summary, and three interactive requests. It uses B0 for answer compatibility and does not reuse OpenAI extraction.

On the approved GPU, install the exact vLLM revision from `requirements-qwen.txt`, set `HF_TOKEN` and a private `VLLM_API_KEY`, then start:

```bash
scripts/run_qwen_vllm.sh
```

Record measured model download and startup seconds. From the local machine, tunnel the private endpoint and run:

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_preflight \
  --requests results/evaluation/qwen35-27b-fp8-v1/stage1/compatibility-requests.jsonl \
  --output results/evaluation/qwen35-27b-fp8-v1/stage1/run-001 \
  --base-url http://127.0.0.1:6006 \
  --model qwen35-27b-fp8-v1 \
  --api-key "$VLLM_API_KEY" \
  --hourly-rate-inr 93.96 \
  --stage-cap-inr 200 \
  --setup-seconds <MEASURED_SECONDS>
```

The runner makes no automatic retry. It checkpoints each response and records tokenizer count, inference time, throughput, and GPU cost. Download the complete Stage 1 directory before pausing the instance.

## Stage 2: INR 300 cap

Stage 2 covers the complete two-user development slice: 932 requests. Start it only after reviewing Stage 1 schema compatibility. Qwen must regenerate extraction; do not read or copy OpenAI extraction predictions into this series.

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_benchmark \
  --split development \
  --output results/evaluation/qwen35-27b-fp8-v1/stage2/run-001 \
  --base-url http://127.0.0.1:6006 \
  --api-key "$VLLM_API_KEY" \
  --hourly-rate-inr <CURRENT_RATE> \
  --stage-cap-inr 300 \
  --cumulative-cap-inr 500 \
  --prior-cost-inr <STAGE_1_COST> \
  --setup-seconds <STAGE_2_SETUP_SECONDS>
```

After an intentional stop, pass `--resume`. Every checkpointed valid or invalid model response is skipped, so invalid output is not retried.

Download each completed batch before proceeding. Stop before the stage or cumulative cap. Load development gold only after all development predictions are complete.

## Stage 3: INR 1,000 cap

Stage 3 is optional. It can start only when Stage 2 has at least 95% structurally valid responses and the projected cost plus a 20% reserve fits the remaining budget. Frozen gold stays closed until corresponding predictions are complete.

The runner enforces that gate before selecting the 3,728 test-split requests:

```bash
PYTHONPATH=src .venv-storage/bin/python -m evaluation.qwen_benchmark \
  --split test \
  --output results/evaluation/qwen35-27b-fp8-v1/stage3/run-001 \
  --base-url http://127.0.0.1:6006 \
  --api-key "$VLLM_API_KEY" \
  --hourly-rate-inr <CURRENT_RATE> \
  --stage-cap-inr 1000 \
  --cumulative-cap-inr 1500 \
  --prior-cost-inr <STAGE_1_PLUS_2_COST> \
  --setup-seconds <STAGE_3_SETUP_SECONDS> \
  --stage2-valid-responses <VALID_COUNT> \
  --stage2-total-responses 932 \
  --projected-cost-inr <PROJECTED_STAGE_3_COST>
```

Pause or destroy the resource as soon as the approved stage ends. Do not switch to H200, change model/runtime revisions, retry invalid output, or extend the budget without a new approval.
