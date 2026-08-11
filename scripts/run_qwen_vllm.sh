#!/usr/bin/env bash
set -euo pipefail

VLLM_API_KEY="${VLLM_API_KEY:-${QWEN_VLLM_API_KEY:-}}"
: "${VLLM_API_KEY:?VLLM_API_KEY or QWEN_VLLM_API_KEY must be set on the rented GPU}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-8b-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT="${PORT:-6006}"
VLLM_BIN="${VLLM_BIN:-vllm}"

exec "${VLLM_BIN}" serve "${MODEL_ID}" \
  --served-model-name "${MODEL_ALIAS}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --seed 42 \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
