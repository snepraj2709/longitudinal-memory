#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set on the rented GPU}"
: "${VLLM_API_KEY:?VLLM_API_KEY must be set on the rented GPU}"

MODEL_PATH="/home/qwen35-27b-fp8-v2-model"
MODEL_ALIAS="qwen35-27b-fp8-v2"
MODE="${1:-production}"

case "${MODE}" in
  preflight)
    MODEL_PATH="/home/qwen35-27b-fp8-v2-metadata"
    EXTRA_ARGS=(--load-format dummy --max-model-len 1024 --max-num-seqs 1)
    ;;
  production)
    EXTRA_ARGS=(--max-model-len 16384 --max-num-seqs 16)
    ;;
  *)
    echo "unsupported Qwen server mode: ${MODE}" >&2
    exit 2
    ;;
esac

exec /home/qwen-v2-env/bin/vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_ALIAS}" \
  --host 0.0.0.0 \
  --port 6006 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --seed 42 \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --language-model-only \
  "${EXTRA_ARGS[@]}"
