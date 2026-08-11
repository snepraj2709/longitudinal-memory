#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set on the rented GPU}"
: "${VLLM_API_KEY:?VLLM_API_KEY must be set on the rented GPU}"

MODEL_ID="Qwen/Qwen3.5-27B-FP8"
MODEL_REVISION="97f5941bf617e31c5e237364a8602ce3f03a551a"
MODEL_ALIAS="qwen35-27b-fp8-v1"

exec vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${MODEL_ALIAS}" \
  --host 0.0.0.0 \
  --port 6006 \
  --api-key "${VLLM_API_KEY}" \
  --task generate \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --generation-config vllm
