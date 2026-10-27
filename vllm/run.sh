#!/usr/bin/env bash
# Phase 6 — serve the same MODEL_ID via vLLM (OpenAI-compatible endpoint).
#
# Requirements: Linux + NVIDIA GPU + vLLM installed.
#   pip install vllm
#
# Usage (from project root, outside the vllm/ subdirectory):
#   source .env && bash vllm/run.sh
#   VLLM_PORT=8001 bash vllm/run.sh
#
# NOTE: Run from the project root, not from inside vllm/ — otherwise
# Python's import system shadows the real vLLM with this directory.
#
# Tuning knobs:
#   --max-num-seqs            max concurrent sequences (default: 256)
#   --gpu-memory-utilization  fraction of GPU VRAM for KV cache (default: 0.90)
#   --max-model-len           context window; reduce to save memory on small GPUs

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID in .env or environment}"
PORT="${VLLM_PORT:-8001}"

echo "Starting vLLM server: model=$MODEL_ID  port=$PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --served-model-name "$MODEL_ID"
