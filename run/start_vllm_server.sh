#!/usr/bin/env bash
# Start the vLLM generation server on GPU 0 (TP=1).
#
# The 7B model (~14 GB) fits on a single A6000 (48 GB) — no tensor parallelism
# needed.  TP=1 avoids inter-GPU all-reduce overhead on every forward pass.
#
# This must be running BEFORE launching the training job.
# The server will be ready when you see:
#   "INFO:     Application startup complete."
#
# To stop the server: Ctrl-C in this terminal, or `kill <PID>` if backgrounded.
#
# Usage:
#   bash run/start_vllm_server.sh                          # default port 8000
#   VLLM_PORT=8001 bash run/start_vllm_server.sh           # custom port
#   MODEL=Qwen/Qwen2-VL-7B-Instruct bash run/start_vllm_server.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"

echo "=== Starting vLLM server ==="
echo "  Model:          ${MODEL}"
echo "  Port:           ${VLLM_PORT}"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  GPU util:       ${GPU_UTIL}"
echo "  GPUs:           0 (TP=1)"
echo ""
echo "Wait for 'Application startup complete' before launching training."
echo ""

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model "${MODEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code
