#!/usr/bin/env bash
# Start the vLLM generation server on GPUs 0–3 (Tensor Parallelism = 4).
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
echo "  GPUs:           0,1,2,3 (TP=4)"
echo ""
echo "Wait for 'Application startup complete' before launching training."
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size 4 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --served-model-name "policy" \
    --generation-config vllm
