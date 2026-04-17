#!/usr/bin/env bash
# Phase 3: GRPO Training — 2/5 GPU split with vLLM Server Mode.
#
# Architecture:
#   GPUs 0–1 → vLLM generation server (TP=2; doubles KV-cache capacity vs TP=1)
#   GPUs 3–7 → GRPOTrainer, DeepSpeed ZeRO-2, 5-way DDP
#
# Why TP=2?  With TP=1, a single A6000 (48 GB) can serve ~48 concurrent
# sequences.  But 5 training GPUs × batch_size=2 × num_generations=8 = 80
# concurrent completion requests per micro-step.  TP=2 splits the model
# across 2 GPUs, roughly doubling available KV-cache memory so all 80
# requests can be served without queuing.
#
# This script:
#   1. Starts the vLLM server in the background on GPUs 0–1 (TP=2).
#   2. Waits for the server health endpoint to respond (up to 120s).
#   3. Launches the GRPOTrainer accelerate job on GPUs 3–7.
#   4. On exit (normal or Ctrl-C), kills the vLLM server process.
#
# Usage:
#   bash run/03_train_grpo_vllm.sh
#   bash run/03_train_grpo_vllm.sh training.num_epochs=5
#   bash run/03_train_grpo_vllm.sh training.steps_per_epoch=200 training.sampling_mode=baseline
#
# Environment overrides:
#   VLLM_PORT=8001             — change default port
#   MAX_MODEL_LEN=2048         — reduce if vLLM OOMs on A6000s
#   GPU_UTIL=0.85              — lower GPU memory utilisation fraction
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-120}"

# ---------------------------------------------------------------------------
# Trap: kill vLLM server on exit
# ---------------------------------------------------------------------------
VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo ""
        echo "=== Shutting down vLLM server (PID ${VLLM_PID}) ==="
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Step 1: Start vLLM server on GPUs 0–1 (TP=2)
# ---------------------------------------------------------------------------
echo "=== Phase 3: GRPO Training (2/5 GPU split) ==="
echo ""
echo "--- Starting vLLM server on GPUs 0–1 (TP=2) ---"
CUDA_VISIBLE_DEVICES=0,2 trl vllm-serve \
    --model "${MODEL}" \
    --tensor-parallel-size 2 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM server PID: ${VLLM_PID}"

# ---------------------------------------------------------------------------
# Step 2: Wait for server to be healthy
# ---------------------------------------------------------------------------
echo ""
echo "--- Waiting for vLLM server to become healthy (timeout: ${VLLM_READY_TIMEOUT}s) ---"
elapsed=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [[ "${elapsed}" -ge "${VLLM_READY_TIMEOUT}" ]]; then
        echo "ERROR: vLLM server did not become healthy within ${VLLM_READY_TIMEOUT}s."
        echo "Check logs above. Possible causes:"
        echo "  - CUDA OOM: reduce MAX_MODEL_LEN (current: ${MAX_MODEL_LEN}) or GPU_UTIL"
        echo "  - Model download in progress (first run)"
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "  ... still waiting (${elapsed}s elapsed)"
done
echo "vLLM server is healthy at http://localhost:${VLLM_PORT}"

# ---------------------------------------------------------------------------
# Step 3: Launch GRPOTrainer on GPUs 2–7
# ---------------------------------------------------------------------------
echo ""
echo "--- Launching GRPOTrainer on GPUs 3–7 (DeepSpeed ZeRO-2, 5-way DDP) ---"
CUDA_VISIBLE_DEVICES=3,4,5,6,7 accelerate launch \
    --config_file=accelerate_config_grpo.yaml \
    scripts/train_grpo.py \
    training=grpo_trl \
    training.vllm_server_port="${VLLM_PORT}" \
    "$@"

echo ""
echo "=== GRPO training complete ==="
