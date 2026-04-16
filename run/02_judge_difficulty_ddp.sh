#!/usr/bin/env bash
# Phase 2 (data-parallel, 8x A6000): score each split with N independent
# vLLM engines, one per GPU. Each rank owns records where
# (index % WORLD_SIZE == rank) and writes to
# data/scored/{stem}.part{rank}-of-{WORLD_SIZE}.jsonl. After all ranks
# finish, scripts/merge_judge_shards.py interleaves shards back into the
# canonical data/scored/{stem}.jsonl and cleans up the shard files.
#
# Override defaults with env vars:
#   WORLD_SIZE=4 bash run/02_judge_difficulty_ddp.sh
#   GPUS=0,1,2,3 WORLD_SIZE=4 bash run/02_judge_difficulty_ddp.sh
#
# Any additional Hydra overrides are forwarded to every rank:
#   bash run/02_judge_difficulty_ddp.sh judge.batch_size=16
#
# Runtime on 8x A6000 with the 7B model: ~15 min for ~13k records
# (compared to ~2 hrs single-GPU). On a COLD HF cache the first rank
# pays the model download (~15 GB) alone; other ranks block briefly on
# HF's filelock and then share the cache — no duplicate downloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

WORLD_SIZE="${WORLD_SIZE:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LOG_DIR="${LOG_DIR:-logs/judge_ddp}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -ne "$WORLD_SIZE" ]]; then
    echo "ERROR: GPUS has ${#GPU_ARRAY[@]} entries but WORLD_SIZE=$WORLD_SIZE" >&2
    exit 1
fi

# Sanity check — warn if user asked for more GPUs than nvidia-smi reports.
if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [[ "$WORLD_SIZE" -gt "$VISIBLE" ]]; then
        echo "ERROR: WORLD_SIZE=$WORLD_SIZE exceeds visible GPU count ($VISIBLE)" >&2
        exit 1
    fi
fi

mkdir -p "$LOG_DIR"

echo "=== Phase 2 (DDP): Difficulty Judging | world_size=$WORLD_SIZE | GPUs=$GPUS ==="

# --- Child-process cleanup on Ctrl-C / SIGTERM -------------------------------
# Without this, ampersand-launched ranks get reparented to init and keep
# chewing GPU if the user hits Ctrl-C. The trap sends SIGTERM to the whole
# process group so all ranks die together.
cleanup() {
    echo "=== Received signal; killing all ranks... ===" >&2
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        kill -- "${PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}
PIDS=()
trap cleanup INT TERM

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    GPU_ID="${GPU_ARRAY[$RANK]}"
    LOG_FILE="$LOG_DIR/rank${RANK}_gpu${GPU_ID}.log"
    echo "launching rank=$RANK on GPU=$GPU_ID | log=$LOG_FILE"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python scripts/judge_difficulty.py \
        judge.world_size="$WORLD_SIZE" \
        judge.rank="$RANK" \
        judge.tensor_parallel_size=1 \
        "hydra.run.dir=./outputs/judge_ddp/rank${RANK}" \
        "$@" \
        >"$LOG_FILE" 2>&1 &

    PIDS+=("$!")
done

# --- Wait for all ranks; propagate any non-zero exit --------------------------
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: rank $i (pid=${PIDS[$i]}) exited non-zero — see $LOG_DIR/rank${i}_*.log" >&2
        FAIL=1
    fi
done

trap - INT TERM

if [[ "$FAIL" -ne 0 ]]; then
    echo "=== DDP judging FAILED. Shards left in place for debugging. ===" >&2
    exit 1
fi

echo "=== All $WORLD_SIZE ranks finished. Merging shards. ==="
python scripts/merge_judge_shards.py --world-size "$WORLD_SIZE"

echo "=== Phase 2 (DDP) complete. ==="
