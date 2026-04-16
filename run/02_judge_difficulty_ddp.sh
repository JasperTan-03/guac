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
# (compared to ~2 hrs single-GPU). Rank 0 does the HF download on
# the first invocation; subsequent ranks share the same HF cache so
# only one network fetch happens.
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

mkdir -p "$LOG_DIR"

echo "=== Phase 2 (DDP): Difficulty Judging | world_size=$WORLD_SIZE | GPUs=$GPUS ==="

# --- Warm the HF cache on rank 0 (serial) before fanning out -----------------
# Avoids N simultaneous downloads of the VLM on first run. If the model is
# already cached this is a ~10s no-op. Sample one record via --splits=val
# and --batch_size=1 is overkill — rely on the user having run the single-GPU
# script once before, or accept a brief serial fetch on the first rank.
# (Intentionally no preload here — HF filelock handles concurrent cache
#  reads; on a cold cache the first rank claims the lock and the others
#  wait, which is safe but slow on cold start.)

PIDS=()
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

# --- Wait for all ranks; if any fails, kill the rest -------------------------
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: rank $i (pid=${PIDS[$i]}) exited non-zero — see $LOG_DIR/rank${i}_*.log" >&2
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "=== DDP judging FAILED. Shards left in place for debugging. ===" >&2
    exit 1
fi

echo "=== All $WORLD_SIZE ranks finished. Merging shards. ==="
python scripts/merge_judge_shards.py --world-size "$WORLD_SIZE"

echo "=== Phase 2 (DDP) complete. ==="
