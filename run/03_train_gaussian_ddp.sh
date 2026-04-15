#!/usr/bin/env bash
# Phase 3 (Gaussian curriculum, 8x A6000 DDP): GRPO training distributed
# across all visible GPUs via HuggingFace accelerate.
#
# Effective global batch per optimizer step (with the override below):
#   batch_size=2 * group_size=4 * world_size=8 * grad_accum=1 = 64 rollouts
# — same order of magnitude as the single-GPU default (batch 2 * group 4 *
#   accum 4 = 32), so the curriculum update rule keeps similar granularity.
# Remove the `gradient_accumulation_steps=1` override below if you want
# a 4x larger batch at the cost of coarser curriculum updates.
#
# Saves LoRA checkpoints to checkpoints/step_N/ every 500 steps (rank 0
# only).  Logs training metrics to MLflow under the 'grpo_curriculum'
# experiment.  All reduction barriers for R_avg, loss, and flat-group
# counters happen inside GRPOTrainer.train() — see trainer.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 3: GRPO Training (Gaussian curriculum, 8x DDP) ==="
accelerate launch \
    --config_file=accelerate_config.yaml \
    scripts/train.py \
    training.sampling_mode=gaussian \
    training.gradient_accumulation_steps=1 \
    "$@"
