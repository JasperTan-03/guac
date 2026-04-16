#!/usr/bin/env bash
# Phase 3: REINFORCE training — 8× DDP via accelerate, Gaussian curriculum
#
# Effective batch per optimizer step:
#   batch_size=8 × world_size=8 × grad_accum=1 = 64 forward passes per step.
#   Override grad_accum if you want larger or smaller effective batches.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Phase 3: REINFORCE Training (Gaussian curriculum, 8× DDP) ==="
accelerate launch \
    --config_file=accelerate_config.yaml \
    scripts/train.py \
    training.sampling_mode=gaussian \
    training.gradient_accumulation_steps=1 \
    "$@"
