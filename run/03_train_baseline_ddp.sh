#!/usr/bin/env bash
# Phase 3 (Baseline / AdaRFT-style curriculum, 8x A6000 DDP): GRPO training
# distributed across all visible GPUs via HuggingFace accelerate.
#
# See 03_train_gaussian_ddp.sh for the effective-batch math and the
# grad_accum rationale.  The only difference here is sampling_mode=baseline
# — deterministic top-B*world_size selection, sliced disjoint-per-rank.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 3: GRPO Training (Baseline curriculum, 8x DDP) ==="
accelerate launch \
    --config_file=accelerate_config.yaml \
    scripts/train.py \
    training.sampling_mode=baseline \
    training.gradient_accumulation_steps=1 \
    training.batch_size=4 \
    "$@"
