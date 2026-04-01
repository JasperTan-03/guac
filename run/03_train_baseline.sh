#!/usr/bin/env bash
# Phase 3 (Baseline): GRPO training with nearest-neighbour curriculum sampling
# Saves LoRA checkpoints to checkpoints/step_N/ every 500 steps.
# Logs training metrics to MLflow (mlruns/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 3: GRPO Training (baseline curriculum) ==="
python scripts/train.py training.sampling_mode=baseline "$@"
