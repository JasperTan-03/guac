#!/usr/bin/env bash
# Phase 3 (Gaussian curriculum): GRPO training for 5000 steps
# Saves LoRA checkpoints to checkpoints/step_N/ every 500 steps.
# Logs training metrics to MLflow (mlruns/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 3: GRPO Training (gaussian curriculum) ==="
python scripts/train.py training.sampling_mode=gaussian "$@"
