#!/usr/bin/env bash
# Phase 3: REINFORCE training — single GPU, Gaussian curriculum
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Phase 3: REINFORCE Training (Gaussian curriculum, single GPU) ==="
python scripts/train.py \
    training.sampling_mode=gaussian \
    "$@"
