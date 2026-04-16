#!/usr/bin/env bash
# 2-process gloo-backend DDP smoke test.
#
# Verifies the entire distributed training path on CPU:
#  - Both processes init Accelerator, load the tiny model, run 2 steps.
#  - backward() fires on both ranks on every micro-step (no NCCL hang).
#  - Rank 0 logs to MLflow; rank 1 does not.
#  - Curriculum T updates.  Checkpoint saved by rank 0 only.
#
# Runs in ~20 s on a Mac.  If it hangs, the DDP plumbing is broken.
#
# Usage:
#     bash run/00_smoke_test_ddp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export TOKENIZERS_PARALLELISM=false

PY="${PYTHON:-python3}"

echo "=== GUAC DDP smoke test (2x gloo on CPU) ==="

# Uses torch.multiprocessing.spawn + gloo backend (no NCCL/GPU).
# Runs GRPOTrainer.train() for 2 steps on each of 2 ranks.
PYTHONPATH=src "$PY" tests/test_ddp_cpu.py

echo "=== DDP smoke test PASSED ==="
