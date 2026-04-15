#!/usr/bin/env bash
# CPU-runnable smoke tests for the GRPO training loop.
#
# Runs ~10 seconds on a Mac, uses a 5M-param random-weights Qwen2-VL
# checkpoint + 8 synthetic records with no images, and verifies the
# reward cascade (exact / extracted / SymPy / substring), flat-group
# skipping, GRPO loss math, and gradient flow into LoRA parameters.
#
# Uses the system Python — no .venv required (the committed .venv at
# .venv/ is a Linux/uv venv intended for the GPU box and does not work
# on Mac).  Install these once:
#
#     python3 -m pip install torch transformers peft mlflow omegaconf \
#                            sympy torchvision pytest
#
# Then:
#
#     bash run/00_smoke_test.sh
#
# Intended for local development and CI — NOT a substitute for a real
# GPU training run.  For that, use run/03_train_gaussian.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Force CPU: the tests must be reproducible on machines without a GPU.
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

# Resolve pytest via the system Python (no .venv activation).
PY="${PYTHON:-python3}"

echo "=== GUAC smoke test (CPU, no GPU required) ==="
PYTHONPATH=src "$PY" -m pytest tests/test_training_smoke.py -v "$@"
