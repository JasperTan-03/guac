#!/usr/bin/env bash
# Phase 4: Evaluate the final LoRA checkpoint on MathVista and MMMU.
# Results written to results/eval_results.json and logged to MLflow.
#
# Usage:
#   bash run/04_evaluate.sh                                  # evaluate checkpoints/final
#   bash run/04_evaluate.sh checkpoint_path=checkpoints/step_500
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 4: Evaluation ==="
# Default to checkpoints/final; any $@ arg of the form checkpoint_path=... overrides it.
python scripts/evaluate.py checkpoint_path=checkpoints/final "$@"
