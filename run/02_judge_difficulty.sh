#!/usr/bin/env bash
# Phase 2: Score all splits with vLLM and write data/scored/
# Runtime: ~2 hrs for ~13k records on H100 (batch_size=32, ~90 rows/sec)
# Resumes automatically from checkpoint on interruption.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 2: Difficulty Judging ==="
python scripts/judge_difficulty.py "$@"
