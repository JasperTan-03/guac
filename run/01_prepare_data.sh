#!/usr/bin/env bash
# Phase 1: Load all dataset splits and write data/raw/ + data/processed/
# Runtime: ~5-10 min (depends on HuggingFace download speed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "=== Phase 1: Data Preparation ==="
python scripts/prepare_data.py "$@"
