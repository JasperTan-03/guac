"""Shared pytest configuration for GUAC tests.

Forces CPU-only execution so the smoke tests are reproducible on Macs and
CI runners that have no GPU.  Real training runs still use GPUs via the
normal `scripts/train.py` entry point.
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the test process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Silence tokenizer fork warnings in the test session.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Disable HuggingFace telemetry on CI.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
