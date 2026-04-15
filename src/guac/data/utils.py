"""Shared image and text utilities used across all data processing phases.

Ported from ``src/guac/data_preprocessing/utils.py`` on ``main``.  The
trainer, judge, and evaluation code all import from ``guac.data.*`` by
path — the ``data_preprocessing`` directory on ``main`` is misnamed and
its internal imports already reference ``guac.data``.  This module
restores the expected package layout on the current branch.

Compatible with: Pillow>=9.0
"""

import base64
import io
import logging
import re
from pathlib import Path
from typing import Optional

import PIL.Image

logger = logging.getLogger(__name__)


def encode_image(img: PIL.Image.Image) -> str:
    """Base64-encode a PIL Image as PNG."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image(b64_str: str) -> Optional[PIL.Image.Image]:
    """Decode a base64 PNG string to a PIL Image, or None on failure."""
    try:
        raw = base64.b64decode(b64_str)
        return PIL.Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def strip_mc_from_text(text: str) -> str:
    """Remove embedded multiple-choice option lines from text."""
    mc_patterns = [
        r"\n?\s*(?:Options?|Choices?)\s*:.*",
        r"\n?\s*\(?[A-Da-d]\)?[\.\):]\s*[^\n]+",
        r"\n?\s*\d+[\.\)]\s*[^\n]+",
    ]
    clean = text
    for pattern in mc_patterns:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE | re.DOTALL)
    return clean.strip()


def safe_load_image(image_field) -> Optional[PIL.Image.Image]:
    """Coerce various image-field representations to a PIL Image in RGB mode."""
    if image_field is None:
        return None

    try:
        if isinstance(image_field, PIL.Image.Image):
            return image_field.convert("RGB")

        if isinstance(image_field, bytes):
            return PIL.Image.open(io.BytesIO(image_field)).convert("RGB")

        if isinstance(image_field, (str, Path)):
            return PIL.Image.open(image_field).convert("RGB")

        if isinstance(image_field, dict):
            raw_bytes = image_field.get("bytes")
            if raw_bytes:
                return PIL.Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            path = image_field.get("path")
            if path:
                return PIL.Image.open(path).convert("RGB")

        logger.warning(
            f"safe_load_image: unrecognised image_field type {type(image_field)}"
        )
        return None

    except Exception as e:
        logger.warning(f"safe_load_image: failed to load image: {e}")
        return None
