"""Shared image and text utilities used across all data processing phases.

Compatible with: datasets>=2.14, Pillow>=9.0
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
    """Base64-encode a PIL Image as PNG.

    Args:
        img: A PIL Image object to encode.

    Returns:
        A base64-encoded UTF-8 string representing the image in PNG format.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image(b64_str: str) -> Optional[PIL.Image.Image]:
    """Decode a base64 PNG string to a PIL Image.

    Args:
        b64_str: A base64-encoded UTF-8 string representing a PNG image.

    Returns:
        A PIL Image object, or None if decoding fails.
    """
    try:
        raw = base64.b64decode(b64_str)
        return PIL.Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def strip_mc_from_text(text: str) -> str:
    """Remove embedded multiple-choice option lines from text.

    Handles the following patterns:
      - ``(A) option text`` / ``(a) option text``
      - ``A. option text`` / ``A) option text``
      - ``1. option text`` / ``1) option text``
      - Lines starting with ``Options:`` or ``Choices:`` (and everything after)

    Args:
        text: The raw prompt or question string that may contain MC options.

    Returns:
        The cleaned text with MC option lines removed and trailing whitespace
        stripped.
    """
    # Patterns ordered from most specific to least; DOTALL used on trailing
    # "Options:/Choices:" blocks so everything after the keyword is consumed.
    mc_patterns = [
        # Options:/Choices: header and all text that follows
        r"\n?\s*(?:Options?|Choices?)\s*:.*",
        # Lettered options: (A), A., A), a., a) — captures rest of line
        r"\n?\s*\(?[A-Da-d]\)?[\.\):]\s*[^\n]+",
        # Numbered options: 1., 1), 2., 2) — captures rest of line
        r"\n?\s*\d+[\.\)]\s*[^\n]+",
    ]
    clean = text
    for pattern in mc_patterns:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE | re.DOTALL)
    return clean.strip()


def safe_load_image(image_field) -> Optional[PIL.Image.Image]:
    """Coerce various image field types to a PIL Image.

    Accepts:
      - ``None`` — returns None immediately.
      - ``PIL.Image.Image`` — returned as-is (converted to RGB).
      - ``bytes`` — decoded via BytesIO.
      - ``str`` or ``pathlib.Path`` — treated as a file path.
      - ``dict`` — inspected for ``"bytes"`` or ``"path"`` keys (as produced
        by the HuggingFace ``datasets`` image feature decoder).

    Args:
        image_field: The image value from a dataset row. May be any of the
            types listed above.

    Returns:
        A PIL Image in RGB mode, or None if the input is None or loading fails.
    """
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
            # HuggingFace datasets Image feature: {"bytes": ..., "path": ...}
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
