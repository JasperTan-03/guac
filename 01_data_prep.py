import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import base64
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import PIL.Image
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with max_samples attribute.
    """
    parser = argparse.ArgumentParser(
        description="PoC data prep: load Geometry3K and ScienceQA, standardise, save as JSONL."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of rows to process per dataset (for quick testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def encode_image(img: PIL.Image.Image) -> str:
    """Base64-encode a PIL Image as a PNG byte string.

    Args:
        img: A PIL Image object.

    Returns:
        A base64-encoded UTF-8 string representing the PNG-encoded image.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def strip_mc_from_text(text: str) -> str:
    """Remove embedded multiple-choice option lines from a text string.

    Handles patterns such as:
        - ``(A) option text``
        - ``A. option text``
        - ``A) option text``
        - ``Options: ...`` / ``Choices: ...`` trailing blocks

    Args:
        text: The raw prompt or question text that may contain MC options.

    Returns:
        The text with MC option lines removed and leading/trailing whitespace
        stripped.
    """
    # Match lines that start with a letter option marker (with optional leading
    # whitespace / newline).  Use DOTALL so '.' does not cross newlines, and
    # match each option line individually.
    patterns = [
        # (A) option  /  (a) option
        r"\n?\s*\(?[A-Da-d]\)?[\.\):\s]\s*[^\n]+",
        # A. option  /  A) option  /  A: option (at start of token)
        r"\n?\s*[A-Da-d][\.\)]\s*[^\n]+",
        # "Options:" or "Choices:" and everything after on that line
        r"\n?\s*(?:Options?|Choices?)\s*:.*",
    ]
    clean = text
    for pat in patterns:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE)
    return clean.strip()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_geometry3k(max_samples: Optional[int]) -> List[Dict]:
    """Load and standardise the Geometry3K dataset from HuggingFace (hiyouga/geometry3k).

    The dataset has fields: ``images`` (list of PIL Images), ``problem`` (str),
    ``answer`` (str, already open-ended — no MC choices).  The first image in
    ``images`` is used.  ``<image>`` placeholder tokens are stripped from the
    prompt.

    Args:
        max_samples: If provided, limit processing to this many rows.

    Returns:
        A list of dicts conforming to the unified schema:
        ``{"id": str, "image": str | None, "prompt": str, "answer": str}``.
    """
    dataset_name = "hiyouga/geometry3k"
    logger.info("Loading Geometry3K from '%s' …", dataset_name)

    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as exc:
        logger.error("Failed to load Geometry3K: %s", exc)
        return []

    logger.info("Geometry3K loaded — %d rows in train split.", len(ds))
    logger.info("Geometry3K features: %s", list(ds.features.keys()))

    records: List[Dict] = []
    skipped = 0

    iterable = ds
    if max_samples is not None:
        iterable = ds.select(range(min(max_samples, len(ds))))

    for idx, row in enumerate(iterable):
        row_id = f"geometry3k_{idx}"
        try:
            # --- Prompt: strip <image> placeholder tokens ---
            raw_prompt = row.get("problem") or ""
            if not raw_prompt:
                logger.warning("geometry3k row %s: empty problem, skipping.", row_id)
                skipped += 1
                continue
            clean_prompt = re.sub(r"<image>", "", raw_prompt, flags=re.IGNORECASE).strip()
            if not clean_prompt:
                logger.warning("geometry3k row %s: prompt empty after stripping, skipping.", row_id)
                skipped += 1
                continue

            # --- Answer: already open-ended text ---
            answer_text = str(row.get("answer") or "").strip()
            if not answer_text:
                logger.warning("geometry3k row %s: empty answer, skipping.", row_id)
                skipped += 1
                continue

            # --- Image: first item in 'images' list ---
            image_b64: Optional[str] = None
            images_field = row.get("images") or []
            if images_field:
                raw_img = images_field[0]
                try:
                    if isinstance(raw_img, PIL.Image.Image):
                        image_b64 = encode_image(raw_img)
                    elif isinstance(raw_img, (str, Path)):
                        image_b64 = encode_image(PIL.Image.open(raw_img))
                    elif isinstance(raw_img, bytes):
                        image_b64 = encode_image(PIL.Image.open(io.BytesIO(raw_img)))
                    else:
                        logger.warning("geometry3k row %s: unknown image type %s.", row_id, type(raw_img))
                except Exception as img_exc:
                    logger.warning("geometry3k row %s: image encode failed: %s", row_id, img_exc)

            records.append({"id": row_id, "image": image_b64, "prompt": clean_prompt, "answer": answer_text})

        except Exception as exc:
            logger.warning("geometry3k row %s: unexpected error — %s, skipping.", row_id, exc)
            skipped += 1

    logger.info("Geometry3K: %d records produced, %d rows skipped.", len(records), skipped)
    return records


def load_scienceqa(max_samples: Optional[int]) -> List[Dict]:
    """Load and standardise the ScienceQA dataset from HuggingFace.

    Skips rows where the ``image`` field is None (text-only questions).
    Maps the integer ``answer`` index to the corresponding choice text.

    Args:
        max_samples: If provided, limit processing to this many rows.

    Returns:
        A list of dicts conforming to the unified schema:
        ``{"id": str, "image": str | None, "prompt": str, "answer": str}``.
    """
    dataset_name = "derek-thomas/ScienceQA"
    logger.info("Loading ScienceQA from '%s' …", dataset_name)

    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as exc:
        logger.error("Failed to load ScienceQA: %s", exc)
        return []

    logger.info("ScienceQA loaded — %d rows in train split.", len(ds))

    features = ds.features
    logger.info("ScienceQA features: %s", list(features.keys()))

    records: List[Dict] = []
    skipped_no_image = 0
    skipped_bad = 0

    collected = 0
    for idx, row in enumerate(ds):
        row_id = f"scienceqa_{idx}"
        try:
            # --- Filter: skip rows without an image ---
            raw_img = row.get("image")
            if raw_img is None:
                skipped_no_image += 1
                continue

            # --- Prompt ---
            raw_prompt = row.get("question") or ""
            if not raw_prompt:
                logger.warning("scienceqa row %s: empty question, skipping.", row_id)
                skipped_bad += 1
                continue
            clean_prompt = strip_mc_from_text(raw_prompt)
            if not clean_prompt:
                logger.warning(
                    "scienceqa row %s: prompt became empty after MC stripping, skipping.", row_id
                )
                skipped_bad += 1
                continue

            # --- Answer ---
            choices = row.get("choices") or []
            raw_answer = row.get("answer")
            if raw_answer is None:
                logger.warning("scienceqa row %s: missing answer field, skipping.", row_id)
                skipped_bad += 1
                continue

            if isinstance(raw_answer, int):
                if choices and 0 <= raw_answer < len(choices):
                    answer_text = str(choices[raw_answer]).strip()
                else:
                    logger.warning(
                        "scienceqa row %s: answer index %d out of range (choices len=%d), skipping.",
                        row_id, raw_answer, len(choices),
                    )
                    skipped_bad += 1
                    continue
            else:
                # Fallback: treat as text directly.
                answer_text = str(raw_answer).strip()

            if not answer_text:
                logger.warning("scienceqa row %s: empty answer, skipping.", row_id)
                skipped_bad += 1
                continue

            # --- Image encoding ---
            image_b64: Optional[str] = None
            try:
                if isinstance(raw_img, PIL.Image.Image):
                    image_b64 = encode_image(raw_img)
                elif isinstance(raw_img, (str, Path)):
                    pil_img = PIL.Image.open(raw_img).convert("RGB")
                    image_b64 = encode_image(pil_img)
                elif isinstance(raw_img, bytes):
                    pil_img = PIL.Image.open(io.BytesIO(raw_img)).convert("RGB")
                    image_b64 = encode_image(pil_img)
                else:
                    logger.warning(
                        "scienceqa row %s: unknown image type %s, storing null.",
                        row_id, type(raw_img),
                    )
            except Exception as img_exc:
                logger.warning("scienceqa row %s: failed to encode image: %s", row_id, img_exc)
                # Keep image_b64 as None — do not skip the row entirely.

            records.append(
                {
                    "id": row_id,
                    "image": image_b64,
                    "prompt": clean_prompt,
                    "answer": answer_text,
                }
            )
            collected += 1
            if max_samples is not None and collected >= max_samples:
                break

        except Exception as exc:
            logger.warning("scienceqa row %s: unexpected error — %s, skipping.", row_id, exc)
            skipped_bad += 1

    logger.info(
        "ScienceQA: %d records produced, %d rows skipped (no image), %d rows skipped (other).",
        len(records), skipped_no_image, skipped_bad,
    )
    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_jsonl(records: List[Dict], path: str) -> None:
    """Write a list of dicts to a JSONL file, one JSON object per line.

    Creates parent directories if they do not already exist.

    Args:
        records: List of dicts to serialise.
        path: Destination file path (absolute or relative).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), out_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate dataset loading, standardisation, and JSONL output."""
    args = parse_args()
    output_path = "/home/jasper/guac/data/poc_train_data.jsonl"

    all_records: List[Dict] = []

    # --- Geometry3K ---
    geo3k_records = load_geometry3k(max_samples=args.max_samples)
    all_records.extend(geo3k_records)

    # --- ScienceQA ---
    sqa_records = load_scienceqa(max_samples=args.max_samples)
    all_records.extend(sqa_records)

    # --- Save ---
    save_jsonl(all_records, output_path)

    logger.info(
        "Pipeline complete. Total rows written: %d  (geometry3k=%d, scienceqa=%d).",
        len(all_records), len(geo3k_records), len(sqa_records),
    )


if __name__ == "__main__":
    main()
