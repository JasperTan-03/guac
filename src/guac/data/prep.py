"""Dataset loaders that produce per-dataset per-split JSONL files, then merge
splits across datasets into unified train/val/test files.

Compatible with: datasets>=2.14, omegaconf>=2.3
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig

from guac.data.utils import encode_image, safe_load_image, strip_mc_from_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------


def load_geometry3k(hf_id: str, split: str) -> List[Dict]:
    """Load the Geometry3K dataset for the given split.

    Uses the ``hiyouga/geometry3k`` HuggingFace dataset. Each row is expected
    to contain:
      - ``images``: a list of PIL Images embedded via the HF Image feature.
        ``images[0]`` is used as the row's image.
      - ``problem``: the question string, which may contain ``<image>`` tokens
        that mark image placement positions. These are stripped to produce a
        clean prompt.
      - ``answer``: an open-ended text answer (already not multiple-choice).

    Args:
        hf_id: The HuggingFace dataset identifier (e.g. ``"hiyouga/geometry3k"``).
        split: The dataset split to load (e.g. ``"train"``, ``"validation"``).

    Returns:
        A list of dicts with keys ``id``, ``image`` (base64 PNG str or None),
        ``prompt`` (str), and ``answer`` (str).
    """
    import datasets as hf_datasets

    logger.info(f"Loading {hf_id} split={split}")
    ds = hf_datasets.load_dataset(hf_id, split=split)

    records: List[Dict] = []
    skipped = 0

    for idx, row in enumerate(ds):
        row_id = f"geometry3k_{split}_{idx}"

        # --- prompt ---
        raw_problem = row.get("problem", "") or ""
        # Remove <image> placeholder tokens inserted by the dataset
        prompt = re.sub(r"<image\s*/?>", "", raw_problem, flags=re.IGNORECASE).strip()
        prompt = strip_mc_from_text(prompt)

        # --- answer ---
        answer = str(row.get("answer", "") or "").strip()

        if not prompt or not answer:
            logger.debug(
                f"geometry3k: skipping {row_id} — empty prompt or answer "
                f"(prompt={bool(prompt)}, answer={bool(answer)})"
            )
            skipped += 1
            continue

        # --- image ---
        image_b64: Optional[str] = None
        images_field = row.get("images")
        if images_field and len(images_field) > 0:
            pil_img = safe_load_image(images_field[0])
            if pil_img is not None:
                image_b64 = encode_image(pil_img)

        records.append(
            {"id": row_id, "image": image_b64, "prompt": prompt, "answer": answer}
        )

    logger.info(
        f"geometry3k split={split}: {len(records)} records kept, {skipped} skipped"
    )
    return records


def load_scienceqa(
    hf_id: str, split: str, filter_no_image: bool = True
) -> List[Dict]:
    """Load the ScienceQA dataset for the given split.

    Uses the ``derek-thomas/ScienceQA`` HuggingFace dataset. Each row
    contains:
      - ``question``: the question string (may embed MC options in some rows).
      - ``choices``: list of answer strings.
      - ``answer``: integer index into ``choices`` selecting the correct answer.
      - ``image``: a PIL Image or None for text-only rows.

    MC options that appear embedded in the ``question`` field are stripped.
    The correct answer label is resolved via ``choices[answer]``.

    Args:
        hf_id: The HuggingFace dataset identifier
            (e.g. ``"derek-thomas/ScienceQA"``).
        split: The dataset split to load
            (e.g. ``"train"``, ``"validation"``, ``"test"``).
        filter_no_image: If True, rows where ``image`` is None are dropped.
            Defaults to True.

    Returns:
        A list of dicts with keys ``id``, ``image`` (base64 PNG str or None),
        ``prompt`` (str), and ``answer`` (str).
    """
    import datasets as hf_datasets

    logger.info(
        f"Loading {hf_id} split={split} filter_no_image={filter_no_image}"
    )
    ds = hf_datasets.load_dataset(hf_id, split=split)

    records: List[Dict] = []
    skipped = 0

    for idx, row in enumerate(ds):
        row_id = f"scienceqa_{split}_{idx}"

        # --- image ---
        image_b64: Optional[str] = None
        pil_img = safe_load_image(row.get("image"))

        if filter_no_image and pil_img is None:
            logger.debug(f"scienceqa: skipping {row_id} — no image")
            skipped += 1
            continue

        if pil_img is not None:
            image_b64 = encode_image(pil_img)

        # --- prompt ---
        raw_question = row.get("question", "") or ""
        prompt = strip_mc_from_text(raw_question)

        # --- answer ---
        choices = row.get("choices") or []
        answer_idx = row.get("answer")
        answer = ""
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
            answer = str(choices[answer_idx]).strip()
        else:
            logger.debug(
                f"scienceqa: skipping {row_id} — invalid answer index "
                f"{answer_idx!r} for choices of length {len(choices)}"
            )
            skipped += 1
            continue

        if not prompt or not answer:
            logger.debug(
                f"scienceqa: skipping {row_id} — empty prompt or answer"
            )
            skipped += 1
            continue

        records.append(
            {"id": row_id, "image": image_b64, "prompt": prompt, "answer": answer}
        )

    logger.info(
        f"scienceqa split={split}: {len(records)} records kept, {skipped} skipped"
    )
    return records


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_mathverse(hf_id: str, split: str) -> List[Dict]:
    """Load the MathVerse dataset for the given split.

    Args:
        hf_id: The HuggingFace dataset identifier (e.g. ``"AI4Math/MathVerse"``).
        split: The dataset split to load.

    Returns:
        A list of dicts with keys ``id``, ``image``, ``prompt``, and ``answer``.
    """
    import datasets as hf_datasets

    logger.info(f"Loading {hf_id} split={split}")
    # MathVerse uses 'testmini' config. If we need something else later we can adapt.
    ds = hf_datasets.load_dataset(hf_id, "testmini", split=split)

    records: List[Dict] = []
    skipped = 0

    for idx, row in enumerate(ds):
        row_id = f"mathverse_{split}_{idx}"

        # --- image ---
        image_b64: Optional[str] = None
        pil_img = safe_load_image(row.get("image"))

        if pil_img is not None:
            image_b64 = encode_image(pil_img)

        # --- prompt ---
        # The prompt is in the 'question' field (sometimes 'query_wo' or 'query_cot')
        # Here we use 'question' and strip MC formatting if present
        raw_question = row.get("question", "") or ""
        prompt = strip_mc_from_text(raw_question)

        # --- answer ---
        answer = str(row.get("answer", "")).strip()

        if not prompt or not answer:
            logger.debug(
                f"mathverse: skipping {row_id} — empty prompt or answer"
            )
            skipped += 1
            continue

        records.append(
            {"id": row_id, "image": image_b64, "prompt": prompt, "answer": answer}
        )

    logger.info(
        f"mathverse split={split}: {len(records)} records kept, {skipped} skipped"
    )
    return records


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def save_jsonl(records: List[Dict], path: str) -> None:
    """Write records to a JSONL file (one JSON object per line).

    Creates parent directories if they do not exist.

    Args:
        records: A list of dicts to serialise. Each dict must be
            JSON-serialisable.
        path: Destination file path as a string.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug(f"Wrote {len(records)} records to {dest}")


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of parsed dicts, one per non-empty line.
    """
    src = Path(path)
    records: List[Dict] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.debug(f"Loaded {len(records)} records from {src}")
    return records


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

# Maps dataset name (as declared in cfg.data.datasets[*].name) to its loader
# function.  Each loader must accept (hf_id: str, split: str, **kwargs) and
# return List[Dict].
_LOADERS = {
    "geometry3k": load_geometry3k,
    "scienceqa": load_scienceqa,
    "mathverse": load_mathverse,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_all(cfg: DictConfig) -> None:
    """Main entry point for the data preparation pipeline.

    Iterates over every dataset and split declared in ``cfg.data.datasets``.
    For each combination it:

    1. Calls the appropriate loader to fetch records.
    2. Saves the records to
       ``{cfg.data.raw_dir}/{dataset_name}_{split_stem}.jsonl``.
    3. Logs the record count.

    After all per-dataset files are written, merges across datasets for each
    split stem (``train``, ``val``, ``test``) and saves the combined files to
    ``{cfg.data.processed_dir}/{split_stem}.jsonl``.

    A final summary table is printed to the log at INFO level.

    Args:
        cfg: A Hydra ``DictConfig`` with the following structure::

            data:
              raw_dir: "data/raw"
              processed_dir: "data/processed"
              split_name_map:           # HF split -> output stem
                train: train
                validation: val
                test: test
              datasets:
                - name: geometry3k
                  hf_id: hiyouga/geometry3k
                  splits: [train, validation]
                  filter_no_image: false   # optional, default false
                - name: scienceqa
                  hf_id: derek-thomas/ScienceQA
                  splits: [train, validation, test]
                  filter_no_image: true    # optional, default true
    """
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    split_name_map: Dict[str, str] = dict(cfg.data.split_name_map)

    # summary_rows accumulates (dataset_name, split, count, path) tuples
    summary_rows: List[tuple] = []

    # per_stem tracks which raw files belong to each output stem so we can
    # merge them afterwards.  {stem: [path, ...]}
    stem_files: Dict[str, List[str]] = defaultdict(list)

    for ds_cfg in cfg.data.datasets:
        ds_name: str = ds_cfg.name
        hf_id: str = ds_cfg.hf_id
        splits: List[str] = list(ds_cfg.splits)
        filter_no_image: bool = getattr(ds_cfg, "filter_no_image", False)

        loader = _LOADERS.get(ds_name)
        if loader is None:
            logger.error(
                f"No loader registered for dataset '{ds_name}'. "
                f"Registered loaders: {list(_LOADERS)}. Skipping."
            )
            continue

        for hf_split in splits:
            split_stem = split_name_map.get(hf_split, hf_split)
            out_path = raw_dir / f"{ds_name}_{split_stem}.jsonl"

            try:
                if ds_name == "scienceqa":
                    records = loader(
                        hf_id, hf_split, filter_no_image=filter_no_image
                    )
                else:
                    records = loader(hf_id, hf_split)
            except Exception as exc:
                logger.error(
                    f"Failed to load {ds_name} split={hf_split}: {exc}",
                    exc_info=True,
                )
                summary_rows.append((ds_name, hf_split, "ERROR", str(out_path)))
                continue

            try:
                save_jsonl(records, str(out_path))
            except Exception as exc:
                logger.error(
                    f"Failed to save {out_path}: {exc}", exc_info=True
                )
                summary_rows.append((ds_name, hf_split, "SAVE_ERROR", str(out_path)))
                continue

            stem_files[split_stem].append(str(out_path))
            summary_rows.append((ds_name, hf_split, len(records), str(out_path)))
            logger.info(
                f"Saved {len(records):,} records -> {out_path}"
            )

    # --- Merge per-stem ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    merge_summary: List[tuple] = []

    for stem, file_paths in stem_files.items():
        merged: List[Dict] = []
        for fp in file_paths:
            try:
                merged.extend(load_jsonl(fp))
            except Exception as exc:
                logger.error(f"Failed to read {fp} during merge: {exc}", exc_info=True)

        merged_path = processed_dir / f"{stem}.jsonl"
        try:
            save_jsonl(merged, str(merged_path))
            merge_summary.append((stem, len(merged), str(merged_path)))
            logger.info(
                f"Merged {stem}: {len(merged):,} total records -> {merged_path}"
            )
        except Exception as exc:
            logger.error(f"Failed to save merged {merged_path}: {exc}", exc_info=True)
            merge_summary.append((stem, "SAVE_ERROR", str(merged_path)))

    # --- Summary table ---
    _log_summary(summary_rows, merge_summary)


def _log_summary(
    per_dataset: List[tuple],
    merged: List[tuple],
) -> None:
    """Log a formatted summary table of all processing results.

    Args:
        per_dataset: List of ``(dataset_name, split, count_or_status, path)``
            tuples from per-dataset loading.
        merged: List of ``(split_stem, count_or_status, path)`` tuples from
            the merge step.
    """
    sep = "-" * 90
    logger.info(sep)
    logger.info("DATA PREPARATION SUMMARY")
    logger.info(sep)
    logger.info(f"{'Dataset':<20} {'Split':<15} {'Records':>10}  {'Output path'}")
    logger.info(sep)
    for dataset_name, split, count, path in per_dataset:
        count_str = f"{count:,}" if isinstance(count, int) else str(count)
        logger.info(f"{dataset_name:<20} {split:<15} {count_str:>10}  {path}")
    logger.info(sep)
    logger.info("MERGED FILES")
    logger.info(sep)
    logger.info(f"{'Split stem':<20} {'Records':>10}  {'Output path'}")
    logger.info(sep)
    for stem, count, path in merged:
        count_str = f"{count:,}" if isinstance(count, int) else str(count)
        logger.info(f"{stem:<20} {count_str:>10}  {path}")
    logger.info(sep)

