"""Data preparation entrypoint.

Loads all configured datasets for all splits and writes:
  - ``data/raw/{dataset}_{split}.jsonl``  — per-dataset per-split files
  - ``data/processed/{split}.jsonl``       — merged files across datasets

Usage::

    python scripts/prepare_data.py
    python scripts/prepare_data.py gpu_id=0
    python scripts/prepare_data.py data.processed_dir=data/processed_custom
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from guac.data.prep import prepare_all

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the data preparation pipeline.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    log.info("Starting data preparation (gpu_id=%s, seed=%d)", cfg.gpu_id, cfg.seed)
    log.info(
        "Output dirs — raw: %s | processed: %s",
        cfg.data.raw_dir,
        cfg.data.processed_dir,
    )

    prepare_all(cfg)

    log.info("Data preparation complete.")


if __name__ == "__main__":
    main()
