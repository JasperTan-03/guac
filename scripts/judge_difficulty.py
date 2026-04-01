"""Difficulty judging entrypoint.

Scores every record in ``data/processed/{train,val,test}.jsonl`` using a local
vLLM engine and writes annotated files to ``data/scored/{split}.jsonl``.

Each output record adds three fields to the input:
  - ``difficulty``             — float in [0.0, 1.0] (AoPS score / 10), or null on parse failure
  - ``difficulty_raw_response``— raw VLM output string
  - ``difficulty_parse_error`` — bool

The scorer resumes from a ``.ckpt`` file on crash, so re-running after
interruption picks up where it left off.

Usage::

    python scripts/judge_difficulty.py
    python scripts/judge_difficulty.py judge.batch_size=16
    python scripts/judge_difficulty.py judge.model_name=Qwen/Qwen2-VL-72B-Instruct
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from guac.judge.difficulty import DifficultyJudge

log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the difficulty judging pipeline for all splits.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    processed_dir = Path(cfg.data.processed_dir)
    scored_dir = Path(cfg.data.scored_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting difficulty judging | model=%s | batch_size=%d",
        cfg.judge.model_name,
        cfg.judge.batch_size,
    )

    judge = DifficultyJudge(cfg)

    for split in SPLITS:
        input_path = processed_dir / f"{split}.jsonl"
        output_path = scored_dir / f"{split}.jsonl"

        if not input_path.exists():
            log.warning("Input file not found, skipping: %s", input_path)
            continue

        log.info("Scoring split=%s | input=%s | output=%s", split, input_path, output_path)
        summary = judge.score_split(str(input_path), str(output_path))
        log.info(
            "split=%s done | total=%d | scored=%d | parse_errors=%d",
            split,
            summary["total"],
            summary["scored"],
            summary["parse_errors"],
        )

    log.info("Difficulty judging complete.")


if __name__ == "__main__":
    main()
