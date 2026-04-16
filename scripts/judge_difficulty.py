"""Difficulty judging entrypoint.

Scores every record in ``{cfg.data.processed_dir}/{stem}.jsonl`` for each
``stem`` in ``cfg.judge.splits`` using a local vLLM engine, and writes
annotated files to ``{cfg.data.scored_dir}/{stem}.jsonl``.

Each output record carries these difficulty fields:
  - ``difficulty``              — continuous float in (0, 1), or null on failure
  - ``difficulty_integer``      — argmax integer (diagnostic)
  - ``difficulty_probs``        — softmax probs over the supported single-token
                                  digit labels (``"1"`` through
                                  ``str(min(score_max, 9))``) — not necessarily
                                  every integer in ``1..score_max``
  - ``difficulty_raw_response`` — raw VLM output string
  - ``difficulty_parse_error``  — bool

The scorer resumes from a ``.ckpt`` file on crash. After changing the rubric
(``cfg.judge.system_prompt``) or score range (``cfg.judge.score_max``), delete
the stale ``{cfg.data.scored_dir}/*.ckpt`` before rerunning, otherwise old
records will be reused verbatim.

Onboarding a new dataset (zero judge-code edits required):
  1. Add a loader to ``src/guac/data/prep.py`` and register it in ``_LOADERS``.
  2. Add a ``datasets:`` entry in ``conf/data/default.yaml``.
  3. ``bash run/01_prepare_data.sh`` to re-merge into ``data/processed/*.jsonl``.
  4. ``bash run/02_judge_difficulty.sh`` (optionally with a different rubric:
     ``judge=<your_yaml>`` or ``'judge.system_prompt="..."' judge.score_max=5``).

Usage::

    python scripts/judge_difficulty.py
    python scripts/judge_difficulty.py judge.batch_size=16
    python scripts/judge_difficulty.py judge.splits=[val]
    # Swap rubric inline (no new yaml needed):
    python scripts/judge_difficulty.py 'judge.system_prompt="<your rubric>"' judge.score_max=5
    # Or author a sibling yaml under conf/judge/ (e.g. conf/judge/vllm_coding.yaml)
    # and reference it by group name:
    #   python scripts/judge_difficulty.py judge=vllm_coding
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from guac.judge.difficulty import DifficultyJudge

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the difficulty judging pipeline for every split in ``cfg.judge.splits``.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    processed_dir = Path(cfg.data.processed_dir)
    scored_dir = Path(cfg.data.scored_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    splits = list(cfg.judge.splits)

    log.info(
        "Starting difficulty judging | model=%s | batch_size=%d | splits=%s | score_max=%d",
        cfg.judge.model_name,
        cfg.judge.batch_size,
        splits,
        cfg.judge.score_max,
    )

    judge = DifficultyJudge(cfg)

    for split in splits:
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
