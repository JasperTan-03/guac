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

Distributed judging (data-parallel across N GPUs):
  Run N copies with ``judge.world_size=N judge.rank={0..N-1}``. Each rank
  scores ``index % N == rank`` of the records and writes to
  ``{scored_dir}/{stem}.part{rank}-of-{world_size}.jsonl`` (and a matching
  ckpt). After all ranks finish, run ``scripts/merge_judge_shards.py`` to
  stitch the shards into a single canonical ``{stem}.jsonl``. See
  ``run/02_judge_difficulty_ddp.sh`` for an 8-GPU launcher.

  The bash launcher assigns each rank its own GPU via ``CUDA_VISIBLE_DEVICES``
  before entering Python; this script defers to a pre-set ``CUDA_VISIBLE_DEVICES``
  env var (it only applies ``cfg.gpu_id`` when the env var is absent).

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
    # Respect an externally-set CUDA_VISIBLE_DEVICES (the DDP launcher pins
    # each rank to a specific GPU before entering Python). Only apply
    # cfg.gpu_id when the env var is absent, mirroring the LOCAL_RANK
    # convention in scripts/train.py.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    processed_dir = Path(cfg.data.processed_dir)
    scored_dir = Path(cfg.data.scored_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    splits = list(cfg.judge.splits)
    world_size = int(cfg.judge.world_size)
    rank = int(cfg.judge.rank)

    if world_size < 1:
        raise ValueError(f"judge.world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(
            f"judge.rank={rank} out of range for judge.world_size={world_size}"
        )

    log.info(
        "Starting difficulty judging | model=%s | batch_size=%d | splits=%s | score_max=%d | rank=%d/%d",
        cfg.judge.model_name,
        cfg.judge.batch_size,
        splits,
        cfg.judge.score_max,
        rank,
        world_size,
    )

    judge = DifficultyJudge(cfg)

    for split in splits:
        input_path = processed_dir / f"{split}.jsonl"
        if world_size > 1:
            output_path = (
                scored_dir / f"{split}.part{rank}-of-{world_size}.jsonl"
            )
        else:
            output_path = scored_dir / f"{split}.jsonl"

        if not input_path.exists():
            log.warning("Input file not found, skipping: %s", input_path)
            continue

        log.info("Scoring split=%s | input=%s | output=%s", split, input_path, output_path)
        summary = judge.score_split(
            str(input_path),
            str(output_path),
            rank=rank,
            world_size=world_size,
        )
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
