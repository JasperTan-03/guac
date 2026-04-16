"""One-off script to score ChartQA train data with the difficulty judge.

Loads ``conf/judge/vllm.yaml`` for the rubric, score_max, logprobs_k, and
sampling defaults, then overrides a few fields for this ad-hoc run.
"""

import logging
from pathlib import Path

from omegaconf import OmegaConf

from guac.judge.difficulty import DifficultyJudge

logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parent


def judge_chartqa():
    # Start from the canonical judge config so new keys (system_prompt,
    # score_max, logprobs_k, splits, world_size, rank) are picked up
    # automatically without this script having to track them.
    judge_cfg = OmegaConf.load(REPO_ROOT / "conf" / "judge" / "vllm.yaml")
    judge_cfg.max_tokens = 128  # longer fallback; ChartQA answers can run wider.

    cfg = OmegaConf.create({"judge": judge_cfg})

    judge = DifficultyJudge(cfg)
    summary = judge.score_split(
        input_path="/data/troy/datasets/guac/raw/chartqa_train.jsonl",
        output_path="/data/troy/datasets/guac/scored/chartqa_train.jsonl",
    )
    print("ChartQA train judged summary:", summary)


if __name__ == "__main__":
    judge_chartqa()
