"""GRPO training entrypoint.

Trains a LoRA-adapted Qwen2-VL model with GRPO and a dynamic difficulty
curriculum.  Reads training data from ``data/scored/train.jsonl`` (requires
Phase 2 to have completed).  Saves LoRA checkpoints every ``save_steps`` steps
to ``checkpoints/step_{N}/`` and a final checkpoint to ``checkpoints/final/``.

Training and curriculum metrics are logged to MLflow at every ``log_steps``
optimizer steps.

Usage::

    python scripts/train.py
    python scripts/train.py training.sampling_mode=baseline
    python scripts/train.py training.num_train_steps=2000 training.batch_size=4
    python scripts/train.py gpu_id=0
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from guac.training.trainer import GRPOTrainer

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the GRPO training loop.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.
    """
    # ``accelerate launch`` sets LOCAL_RANK per process and manages
    # CUDA_VISIBLE_DEVICES itself — overriding it here would collapse
    # all 8 ranks onto the same GPU.  Only pin to cfg.gpu_id when
    # invoked as plain ``python scripts/train.py``.
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    log.info(
        "Starting GRPO training | model=%s | steps=%d | sampling_mode=%s | gpu_id=%s",
        cfg.model.name,
        cfg.training.num_train_steps,
        cfg.training.sampling_mode,
        cfg.gpu_id,
    )

    trainer = GRPOTrainer(cfg)
    trainer.train()

    log.info("Training complete.")


if __name__ == "__main__":
    main()
