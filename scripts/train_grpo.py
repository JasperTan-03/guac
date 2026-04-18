"""GRPO training entrypoint (TRL GRPOTrainer + vLLM Server Mode).

Runs the discrete epoch-level GRPO curriculum loop.  Reads training data from
the scored directory configured in ``conf/data/default.yaml``.  Saves LoRA
checkpoints after each curriculum epoch and a final checkpoint when done.

The vLLM generation server must already be running on the configured host/port
before launching this script.  Use ``run/start_vllm_server.sh`` to start it, or
run the full ``run/03_train_grpo_vllm.sh`` wrapper which handles both.

Usage (with accelerate — typical)::

    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \\
        --config_file=accelerate_config_grpo.yaml \\
        scripts/train_grpo.py

    # Override config values at the CLI:
    scripts/train_grpo.py training.num_epochs=5 training.steps_per_epoch=200
    scripts/train_grpo.py training.sampling_mode=baseline

Usage (single GPU, for debugging)::

    CUDA_VISIBLE_DEVICES=4 python scripts/train_grpo.py \\
        training.num_epochs=1 training.steps_per_epoch=2
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from guac.training.grpo_trainer import GUACGRPOTrainer

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the GRPO curriculum training loop.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.
             Override ``training`` to ``grpo_trl`` via ``conf/config.yaml``
             or pass ``+training=grpo_trl`` on the CLI.
    """
    # ``accelerate launch`` manages CUDA_VISIBLE_DEVICES via LOCAL_RANK.
    # Only pin to cfg.gpu_id when invoked as a plain ``python`` call.
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    log.info(
        "Starting GRPO training | model=%s | epochs=%d | steps_per_epoch=%d | sampling_mode=%s | vllm=%s:%d",
        cfg.model.name,
        cfg.training.num_epochs,
        cfg.training.steps_per_epoch,
        cfg.training.sampling_mode,
        cfg.training.vllm_server_host,
        cfg.training.vllm_server_port,
    )

    trainer = GUACGRPOTrainer(cfg)
    trainer.train()

    log.info("GRPO training complete.")


if __name__ == "__main__":
    main()
