"""Evaluation entrypoint.

Evaluates a LoRA checkpoint on MathVista and MMMU benchmarks and writes a
JSON results file to ``results/eval_results.json`` (or the path in
``cfg.evaluation.output_path``).

Metrics are also logged to MLflow under the ``evaluation`` experiment.

Usage::

    # Evaluate the final checkpoint
    python scripts/evaluate.py checkpoint_path=checkpoints/final

    # Evaluate a specific step checkpoint
    python scripts/evaluate.py checkpoint_path=checkpoints/step_500

    # Override the output path
    python scripts/evaluate.py checkpoint_path=checkpoints/final \\
        evaluation.output_path=results/step_500.json

    # Evaluate on a different GPU
    python scripts/evaluate.py checkpoint_path=checkpoints/final gpu_id=0
"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from guac.evaluation.evaluator import run_evaluation

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the evaluation pipeline.

    Args:
        cfg: Hydra-composed DictConfig from ``conf/config.yaml``.  Must include
             a ``checkpoint_path`` key (set via CLI override).

    Raises:
        ValueError: If ``checkpoint_path`` is not provided.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    checkpoint_path: str = OmegaConf.select(cfg, "checkpoint_path", default="")
    if not checkpoint_path:
        raise ValueError(
            "checkpoint_path must be provided via CLI: "
            "python scripts/evaluate.py checkpoint_path=checkpoints/final"
        )

    log.info(
        "Starting evaluation | checkpoint=%s | gpu_id=%s",
        checkpoint_path,
        cfg.gpu_id,
    )

    results = run_evaluation(cfg, checkpoint_path)

    log.info(
        "Evaluation complete | MathVista=%.4f | MMMU=%.4f | macro=%.4f",
        results["benchmarks"]["MathVista"]["accuracy"],
        results["benchmarks"]["MMMU"]["accuracy"],
        results["macro_average"],
    )


if __name__ == "__main__":
    main()
