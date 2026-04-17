"""GRPO trainer wrapper using TRL's GRPOTrainer in vLLM Server Mode.

Architecture: 4/4 GPU split.
  - GPUs 0–3: vLLM server (TP=4) — handles generation (started externally).
  - GPUs 4–7: This process — GRPOTrainer with DeepSpeed ZeRO-2 (LoRA r=8).

Curriculum updates happen at the *epoch* level rather than per-step.  After
each ``steps_per_epoch`` optimizer steps, ``R_avg`` is read from the trainer's
log history, the curriculum target T is updated via Equation 1, and the dataset
is rebuilt by re-sampling from the Gaussian distribution centred on T.  This
avoids the distributed-sampler synchronisation complexity that per-step updates
would require.

Per-epoch logic:
  1. Sample a ``datasets.Dataset`` from the Gaussian curriculum.
  2. Run ``GRPOTrainer.train()`` for ``steps_per_epoch`` steps.
  3. Read ``R_avg`` from ``trainer.state.log_history``.
  4. Update ``CurriculumState.T`` (Equation 1).
  5. Save a LoRA checkpoint.
  6. Repeat for ``num_epochs`` epochs.

Reward function (injected into GRPOTrainer):
  TRL calls ``reward_func(completions, **kwargs)`` where ``completions`` is a
  list of decoded strings and kwargs contains the dataset columns (including
  ``answer``).  We wrap ``compute_reward`` from ``rewards.py`` in a closure
  that reads the ground-truth answer from kwargs.

Compatible with: trl>=1.1.0, transformers>=4.45, peft>=0.12, vllm>=0.17.1,
                 accelerate>=1.0, datasets>=2.14, wandb>=0.18, omegaconf>=2.3
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress spurious vLLM version warning from TRL import (harmless with 0.19.x)
warnings.filterwarnings(
    "ignore",
    message="TRL currently supports vLLM versions from",
    category=UserWarning,
)

import wandb
import datasets as hf_datasets
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from guac.data.prep import load_jsonl
from guac.data.utils import decode_image
from guac.training.curriculum import CurriculumSampler, CurriculumState
from guac.training.rewards import compute_reward

logger = logging.getLogger(__name__)

# System prompt injected at the start of every conversation turn.
_SYSTEM_PROMPT = (
    "You are a concise math and science problem solver. "
    "Think step by step inside <think>...</think> tags, then provide your "
    "final answer as a number, expression, or short phrase after </think>."
)


class GUACGRPOTrainer:
    """Orchestration wrapper around TRL's GRPOTrainer for the GUAC pipeline.

    Handles:
    - Loading and filtering the scored JSONL dataset.
    - Building curriculum-sampled ``datasets.Dataset`` objects each epoch.
    - Constructing the TRL-compatible reward function from ``compute_reward``.
    - Managing the discrete epoch-level curriculum update loop.
    - W&B logging (rank 0 only) for curriculum metrics not captured by TRL.

    Args:
        cfg: Full Hydra DictConfig (``model``, ``training``, ``data`` sub-configs).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        tcfg = cfg.training

        # ------------------------------------------------------------------
        # Resolve rank for W&B gating (GRPOTrainer sets this via Accelerate)
        # ------------------------------------------------------------------
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._is_main = self._local_rank == 0

        # ------------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------------
        self.output_dir = Path(tcfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Dataset — load from {scored_dir}/train.jsonl, filter null difficulty
        # ------------------------------------------------------------------
        train_path = Path(cfg.data.scored_dir) / "train.jsonl"
        logger.info("Loading training data from %s ...", train_path)
        all_records = load_jsonl(str(train_path))

        self.records: List[Dict[str, Any]] = [
            r for r in all_records if r.get("difficulty") is not None
        ]
        n_dropped = len(all_records) - len(self.records)
        if not self.records:
            raise ValueError(
                f"No records with non-null difficulty in {train_path}. "
                "Ensure the difficulty judging phase has completed."
            )
        logger.info(
            "Dataset: %d records retained, %d dropped (null difficulty).",
            len(self.records),
            n_dropped,
        )

        difficulties: List[float] = [float(r["difficulty"]) for r in self.records]

        # ------------------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------------------
        self.curriculum = CurriculumState(
            T=float(tcfg.T_init),
            eta=float(tcfg.eta),
            alpha=float(tcfg.alpha),
            beta=float(tcfg.beta),
            d_min=float(tcfg.d_min),
            d_max=float(tcfg.d_max),
        )
        # Epoch-level sampling: we re-sample steps_per_epoch * per_device_batch_size
        # examples per epoch.  The sampler draws independently; we set world_size=1
        # because GRPOTrainer handles the DDP data sharding internally via its own
        # DataLoader — we just need to provide the full epoch-sized dataset.
        self.sampler = CurriculumSampler(
            difficulties=difficulties,
            mode=str(tcfg.sampling_mode),
            sigma=float(tcfg.sigma),
            seed=int(cfg.get("seed", 42)),
            rank=0,
            world_size=1,
        )

        # ------------------------------------------------------------------
        # LoRA config (passed to GRPOTrainer)
        # ------------------------------------------------------------------
        lora_cfg = cfg.model.lora
        self.peft_config = LoraConfig(
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.lora_alpha),
            target_modules=list(lora_cfg.target_modules),
            lora_dropout=float(lora_cfg.lora_dropout),
            bias=str(lora_cfg.bias),
            task_type=TaskType.CAUSAL_LM,
        )

        # ------------------------------------------------------------------
        # W&B (rank 0 only — TRL also logs to W&B, but we need extra metrics)
        # ------------------------------------------------------------------
        self._wandb_run = None
        if self._is_main:
            wandb_kwargs: Dict[str, Any] = {
                "project": str(tcfg.get("wandb_project", "guac")),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "name": f"grpo_{tcfg.sampling_mode}",
                "reinit": True,
            }
            wandb_entity = tcfg.get("wandb_entity", None)
            if wandb_entity:
                wandb_kwargs["entity"] = str(wandb_entity)
            self._wandb_run = wandb.init(**wandb_kwargs)
            logger.info("W&B run started: %s", self._wandb_run.id)

        logger.info(
            "GUACGRPOTrainer initialised: model=%s, T_init=%.3f, "
            "num_epochs=%d, steps_per_epoch=%d, num_generations=%d",
            cfg.model.name,
            self.curriculum.T,
            int(tcfg.num_epochs),
            int(tcfg.steps_per_epoch),
            int(tcfg.num_generations),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the discrete epoch-level GRPO curriculum training loop.

        Each epoch:
          1. Sample a dataset slice from the curriculum.
          2. Build a fresh GRPOTrainer and train for steps_per_epoch steps.
          3. Read R_avg from the trainer's log history.
          4. Update curriculum T (Equation 1).
          5. Save a LoRA checkpoint.
        """
        tcfg = self.cfg.training
        num_epochs = int(tcfg.num_epochs)
        steps_per_epoch = int(tcfg.steps_per_epoch)
        per_device_batch = int(tcfg.per_device_train_batch_size)
        num_generations = int(tcfg.num_generations)
        # How many unique prompts to include in each epoch-dataset.
        # 4 training GPUs × per_device_batch × steps_per_epoch gives the
        # total prompt slots; we provide 2× for shuffling headroom.
        epoch_dataset_size = per_device_batch * steps_per_epoch * 4

        for epoch in range(1, num_epochs + 1):
            logger.info(
                "=== Curriculum Epoch %d/%d | T=%.4f ===",
                epoch, num_epochs, self.curriculum.T,
            )

            # --------------------------------------------------------------
            # Step 1: Build curriculum-sampled dataset for this epoch
            # --------------------------------------------------------------
            train_dataset = self._build_epoch_dataset(epoch_dataset_size)
            logger.info(
                "Epoch %d dataset: %d examples (T=%.4f, mode=%s)",
                epoch, len(train_dataset), self.curriculum.T, tcfg.sampling_mode,
            )

            # --------------------------------------------------------------
            # Step 2: Build GRPOConfig and GRPOTrainer
            # --------------------------------------------------------------
            resume_path = self._latest_checkpoint()
            total_max_steps = num_epochs * steps_per_epoch
            target_step = epoch * steps_per_epoch

            grpo_config = self._build_grpo_config(
                epoch=epoch, max_steps=total_max_steps,
            )

            from transformers import TrainerCallback
            class EpochStopCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step >= target_step:
                        control.should_training_stop = True

            trainer = GRPOTrainer(
                model=str(self.cfg.model.name),
                reward_funcs=[self._make_reward_fn()],
                args=grpo_config,
                train_dataset=train_dataset,
                peft_config=self.peft_config,
                callbacks=[EpochStopCallback()],
            )

            # --------------------------------------------------------------
            # Step 3: Train for steps_per_epoch steps
            # --------------------------------------------------------------
            if resume_path is not None:
                logger.info("Resuming from checkpoint: %s", resume_path)
                trainer.train(resume_from_checkpoint=str(resume_path))
            else:
                trainer.train()

            # --------------------------------------------------------------
            # Step 4: Extract R_avg and update curriculum T
            # --------------------------------------------------------------
            R_avg = self._extract_avg_reward(trainer)
            self.curriculum.update(R_avg)
            logger.info(
                "Epoch %d complete | R_avg=%.4f | T_new=%.4f",
                epoch, R_avg, self.curriculum.T,
            )

            # Log curriculum metrics to W&B
            if self._is_main and self._wandb_run is not None:
                wandb.log(
                    {
                        "curriculum/T": self.curriculum.T,
                        "curriculum/R_avg": R_avg,
                        "curriculum/epoch": epoch,
                    },
                    step=epoch * steps_per_epoch,
                )

            # --------------------------------------------------------------
            # Step 5: Save epoch checkpoint
            # --------------------------------------------------------------
            ckpt_dir = self.output_dir / f"epoch_{epoch:04d}"
            if self._is_main:
                trainer.save_model(str(ckpt_dir))
                logger.info("Checkpoint saved: %s", ckpt_dir)

        # ------------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------------
        if self._is_main:
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(final_dir))
            logger.info("Training complete. Final checkpoint: %s", final_dir)
            if self._wandb_run is not None:
                wandb.finish()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_epoch_dataset(self, size: int) -> hf_datasets.Dataset:
        """Sample ``size`` examples from the curriculum and return a Dataset.

        Each example is formatted as:
          {
            "prompt":  [{"role": "system", ...}, {"role": "user", ...}],
            "answer":  "...",
            "image":   PIL.Image (optional),
          }

        TRL's GRPOTrainer (with a Qwen2-VLProcessor) accepts ``prompt`` as a
        list-of-dicts (the chat-template format) and PIL images in an ``images``
        column — it handles apply_chat_template internally.

        Args:
            size: Target number of examples.  Clipped to the dataset length.

        Returns:
            A ``datasets.Dataset`` with columns ``prompt``, ``answer``, and
            optionally ``images``.
        """
        size = min(size, len(self.records))
        indices = self.sampler.sample(size, self.curriculum.T)

        prompts: List[List[Dict[str, Any]]] = []
        answers: List[str] = []
        images: List[Optional[Any]] = []

        for idx in indices:
            record = self.records[idx]
            question = str(record.get("prompt", record.get("question", "")))
            answer = str(record.get("answer", ""))
            image_b64: Optional[str] = record.get("image")
            pil_image = decode_image(image_b64) if image_b64 else None

            # Build the user message content list.  TRL's GRPOTrainer passes
            # this directly to the processor's apply_chat_template, so we use
            # the Qwen2-VL dict format: {"type": "image"} and {"type": "text"}.
            user_content: List[Dict[str, Any]] = []
            if pil_image is not None:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": question})

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            prompts.append(conversation)
            answers.append(answer)
            images.append(pil_image)

        has_images = any(img is not None for img in images)
        data: Dict[str, List[Any]] = {"prompt": prompts, "answer": answers}
        if has_images:
            data["images"] = images

        return hf_datasets.Dataset.from_dict(data)

    def _make_reward_fn(self):
        """Return a TRL-compatible reward function wrapping ``compute_reward``.

        TRL's GRPOTrainer calls::

            rewards = reward_fn(completions, **kwargs)

        where ``completions`` is a list of decoded strings for the current
        batch and ``kwargs`` contains the dataset columns — including ``answer``.

        Returns:
            Callable with signature ``(completions, **kwargs) -> List[float]``.
        """
        def reward_fn(completions: List[str], **kwargs) -> List[float]:
            answers = kwargs.get("answer", [])
            return [
                compute_reward(completion, gt)
                for completion, gt in zip(completions, answers)
            ]

        return reward_fn

    def _build_grpo_config(self, epoch: int, max_steps: int) -> GRPOConfig:
        """Build a ``GRPOConfig`` for one curriculum epoch.

        Each epoch writes to a subdirectory so checkpoints don't overwrite
        each other.  ``max_steps`` caps training at ``steps_per_epoch``.

        Args:
            epoch: 1-indexed curriculum epoch number (used for output subdir).
            max_steps: Number of optimizer steps to run this epoch.

        Returns:
            A fully configured ``GRPOConfig`` instance.
        """
        tcfg = self.cfg.training
        epoch_output = str(self.output_dir / f"epoch_{epoch:04d}")

        return GRPOConfig(
            # Output
            output_dir=epoch_output,

            # Batch size / steps
            per_device_train_batch_size=int(tcfg.per_device_train_batch_size),
            gradient_accumulation_steps=int(tcfg.gradient_accumulation_steps),
            max_steps=max_steps,
            num_generations=int(tcfg.num_generations),

            # Optimiser
            learning_rate=float(tcfg.learning_rate),
            weight_decay=float(tcfg.weight_decay),
            max_grad_norm=float(tcfg.max_grad_norm),
            warmup_steps=int(tcfg.warmup_steps),

            # Generation parameters
            max_completion_length=int(tcfg.max_completion_length),
            temperature=float(tcfg.temperature),
            top_p=float(tcfg.top_p),

            # vLLM server integration
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=str(tcfg.vllm_server_host),
            vllm_server_port=int(tcfg.vllm_server_port),

            # Precision (A6000s support bf16)
            bf16=True,

            # Logging
            logging_steps=int(tcfg.log_steps),
            save_steps=int(tcfg.save_steps),

            # Report to W&B (TRL logs train/* metrics natively)
            report_to="wandb" if self._is_main else "none",

            # Don't run eval during GRPO epochs (no eval set configured)
            do_eval=False,
            
            # Don't skip batches in the new epoch's dataset when resuming
            ignore_data_skip=True,
        )

    def _latest_checkpoint(self) -> Optional[Path]:
        """Return the most recent epoch checkpoint dir, or None on first epoch.

        Looks for ``epoch_NNNN`` subdirs in ``self.output_dir`` and returns
        the lexicographically last one (which is the numerically largest)
        that contains a valid trainer_state.json.

        Returns:
            Path to the latest checkpoint directory, or None if none exist.
        """
        ckpts = sorted(self.output_dir.glob("epoch_????"))
        valid_ckpts = [c for c in ckpts if (c / "trainer_state.json").is_file()]
        return valid_ckpts[-1] if valid_ckpts else None

    @staticmethod
    def _extract_avg_reward(trainer: GRPOTrainer) -> float:
        """Extract the mean reward over the epoch from the trainer's log history.

        ``GRPOTrainer`` logs ``train/reward`` (or ``reward``) to
        ``trainer.state.log_history`` at every ``logging_steps`` steps.  We
        average all reward entries from the most recent epoch.

        Falls back to 0.0 if no reward entries are found (e.g. on a very short
        smoke-test run where no logging step fired).

        Args:
            trainer: A ``GRPOTrainer`` instance after ``.train()`` has returned.

        Returns:
            Float mean reward, or 0.0 if no entries were logged.
        """
        reward_values = []
        for entry in trainer.state.log_history:
            # TRL 1.x logs 'reward' directly; older versions may use 'train/reward'
            for key in ("reward", "train/reward"):
                if key in entry:
                    reward_values.append(float(entry[key]))
                    break

        if not reward_values:
            logger.warning(
                "No reward entries found in trainer log history. "
                "Returning R_avg=0.0.  This is normal on a 1-step smoke test."
            )
            return 0.0

        return sum(reward_values) / len(reward_values)
