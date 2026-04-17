"""GRPO trainer wrapper using TRL's GRPOTrainer in vLLM Server Mode.

Architecture: 2/6 GPU split.
  - GPUs 0–1: vLLM generation server (TP=2; doubles KV-cache for 96 concurrent seqs).
  - GPUs 2–7: This process — GRPOTrainer with DeepSpeed ZeRO-2 (LoRA r=8).

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

from tqdm import tqdm

# Suppress spurious vLLM version warning from TRL import (harmless with 0.19.x)
warnings.filterwarnings(
    "ignore",
    message="TRL currently supports vLLM versions from",
    category=UserWarning,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import datasets as hf_datasets
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from guac.data.prep import load_jsonl
from guac.data.utils import decode_image
from guac.training.curriculum import CurriculumSampler, CurriculumState
from guac.training.rewards import compute_reward


class CustomGRPOTrainer(GRPOTrainer):
    def create_optimizer(self):
        """Override to filter out empty parameter groups (e.g., no_decay group in LoRA).

        DeepSpeed ZeRO-2 drops empty parameter groups automatically. If the Trainer
        creates an empty group (e.g., for biases/LayerNorms when only LoRA is active),
        DeepSpeed drops it, but the LR scheduler expects the original number of groups,
        causing a `ValueError: zip() argument 2 is longer than argument 1`.
        """
        optimizer = super().create_optimizer()
        if optimizer is not None:
            # Filter out any parameter groups that have no parameters
            optimizer.param_groups = [
                group for group in optimizer.param_groups if len(group["params"]) > 0
            ]
        return optimizer

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
        if self._is_main and _WANDB_AVAILABLE:
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
          5. Save a LoRA checkpoint + trainer state (required for resume).
        """
        from transformers import TrainerCallback

        # Silence verbose library logging on all non-main ranks so the
        # tqdm bar on rank-0 is the primary source of progress information.
        if not self._is_main:
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("trl").setLevel(logging.WARNING)
            logging.getLogger("accelerate").setLevel(logging.WARNING)

        tcfg = self.cfg.training
        num_epochs = int(tcfg.num_epochs)
        steps_per_epoch = int(tcfg.steps_per_epoch)
        per_device_batch = int(tcfg.per_device_train_batch_size)
        # How many unique prompts to include in each epoch-dataset.
        # world_size × per_device_batch × steps_per_epoch gives the total
        # prompt slots consumed; we provide 2× headroom for shuffling.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        epoch_dataset_size = per_device_batch * steps_per_epoch * world_size * 2
        total_max_steps = num_epochs * steps_per_epoch

        # Single tqdm bar for the entire training run (main process only).
        pbar = (
            tqdm(
                total=total_max_steps,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
            if self._is_main
            else None
        )
        # Track how many steps have been completed so far (for bar increments).
        _steps_done = [0]  # list so the closure can mutate it
        # Keep reference to the previous epoch's trainer so we can tear down
        # its vLLM communicator before a new one is opened.
        _prev_trainer = [None]  # list so the closure can hold the ref

        for epoch in range(1, num_epochs + 1):
            if self._is_main and pbar is not None:
                pbar.set_description(f"Epoch {epoch}/{num_epochs} (T={self.curriculum.T:.3f})")

            # --------------------------------------------------------------
            # Step 1: Build curriculum-sampled dataset for this epoch
            # --------------------------------------------------------------
            train_dataset = self._build_epoch_dataset(epoch_dataset_size)

            # --------------------------------------------------------------
            # Step 2: Build GRPOConfig and GRPOTrainer
            # --------------------------------------------------------------
            # Load the previous epoch's LoRA adapter so weights carry forward.
            # We cannot resume a DeepSpeed checkpoint (those files aren't saved);
            # instead we load the PEFT adapter saved by save_model() at the end
            # of the previous epoch.  For epoch 1 there is no prior adapter.
            prev_ckpt = self._latest_checkpoint()

            grpo_config = self._build_grpo_config(
                epoch=epoch, max_steps=steps_per_epoch,
            )

            # tqdm update callback — max_steps=steps_per_epoch means the
            # trainer stops automatically; no manual stop logic required.
            is_main = self._is_main
            pbar_ref = pbar
            steps_done_ref = _steps_done

            class StepCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if is_main and pbar_ref is not None:
                        new_total = steps_done_ref[0] + state.global_step
                        increment = new_total - pbar_ref.n
                        if increment > 0:
                            reward = None
                            for entry in reversed(state.log_history):
                                for key in ("reward", "train/reward"):
                                    if key in entry:
                                        reward = entry[key]
                                        break
                                if reward is not None:
                                    break
                            postfix = {"reward": f"{reward:.4f}"} if reward is not None else {}
                            pbar_ref.update(increment)
                            pbar_ref.set_postfix(postfix)

            # ---------------------------------------------------------------
            # Close the previous epoch's vLLM communicator BEFORE creating a
            # new GRPOTrainer.  Each trainer's __init__ calls
            # vllm_client.init_communicator() on the server; the server raises
            # "Weight update group already initialized" if the previous NCCL
            # group is still open.
            # ---------------------------------------------------------------
            if _prev_trainer[0] is not None:
                prev = _prev_trainer[0]
                try:
                    if (
                        self._is_main
                        and hasattr(prev, "vllm_generation")
                        and hasattr(prev.vllm_generation, "vllm_client")
                    ):
                        prev.vllm_generation.vllm_client.close_communicator()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("close_communicator raised (ignored): %s", exc)
                # Let all ranks synchronise so rank-0 has finished closing
                # before any rank proceeds to init_communicator again.
                if hasattr(prev, "accelerator"):
                    prev.accelerator.wait_for_everyone()
                _prev_trainer[0] = None
                del prev

            # If a saved LoRA adapter exists from the previous epoch, load it
            # so training continues from where it left off.  Otherwise start
            # from the base HuggingFace model with a fresh LoRA init.
            if prev_ckpt is not None:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForVision2Seq

                base = AutoModelForVision2Seq.from_pretrained(
                    str(self.cfg.model.name),
                    torch_dtype=torch.bfloat16,
                    device_map=None,  # DeepSpeed handles placement
                )
                model_arg = PeftModel.from_pretrained(base, str(prev_ckpt))
                peft_config_arg = None  # already a PeftModel; skip re-wrapping
            else:
                model_arg = str(self.cfg.model.name)
                peft_config_arg = self.peft_config

            trainer = CustomGRPOTrainer(
                model=model_arg,
                reward_funcs=[self._make_reward_fn()],
                args=grpo_config,
                train_dataset=train_dataset,
                peft_config=peft_config_arg,
                callbacks=[StepCallback()],
            )
            _prev_trainer[0] = trainer

            # --------------------------------------------------------------
            # Step 3: Train for steps_per_epoch steps (max_steps handles stop)
            # --------------------------------------------------------------
            trainer.train()
            _steps_done[0] += steps_per_epoch

            # --------------------------------------------------------------
            # Step 4: Extract R_avg and update curriculum T
            # --------------------------------------------------------------
            R_avg = self._extract_avg_reward(trainer, steps_per_epoch)
            self.curriculum.update(R_avg)

            if self._is_main and pbar is not None:
                pbar.set_description(
                    f"Epoch {epoch}/{num_epochs} done | R_avg={R_avg:.4f} | T={self.curriculum.T:.3f}"
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
            # save_model writes the LoRA adapter weights (rank 0 only for ZeRO-2).
            # save_state writes trainer_state.json (global_step etc.) to the same
            # directory — this is what _latest_checkpoint() checks for, and what
            # resume_from_checkpoint reads to restore global_step in the next epoch.
            ckpt_dir = self.output_dir / f"epoch_{epoch:04d}"
            if self._is_main:
                trainer.save_model(str(ckpt_dir))
            trainer.save_state()  # all ranks; non-zero ranks return immediately

        # ------------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------------
        if pbar is not None:
            pbar.close()
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
        # TRL's prepare_multimodal_messages calls len() on each row's images
        # value, so each entry must be a list of PIL images (not a bare image).
        # Text-only examples use an empty list.
        images: List[List[Any]] = []

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
            images.append([pil_image] if pil_image is not None else [])

        data: Dict[str, List[Any]] = {"prompt": prompts, "answer": answers, "images": images}
        return hf_datasets.Dataset.from_dict(data)

    @staticmethod
    def _extract_completion_text(completion: Any) -> str:
        """Extract the plain-text string from a TRL completion value.

        TRL passes completions in one of two formats depending on the model
        and processor:
          - Plain string (most text-only models).
          - List of message dicts (conversational / multimodal format), where
            the last dict is the assistant turn.  The ``content`` field is
            either a plain string *or* a list of content-part dicts such as
            ``[{"type": "text", "text": "..."}]`` (Qwen2-VL multimodal).

        Args:
            completion: Raw completion value from TRL.

        Returns:
            Plain text content of the assistant's response.
        """
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            # Conversational format — take the last message (assistant turn).
            content = completion[-1].get("content", "")
        else:
            content = completion

        # Multimodal content-part list: [{"type": "text", "text": "..."}, ...]
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content)

    def _make_reward_fn(self):
        """Return a TRL-compatible reward function wrapping ``compute_reward``.

        TRL's GRPOTrainer calls::

            rewards = reward_fn(completions, **kwargs)

        where ``completions`` is a list of decoded strings for the current
        batch and ``kwargs`` contains the dataset columns — including ``answer``.

        Returns:
            Callable with signature ``(completions, **kwargs) -> List[float]``.
        """
        extract = self._extract_completion_text

        def reward_fn(completions: List[Any], **kwargs) -> List[float]:
            answers = kwargs.get("answer", [])
            rewards = []
            for completion, gt in zip(completions, answers):
                completion_text = extract(completion)
                rewards.append(compute_reward(completion_text, gt))
            return rewards

        return reward_fn

    @staticmethod
    def _make_format_reward_fn():
        """Return a TRL-compatible reward function that scores output format.

        Rewards the model for following the expected ``<think>…</think>``
        reasoning + short final-answer structure.  This provides a non-zero
        signal early in training when accuracy is near 0% and every
        correctness-reward group is flat (all zeros → no GRPO gradient).

        The maximum format reward is 0.2, so it cannot dominate the
        correctness signal once the model starts answering correctly.

        Returns:
            Callable with signature ``(completions, **kwargs) -> List[float]``.
        """
        import re as _re
        _think_open = _re.compile(r"<think>", _re.IGNORECASE)
        _think_close = _re.compile(r"</think>", _re.IGNORECASE)
        _extract = GUACGRPOTrainer._extract_completion_text

        def format_reward_fn(completions: List[Any], **kwargs) -> List[float]:
            rewards = []
            for completion in completions:
                text = _extract(completion)
                score = 0.0
                # +0.1 for opening <think> tag
                if _think_open.search(text):
                    score += 0.1
                # +0.1 for closing </think> tag (model completed its reasoning)
                if _think_close.search(text):
                    score += 0.1
                rewards.append(score)
            return rewards

        return format_reward_fn

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

            # Batch size / steps — max_steps is set to steps_per_epoch so
            # each trainer instance runs for exactly one curriculum epoch.
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

            # KL penalty coefficient.  TRL default is 0.04 (GRPO paper value),
            # but 0.1 better prevents rapid policy divergence on VLMs where the
            # reward signal is sparse early in training.
            beta=float(tcfg.get("grpo_beta", 0.1)),

            # On-policy: use each group of completions for exactly 1 gradient
            # update.  num_iterations > 1 would make GRPO off-policy.
            num_iterations=1,

            # Precision (A6000s support bf16)
            bf16=True,

            # Logging — only log to W&B; suppress TRL's own console progress
            # bar and step-level prints so the tqdm bar is the sole output.
            logging_steps=int(tcfg.log_steps),
            save_steps=int(tcfg.save_steps),
            disable_tqdm=True,
            log_on_each_node=False,

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
    def _extract_avg_reward(
        trainer: GRPOTrainer, steps_per_epoch: int
    ) -> float:
        """Extract the mean reward over the *current* epoch from log history.

        ``GRPOTrainer`` logs ``reward`` (or ``train/reward``) to
        ``trainer.state.log_history`` at every ``logging_steps`` steps.
        When resuming from a checkpoint, the log history contains entries
        from **all** previous epochs.  We filter to only the entries whose
        ``step`` falls within the current epoch's range::

            (global_step - steps_per_epoch, global_step]

        This ensures the curriculum update responds to current-epoch
        performance rather than a lagged historical average.

        Falls back to 0.0 if no reward entries are found (e.g. on a very
        short smoke-test run where no logging step fired).

        Args:
            trainer: A ``GRPOTrainer`` instance after ``.train()`` has
                     returned.
            steps_per_epoch: Number of optimizer steps in one curriculum
                             epoch.  Used to compute the step-range filter.

        Returns:
            Float mean reward for the current epoch, or 0.0 if no entries
            were logged.
        """
        current_step = trainer.state.global_step
        epoch_start = current_step - steps_per_epoch

        reward_values = []
        for entry in trainer.state.log_history:
            step = entry.get("step", 0)
            if step <= epoch_start:
                continue  # belongs to a previous epoch
            # TRL 1.x logs 'reward' directly; older versions may use 'train/reward'
            for key in ("reward", "train/reward"):
                if key in entry:
                    reward_values.append(float(entry[key]))
                    break

        if not reward_values:
            logger.warning(
                "No reward entries found in trainer log history for "
                "steps (%d, %d].  Returning R_avg=0.0.  "
                "This is normal on a 1-step smoke test.",
                epoch_start, current_step,
            )
            return 0.0

        return sum(reward_values) / len(reward_values)
