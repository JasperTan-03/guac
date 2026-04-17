"""REINFORCE trainer with EMA baseline, dynamic difficulty curriculum, and W&B logging.

Replaces the GRPO trainer that suffered from flat-group collapse (all rollouts
in a group earning the same binary reward → zero advantages → zero gradient).

The key difference: REINFORCE computes advantages against a *running historical
average* (EMA baseline) instead of within-group normalization.  Even when every
response in a batch earns the same reward, the advantage is non-zero as long as
the EMA baseline hasn't converged to that reward level.

Per-step logic:
  1. Sample a batch of examples from the dataset using ``CurriculumSampler``.
  2. Generate **one** response per prompt (G=1) via ``model.generate()``.
  3. Score each response with exact-match reward (``compute_reward``).
  4. Compute ``R_avg`` and update curriculum target T (Equation 1).
  5. Compute per-sequence policy log-probs via teacher forcing.
  6. Compute per-sequence reference log-probs via ``model.disable_adapter()``
     (zero extra VRAM — no separate reference model needed).
  7. Compute KL penalty and adjusted rewards.
  8. Compute advantages: ``A_i = R_hat_i - b`` (EMA baseline).
  9. Update EMA baseline: ``b = decay * b + (1 - decay) * R_avg``.
  10. Compute REINFORCE loss: ``L = -mean(A.detach() * policy_logprobs)``.
  11. Accumulate gradients, clip, and step optimizer.
  12. Log metrics to W&B every ``log_steps`` optimizer steps.
  13. Save a LoRA checkpoint every ``save_steps`` optimizer steps.

Compatible with: transformers>=4.45, peft>=0.12, torch>=2.3, wandb>=0.18,
                 omegaconf>=2.3
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup

from guac.data.prep import load_jsonl
from guac.data.utils import decode_image
from guac.training.curriculum import CurriculumSampler, CurriculumState
from guac.training.rewards import compute_reward

logger = logging.getLogger(__name__)


class ReinforceTrainer:
    """REINFORCE trainer with EMA baseline and dynamic difficulty curriculum.

    Loads model and data according to the Hydra config, applies LoRA, and
    runs a REINFORCE training loop with gradient accumulation, an adaptive
    curriculum over example difficulty, and KL-penalised rewards.

    Unlike GRPOTrainer, this generates exactly one response per prompt (G=1),
    which eliminates the flat-group problem and significantly reduces VRAM.

    Config structure expected (all fields accessed via ``cfg.*``)::

        model:
          name: "Qwen/Qwen2-VL-7B-Instruct"
          dtype: "bfloat16"
          lora:
            r: 8
            lora_alpha: 16
            target_modules: [q_proj, k_proj, v_proj, o_proj]
            lora_dropout: 0.05
            bias: none

        training:
          output_dir: "checkpoints"
          batch_size: 8
          gradient_accumulation_steps: 2
          learning_rate: 1.0e-5
          weight_decay: 0.01
          max_grad_norm: 1.0
          num_train_steps: 5000
          warmup_steps: 100
          save_steps: 500
          log_steps: 10
          max_new_tokens: 128
          temperature: 1.0
          top_p: 0.9
          ema_decay: 0.95
          kl_coeff: 0.1
          sampling_mode: "gaussian"
          T_init: 0.1
          eta: 0.05
          alpha: 2.0
          beta: 0.5
          wandb_project: "guac"
          wandb_entity: null
          sigma: 0.15
          d_min: 0.1
          d_max: 0.7
          wandb_project: "guac"

        data:
          scored_dir: "/data/troy/datasets/guac/scored"

    Args:
        cfg: Full Hydra ``DictConfig`` containing ``model``, ``training``,
             and ``data`` sub-configs.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Set up model, LoRA, optimizer, scheduler, curriculum, and W&B.

        Args:
            cfg: Full Hydra DictConfig (see class docstring for structure).

        Raises:
            FileNotFoundError: If the training JSONL file does not exist.
            ValueError: If no valid records remain after difficulty filtering.
        """
        self.cfg = cfg
        tcfg = cfg.training

        # ------------------------------------------------------------------
        # Accelerator / device / rank
        # ------------------------------------------------------------------
        self.accelerator = Accelerator(mixed_precision="no")
        self.device = self.accelerator.device
        self.is_main: bool = self.accelerator.is_main_process
        self.rank: int = self.accelerator.process_index
        self.world_size: int = self.accelerator.num_processes
        if self.device.type != "cuda":
            logger.warning(
                "No CUDA device found. Training on CPU will be extremely slow "
                "and is not recommended for production runs."
            )
        if self.world_size > 1:
            logger.info(
                "Distributed training detected: rank=%d/%d device=%s",
                self.rank, self.world_size, self.device,
            )

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
        self.sampler = CurriculumSampler(
            difficulties=difficulties,
            mode=str(tcfg.sampling_mode),
            sigma=float(tcfg.sigma),
            seed=int(cfg.get("seed", 42)),
            rank=self.rank,
            world_size=self.world_size,
        )

        # ------------------------------------------------------------------
        # REINFORCE-specific state
        # ------------------------------------------------------------------
        self.ema_baseline: float = 0.0
        self.ema_decay = float(tcfg.ema_decay)
        self.kl_coeff = float(tcfg.kl_coeff)

        # ------------------------------------------------------------------
        # Processor — image resolution cap
        # ------------------------------------------------------------------
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if hasattr(cfg.model, "max_pixels"):
            processor_kwargs["max_pixels"] = int(cfg.model.max_pixels)
        if hasattr(cfg.model, "min_pixels"):
            processor_kwargs["min_pixels"] = int(cfg.model.min_pixels)
        logger.info("Loading processor from %s ...", cfg.model.name)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model.name,
            **processor_kwargs,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        # ------------------------------------------------------------------
        # Base model (bfloat16)
        # ------------------------------------------------------------------
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_dtype = dtype_map.get(str(cfg.model.dtype), torch.bfloat16)

        if model_dtype == torch.bfloat16 and self.device.type != "cuda":
            logger.warning(
                "Device %s does not reliably support bfloat16; coercing model "
                "dtype to float32 for this run.",
                self.device.type,
            )
            model_dtype = torch.float32
        self._model_dtype = model_dtype
        self._autocast_enabled = self.device.type == "cuda" and model_dtype in (
            torch.bfloat16, torch.float16,
        )
        self._autocast_dtype = model_dtype if self._autocast_enabled else torch.float32

        logger.info("Loading base model from %s (%s) ...", cfg.model.name, cfg.model.dtype)
        base_model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.name,
            torch_dtype=model_dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------
        # LoRA
        # ------------------------------------------------------------------
        lora_cfg = cfg.model.lora
        lora_config = LoraConfig(
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.lora_alpha),
            target_modules=list(lora_cfg.target_modules),
            lora_dropout=float(lora_cfg.lora_dropout),
            bias=str(lora_cfg.bias),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)
        if self.is_main:
            self.model.print_trainable_parameters()

        # ------------------------------------------------------------------
        # Optimizer and LR scheduler
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(tcfg.learning_rate),
            weight_decay=float(tcfg.weight_decay),
        )
        total_steps = int(tcfg.num_train_steps)
        warmup_steps = int(tcfg.warmup_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ------------------------------------------------------------------
        # Distributed wrapping
        # ------------------------------------------------------------------
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler,
        )

        # ------------------------------------------------------------------
        # W&B (rank 0 only)
        # ------------------------------------------------------------------
        self._wandb_run = None
        if self.is_main and _WANDB_AVAILABLE:
            wandb_kwargs: Dict[str, Any] = {
                "project": str(tcfg.get("wandb_project", "guac")),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "name": f"reinforce_{tcfg.sampling_mode}",
                "reinit": True,
            }
            wandb_entity = tcfg.get("wandb_entity", None)
            if wandb_entity:
                wandb_kwargs["entity"] = str(wandb_entity)
            self._wandb_run = wandb.init(**wandb_kwargs)
            logger.info(
                "W&B run started: %s", self._wandb_run.id
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full REINFORCE training loop.

        The outer loop counts *optimizer steps* (``global_step``).  Each
        optimizer step consists of ``gradient_accumulation_steps`` micro-steps,
        each of which processes a fresh batch sampled from the curriculum.

        All B examples in a micro-step are processed together in batched
        GPU operations (one ``generate()`` call, one policy forward, one
        reference forward) for maximum GPU utilization.

        At the end of training the final LoRA adapter is saved to
        ``{output_dir}/final/`` and the path is logged to W&B.
        """
        tcfg = self.cfg.training
        num_train_steps = int(tcfg.num_train_steps)
        grad_accum = int(tcfg.gradient_accumulation_steps)
        batch_size = int(tcfg.batch_size)
        log_steps = int(tcfg.log_steps)
        save_steps = int(tcfg.save_steps)
        max_grad_norm = float(tcfg.max_grad_norm)
        max_new_tokens = int(tcfg.max_new_tokens)
        temperature = float(tcfg.temperature)
        top_p = float(tcfg.top_p)

        global_step = 0
        micro_step = 0
        running_loss = 0.0
        running_reward = 0.0
        running_kl = 0.0

        self.model.train()
        self.optimizer.zero_grad()

        pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        logger.info(
            "Starting REINFORCE training: %d steps, batch_size=%d, "
            "grad_accum=%d, sampling_mode=%s, T_init=%.3f, "
            "ema_decay=%.3f, kl_coeff=%.3f",
            num_train_steps,
            batch_size,
            grad_accum,
            tcfg.sampling_mode,
            self.curriculum.T,
            self.ema_decay,
            self.kl_coeff,
        )

        # tqdm progress bar — only on main process to avoid 8× DDP spam.
        pbar = tqdm(
            total=num_train_steps,
            desc="Training",
            disable=not self.is_main,
            dynamic_ncols=True,
        )

        while global_step < num_train_steps:
            # ----------------------------------------------------------
            # Step 1: Sample batch from curriculum
            # ----------------------------------------------------------
            batch_indices = self.sampler.sample(batch_size, self.curriculum.T)
            batch_records = [self.records[i] for i in batch_indices]
            ground_truths = [str(r.get("answer", "")) for r in batch_records]

            # ----------------------------------------------------------
            # Step 2: Generate ONE response per prompt (individual calls).
            # Qwen2-VL images have variable resolutions, so batched
            # generate() causes image_token / feature mismatches.  The
            # autoregressive decode is the inevitable bottleneck anyway;
            # the real speed win is batching the log-prob forward passes.
            # ----------------------------------------------------------
            batch_rewards: List[float] = []
            per_example_gen_ids: List[torch.Tensor] = []
            _log_sample: Optional[Dict[str, Any]] = None

            self.model.eval()
            gen_model = self.accelerator.unwrap_model(self.model)

            for i, record in enumerate(batch_records):
                prompt_inputs = self._build_inputs(record)
                prompt_len = prompt_inputs["input_ids"].shape[1]

                with torch.no_grad():
                    gen_out = gen_model.generate(
                        **prompt_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    gen_ids_i = gen_out[0, prompt_len:]  # (gen_len,)

                per_example_gen_ids.append(gen_ids_i)

                decoded = self.processor.tokenizer.decode(
                    gen_ids_i, skip_special_tokens=True,
                ).strip()
                reward = compute_reward(decoded, ground_truths[i])
                batch_rewards.append(reward)

                if _log_sample is None:
                    _log_sample = {
                        "pred": decoded,
                        "gt": ground_truths[i],
                        "reward": reward,
                    }

            # ----------------------------------------------------------
            # Step 4: Batched policy log-probs (with gradient)
            # ----------------------------------------------------------
            self.model.train()
            policy_logprobs = self._compute_batch_log_probs(
                batch_records, per_example_gen_ids, use_ref=False,
            )

            # ----------------------------------------------------------
            # Step 5: Batched reference log-probs (no grad)
            # ----------------------------------------------------------
            with torch.no_grad():
                gen_model_unwrapped = self.accelerator.unwrap_model(self.model)
                gen_model_unwrapped.eval()
                with gen_model_unwrapped.disable_adapter():
                    ref_logprobs = self._compute_batch_log_probs(
                        batch_records, per_example_gen_ids, use_ref=True,
                    )
            self.model.train()

            # ----------------------------------------------------------
            # Step 6: KL, advantages, EMA update, loss
            # ----------------------------------------------------------
            rewards_tensor = torch.tensor(
                batch_rewards, dtype=torch.float32, device=self.device,
            )
            R_avg = rewards_tensor.mean().item()

            kl_tensor = policy_logprobs.detach() - ref_logprobs  # (B,)
            adjusted_rewards = rewards_tensor - self.kl_coeff * kl_tensor

            advantages = adjusted_rewards - self.ema_baseline

            self.ema_baseline = (
                self.ema_decay * self.ema_baseline
                + (1 - self.ema_decay) * R_avg
            )

            loss = -(advantages.detach() * policy_logprobs).mean()
            scaled_loss = loss / grad_accum
            self.accelerator.backward(scaled_loss)

            running_loss += loss.item()
            running_reward += R_avg
            running_kl += kl_tensor.mean().item()
            micro_step += 1

            # ----------------------------------------------------------
            # Optimizer step (once per grad_accum micro-steps)
            # ----------------------------------------------------------
            if micro_step % grad_accum == 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1

                # Curriculum update (Equation 1)
                avg_reward_this_step = running_reward / grad_accum
                self.curriculum.update(avg_reward_this_step)

                # Logging
                avg_loss = running_loss / grad_accum
                avg_kl = running_kl / grad_accum

                should_log = (
                    global_step % log_steps == 0 or global_step == 1
                )

                # Always update tqdm postfix for live terminal feedback.
                if self.is_main:
                    current_lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        reward=f"{avg_reward_this_step:.3f}",
                        T=f"{self.curriculum.T:.3f}",
                        ema=f"{self.ema_baseline:.3f}",
                        lr=f"{current_lr:.1e}",
                        ordered=True,
                    )
                    pbar.update(1)

                if should_log and self.is_main:
                    if _log_sample is not None:
                        logger.info(
                            "  sample | pred=%r | gt=%r | reward=%.1f",
                            _log_sample["pred"][:120],
                            _log_sample["gt"][:60],
                            _log_sample["reward"],
                        )
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/reward": avg_reward_this_step,
                            "curriculum/T": self.curriculum.T,
                            "train/learning_rate": current_lr,
                            "train/kl_div": avg_kl,
                            "train/ema_baseline": self.ema_baseline,
                        },
                        step=global_step,
                    )

                # Checkpoint
                if global_step % save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    if self.is_main:
                        self._save_checkpoint(global_step)

                # Reset accumulators
                running_loss = 0.0
                running_reward = 0.0
                running_kl = 0.0

            if global_step >= num_train_steps:
                break

        # ------------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------------
        pbar.close()
        self.accelerator.wait_for_everyone()
        if self.is_main:
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(
                str(final_dir)
            )
            self.processor.save_pretrained(str(final_dir))
            logger.info(
                "Training complete. Final checkpoint saved to %s", final_dir,
            )
            if self._wandb_run is not None:
                wandb.finish()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_batch_inputs(
        self, records: List[Dict[str, Any]]
    ) -> tuple:
        """Tokenise a batch of records in one processor call.

        Returns:
            Tuple of (batch_inputs dict on device, prompt_lens list).
            ``prompt_lens[i]`` is the number of prompt tokens for record i
            (needed to split prompt from generated tokens after generate()).
        """
        texts: List[str] = []
        images_flat: List[Any] = []
        prompt_lens: List[int] = []

        for record in records:
            question = str(record.get("prompt", record.get("question", "")))
            image_b64: Optional[str] = record.get("image")
            pil_image = decode_image(image_b64) if image_b64 else None

            content: List[Dict[str, Any]] = []
            if pil_image is not None:
                content.append({"type": "image", "image": pil_image})
                images_flat.append(pil_image)
            content.append({"type": "text", "text": question})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise math and science problem solver. "
                        "Provide only your final answer as a number, expression, or short phrase. "
                        "Do not show working or explanation."
                    ),
                },
                {"role": "user", "content": content},
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Tokenize all examples in one call (the processor handles left-padding).
        batch_inputs = self.processor(
            text=texts,
            images=images_flat if images_flat else None,
            return_tensors="pt",
            padding=True,
        )
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

        # Compute per-example prompt lengths from the attention mask:
        # with left-padding, prompt_len = total_len - pad_len = number of 1s.
        attn_mask = batch_inputs["attention_mask"]
        for i in range(len(records)):
            prompt_lens.append(int(attn_mask[i].sum().item()))

        return batch_inputs, prompt_lens

    def _build_inputs(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenise a single record (image + prompt) using the processor.

        Kept for backward compatibility with smoke tests.

        Args:
            record: A single dataset record dict with ``"prompt"`` and
                    ``"image"`` keys.

        Returns:
            Dict of processor outputs on ``self.device``.
        """
        question = str(record.get("prompt", record.get("question", "")))
        image_b64: Optional[str] = record.get("image")
        pil_image = decode_image(image_b64) if image_b64 else None

        content: List[Dict[str, Any]] = []
        if pil_image is not None:
            content.append({"type": "image", "image": pil_image})
        content.append({"type": "text", "text": question})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise math and science problem solver. "
                    "Provide only your final answer as a number, expression, or short phrase. "
                    "Do not show working or explanation."
                ),
            },
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images_list = [pil_image] if pil_image is not None else None
        inputs = self.processor(
            text=[text],
            images=images_list,
            return_tensors="pt",
            padding=True,
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_batch_log_probs(
        self,
        records: List[Dict[str, Any]],
        gen_ids_list: List[torch.Tensor],
        use_ref: bool = False,
    ) -> torch.Tensor:
        """Compute per-example mean log-probs for a batch.

        Since each example can have a different generated length, we process
        them individually but use the same model state. For the reference model
        path (``use_ref=True``), the caller wraps this in
        ``disable_adapter()`` + ``no_grad()``.

        Args:
            records: The batch of dataset records (for re-tokenization).
            gen_ids_list: List of generated token ID tensors, one per example.
            use_ref: If True, use the unwrapped model (for reference log-probs).

        Returns:
            Tensor of shape ``(B,)`` with per-example mean log-probs.
        """
        log_probs_list: List[torch.Tensor] = []
        max_seq = int(self.cfg.training.get("max_seq_length", 2048))

        for record, gen_ids in zip(records, gen_ids_list):
            # Re-tokenize just the prompt for this example.
            prompt_inputs = self._build_inputs(record)
            prompt_len = prompt_inputs["input_ids"].shape[1]
            gen_len = gen_ids.shape[0]

            gen_ids_2d = gen_ids.unsqueeze(0)  # (1, gen_len)
            full_ids = torch.cat(
                [prompt_inputs["input_ids"], gen_ids_2d], dim=1,
            )
            full_mask = torch.cat([
                prompt_inputs["attention_mask"],
                torch.ones(
                    (1, gen_len),
                    dtype=prompt_inputs["attention_mask"].dtype,
                    device=self.device,
                ),
            ], dim=1)

            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            extra_kwargs: Dict[str, torch.Tensor] = {}
            for k in ("pixel_values", "image_grid_thw"):
                if k in prompt_inputs:
                    extra_kwargs[k] = prompt_inputs[k]

            if full_ids.shape[1] > max_seq:
                full_ids = full_ids[:, :max_seq]
                full_mask = full_mask[:, :max_seq]
                labels = labels[:, :max_seq]

            with torch.amp.autocast(
                self.device.type,
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                if use_ref:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    outputs = unwrapped(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        labels=None,
                        **extra_kwargs,
                    )
                else:
                    outputs = self.model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        labels=None,
                        **extra_kwargs,
                    )

            logits = outputs.logits.float()
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(
                dim=-1,
                index=shift_labels.clamp(min=0).unsqueeze(-1),
            ).squeeze(-1)

            valid_mask = (shift_labels != -100).float()
            n_tokens = valid_mask.sum().clamp(min=1.0)
            seq_lp = (token_lp * valid_mask).sum() / n_tokens

            if use_ref:
                seq_lp = seq_lp.detach()
            log_probs_list.append(seq_lp)

        return torch.stack(log_probs_list)

    # Keep the old single-example methods for backward compat with tests.
    def _compute_seq_log_probs(
        self,
        prompt_inputs: Dict[str, torch.Tensor],
        gen_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token mean log-prob for the generated tokens (with grad).

        Concatenates prompt and generated tokens, runs a teacher-forcing
        forward pass, and sums log-probs over the generated token positions.

        Args:
            prompt_inputs: Tokenized prompt inputs (input_ids, attention_mask,
                          and optionally pixel_values, image_grid_thw).
            gen_ids: Generated token IDs, shape ``(1, gen_len)``.

        Returns:
            Scalar tensor: mean of log-probs over generated token positions.
        """
        prompt_len = prompt_inputs["input_ids"].shape[1]
        gen_len = gen_ids.shape[1]

        full_ids = torch.cat(
            [prompt_inputs["input_ids"], gen_ids], dim=1
        )
        full_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones(
                (1, gen_len),
                dtype=prompt_inputs["attention_mask"].dtype,
                device=self.device,
            ),
        ], dim=1)

        labels = full_ids.clone()
        labels[:, :prompt_len] = -100

        extra_kwargs: Dict[str, torch.Tensor] = {}
        for k in ("pixel_values", "image_grid_thw"):
            if k in prompt_inputs:
                extra_kwargs[k] = prompt_inputs[k]

        max_seq = int(self.cfg.training.get("max_seq_length", 2048))
        if full_ids.shape[1] > max_seq:
            logger.warning(
                "Sequence length %d exceeds max_seq_length=%d; truncating.",
                full_ids.shape[1], max_seq,
            )
            full_ids = full_ids[:, :max_seq]
            full_mask = full_mask[:, :max_seq]
            labels = labels[:, :max_seq]

        with torch.amp.autocast(
            self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                labels=None,
                **extra_kwargs,
            )

        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(
            dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)

        valid_mask = (shift_labels != -100).float()
        n_tokens = valid_mask.sum().clamp(min=1.0)
        return (token_lp * valid_mask).sum() / n_tokens

    def _compute_seq_log_probs_no_grad(
        self,
        prompt_inputs: Dict[str, torch.Tensor],
        gen_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Same as _compute_seq_log_probs but without gradient tracking.

        Used for the reference model (base model with adapter disabled).
        Detaches the result.

        Args:
            prompt_inputs: Tokenized prompt inputs.
            gen_ids: Generated token IDs, shape ``(1, gen_len)``.

        Returns:
            Scalar tensor (detached): mean of log-probs over generated tokens.
        """
        prompt_len = prompt_inputs["input_ids"].shape[1]
        gen_len = gen_ids.shape[1]

        full_ids = torch.cat(
            [prompt_inputs["input_ids"], gen_ids], dim=1
        )
        full_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones(
                (1, gen_len),
                dtype=prompt_inputs["attention_mask"].dtype,
                device=self.device,
            ),
        ], dim=1)

        labels = full_ids.clone()
        labels[:, :prompt_len] = -100

        extra_kwargs: Dict[str, torch.Tensor] = {}
        for k in ("pixel_values", "image_grid_thw"):
            if k in prompt_inputs:
                extra_kwargs[k] = prompt_inputs[k]

        max_seq = int(self.cfg.training.get("max_seq_length", 2048))
        if full_ids.shape[1] > max_seq:
            full_ids = full_ids[:, :max_seq]
            full_mask = full_mask[:, :max_seq]
            labels = labels[:, :max_seq]

        with torch.amp.autocast(
            self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            unwrapped = self.accelerator.unwrap_model(self.model)
            outputs = unwrapped(
                input_ids=full_ids,
                attention_mask=full_mask,
                labels=None,
                **extra_kwargs,
            )

        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(
            dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)

        valid_mask = (shift_labels != -100).float()
        n_tokens = valid_mask.sum().clamp(min=1.0)
        return ((token_lp * valid_mask).sum() / n_tokens).detach()

    def _save_checkpoint(self, step: int) -> str:
        """Save the LoRA adapter weights to ``{output_dir}/step_{step}/``.

        Args:
            step: Current optimizer step number.

        Returns:
            Absolute path to the checkpoint directory.
        """
        ckpt_dir = self.output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(
            str(ckpt_dir)
        )
        self.processor.save_pretrained(str(ckpt_dir))
        logger.info("Checkpoint saved to %s", ckpt_dir)
        return str(ckpt_dir)

