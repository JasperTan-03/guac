"""GRPO trainer with dynamic difficulty curriculum and MLflow logging.

The trainer implements a custom GRPO training loop for vision-language models
(Qwen2-VL by default) with the following per-step logic:

  1. Sample a batch of examples from the dataset using ``CurriculumSampler``.
  2. Generate ``group_size`` candidate responses per prompt via ``model.generate()``.
  3. Score each response with exact-match reward (``compute_reward``).
  4. Normalise rewards within each group (``normalize_rewards``).
  5. Compute per-sequence log-probs via teacher forcing (``compute_log_probs``).
  6. Compute GRPO loss, optionally adding a KL penalty (``grpo_loss``,
     ``compute_kl_penalty``).
  7. Accumulate gradients for ``gradient_accumulation_steps`` micro-steps,
     then clip and step the optimizer.
  8. Update ``CurriculumState.T`` once per optimizer step.
  9. Log metrics to MLflow every ``log_steps`` optimizer steps.
  10. Save a LoRA checkpoint every ``save_steps`` optimizer steps.

Compatible with: transformers>=4.45, peft>=0.12, torch>=2.3, mlflow>=2.10,
                 omegaconf>=2.3
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup

from guac.data.prep import load_jsonl
from guac.data.utils import decode_image
from guac.training.curriculum import CurriculumSampler, CurriculumState
from guac.training.rewards import (
    compute_kl_penalty,
    compute_reward,
    grpo_loss,
    is_informative_group,
    normalize_rewards,
)

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO trainer with dynamic difficulty curriculum and MLflow logging.

    Loads model and data according to the Hydra config, applies LoRA, and
    runs a custom GRPO training loop with gradient accumulation and an
    adaptive curriculum over example difficulty.

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
          batch_size: 2
          group_size: 4
          gradient_accumulation_steps: 4
          learning_rate: 1.0e-5
          weight_decay: 0.01
          max_grad_norm: 1.0
          num_train_steps: 5000
          warmup_steps: 100
          save_steps: 500
          log_steps: 10
          max_new_tokens: 128
          temperature: 0.7
          top_p: 0.9
          kl_coeff: 0.0
          sampling_mode: "gaussian"
          T_init: 0.5
          eta: 0.05
          alpha: 2.0
          beta: 0.5
          sigma: 0.15
          d_min: 0.0
          d_max: 1.0
          mlflow_tracking_uri: "mlruns"
          mlflow_experiment: "grpo_curriculum"

        data:
          scored_dir: "data/scored"

    Args:
        cfg: Full Hydra ``DictConfig`` containing ``model``, ``training``,
             and ``data`` sub-configs.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Set up model, LoRA, optimizer, scheduler, curriculum, and MLflow.

        Args:
            cfg: Full Hydra DictConfig (see class docstring for structure).

        Raises:
            FileNotFoundError: If the training JSONL file does not exist.
            ValueError: If no valid records remain after difficulty filtering.
            RuntimeError: If CUDA is unavailable and training is attempted.
        """
        self.cfg = cfg
        tcfg = cfg.training

        # ------------------------------------------------------------------
        # Accelerator / device / rank
        # ------------------------------------------------------------------
        # mixed_precision="no" because bf16 weights + manual autocast are
        # already managed by this class (see self._autocast_* below).
        # Letting accelerate layer its own bf16 autocast on top would
        # double-cast the policy forward pass.
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
        # Processor
        # TODO: adjust trust_remote_code and padding_side for VLMs other than Qwen2-VL.
        # ------------------------------------------------------------------
        logger.info("Loading processor from %s ...", cfg.model.name)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model.name,
            trust_remote_code=True,
        )
        if hasattr(self.processor, "tokenizer"):
            # Left-padding is required for correct generation with batch inference.
            self.processor.tokenizer.padding_side = "left"

        # ------------------------------------------------------------------
        # Base model (bfloat16)
        # TODO: Replace AutoModelForVision2Seq with the appropriate class for
        #       your VLM if the auto class does not resolve correctly.
        # ------------------------------------------------------------------
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_dtype = dtype_map.get(str(cfg.model.dtype), torch.bfloat16)

        # bfloat16 matmul isn't reliably supported on CPU / MPS.  Coerce to
        # float32 off-CUDA so the smoke test (and any CPU fallback) works.
        if model_dtype == torch.bfloat16 and self.device.type != "cuda":
            logger.warning(
                "Device %s does not reliably support bfloat16; coercing model "
                "dtype to float32 for this run.",
                self.device.type,
            )
            model_dtype = torch.float32
        self._model_dtype = model_dtype
        # Autocast is only useful on CUDA with bf16/fp16; elsewhere it's a
        # no-op contextmanager.
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
        # TODO: verify that target_modules match the attention projection names
        #       in your specific Qwen2-VL variant.
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
        # Optional frozen reference model for KL penalty
        # Only loaded when kl_coeff > 0 to preserve VRAM.
        # ------------------------------------------------------------------
        self.ref_model: Optional[torch.nn.Module] = None
        self.kl_coeff = float(tcfg.kl_coeff)
        if self.kl_coeff > 0.0:
            logger.info(
                "Loading frozen reference model for KL penalty "
                "(kl_coeff=%.4f) ...",
                self.kl_coeff,
            )
            # Load a clean copy of the *base* (non-LoRA) model as the reference.
            self.ref_model = AutoModelForVision2Seq.from_pretrained(
                cfg.model.name,
                torch_dtype=model_dtype,
                device_map={"": self.device},
                trust_remote_code=True,
            )
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)

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
        # Under DDP this returns a DistributedDataParallel wrapper.  The
        # wrapper's ``forward`` triggers gradient all-reduce at
        # ``backward`` time.  In single-process (rank=0, world_size=1)
        # this is a no-op passthrough.  The reference model (if any) is
        # intentionally NOT prepared — it's frozen, no optimizer, no
        # backward, so wrapping it would just waste registering unused
        # reducers.
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler,
        )

        # ------------------------------------------------------------------
        # MLflow (rank 0 only — other ranks leave _mlflow_run as None so
        # the logging guards below short-circuit cleanly).
        # ------------------------------------------------------------------
        self._mlflow_run = None
        if self.is_main:
            mlflow.set_tracking_uri(str(tcfg.mlflow_tracking_uri))
            mlflow.set_experiment(str(tcfg.mlflow_experiment))
            self._mlflow_run = mlflow.start_run()
            # Log the full resolved config as flat params.
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
            logger.info(
                "MLflow run started: %s", self._mlflow_run.info.run_id
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full GRPO training loop.

        The outer loop counts *optimizer steps* (``global_step``).  Each
        optimizer step consists of ``gradient_accumulation_steps`` micro-steps,
        each of which processes a fresh batch sampled from the curriculum.

        Steps per optimizer step:
            1. Sample ``batch_size`` examples from ``CurriculumSampler``.
            2. Generate ``group_size`` responses per prompt with
               ``model.generate()`` (no-grad, eval mode).
            3. Compute exact-match reward for each response.
            4. Normalise rewards within each group.
            5. Compute per-sequence log-probs via teacher forcing (train mode).
            6. Compute GRPO loss; add optional KL penalty.
            7. Scale by ``1 / gradient_accumulation_steps`` and backpropagate.
            8. After accumulating: clip gradients, step optimizer and scheduler.
            9. Update ``CurriculumState.T`` once per optimizer step.
            10. Log metrics to MLflow every ``log_steps`` steps.
            11. Save LoRA checkpoint every ``save_steps`` steps.

        At the end of training the final LoRA adapter is saved to
        ``{output_dir}/final/`` and the path is logged to MLflow.
        """
        tcfg = self.cfg.training
        num_train_steps = int(tcfg.num_train_steps)
        grad_accum = int(tcfg.gradient_accumulation_steps)
        batch_size = int(tcfg.batch_size)
        group_size = int(tcfg.group_size)
        log_steps = int(tcfg.log_steps)
        save_steps = int(tcfg.save_steps)
        max_grad_norm = float(tcfg.max_grad_norm)
        max_new_tokens = int(tcfg.max_new_tokens)
        temperature = float(tcfg.temperature)
        top_p = float(tcfg.top_p)

        global_step = 0     # completed optimizer steps
        micro_step = 0      # forward-backward passes (resets at each optimizer step)
        running_loss = 0.0
        running_reward = 0.0
        running_informative_groups = 0
        running_flat_groups = 0
        # Tracks whether any informative micro-step contributed gradient in
        # the current grad_accum window.  When every micro-step in the
        # window is flat, we advance global_step and the curriculum anyway
        # so the loop always terminates at num_train_steps, but we skip
        # optimizer.step (there's nothing to update).
        had_gradient_this_window = False

        self.model.train()
        self.optimizer.zero_grad()

        logger.info(
            "Starting GRPO training: %d steps, batch_size=%d, group_size=%d, "
            "grad_accum=%d, sampling_mode=%s, T_init=%.3f",
            num_train_steps,
            batch_size,
            group_size,
            grad_accum,
            tcfg.sampling_mode,
            self.curriculum.T,
        )

        while global_step < num_train_steps:
            # ----------------------------------------------------------
            # Step 1: Sample batch
            # ----------------------------------------------------------
            batch_indices = self.sampler.sample(batch_size, self.curriculum.T)
            batch_records = [self.records[i] for i in batch_indices]

            # Accumulators for this micro-step.
            all_log_probs: List[torch.Tensor] = []
            all_advantages: List[float] = []
            batch_raw_rewards: List[float] = []
            # KL data: (prompt_input_ids, prompt_attn_mask, generated_ids_list,
            # vision_kwargs).  Only populated for *informative* groups — flat
            # groups contribute nothing to the loss and must not contribute to
            # the KL term either.
            kl_data: List[
                Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
            ] = []
            # Count informative vs flat groups in this micro-step so the
            # diagnostic can surface advantage collapse instead of silently
            # producing zero gradients.
            informative_groups = 0
            flat_groups = 0
            # Capture the first sample each micro-step for diagnostic logging.
            _log_sample: Optional[Dict[str, Any]] = None

            for record in batch_records:
                ground_truth: str = str(record.get("answer", ""))

                # ----------------------------------------------------------
                # Step 2: Build prompt inputs and generate group_size responses
                # ----------------------------------------------------------
                prompt_inputs = self._build_inputs(record)
                prompt_len = prompt_inputs["input_ids"].shape[1]

                group_rewards: List[float] = []
                group_generated_ids: List[torch.Tensor] = []

                pad_token_id = (
                    self.processor.tokenizer.pad_token_id
                    if self.processor.tokenizer.pad_token_id is not None
                    else self.processor.tokenizer.eos_token_id
                )

                # ----------------------------------------------------------
                # Step 2 (cont.): Rollout generation — no_grad, eval mode.
                # Under DDP, ``self.model`` is a DistributedDataParallel
                # wrapper that only exposes ``forward()`` — ``generate()``
                # lives on the inner module.  ``unwrap_model`` is a no-op
                # in single-process mode.
                # ----------------------------------------------------------
                self.model.eval()
                gen_model = self.accelerator.unwrap_model(self.model)
                with torch.no_grad():
                    for _ in range(group_size):
                        gen_out = gen_model.generate(
                            **prompt_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                        # Strip the prompt prefix to keep only new tokens.
                        gen_ids = gen_out[:, prompt_len:]  # (1, gen_len)
                        group_generated_ids.append(gen_ids)

                        # Step 3: Compute reward
                        decoded = self.processor.tokenizer.decode(
                            gen_ids[0], skip_special_tokens=True
                        ).strip()
                        r = compute_reward(decoded, ground_truth)
                        group_rewards.append(r)

                        # Capture first sample per micro-step for logging.
                        if _log_sample is None:
                            _log_sample = {
                                "pred": decoded,
                                "gt": ground_truth,
                                "reward": r,
                            }

                batch_raw_rewards.extend(group_rewards)

                # Step 4: Detect flat groups.  If every rollout in this group
                # got the same reward (all 0 or all 1), normalising would
                # yield all-zero advantages and the GRPO term for this group
                # would be exactly 0 — a wasted forward/backward pass.  Skip
                # it, but still include its rewards in batch_raw_rewards so
                # the curriculum update sees the true success rate.
                if not is_informative_group(group_rewards):
                    flat_groups += 1
                    continue
                informative_groups += 1

                # Step 4 (cont.): Normalise rewards within the informative group.
                norm_rewards = normalize_rewards(group_rewards)

                # Stash for optional KL computation (including vision tensors).
                vision_kwargs: Dict[str, torch.Tensor] = {
                    k: prompt_inputs[k] for k in ("pixel_values", "image_grid_thw")
                    if k in prompt_inputs
                }
                kl_data.append((
                    prompt_inputs["input_ids"],
                    prompt_inputs["attention_mask"],
                    group_generated_ids,
                    vision_kwargs,
                ))

                # ----------------------------------------------------------
                # Steps 5–6: Teacher-forcing log-probs (train mode)
                # ----------------------------------------------------------
                self.model.train()
                for gen_ids, advantage in zip(
                    group_generated_ids, norm_rewards
                ):
                    gen_len = gen_ids.shape[1]
                    full_ids = torch.cat(
                        [prompt_inputs["input_ids"], gen_ids], dim=1
                    )  # (1, prompt_len + gen_len)

                    # Preserve prompt padding mask; generated tokens always attend.
                    full_mask = torch.cat([
                        prompt_inputs["attention_mask"],
                        torch.ones((1, gen_len), dtype=prompt_inputs["attention_mask"].dtype, device=self.device),
                    ], dim=1)

                    # Mask prompt positions in labels so they do not contribute
                    # to the log-prob computation.
                    labels = full_ids.clone()
                    labels[:, :prompt_len] = -100

                    # Pass pixel_values and image_grid_thw through if present.
                    extra_kwargs: Dict[str, torch.Tensor] = {}
                    for k in ("pixel_values", "image_grid_thw"):
                        if k in prompt_inputs:
                            extra_kwargs[k] = prompt_inputs[k]

                    # Standard GRPO: use the current policy's log-prob directly.
                    # L = -E[Â_i · log π_θ(y_i | x_i)]
                    # _compute_seq_log_probs returns mean per-token log-prob,
                    # so the signal is already length-normalised.
                    log_prob = self._compute_seq_log_probs(
                        full_ids, full_mask, labels, extra_kwargs
                    )
                    all_log_probs.append(log_prob)
                    all_advantages.append(advantage)

            # ----------------------------------------------------------
            # Step 6 (cont.): GRPO loss
            # ----------------------------------------------------------
            # Always update observability counters (even on all-flat).
            running_informative_groups += informative_groups
            running_flat_groups += flat_groups
            running_reward += (
                sum(batch_raw_rewards) / len(batch_raw_rewards)
                if batch_raw_rewards else 0.0
            )

            # ----------------------------------------------------------
            # DDP safety: every rank must either call backward() together
            # or skip it together.  If rank A does backward (firing DDP's
            # gradient allreduce) while rank B does not, B hangs.  A cheap
            # boolean all-reduce lets us make a unanimous decision.
            # ----------------------------------------------------------
            local_has_signal = torch.tensor(
                1 if all_log_probs else 0,
                device=self.device, dtype=torch.long,
            )
            if self.world_size > 1:
                global_has_signal = self.accelerator.reduce(
                    local_has_signal, reduction="sum",
                ).item() > 0
            else:
                global_has_signal = bool(local_has_signal.item())

            if all_log_probs:
                lp_tensor = torch.stack(all_log_probs)        # (N,)
                adv_tensor = torch.tensor(
                    all_advantages,
                    dtype=torch.float32,
                    device=self.device,
                )
                loss = grpo_loss(lp_tensor, adv_tensor)

                # Optional KL penalty, gated by kl_coeff > 0.
                if self.kl_coeff > 0.0 and self.ref_model is not None:
                    kl_total = self._compute_batch_kl(kl_data)
                    loss = loss + self.kl_coeff * kl_total

                # ------------------------------------------------------
                # Step 7: Gradient accumulation
                # ------------------------------------------------------
                scaled_loss = loss / grad_accum
                self.accelerator.backward(scaled_loss)
                running_loss += loss.item()
                had_gradient_this_window = True
            elif global_has_signal:
                # This rank is flat but at least one other rank has signal
                # and is calling backward().  We MUST participate in DDP's
                # gradient allreduce or we deadlock.  Synthesize a
                # differentiable-zero loss that touches every trainable
                # parameter, so DDP's reducers fire uniformly and we
                # contribute exactly zero gradient.
                trainable = [
                    p for p in self.model.parameters() if p.requires_grad
                ]
                if trainable:
                    dummy = torch.stack([p.sum() for p in trainable]).sum() * 0.0
                else:  # pragma: no cover — LoRA always has trainable params
                    dummy = torch.zeros((), device=self.device, requires_grad=True)
                self.accelerator.backward(dummy / grad_accum)
                # running_loss stays at 0 for this micro-step — honest
                # signal for MLflow (flat_groups_frac captures the collapse).
                had_gradient_this_window = True
            else:
                # Every rank is flat this micro-step: skip backward
                # entirely.  Advance micro_step/global_step below so the
                # loop terminates at num_train_steps regardless.
                logger.warning(
                    "All %d groups on rank %d were flat this micro-step "
                    "(and no other rank had signal); skipping backward. "
                    "If this persists, check reward shaping or curriculum.",
                    flat_groups, self.rank,
                )

            micro_step += 1

            # ----------------------------------------------------------
            # Optimizer step — fire every grad_accum micro-steps.
            # When every micro-step in the window was flat, skip
            # optimizer.step (no gradient to apply) but still advance the
            # scheduler, curriculum, and global_step so training always
            # terminates cleanly at num_train_steps.
            # ----------------------------------------------------------
            if micro_step % grad_accum == 0:
                if had_gradient_this_window:
                    # accelerator.clip_grad_norm_ unwraps the DDP wrapper
                    # and works correctly in single-process mode too.
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1

                # ------------------------------------------------------
                # Step 8: All-reduce running metrics so every rank's
                # curriculum.update receives the same R_avg (otherwise T
                # drifts across ranks and the 8 gaussians sample from
                # incompatible difficulty distributions).
                # ------------------------------------------------------
                R_avg_local = running_reward / grad_accum
                if self.world_size > 1:
                    r_t = torch.tensor(R_avg_local, device=self.device)
                    R_avg = self.accelerator.reduce(r_t, reduction="mean").item()
                else:
                    R_avg = R_avg_local
                self.curriculum.update(R_avg)

                # ------------------------------------------------------
                # Step 9: Logging — all-reduce metrics for MLflow, then
                # log on rank 0 only.  loss/reward use mean-reduction,
                # flat/informative counts use sum-reduction.
                # ------------------------------------------------------
                if self.world_size > 1:
                    loss_t = torch.tensor(
                        running_loss / grad_accum, device=self.device,
                    )
                    flat_t = torch.tensor(
                        float(running_flat_groups), device=self.device,
                    )
                    info_t = torch.tensor(
                        float(running_informative_groups), device=self.device,
                    )
                    avg_loss = self.accelerator.reduce(
                        loss_t, reduction="mean",
                    ).item()
                    total_flat = int(self.accelerator.reduce(
                        flat_t, reduction="sum",
                    ).item())
                    total_info = int(self.accelerator.reduce(
                        info_t, reduction="sum",
                    ).item())
                else:
                    avg_loss = running_loss / grad_accum
                    total_flat = running_flat_groups
                    total_info = running_informative_groups

                total_groups = total_info + total_flat
                flat_frac = (
                    total_flat / total_groups if total_groups > 0 else 0.0
                )
                if global_step % log_steps == 0 and self.is_main:
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "step=%d | loss=%.4f | reward=%.4f | T=%.4f | lr=%.2e "
                        "| flat_groups=%d/%d",
                        global_step,
                        avg_loss,
                        R_avg,
                        self.curriculum.T,
                        current_lr,
                        total_flat,
                        total_groups,
                    )
                    if _log_sample is not None:
                        logger.info(
                            "  sample | pred=%r | gt=%r | reward=%.1f",
                            _log_sample["pred"][:120],
                            _log_sample["gt"][:60],
                            _log_sample["reward"],
                        )
                    mlflow.log_metrics(
                        {
                            "train/loss": avg_loss,
                            "train/reward": R_avg,
                            "curriculum/T": self.curriculum.T,
                            "train/learning_rate": current_lr,
                            "train/flat_groups_frac": flat_frac,
                            "train/informative_groups": total_info,
                            "train/flat_groups": total_flat,
                        },
                        step=global_step,
                    )

                # Step 10: Checkpoint — rank 0 only; wait for all ranks
                # to finish this optimizer step first so the saved
                # weights are consistent with a completed backward.
                if global_step % save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    if self.is_main:
                        ckpt_path = self._save_checkpoint(global_step)
                        mlflow.log_artifact(ckpt_path)

                # Reset running accumulators for next optimizer step.
                running_loss = 0.0
                running_reward = 0.0
                running_informative_groups = 0
                running_flat_groups = 0
                had_gradient_this_window = False

            # Free fragmented GPU memory after each micro-step.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if global_step >= num_train_steps:
                break

        # ------------------------------------------------------------------
        # Final checkpoint — rank 0 only.  Barrier first so every rank
        # has exited the training loop before we serialise weights.
        # ------------------------------------------------------------------
        self.accelerator.wait_for_everyone()
        if self.is_main:
            final_dir = self.output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            # unwrap_model so the saved adapter is plain PEFT weights,
            # not a DDP-wrapped state_dict.
            self.accelerator.unwrap_model(self.model).save_pretrained(
                str(final_dir)
            )
            self.processor.save_pretrained(str(final_dir))
            logger.info(
                "Training complete. Final checkpoint saved to %s", final_dir,
            )
            mlflow.log_artifact(str(final_dir))
            mlflow.end_run()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_inputs(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenise a single record (image + prompt) using the processor.

        Constructs the Qwen2-VL chat template from the record fields and
        calls the processor to produce tensors ready for ``model.generate()``
        or teacher-forcing forward passes.

        Args:
            record: A single dataset record dict.  Expected keys:
                - ``"prompt"`` or ``"question"``: the question text (str).
                - ``"image"``: base64-encoded PNG string, or None.

        Returns:
            Dict of processor outputs (``input_ids``, ``attention_mask``,
            and optionally ``pixel_values``, ``image_grid_thw``), all on
            ``self.device`` as ``torch.Tensor`` values.

        # TODO: adjust for your VLM processor if the chat template format or
        #       image-passing convention differs from Qwen2-VL.
        """
        question = str(record.get("prompt", record.get("question", "")))
        image_b64: Optional[str] = record.get("image")
        pil_image = decode_image(image_b64) if image_b64 else None

        # Build the messages list in Qwen2-VL chat format.
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

        # TODO: adjust apply_chat_template kwargs for your VLM processor.
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

    def _compute_seq_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        extra_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run teacher-forcing forward pass and return the scalar log-prob sum.

        This wrapper passes any vision-specific tensors (``pixel_values``,
        ``image_grid_thw``) through to the model alongside the standard
        language-modelling inputs.

        Args:
            input_ids: Full sequence token IDs, shape ``(1, L)``.
            attention_mask: Attention mask, shape ``(1, L)``.
            labels: Token IDs with prompt positions masked to ``-100``,
                    shape ``(1, L)``.
            extra_kwargs: Additional tensors to forward (e.g.
                          ``pixel_values``, ``image_grid_thw``).

        Returns:
            Scalar tensor: sum of log-probs over generated (non-masked)
            token positions.
        """
        with torch.amp.autocast(
            self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                **extra_kwargs,
            )

        logits = outputs.logits.float()          # upcasted for stability

        shift_logits = logits[:, :-1, :]         # (1, L-1, V)
        shift_labels = labels[:, 1:]             # (1, L-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(
            dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)                            # (1, L-1)

        valid_mask = (shift_labels != -100).float()
        n_tokens = valid_mask.sum().clamp(min=1.0)
        return (token_lp * valid_mask).sum() / n_tokens  # mean per-token log-prob

    def _compute_batch_kl(
        self,
        kl_data: List[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        """Compute mean KL(policy || reference) over all responses in the batch.

        For each (prompt, generated sequence) pair, concatenates prompt and
        generation, runs both policy and reference models, then calls
        ``compute_kl_penalty`` on the generated-token slice of the logits.

        Args:
            kl_data: List of ``(prompt_ids, prompt_mask, gen_ids_list,
                     vision_kwargs)`` tuples — one per prompt in the batch.

        Returns:
            Scalar mean KL divergence tensor (with gradient through the policy
            model parameters).
        """
        kl_total = torch.tensor(0.0, device=self.device)
        n_terms = 0

        for p_ids, p_mask, gen_ids_list, v_kwargs in kl_data:
            for gen_ids in gen_ids_list:
                full_ids = torch.cat([p_ids, gen_ids], dim=1)
                gen_len = gen_ids.shape[1]
                # Preserve the prompt's padding mask; generated tokens attend.
                full_mask = torch.cat([
                    p_mask,
                    torch.ones((1, gen_len), dtype=p_mask.dtype, device=self.device),
                ], dim=1)

                # Reference forward pass (frozen, no grad).
                with torch.no_grad():
                    ref_logits = self.ref_model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        **v_kwargs,
                    ).logits[:, -gen_len:, :].float()

                # Policy forward pass (with grad for loss backprop).
                with torch.amp.autocast(
                    self.device.type,
                    dtype=self._autocast_dtype,
                    enabled=self._autocast_enabled,
                ):
                    policy_logits = self.model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        **v_kwargs,
                    ).logits[:, -gen_len:, :].float()

                policy_lp = F.log_softmax(policy_logits, dim=-1)
                ref_lp = F.log_softmax(ref_logits, dim=-1)

                kl_total = kl_total + compute_kl_penalty(policy_lp, ref_lp)
                n_terms += 1

        if n_terms > 0:
            kl_total = kl_total / n_terms

        return kl_total

    def _save_checkpoint(self, step: int) -> str:
        """Save the LoRA adapter weights to ``{output_dir}/step_{step}/``.

        Also saves the processor (tokenizer + image processor) alongside the
        adapter so the checkpoint directory is self-contained.

        Args:
            step: Current optimizer step number, used to name the directory.

        Returns:
            Absolute path to the checkpoint directory as a string.
        """
        ckpt_dir = self.output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # unwrap_model so the adapter directory contains plain PEFT
        # weights, not a DDP-wrapped state_dict.  No-op single-process.
        self.accelerator.unwrap_model(self.model).save_pretrained(
            str(ckpt_dir)
        )
        self.processor.save_pretrained(str(ckpt_dir))
        logger.info("Checkpoint saved to %s", ckpt_dir)
        return str(ckpt_dir)
