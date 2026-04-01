import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ============================================================
# Phase 3 — GRPO Training with Dynamic Curriculum Learning
# Model: Qwen/Qwen2-VL-7B-Instruct
# Requires: transformers>=4.45, peft>=0.12, torch>=2.3, pillow
# ============================================================

import argparse
import base64
import importlib.util
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType  # peft>=0.12
from transformers import (  # transformers>=4.45
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Argument Parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRPO curriculum training script.

    Returns:
        argparse.Namespace: Parsed argument namespace with all hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Phase 3 — GRPO training with dynamic curriculum for Qwen2-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model & Data ---
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/poc_scored_data.jsonl",
        help="Path to JSONL dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving LoRA checkpoints.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Dataset column containing base64-encoded images.",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default="prompt",
        help="Dataset column containing the question / prompt text.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="answer",
        help="Dataset column containing the ground-truth answer.",
    )
    parser.add_argument(
        "--difficulty_column",
        type=str,
        default="difficulty",
        help="Dataset column containing float difficulty scores.",
    )

    # --- Curriculum ---
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=["baseline", "gaussian"],
        default="gaussian",
        help="Curriculum sampling strategy.",
    )
    parser.add_argument(
        "--T_init",
        type=float,
        default=0.5,
        help="Initial target difficulty T.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="Curriculum update step size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Sensitivity to reward deviation in curriculum update.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Target reward baseline for curriculum update.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="Bandwidth (std dev) for Gaussian sampling mode.",
    )
    parser.add_argument(
        "--d_min",
        type=float,
        default=0.0,
        help="Minimum difficulty value (clip lower bound).",
    )
    parser.add_argument(
        "--d_max",
        type=float,
        default=1.0,
        help="Maximum difficulty value (clip upper bound).",
    )

    # --- Training ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of prompts per step (before group expansion).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="AdamW peak learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1000,
        help="Total number of optimizer steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Linear warmup steps for LR scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save LoRA checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log metrics every N optimizer steps.",
    )

    # --- GRPO ---
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Number of responses to generate per prompt (GRPO group size G).",
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.0,
        help="KL penalty coefficient against frozen reference model (0 = disabled).",
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        default=True,
        help="Skip vLLM; use model.generate() for rollouts (default=True for PoC).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollout generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling probability.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per rollout.",
    )

    # --- Reward ---
    parser.add_argument(
        "--reward_fn_path",
        type=str,
        default=None,
        help=(
            "Optional path to a Python file exporting "
            "`verify(prediction: str, ground_truth: str, example: dict) -> float`."
        ),
    )

    return parser.parse_args()


# ============================================================
# 2. CurriculumState dataclass
# ============================================================

@dataclass
class CurriculumState:
    """Tracks and updates the current target difficulty T.

    The update rule after each optimizer step is:
        T_new = clip(T + eta * tanh(alpha * (R_avg - beta)), d_min, d_max)

    Attributes:
        T: Current target difficulty in [d_min, d_max].
        eta: Step size / learning rate for the curriculum update.
        alpha: Sensitivity parameter scaling the reward deviation.
        beta: Baseline reward level at which T does not change.
        d_min: Lower bound for T.
        d_max: Upper bound for T.
    """

    T: float = 0.5
    eta: float = 0.05
    alpha: float = 2.0
    beta: float = 0.5
    d_min: float = 0.0
    d_max: float = 1.0

    def update(self, R_avg: float) -> None:
        """Update target difficulty T based on average batch reward.

        Args:
            R_avg: Mean reward over the current batch's group responses.
        """
        delta = self.eta * math.tanh(self.alpha * (R_avg - self.beta))
        self.T = float(max(self.d_min, min(self.d_max, self.T + delta)))


# ============================================================
# 3. CurriculumSampler
# ============================================================

class CurriculumSampler:
    """Samples dataset indices according to the current target difficulty T.

    Two modes are supported:

    - **baseline**: Greedily selects the B examples with smallest
      ``|d_i - T|`` (ties broken by random shuffle).
    - **gaussian**: Assigns probability proportional to
      ``exp(-(d_i - T)^2 / (2 * sigma^2))`` and samples without
      replacement using ``torch.multinomial``.

    Args:
        difficulties: List of per-example difficulty scores (float in [0, 1]).
        mode: ``'baseline'`` or ``'gaussian'``.
        sigma: Bandwidth for Gaussian mode (ignored in baseline mode).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        difficulties: List[float],
        mode: str = "gaussian",
        sigma: float = 0.15,
        seed: int = 42,
    ) -> None:
        if mode not in ("baseline", "gaussian"):
            raise ValueError(f"sampling_mode must be 'baseline' or 'gaussian', got {mode!r}")
        self.difficulties = difficulties
        self.mode = mode
        self.sigma = sigma
        self.rng = random.Random(seed)
        # Pre-compute tensor of difficulties for fast ops
        self._diff_tensor = torch.tensor(difficulties, dtype=torch.float32)

    def sample(self, batch_size: int, T: float) -> List[int]:
        """Return a list of ``batch_size`` dataset indices for the current T.

        Args:
            batch_size: Number of indices to return.
            T: Current target difficulty.

        Returns:
            List of integer dataset indices.

        Raises:
            ValueError: If ``batch_size`` exceeds the dataset size.
        """
        n = len(self.difficulties)
        if batch_size > n:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds dataset size ({n}). "
                "Reduce --batch_size or provide a larger dataset."
            )

        if self.mode == "baseline":
            return self._sample_baseline(batch_size, T)
        else:
            return self._sample_gaussian(batch_size, T)

    def _sample_baseline(self, batch_size: int, T: float) -> List[int]:
        """Baseline mode: top-B nearest neighbours by |d_i - T|.

        Args:
            batch_size: Number of indices to select.
            T: Current target difficulty.

        Returns:
            List of selected integer indices.
        """
        distances = [(abs(d - T), i) for i, d in enumerate(self.difficulties)]
        # Shuffle first to randomise tie-breaking, then stable-sort by distance
        self.rng.shuffle(distances)
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:batch_size]]

    def _sample_gaussian(self, batch_size: int, T: float) -> List[int]:
        """Gaussian mode: weighted sampling without replacement.

        Args:
            batch_size: Number of indices to sample.
            T: Current target difficulty.

        Returns:
            List of sampled integer indices.
        """
        # w_i = exp(-(d_i - T)^2 / (2 * sigma^2))
        diff = self._diff_tensor - T
        log_w = -(diff ** 2) / (2.0 * self.sigma ** 2)
        # Subtract max for numerical stability before exp
        log_w = log_w - log_w.max()
        weights = torch.exp(log_w)
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False)
        return indices.tolist()


# ============================================================
# 4. Dataset Loading
# ============================================================

def load_dataset_from_jsonl(
    path: str,
    difficulty_col: str = "difficulty",
) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file and filter out rows with null difficulty.

    Args:
        path: Filesystem path to the ``.jsonl`` file.
        difficulty_col: Name of the column holding float difficulty scores.

    Returns:
        List of record dicts, each guaranteed to have a non-null ``difficulty_col``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no valid records remain after filtering.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path!r}. "
            "Pass a valid path via --dataset_path."
        )

    records: List[Dict[str, Any]] = []
    skipped = 0
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)
                skipped += 1
                continue

            if record.get(difficulty_col) is None:
                skipped += 1
                continue
            records.append(record)

    if not records:
        raise ValueError(
            f"No valid records in {path!r} after filtering null '{difficulty_col}'. "
            "Check your dataset."
        )

    logger.info(
        "Loaded %d records from %s (skipped %d with null difficulty).",
        len(records),
        path,
        skipped,
    )
    return records


# ============================================================
# 5. Image Decoding
# ============================================================

def decode_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    """Decode a base64-encoded PNG/JPEG string into a PIL Image.

    Args:
        b64_str: Base64-encoded image bytes, or ``None``.

    Returns:
        A PIL ``Image`` in RGB mode, or ``None`` if ``b64_str`` is falsy.
    """
    if not b64_str:
        return None
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Failed to decode image (%s); treating as text-only.", exc)
        return None


# ============================================================
# 6. Reward Computation
# ============================================================

def _load_custom_reward_fn(path: str):
    """Dynamically import a ``verify`` function from a user-supplied file.

    Args:
        path: Filesystem path to a ``.py`` file that defines
              ``verify(prediction: str, ground_truth: str, example: dict) -> float``.

    Returns:
        The callable ``verify`` function.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file does not export ``verify``.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"--reward_fn_path {path!r} does not exist."
        )
    spec = importlib.util.spec_from_file_location("custom_reward_module", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "verify"):
        raise AttributeError(
            f"Custom reward file {path!r} must define "
            "`verify(prediction: str, ground_truth: str, example: dict) -> float`."
        )
    return module.verify


def compute_reward(
    prediction: str,
    ground_truth: str,
    example: Optional[Dict[str, Any]] = None,
    custom_fn=None,
) -> float:
    """Compute reward for a single (prediction, ground_truth) pair.

    If a custom reward function is provided it takes precedence.
    Otherwise uses exact string match (after normalising whitespace and case).

    Args:
        prediction: The model-generated answer string.
        ground_truth: The reference answer string.
        example: The full example dict (passed to custom reward fn if provided).
        custom_fn: Optional callable with signature
                   ``(prediction, ground_truth, example) -> float``.

    Returns:
        Reward scalar (float), typically in [0.0, 1.0].
    """
    if custom_fn is not None:
        return float(custom_fn(prediction, ground_truth, example or {}))

    # Exact match with normalisation
    pred_norm = " ".join(prediction.strip().lower().split())
    gt_norm = " ".join(ground_truth.strip().lower().split())
    return 1.0 if pred_norm == gt_norm else 0.0


def normalize_rewards(rewards: List[float]) -> List[float]:
    """Normalise a list of rewards within a group (zero-mean, unit-variance).

    Args:
        rewards: Raw reward scalars for one group (G responses to one prompt).

    Returns:
        Normalised rewards of the same length.
    """
    if len(rewards) == 0:
        raise ValueError("Cannot normalise an empty reward list.")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance) + 1e-8
    return [(r - mean) / std for r in rewards]


# ============================================================
# 7. Log-probability Computation
# ============================================================

def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of log-probabilities of label tokens under the model.

    Runs the model in teacher-forcing mode and gathers the log-prob of
    each token in ``labels`` (positions where ``labels != -100``).

    Args:
        model: The policy model (with or without LoRA adapters).
        input_ids: Token ids of shape ``(1, seq_len)``.
        labels: Token ids of shape ``(1, seq_len)``; positions to *ignore*
                must be set to ``-100``.
        attention_mask: Binary mask of shape ``(1, seq_len)``.

    Returns:
        Scalar tensor: sum of log-probs over all non-ignored label positions.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We compute loss manually
        )
    # logits: (1, seq_len, vocab_size)
    logits = outputs.logits.float()  # upcasted for numerical stability

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]          # (1, seq_len-1, vocab)
    shift_labels = labels[:, 1:]              # (1, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, seq_len-1, vocab)

    # Gather log-prob of the actual next token
    # shape: (1, seq_len-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)

    # Mask out positions where labels == -100 (prompt / padding)
    valid_mask = (shift_labels != -100).float()
    summed = (token_log_probs * valid_mask).sum()
    return summed


# ============================================================
# 8. GRPO Loss
# ============================================================

def grpo_loss(
    log_probs_list: List[torch.Tensor],
    advantages: List[float],
) -> torch.Tensor:
    """Compute the GRPO policy gradient loss.

    Loss = -mean(advantage_i * log_prob_i)  averaged over all responses.

    Args:
        log_probs_list: List of scalar tensors, one per response in the batch
                        (concatenated across all prompts).
        advantages: Corresponding normalised advantage values (float per response).

    Returns:
        Scalar loss tensor (requires grad through ``log_probs_list``).

    Raises:
        ValueError: If ``log_probs_list`` and ``advantages`` have different lengths.
    """
    if len(log_probs_list) != len(advantages):
        raise ValueError(
            f"log_probs_list length {len(log_probs_list)} != "
            f"advantages length {len(advantages)}."
        )
    if not log_probs_list:
        raise ValueError("grpo_loss received empty log_probs_list.")

    adv_tensor = torch.tensor(
        advantages,
        dtype=torch.float32,
        device=log_probs_list[0].device,
    )
    lp_stack = torch.stack(log_probs_list)  # (N,)
    loss = -(adv_tensor * lp_stack).mean()
    return loss


# ============================================================
# 9. LR Scheduler helper
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine LR schedule with linear warmup.

    Args:
        optimizer: The AdamW optimizer.
        warmup_steps: Number of steps for linear LR warm-up from 0.
        total_steps: Total training steps (used for cosine decay).

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 10. Processor helper — build input for one example
# ============================================================

def build_model_inputs(
    processor,
    question: str,
    image: Optional[Image.Image],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Tokenise a (question, image) pair using the Qwen2-VL processor.

    Follows the Qwen2-VL chat template: wraps the user turn in the
    ``<|im_start|>user ... <|im_end|>`` structure expected by the model.

    Args:
        processor: ``AutoProcessor`` instance for Qwen2-VL.
        question: The prompt / question text.
        image: Optional PIL Image. If ``None``, text-only input is produced.
        device: Target device for returned tensors.

    Returns:
        Dict with keys ``input_ids`` and ``attention_mask``
        (and ``pixel_values`` / ``image_grid_thw`` when an image is present),
        all on ``device``.

    # TODO: adjust for your VLM processor if using a model other than Qwen2-VL.
    """
    # Build the messages list in Qwen2-VL chat format
    content: List[Dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    # Apply chat template to get the formatted text string
    # TODO: adjust for your VLM processor if the chat template API differs.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images_list = [image] if image is not None else None

    inputs = processor(
        text=[text],
        images=images_list,
        return_tensors="pt",
        padding=True,
    )

    return {k: v.to(device) for k, v in inputs.items()}


# ============================================================
# 11. KL divergence helper (optional)
# ============================================================

def compute_kl_penalty(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute mean per-token KL(policy || reference) over generated tokens.

    Args:
        policy_model: Current policy (LoRA-adapted) model.
        ref_model: Frozen reference model (original weights).
        input_ids: Input prompt token ids, shape ``(1, prompt_len)``.
        attention_mask: Attention mask for input, shape ``(1, prompt_len)``.
        generated_ids: Generated token ids, shape ``(1, gen_len)``.

    Returns:
        Scalar tensor representing mean KL divergence over generated positions.
    """
    # Full sequence = prompt + generation
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    gen_len = generated_ids.shape[1]
    full_mask = torch.ones(
        full_ids.shape, dtype=attention_mask.dtype, device=full_ids.device
    )

    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=full_ids,
            attention_mask=full_mask,
        ).logits[:, -gen_len:, :]

    policy_logits = policy_model(
        input_ids=full_ids,
        attention_mask=full_mask,
    ).logits[:, -gen_len:, :]

    ref_logits = ref_logits.float()
    policy_logits = policy_logits.float()

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)

    # KL(policy || ref) = sum_v [ policy * (log_policy - log_ref) ]
    kl = F.kl_div(
        input=policy_log_probs,
        target=ref_probs,
        reduction="batchmean",
        log_target=False,
    )
    return kl


# ============================================================
# 12. Main training loop
# ============================================================

def main() -> None:
    """Entry point — wires dataset loading, model init, and GRPO training loop."""
    args = parse_args()

    # ----------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if device.type != "cuda":
        logger.warning(
            "No CUDA device found. Training on CPU will be extremely slow."
        )

    # ----------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    logger.info("Loading dataset from %s ...", args.dataset_path)
    records = load_dataset_from_jsonl(args.dataset_path, args.difficulty_column)
    difficulties = [float(r[args.difficulty_column]) for r in records]

    # ----------------------------------------------------------
    # Curriculum state and sampler
    # ----------------------------------------------------------
    curriculum = CurriculumState(
        T=args.T_init,
        eta=args.eta,
        alpha=args.alpha,
        beta=args.beta,
        d_min=args.d_min,
        d_max=args.d_max,
    )

    sampler = CurriculumSampler(
        difficulties=difficulties,
        mode=args.sampling_mode,
        sigma=args.sigma,
        seed=args.seed,
    )

    # ----------------------------------------------------------
    # Optional custom reward function
    # ----------------------------------------------------------
    custom_reward_fn = None
    if args.reward_fn_path is not None:
        logger.info("Loading custom reward function from %s", args.reward_fn_path)
        custom_reward_fn = _load_custom_reward_fn(args.reward_fn_path)

    # ----------------------------------------------------------
    # Load processor
    # TODO: adjust for your VLM processor if using a model other than Qwen2-VL.
    # ----------------------------------------------------------
    logger.info("Loading processor from %s ...", args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # Ensure the processor does not add pad tokens to the right by default
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    # ----------------------------------------------------------
    # Load base model in bfloat16
    # TODO: adjust model class for a VLM other than Qwen2-VL.
    # ----------------------------------------------------------
    logger.info("Loading base model from %s (bfloat16) ...", args.model_name_or_path)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    # ----------------------------------------------------------
    # Apply LoRA
    # TODO: verify target_modules are correct for your Qwen2-VL variant.
    # ----------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ----------------------------------------------------------
    # Optional frozen reference model for KL penalty
    # ----------------------------------------------------------
    ref_model: Optional[torch.nn.Module] = None
    if args.kl_coeff > 0.0:
        logger.info("Loading frozen reference model for KL penalty ...")
        ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # ----------------------------------------------------------
    # Optimizer and LR scheduler
    # ----------------------------------------------------------
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_train_steps,
    )

    # ----------------------------------------------------------
    # Training state
    # ----------------------------------------------------------
    global_step = 0          # counts completed optimizer steps
    micro_step = 0           # counts forward-backward passes
    running_loss = 0.0
    running_reward = 0.0
    t0 = time.time()

    model.train()
    optimizer.zero_grad()

    logger.info(
        "Starting GRPO training — %d steps, batch_size=%d, group_size=%d, "
        "grad_accum=%d, mode=%s, T_init=%.3f",
        args.num_train_steps,
        args.batch_size,
        args.group_size,
        args.gradient_accumulation_steps,
        args.sampling_mode,
        curriculum.T,
    )

    # ----------------------------------------------------------
    # Main loop — one iteration = one gradient-accumulation micro-step
    # ----------------------------------------------------------
    while global_step < args.num_train_steps:

        # ---- 3a. Sample batch indices ----------------------------------------
        batch_indices = sampler.sample(args.batch_size, curriculum.T)
        batch_records = [records[i] for i in batch_indices]

        # ---- 3b-c. Generate G responses and compute rewards -------------------
        all_log_probs: List[torch.Tensor] = []
        all_advantages: List[float] = []
        batch_raw_rewards: List[float] = []  # for curriculum update

        # Collect (prompt_inputs, generated_ids_list) per example for KL computation.
        # Each entry: (prompt_inputs_dict, List[Tensor(1, gen_len)])
        kl_data: List[Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]] = []

        for example in batch_records:
            question: str = example[args.question_column]
            ground_truth: str = str(example[args.answer_column])
            image_b64: Optional[str] = example.get(args.image_column)
            pil_image = decode_image(image_b64)

            # Build prompt inputs once (shared across G rollouts)
            prompt_inputs = build_model_inputs(processor, question, pil_image, device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            group_rewards: List[float] = []
            group_generated_ids: List[torch.Tensor] = []

            # Generate G candidate responses
            model.eval()
            with torch.no_grad():
                pad_token = (
                    processor.tokenizer.pad_token_id
                    if processor.tokenizer.pad_token_id is not None
                    else processor.tokenizer.eos_token_id
                )
                for _ in range(args.group_size):
                    gen_out = model.generate(
                        **prompt_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=pad_token,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                    # Only keep the newly generated tokens (strip the prompt prefix)
                    gen_ids = gen_out[:, prompt_len:]  # (1, gen_len)
                    group_generated_ids.append(gen_ids)

                    # Decode and compute reward
                    decoded = processor.tokenizer.decode(
                        gen_ids[0],
                        skip_special_tokens=True,
                    ).strip()
                    r = compute_reward(
                        decoded,
                        ground_truth,
                        example=example,
                        custom_fn=custom_reward_fn,
                    )
                    group_rewards.append(r)

            batch_raw_rewards.extend(group_rewards)

            # Normalise rewards within the group: r_norm = (r - mean) / (std + 1e-8)
            norm_rewards = normalize_rewards(group_rewards)

            # Stash (prompt_inputs, generated_ids) for optional KL computation
            kl_data.append((prompt_inputs, group_generated_ids))

            # ---- 3d-e. Compute log-probs in training mode (teacher forcing) ----
            model.train()
            for gen_ids, advantage in zip(group_generated_ids, norm_rewards):
                # Full sequence: [prompt tokens | generated tokens]
                full_ids = torch.cat(
                    [prompt_inputs["input_ids"], gen_ids], dim=1
                )  # (1, prompt_len + gen_len)

                full_mask = torch.ones(
                    full_ids.shape,
                    dtype=prompt_inputs["attention_mask"].dtype,
                    device=device,
                )

                # Labels: -100 masks the prompt; generated positions carry their token ids
                labels = full_ids.clone()
                labels[:, :prompt_len] = -100

                lp = compute_log_probs(
                    model,
                    input_ids=full_ids,
                    labels=labels,
                    attention_mask=full_mask,
                )
                all_log_probs.append(lp)
                all_advantages.append(advantage)

        # ---- 3e. GRPO policy gradient loss -----------------------------------
        loss = grpo_loss(all_log_probs, all_advantages)

        # ---- KL penalty (optional, gated by --kl_coeff > 0) ------------------
        if args.kl_coeff > 0.0 and ref_model is not None:
            kl_total = torch.tensor(0.0, device=device)
            num_kl_terms = 0
            for ex_prompt_inputs, ex_gen_ids_list in kl_data:
                for gen_ids in ex_gen_ids_list:
                    kl_total = kl_total + compute_kl_penalty(
                        model,
                        ref_model,
                        ex_prompt_inputs["input_ids"],
                        ex_prompt_inputs["attention_mask"],
                        gen_ids,
                    )
                    num_kl_terms += 1
            if num_kl_terms > 0:
                kl_total = kl_total / num_kl_terms
            loss = loss + args.kl_coeff * kl_total

        # ---- 3f. Scale loss for gradient accumulation -----------------------
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        running_loss += loss.item()
        running_reward += (
            sum(batch_raw_rewards) / len(batch_raw_rewards)
            if batch_raw_rewards else 0.0
        )
        micro_step += 1

        # ---- Optimizer step every gradient_accumulation_steps micro-steps ----
        if micro_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # ---- 3g. Curriculum update (after optimizer step) ----------------
            R_avg = running_reward / args.gradient_accumulation_steps
            curriculum.update(R_avg)

            # ---- 3h. Logging -------------------------------------------------
            if global_step % args.log_steps == 0:
                avg_loss = running_loss / args.gradient_accumulation_steps
                elapsed = time.time() - t0
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d | loss=%.4f | reward=%.4f | T=%.4f | lr=%.2e | elapsed=%.1fs",
                    global_step,
                    avg_loss,
                    R_avg,
                    curriculum.T,
                    current_lr,
                    elapsed,
                )

            # ---- 3i. Checkpoint ---------------------------------------------
            if global_step % args.save_steps == 0:
                ckpt_dir = output_path / f"step_{global_step}"
                model.save_pretrained(str(ckpt_dir))
                # Also save processor for convenience
                processor.save_pretrained(str(ckpt_dir))
                logger.info("Saved checkpoint to %s", ckpt_dir)

            # Reset running accumulators
            running_loss = 0.0
            running_reward = 0.0

        # Free GPU memory after each batch
        torch.cuda.empty_cache()

        if global_step >= args.num_train_steps:
            break

    # ----------------------------------------------------------
    # Final checkpoint
    # ----------------------------------------------------------
    final_dir = output_path / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Training complete. Final checkpoint saved to %s", final_dir)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
