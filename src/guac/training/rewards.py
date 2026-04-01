"""Reward computation utilities for GRPO training.

Provides:
  - ``compute_reward`` — binary exact-match reward.
  - ``normalize_rewards`` — within-group reward normalisation.
  - ``compute_log_probs`` — teacher-forcing log-probability computation.
  - ``compute_kl_penalty`` — per-token KL divergence between policy and reference.
  - ``grpo_loss`` — GRPO policy gradient loss.

Compatible with: torch>=2.3, transformers>=4.45
"""

import math
import re
from typing import List, Optional

import torch
import torch.nn.functional as F


def _extract_final_answer(text: str) -> Optional[str]:
    """Extract a final answer token from model output.

    Tries the following strategies in order:
    1. Strip ``<think>...</think>`` reasoning traces.
    2. Extract the last ``\\boxed{...}`` expression.
    3. Match explicit answer-prefix patterns (e.g. "the answer is 12").
    4. Match a trailing ``= <value>`` expression.
    5. Return the last integer or decimal number in the text.

    Args:
        text: Raw decoded output string from the model.

    Returns:
        Normalised (stripped, lowercased) answer string, or None if nothing
        could be extracted.
    """
    if not text:
        return None

    # Strip <think>...</think> reasoning traces
    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        text = think_match.group(1)
    text = text.strip()

    # Strategy 1: \boxed{...} (last occurrence wins)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip().lower()

    # Strategy 2: explicit answer-prefix patterns
    prefix_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"(?:therefore|thus|so)[,\s]+(?:the\s+answer\s+is\s+)?([^\s,.]+)",
    ]
    for pat in prefix_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()

    # Strategy 3: trailing "= value"
    eq_match = re.search(r"=\s*([^\s=,]+)\s*$", text)
    if eq_match:
        return eq_match.group(1).strip().lower()

    # Strategy 4: last number (integer or decimal)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].lower()

    return None


def compute_reward(prediction: str, ground_truth: str) -> float:
    """Compute a reward for a predicted answer against the ground truth.

    Uses a cascade of matching strategies so that common model verbosity
    (e.g. "The answer is 12." vs. "12") still yields a positive reward:

    1. Case-insensitive exact match after whitespace normalisation.
    2. Both sides parsed through ``_extract_final_answer``; extracted tokens
       compared.
    3. Substring containment: ground truth (≥ 3 chars) appears verbatim
       inside the normalised prediction.

    Args:
        prediction: Model's decoded output string.
        ground_truth: Ground-truth answer string.

    Returns:
        1.0 if any matching strategy succeeds, 0.0 otherwise.
    """
    pred_norm = " ".join(prediction.strip().lower().split())
    gt_norm = " ".join(ground_truth.strip().lower().split())

    # Strategy 1: exact match
    if pred_norm == gt_norm:
        return 1.0

    # Strategy 2: extracted-answer match
    pred_extracted = _extract_final_answer(prediction)
    gt_extracted = _extract_final_answer(ground_truth)
    if pred_extracted is not None and gt_extracted is not None:
        if pred_extracted == gt_extracted:
            return 1.0

    # Strategy 3: ground truth appears verbatim in prediction
    # (guards against false positives from very short strings like "a")
    if len(gt_norm) >= 3 and gt_norm in pred_norm:
        return 1.0

    return 0.0


def normalize_rewards(rewards: List[float]) -> List[float]:
    """Normalise rewards within a group to zero mean and unit variance.

    The standard GRPO normalisation formula is::

        r_i_normalised = (r_i - mean(r)) / (std(r) + 1e-8)

    This keeps the advantage signal centred and prevents large reward
    magnitudes from destabilising training.

    Args:
        rewards: Raw reward values for one group (all G responses to a
                 single prompt).  Must be non-empty.

    Returns:
        List of normalised reward values of the same length.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        raise ValueError("Cannot normalise an empty reward list.")

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance) + 1e-8
    return [(r - mean) / std for r in rewards]


def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of log-probabilities for generated tokens via teacher forcing.

    The model is run with the full concatenated sequence (prompt + generated
    tokens).  The labels tensor marks prompt positions with ``-100`` so that
    only the generated token positions contribute to the returned sum.

    Logits are upcast to float32 before the log-softmax to avoid bfloat16
    precision loss, as recommended in the VLM training reference notes.

    Args:
        model: The language model (policy, with or without LoRA adapters).
               Expected to return an object with a ``logits`` attribute of
               shape ``(B, L, V)``.
        input_ids: Full sequence token IDs (prompt + generated), shape ``(B, L)``.
        attention_mask: Attention mask for the full sequence, shape ``(B, L)``.
        labels: Token IDs with prompt positions set to ``-100``, shape ``(B, L)``.
                Generated positions carry the actual token IDs.

    Returns:
        Per-sequence sum of log-probabilities over generated (non -100) token
        positions, shape ``(B,)``.

    # TODO: adjust for your VLM processor if it returns additional inputs
    #       (e.g. pixel_values, image_grid_thw) that must be forwarded here.
    """
    # Use autocast for memory efficiency; logits are upcast to float32 below.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We compute the loss manually.
        )

    # Upcasted for numerical stability during log_softmax.
    logits = outputs.logits.float()  # (B, L, V)

    # Causal LM convention: position t predicts token t+1.
    shift_logits = logits[:, :-1, :]    # (B, L-1, V)
    shift_labels = labels[:, 1:]        # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)

    # Gather the log-prob of the actual next token at each position.
    # clamp(min=0) prevents gather errors at -100 positions (masked out below).
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)  # (B, L-1)

    # Mask out prompt and padding positions (labels == -100).
    valid_mask = (shift_labels != -100).float()  # (B, L-1)
    # Sum over the generated token positions for each sequence.
    return (token_log_probs * valid_mask).sum(dim=-1)  # (B,)


def compute_kl_penalty(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean per-token KL divergence KL(policy || reference).

    Uses the identity::

        KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))

    which is equivalent to ``F.kl_div(log_Q, P, reduction='batchmean')``.

    Args:
        policy_log_probs: Log-probabilities from the policy model, shape
                          ``(B, L, V)``.  Should be the output of
                          ``F.log_softmax`` in float32.
        ref_log_probs: Log-probabilities from the frozen reference model,
                       shape ``(B, L, V)``.  Same requirements.

    Returns:
        Scalar mean KL divergence averaged over all batch and token positions.

    Raises:
        ValueError: If the two tensors have different shapes.
    """
    if policy_log_probs.shape != ref_log_probs.shape:
        raise ValueError(
            f"Shape mismatch: policy_log_probs {tuple(policy_log_probs.shape)} "
            f"!= ref_log_probs {tuple(ref_log_probs.shape)}."
        )

    # Convert policy log-probs to probabilities for the KL formula.
    policy_probs = torch.exp(policy_log_probs)  # (B, L, V)

    # F.kl_div expects log-space input Q and probability-space target P:
    #   KL(P || Q) = sum P * (log P - log Q) = F.kl_div(log_Q, P, ...)
    kl = F.kl_div(
        input=ref_log_probs,           # log Q
        target=policy_probs,           # P
        reduction="batchmean",
        log_target=False,
    )
    return kl


def grpo_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute the GRPO policy gradient loss.

    The loss is::

        L = -mean(advantages * log_probs)

    A negative sign is applied because optimisers minimise the loss, while
    the policy gradient objective is maximised.

    Args:
        log_probs: Per-sequence sum of log-probabilities, shape ``(N,)``
                   where ``N = batch_size * group_size``.  Must require grad.
        advantages: Normalised advantage values, shape ``(N,)``.

    Returns:
        Scalar loss tensor (with gradient through ``log_probs``).

    Raises:
        ValueError: If ``log_probs`` and ``advantages`` have different shapes.
        ValueError: If ``log_probs`` is empty.
    """
    if log_probs.shape != advantages.shape:
        raise ValueError(
            f"Shape mismatch: log_probs {tuple(log_probs.shape)} "
            f"!= advantages {tuple(advantages.shape)}."
        )
    if log_probs.numel() == 0:
        raise ValueError("grpo_loss received empty tensors.")

    return -(advantages * log_probs).mean()
