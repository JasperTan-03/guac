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

    # Strategy 2: explicit answer-prefix patterns.
    #
    # The terminator is ``.\s``, ``.$``, or end-of-string — *not* a bare
    # ``.`` — so decimal numbers like ``0.5`` aren't truncated to ``0``
    # by the lazy ``.+?`` capture group.
    prefix_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.\s|\.$|$)",
        r"(?:therefore|thus|so)[,\s]+(?:the\s+answer\s+is\s+)?([^\s,]+?)(?:\.\s|\.$|,|$)",
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


_LATEX_CLEAN_REPLACEMENTS = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\cdot", "*"),
    (r"\\times", "*"),
    (r"\\pi\b", "pi"),
    (r"\\sqrt", "sqrt"),
    (r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)"),
    (r"\^\s*\{([^{}]+)\}", r"**(\1)"),
    (r"\^", "**"),
    (r"\$+", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\!", ""),
    (r"\\\\", ""),
]


def _normalize_latex_for_sympy(s: str) -> str:
    """Best-effort rewrite of LaTeX-ish math into SymPy-parsable syntax.

    The transformations are intentionally conservative — we only rewrite
    constructs that have an unambiguous SymPy equivalent.  Anything we
    don't recognise is left in place so ``sympify`` can try to parse it.
    """
    out = s.strip()
    # Drop a leading "x = " / "X = " style prefix: ground-truth answers are
    # sometimes written as "x = 3" while the model outputs just "3".
    out = re.sub(r"^\s*[a-zA-Z][a-zA-Z_0-9]*\s*=\s*", "", out)
    # Drop trailing units / text after whitespace-free numerics.
    for pattern, repl in _LATEX_CLEAN_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    # Remove stray braces that survive after the LaTeX rewrites.
    out = out.replace("{", "").replace("}", "")
    return out.strip()


# Hard upper bound on how much text we'll hand to SymPy.  Real answers
# are typically < 50 chars; long model rambles force O(L) parsing work
# that never produces a match and becomes a CPU hotspot early in training
# (when most rollouts are wrong).
_SYMPY_MAX_INPUT_LEN = 200

# Cheap pre-filter: if a string has no digit, math operator, or LaTeX
# math token, SymPy will never succeed.  Skipping the parse when this is
# false is ~1000x cheaper than sympify+simplify on a wrong-but-textual
# rollout.
_MATH_LIKELY_RE = re.compile(
    r"[0-9]|[+\-*/^=(){}]|\\frac|\\sqrt|\\pi|\\cdot|\\times"
)


def _looks_like_math(s: str) -> bool:
    """Return True if ``s`` is worth handing to SymPy.

    False-positive is fine (we'll just do a wasted parse); false-negative
    means we miss a SymPy equivalence, so err on the side of inclusion.
    """
    if not s or len(s) > _SYMPY_MAX_INPUT_LEN:
        return False
    return _MATH_LIKELY_RE.search(s) is not None


def _sympy_equivalent(a: str, b: str) -> bool:
    """Return True iff ``a`` and ``b`` represent the same mathematical value.

    Uses :func:`sympy.sympify` on the LaTeX-normalised inputs and checks
    ``simplify(A - B) == 0``.  Returns False (not raises) on any parse
    error, non-numeric input, or unevaluated difference — the substring
    branch of ``compute_reward`` will still handle those cases.

    A cheap ``_looks_like_math`` pre-filter guards the sympify call so
    that common word-answer rollouts (e.g. ``"the capital is paris"``)
    don't pay the full parse cost during training.

    Examples that return True:
        ``"3"`` vs ``"x = 3"``
        ``"\\frac{1}{2}"`` vs ``"0.5"``
        ``"\\boxed{3}"`` vs ``"3"``   (caller is expected to strip \\boxed)
        ``"2*pi"`` vs ``"pi*2"``
    """
    # Fast path: if either side doesn't look like math, SymPy can't help.
    if not (_looks_like_math(a) and _looks_like_math(b)):
        return False

    try:
        import sympy
        from sympy import sympify
    except ImportError:  # pragma: no cover — sympy is a hard dep
        return False

    try:
        a_expr = sympify(_normalize_latex_for_sympy(a))
        b_expr = sympify(_normalize_latex_for_sympy(b))
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        return False
    except Exception:  # noqa: BLE001 — sympy raises a wide range of errors
        return False

    try:
        diff = sympy.simplify(a_expr - b_expr)
    except Exception:  # noqa: BLE001
        return False

    # Structural equality via simplify → 0.  Fall back to a tight numeric
    # comparison for floats that don't simplify symbolically.
    if diff == 0:
        return True
    try:
        return abs(float(diff)) < 1e-6
    except (TypeError, ValueError):
        return False


def compute_reward(prediction: str, ground_truth: str) -> float:
    """Compute a reward for a predicted answer against the ground truth.

    Uses a cascade of matching strategies so that common model verbosity
    (e.g. "The answer is 12." vs. "12") still yields a positive reward:

    1. Case-insensitive exact match after whitespace normalisation.
    2. Both sides parsed through ``_extract_final_answer``; extracted tokens
       compared.
    3. Symbolic / numeric equivalence via SymPy on the extracted answers
       (catches "x=3" vs "3", "\\frac{1}{2}" vs "0.5", "2*pi" vs "pi*2").
    4. Substring containment: ground truth (≥ 3 chars) appears verbatim
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

    # Strategy 3: SymPy symbolic / numeric equivalence.  Matches AdaRFT's
    # answer-check scheme (Shi et al. 2025) and catches LaTeX↔decimal
    # mismatches like "0.5" vs "\\frac{1}{2}" that the regex extractor
    # can't handle (it would pull only the denominator).  We try both the
    # extracted pred and the raw prediction against the raw ground truth,
    # since the LaTeX normaliser handles the raw GT better than the
    # regex-based extractor would.
    pred_candidates = [c for c in (pred_extracted, prediction) if c]
    gt_candidates = [c for c in (gt_extracted, ground_truth) if c]
    for p_cand in pred_candidates:
        for g_cand in gt_candidates:
            if _sympy_equivalent(p_cand, g_cand):
                return 1.0

    # Strategy 4: ground truth appears as a whole word/number in prediction.
    # We require a word-boundary match to avoid false positives like "120"
    # matching inside "1200" or "12000".  The `re.escape` handles dots and
    # other special chars in numeric answers.
    if len(gt_norm) >= 3:
        if re.search(r"(?<![0-9a-z])" + re.escape(gt_norm) + r"(?![0-9a-z])", pred_norm):
            return 1.0

    return 0.0


# A group whose reward std falls below this threshold is treated as
# uninformative: every rollout got the same reward, so normalised
# advantages would all be zero and the resulting GRPO loss term would
# vanish.  The trainer skips such groups rather than backpropagating
# through a zero signal.
FLAT_GROUP_STD_EPS = 1e-6


def is_informative_group(rewards: List[float]) -> bool:
    """Return True iff the reward group carries a usable GRPO signal.

    A group is uninformative when every rollout earned (approximately)
    the same reward — e.g. ``[0, 0, 0, 0]`` or ``[1, 1, 1, 1]``.  In that
    case ``normalize_rewards`` returns all zeros, so the GRPO loss term
    ``-mean(advantages * log_probs)`` is exactly zero and ``backward()``
    produces no gradient.  Callers should skip these groups so that the
    reported loss reflects only informative examples.

    Args:
        rewards: Raw reward values for one group.

    Returns:
        ``True`` if the group's reward std exceeds ``FLAT_GROUP_STD_EPS``.
    """
    if len(rewards) < 2:
        return False
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return math.sqrt(variance) > FLAT_GROUP_STD_EPS


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
        List of normalised reward values of the same length.  When the
        group is flat (variance below ``FLAT_GROUP_STD_EPS``), returns a
        list of zeros — callers should detect this via
        :func:`is_informative_group` and skip the group.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        raise ValueError("Cannot normalise an empty reward list.")

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    if std <= FLAT_GROUP_STD_EPS:
        return [0.0] * n
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
