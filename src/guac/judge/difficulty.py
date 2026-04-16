"""Difficulty judging module for the VLM RL pipeline.

Scores problems on a configurable integer rubric (default: AoPS 1–10) using a
local vLLM engine. Instead of only keeping the argmax integer, we extract
top-k logprobs at the first generated token position and compute a *continuous*
expected score:

    E[score] = sum_{i=1..min(score_max, 9)}  i * softmax(logp_i)

``difficulty`` is then ``E[score] / score_max``, a float in approximately
``(1/score_max, min(score_max, 9)/score_max)`` — dense and continuous rather
than bucketed to ``score_max`` discrete values. When ``score_max == 10``, the
top end caps at ``0.9`` because the expectation excludes the ``"10"`` token
(two-token in Qwen2; its first token is ``"1"`` and would collide with the
``i=1`` bucket). The argmax integer is still stored as ``difficulty_integer``
for diagnostics.

The rubric text, score range, top-k, and split list are all read from
``cfg.judge`` so the same code can score any new dataset by overriding
config values — no judge-code edits required.

Compatible with: vllm>=0.4, omegaconf>=2.3, tqdm>=4.0
"""

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams

from guac.data.prep import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output parser (argmax from raw text — kept for diagnostics + text fallback)
# ---------------------------------------------------------------------------
def parse_difficulty_score(response: str, score_max: int = 10) -> Optional[int]:
    """Extract the first integer in ``[1, score_max]`` from a raw VLM response.

    Handles common malformed patterns such as:

    - ``"Level 7"``        -> 7
    - ``"7/10"``           -> 7
    - ``"Difficulty: 7"``  -> 7
    - ``"7.5"``            -> 7
    - Multi-line output    -> strip and take first numeric token

    The upper bound is configurable so the parser stays consistent with the
    active rubric. For a ``score_max=5`` rubric, a model output of ``"7"`` is
    out-of-range and will be rejected (returns ``None``) rather than silently
    producing a normalized ``difficulty > 1.0`` downstream.

    Args:
        response: Raw string returned by the VLM.
        score_max: Rubric upper bound (inclusive). Defaults to 10.

    Returns:
        An integer in ``[1, score_max]`` if one can be reliably extracted,
        else ``None``.
    """
    if not response:
        return None

    first_line = response.strip().split("\n")[0].strip()
    search_text = first_line if first_line else response.strip()

    matches = re.findall(r"\b(\d+)\b", search_text)

    if not matches:
        matches = re.findall(r"\b(\d+)\b", response)

    for match in matches:
        value = int(match)
        if 1 <= value <= score_max:
            return value

    return None


# ---------------------------------------------------------------------------
# Continuous expectation from top-k logprobs
# ---------------------------------------------------------------------------
def compute_continuous_difficulty(
    logprobs_at_pos_0: Optional[Dict[int, Any]],
    tokenizer: Any,
    score_max: int,
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """Compute an expected difficulty score from top-k logprobs.

    Reads the top-k logprob dict at generation position 0, decodes each
    token id to its string form, and matches against digit strings
    ``"1"`` through ``str(min(score_max, 9))``. Log-probs over the matched
    subset are softmax-normalised to a proper probability distribution and
    the expectation ``sum(i * p_i)`` is returned.

    ``"10"`` is deliberately excluded when ``score_max == 10``: in the
    Qwen2 tokenizer ``"10"`` is two tokens whose first token is ``"1"``,
    which would collide with the ``i=1`` bucket. In our data the 1.0
    difficulty bucket is effectively empty, so this is a benign
    approximation.

    When multiple tokens decode to the same digit (e.g. ``"2"`` and
    ``" 2"``), we keep the one with the highest logprob.

    Args:
        logprobs_at_pos_0: Mapping ``token_id -> Logprob`` for the first
            generated token, as returned by vLLM when ``SamplingParams``
            has ``logprobs=k``. May be ``None`` if logprobs were not
            requested or inference failed.
        tokenizer: The vLLM model's tokenizer (used to decode token ids).
        score_max: Upper bound of the rubric (1..score_max). Digit tokens
            up to ``min(score_max, 9)`` are read.

    Returns:
        A ``(expectation, probs)`` tuple. ``expectation`` is a float in
        ``[1.0, min(score_max, 9)]``. ``probs`` is a dict mapping
        str-digit (``"1"``, ``"2"``, ...) to its softmax-normalised
        probability — str keys so the dict round-trips cleanly through
        JSON. Returns ``(None, None)`` when ``logprobs_at_pos_0`` is
        missing/empty or no digit tokens were found.
    """
    if not logprobs_at_pos_0:
        return None, None

    max_digit = min(int(score_max), 9)
    digit_targets = {str(i): i for i in range(1, max_digit + 1)}

    digit_logprobs: Dict[int, float] = {}
    for token_id, lp_obj in logprobs_at_pos_0.items():
        try:
            decoded = tokenizer.decode([int(token_id)])
        except Exception:
            continue
        key = decoded.strip()
        if key not in digit_targets:
            continue
        digit = digit_targets[key]
        lp = float(lp_obj.logprob)
        if digit not in digit_logprobs or lp > digit_logprobs[digit]:
            digit_logprobs[digit] = lp

    if not digit_logprobs:
        return None, None

    max_lp = max(digit_logprobs.values())
    exps = {d: math.exp(lp - max_lp) for d, lp in digit_logprobs.items()}
    z = sum(exps.values())
    probs = {str(d): e / z for d, e in exps.items()}
    expectation = sum(int(d) * p for d, p in probs.items())

    return expectation, probs


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_messages(
    system_prompt: str, prompt: str, image_b64: Optional[str]
) -> List[Dict]:
    """Build an OpenAI-style chat message list for vLLM chat inference.

    The system turn contains the rubric (passed in from config). The user
    turn contains the problem image (when present) and problem text as
    typed content blocks. Text-only rows omit the image block entirely.

    Args:
        system_prompt: The rubric / instruction string to use as the
            system message. Typically sourced from ``cfg.judge.system_prompt``.
        prompt: The problem text to be scored.
        image_b64: Base64-encoded PNG string, or None for text-only problems.

    Returns:
        A list of message dicts compatible with ``vllm.LLM.chat()``.
    """
    system_message: Dict = {
        "role": "system",
        "content": system_prompt,
    }

    if image_b64:
        user_content: List[Dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = [{"type": "text", "text": prompt}]

    user_message: Dict = {
        "role": "user",
        "content": user_content,
    }

    return [system_message, user_message]


# ---------------------------------------------------------------------------
# DifficultyJudge
# ---------------------------------------------------------------------------
class DifficultyJudge:
    """vLLM-based difficulty scorer for math and vision problems.

    Loads a multimodal model via vLLM and scores each problem on a
    configurable integer rubric (``cfg.judge.score_max``). Uses top-k
    logprobs at the first generated token to compute a continuous
    expected score.

    Args:
        cfg: Hydra DictConfig whose ``judge`` sub-config must contain:

            - ``model_name`` (str): HuggingFace model identifier.
            - ``gpu_memory_utilization`` (float): Fraction of GPU VRAM to use.
            - ``tensor_parallel_size`` (int): Number of GPUs for tensor parallelism.
            - ``max_model_len`` (int): Maximum token context length.
            - ``batch_size`` (int): Items per inference batch.
            - ``temperature`` (float): Sampling temperature (0.0 for greedy).
            - ``max_tokens`` (int): Maximum tokens to generate per response.
            - ``logprobs_k`` (int): Top-k logprobs to extract per position.
            - ``checkpoint_interval`` (int): Rows between checkpoint writes.
            - ``score_max`` (int): Rubric upper bound for normalization.
            - ``system_prompt`` (str): Rubric text used as the system turn.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the vLLM engine from ``cfg.judge`` settings.

        Args:
            cfg: Top-level Hydra DictConfig containing a ``judge`` sub-config.
        """
        jcfg = cfg.judge
        logger.info(
            "Initialising DifficultyJudge | model=%s | tp=%d | gpu_mem=%.2f | max_len=%d | score_max=%d | logprobs_k=%d",
            jcfg.model_name,
            jcfg.tensor_parallel_size,
            jcfg.gpu_memory_utilization,
            jcfg.max_model_len,
            jcfg.score_max,
            jcfg.logprobs_k,
        )

        self._cfg = jcfg
        self._system_prompt: str = str(jcfg.system_prompt)
        self._score_max: int = int(jcfg.score_max)

        self._llm = LLM(
            model=jcfg.model_name,
            gpu_memory_utilization=jcfg.gpu_memory_utilization,
            tensor_parallel_size=jcfg.tensor_parallel_size,
            max_model_len=jcfg.max_model_len,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=jcfg.temperature,
            max_tokens=jcfg.max_tokens,
            logprobs=int(jcfg.logprobs_k),
        )
        logger.info("vLLM engine ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def score_split(
        self,
        input_path: str,
        output_path: str,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """Score all records in a JSONL file and save annotated output.

        Loads input records, optionally resumes from a checkpoint, runs batch
        inference, computes continuous difficulty via logit expectation,
        checkpoints every N rows, and writes the final annotated JSONL.

        Args:
            input_path: Path to the input JSONL file. Each line must be a JSON
                object with at minimum ``id``, ``prompt``, ``answer``, and
                optionally ``image`` (base64 PNG string or null).
            output_path: Destination path for the annotated JSONL file.
            checkpoint_path: Optional path for resume-on-crash checkpointing.
                Defaults to ``{output_path}.ckpt`` when not supplied.

        Returns:
            Summary dict with keys ``total`` (int), ``scored`` (int), and
            ``parse_errors`` (int).
        """
        in_path = Path(input_path)
        out_path = Path(output_path)
        ckpt_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else out_path.with_suffix(out_path.suffix + ".ckpt")
        )

        # ------------------------------------------------------------------
        # 1. Load input records
        # ------------------------------------------------------------------
        logger.info("Loading records from %s", in_path)
        records: List[Dict] = load_jsonl(str(in_path))
        total_input = len(records)

        null_image_count = sum(1 for r in records if not r.get("image"))
        logger.info(
            "Dataset stats | total=%d | with_image=%d | text_only=%d",
            total_input,
            total_input - null_image_count,
            null_image_count,
        )

        # ------------------------------------------------------------------
        # 2. Checkpoint resume
        # ------------------------------------------------------------------
        completed_results: List[Dict] = []
        processed_ids: set = set()

        if ckpt_path.exists():
            logger.info("Checkpoint found at %s — resuming.", ckpt_path)
            completed_results = load_jsonl(str(ckpt_path))
            processed_ids = {r["id"] for r in completed_results}
            logger.info(
                "%d rows already processed (from checkpoint).", len(completed_results)
            )

        pending: List[Dict] = [r for r in records if r["id"] not in processed_ids]
        logger.info("%d records pending inference.", len(pending))

        if not pending:
            logger.info("All records already processed — writing final output.")
            save_jsonl(completed_results, str(out_path))
            return self._build_summary(completed_results)

        # ------------------------------------------------------------------
        # 3. Batch inference loop
        # ------------------------------------------------------------------
        new_results: List[Dict] = []
        rows_since_ckpt = 0
        batch_size = self._cfg.batch_size
        checkpoint_interval = self._cfg.checkpoint_interval

        pbar = tqdm(total=len(pending), desc="Scoring difficulty", unit="row")

        for batch_start in range(0, len(pending), batch_size):
            batch_records = pending[batch_start : batch_start + batch_size]

            raw_results = self._run_batch(batch_records)

            for rec, (raw_response, lp0) in zip(batch_records, raw_results):
                expectation, probs = compute_continuous_difficulty(
                    lp0, self._tokenizer, self._score_max
                )
                score_int = parse_difficulty_score(raw_response, self._score_max)

                if expectation is not None:
                    difficulty: Optional[float] = expectation / self._score_max
                elif score_int is not None:
                    difficulty = score_int / self._score_max
                else:
                    difficulty = None

                parse_error = difficulty is None

                if parse_error:
                    logger.warning(
                        "Parse error | id=%s | raw=%r", rec.get("id"), raw_response
                    )

                annotated: Dict = {
                    "id": rec["id"],
                    "image": rec.get("image"),
                    "prompt": rec["prompt"],
                    "answer": rec["answer"],
                    "difficulty": difficulty,
                    "difficulty_integer": score_int,
                    "difficulty_probs": probs,
                    "difficulty_raw_response": raw_response,
                    "difficulty_parse_error": parse_error,
                }
                new_results.append(annotated)
                rows_since_ckpt += 1

            pbar.update(len(batch_records))

            if rows_since_ckpt >= checkpoint_interval:
                all_so_far = completed_results + new_results
                logger.info(
                    "Checkpointing %d rows -> %s", len(all_so_far), ckpt_path
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                save_jsonl(all_so_far, str(ckpt_path))
                rows_since_ckpt = 0

        pbar.close()

        # ------------------------------------------------------------------
        # 4. Final save
        # ------------------------------------------------------------------
        all_results = completed_results + new_results

        logger.info("Writing final checkpoint -> %s", ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(all_results, str(ckpt_path))

        logger.info("Writing final output -> %s", out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(all_results, str(out_path))

        summary = self._build_summary(all_results)
        logger.info(
            "Done | total=%d | scored=%d | parse_errors=%d",
            summary["total"],
            summary["scored"],
            summary["parse_errors"],
        )
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _run_batch(
        self, batch: List[Dict]
    ) -> List[Tuple[str, Optional[Dict[int, Any]]]]:
        """Run inference on a batch of records via ``llm.chat()``.

        Builds conversation lists for every record in ``batch`` and submits
        them in a single ``llm.chat()`` call. If that call raises an exception,
        falls back to processing each item individually so one bad input cannot
        drop the whole batch.

        Args:
            batch: List of raw input record dicts, each containing at minimum
                ``prompt`` and optionally ``image`` (base64 PNG or null).

        Returns:
            A list of ``(raw_text, logprobs_at_pos_0)`` tuples, one per
            record. ``logprobs_at_pos_0`` is a dict mapping token_id to a
            vLLM ``Logprob`` object, or ``None`` if unavailable.
            Items that fail even in single-item fallback mode are
            returned as ``("", None)``.
        """
        conversations: List[List[Dict]] = []
        for rec in batch:
            image_b64: Optional[str] = rec.get("image") or None
            messages = build_messages(
                self._system_prompt, rec["prompt"], image_b64
            )
            conversations.append(messages)

        def _extract(output_obj: Any) -> Tuple[str, Optional[Dict[int, Any]]]:
            gen = output_obj.outputs[0]
            text = gen.text.strip()
            lp_list = getattr(gen, "logprobs", None)
            lp0 = lp_list[0] if lp_list else None
            return text, lp0

        try:
            outputs = self._llm.chat(
                conversations, sampling_params=self._sampling_params
            )
            return [_extract(out) for out in outputs]

        except Exception as exc:
            logger.error(
                "Batch inference failed (%s) — falling back to single-item processing.",
                exc,
            )
            results: List[Tuple[str, Optional[Dict[int, Any]]]] = []
            for conversation, rec in zip(conversations, batch):
                try:
                    single_out = self._llm.chat(
                        [conversation], sampling_params=self._sampling_params
                    )
                    results.append(_extract(single_out[0]))
                except Exception as item_exc:
                    logger.error(
                        "Single-item inference failed | id=%s | error=%s",
                        rec.get("id"),
                        item_exc,
                    )
                    results.append(("", None))
            return results

    @staticmethod
    def _build_summary(results: List[Dict]) -> Dict:
        """Compute processing summary counts from a list of annotated records.

        Args:
            results: Fully annotated list of output record dicts.

        Returns:
            Dict with keys:

            - ``total``: Total number of records.
            - ``scored``: Records where parsing succeeded (difficulty is not None).
            - ``parse_errors``: Records where parsing failed.
        """
        total = len(results)
        parse_errors = sum(1 for r in results if r.get("difficulty_parse_error"))
        scored = total - parse_errors
        return {"total": total, "scored": scored, "parse_errors": parse_errors}
