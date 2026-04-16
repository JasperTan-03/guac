"""Difficulty judging module for the VLM RL pipeline.

Scores math/vision problems on the AoPS 1–10 scale using a local vLLM engine.
Output difficulty values are normalised to [0.0, 1.0] (AoPS score / 10.0).

Compatible with: vllm>=0.4, omegaconf>=2.3, tqdm>=4.0
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams

from guac.data.prep import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AoPS judging prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a mathematics subject matter expert and competition problem classifier.\n"
    "Your task is to assess the difficulty of the following mathematical problem on a continuous scale from 1 to 10\n"
    "using the Art of Problem Solving (AoPS) difficulty rubric:\n"
    "\n"
    "AoPS Difficulty Scale:\n"
    "  1 - Trivial arithmetic or basic definitions (e.g., 2+2, naming a shape)\n"
    "  2 - Elementary school competition level (e.g., simple word problems, basic fractions)\n"
    "  3 - Middle school competition level (e.g., AMC 8 easy, MATHCOUNTS Sprint easy)\n"
    "  4 - AMC 8 hard / AMC 10 easy (e.g., basic algebra, geometry, number theory)\n"
    "  5 - AMC 10 mid / AMC 12 easy (e.g., intermediate algebra, combinatorics intro)\n"
    "  6 - AMC 10 hard / AMC 12 mid (e.g., coordinate geometry, modular arithmetic)\n"
    "  7 - AMC 12 hard / AIME easy (e.g., complex numbers, advanced combinatorics)\n"
    "  8 - AIME mid (e.g., multi-step proofs, advanced number theory, 3D geometry)\n"
    "  9 - AIME hard / USAMO easy (e.g., olympiad-level proofs, elegant constructions)\n"
    "  10 - USAMO/Putnam hard (e.g., research-adjacent, highly non-trivial proofs)\n"
    "\n"
    "Instructions:\n"
    "- Carefully read the problem text and examine any provided image.\n"
    "- Consider the mathematical concepts required, the number of steps, and the creativity needed.\n"
    "- Respond with ONLY a single continuous number between 1 and 10 (e.g., 5.5, 7.2). No explanation, no punctuation, no extra text."
)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------
def parse_difficulty_score(response: str) -> Optional[float]:
    """Extract the first number in [1.0, 10.0] from a raw VLM response string.

    Handles common malformed patterns such as:

    - ``"Level 7"``        -> 7.0
    - ``"7.5/10"``         -> 7.5
    - ``"Difficulty: 7.2"``-> 7.2
    - ``"7.5"``            -> 7.5
    - Multi-line output    -> strip and take first numeric token

    Args:
        response: Raw string returned by the VLM.

    Returns:
        A float in [1.0, 10.0] if one can be reliably extracted, else None.
    """
    if not response:
        return None

    # Normalise whitespace; work on the first line first to handle responses
    # where the model adds an explanation after the answer.
    first_line = response.strip().split("\n")[0].strip()
    search_text = first_line if first_line else response.strip()

    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", search_text)

    if not matches:
        # Last-ditch: scan the entire response.
        matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)

    for match in matches:
        try:
            value = float(match)
            if 1.0 <= value <= 10.0:
                return value
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_messages(prompt: str, image_b64: Optional[str]) -> List[Dict]:
    """Build an OpenAI-style chat message list for vLLM chat inference.

    The system turn contains the full AoPS difficulty rubric. The user turn
    contains the problem image (when present) and problem text as typed content
    blocks. Text-only rows omit the image block entirely.

    Args:
        prompt: The problem text to be scored.
        image_b64: Base64-encoded PNG string, or None for text-only problems.

    Returns:
        A list of message dicts compatible with ``vllm.LLM.chat()``.
    """
    system_message: Dict = {
        "role": "system",
        "content": SYSTEM_PROMPT,
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
        # Plain string content is acceptable when there is no image.
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

    Loads a multimodal model via vLLM and scores each problem on the AoPS
    1–10 scale, normalising to [0.0, 1.0] before writing output.

    Args:
        cfg: Hydra DictConfig whose ``judge`` sub-config must contain:

            - ``model_name`` (str): HuggingFace model identifier.
            - ``gpu_memory_utilization`` (float): Fraction of GPU VRAM to use.
            - ``tensor_parallel_size`` (int): Number of GPUs for tensor parallelism.
            - ``max_model_len`` (int): Maximum token context length.
            - ``batch_size`` (int): Items per inference batch.
            - ``temperature`` (float): Sampling temperature (0.0 for greedy).
            - ``max_tokens`` (int): Maximum tokens to generate per response.
            - ``checkpoint_interval`` (int): Rows between checkpoint writes.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the vLLM engine from ``cfg.judge`` settings.

        Args:
            cfg: Top-level Hydra DictConfig containing a ``judge`` sub-config.
        """
        jcfg = cfg.judge
        logger.info(
            "Initialising DifficultyJudge | model=%s | tp=%d | gpu_mem=%.2f | max_len=%d",
            jcfg.model_name,
            jcfg.tensor_parallel_size,
            jcfg.gpu_memory_utilization,
            jcfg.max_model_len,
        )

        self._cfg = jcfg
        self._llm = LLM(
            model=jcfg.model_name,
            gpu_memory_utilization=jcfg.gpu_memory_utilization,
            tensor_parallel_size=jcfg.tensor_parallel_size,
            max_model_len=jcfg.max_model_len,
        )
        self._sampling_params = SamplingParams(
            temperature=jcfg.temperature,
            max_tokens=jcfg.max_tokens,
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
        inference, normalises scores to [0.0, 1.0], checkpoints every N rows,
        and writes the final annotated JSONL.

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

            raw_responses = self._run_batch(batch_records)

            for rec, raw_response in zip(batch_records, raw_responses):
                score_val = parse_difficulty_score(raw_response)
                parse_error = score_val is None

                if parse_error:
                    logger.warning(
                        "Parse error | id=%s | raw=%r", rec.get("id"), raw_response
                    )

                difficulty: Optional[float] = (
                    score_val / 10.0 if score_val is not None else None
                )

                annotated: Dict = {
                    "id": rec["id"],
                    "image": rec.get("image"),
                    "prompt": rec["prompt"],
                    "answer": rec["answer"],
                    "difficulty": difficulty,
                    "difficulty_raw_response": raw_response,
                    "difficulty_parse_error": parse_error,
                }
                new_results.append(annotated)
                rows_since_ckpt += 1

            pbar.update(len(batch_records))

            # Checkpoint flush
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
    def _run_batch(self, batch: List[Dict]) -> List[str]:
        """Run inference on a batch of records via ``llm.chat()``.

        Builds conversation lists for every record in ``batch`` and submits
        them in a single ``llm.chat()`` call. If that call raises an exception,
        falls back to processing each item individually so one bad input cannot
        drop the whole batch.

        Args:
            batch: List of raw input record dicts, each containing at minimum
                ``prompt`` and optionally ``image`` (base64 PNG or null).

        Returns:
            List of raw response strings, one per record. Items that fail even
            in single-item fallback mode are returned as empty strings.
        """
        conversations: List[List[Dict]] = []
        for rec in batch:
            image_b64: Optional[str] = rec.get("image") or None
            messages = build_messages(rec["prompt"], image_b64)
            conversations.append(messages)

        try:
            outputs = self._llm.chat(
                conversations, sampling_params=self._sampling_params
            )
            return [out.outputs[0].text.strip() for out in outputs]

        except Exception as exc:
            logger.error(
                "Batch inference failed (%s) — falling back to single-item processing.",
                exc,
            )
            results: List[str] = []
            for conversation, rec in zip(conversations, batch):
                try:
                    single_out = self._llm.chat(
                        [conversation], sampling_params=self._sampling_params
                    )
                    text = single_out[0].outputs[0].text.strip()
                except Exception as item_exc:
                    logger.error(
                        "Single-item inference failed | id=%s | error=%s",
                        rec.get("id"),
                        item_exc,
                    )
                    text = ""
                results.append(text)
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
