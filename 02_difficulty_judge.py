import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judging prompt (system message content)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a mathematics subject matter expert and competition problem classifier.\n"
    "Your task is to assess the difficulty of the following mathematical problem on a scale from 1 to 10\n"
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
    "- Respond with ONLY a single integer between 1 and 10. No explanation, no punctuation, no extra text."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the difficulty judging pipeline.

    Returns:
        argparse.Namespace: Parsed arguments with all pipeline configuration.
    """
    parser = argparse.ArgumentParser(
        description="Score mathematical problem difficulty using a local vLLM inference engine."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/poc_train_data.jsonl",
        help="Path to the input JSONL file (default: data/poc_train_data.jsonl).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/poc_scored_data.jsonl",
        help="Path for the annotated output JSONL file (default: data/poc_scored_data.jsonl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HuggingFace model identifier for vLLM (default: Qwen/Qwen2-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of items per vLLM inference batch (default: 32).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Truncate dataset to this many samples for quick testing (default: None = all).",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save intermediate checkpoint every N rows (default: 500).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------
def parse_difficulty_score(response: str) -> Optional[int]:
    """Extract the first integer in [1, 10] from a raw VLM response string.

    Handles common malformed patterns such as:
      - "Level 7"          -> 7
      - "7/10"             -> 7
      - "Difficulty: 7"    -> 7
      - "7.5"              -> 7
      - Multi-line output  -> strip and take first numeric token

    Args:
        response: Raw string returned by the VLM.

    Returns:
        An integer in [1, 10] if one can be reliably extracted, else None.
    """
    if not response:
        return None

    # Normalise whitespace and take only the first line to handle multi-line
    # responses where the answer appears before any explanation.
    first_line = response.strip().split("\n")[0].strip()

    # Find all digit sequences in the first line; fall back to full response.
    candidates_str = first_line if first_line else response.strip()
    matches = re.findall(r"\b(\d+)\b", candidates_str)

    if not matches:
        # Last-ditch: scan the full response for any digit sequence.
        matches = re.findall(r"\b(\d+)\b", response)

    for match in matches:
        value = int(match)
        if 1 <= value <= 10:
            return value

    return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_messages(prompt: str, image_b64: Optional[str]) -> List[Dict]:
    """Build an OpenAI-style multi-turn message list for vLLM chat inference.

    The system turn contains the AoPS difficulty rubric.  The user turn
    contains the problem text and, when available, the problem image encoded
    as a data-URI.

    Args:
        prompt: The problem text to be scored.
        image_b64: Base64-encoded PNG image string, or None for text-only rows.

    Returns:
        A list of message dicts compatible with vLLM's LLM.chat() interface.
    """
    system_message: Dict = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    # Build user content as a list of typed content blocks.
    user_content: List[Dict] = [{"type": "text", "text": prompt}]

    if image_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    user_message: Dict = {
        "role": "user",
        "content": user_content,
    }

    return [system_message, user_message]


# ---------------------------------------------------------------------------
# vLLM inference runner
# ---------------------------------------------------------------------------
def run_inference(
    llm: LLM,
    batch: List[List[Dict]],
    sampling_params: SamplingParams,
) -> List[str]:
    """Run a batch of conversations through vLLM's chat interface.

    Args:
        llm: An initialised vLLM LLM instance.
        batch: A list of conversations; each conversation is a list of
            OpenAI-style message dicts (the output of build_messages()).
        sampling_params: vLLM SamplingParams controlling generation behaviour.

    Returns:
        A list of raw response strings, one per input conversation.  On a
        per-item failure the corresponding entry is an empty string so that
        the caller can mark it as a parse error without crashing.
    """
    try:
        outputs = llm.chat(batch, sampling_params=sampling_params)
    except Exception as exc:  # pragma: no cover
        logger.error("Batch inference failed (%s). Falling back to single-item.", exc)
        # Fall back to item-by-item inference so one bad input cannot drop the
        # whole batch.
        outputs_list: List[str] = []
        for conversation in batch:
            try:
                single_out = llm.chat([conversation], sampling_params=sampling_params)
                text = single_out[0].outputs[0].text.strip()
            except Exception as item_exc:
                logger.error("Single-item inference failed: %s", item_exc)
                text = ""
            outputs_list.append(text)
        return outputs_list

    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Load data, run vLLM difficulty scoring, checkpoint, and save results.

    Orchestrates the full pipeline:
      1. Parse CLI args.
      2. Load input JSONL and optionally resume from a checkpoint.
      3. Initialise vLLM with Qwen2-VL.
      4. Iterate in batches, scoring each problem.
      5. Write intermediate checkpoints every N rows.
      6. Write the final output JSONL and log a summary.
    """
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".ckpt")

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    logger.info("Loading input data from %s", input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    records: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), input_path)

    # Log basic statistics.
    null_image_count = sum(1 for r in records if not r.get("image"))
    logger.info(
        "Dataset stats: total=%d, with_image=%d, text_only=%d",
        len(records),
        len(records) - null_image_count,
        null_image_count,
    )

    # Optional truncation for quick testing.
    if args.max_samples is not None:
        records = records[: args.max_samples]
        logger.info("Truncated to %d samples (--max_samples)", len(records))

    # ------------------------------------------------------------------
    # 2. Checkpoint resume
    # ------------------------------------------------------------------
    completed_results: List[Dict] = []
    processed_ids: set = set()

    if checkpoint_path.exists():
        logger.info("Checkpoint found at %s — resuming.", checkpoint_path)
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    completed_results.append(row)
                    processed_ids.add(row["id"])
        logger.info("Resuming from checkpoint: %d rows already processed.", len(completed_results))

    # Filter to only unprocessed records.
    pending_records = [r for r in records if r["id"] not in processed_ids]
    logger.info("%d records remain to be processed.", len(pending_records))

    if not pending_records:
        logger.info("All records already processed. Writing final output.")
        _write_jsonl(completed_results, output_path)
        _log_summary(completed_results)
        return

    # ------------------------------------------------------------------
    # 3. Initialise vLLM
    # ------------------------------------------------------------------
    logger.info("Initialising vLLM with model: %s", args.model_name)
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    logger.info("vLLM initialised successfully.")

    # ------------------------------------------------------------------
    # 4. Batch inference loop
    # ------------------------------------------------------------------
    new_results: List[Dict] = []
    rows_since_ckpt = 0

    total = len(pending_records)
    pbar = tqdm(total=total, desc="Scoring difficulty", unit="row")

    for batch_start in range(0, total, args.batch_size):
        batch_records = pending_records[batch_start : batch_start + args.batch_size]

        # Build conversation lists for vLLM.
        conversations: List[List[Dict]] = []
        for rec in batch_records:
            image_b64: Optional[str] = rec.get("image") or None
            messages = build_messages(rec["prompt"], image_b64)
            conversations.append(messages)

        # Run inference.
        raw_responses = run_inference(llm, conversations, sampling_params)

        # Parse and annotate.
        for rec, raw_response in zip(batch_records, raw_responses):
            score_int = parse_difficulty_score(raw_response)
            parse_error = score_int is None

            if parse_error:
                logger.warning(
                    "Parse error for id=%s | raw_response=%r",
                    rec.get("id"),
                    raw_response,
                )

            difficulty: Optional[float] = (score_int / 10.0) if score_int is not None else None

            annotated = {
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

        # ------------------------------------------------------------------
        # 5. Checkpoint
        # ------------------------------------------------------------------
        if rows_since_ckpt >= args.checkpoint_interval:
            all_so_far = completed_results + new_results
            logger.info(
                "Writing checkpoint at %d total rows -> %s",
                len(all_so_far),
                checkpoint_path,
            )
            _write_jsonl(all_so_far, checkpoint_path)
            rows_since_ckpt = 0

    pbar.close()

    # ------------------------------------------------------------------
    # 6. Final save
    # ------------------------------------------------------------------
    all_results = completed_results + new_results

    # Write any remaining checkpoint rows that haven't been flushed yet.
    logger.info("Writing final checkpoint -> %s", checkpoint_path)
    _write_jsonl(all_results, checkpoint_path)

    logger.info("Writing final output -> %s", output_path)
    _write_jsonl(all_results, output_path)

    _log_summary(all_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(records: List[Dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed.

    Args:
        records: List of dicts to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _log_summary(results: List[Dict]) -> None:
    """Log a final summary of processing counts and parse outcomes.

    Args:
        results: Fully annotated list of result dicts.
    """
    total = len(results)
    parse_errors = sum(1 for r in results if r.get("difficulty_parse_error"))
    successes = total - parse_errors

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("  Total processed : %d", total)
    logger.info("  Successful parses: %d", successes)
    logger.info("  Parse errors     : %d", parse_errors)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
