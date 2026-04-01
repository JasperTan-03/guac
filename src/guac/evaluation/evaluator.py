# Dependencies:
#   pip install transformers peft accelerate torch datasets tqdm pillow omegaconf mlflow

"""VLM evaluation module for LoRA checkpoints.

Evaluates a LoRA-adapted Qwen2-VL model on MathVista and MMMU benchmarks.
Designed to be called from scripts/evaluate.py with a Hydra DictConfig.
"""

import ast
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from peft import PeftModel
except ImportError as exc:
    raise ImportError("peft is required: pip install peft") from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_processor(
    checkpoint_path: str,
    base_model_id: str,
) -> Tuple[Any, Any]:
    """Load base model + LoRA adapter + processor.

    Loads the base vision-language model via AutoModelForVision2Seq, then
    attaches the LoRA adapter with PeftModel.from_pretrained. The processor
    is always loaded from the base model ID (adapters do not ship processors).

    Args:
        checkpoint_path: Path to the LoRA adapter directory.
        base_model_id: HuggingFace model ID or local path for the base model.

    Returns:
        A (model, processor) tuple. The model is in eval mode on bfloat16.
    """
    logger.info("Loading base model: %s", base_model_id)
    base = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Attaching LoRA adapter: %s", checkpoint_path)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()

    logger.info("Loading processor from base model: %s", base_model_id)
    processor = AutoProcessor.from_pretrained(base_model_id)

    return model, processor


# ---------------------------------------------------------------------------
# Image coercion
# ---------------------------------------------------------------------------


def _extract_pil_image(image_field: Any) -> Optional[Image.Image]:
    """Coerce an HF image field (PIL.Image, dict with bytes/path) to a PIL Image.

    HuggingFace datasets can deliver image columns as PIL Images, dicts with
    a ``bytes`` or ``path`` key, or None. This function normalises all three
    forms.

    Args:
        image_field: Raw value of an image column from a HuggingFace dataset row.

    Returns:
        An RGB PIL Image, or None if the field is absent or cannot be decoded.
    """
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        raw_bytes = image_field.get("bytes")
        if raw_bytes:
            try:
                return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception as exc:
                logger.debug("Failed to decode image from bytes: %s", exc)
        path = image_field.get("path")
        if path:
            try:
                return Image.open(path).convert("RGB")
            except Exception as exc:
                logger.debug("Failed to open image from path %s: %s", path, exc)
    return None


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def parse_mc_answer(text: str) -> Optional[str]:
    """Extract a multiple-choice letter (A/B/C/D) from model output.

    Processing order:
    1. Strip any ``<think>...</think>`` reasoning trace.
    2. Look for explicit answer-prefix patterns (e.g. "the answer is A", "Answer: B").
    3. Look for bold-formatted letter (e.g. ``**A**``).
    4. Fall back to the first standalone A/B/C/D word boundary match.

    Args:
        text: Raw decoded output string from the model.

    Returns:
        Uppercase letter ``'A'``, ``'B'``, ``'C'``, or ``'D'``, or None if not found.
    """
    if not text:
        return None

    # Strip <think>...</think> reasoning traces
    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        text = think_match.group(1)

    text = text.strip()

    # Priority 1: explicit answer prefixes
    prefix_patterns = [
        r"(?:the\s+)?answer\s+is[:\s]+([ABCD])\b",
        r"(?:answer|option)[:\s]+([ABCD])\b",
        r"\b([ABCD])\s*\.",
    ]
    for pat in prefix_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 2: bold-formatted letter (**A**)
    bold_match = re.search(r"\*\*([ABCD])\*\*", text, re.IGNORECASE)
    if bold_match:
        return bold_match.group(1).upper()

    # Priority 3: first standalone letter
    standalone = re.search(r"\b([ABCD])\b", text)
    if standalone:
        return standalone.group(1).upper()

    return None


def parse_numeric_answer(text: str) -> Optional[str]:
    """Extract a final numeric answer from model output.

    First attempts to find a ``\\boxed{...}`` expression (taking the last one
    if multiple are present). Falls back to the last integer or decimal in the
    text.

    Args:
        text: Raw decoded output string from the model.

    Returns:
        String form of the extracted number, or None if nothing parseable is found.
    """
    if not text:
        return None

    # Strip <think>...</think> reasoning traces
    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        text = think_match.group(1)

    text = text.strip()

    # Try \boxed{...} first (last occurrence wins)
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Fall back to last integer or decimal number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_mathvista_prompt(question: str, choices: Optional[List]) -> str:
    """Append choice instructions or a numeric-answer instruction to a question.

    Args:
        question: The raw question text from the MathVista dataset.
        choices: List of answer choice strings for MC questions, or None for
            free-form numeric questions.

    Returns:
        A formatted prompt string ready to be included in the chat template.
    """
    if choices:
        choices_str = ", ".join(str(c) for c in choices)
        return (
            f"{question}\n"
            f"Choices: {choices_str}\n"
            "Answer with the correct option letter only."
        )
    return f"{question}\nProvide only the final numerical answer."


def format_mmmu_prompt(question: str, options: List) -> str:
    """Append option labels and a single-letter instruction to a question.

    Args:
        question: The raw question text from the MMMU dataset.
        options: List of option strings (typically A through D).

    Returns:
        A formatted prompt string ready to be included in the chat template.
    """
    options_str = ", ".join(str(o) for o in options)
    return (
        f"{question}\n"
        f"Options: {options_str}\n"
        "Answer with a single letter (A, B, C, or D) only."
    )


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def run_inference_batch(
    model: Any,
    processor: Any,
    batch_items: List[Dict],
) -> List[str]:
    """Run batched inference for a list of prepared chat-format samples.

    Each item in ``batch_items`` must contain a ``'messages'`` key holding a
    list of chat messages formatted for the Qwen2-VL chat template. Image
    content parts use ``{"type": "image", "image": <PIL.Image>}``.

    The function:
    1. Applies the chat template to each item to produce a text string.
    2. Collects PIL images from the content parts (flat list aligned to texts).
    3. Calls the processor and moves inputs to the model's device.
    4. Generates tokens with greedy decoding.
    5. Slices off prompt tokens and batch-decodes the generated portion.

    Args:
        model: The loaded PeftModel-wrapped vision-language model.
        processor: The AutoProcessor for the base model.
        batch_items: List of dicts, each with a ``'messages'`` key.

    Returns:
        List of decoded output strings, one per input item. Returns empty
        strings for items that caused a processor error.
    """
    texts: List[str] = []
    images_per_sample: List[Optional[List[Image.Image]]] = []

    for item in batch_items:
        text = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

        pil_images: List[Image.Image] = []
        for msg in item["messages"]:
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    img = part.get("image")
                    if img is not None:
                        pil_images.append(img)
        images_per_sample.append(pil_images if pil_images else None)

    # Build flat image list for the processor.
    # Samples without images must not contribute entries here — the processor
    # matches images to <image> tokens, so the counts must stay aligned.
    has_any_image = any(imgs is not None for imgs in images_per_sample)
    all_images: List[Image.Image] = []
    if has_any_image:
        for imgs in images_per_sample:
            if imgs:
                all_images.extend(imgs)

    try:
        inputs = processor(
            text=texts,
            images=all_images if has_any_image else None,
            return_tensors="pt",
            padding=True,
        )
    except Exception as exc:
        logger.warning("Processor error for batch: %s — returning empty outputs", exc)
        return [""] * len(batch_items)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

    # Strip prompt tokens before decoding
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------


def evaluate_mathvista(
    model: Any,
    processor: Any,
    cfg: DictConfig,
) -> Dict:
    """Evaluate the model on MathVista (testmini split).

    Loads the dataset from ``cfg.evaluation.benchmarks.mathvista.hf_id`` at
    the split specified by ``cfg.evaluation.benchmarks.mathvista.split``.

    Image field priority: ``decoded_image`` (PIL) -> ``image`` (str path).
    Questions with a non-empty ``choices`` field are treated as MC; all others
    are scored as free-form numeric.

    Args:
        model: Loaded vision-language model (PeftModel).
        processor: Corresponding AutoProcessor.
        cfg: Hydra DictConfig rooted at the project config.

    Returns:
        Dict with keys ``accuracy``, ``correct``, ``total``, ``skipped``.
    """
    mv_cfg = cfg.evaluation.benchmarks.mathvista
    hf_id: str = mv_cfg.hf_id
    split: str = mv_cfg.split
    batch_size: int = cfg.evaluation.batch_size

    logger.info("Loading MathVista dataset (%s, split=%s)...", hf_id, split)
    try:
        dataset = load_dataset(hf_id, split=split, trust_remote_code=True)
    except Exception as exc:
        logger.error("Failed to load MathVista: %s", exc, exc_info=True)
        return {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    samples = list(dataset)
    total = len(samples)
    correct = 0
    skipped = 0

    for batch_start in tqdm(
        range(0, total, batch_size), desc="MathVista", unit="batch"
    ):
        batch_raw = samples[batch_start : batch_start + batch_size]
        batch_items: List[Dict] = []

        for idx, sample in enumerate(batch_raw):
            # Prefer decoded_image (PIL), fall back to image (may be a path)
            image = _extract_pil_image(
                sample.get("decoded_image") or sample.get("image")
            )
            if image is None:
                logger.warning(
                    "MathVista sample %d: missing image, skipping.",
                    batch_start + idx,
                )
                skipped += 1
                continue

            choices = sample.get("choices") or sample.get("options")
            question_text = format_mathvista_prompt(
                sample.get("question", ""), choices
            )
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]
            messages = [{"role": "user", "content": content}]
            batch_items.append(
                {"messages": messages, "raw": sample, "choices": choices}
            )

        if not batch_items:
            continue

        try:
            outputs = run_inference_batch(model, processor, batch_items)
        except Exception as exc:
            logger.warning(
                "Inference error on MathVista batch at %d: %s", batch_start, exc
            )
            skipped += len(batch_items)
            continue

        for item, output in zip(batch_items, outputs):
            raw = item["raw"]
            choices = item["choices"]
            ground_truth = str(raw.get("answer", "")).strip()

            predicted = (
                parse_mc_answer(output) if choices else parse_numeric_answer(output)
            )
            if predicted is None:
                skipped += 1
                continue

            if predicted.strip().lower() == ground_truth.strip().lower():
                correct += 1

    accuracy = correct / (total - skipped) if (total - skipped) > 0 else 0.0
    logger.info(
        "MathVista -- accuracy: %.4f (%d/%d), skipped: %d",
        accuracy,
        correct,
        total - skipped,
        skipped,
    )
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}


def evaluate_mmmu(
    model: Any,
    processor: Any,
    cfg: DictConfig,
) -> Dict:
    """Evaluate the model on MMMU (Math config, validation split).

    Loads the dataset from ``cfg.evaluation.benchmarks.mmmu.hf_id`` using the
    config and split declared in the same sub-key. If loading with an explicit
    config fails, retries without specifying a config name.

    Image field: ``image_1`` (PIL or dict). Samples without an image are still
    evaluated as text-only — they are not skipped.

    Options parsing: the ``options`` field may arrive as a Python-literal string
    (e.g. ``"['opt1', 'opt2']"``); this is coerced with ``ast.literal_eval``.

    Args:
        model: Loaded vision-language model (PeftModel).
        processor: Corresponding AutoProcessor.
        cfg: Hydra DictConfig rooted at the project config.

    Returns:
        Dict with keys ``accuracy``, ``correct``, ``total``, ``skipped``.
    """
    mmmu_cfg = cfg.evaluation.benchmarks.mmmu
    hf_id: str = mmmu_cfg.hf_id
    config_name: str = mmmu_cfg.config
    split: str = mmmu_cfg.split
    batch_size: int = cfg.evaluation.batch_size

    logger.info(
        "Loading MMMU dataset (%s, config=%s, split=%s)...",
        hf_id,
        config_name,
        split,
    )
    try:
        dataset = load_dataset(hf_id, config_name, split=split)
    except Exception as exc:
        logger.warning(
            "Failed to load MMMU with config '%s': %s. Retrying without config.",
            config_name,
            exc,
        )
        try:
            dataset = load_dataset(hf_id, split=split)
        except Exception as exc2:
            logger.error("Failed to load MMMU: %s", exc2, exc_info=True)
            return {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    samples = list(dataset)
    total = len(samples)
    correct = 0
    skipped = 0

    for batch_start in tqdm(
        range(0, total, batch_size), desc="MMMU", unit="batch"
    ):
        batch_raw = samples[batch_start : batch_start + batch_size]
        batch_items: List[Dict] = []

        for idx, sample in enumerate(batch_raw):
            # image_1 is the primary image; may be None for text-only samples
            image = _extract_pil_image(sample.get("image_1"))
            if image is None:
                logger.debug(
                    "MMMU sample %d has no image_1; proceeding text-only.",
                    batch_start + idx,
                )

            options = sample.get("options") or []
            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    try:
                        options = json.loads(options)
                    except Exception:
                        options = [options]

            question_text = format_mmmu_prompt(sample.get("question", ""), options)

            content: List[Dict] = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": question_text})

            messages = [{"role": "user", "content": content}]
            batch_items.append({"messages": messages, "raw": sample})

        if not batch_items:
            continue

        try:
            outputs = run_inference_batch(model, processor, batch_items)
        except Exception as exc:
            logger.warning(
                "Inference error on MMMU batch at %d: %s", batch_start, exc
            )
            skipped += len(batch_items)
            continue

        for item, output in zip(batch_items, outputs):
            raw = item["raw"]
            ground_truth = str(raw.get("answer", "")).strip().upper()

            predicted = parse_mc_answer(output)
            if predicted is None:
                skipped += 1
                continue

            if predicted.strip().upper() == ground_truth:
                correct += 1

    accuracy = correct / (total - skipped) if (total - skipped) > 0 else 0.0
    logger.info(
        "MMMU -- accuracy: %.4f (%d/%d), skipped: %d",
        accuracy,
        correct,
        total - skipped,
        skipped,
    )
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_evaluation(cfg: DictConfig, checkpoint_path: str) -> Dict:
    """Run the full evaluation pipeline for a LoRA checkpoint.

    Steps:
    1. Load the model and processor from ``checkpoint_path`` using
       ``cfg.model.name`` as the base model ID.
    2. Initialise an MLflow run under the configured experiment and log
       key parameters.
    3. Evaluate on MathVista and MMMU (each wrapped in a try/except so one
       failure does not abort the other).
    4. Compute the macro-average accuracy across both benchmarks.
    5. Log per-benchmark and macro-average metrics to MLflow.
    6. Write a results JSON file to ``cfg.evaluation.output_path``.
    7. Return the complete results dict.

    Args:
        cfg: Hydra DictConfig rooted at the project config.  Expected keys:
            ``cfg.model.name``, ``cfg.evaluation.output_path``,
            ``cfg.evaluation.batch_size``, ``cfg.evaluation.benchmarks.*``,
            ``cfg.evaluation.mlflow_tracking_uri``,
            ``cfg.evaluation.mlflow_experiment``.
        checkpoint_path: Filesystem path to the LoRA adapter directory.

    Returns:
        Results dict matching the schema::

            {
                "model_checkpoint": str,
                "base_model": str,
                "evaluation_date": "<ISO8601>",
                "benchmarks": {
                    "MathVista": {"accuracy": float, "correct": int, "total": int, "skipped": int},
                    "MMMU":      {"accuracy": float, "correct": int, "total": int, "skipped": int},
                },
                "macro_average": float,
            }
    """
    base_model_id: str = cfg.model.name
    output_path: str = cfg.evaluation.output_path
    tracking_uri: str = cfg.evaluation.mlflow_tracking_uri
    experiment_name: str = cfg.evaluation.mlflow_experiment

    # --- Model loading ---
    model, processor = load_model_and_processor(checkpoint_path, base_model_id)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "checkpoint_path": checkpoint_path,
                "base_model": base_model_id,
                "batch_size": cfg.evaluation.batch_size,
            }
        )

        benchmarks: Dict[str, Dict] = {}

        # MathVista
        try:
            benchmarks["MathVista"] = evaluate_mathvista(model, processor, cfg)
        except Exception as exc:
            logger.error(
                "MathVista evaluation crashed: %s", exc, exc_info=True
            )
            benchmarks["MathVista"] = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "skipped": 0,
            }

        # MMMU
        try:
            benchmarks["MMMU"] = evaluate_mmmu(model, processor, cfg)
        except Exception as exc:
            logger.error("MMMU evaluation crashed: %s", exc, exc_info=True)
            benchmarks["MMMU"] = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "skipped": 0,
            }

        macro_average = (
            sum(b["accuracy"] for b in benchmarks.values()) / len(benchmarks)
            if benchmarks
            else 0.0
        )

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "eval/mathvista_accuracy": benchmarks["MathVista"]["accuracy"],
                "eval/mmmu_accuracy": benchmarks["MMMU"]["accuracy"],
                "eval/macro_average": macro_average,
            }
        )

        results: Dict = {
            "model_checkpoint": checkpoint_path,
            "base_model": base_model_id,
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
            "benchmarks": benchmarks,
            "macro_average": macro_average,
        }

        # Save JSON results
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results saved to %s", out_path)
        mlflow.log_artifact(str(out_path))

    # Human-readable summary
    sep = "=" * 52
    logger.info(sep)
    logger.info("Evaluation Summary")
    logger.info(sep)
    for name, metrics in benchmarks.items():
        logger.info(
            "  %-12s accuracy=%.4f  (%d/%d, skipped=%d)",
            name,
            metrics["accuracy"],
            metrics["correct"],
            metrics["total"] - metrics["skipped"],
            metrics["skipped"],
        )
    logger.info("  %-12s %.4f", "MacroAvg", macro_average)
    logger.info(sep)

    return results


if __name__ == "__main__":
    # This module is intended to be called from scripts/evaluate.py.
    # Running it directly is not supported; use the entrypoint script instead.
    raise SystemExit(
        "Run via scripts/evaluate.py, not directly. "
        "Example: python scripts/evaluate.py checkpoint_path=<path>"
    )
