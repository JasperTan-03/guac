import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Dependencies:
#   pip install transformers peft accelerate torch datasets tqdm pillow

import argparse
import ast
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from peft import PeftModel
except ImportError as e:
    raise ImportError("peft is required: pip install peft") from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA fine-tuned Qwen2-VL-7B checkpoint on MathVista and MMMU."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="poc_results.json",
        help="Path where poc_results.json will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples to process per batch.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (None = all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map strategy: 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(
    checkpoint: str,
    base_model_id: str,
    device: str,
) -> Tuple[Any, Any]:
    """Load the base Qwen2-VL model and merge the LoRA adapter.

    Args:
        checkpoint: Path to the LoRA adapter directory.
        base_model_id: HuggingFace model ID or local path for the base model.
        device: Device map strategy passed to from_pretrained ('auto', etc.).

    Returns:
        A (model, processor) tuple ready for inference.
    """
    logger.info("Loading base model: %s", base_model_id)
    base = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    logger.info("Attaching LoRA adapter: %s", checkpoint)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()

    logger.info("Loading processor: %s", base_model_id)
    processor = AutoProcessor.from_pretrained(base_model_id)

    return model, processor


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_mc_answer(text: str) -> Optional[str]:
    """Extract a multiple-choice letter (A/B/C/D) from model output.

    Looks for common answer-prefix patterns first, then falls back to the
    first standalone letter found anywhere in the text.

    Args:
        text: Raw decoded output from the model.

    Returns:
        Uppercase letter 'A', 'B', 'C', or 'D', or None if not found.
    """
    if not text:
        return None

    # Strip reasoning traces wrapped in <think>...</think>
    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        text = think_match.group(1)

    text = text.strip()

    # Pattern 1: explicit answer prefixes
    prefixes = [
        r"(?:the\s+)?answer\s+is[:\s]+([ABCD])\b",
        r"(?:answer|option)[:\s]+([ABCD])\b",
        r"\b([ABCD])\s*\.",
        r"\*\*([ABCD])\*\*",
    ]
    for pat in prefixes:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Pattern 2: first standalone A/B/C/D in the text
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()

    return None


def parse_numeric_answer(text: str) -> Optional[str]:
    """Extract the last number or mathematical expression from model output.

    Args:
        text: Raw decoded output from the model.

    Returns:
        String representation of the last number found, or None.
    """
    if not text:
        return None

    # Strip reasoning traces wrapped in <think>...</think>
    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        text = think_match.group(1)

    text = text.strip()

    # Try boxed{} expression first (common in math outputs)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Fall back to last integer or decimal number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_mathvista_prompt(question: str, choices: Optional[List]) -> str:
    """Build the text portion of a MathVista prompt.

    Args:
        question: The raw question string from the dataset.
        choices: List of answer choices, or None for free-form questions.

    Returns:
        Formatted question string ready to be inserted into the chat template.
    """
    if choices:
        choices_str = ", ".join(str(c) for c in choices)
        return f"{question}\nChoices: {choices_str}\nAnswer with the correct option letter."
    return f"{question}\nProvide only the final numerical answer."


def format_mmmu_prompt(question: str, options: List) -> str:
    """Build the text portion of an MMMU prompt.

    Args:
        question: The raw question string from the dataset.
        options: List of option strings (typically four).

    Returns:
        Formatted question string ready to be inserted into the chat template.
    """
    options_str = ", ".join(str(o) for o in options)
    return (
        f"{question}\nOptions: {options_str}\n"
        "Answer with a single letter (A, B, C, or D)."
    )


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _extract_pil_image(image_field: Any) -> Optional[Image.Image]:
    """Coerce a dataset image field to a PIL Image.

    HuggingFace datasets can return image fields as dicts with 'bytes'/'path'
    keys or as PIL Images directly.

    Args:
        image_field: Value of an image column from a HuggingFace dataset row.

    Returns:
        PIL Image, or None if the field is absent / cannot be decoded.
    """
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"]:
            import io
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if "path" in image_field and image_field["path"]:
            try:
                return Image.open(image_field["path"]).convert("RGB")
            except Exception:
                return None
    return None


def run_inference_batch(
    model: Any,
    processor: Any,
    batch_items: List[Dict],
) -> List[str]:
    """Run batched inference for a list of prepared samples.

    Each item in batch_items must have:
        - 'messages': list formatted for the Qwen2-VL chat template.

    Args:
        model: The loaded (PeftModel-wrapped) vision-language model.
        processor: The AutoProcessor for the base model.
        batch_items: List of dicts, each with a 'messages' key.

    Returns:
        List of decoded output strings (one per input item).
    """
    # Build list-of-conversation-texts using the chat template
    texts = []
    images_per_sample = []
    for item in batch_items:
        text = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

        # Collect PIL images in the order they appear in messages
        pil_images = []
        for msg in item["messages"]:
            for content_part in msg.get("content", []):
                if isinstance(content_part, dict) and content_part.get("type") == "image":
                    img = content_part.get("image")
                    if img is not None:
                        pil_images.append(img)
        images_per_sample.append(pil_images if pil_images else None)

    # Flatten images for the processor (it expects a flat list aligned with texts)
    # Qwen2-VL processor accepts images=None when no images are present
    all_images = []
    has_any_image = any(imgs is not None for imgs in images_per_sample)

    if has_any_image:
        for imgs in images_per_sample:
            if imgs:
                all_images.extend(imgs)
            # If a sample has no image we don't append anything; the processor
            # matches images to <image> tokens in the text, so this is safe as
            # long as samples without images have no <image> token in their text.

    try:
        inputs = processor(
            text=texts,
            images=all_images if has_any_image else None,
            return_tensors="pt",
            padding=True,
        )
    except Exception as e:
        logger.warning("Processor error for batch: %s — returning empty outputs", e)
        return [""] * len(batch_items)

    # Move tensors to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

    # Slice off prompt tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]

    decoded = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_mathvista(
    model: Any,
    processor: Any,
    args: argparse.Namespace,
) -> Dict:
    """Evaluate the model on MathVista (testmini split).

    Args:
        model: Loaded vision-language model.
        processor: Corresponding processor.
        args: Parsed CLI arguments.

    Returns:
        Dict with keys 'accuracy', 'correct', 'total', 'skipped'.
    """
    logger.info("Loading MathVista dataset (AI4Math/MathVista, split=testmini)...")
    try:
        dataset = load_dataset("AI4Math/MathVista", split="testmini", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load MathVista: %s", e)
        return {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    skipped = 0
    total = len(dataset)

    batch_size = args.batch_size
    samples = list(dataset)

    for batch_start in tqdm(
        range(0, total, batch_size), desc="MathVista", unit="batch"
    ):
        batch_raw = samples[batch_start : batch_start + batch_size]
        batch_items = []
        batch_indices = []  # track which raw samples remain valid

        for idx, sample in enumerate(batch_raw):
            # MathVista uses 'decoded_image' (PIL) not 'image' (str path)
            image = _extract_pil_image(sample.get("decoded_image") or sample.get("image"))
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
            batch_items.append({"messages": messages, "raw": sample, "choices": choices})
            batch_indices.append(batch_start + idx)

        if not batch_items:
            continue

        try:
            outputs = run_inference_batch(model, processor, batch_items)
        except Exception as e:
            logger.warning("Inference error on MathVista batch starting at %d: %s", batch_start, e)
            skipped += len(batch_items)
            continue

        for item, output in zip(batch_items, outputs):
            raw = item["raw"]
            choices = item["choices"]
            ground_truth = str(raw.get("answer", "")).strip()

            if choices:
                predicted = parse_mc_answer(output)
            else:
                predicted = parse_numeric_answer(output)

            if predicted is None:
                skipped += 1
                continue

            if predicted.strip().lower() == ground_truth.strip().lower():
                correct += 1

    accuracy = correct / (total - skipped) if (total - skipped) > 0 else 0.0
    logger.info(
        "MathVista — accuracy: %.4f (%d/%d), skipped: %d",
        accuracy, correct, total - skipped, skipped,
    )
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}


def evaluate_mmmu(
    model: Any,
    processor: Any,
    args: argparse.Namespace,
) -> Dict:
    """Evaluate the model on MMMU (validation split, all subjects).

    Args:
        model: Loaded vision-language model.
        processor: Corresponding processor.
        args: Parsed CLI arguments.

    Returns:
        Dict with keys 'accuracy', 'correct', 'total', 'skipped'.
    """
    # Use 'Math' config for this PoC (MMMU has no single 'all' config)
    mmmu_config = "Math"
    logger.info("Loading MMMU dataset (MMMU/MMMU, config=%s, split=validation)...", mmmu_config)
    try:
        dataset = load_dataset("MMMU/MMMU", mmmu_config, split="validation")
    except Exception as e:
        logger.error("Failed to load MMMU: %s", e)
        return {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    skipped = 0
    total = len(dataset)

    batch_size = args.batch_size
    samples = list(dataset)

    for batch_start in tqdm(
        range(0, total, batch_size), desc="MMMU", unit="batch"
    ):
        batch_raw = samples[batch_start : batch_start + batch_size]
        batch_items = []

        for idx, sample in enumerate(batch_raw):
            image = _extract_pil_image(sample.get("image_1"))
            # image_1 may be absent — MMMU samples without images still have text

            options = sample.get("options") or []
            # MMMU 'options' is a string like "['A', 'B', 'C']" — use ast.literal_eval
            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = [options]

            question_text = format_mmmu_prompt(sample.get("question", ""), options)

            content: List[Dict] = []
            if image is not None:
                content.append({"type": "image", "image": image})
            else:
                logger.debug(
                    "MMMU sample %d has no image_1; proceeding text-only.",
                    batch_start + idx,
                )
            content.append({"type": "text", "text": question_text})

            messages = [{"role": "user", "content": content}]
            batch_items.append({"messages": messages, "raw": sample})

        if not batch_items:
            continue

        try:
            outputs = run_inference_batch(model, processor, batch_items)
        except Exception as e:
            logger.warning("Inference error on MMMU batch starting at %d: %s", batch_start, e)
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
        "MMMU — accuracy: %.4f (%d/%d), skipped: %d",
        accuracy, correct, total - skipped, skipped,
    )
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: Dict, output_path: str) -> None:
    """Write the results dict to a JSON file.

    Args:
        results: Fully populated results dictionary.
        output_path: Destination file path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, load model, run benchmarks, save results."""
    args = parse_args()

    model, processor = load_model_and_processor(
        checkpoint=args.checkpoint,
        base_model_id=args.base_model,
        device=args.device,
    )

    benchmarks: Dict[str, Dict] = {}

    # MathVista
    try:
        benchmarks["MathVista"] = evaluate_mathvista(model, processor, args)
    except Exception as e:
        logger.error("MathVista evaluation crashed: %s", e, exc_info=True)
        benchmarks["MathVista"] = {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    # MMMU
    try:
        benchmarks["MMMU"] = evaluate_mmmu(model, processor, args)
    except Exception as e:
        logger.error("MMMU evaluation crashed: %s", e, exc_info=True)
        benchmarks["MMMU"] = {"accuracy": 0.0, "correct": 0, "total": 0, "skipped": 0}

    macro_avg = sum(b["accuracy"] for b in benchmarks.values()) / len(benchmarks)

    results = {
        "model_checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "macro_average": macro_avg,
    }

    save_results(results, args.output_path)

    # Human-readable summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    for name, metrics in benchmarks.items():
        print(
            f"  {name:<12} accuracy={metrics['accuracy']:.4f}  "
            f"({metrics['correct']}/{metrics['total']}, skipped={metrics['skipped']})"
        )
    print(f"  {'MacroAvg':<12} {macro_avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
