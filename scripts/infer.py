"""Inference script for testing a LoRA checkpoint interactively.

Loads a base model + LoRA adapter and runs greedy inference on a single
question, optionally with an image.  Useful for verifying that a checkpoint
has learned something before running full benchmark evaluation.

Usage::

    # Test the final checkpoint with a text-only question
    python scripts/infer.py \\
        checkpoint_path=checkpoints/final \\
        question="What is the capital of France?"

    # Test a mid-training checkpoint with a geometry image
    python scripts/infer.py \\
        checkpoint_path=checkpoints/step_500 \\
        question="Find the value of x." \\
        image_path=data/sample.png

    # Use the base model without any adapter (for comparison)
    python scripts/infer.py \\
        checkpoint_path=none \\
        question="What is 12 times 7?"

    # Override GPU
    python scripts/infer.py \\
        checkpoint_path=checkpoints/final \\
        question="Find x." \\
        gpu_id=0
"""

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run a single inference with the specified checkpoint.

    Args:
        cfg: Hydra DictConfig.  Expects additional CLI overrides:
            - ``checkpoint_path`` (str): Path to LoRA adapter directory,
              or ``"none"`` to use the base model only.
            - ``question`` (str): The question text to answer.
            - ``image_path`` (str, optional): Path to an image file.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    checkpoint_path: str = OmegaConf.select(cfg, "checkpoint_path", default="")
    question: str = OmegaConf.select(cfg, "question", default="")
    image_path: str = OmegaConf.select(cfg, "image_path", default="")

    if not question:
        raise ValueError(
            "question must be provided: python scripts/infer.py checkpoint_path=... question='your question here'"
        )

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    log.info("Loading base model: %s", cfg.model.name)
    base = AutoModelForVision2Seq.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # Attach LoRA adapter (optional)
    # ------------------------------------------------------------------
    use_adapter = checkpoint_path and checkpoint_path.lower() != "none"
    if use_adapter:
        try:
            from peft import PeftModel

            log.info("Attaching LoRA adapter from: %s", checkpoint_path)
            model = PeftModel.from_pretrained(base, checkpoint_path)
            log.info("Adapter loaded successfully.")
        except Exception as exc:
            log.warning("Failed to load adapter (%s) — using base model.", exc)
            model = base
    else:
        log.info("No adapter specified — using base model for comparison.")
        model = base

    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.model.name, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Optionally load image
    # ------------------------------------------------------------------
    pil_image: Optional[Image.Image] = None
    if image_path:
        p = Path(image_path)
        if p.exists():
            pil_image = Image.open(p).convert("RGB")
            log.info("Image loaded from: %s (%dx%d)", p, pil_image.width, pil_image.height)
        else:
            log.warning("image_path=%s not found — proceeding text-only.", image_path)

    # ------------------------------------------------------------------
    # Build chat input
    # ------------------------------------------------------------------
    content = []
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[pil_image] if pil_image is not None else None,
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Generate (greedy, no sampling)
    # ------------------------------------------------------------------
    log.info("Generating answer...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    answer = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"Checkpoint : {checkpoint_path or 'base model'}")
    print(f"Question   : {question}")
    if image_path:
        print(f"Image      : {image_path}")
    print(sep)
    print(f"Answer     : {answer}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
