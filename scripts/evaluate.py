import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from guac.evaluation.evaluator import (
    _extract_pil_image,
    parse_mc_answer,
    parse_numeric_answer,
)

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the evaluation pipeline on local datasets using the base model."""
    # Use the base model rather than a LoRA checkpoint
    base_model_id: str = cfg.model.name
    log.info("Loading raw base model: %s", base_model_id)

    llm = LLM(model=base_model_id, tensor_parallel_size=4, pipeline_parallel_size=2, disable_log_stats=True)

    # Read from local dataset instead of huggingface benchmarks
    # Map to local dataset 
    test_file = Path(cfg.data.processed_dir) / "test.jsonl"
    log.info("Loading local test dataset from %s", test_file)
    
    samples = []
    if test_file.exists():
        with open(test_file, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        # Reduce samples by 1/10th for quick testing
        # samples = samples[:max(1, len(samples) // 10)]
    else:
        log.warning("Test file %s not found.", test_file)

    # Output metrics logic
    output_path = Path(cfg.evaluation.output_path)
    tracking_uri = cfg.evaluation.mlflow_tracking_uri
    experiment_name = cfg.evaluation.mlflow_experiment

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "base_model": base_model_id,
            "batch_size": cfg.evaluation.batch_size,
            "dataset_path": str(test_file)
        })

        benchmarks: Dict[str, Dict] = {}
        
        # Evaluate on the loaded local dataset
        total = 0
        skipped = 0
        from collections import defaultdict
        
        counter = lambda: {"total": 0, "correct": 0, "skipped": 0}
        dataset_metrics = defaultdict(counter)

        import base64
        import io
        
        batch_size = 50

        sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)

        for i in range(0, len(samples), batch_size):
            batch_raw = samples[i:i+batch_size]
            all_messages = []
            
            for item in batch_raw:
                prompt_text = str(item.get("prompt", "")).strip()
                
                # Some datasets might not have images, check if "image" is in item and not None
                content = []
                img_data = item.get("image")
                if img_data:
                    # decode base64
                    try:
                        # Append base64 data URI format for vLLM openai format
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}})
                    except Exception as e:
                        log.warning(f"Failed to load base64 image: {e}")
                
                content.append({"type": "text", "text": prompt_text})
                
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
                all_messages.append(messages)

            outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

            for item, output_obj in zip(batch_raw, outputs):
                output = output_obj.outputs[0].text
                dataset = item.get("id", "Unknown_").split("_")[0]
                dataset_metrics[dataset]["total"] += 1
                total += 1
                
                if not output:
                    dataset_metrics[dataset]["skipped"] += 1
                    skipped += 1
                    continue
                    
                target = str(item.get("answer", "")).strip().upper()
                is_correct = False
                
                if total <= 10:  # Print first 10 for validation
                    log.info(f"Target: {target} | Output (raw): {output[:100]}...")

                if item.get("is_multiple_choice", False):
                    pred = parse_mc_answer(output)
                    if pred and target and pred == target:
                        is_correct = True
                else:
                    pred = parse_numeric_answer(output)
                    if pred and target and pred == target:
                        is_correct = True

                if total <= 10:
                    log.info(f"  -> Extracted pred: {pred} | Correct? {is_correct}")
                        
                if is_correct:
                    dataset_metrics[dataset]["correct"] += 1
                    
        # --- Evaluate MathVista ---
        from datasets import load_dataset
        import ast

        log.info("Loading MathVista dataset...")
        try:
            mv_dataset = load_dataset('AI4Math/MathVista', split='testmini', trust_remote_code=True)
            mv_samples = list(mv_dataset)
            # Reduce samples by 1/10th for quick testing as well
            mv_samples = mv_samples[:max(1, len(mv_samples) // 10)]
            for i in range(0, len(mv_samples), batch_size):
                batch_raw = mv_samples[i:i+batch_size]
                all_messages = []
                for item in batch_raw:
                    question = str(item.get("question", "")).strip()
                    choices = item.get("choices")
                    image = item.get("decoded_image") or item.get("image")
                    
                    if choices:
                        choices_str = ", ".join(str(c) for c in choices)
                        prompt_text = f"{question}\nChoices: {choices_str}\nAnswer with the correct option letter only."
                    else:
                        prompt_text = f"{question}\nProvide only the final numerical answer."

                    content = []
                    if image and isinstance(image, Image.Image):
                        buffered = io.BytesIO()
                        image.convert("RGB").save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
                    
                    content.append({"type": "text", "text": prompt_text})
                    all_messages.append([{"role": "user", "content": content}])

                outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)
                for item, output_obj in zip(batch_raw, outputs):
                    output = output_obj.outputs[0].text
                    dataset_metrics["MathVista"]["total"] += 1
                    
                    if not output:
                        dataset_metrics["MathVista"]["skipped"] += 1
                        continue
                        
                    target = str(item.get("answer", "")).strip()
                    is_correct = False
                    
                    choices = item.get("choices")
                    if choices:
                        pred = parse_mc_answer(output)
                    else:
                        pred = parse_numeric_answer(output)
                        
                    if pred and target and pred.strip().lower() == target.lower():
                        is_correct = True
                        
                    if is_correct:
                        dataset_metrics["MathVista"]["correct"] += 1
        except Exception as e:
            log.warning(f"MathVista failed: {e}")

        # --- Evaluate MMMU ---
        log.info("Loading MMMU dataset...")
        try:
            mmmu_dataset = load_dataset('MMMU/MMMU', 'Math', split='validation')
            mmmu_samples = list(mmmu_dataset)
            # Reduce samples by 1/10th for quick testing as well
            mmmu_samples = mmmu_samples[:max(1, len(mmmu_samples) // 10)]
            for i in range(0, len(mmmu_samples), batch_size):
                batch_raw = mmmu_samples[i:i+batch_size]
                all_messages = []
                for item in batch_raw:
                    question = str(item.get("question", "")).strip()
                    options_str = item.get("options", "[]")
                    try:
                        options = ast.literal_eval(options_str)
                    except:
                        options = []
                    
                    opt_str = ", ".join(str(o) for o in options)
                    prompt_text = f"{question}\nOptions: {opt_str}\nAnswer with a single letter (A, B, C, or D) only."

                    content = []
                    image = item.get("image_1")
                    if image and isinstance(image, Image.Image):
                        buffered = io.BytesIO()
                        image.convert("RGB").save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
                    
                    content.append({"type": "text", "text": prompt_text})
                    all_messages.append([{"role": "user", "content": content}])

                outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)
                for item, output_obj in zip(batch_raw, outputs):
                    output = output_obj.outputs[0].text
                    dataset_metrics["MMMU"]["total"] += 1
                    
                    if not output:
                        dataset_metrics["MMMU"]["skipped"] += 1
                        continue
                        
                    target = str(item.get("answer", "")).strip()
                    is_correct = False
                    
                    pred = parse_mc_answer(output)
                    if pred and target and pred.strip().lower() == target.lower():
                        is_correct = True
                        
                    if is_correct:
                        dataset_metrics["MMMU"]["correct"] += 1
        except Exception as e:
            log.warning(f"MMMU failed: {e}")
                    
        macro_accs = []
        for dataset, mets in dataset_metrics.items():
            acc = mets["correct"] / max(mets["total"] - mets["skipped"], 1) if mets["total"] > 0 else 0.0
            benchmarks[dataset] = {
                "accuracy": acc,
                "correct": mets["correct"],
                "total": mets["total"],
                "skipped": mets["skipped"]
            }
            macro_accs.append(acc)

        macro_average = sum(macro_accs) / max(len(macro_accs), 1)

        mlflow.log_metrics({
            "eval/macro_average": macro_average,
        })
        for dataset, mets in benchmarks.items():
            mlflow.log_metric(f"eval/{dataset}_accuracy", mets["accuracy"])

        results = {
            "base_model": base_model_id,
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
            "benchmarks": benchmarks,
            "macro_average": macro_average,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
            
        log.info("Results saved to %s", output_path)
        mlflow.log_artifact(str(output_path))
        
        # Human-readable summary
        sep = "=" * 52
        print("\n" + sep)
        print("EVALUATION SUMMARY")
        print(sep)
        for name, metrics in benchmarks.items():
            print(
                f"  {name:<12} accuracy={metrics['accuracy']:.4f}  "
                f"({metrics['correct']}/{metrics['total'] - metrics['skipped']}, skipped={metrics['skipped']})"
            )
        print(f"  {'MacroAvg':<12} {macro_average:.4f}")
        print(sep + "\n")
        log.info(sep)
        log.info("Evaluation Summary")
        log.info(sep)
        for name, metrics in benchmarks.items():
            log.info(
                "  %-12s accuracy=%.4f  (%d/%d, skipped=%d)",
                name,
                metrics["accuracy"],
                metrics["correct"],
                metrics["total"] - metrics["skipped"],
                metrics["skipped"],
            )
        log.info("  %-12s %.4f", "MacroAvg", macro_average)
        log.info(sep)

if __name__ == "__main__":
    main()
