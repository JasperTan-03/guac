import json
from pathlib import Path
import logging
from guac.data.utils import encode_image, safe_load_image, strip_mc_from_text
import datasets as hf_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chartqa_prep")

def process_chartqa_train():
    split = "train"
    hf_id = "HuggingFaceM4/ChartQA"
    logger.info(f"Loading {hf_id} split={split}")
    ds = hf_datasets.load_dataset(hf_id, split=split)
    
    records = []
    skipped = 0
    for idx, row in enumerate(ds):
        row_id = f"chartqa_{split}_{idx}"
        
        image_b64 = None
        pil_img = safe_load_image(row.get("image"))
        if pil_img is not None:
            image_b64 = encode_image(pil_img)
            
        prompt = row.get("query", "")
        prompt = strip_mc_from_text(prompt)
        
        label = row.get("label", [])
        if isinstance(label, list) and len(label) > 0:
            answer = str(label[0]).strip()
        else:
            answer = str(label).strip()
            
        if not prompt or not answer:
            skipped += 1
            continue
            
        records.append({
            "id": row_id,
            "image": image_b64,
            "prompt": prompt,
            "answer": answer
        })
    
    logger.info(f"Kept {len(records)}, skipped {skipped}")
    
    # Save to raw
    raw_path = Path("/data/troy/datasets/guac/raw/chartqa_train.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info(f"Saved to {raw_path}")
    
    # Print 3 examples
    print("\n--- 3 PROCESSED EXAMPLES ---")
    for r in records[:3]:
        print(f"ID: {r['id']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Answer: {r['answer']}")
        print(f"Has Image: {r['image'] is not None}")
        print("-" * 20)

if __name__ == "__main__":
    process_chartqa_train()
