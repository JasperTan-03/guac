# guac

A proof-of-concept pipeline for training vision-language models (VLMs) with Group Relative Policy Optimization (GRPO) and a dynamic difficulty curriculum. The pipeline runs end-to-end from raw HuggingFace datasets through difficulty scoring, RL training, and benchmark evaluation.

## Pipeline Overview

```
Phase 1 (01_data_prep.py)
  Geometry3K + ScienceQA → data/poc_train_data.jsonl
        ↓
Phase 2 (02_difficulty_judge.py)
  vLLM (Qwen2-VL-7B) scores each problem on AoPS 1–10 rubric
  → data/poc_scored_data.jsonl
        ↓
Phase 3 (03_grpo_train.py)
  GRPO + LoRA fine-tuning with dynamic curriculum
  → checkpoints/
        ↓
Phase 4 (04_evaluate.py)
  Zero-shot evaluation on MathVista + MMMU
  → poc_results.json
```

## Requirements

- Python 3.11+
- NVIDIA GPU with 80GB VRAM (tested on H100 80GB); all scripts pin to `CUDA_VISIBLE_DEVICES=7`
- CUDA 12.x
- [uv](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
git clone <repo>
cd guac

# Create virtual environment and install all dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

> **Note:** `vllm` pulls in `torch` automatically. If you need a specific CUDA build of torch, install it first before running `uv pip install vllm`.

## Phase 1 — Data Engineering

Loads Geometry3K and ScienceQA from HuggingFace, strips multiple-choice formatting, and saves a unified `(id, image, prompt, answer)` schema as JSONL. Images are base64-encoded PNG strings.

```bash
# Full run (~8300 records)
python 01_data_prep.py

# Quick test (50 samples per dataset)
python 01_data_prep.py --max_samples 50
```

**Output:** `data/poc_train_data.jsonl`

| Dataset | HF ID | Records |
|---|---|---|
| Geometry3K | `hiyouga/geometry3k` | 2,101 |
| ScienceQA (vision rows only) | `derek-thomas/ScienceQA` | ~6,200 |

## Phase 2 — VLM Difficulty Judging

Scores each problem on the Art of Problem Solving (AoPS) 1–10 difficulty scale using local `Qwen/Qwen2-VL-7B-Instruct` inference via vLLM. Scores are normalized to `[0.0, 1.0]` for use in the curriculum.

```bash
# Full run
python 02_difficulty_judge.py

# Quick test (10 samples)
python 02_difficulty_judge.py --max_samples 10

# Custom model or batch size
python 02_difficulty_judge.py --model_name Qwen/Qwen2-VL-7B-Instruct --batch_size 32
```

**Output:** `data/poc_scored_data.jsonl` — adds `difficulty` (float), `difficulty_raw_response` (str), `difficulty_parse_error` (bool) to each record.

Checkpoints every 500 rows to `data/poc_scored_data.jsonl.ckpt`; the script resumes automatically if interrupted.

## Phase 3 — GRPO RL Training

Custom GRPO training loop with LoRA fine-tuning and a dynamic difficulty curriculum. After each optimizer step, the target difficulty `T` updates as:

```
T_new = clip(T + eta * tanh(alpha * (R_avg - beta)), d_min, d_max)
```

Two batch sampling modes are available:

- **`baseline`** — selects the B examples with smallest `|d_i - T|` (nearest-neighbor)
- **`gaussian`** — samples B examples proportional to `exp(-(d_i - T)² / (2σ²))`

```bash
# Minimal test (5 steps)
python 03_grpo_train.py \
  --num_train_steps 5 \
  --batch_size 1 \
  --group_size 2 \
  --max_new_tokens 32

# Gaussian sampling (default)
python 03_grpo_train.py --sampling_mode gaussian --num_train_steps 1000

# Baseline sampling
python 03_grpo_train.py --sampling_mode baseline --num_train_steps 1000

# Key arguments
python 03_grpo_train.py \
  --dataset_path data/poc_scored_data.jsonl \
  --output_dir ./checkpoints \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --sampling_mode gaussian \
  --T_init 0.5 \
  --eta 0.05 \
  --alpha 2.0 \
  --beta 0.5 \
  --sigma 0.15 \
  --batch_size 2 \
  --group_size 4 \
  --num_train_steps 1000 \
  --save_steps 100
```

**Output:** LoRA adapter checkpoints in `checkpoints/step_N/` and `checkpoints/final/`.

LoRA config: `r=8`, `lora_alpha=16`, targets `q_proj k_proj v_proj o_proj`. The 7B model in bfloat16 occupies ~14GB; LoRA keeps the full training loop within 80GB.

## Phase 4 — Evaluation

Zero-shot batched evaluation on MathVista and MMMU using the LoRA fine-tuned checkpoint.

```bash
# Full evaluation
python 04_evaluate.py --checkpoint ./checkpoints/final

# Quick test (5 samples per benchmark)
python 04_evaluate.py --checkpoint ./checkpoints/final --max_samples 5

# All options
python 04_evaluate.py \
  --checkpoint ./checkpoints/final \
  --base_model Qwen/Qwen2-VL-7B-Instruct \
  --output_path poc_results.json \
  --batch_size 4
```

**Output:** `poc_results.json`

```json
{
  "model_checkpoint": "./checkpoints/final",
  "base_model": "Qwen/Qwen2-VL-7B-Instruct",
  "evaluation_date": "...",
  "benchmarks": {
    "MathVista": {"accuracy": 0.0, "correct": 0, "total": 1000, "skipped": 0},
    "MMMU": {"accuracy": 0.0, "correct": 0, "total": 30, "skipped": 0}
  },
  "macro_average": 0.0
}
```

| Benchmark | HF ID | Split | Config |
|---|---|---|---|
| MathVista | `AI4Math/MathVista` | `testmini` | — |
| MMMU | `MMMU/MMMU` | `validation` | `Math` |

## Data Schema

All intermediate files are JSONL with one JSON object per line.

**`poc_train_data.jsonl`** (Phase 1 output):
```json
{"id": "geometry3k_0", "image": "<base64 PNG>", "prompt": "Find x.", "answer": "3"}
```

**`poc_scored_data.jsonl`** (Phase 2 output, Phase 3 input):
```json
{"id": "geometry3k_0", "image": "<base64 PNG>", "prompt": "Find x.", "answer": "3",
 "difficulty": 0.3, "difficulty_raw_response": "3", "difficulty_parse_error": false}
```

## Project Structure

```
guac/
├── 01_data_prep.py          # Phase 1: data engineering
├── 02_difficulty_judge.py   # Phase 2: VLM difficulty scoring
├── 03_grpo_train.py         # Phase 3: GRPO RL training
├── 04_evaluate.py           # Phase 4: benchmark evaluation
├── data/
│   ├── poc_train_data.jsonl
│   ├── poc_scored_data.jsonl
│   └── poc_scored_data.jsonl.ckpt
├── checkpoints/             # LoRA adapter weights
├── poc_results.json         # Evaluation results
└── pyproject.toml
```
