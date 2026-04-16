# GUAC Handoff Guide — Running on 8×A6000

This document covers everything you need to deploy and run the GUAC training pipeline on the 8×A6000 cluster. It was written alongside the implementation work that got the codebase from a broken state to fully trainable.

---

## Quick Start (TL;DR)

```bash
git clone <repo-url> && cd guac
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

bash run/01_prepare_data.sh           # ~10 min  — downloads datasets
bash run/02_judge_difficulty.sh       # ~2 hrs   — scores difficulty
bash run/03_train_gaussian.sh         # ~hours   — GRPO training
bash run/04_evaluate.sh               # ~30 min  — MathVista + MMMU eval
```

---

## 1. Hardware & Environment

| Item | Spec |
|------|------|
| GPUs | 8× NVIDIA A6000 (48 GB each) |
| CUDA | 12.x required |
| Python | 3.11+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |

### Install

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

> `vllm` pulls in PyTorch automatically. If you need a specific CUDA build of torch, install it before running `uv pip install`.

---

## 2. GPU Assignment

Every script reads `gpu_id` from `conf/config.yaml` (default: `0`) and sets `CUDA_VISIBLE_DEVICES` to that single GPU. The 7B model with LoRA fits comfortably on a single 48 GB A6000, so **no multi-GPU / DDP is needed for training**.

Override at runtime to use a different GPU:

```bash
# Run training on GPU 2
bash run/03_train_gaussian.sh gpu_id=2

# Run judging on GPU 5 while training runs on GPU 0
bash run/02_judge_difficulty.sh gpu_id=5
```

This means you can **run multiple phases in parallel on different GPUs** — e.g. re-run judging on GPU 5 while training on GPU 0.

---

## 3. Data: Where It Goes & How It Gets There

### Directory Layout

All data lives under `data/` at the project root (gitignored):

```
data/
├── raw/                    # Phase 1 output — per-dataset per-split
│   ├── geometry3k_train.jsonl
│   ├── geometry3k_val.jsonl
│   ├── geometry3k_test.jsonl
│   ├── scienceqa_train.jsonl
│   ├── scienceqa_val.jsonl
│   └── scienceqa_test.jsonl
├── processed/              # Phase 1 output — merged across datasets
│   ├── train.jsonl         # ~8,300 records
│   ├── val.jsonl           # ~2,400 records
│   └── test.jsonl          # ~2,600 records
└── scored/                 # Phase 2 output — with difficulty scores
    ├── train.jsonl         # same records + difficulty field
    ├── val.jsonl
    └── test.jsonl
```

### Paths Are Configurable

The data directories are set in `conf/data/default.yaml`:

```yaml
raw_dir: data/raw
processed_dir: data/processed
scored_dir: data/scored
```

These are **relative to the project root** (Hydra's `version_base="1.2"` keeps CWD at the project root). To change them:

```bash
# Use a different data location
python scripts/prepare_data.py data.raw_dir=/mnt/fast/guac/raw data.processed_dir=/mnt/fast/guac/processed
```

### If You Already Have the Data

If you already have a `data/` directory from a previous run (or someone shared it), just place it at the project root:

```bash
cp -r /path/to/existing/data ./data/
```

Then **skip Phase 1 and Phase 2** and go straight to training:

```bash
bash run/03_train_gaussian.sh
```

Phase 3 reads from `data/scored/train.jsonl`. Phase 4 doesn't need any of this data (it downloads eval benchmarks directly from HuggingFace).

### If You Need To Regenerate Data

Phase 1 downloads from HuggingFace (needs internet). Phase 2 runs the VLM judge (needs a GPU). Both are idempotent — rerunning overwrites previous output.

---

## 4. Record Schema

Every JSONL record follows this format:

```json
{
  "id": "geometry3k_train_0",
  "image": "<base64-encoded PNG string>",
  "prompt": "Find x.",
  "answer": "3"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | Unique identifier, format: `{dataset}_{split}_{index}` |
| `image` | `str` | Base64-encoded PNG. All records have images (text-only ScienceQA rows are filtered out) |
| `prompt` | `str` | The question text. Geometry3K `<image>` tokens are stripped |
| `answer` | `str` | Ground truth. Geometry3K: open-ended (e.g. `"3"`). ScienceQA: resolved choice text (e.g. `"West Virginia"`) |

After Phase 2 (difficulty judging), three fields are added:

| Field | Type | Notes |
|-------|------|-------|
| `difficulty` | `float \| null` | Normalised to [0.0, 1.0] (AoPS score ÷ 10). `null` if parsing failed |
| `difficulty_raw_response` | `str` | Raw LLM output before parsing |
| `difficulty_parse_error` | `bool` | `true` if the score couldn't be parsed |

---

## 5. The 4-Phase Pipeline in Detail

### Phase 1 — Data Preparation

```bash
bash run/01_prepare_data.sh
```

- Downloads **Geometry3K** (`hiyouga/geometry3k`) and **ScienceQA** (`derek-thomas/ScienceQA`) from HuggingFace
- Strips `<image>` tokens from Geometry3K problems
- Filters ScienceQA to vision-only rows (drops text-only)
- Resolves ScienceQA integer answer indices to choice text
- Encodes all images as base64 PNG strings
- Writes `data/raw/` (per-dataset) and `data/processed/` (merged)
- **Runtime:** ~5–10 min (mostly download time)
- **GPU required:** No

### Phase 2 — Difficulty Judging

```bash
bash run/02_judge_difficulty.sh
```

- Loads Qwen2-VL-7B via vLLM as a zero-shot difficulty judge
- Scores each problem on the AoPS 1–10 difficulty scale
- Normalises to [0.0, 1.0] and writes to `data/scored/`
- **Crash-safe:** checkpoints every 500 rows (`.ckpt` files). Rerunning resumes from where it left off
- **Runtime:** ~2 hours for ~13K records on A6000 at batch_size=32
- **GPU required:** Yes (1 GPU, ~40 GB VRAM)

Key vLLM settings (`conf/judge/vllm.yaml`):

```yaml
gpu_memory_utilization: 0.85    # tuned for 48 GB A6000
tensor_parallel_size: 1         # 7B model fits on 1 GPU
batch_size: 32
```

### Phase 3 — GRPO Training

```bash
# Gaussian curriculum (the novel method):
bash run/03_train_gaussian.sh

# Baseline (nearest-neighbour) for comparison:
bash run/03_train_baseline.sh
```

- Loads scored training data from `data/scored/train.jsonl`
- Applies LoRA (rank 8) to Qwen2-VL-7B
- Runs GRPO with adaptive difficulty curriculum for 5000 optimizer steps
- Logs to MLflow every 10 steps
- Saves LoRA checkpoints every 500 steps → `checkpoints/step_N/`
- Final checkpoint → `checkpoints/final/`
- **GPU required:** Yes (1 GPU, ~23 GB VRAM with the `max_pixels` cap; ~28 GB with `batch_size=4`)

The two sampling modes are the experiment:

| Mode | How it picks training examples |
|------|-------------------------------|
| `gaussian` | Probability ∝ exp(−(d_i − T)² / 2σ²) — stochastic, explores via tails |
| `baseline` | Picks the B closest examples to T — deterministic, no exploration |

The hypothesis is that `gaussian` converges faster and avoids catastrophic forgetting.

### Phase 4 — Evaluation

```bash
bash run/04_evaluate.sh
# Or evaluate a specific checkpoint:
bash run/04_evaluate.sh checkpoint_path=checkpoints/step_2000
```

- Loads the LoRA checkpoint + base model
- Evaluates zero-shot on **MathVista** (testmini) and **MMMU** (Math, validation)
- Writes `results/eval_results.json` and logs to MLflow
- **GPU required:** Yes (1 GPU)

---

## 6. Configuration Reference

All config lives in `conf/`. Override any value via CLI args.

### `conf/config.yaml` — Root

```yaml
gpu_id: 0       # which GPU to use (0-7 on the A6000 cluster)
seed: 42
```

### `conf/data/default.yaml` — Data Paths

```yaml
raw_dir: data/raw
processed_dir: data/processed
scored_dir: data/scored

datasets:
  geometry3k:
    hf_id: hiyouga/geometry3k
  scienceqa:
    hf_id: derek-thomas/ScienceQA
```

### `conf/training/grpo.yaml` — Training Hyperparameters

| Key | Default | What it does |
|-----|---------|-------------|
| `num_train_steps` | 5000 | Total optimizer steps |
| `batch_size` | 2 | Examples per micro-step |
| `group_size` | 4 | Rollouts generated per example |
| `gradient_accumulation_steps` | 4 | Micro-steps before optimizer step |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `sampling_mode` | gaussian | `gaussian` or `baseline` |
| `T_init` | 0.5 | Initial curriculum target difficulty |
| `eta` | 0.05 | Curriculum update step size |
| `sigma` | 0.15 | Gaussian sampling bandwidth |
| `d_min` | 0.3 | Minimum target difficulty |
| `d_max` | 1.0 | Maximum target difficulty |
| `save_steps` | 500 | Checkpoint frequency |
| `log_steps` | 10 | MLflow logging frequency |

**Effective batch size per optimizer step:** `batch_size × group_size × gradient_accumulation_steps` = 2 × 4 × 4 = 32 forward passes.

### `conf/model/qwen2_vl_7b.yaml` — Model & LoRA

```yaml
name: Qwen/Qwen2-VL-7B-Instruct
dtype: bfloat16
lora:
  r: 8
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_dropout: 0.05
  bias: none
```

### `conf/judge/vllm.yaml` — Difficulty Judge

```yaml
model_name: Qwen/Qwen2-VL-7B-Instruct
gpu_memory_utilization: 0.85     # 48 GB A6000 headroom
tensor_parallel_size: 1
max_model_len: 4096
batch_size: 32
checkpoint_interval: 500
```

### `conf/evaluation/default.yaml` — Eval Benchmarks

```yaml
output_path: results/eval_results.json
batch_size: 4
benchmarks:
  mathvista:
    hf_id: AI4Math/MathVista
    split: testmini
  mmmu:
    hf_id: MMMU/MMMU
    config: Math
    split: validation
```

---

## 7. MLflow Monitoring

Training and evaluation both log to `mlruns/` (gitignored).

**On the GPU machine:**

```bash
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

**On your local machine (SSH tunnel):**

```bash
ssh -L 5000:127.0.0.1:5000 user@gpu-server-ip
# Then open http://localhost:5000
```

**Training metrics** (logged every 10 steps): `train/loss`, `train/reward`, `curriculum/T`, `train/learning_rate`

**Eval metrics**: `eval/mathvista_accuracy`, `eval/mmmu_accuracy`, `eval/macro_average`

---

## 8. Running Phases in Parallel on Multiple GPUs

Since each phase only uses 1 GPU, you can run independent phases simultaneously:

```bash
# Terminal 1: training on GPU 0
bash run/03_train_gaussian.sh gpu_id=0

# Terminal 2: baseline training on GPU 1 (for comparison)
bash run/03_train_baseline.sh gpu_id=1

# Terminal 3: evaluate a checkpoint on GPU 2
bash run/04_evaluate.sh checkpoint_path=checkpoints/step_1000 gpu_id=2
```

---

## 9. Quick Inference Test

To sanity-check a checkpoint before running full eval:

```bash
python scripts/infer.py \
    checkpoint_path=checkpoints/step_500 \
    question="What is the area of a triangle with base 6 and height 4?" \
    gpu_id=3
```

To compare against the base model (no LoRA):

```bash
python scripts/infer.py \
    checkpoint_path=none \
    question="What is 12 times 7?"
```

---

## 10. What Was Fixed to Get Here

The original codebase (by Jasper) was missing several pieces. Here's what was added/fixed:

### New Files Created

| File | What it provides |
|------|-----------------|
| `src/guac/data/__init__.py` | Package init — makes `guac.data` importable |
| `src/guac/data/utils.py` | `encode_image()` / `decode_image()` — base64 PNG round-trip for JSONL storage |
| `src/guac/data/prep.py` | `load_jsonl()` / `save_jsonl()` — JSONL I/O with atomic writes. `prepare_all()` — full Phase 1 data pipeline |
| `conf/data/default.yaml` | Hydra data config (paths + HuggingFace dataset IDs). Without this, every script crashed on startup |

### Config Changes

| File | Change | Why |
|------|--------|-----|
| `conf/config.yaml` | `gpu_id: 7` → `0` | Old default was for the H100 machine's GPU index. Standard default for A6000 cluster |
| `conf/judge/vllm.yaml` | `gpu_memory_utilization: 0.90` → `0.85` | 90% of 48 GB = 43 GB is too tight. 85% = 40.8 GB gives headroom |

### Bug Fixes in `src/guac/training/trainer.py`

Three bugs in `_compute_batch_kl()` (the KL penalty path). These are dormant when `kl_coeff=0.0` (the default), but would cause incorrect training if KL penalty were enabled:

1. **Missing vision inputs**: forward passes omitted `pixel_values` and `image_grid_thw`, so the model had no image context during KL computation
2. **Wrong attention mask**: used all-ones instead of preserving the prompt's padding mask, causing attention to padding tokens
3. **Dead parameter**: removed unused `prompt_inputs` argument

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'guac'`**
→ You need to install the package: `uv pip install -e ".[dev]"`

**`MissingConfigException: ... data/default.yaml`**
→ The `conf/data/default.yaml` file is missing. Make sure you have the latest code.

**`CUDA out of memory` during training**
→ The `max_pixels` cap in `conf/model/qwen2_vl_7b.yaml` (default 401408 = 512×28×28) prevents the largest images from blowing up activation memory.  If you still OOM, reduce `batch_size` to 1 or `group_size` to 2, or lower `max_pixels`:
```bash
bash run/03_train_gaussian.sh training.batch_size=1 training.group_size=2
# Or lower image resolution further:
bash run/03_train_gaussian.sh model.max_pixels=200704
```

**`CUDA out of memory` during judging**
→ Lower the vLLM batch size and memory utilization:
```bash
bash run/02_judge_difficulty.sh judge.batch_size=16 judge.gpu_memory_utilization=0.75
```

**Phase 2 was interrupted mid-run**
→ Just rerun it. The `.ckpt` files in `data/scored/` enable automatic resume.

**`FileNotFoundError: data/scored/train.jsonl`**
→ Phase 2 hasn't been run yet. Run `bash run/02_judge_difficulty.sh` first, or copy in pre-existing scored data.

**Training loss is 0.0 / reward is stuck at 0.0**
→ This was a known bug (fixed in commit `49745fd`). If you still see it, check that `data/scored/train.jsonl` has non-null difficulty values and that the model is generating non-empty outputs (check the sample logging in the training output).
