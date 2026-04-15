# GUAC: GRPO Utilizing Adaptive Curriculums for Vision Language Models

**Team:** Jasper Tan, Troy Dutton, Varun Arumugam | *The University of Texas at Austin*
📄 **[Read the Full Paper / Milestone Report Here](./Deep_RL_Project_Milestone.pdf)**

**TL;DR:** Standard reinforcement learning often trains VLMs on randomly sampled batches, ignoring the inherent difficulty of individual problems. GUAC is an end-to-end pipeline that fine-tunes Qwen2-VL-7B-Instruct using GRPO, testing a novel **Gaussian sampling schedule** against a standard minimum-distance baseline to dynamically pace problem difficulty during training.

---

```
Phase 1 — Data Prep         Phase 2 — Difficulty Judging
Geometry3K + ScienceQA  →   vLLM scores each problem
data/raw/ + processed/      on AoPS 1–10 scale
                                      ↓
Phase 4 — Evaluation    ←   Phase 3 — GRPO Training
MathVista + MMMU            LoRA + dynamic curriculum
results/eval_results.json   checkpoints/
```

## Requirements

- Python 3.11+
- NVIDIA GPU with ≥48 GB VRAM (tested on A6000 48GB)
- CUDA 12.x
- [uv](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
git clone <repo>
cd guac

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

> **Note:** `vllm` pulls in `torch` automatically. If you need a specific CUDA build, install torch first.

## Running the Pipeline

Each phase has a shell wrapper in `run/` and a Hydra script in `scripts/`. The shell wrappers activate the venv for you.

```bash
bash run/01_prepare_data.sh       # ~5–10 min: download + write data/raw/ + data/processed/
bash run/02_judge_difficulty.sh   # ~2 hrs:   score all splits → data/scored/
bash run/03_train_gaussian.sh     # variable: 5000 steps → checkpoints/
bash run/04_evaluate.sh           # ~30 min:  MathVista + MMMU → results/
```

Pass Hydra overrides as extra arguments:

```bash
bash run/03_train_gaussian.sh training.num_train_steps=2000
bash run/04_evaluate.sh checkpoint_path=checkpoints/step_2000
```

Or call scripts directly for full flexibility:

```bash
python scripts/train.py training.sampling_mode=baseline training.batch_size=4
python scripts/evaluate.py checkpoint_path=checkpoints/final gpu_id=0
```

## Configuration

All hyperparameters live in `conf/`. The root config is `conf/config.yaml`.

```
conf/
├── config.yaml              # gpu_id, seed, defaults list
├── data/default.yaml        # dataset HF IDs, splits, directories
├── model/qwen2_vl_7b.yaml   # model name, dtype, LoRA settings
├── training/grpo.yaml       # steps, LR, curriculum, MLflow
├── judge/vllm.yaml          # vLLM engine settings
└── evaluation/default.yaml  # benchmark configs, output path
```

Override any value at runtime:

```bash
python scripts/train.py training.learning_rate=2e-5 training.group_size=8
python scripts/judge_difficulty.py judge.batch_size=16
python scripts/train.py gpu_id=0
```

## Datasets

| Dataset | HF ID | Splits | Records (vision-only) |
|---|---|---|---|
| Geometry3K | `hiyouga/geometry3k` | train / val / test | 2101 / 300 / 601 |
| ScienceQA | `derek-thomas/ScienceQA` | train / val / test | ~6200 / ~2000 / ~2000 |
| **Combined** | — | train / val / test | ~8300 / ~2300 / ~2600 |

Phase 1 writes:
- `data/raw/{dataset}_{split}.jsonl` — per-dataset per-split
- `data/processed/{train,val,test}.jsonl` — merged across datasets

## Difficulty Judging (Phase 2)

Uses `Qwen/Qwen2-VL-7B-Instruct` via vLLM to score each problem on the [Art of Problem Solving](https://artofproblemsolving.com) 1–10 scale. Scores are normalised to `[0.0, 1.0]` (AoPS score ÷ 10).

- Runs at ~90 rows/sec on H100 (~60 rows/sec on A6000) with `batch_size=32`
- Checkpoints every 500 rows; re-running after a crash resumes automatically
- Output: `data/scored/{train,val,test}.jsonl` — adds `difficulty`, `difficulty_raw_response`, `difficulty_parse_error`

## Training (Phase 3)

Custom GRPO loop with LoRA and a dynamic difficulty curriculum.

**Curriculum update rule** (per optimizer step):
```
T_new = clip(T + η · tanh(α · (R_avg − β)), d_min, d_max)
```

**Sampling modes** (set via `training.sampling_mode`):
- `gaussian` *(default)* — weighted sampling: `w_i ∝ exp(−(d_i − T)² / (2σ²))`
- `baseline` — nearest-neighbour: select B examples with smallest `|d_i − T|`

**LoRA config:** `r=8`, `lora_alpha=16`, targets `q_proj k_proj v_proj o_proj`. The 7B model in bfloat16 (~14 GB) + LoRA fits within 48 GB (A6000). If OOM, override: `training.batch_size=1 training.gradient_accumulation_steps=8`.

Checkpoints saved to `checkpoints/step_N/` every `save_steps` steps and `checkpoints/final/` at completion.

Key hyperparameters (all in `conf/training/grpo.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `num_train_steps` | 5000 | Total optimizer steps |
| `batch_size` | 2 | Examples per micro-step |
| `group_size` | 4 | Rollouts per example |
| `gradient_accumulation_steps` | 4 | Micro-steps per optimizer step |
| `learning_rate` | 1e-5 | AdamW LR |
| `T_init` | 0.5 | Initial curriculum difficulty |
| `eta` | 0.05 | Curriculum step size |
| `sigma` | 0.15 | Gaussian bandwidth |
| `kl_coeff` | 0.0 | KL penalty (0 = disabled) |

## Evaluation (Phase 4)

Zero-shot batched evaluation on two benchmarks:

| Benchmark | HF ID | Split | Config | Task |
|---|---|---|---|---|
| MathVista | `AI4Math/MathVista` | `testmini` | — | MC + numeric |
| MMMU | `MMMU/MMMU` | `validation` | `Math` | MC |

Results are written to `results/eval_results.json` and logged to MLflow.

```json
{
  "model_checkpoint": "checkpoints/final",
  "base_model": "Qwen/Qwen2-VL-7B-Instruct",
  "evaluation_date": "2025-...",
  "benchmarks": {
    "MathVista": {"accuracy": 0.0, "correct": 0, "total": 1000, "skipped": 0},
    "MMMU":      {"accuracy": 0.0, "correct": 0, "total": 30,   "skipped": 0}
  },
  "macro_average": 0.0
}
```

## MLflow Experiment Tracking

Training and evaluation both log to `mlruns/` (gitignored).

**Start the MLflow server on the remote GPU machine:**

```bash
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

**On your local machine, open an SSH tunnel:**

```bash
ssh -L 5000:127.0.0.1:5000 user@remote-server-ip
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

**Metrics logged during training** (every `log_steps` steps):

| Metric | Description |
|---|---|
| `train/loss` | GRPO loss |
| `train/reward` | Mean reward across the batch |
| `curriculum/T` | Current curriculum target difficulty |
| `train/learning_rate` | Scheduler LR |

**Metrics logged during evaluation:**

| Metric | Description |
|---|---|
| `eval/mathvista_accuracy` | MathVista accuracy |
| `eval/mmmu_accuracy` | MMMU accuracy |
| `eval/macro_average` | Macro-average across both |

## Data Schema

All JSONL files follow the same base schema:

```json
{"id": "geometry3k_train_0", "image": "<base64 PNG or null>", "prompt": "Find x.", "answer": "3"}
```

After Phase 2, scored files add:

```json
{"difficulty": 0.3, "difficulty_raw_response": "3", "difficulty_parse_error": false}
```

## Project Structure

```
guac/
├── conf/                          # Hydra configs
│   ├── config.yaml
│   ├── data/default.yaml
│   ├── model/qwen2_vl_7b.yaml
│   ├── training/grpo.yaml
│   ├── judge/vllm.yaml
│   └── evaluation/default.yaml
├── src/guac/                      # Importable Python package
│   ├── data/                      # Dataset loaders, JSONL I/O, image utils
│   ├── judge/                     # vLLM-based difficulty scorer
│   ├── training/                  # Curriculum, rewards, GRPO trainer
│   └── evaluation/                # MathVista + MMMU evaluators
├── scripts/                       # Hydra entrypoints
│   ├── prepare_data.py
│   ├── judge_difficulty.py
│   ├── train.py
│   └── evaluate.py
├── run/                           # Shell wrappers
│   ├── 01_prepare_data.sh
│   ├── 02_judge_difficulty.sh
│   ├── 03_train_gaussian.sh
│   ├── 03_train_baseline.sh
│   └── 04_evaluate.sh
├── data/                          # gitignored
│   ├── raw/                       # Per-dataset per-split JSONL
│   ├── processed/                 # Cross-dataset merged splits
│   └── scored/                    # Difficulty-annotated splits
├── checkpoints/                   # gitignored — LoRA adapters
├── mlruns/                        # gitignored — MLflow tracking store
├── results/                       # gitignored — eval JSON outputs
└── pyproject.toml
```

## Linting

```bash
.venv/bin/ruff check src/ scripts/
.venv/bin/ruff check src/ scripts/ --fix
```
