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

| Use case | Hardware | Deps |
|---|---|---|
| **Full training + evaluation** | 1 NVIDIA GPU with ≥ 48 GB VRAM (A6000 48 GB or H100 80 GB), CUDA 12.x | Python 3.11+, [uv](https://docs.astral.sh/uv/), `vllm` (judge phase) |
| **Distributed training** (optional) | 8× NVIDIA A6000 48 GB on one node — DDP via `accelerate launch` | Same as above; `accelerate` is a dep |
| **Local smoke tests** (Mac / CI / any box w/o a GPU) | CPU (≥ 8 GB RAM), any platform | Python 3.11+, `torch`, `transformers`, `peft`, `mlflow`, `omegaconf`, `sympy`, `torchvision`, `pytest` |

The full pipeline is GPU-only in practice (vLLM for Phase 2, 7 B model fine-tuning for Phase 3).  The smoke test exercises the GRPO training *control flow* on a 5 M-param random-weights Qwen2-VL checkpoint so you can validate reward/loss math, flat-group detection, and gradient flow without waiting for a GPU box.

## Setup

### GPU box (full pipeline)

```bash
git clone <repo>
cd guac

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

> **Note:** `vllm` pulls in `torch` automatically. If you need a specific CUDA build, install torch first.

### Local machine (smoke tests only)

No venv required — the committed `.venv/` is a Linux/uv venv intended for the GPU box and will not work on macOS.  Install the runtime deps into your system Python (or a local venv) once:

```bash
python3 -m pip install torch transformers peft mlflow omegaconf sympy torchvision pytest
```

Then run the smoke suite:

```bash
bash run/00_smoke_test.sh          # 24 tests, ~10 s on a Mac
```

This forces `CUDA_VISIBLE_DEVICES=""` so it's reproducible on any machine.

## Running the Pipeline (GPU)

Each phase has a shell wrapper in `run/` and a Hydra script in `scripts/`. The shell wrappers activate the venv for you.

```bash
bash run/00_smoke_test.sh         # optional: 10 s sanity check before burning GPU time
bash run/01_prepare_data.sh       # ~5–10 min: download + write data/raw/ + data/processed/
bash run/02_judge_difficulty.sh   # ~2 hrs:   score all splits → data/scored/
bash run/03_train_gaussian.sh     # single GPU:  5000 steps → checkpoints/
bash run/03_train_gaussian_ddp.sh # 8 GPUs:      distributed via accelerate (see below)
bash run/04_evaluate.sh           # ~30 min:  MathVista + MMMU → results/
```

### Multi-GPU training (8× A6000)

The trainer supports data-parallel DDP via HuggingFace `accelerate`. The committed `accelerate_config.yaml` at the repo root declares 8-process single-node DDP with `mixed_precision: 'no'` (the trainer already manages bf16 weights + manual autocast — letting accelerate layer another bf16 autocast would double-cast).

```bash
bash run/03_train_gaussian_ddp.sh     # wraps: accelerate launch --config_file=accelerate_config.yaml scripts/train.py ...
bash run/03_train_baseline_ddp.sh     # same, sampling_mode=baseline
```

**Effective batch per optimizer step**: with the `gradient_accumulation_steps=1` override baked into the DDP wrapper,
`batch_size=2 * group_size=4 * world_size=8 * grad_accum=1 = 64 rollouts per step` — same order of magnitude as the single-GPU default (32), so the curriculum update rule keeps similar granularity. Drop the override if you want a 4× larger batch at the cost of coarser curriculum updates.

**Under-the-hood correctness**:
- `CurriculumSampler` takes `rank` and `world_size` at init. Gaussian mode uses a per-rank `torch.Generator` (seeded via Knuth's multiplicative hash so ranks don't share bit patterns) so the 8 ranks draw disjoint stochastic batches. Baseline mode picks the top-`B·W` globally and slices the rank's share.
- Every optimizer step all-reduces `R_avg` before `curriculum.update(...)`, so `T` stays bit-identical across ranks and the 8 Gaussians sample from the same distribution family.
- **Flat-group DDP deadlock guard**: if all rollouts on a rank earn the same reward, normalized advantages collapse to zero and that rank has no gradient signal. A `sum`-reduce of a boolean flag lets every rank make the same decision: if any rank has signal, flat ranks synthesize a differentiable-zero loss that touches every trainable parameter so DDP's gradient allreduce fires uniformly (zero contribution). If no rank has signal, all ranks skip backward together.
- Rank 0 exclusively handles MLflow logging, checkpoint saves (`accelerator.unwrap_model(self.model).save_pretrained(...)`), and `print_trainable_parameters`. Other ranks leave MLflow alone.
- Before each checkpoint save, `accelerator.wait_for_everyone()` acts as a barrier so saved weights correspond to a completed optimizer step.

**Single-GPU vs DDP launch differences**: `scripts/train.py` sets `CUDA_VISIBLE_DEVICES=cfg.gpu_id` only when `LOCAL_RANK` is not in the environment. Under `accelerate launch`, accelerate manages device assignment itself; under plain `python scripts/train.py`, the old single-GPU pinning behavior is preserved.

Expected throughput gain: ~6–7× single-GPU (not a full 8× because of DDP allreduce latency + the sync barriers for `R_avg` and running metrics). If it's less than 4×, check `nvidia-smi` for GPU imbalance and look at the per-step logs for ranks diverging on the global-signal collective.

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
| `T_init` | 0.3 | Initial curriculum difficulty (= `d_min` so the cold start draws from the easiest bucket — see [AdaRFT Fig. 4](https://arxiv.org/abs/2504.05520)) |
| `d_min` / `d_max` | 0.3 / 1.0 | Clipping bounds for `T` |
| `eta` | 0.05 | Curriculum step size |
| `sigma` | 0.15 | Gaussian bandwidth |
| `kl_coeff` | 0.0 | KL penalty (0 = disabled) |

### Flat-group diagnostic

With `group_size=4` and a cold policy, every rollout in a group can earn the same binary reward (all 0 or all 1).  Normalised advantages then collapse to `[0,0,0,0]` and the GRPO loss term is exactly zero — no gradient.  The trainer now **skips** these flat groups and surfaces three MLflow metrics every `log_steps`:

- `train/flat_groups_frac` — fraction of groups that collapsed this interval
- `train/informative_groups` — groups that actually contributed to the loss
- `train/flat_groups` — groups that were skipped

**Healthy run:** `flat_groups_frac < 0.3` within ~50 steps; `train/reward` climbs toward `β = 0.5`.
**Stuck run:** `flat_groups_frac > 0.7` — the curriculum isn't in the productive-struggle zone.  Try widening `sigma` (0.15 → 0.25) or lowering `d_min`.

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
| `train/loss` | GRPO loss (averaged over informative groups in the window) |
| `train/reward` | Mean raw reward across the batch |
| `train/learning_rate` | Scheduler LR |
| `train/flat_groups_frac` | Fraction of groups skipped because every rollout had the same reward |
| `train/informative_groups` | Groups that contributed to the loss |
| `train/flat_groups` | Groups that were skipped |
| `curriculum/T` | Current curriculum target difficulty |

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
│   ├── 00_smoke_test.sh           # CPU pytest — no GPU or venv required
│   ├── 01_prepare_data.sh
│   ├── 02_judge_difficulty.sh
│   ├── 03_train_gaussian.sh
│   ├── 03_train_baseline.sh
│   └── 04_evaluate.sh
├── tests/                         # CPU-runnable smoke tests (24 tests, ~10 s)
│   ├── conftest.py
│   ├── fixtures/smoke/train.jsonl
│   └── test_training_smoke.py
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
.venv/bin/ruff check src/ scripts/ tests/
.venv/bin/ruff check src/ scripts/ tests/ --fix
```

On a local machine without the `.venv`:

```bash
python3 -m ruff check src/ scripts/ tests/
```

## Troubleshooting

**Training loss stuck at 0.** The binary reward collapses every group to a single value (all 0 or all 1), which zeroes the normalised advantages.  Watch `train/flat_groups_frac` in MLflow — a sustained value above 0.7 means the curriculum is out of the productive-struggle zone.  See the [Flat-group diagnostic](#flat-group-diagnostic) section above.  Run `bash run/00_smoke_test.sh` locally before burning GPU time; the suite catches reward-pipeline regressions in ~10 s.

**`ModuleNotFoundError: No module named 'guac.data'`.** The package lives under `src/guac/data/`.  Either `pip install -e .` (inside the venv) or export `PYTHONPATH=src` when running scripts directly.

**`torch.cuda.OutOfMemoryError` during training.** Drop `training.batch_size`, `training.group_size`, or `training.max_new_tokens` via Hydra override.  The A6000 48 GB fits `batch_size=2 group_size=4 max_new_tokens=128` with LoRA `r=8`.
