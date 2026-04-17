# GUAC: Gaussian Utilizing Adaptive Curriculums for Vision Language Models

**Team:** Jasper Tan, Troy Dutton, Varun Arumugam | *The University of Texas at Austin*
📄 **[Read the Full Paper / Milestone Report Here](./Deep_RL_Project_Milestone.pdf)**

**TL;DR:** Standard reinforcement learning often trains VLMs on randomly sampled batches, ignoring the inherent difficulty of individual problems. GUAC is an end-to-end pipeline that fine-tunes Qwen2-VL-7B-Instruct using **REINFORCE with an EMA baseline**, paired with a novel **Gaussian sampling schedule** that dynamically paces problem difficulty during training.

> **Why REINFORCE?** Our initial implementation used GRPO, which requires generating multiple rollouts per prompt (group_size=G) to compute within-group advantages. With binary rewards and small G (constrained by VRAM), every group collapsed to all-0 or all-1 rewards — advantages zeroed out, producing no gradient signal across 100% of training steps. REINFORCE computes advantages against a running EMA baseline, completely bypassing this flat-group problem.

---

```
Phase 1 — Data Prep         Phase 2 — Difficulty Judging
Geometry3K + ScienceQA  →   vLLM scores each problem
data/raw/ + processed/      on AoPS 1–10 scale
                                      ↓
Phase 4 — Evaluation    ←   Phase 3 — REINFORCE Training
MathVista + MMMU            LoRA + dynamic curriculum
results/eval_results.json   checkpoints/
```

## Requirements

| Use case | Hardware | Deps |
|---|---|---|
| **Full training + evaluation** | 1 NVIDIA GPU with ≥ 48 GB VRAM (A6000 48 GB or H100 80 GB), CUDA 12.x | Python 3.11+, [uv](https://docs.astral.sh/uv/), `vllm` (judge phase) |
| **Distributed training** (optional) | 8× NVIDIA A6000 48 GB on one node — DDP via `accelerate launch` | Same as above; `accelerate` is a dep |
| **Local smoke tests** (Mac / CI / any box w/o a GPU) | CPU (≥ 8 GB RAM), any platform | Python 3.11+, `torch`, `transformers`, `peft`, `mlflow`, `omegaconf`, `sympy`, `torchvision`, `pytest` |

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
bash run/00_smoke_test.sh          # ~10 s on a Mac
```

## Running the Pipeline (GPU)

Each phase has a shell wrapper in `run/` and a Hydra script in `scripts/`. The shell wrappers activate the venv for you.

```bash
bash run/00_smoke_test.sh             # optional: quick sanity check
bash run/01_prepare_data.sh           # ~5–10 min: download + write data/raw/ + data/processed/
bash run/02_judge_difficulty.sh       # ~2 hrs:   score all splits → data/scored/
bash run/03_train_reinforce.sh        # single GPU:  5000 steps → checkpoints/
bash run/03_train_reinforce_ddp.sh    # 8 GPUs:      distributed via accelerate
bash run/04_evaluate.sh               # ~30 min:  MathVista + MMMU → results/
```

### Multi-GPU training (8× A6000)

The trainer supports data-parallel DDP via HuggingFace `accelerate`. The `accelerate_config.yaml` at the repo root declares 8-process single-node DDP with `mixed_precision: 'no'`.

```bash
bash run/03_train_reinforce_ddp.sh
```

**Effective batch per optimizer step**: with `gradient_accumulation_steps=1` override, `batch_size=8 × world_size=8 × grad_accum=1 = 64 forward passes per step`.

**Under-the-hood DDP behaviour**:
- `CurriculumSampler` takes `rank` and `world_size` at init. Gaussian mode uses per-rank seeded `torch.Generator` for independent sampling.
- Each rank computes R_avg from its local batch and updates curriculum T independently. T drift across ranks is bounded by `eta=0.05` per step.
- Rank 0 exclusively handles MLflow logging and checkpoint saves.
- The EMA baseline is updated independently per rank — this is fine because B=8 provides sufficient variance smoothing.
- No user-initiated collectives in the hot path, avoiding the NCCL sequence-number drift that caused DDP deadlocks in the GRPO implementation.

Pass Hydra overrides as extra arguments:

```bash
bash run/03_train_reinforce.sh training.num_train_steps=2000
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
├── config.yaml                # gpu_id, seed, defaults list
├── data/default.yaml          # dataset HF IDs, splits, directories
├── model/qwen2_vl_7b.yaml    # model name, dtype, LoRA settings
├── training/reinforce.yaml    # REINFORCE: steps, LR, EMA, KL, curriculum
├── training/grpo.yaml         # [DEPRECATED] old GRPO config
├── judge/vllm.yaml            # vLLM engine settings
└── evaluation/default.yaml    # benchmark configs, output path
```

Override any value at runtime:

```bash
python scripts/train.py training.learning_rate=2e-5 training.ema_decay=0.9
python scripts/train.py training.kl_coeff=0.05 training.batch_size=16
python scripts/judge_difficulty.py judge.batch_size=16
```

## Datasets

| Dataset | HF ID | Splits | Records (vision-only) |
|---|---|---|---|
| Geometry3K | `hiyouga/geometry3k` | train / val / test | 2101 / 300 / 601 |
| ScienceQA | `derek-thomas/ScienceQA` | train / val / test | ~6200 / ~2000 / ~2000 |
| **Combined** | — | train / val / test | ~8300 / ~2300 / ~2600 |

Data is stored at `/data/troy/datasets/guac/` (hardcoded in `conf/data/default.yaml`):
- `raw/{dataset}_{split}.jsonl` — per-dataset per-split
- `processed/{train,val,test}.jsonl` — merged across datasets
- `scored/{train,val,test}.jsonl` — with difficulty annotations

## Difficulty Judging (Phase 2)

Uses `Qwen/Qwen2-VL-7B-Instruct` via vLLM to score each problem on the [Art of Problem Solving](https://artofproblemsolving.com) 1–10 scale. Scores are normalised to `[0.0, 1.0]` (AoPS score ÷ 10).

- Runs at ~90 rows/sec on H100 (~60 rows/sec on A6000) with `batch_size=32`
- Checkpoints every 500 rows; re-running after a crash resumes automatically
- Output: `data/scored/{train,val,test}.jsonl` — adds `difficulty`, `difficulty_raw_response`, `difficulty_parse_error`

## Training (Phase 3) — REINFORCE with EMA Baseline

Custom REINFORCE loop with LoRA, a dynamic difficulty curriculum, and KL-penalised rewards.

### Why REINFORCE over GRPO

GRPO requires generating G rollouts per prompt to compute within-group advantages. With binary rewards (0 or 1) and small G (limited by VRAM), the probability that all G rollouts earn the same reward is very high:
- Easy problems: all correct → `[1,1,1,1]` → advantages = `[0,0,0,0]` → zero gradient
- Hard problems: all wrong → `[0,0,0,0]` → advantages = `[0,0,0,0]` → zero gradient

In our 8×A6000 DDP run with `group_size=4`, **100% of groups were flat across 180 optimizer steps**. The loss was exactly `0.0000` on every single step. The model never learned.

REINFORCE with an EMA baseline computes advantages against a *running historical average* `b`, not the current batch. Even if every response in a batch gets reward 1.0, the advantage is `1.0 - b > 0` (as long as `b` hasn't converged to 1.0 yet), providing a clear gradient signal.

### Algorithm

Per optimizer step:

1. **Sample batch**: B prompts via Gaussian curriculum (Equation 2: `w_i ∝ exp(−(d_i − T)² / 2σ²)`)
2. **Generate**: One response per prompt (G=1) — 8× less VRAM than GRPO with G=4
3. **Reward**: Binary exact-match score R_i ∈ {0, 1}
4. **Policy log-probs**: Teacher-forcing forward pass with LoRA adapters active
5. **Reference log-probs**: Same forward pass with `model.disable_adapter()` — zero extra VRAM
6. **KL penalty**: `KL_i = policy_logprob_i - ref_logprob_i`
7. **Adjusted reward**: `R̂_i = R_i - λ · KL_i`
8. **Advantage**: `A_i = R̂_i - b` (where `b` is the EMA baseline)
9. **EMA update**: `b ← α · b + (1 − α) · R_avg`
10. **Loss**: `L = −mean(A.detach() · policy_logprobs)`

**Curriculum update rule** (per optimizer step):
```
T_new = clip(T + η · tanh(α · (R_avg − β)), d_min, d_max)
```

**Sampling modes** (set via `training.sampling_mode`):
- `gaussian` *(default)* — weighted sampling: `w_i ∝ exp(−(d_i − T)² / (2σ²))`
- `baseline` — nearest-neighbour: select B examples with smallest `|d_i − T|`

**LoRA config:** `r=8`, `lora_alpha=16`, targets `q_proj k_proj v_proj o_proj`. The 7B model in bfloat16 (~14 GB) + LoRA fits within 48 GB (A6000). KL reference via `disable_adapter()` costs zero extra VRAM.

Checkpoints saved to `checkpoints/step_N/` every `save_steps` steps and `checkpoints/final/` at completion.

### Key Hyperparameters

All in `conf/training/reinforce.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `num_train_steps` | 5000 | Total optimizer steps |
| `batch_size` | 8 | Prompts per micro-step (G=1, so 8 forward passes) |
| `gradient_accumulation_steps` | 2 | Micro-steps per optimizer step |
| `learning_rate` | 1e-5 | AdamW LR |
| `ema_decay` | 0.95 | EMA baseline decay factor (α) |
| `kl_coeff` | 0.1 | KL penalty coefficient (λ) |
| `T_init` | 0.1 | Initial curriculum difficulty (= `d_min`, cold start on easiest problems) |
| `d_min` / `d_max` | 0.1 / 0.7 | Clipping bounds for T (matches dataset range) |
| `eta` | 0.05 | Curriculum step size |
| `sigma` | 0.15 | Gaussian bandwidth |
| `temperature` | 1.0 | Generation temperature |

**Effective batch size per optimizer step:** `batch_size × gradient_accumulation_steps` = 8 × 2 = 16 forward passes.

## Evaluation (Phase 4)

Zero-shot batched evaluation on two benchmarks:

| Benchmark | HF ID | Split | Config | Task |
|---|---|---|---|---|
| MathVista | `AI4Math/MathVista` | `testmini` | — | MC + numeric |
| MMMU | `MMMU/MMMU` | `validation` | `Math` | MC |

Results are written to `results/eval_results.json` and logged to MLflow.

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
| `train/loss` | REINFORCE policy gradient loss |
| `train/reward` | Mean raw reward across the batch |
| `train/learning_rate` | Scheduler LR |
| `train/kl_div` | Mean KL divergence between policy and reference |
| `train/ema_baseline` | Current EMA baseline value |
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
│   ├── training/reinforce.yaml    # ← active training config
│   ├── training/grpo.yaml         # [DEPRECATED]
│   ├── judge/vllm.yaml
│   └── evaluation/default.yaml
├── src/guac/                      # Importable Python package
│   ├── data/                      # Dataset loaders, JSONL I/O, image utils
│   ├── judge/                     # vLLM-based difficulty scorer
│   ├── training/                  # Curriculum, rewards, REINFORCE trainer
│   │   ├── curriculum.py          # CurriculumState + CurriculumSampler
│   │   ├── rewards.py             # compute_reward, log-prob helpers
│   │   ├── reinforce_trainer.py   # ← active trainer
│   │   └── trainer.py             # [DEPRECATED] old GRPO trainer
│   └── evaluation/                # MathVista + MMMU evaluators
├── scripts/                       # Hydra entrypoints
│   ├── prepare_data.py
│   ├── judge_difficulty.py
│   ├── train.py                   # Uses ReinforceTrainer
│   └── evaluate.py
├── run/                           # Shell wrappers
│   ├── 00_smoke_test.sh
│   ├── 01_prepare_data.sh
│   ├── 02_judge_difficulty.sh
│   ├── 03_train_reinforce.sh      # ← single-GPU training
│   ├── 03_train_reinforce_ddp.sh  # ← 8× DDP training
│   ├── 03_train_gaussian.sh       # [DEPRECATED]
│   ├── 03_train_baseline.sh       # [DEPRECATED]
│   └── 04_evaluate.sh
├── tests/                         # CPU-runnable smoke tests
│   ├── conftest.py
│   ├── fixtures/smoke/train.jsonl
│   └── test_training_smoke.py
├── data/                          # symlink or gitignored
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

## Troubleshooting

**Training loss stuck at 0.** If you see `train/loss = 0.0` on every step, you may have accidentally switched back to the GRPO trainer. Verify `conf/config.yaml` has `training: reinforce`, and `scripts/train.py` imports `ReinforceTrainer`. With REINFORCE + EMA baseline, the loss should be non-zero from step 1.

**`ModuleNotFoundError: No module named 'guac.data'`.** The package lives under `src/guac/data/`.  Either `pip install -e .` (inside the venv) or export `PYTHONPATH=src` when running scripts directly.

**`torch.cuda.OutOfMemoryError` during training.** Drop `training.batch_size` or `training.max_new_tokens` via Hydra override. With G=1, the REINFORCE trainer is significantly more memory-efficient than the old GRPO trainer. The A6000 48 GB should comfortably fit `batch_size=8 max_new_tokens=128` with LoRA `r=8`.

**KL divergence explodes.** If `train/kl_div` climbs rapidly, increase `kl_coeff` to penalise drift more aggressively, or decrease `learning_rate`.

**EMA baseline converges too slowly.** If `train/ema_baseline` barely moves and advantages are noisy, try `ema_decay=0.9` (faster tracking) instead of the default `0.95`.
