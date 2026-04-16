# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All work uses a `uv` virtual environment at `.venv/`. Activate before running any command:

```bash
source .venv/bin/activate
```

Install the package and all deps:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**GPU constraint** (single-GPU launch): scripts pin to GPU 0 by default. The `gpu_id` value in `conf/config.yaml` is set at the top of each script via `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`.

**Multi-GPU launch**: `bash run/03_train_reinforce_ddp.sh` wraps `accelerate launch --config_file=accelerate_config.yaml scripts/train.py ...`. In that path, accelerate sets `LOCAL_RANK` and manages device assignment itself; `scripts/train.py` only applies the `cfg.gpu_id` override when `LOCAL_RANK` is absent.

**Data location**: The scored dataset lives at `/data/troy/datasets/guac/scored/`. This is hardcoded in `conf/data/default.yaml`. Phase 1/2 outputs also point there.

Lint before committing:

```bash
.venv/bin/ruff check src/ scripts/
.venv/bin/ruff check src/ scripts/ --fix
```

## Running the Pipeline

```bash
bash run/01_prepare_data.sh            # loads all splits → data/raw/ + data/processed/
bash run/02_judge_difficulty.sh        # scores all splits → data/scored/
bash run/03_train_reinforce.sh         # single-GPU REINFORCE training → checkpoints/
bash run/03_train_reinforce_ddp.sh     # 8× DDP via accelerate → checkpoints/
bash run/04_evaluate.sh                # eval → results/eval_results.json
```

Scripts accept Hydra overrides appended directly:

```bash
python scripts/train.py training.sampling_mode=baseline training.num_train_steps=2000
python scripts/evaluate.py checkpoint_path=checkpoints/step_500 gpu_id=0
```

## Architecture

The project is a 4-phase VLM RL pipeline: data prep → difficulty judging → REINFORCE training → evaluation.

**Data flow**: `data/processed/{train,val,test}.jsonl` → difficulty judging → `data/scored/{train,val,test}.jsonl` → REINFORCE training → `checkpoints/` → evaluation → `results/eval_results.json`.

**Record schema**: Every JSONL record carries `id` (str), `image` (base64 PNG string or null), `prompt` (str), `answer` (str). Scored records additionally carry `difficulty` (float in [0.0, 1.0], AoPS score ÷ 10), `difficulty_raw_response`, `difficulty_parse_error`.

**Package layout** (`src/guac/`):
- `data/utils.py` — image encode/decode helpers (base64 PNG) shared by all phases
- `data/prep.py` — `prepare_all(cfg)` loads geometry3k + ScienceQA for all splits; `load_jsonl`/`save_jsonl` used everywhere
- `judge/difficulty.py` — `DifficultyJudge` wraps vLLM; `score_split(input, output)` adds difficulty fields; resumes from `.ckpt` file on crash
- `training/curriculum.py` — `CurriculumState` dataclass (T update rule) + `CurriculumSampler` (baseline/gaussian modes)
- `training/rewards.py` — reward computation (exact match, SymPy equivalence, substring), log-prob helpers
- `training/reinforce_trainer.py` — `ReinforceTrainer` uses REINFORCE + EMA baseline; generates G=1 response per prompt; KL via `disable_adapter()`; logs to MLflow
- `training/trainer.py` — **[DEPRECATED]** old `GRPOTrainer`, kept for reference
- `evaluation/evaluator.py` — `run_evaluation(cfg, checkpoint_path)` evaluates on MathVista and MMMU; logs to MLflow

**Training algorithm (REINFORCE + EMA baseline)**:
1. Sample batch of B prompts via Gaussian curriculum (Equation 2)
2. Generate **one** response per prompt (G=1)
3. Score with exact-match reward → R_i
4. Policy log-probs via teacher forcing; reference log-probs via `model.disable_adapter()` (zero extra VRAM)
5. KL penalty: KL_i = policy_logprob - ref_logprob
6. Adjusted reward: R_hat_i = R_i - λ·KL_i
7. Advantage: A_i = R_hat_i - b (EMA baseline)
8. EMA update: b ← α·b + (1-α)·R_avg
9. Loss: L = -mean(A.detach() · policy_logprobs)

**Curriculum update rule**: `T_new = clip(T + η·tanh(α·(R_avg - β)), d_min, d_max)` — called once per optimizer step.

**Training memory**: LoRA (r=8, targets q/k/v/o_proj) keeps the 7B model trainable within 48 GB VRAM (A6000). KL reference uses `disable_adapter()` — no second model loaded.

## Config System

All hyperparameters live in `conf/`. Root `conf/config.yaml` composes sub-configs via Hydra's defaults list. Key sub-configs:
- `conf/training/reinforce.yaml` — steps (5000), batch size (8), EMA decay (0.95), KL coeff (0.1), sampling mode, MLflow experiment name
- `conf/model/qwen2_vl_7b.yaml` — LoRA settings (r=8, lora_alpha=16)
- `conf/judge/vllm.yaml` — vLLM batch size (32), checkpoint interval (500 rows)

**Legacy**: `conf/training/grpo.yaml` is preserved but no longer the default. It can be activated with `python scripts/train.py training=grpo` but the GRPOTrainer will need to be re-imported in `scripts/train.py`.

## MLflow

Training and evaluation log to `mlruns/` (gitignored).

```bash
# Start server on remote GPU machine:
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# SSH tunnel on local machine:
ssh -L 5000:127.0.0.1:5000 user@remote-server-ip
# Open: http://localhost:5000
```

## Dataset Notes

- **Geometry3K**: HF ID `hiyouga/geometry3k`. Fields: `images` (list of PIL), `problem` (contains `<image>` placeholder tokens that are stripped), `answer` (already open-ended).
- **ScienceQA**: HF ID `derek-thomas/ScienceQA`. Filter rows where `image is None` (text-only). `answer` is an integer index into `choices`.
- **MathVista**: Image is in the `decoded_image` field (PIL), not `image` (string path).
- **MMMU**: Requires a subject config (e.g., `Math`); no `all` config exists. The `options` field is a Python-literal string — parse with `ast.literal_eval`.
