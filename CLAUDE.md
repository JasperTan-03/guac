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

**GPU constraint**: All scripts pin to GPU 7 by default. The `gpu_id` value in `conf/config.yaml` is set at the top of each script via `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`.

Lint before committing:

```bash
.venv/bin/ruff check src/ scripts/
.venv/bin/ruff check src/ scripts/ --fix
```

## Running the Pipeline

```bash
bash run/01_prepare_data.sh           # loads all splits → data/raw/ + data/processed/
bash run/02_judge_difficulty.sh       # scores all splits → data/scored/
bash run/03_train_gaussian.sh         # GRPO training → checkpoints/
bash run/04_evaluate.sh               # eval → results/eval_results.json
```

Scripts accept Hydra overrides appended directly:

```bash
python scripts/train.py training.sampling_mode=baseline training.num_train_steps=2000
python scripts/evaluate.py checkpoint_path=checkpoints/step_500 gpu_id=0
```

## Architecture

The project is a 4-phase VLM RL pipeline: data prep → difficulty judging → GRPO training → evaluation.

**Data flow**: `data/processed/{train,val,test}.jsonl` → difficulty judging → `data/scored/{train,val,test}.jsonl` → GRPO training → `checkpoints/` → evaluation → `results/eval_results.json`.

**Record schema**: Every JSONL record carries `id` (str), `image` (base64 PNG string or null), `prompt` (str), `answer` (str). Scored records additionally carry `difficulty` (float in [0.0, 1.0], AoPS score ÷ 10), `difficulty_raw_response`, `difficulty_parse_error`.

**Package layout** (`src/guac/`):
- `data/utils.py` — image encode/decode helpers (base64 PNG) shared by all phases
- `data/prep.py` — `prepare_all(cfg)` loads geometry3k + ScienceQA for all splits; `load_jsonl`/`save_jsonl` used everywhere
- `judge/difficulty.py` — `DifficultyJudge` wraps vLLM; `score_split(input, output)` adds difficulty fields; resumes from `.ckpt` file on crash
- `training/curriculum.py` — `CurriculumState` dataclass (T update rule) + `CurriculumSampler` (baseline/gaussian modes)
- `training/rewards.py` — reward computation, group normalization, log-prob calculation, GRPO loss
- `training/trainer.py` — `GRPOTrainer` uses all of the above; logs to MLflow at each `log_steps`
- `evaluation/evaluator.py` — `run_evaluation(cfg, checkpoint_path)` evaluates on MathVista and MMMU; logs to MLflow

**Curriculum update rule**: `T_new = clip(T + η·tanh(α·(R_avg - β)), d_min, d_max)` — called once per optimizer step.

**Training memory**: LoRA (r=8, targets q/k/v/o_proj) keeps the 7B model trainable within 80 GB VRAM.

## Config System

All hyperparameters live in `conf/`. Root `conf/config.yaml` composes sub-configs via Hydra's defaults list. Key sub-configs:
- `conf/training/grpo.yaml` — steps (5000), batch size (2), group size (4), sampling mode, MLflow experiment name
- `conf/model/qwen2_vl_7b.yaml` — LoRA settings (r=8, lora_alpha=16)
- `conf/judge/vllm.yaml` — vLLM batch size (32), checkpoint interval (500 rows)

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
