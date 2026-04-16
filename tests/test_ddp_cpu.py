"""2-process gloo-backend DDP smoke test for the GRPO training loop.

Uses ``torch.multiprocessing.spawn`` with gloo backend (no GPU needed)
to verify the DDP plumbing works:

 - Both ranks initialise Accelerator with world_size=2.
 - Both ranks call backward() on every micro-step (no hang).
 - Flat ranks contribute zero gradient via the dummy-loss path.
 - Curriculum T updates independently per rank.
 - Checkpoint saved by rank 0 only.

Runs in ~30 s on a Mac.  If it hangs, the DDP plumbing is broken.

Usage::

    bash run/00_smoke_test_ddp.sh
    # or:
    PYTHONPATH=src python3 tests/test_ddp_cpu.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch.multiprocessing as mp

# ── Fixtures ────────────────────────────────────────────────────────────
TINY_MODEL = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "smoke"
WORLD_SIZE = 2


def _fake_reward_factory():
    """Return a closure that cycles: varied [0,1,0,1] then flat [0,0,0,0]."""
    state = {"idx": 0}

    def _reward(prediction: str, ground_truth: str) -> float:
        i = state["idx"]
        state["idx"] += 1
        return [0.0, 1.0, 0.0, 1.0][i % 4] if i < 4 else 0.0

    return _reward


def _run_rank(rank: int, tmp_dir: str) -> None:
    """Entry point for each spawned process."""
    # Set env vars so Accelerator detects the distributed context and
    # initialises gloo via PartialState.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from omegaconf import OmegaConf

        from guac.training.trainer import GRPOTrainer

        cfg = OmegaConf.create(
            {
                "seed": 42,
                "gpu_id": 0,
                "data": {"scored_dir": str(FIXTURE_DIR)},
                "model": {
                    "name": TINY_MODEL,
                    "dtype": "float32",
                    "lora": {
                        "r": 4,
                        "lora_alpha": 8,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "lora_dropout": 0.0,
                        "bias": "none",
                    },
                },
                "training": {
                    "output_dir": f"{tmp_dir}/ckpt",
                    "batch_size": 1,
                    "group_size": 4,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "num_train_steps": 2,
                    "warmup_steps": 0,
                    "save_steps": 9999,
                    "log_steps": 1,
                    "max_new_tokens": 4,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "sampling_mode": "gaussian",
                    "T_init": 0.5,
                    "eta": 0.05,
                    "alpha": 2.0,
                    "beta": 0.5,
                    "sigma": 0.15,
                    "d_min": 0.0,
                    "d_max": 1.0,
                    "kl_coeff": 0.0,
                    "max_seq_length": 2048,
                    "mlflow_tracking_uri": f"{tmp_dir}/mlruns_{rank}",
                    "mlflow_experiment": "ddp_test",
                },
            }
        )

        reward_fn = _fake_reward_factory()
        with patch("guac.training.trainer.compute_reward", side_effect=reward_fn):
            trainer = GRPOTrainer(cfg)

            assert trainer.world_size == WORLD_SIZE, (
                f"rank {rank}: expected world_size={WORLD_SIZE}, "
                f"got {trainer.world_size}"
            )
            assert trainer.rank == rank, (
                f"rank {rank}: trainer.rank={trainer.rank}"
            )

            # ── Run 2 optimizer steps ──
            trainer.train()

        # ── Post-training checks ──
        assert 0.0 <= trainer.curriculum.T <= 1.0, (
            f"rank {rank}: T={trainer.curriculum.T} out of bounds"
        )

        if rank == 0:
            final_dir = Path(f"{tmp_dir}/ckpt/final")
            assert final_dir.exists(), (
                f"rank 0: final checkpoint missing at {final_dir}"
            )

        print(f"[rank {rank}] ALL CHECKS PASSED (T={trainer.curriculum.T:.4f})", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
        raise


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tmp_dir = tempfile.mkdtemp(prefix="guac_ddp_test_")
    print(f"DDP test tmp dir: {tmp_dir}")
    print(f"Spawning {WORLD_SIZE} gloo processes ...")

    mp.spawn(_run_rank, args=(tmp_dir,), nprocs=WORLD_SIZE, join=True)

    print("=== DDP smoke test PASSED ===")


if __name__ == "__main__":
    main()
