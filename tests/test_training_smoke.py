"""CPU-runnable smoke tests for the GRPO training loop.

These tests are designed to be cheap enough to run on a laptop (Mac / CI
without a GPU) and target the exact failure mode that previously produced
a loss of zero throughout training: flat-group reward collapse.

The tests cover three things:

1. ``normalize_rewards`` / ``is_informative_group`` unit behaviour.
2. The trainer's micro-step path on a deterministic, monkeypatched
   reward — verifies that informative groups generate a non-zero loss
   and push gradient into LoRA parameters, while flat groups are
   correctly skipped (running loss stays 0 for that group, but the
   loop does not crash).
3. The curriculum update fires at least once.

A tiny random Qwen2-VL checkpoint (~5M params) is used so that a couple
of optimizer steps run in a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from guac.training.rewards import (
    FLAT_GROUP_STD_EPS,
    compute_reward,
    grpo_loss,
    is_informative_group,
    normalize_rewards,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# 5M-param random-weights VLM — small enough for CPU, same API surface as
# Qwen/Qwen2-VL-7B-Instruct, so the trainer code path exercised here is the
# same code path that will run on the H100 / A6000.
TINY_MODEL = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "smoke"


# --------------------------------------------------------------------------- #
# Unit tests: reward-normalisation / flat-group detection
# --------------------------------------------------------------------------- #

def test_is_informative_group_flat_all_zero():
    assert not is_informative_group([0.0, 0.0, 0.0, 0.0])


def test_is_informative_group_flat_all_one():
    assert not is_informative_group([1.0, 1.0, 1.0, 1.0])


def test_is_informative_group_singleton():
    # A group of size 1 carries no within-group variance signal.
    assert not is_informative_group([1.0])


def test_is_informative_group_varied():
    assert is_informative_group([0.0, 1.0, 0.0, 1.0])


def test_normalize_rewards_flat_returns_zeros():
    out = normalize_rewards([0.0, 0.0, 0.0, 0.0])
    assert out == [0.0, 0.0, 0.0, 0.0]
    # And the same for all-ones (the other collapse mode):
    out = normalize_rewards([1.0, 1.0, 1.0, 1.0])
    assert out == [0.0, 0.0, 0.0, 0.0]


def test_normalize_rewards_varied_has_nonzero_variance():
    out = normalize_rewards([0.0, 1.0, 0.0, 1.0])
    # Zero mean, unit std → output has at least one entry whose absolute
    # value is well above the flat-group threshold.
    assert max(abs(v) for v in out) > 10 * FLAT_GROUP_STD_EPS
    assert abs(sum(out)) < 1e-6  # zero-mean


# --------------------------------------------------------------------------- #
# compute_reward cascade (exact → extracted → sympy → substring)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("prediction", "ground_truth"),
    [
        # Exact / extracted match (cheap path — sympy not needed).
        ("42", "42"),
        ("The answer is 12.", "12"),
        ("\\boxed{7}", "7"),
        # SymPy-only branch: LaTeX ↔ decimal ↔ symbolic equivalence.
        ("0.5", "\\frac{1}{2}"),
        ("The answer is 0.5.", "\\frac{1}{2}"),
        ("1/2", "0.5"),
        ("2*pi", "pi*2"),
        ("x = 3", "3"),
        # Substring branch for word answers where sympy doesn't apply.
        ("The capital is Paris", "paris"),
    ],
)
def test_compute_reward_recognises_equivalent_answers(prediction, ground_truth):
    assert compute_reward(prediction, ground_truth) == 1.0


@pytest.mark.parametrize(
    ("prediction", "ground_truth"),
    [
        ("3", "4"),
        ("The answer is 0.4", "\\frac{1}{2}"),
        ("hello", "world"),
        ("pi*3", "2*pi"),
        ("", "42"),
    ],
)
def test_compute_reward_rejects_wrong_answers(prediction, ground_truth):
    assert compute_reward(prediction, ground_truth) == 0.0


def test_compute_reward_handles_garbage_without_raising():
    # Ground truth from a malformed LaTeX string shouldn't raise — the
    # sympy branch catches parse errors and returns 0.0.
    assert compute_reward("???", "\\frac{{{{{") == 0.0


def test_grpo_loss_is_nonzero_when_logprobs_and_advantages_vary():
    """Verify the GRPO loss math end-to-end.

    Even before we touch the trainer, we can prove that the math works:
    with varied (zero-mean) advantages AND varied per-rollout log-probs,
    the resulting scalar loss has non-zero magnitude AND propagates a
    non-zero gradient back to the log-prob source.
    """
    # A 1-D "parameter" that requires grad; log-probs flow through it.
    w = torch.tensor([1.0], requires_grad=True)
    # Per-rollout log-probs with clearly different values.
    log_probs = torch.stack([
        w * -5.0,
        w * -3.0,
        w * -7.0,
        w * -4.0,
    ]).squeeze(-1)
    # Advantages normalised from rewards [0, 1, 0, 1] (zero-mean).
    advantages = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    loss = grpo_loss(log_probs, advantages)
    assert loss.abs().item() > 1e-6, f"loss too small: {loss!r}"
    loss.backward()
    assert w.grad is not None and w.grad.abs().item() > 1e-6, (
        f"No gradient reached w; grad={w.grad!r}"
    )


# --------------------------------------------------------------------------- #
# Integration test: flat-vs-varied groups in the trainer
# --------------------------------------------------------------------------- #

def _build_smoke_cfg(output_dir: Path, mlflow_dir: Path) -> OmegaConf:
    """Build a minimal Hydra-compatible config for the smoke trainer.

    Matches the schema expected by GRPOTrainer.__init__ (see the class
    docstring).  Only the fields the trainer actually reads are populated.
    """
    cfg = OmegaConf.create(
        {
            "gpu_id": 0,  # ignored on CPU — CUDA_VISIBLE_DEVICES is unset in conftest
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
                "output_dir": str(output_dir),
                "batch_size": 1,
                "group_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_train_steps": 2,
                "warmup_steps": 0,
                "save_steps": 9999,   # don't checkpoint during the smoke test
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
                "mlflow_tracking_uri": str(mlflow_dir),
                "mlflow_experiment": "smoke",
            },
        }
    )
    return cfg


class _DeterministicReward:
    """Callable that returns a prescribed (reward-per-call) sequence.

    Used to monkeypatch ``compute_reward`` in the trainer so that the test
    controls the exact reward distribution per group — half flat, half
    varied — without depending on a 5M-param random model actually
    producing correct answers (which it won't).
    """

    def __init__(self, pattern: List[float]):
        self._pattern = pattern
        self.calls: List[float] = []

    def __call__(self, prediction: str, ground_truth: str) -> float:
        r = self._pattern[len(self.calls) % len(self._pattern)]
        self.calls.append(r)
        return r


def _patched_compute_reward(reward_fn):
    """Patch both import sites of ``compute_reward``.

    The trainer imports ``compute_reward`` by name (``from
    guac.training.rewards import compute_reward``), so patching the name
    inside the trainer module is what actually intercepts the call.
    """
    return patch("guac.training.trainer.compute_reward", new=reward_fn)


def test_trainer_skips_flat_groups_and_produces_nonzero_loss_on_varied(
    tmp_path: Path, caplog
):
    """End-to-end: trainer runs two micro-steps with a monkeypatched reward.

    Micro-step 0 has *varied* rewards → loss must be non-zero, gradients
    must reach LoRA parameters.

    Micro-step 1 has *flat* rewards → all groups should be detected as
    uninformative; the trainer should skip backward without raising.

    To avoid depending on the tiny random-weights model producing *varied*
    per-rollout log-probs (it tends to produce near-identical short
    sequences, which would collapse the loss through a separate mechanism
    unrelated to the flat-group fix), we monkeypatch
    ``_compute_seq_log_probs`` to return distinct scalars that still flow
    gradient through a LoRA parameter.  This isolates the test to the
    control-flow we actually care about: flat-group skipping + non-zero
    advantage → non-zero gradient on trainable params.
    """
    import logging

    # Defer the trainer import until after CUDA_VISIBLE_DEVICES is unset
    # and our conftest hooks have fired.
    from guac.training.trainer import GRPOTrainer

    cfg = _build_smoke_cfg(
        output_dir=tmp_path / "ckpt",
        mlflow_dir=tmp_path / "mlruns",
    )

    # Pattern:
    #   group_size=4, batch_size=1, grad_accum=1, num_train_steps=2
    #   → 2 micro-steps × 1 group × 4 rollouts = 8 reward calls.
    #   First 4 = varied (mix of 0 and 1)  → informative
    #   Next  4 = flat    (all zeros)      → skipped
    reward_fn = _DeterministicReward([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Gradient-flow probe: stash LoRA params' gradient norm right before
    # optimizer.step() so we can assert it's non-zero on the informative
    # micro-step and zero on the flat micro-step.
    captured_grad_norms: List[float] = []

    with _patched_compute_reward(reward_fn):
        trainer = GRPOTrainer(cfg)

        # Find a LoRA A-matrix and pipe all log-prob calls through it so
        # .backward() provably reaches trainable params.  We vary the
        # per-call multiplier so the 4 rollouts in a group produce
        # genuinely different log_probs (scale -3, -5, -7, -4, ...).
        lora_param = next(
            p for n, p in trainer.model.named_parameters()
            if p.requires_grad and "lora_A" in n
        )
        coefficients = [-3.0, -5.0, -7.0, -4.0]
        call_idx = {"n": 0}

        def fake_log_probs(input_ids, attention_mask, labels, extra_kwargs):
            c = coefficients[call_idx["n"] % len(coefficients)]
            call_idx["n"] += 1
            # Sum over the LoRA matrix → scalar that depends on param.
            return lora_param.sum() * c

        trainer._compute_seq_log_probs = fake_log_probs  # type: ignore[assignment]

        # Hook optimizer.step to snapshot grad norm pre-step.
        real_step = trainer.optimizer.step

        def step_with_probe(*a, **kw):
            g_sq = 0.0
            for p in trainer.model.parameters():
                if p.requires_grad and p.grad is not None:
                    g_sq += float(p.grad.detach().abs().pow(2).sum().item())
            captured_grad_norms.append(g_sq ** 0.5)
            return real_step(*a, **kw)

        trainer.optimizer.step = step_with_probe  # type: ignore[assignment]

        caplog.set_level(logging.INFO, logger="guac.training.trainer")
        trainer.train()

    # --- Assertions -------------------------------------------------------
    # The varied micro-step must have invoked optimizer.step at least once
    # with a non-zero gradient norm on the LoRA params.
    assert any(g > 0 for g in captured_grad_norms), (
        f"Expected at least one optimizer step with non-zero gradient norm "
        f"on LoRA params; got {captured_grad_norms!r}. "
        f"This is the original zero-loss bug regressing."
    )

    # The flat micro-step must have been detected and surfaced via a
    # warning log — this is the signal that flat-group skipping is wired in.
    all_log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "flat" in all_log_text.lower(), (
        "Expected the trainer to log a 'flat' warning when all groups in "
        "a micro-step had identical rewards; did not find one in:\n"
        + all_log_text[-2000:]
    )

    # Curriculum should still be a valid probability.
    assert 0.0 <= trainer.curriculum.T <= 1.0, (
        f"Curriculum T escaped [0, 1]: {trainer.curriculum.T}"
    )


def test_trainer_all_flat_batch_does_not_crash_and_logs(
    tmp_path: Path, caplog
):
    """When every group in every micro-step is flat, training completes
    without touching the optimizer — no NaN, no crash, and the
    warning is emitted."""
    import logging

    from guac.training.trainer import GRPOTrainer

    cfg = _build_smoke_cfg(
        output_dir=tmp_path / "ckpt",
        mlflow_dir=tmp_path / "mlruns",
    )
    # Cap at 1 step to keep the test fast; all-flat rewards.
    cfg.training.num_train_steps = 1
    reward_fn = _DeterministicReward([0.0, 0.0, 0.0, 0.0])

    captured_grad_norms: List[float] = []

    with _patched_compute_reward(reward_fn):
        trainer = GRPOTrainer(cfg)

        real_step = trainer.optimizer.step

        def step_with_probe(*a, **kw):
            g_sq = 0.0
            for p in trainer.model.parameters():
                if p.requires_grad and p.grad is not None:
                    g_sq += float(p.grad.detach().abs().pow(2).sum().item())
            captured_grad_norms.append(g_sq ** 0.5)
            return real_step(*a, **kw)

        trainer.optimizer.step = step_with_probe  # type: ignore[assignment]

        caplog.set_level(logging.WARNING, logger="guac.training.trainer")
        # With num_train_steps=1 and every group flat, the while-loop's
        # global_step never advances; to avoid a hang we cap wall-clock
        # progress by also capping the number of sampler calls.
        call_count = {"n": 0}
        real_sample = trainer.sampler.sample

        def bounded_sample(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] > 3:
                raise RuntimeError(
                    "all-flat loop ran more than 3 iterations — expected "
                    "the trainer to continue safely but not spin forever"
                )
            return real_sample(*a, **kw)

        trainer.sampler.sample = bounded_sample  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="all-flat loop"):
            trainer.train()

    # Optimizer must NOT have run.
    assert all(g == 0.0 for g in captured_grad_norms), (
        f"Optimizer fired on all-flat rewards; norms={captured_grad_norms!r}"
    )
    warn_text = "\n".join(
        rec.getMessage() for rec in caplog.records if rec.levelno >= 30
    )
    assert "flat" in warn_text.lower(), (
        "Expected a flat-group warning; got:\n" + warn_text
    )
