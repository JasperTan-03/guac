"""Curriculum learning components for GRPO training.

Provides:
  - ``CurriculumState`` — dataclass that tracks and updates the target difficulty T.
  - ``CurriculumSampler`` — selects training examples by proximity to T, in two modes.

Compatible with: torch>=2.3
"""

import math
import random
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CurriculumState:
    """Tracks and updates the current target difficulty T.

    The update rule applied after each optimizer step is::

        T_new = clip(T + eta * tanh(alpha * (R_avg - beta)), d_min, d_max)

    When R_avg > beta the model is finding the current difficulty easy, so T
    is increased.  When R_avg < beta the model is struggling, so T is
    decreased.  ``eta`` controls the maximum step size and ``alpha`` controls
    how sharply the signal saturates.

    Attributes:
        T: Current target difficulty in [d_min, d_max].
        eta: Step size / learning rate for the curriculum update.
        alpha: Sensitivity parameter scaling the reward deviation.
        beta: Baseline reward level at which T does not change.
        d_min: Lower bound for T (inclusive).
        d_max: Upper bound for T (inclusive).
    """

    T: float
    eta: float
    alpha: float
    beta: float
    d_min: float
    d_max: float

    def __post_init__(self) -> None:
        """Validate that bounds and initial T are consistent."""
        if self.d_min >= self.d_max:
            raise ValueError(
                f"d_min ({self.d_min}) must be strictly less than d_max ({self.d_max})."
            )
        if not (self.d_min <= self.T <= self.d_max):
            raise ValueError(
                f"T ({self.T}) must lie in [d_min={self.d_min}, d_max={self.d_max}]."
            )
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}.")

    def update(self, R_avg: float) -> None:
        """Update T using the curriculum update rule.

        Args:
            R_avg: Mean reward over the current batch's group responses.
                   Typically in [0.0, 1.0] for binary exact-match rewards.
        """
        delta = self.eta * math.tanh(self.alpha * (R_avg - self.beta))
        self.T = float(max(self.d_min, min(self.d_max, self.T + delta)))


class CurriculumSampler:
    """Samples dataset indices based on current target difficulty T.

    Two sampling modes are supported:

    **baseline**
        At each step, sort all examples by ``|d_i - T|`` (ascending) and
        return the top-``batch_size`` indices.  Ties are broken randomly by
        shuffling before the sort (Python's ``sort`` is stable, so the
        shuffle order is preserved among equal distances).

    **gaussian**
        Compute unnormalised weights::

            w_i = exp(-(d_i - T)^2 / (2 * sigma^2))

        Normalise to a probability distribution and sample ``batch_size``
        indices **without replacement** via ``torch.multinomial``.

    Both modes re-compute the selection from scratch at every call to
    :meth:`sample`, ensuring the latest value of T is always used.

    Args:
        difficulties: Per-example difficulty scores (float in [0, 1]).
        mode: ``'baseline'`` or ``'gaussian'``.
        sigma: Gaussian bandwidth.  Larger values spread probability more
               evenly across all difficulties; smaller values concentrate it
               tightly around T.  Only used in ``'gaussian'`` mode.
        seed: Integer seed for the internal Python RNG (tie-breaking in
              baseline) and the implicit global torch RNG state.

    Raises:
        ValueError: If ``mode`` is not ``'baseline'`` or ``'gaussian'``.
        ValueError: If ``difficulties`` is empty.
    """

    def __init__(
        self,
        difficulties: List[float],
        mode: str,
        sigma: float = 0.15,
        seed: int = 42,
    ) -> None:
        if mode not in ("baseline", "gaussian"):
            raise ValueError(
                f"sampling_mode must be 'baseline' or 'gaussian', got {mode!r}."
            )
        if not difficulties:
            raise ValueError("difficulties list must not be empty.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")

        self.difficulties = difficulties
        self.mode = mode
        self.sigma = sigma
        self._rng = random.Random(seed)
        # Pre-compute a float32 tensor for fast vectorised operations.
        self._diff_tensor = torch.tensor(difficulties, dtype=torch.float32)

    def sample(self, batch_size: int, T: float) -> List[int]:
        """Return ``batch_size`` dataset indices for the current target T.

        Args:
            batch_size: Number of indices to return.
            T: Current target difficulty.

        Returns:
            List of integer dataset indices, length ``batch_size``.

        Raises:
            ValueError: If ``batch_size`` exceeds the dataset size.
        """
        n = len(self.difficulties)
        if batch_size > n:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds dataset size ({n}). "
                "Reduce batch_size or provide a larger dataset."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        if self.mode == "baseline":
            return self._sample_baseline(batch_size, T)
        else:
            return self._sample_gaussian(batch_size, T)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_baseline(self, batch_size: int, T: float) -> List[int]:
        """Greedy nearest-neighbour selection by |d_i - T|.

        Args:
            batch_size: Number of indices to select.
            T: Current target difficulty.

        Returns:
            List of selected integer indices, length ``batch_size``.
        """
        # Build (distance, index) pairs, shuffle first to break ties randomly,
        # then stable-sort by distance so the shuffled order is preserved for
        # equal distances.
        pairs = [(abs(d - T), i) for i, d in enumerate(self.difficulties)]
        self._rng.shuffle(pairs)
        pairs.sort(key=lambda x: x[0])
        return [idx for _, idx in pairs[:batch_size]]

    def _sample_gaussian(self, batch_size: int, T: float) -> List[int]:
        """Probability-weighted sampling without replacement.

        Weights are proportional to a Gaussian centred at T::

            w_i = exp(-(d_i - T)^2 / (2 * sigma^2))

        Args:
            batch_size: Number of indices to sample.
            T: Current target difficulty.

        Returns:
            List of sampled integer indices, length ``batch_size``.
        """
        diff = self._diff_tensor - T          # (N,)
        # Compute log-weights and subtract max for numerical stability.
        log_w = -(diff ** 2) / (2.0 * self.sigma ** 2)
        log_w = log_w - log_w.max()
        weights = torch.exp(log_w)            # (N,)
        probs = weights / weights.sum()       # normalised probability distribution

        indices = torch.multinomial(probs, batch_size, replacement=False)
        return indices.tolist()
