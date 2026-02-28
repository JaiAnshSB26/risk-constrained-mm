# ============================================================================
# risk-constrained-mm :: python/rcmm/regime_wrapper.py
# ============================================================================
"""
Gymnasium wrapper for domain randomization via Hawkes regime shifts.

On each ``reset()``, the wrapper randomly samples Hawkes process parameters
(μ, α, β) from one of two regime distributions — "Normal" or "Flash Crash"
— introducing non-stationarity into the training environment.  This forces
the RL agent to learn policies robust to changing market microstructure.

Regime distributions:

    Normal Regime (stable, low clustering):
        μ ~ clip(N(5.0, 0.5), 1.0, 20.0)
        α ~ clip(N(1.5, 0.3), 0.1, 4.0)
        β ~ clip(N(5.0, 0.5), 2.0, 20.0)
        Branching ratio ≈ 0.30

    Flash Crash Regime (intense clustering, one-sided flow):
        μ ~ clip(N(5.0, 0.5), 1.0, 20.0)
        α ~ clip(N(9.5, 0.5), 5.0, 9.9)
        β ~ clip(N(10.0, 0.5), 6.0, 20.0)
        Branching ratio ≈ 0.95
        buy_prob → 0.15 (heavy sell pressure)

Both distributions enforce the stationarity constraint α < β by clamping
α to at most 0.98 × β after sampling.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np

from rcmm._rcmm_core import EnvConfig, HawkesParams, MarketEnvironment


# ── Regime distribution specifications ───────────────────────────────────────


@dataclasses.dataclass
class RegimeSpec:
    """Sampling distribution for a single Hawkes regime."""

    name: str

    # μ (baseline intensity)
    mu_mean: float = 5.0
    mu_std: float = 0.5
    mu_lo: float = 1.0
    mu_hi: float = 20.0

    # α (excitation jump)
    alpha_mean: float = 1.5
    alpha_std: float = 0.3
    alpha_lo: float = 0.1
    alpha_hi: float = 4.0

    # β (decay rate)
    beta_mean: float = 5.0
    beta_std: float = 0.5
    beta_lo: float = 2.0
    beta_hi: float = 20.0

    # Optional mark overrides (e.g., sell pressure in crash).
    buy_prob: float | None = None

    def sample(self, rng: np.random.Generator) -> HawkesParams:
        """Sample a HawkesParams from this regime's distribution."""
        mu = float(np.clip(rng.normal(self.mu_mean, self.mu_std),
                           self.mu_lo, self.mu_hi))
        alpha = float(np.clip(rng.normal(self.alpha_mean, self.alpha_std),
                              self.alpha_lo, self.alpha_hi))
        beta = float(np.clip(rng.normal(self.beta_mean, self.beta_std),
                             self.beta_lo, self.beta_hi))

        # Enforce stationarity: α < β (clamp α to 0.98·β).
        alpha = min(alpha, 0.98 * beta)

        hp = HawkesParams()
        hp.mu = mu
        hp.alpha = alpha
        hp.beta = beta
        return hp


# Pre-defined regime specifications.
NORMAL_REGIME_SPEC = RegimeSpec(
    name="normal",
    mu_mean=5.0, mu_std=0.5, mu_lo=1.0, mu_hi=20.0,
    alpha_mean=1.5, alpha_std=0.3, alpha_lo=0.1, alpha_hi=4.0,
    beta_mean=5.0, beta_std=0.5, beta_lo=2.0, beta_hi=20.0,
    buy_prob=None,  # keep default (0.5)
)

FLASH_CRASH_REGIME_SPEC = RegimeSpec(
    name="flash_crash",
    mu_mean=5.0, mu_std=0.5, mu_lo=1.0, mu_hi=20.0,
    alpha_mean=9.5, alpha_std=0.5, alpha_lo=5.0, alpha_hi=9.9,
    beta_mean=10.0, beta_std=0.5, beta_lo=6.0, beta_hi=20.0,
    buy_prob=0.15,  # heavy sell pressure
)


# ── Gymnasium Wrapper ────────────────────────────────────────────────────────


class RegimeRandomizationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that randomizes Hawkes regime on each reset().

    On each episode reset, samples from one of the configured regime
    distributions and reconstructs the C++ MarketEnvironment with the
    new parameters.  This produces non-stationary training episodes
    that force the agent to be robust to regime shifts.

    Args:
        env: The base LimitOrderBookEnv to wrap.
        regimes: List of (RegimeSpec, probability) tuples.
            Probabilities must sum to 1.0.
        seed: RNG seed for regime sampling reproducibility.
    """

    def __init__(
        self,
        env: gym.Env,
        regimes: list[tuple[RegimeSpec, float]] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(env)

        if regimes is None:
            regimes = [
                (NORMAL_REGIME_SPEC, 0.8),
                (FLASH_CRASH_REGIME_SPEC, 0.2),
            ]

        self._regime_specs = [r for r, _ in regimes]
        self._regime_probs = np.array([p for _, p in regimes], dtype=np.float64)

        # Normalise in case of floating-point drift.
        self._regime_probs /= self._regime_probs.sum()

        self._rng = np.random.default_rng(seed)

        # Track the current regime for diagnostics.
        self._current_regime: str = "unknown"
        self._current_hawkes: HawkesParams | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset with randomized Hawkes regime."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 1. Sample regime.
        regime_idx = int(self._rng.choice(
            len(self._regime_specs), p=self._regime_probs
        ))
        spec = self._regime_specs[regime_idx]

        # 2. Sample Hawkes parameters from the regime distribution.
        hawkes = spec.sample(self._rng)
        self._current_regime = spec.name
        self._current_hawkes = hawkes

        # 3. Update the inner env's config and reconstruct.
        inner_env = self.env  # type: ignore[attr-defined]
        inner_env._config.hawkes_params = hawkes

        if spec.buy_prob is not None:
            inner_env._config.mark_config.buy_prob = spec.buy_prob
        else:
            inner_env._config.mark_config.buy_prob = 0.5

        # Increment seed for variety across episodes.
        inner_env._config.seed = int(self._rng.integers(0, 2**32))
        inner_env._env = MarketEnvironment(inner_env._config)

        obs = inner_env._env.reset()
        info: dict[str, Any] = {
            "inventory": 0.0,
            "pnl": 0.0,
            "regime": self._current_regime,
            "hawkes_mu": hawkes.mu,
            "hawkes_alpha": hawkes.alpha,
            "hawkes_beta": hawkes.beta,
        }
        return np.asarray(obs, dtype=np.float64), info

    @property
    def current_regime(self) -> str:
        """Name of the currently active regime."""
        return self._current_regime

    @property
    def current_hawkes(self) -> HawkesParams | None:
        """Current episode's Hawkes parameters."""
        return self._current_hawkes
