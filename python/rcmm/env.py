# ============================================================================
# risk-constrained-mm :: python/rcmm/env.py
# ============================================================================
"""
Gymnasium environment wrapping the C++ MarketEnvironment via pybind11.

The observation is a zero-copy NumPy array backed by a flat C++ buffer:

    obs = [bid_price_0, bid_vol_0, ..., bid_price_{N-1}, bid_vol_{N-1},
           ask_price_0, ask_vol_0, ..., ask_price_{N-1}, ask_vol_{N-1},
           normalised_inventory, normalised_pnl]

Action space (Box, 4-dim continuous):
    [bid_spread, ask_spread, bid_size, ask_size]

    - bid_spread / ask_spread: offset from mid-price in ticks (>= 1)
    - bid_size / ask_size: order quantity (>= 1)

Reward function:
    R_t = ΔPnL_t - γ · (Inventory_t)²

    γ (inventory_aversion) defaults to 0.01 and is configurable via
    EnvConfig.inventory_aversion or the `inventory_aversion` kwarg.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# The compiled C++ module is copied into this package directory at build time.
from rcmm._rcmm_core import EnvConfig, MarketEnvironment


class LimitOrderBookEnv(gym.Env):
    """
    Gymnasium wrapper for the C++ risk-constrained market-making environment.

    Zero-copy observations: the returned NumPy array is a view into a
    pre-allocated C++ buffer — no per-step heap allocation on the Python side.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        inventory_aversion: float | None = None,
    ) -> None:
        super().__init__()

        self._config = config if config is not None else EnvConfig()

        # Allow overriding γ from Python side.
        if inventory_aversion is not None:
            self._config.inventory_aversion = inventory_aversion

        self._env = MarketEnvironment(self._config)

        obs_dim = self._env.obs_size()

        # Observation: normalised LOB snapshot + agent state.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64,
        )

        # Action: [bid_spread, ask_spread, bid_size, ask_size].
        # Spreads in [1, 20] ticks; sizes in [1, 50] lots.
        self.action_space = spaces.Box(
            low=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            high=np.array([20.0, 20.0, 50.0, 50.0], dtype=np.float64),
            dtype=np.float64,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a fresh episode."""
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._config.seed = seed
            self._env = MarketEnvironment(self._config)

        obs = self._env.reset()
        info: dict[str, Any] = {"inventory": 0.0, "pnl": 0.0}
        return np.asarray(obs, dtype=np.float64), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one market-making step."""
        bid_spread = float(action[0])
        ask_spread = float(action[1])
        bid_size = float(action[2])
        ask_size = float(action[3])

        obs, reward, terminated, truncated, info = self._env.step(
            bid_spread, ask_spread, bid_size, ask_size
        )

        return (
            np.asarray(obs, dtype=np.float64),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        return self._env.obs_size()
