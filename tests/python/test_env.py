# ============================================================================
# risk-constrained-mm :: tests/python/test_env.py
# ============================================================================
"""
Rigorous pytest suite for the LimitOrderBookEnv (Phase 6).

Covers:
  * C++ module import.
  * Observation space shape matches returned array.
  * Action space validity.
  * Reset returns valid observation.
  * Step returns valid (obs, reward, terminated, truncated, info) tuple.
  * Multiple steps without crash.
  * Inventory and PnL tracking.
  * Episode termination (done flag).
  * Reproducibility (same seed).
  * Zero-copy: the underlying buffer address is stable.
  * Performance benchmark: >10,000 steps/second.
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Create a fresh LimitOrderBookEnv instance."""
    from rcmm.env import LimitOrderBookEnv
    e = LimitOrderBookEnv()
    yield e


@pytest.fixture
def config():
    """Create a test EnvConfig."""
    from rcmm._rcmm_core import EnvConfig
    cfg = EnvConfig()
    cfg.max_steps = 500
    cfg.ticks_per_step = 5
    cfg.warmup_ticks = 100
    cfg.obs_depth = 5
    return cfg


@pytest.fixture
def small_env(config):
    """Create a small env for fast tests."""
    from rcmm.env import LimitOrderBookEnv
    e = LimitOrderBookEnv(config=config)
    yield e


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    """Verify the C++ module and Python wrapper are importable."""

    def test_cpp_module_importable(self) -> None:
        import rcmm._rcmm_core as core
        assert hasattr(core, "MarketEnvironment")
        assert hasattr(core, "EnvConfig")
        assert hasattr(core, "StepResult")

    def test_env_importable(self) -> None:
        from rcmm.env import LimitOrderBookEnv
        assert LimitOrderBookEnv is not None

    def test_package_importable(self) -> None:
        import rcmm
        assert hasattr(rcmm, "LimitOrderBookEnv")


# ---------------------------------------------------------------------------
# Space tests
# ---------------------------------------------------------------------------

class TestSpaces:
    """Observation and action space validation."""

    def test_observation_space_shape(self, env) -> None:
        obs_dim = env.obs_dim
        assert env.observation_space.shape == (obs_dim,)
        # Default obs_depth=5 => 5*4 + 2 = 22
        assert obs_dim == 22

    def test_observation_space_dtype(self, env) -> None:
        assert env.observation_space.dtype == np.float64

    def test_action_space_shape(self, env) -> None:
        assert env.action_space.shape == (4,)

    def test_action_space_dtype(self, env) -> None:
        assert env.action_space.dtype == np.float64

    def test_action_space_bounds(self, env) -> None:
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(low == np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.all(high == np.array([20.0, 20.0, 50.0, 50.0]))


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    """Environment reset validation."""

    def test_reset_returns_tuple(self, env) -> None:
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape(self, env) -> None:
        obs, info = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert obs.dtype == np.float64

    def test_reset_info_dict(self, env) -> None:
        _, info = env.reset()
        assert isinstance(info, dict)
        assert "inventory" in info
        assert "pnl" in info

    def test_reset_obs_is_finite(self, env) -> None:
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_multiple_times(self, env) -> None:
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (env.obs_dim,)


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    """Environment step validation."""

    def test_step_returns_5_tuple(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float64)
        result = small_env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_shape(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0])
        obs, reward, terminated, truncated, info = small_env.step(action)
        assert obs.shape == (small_env.obs_dim,)
        assert obs.dtype == np.float64

    def test_step_reward_is_float(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0])
        _, reward, _, _, _ = small_env.step(action)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0])
        _, _, terminated, truncated, _ = small_env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_has_keys(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0])
        _, _, _, _, info = small_env.step(action)
        assert "inventory" in info
        assert "pnl" in info
        assert "step_num" in info
        assert "fills" in info

    def test_step_obs_is_finite(self, small_env) -> None:
        small_env.reset()
        action = np.array([3.0, 3.0, 5.0, 5.0])
        for _ in range(20):
            obs, _, done, _, _ = small_env.step(action)
            assert np.all(np.isfinite(obs))
            if done:
                break

    def test_multiple_steps_no_crash(self, small_env) -> None:
        small_env.reset()
        action = np.array([5.0, 5.0, 10.0, 10.0])
        for _ in range(100):
            obs, reward, done, truncated, info = small_env.step(action)
            if done:
                break
        # Should have completed at least some steps.
        assert info["step_num"] > 0


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------
