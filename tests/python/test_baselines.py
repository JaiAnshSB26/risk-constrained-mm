# ============================================================================
# risk-constrained-mm :: tests/python/test_baselines.py
# ============================================================================
"""
Pytest suite for the Avellaneda-Stoikov baseline agent.

Tests cover:
  * Interface compatibility with PPO/CPPO agents.
  * Symmetric spreads when inventory is zero.
  * Inventory skew: long → wider bid / tighter ask.
  * Inventory skew: short → tighter bid / wider ask.
  * Action bounds respected.
  * Deterministic (no randomness).
  * Config defaults and overrides.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def agent():
    """Default AS agent."""
    from rcmm.baselines import AvellanedaStoikovAgent
    return AvellanedaStoikovAgent()


@pytest.fixture
def obs_zero_inventory():
    """Observation with zero inventory (normalised inventory = 0.0)."""
    obs = np.zeros(22, dtype=np.float64)
    obs[-2] = 0.0  # normalised inventory = 0
    obs[-1] = 0.0  # normalised PnL
    return obs


@pytest.fixture
def obs_long_inventory():
    """Observation with high positive (long) inventory."""
    obs = np.zeros(22, dtype=np.float64)
    obs[-2] = 0.5  # normalised inventory = +50 (maxinv=100)
    return obs


@pytest.fixture
def obs_short_inventory():
    """Observation with negative (short) inventory."""
    obs = np.zeros(22, dtype=np.float64)
    obs[-2] = -0.5  # normalised inventory = -50
    return obs


# ============================================================================
# Interface Tests
# ============================================================================


class TestASInterface:
    """Baseline agent has the same interface as PPO/CPPO agents."""

    def test_select_action_returns_tuple(self, agent, obs_zero_inventory):
        """select_action returns (action, log_prob, value) tuple."""
        result = agent.select_action(obs_zero_inventory)
        assert len(result) == 3

    def test_action_shape(self, agent, obs_zero_inventory):
        """Action is a 4-element array: [bid_spread, ask_spread, bid_size, ask_size]."""
        action, _, _ = agent.select_action(obs_zero_inventory)
        assert action.shape == (4,)

    def test_log_prob_is_tensor(self, agent, obs_zero_inventory):
        """log_prob is a scalar tensor (dummy)."""
        _, lp, _ = agent.select_action(obs_zero_inventory)
        assert isinstance(lp, torch.Tensor)
        assert lp.shape == ()

    def test_value_is_tensor(self, agent, obs_zero_inventory):
        """value is a scalar tensor (dummy)."""
        _, _, v = agent.select_action(obs_zero_inventory)
        assert isinstance(v, torch.Tensor)
        assert v.shape == ()

    def test_action_finite(self, agent, obs_zero_inventory):
        """All action components are finite."""
        action, _, _ = agent.select_action(obs_zero_inventory)
        assert np.all(np.isfinite(action))


# ============================================================================
# Spread Symmetry Tests
# ============================================================================


class TestASSpreadSymmetry:
    """When inventory is zero, spreads should be symmetric."""

    def test_zero_inventory_symmetric_spreads(self, agent, obs_zero_inventory):
        """With zero inventory, bid_spread ≈ ask_spread."""
        action, _, _ = agent.select_action(obs_zero_inventory)
        bid_spread = action[0]
        ask_spread = action[1]
        assert abs(bid_spread - ask_spread) < 1e-10, (
            f"Asymmetric spreads at zero inventory: bid={bid_spread}, ask={ask_spread}"
        )

    def test_zero_inventory_positive_spreads(self, agent, obs_zero_inventory):
        """Spreads are positive (agent quotes away from mid)."""
        action, _, _ = agent.select_action(obs_zero_inventory)
        assert action[0] > 0  # bid_spread > 0
        assert action[1] > 0  # ask_spread > 0


# ============================================================================
# Inventory Skew Tests (Core AS Logic)
# ============================================================================


class TestASInventorySkew:
    """
    Avellaneda-Stoikov skews quotes based on inventory:
    - Long inventory  → wider bid (less buying), tighter ask (more selling)
    - Short inventory → tighter bid (more buying), wider ask (less selling)
    """

    def test_long_inventory_wider_bid(self, agent):
        """Long inventory → bid spread > ask spread (skew to sell)."""
        action, _, _ = agent.select_action(
            _make_obs(norm_inv=0.5)  # +50 inventory
        )
        bid_spread = action[0]
        ask_spread = action[1]
        assert bid_spread > ask_spread, (
            f"Long inv should widen bid: bid={bid_spread}, ask={ask_spread}"
        )

    def test_short_inventory_wider_ask(self, agent):
        """Short inventory → ask spread > bid spread (skew to buy)."""
        action, _, _ = agent.select_action(
            _make_obs(norm_inv=-0.5)  # -50 inventory
        )
        bid_spread = action[0]
        ask_spread = action[1]
        assert ask_spread > bid_spread, (
            f"Short inv should widen ask: bid={bid_spread}, ask={ask_spread}"
        )

    def test_larger_inventory_larger_skew(self, agent):
        """More extreme inventory produces more spread skew."""
        action_small, _, _ = agent.select_action(_make_obs(norm_inv=0.1))
        action_large, _, _ = agent.select_action(_make_obs(norm_inv=0.5))

        skew_small = abs(action_small[0] - action_small[1])
        skew_large = abs(action_large[0] - action_large[1])

        assert skew_large > skew_small, (
            f"Larger inventory should produce larger skew: "
            f"small={skew_small}, large={skew_large}"
        )

    def test_get_spreads_zero_inventory(self):
        """get_spreads convenience method: zero inventory → symmetric."""
        from rcmm.baselines import AvellanedaStoikovAgent
        agent = AvellanedaStoikovAgent()
        bid_s, ask_s = agent.get_spreads(inventory=0.0)
        assert abs(bid_s - ask_s) < 1e-10

    def test_get_spreads_positive_inventory(self):
        """get_spreads: positive inventory → bid > ask (skew to sell)."""
        from rcmm.baselines import AvellanedaStoikovAgent
        agent = AvellanedaStoikovAgent()
        bid_s, ask_s = agent.get_spreads(inventory=50.0)
        assert bid_s > ask_s

    def test_get_spreads_negative_inventory(self):
        """get_spreads: negative inventory → ask > bid (skew to buy)."""
        from rcmm.baselines import AvellanedaStoikovAgent
        agent = AvellanedaStoikovAgent()
        bid_s, ask_s = agent.get_spreads(inventory=-50.0)
        assert ask_s > bid_s


# ============================================================================
# Action Bounds Tests
# ============================================================================


class TestASActionBounds:
    """Spreads are clamped to [min_spread, max_spread]."""

    def test_extreme_long_inventory_clamped(self):
        """Massive inventory doesn't produce spreads outside bounds."""
        from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig

        cfg = AvellanedaStoikovConfig(gamma=1.0, sigma=10.0, max_inventory=100.0)
        agent = AvellanedaStoikovAgent(cfg)
        action, _, _ = agent.select_action(_make_obs(norm_inv=1.0))

        assert cfg.min_spread <= action[0] <= cfg.max_spread
        assert cfg.min_spread <= action[1] <= cfg.max_spread

    def test_extreme_short_inventory_clamped(self):
        """Massive negative inventory also clamped."""
        from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig

        cfg = AvellanedaStoikovConfig(gamma=1.0, sigma=10.0, max_inventory=100.0)
        agent = AvellanedaStoikovAgent(cfg)
        action, _, _ = agent.select_action(_make_obs(norm_inv=-1.0))

        assert cfg.min_spread <= action[0] <= cfg.max_spread
        assert cfg.min_spread <= action[1] <= cfg.max_spread

    def test_order_size_fixed(self, agent, obs_zero_inventory):
        """Order sizes are fixed at cfg.order_size."""
        action, _, _ = agent.select_action(obs_zero_inventory)
        assert action[2] == agent.cfg.order_size
        assert action[3] == agent.cfg.order_size


# ============================================================================
# Determinism Tests
# ============================================================================


class TestASDeterminism:
    """Baseline is deterministic (no neural network sampling)."""

    def test_same_obs_same_action(self, agent, obs_zero_inventory):
        """Same observation always produces same action."""
        a1, _, _ = agent.select_action(obs_zero_inventory)
        a2, _, _ = agent.select_action(obs_zero_inventory)
        np.testing.assert_array_equal(a1, a2)


# ============================================================================
# Config Tests
# ============================================================================


class TestASConfig:
    """Configuration affects the agent's quoting behavior."""

    def test_higher_gamma_wider_spread(self):
        """Higher risk aversion → wider optimal spread."""
        from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig

        agent_low = AvellanedaStoikovAgent(AvellanedaStoikovConfig(gamma=0.05))
        agent_high = AvellanedaStoikovAgent(AvellanedaStoikovConfig(gamma=0.5))
        obs = _make_obs(norm_inv=0.0)

        a_low, _, _ = agent_low.select_action(obs)
        a_high, _, _ = agent_high.select_action(obs)

        assert a_high[0] > a_low[0], "Higher gamma should produce wider spreads"

    def test_higher_sigma_wider_spread(self):
        """Higher volatility → wider optimal spread."""
        from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig

        agent_low = AvellanedaStoikovAgent(AvellanedaStoikovConfig(sigma=1.0))
        agent_high = AvellanedaStoikovAgent(AvellanedaStoikovConfig(sigma=5.0))
        obs = _make_obs(norm_inv=0.0)

        a_low, _, _ = agent_low.select_action(obs)
        a_high, _, _ = agent_high.select_action(obs)

        assert a_high[0] > a_low[0], "Higher sigma should produce wider spreads"

    def test_default_config_values(self):
        """Default config matches expected values."""
        from rcmm.baselines import AvellanedaStoikovConfig

        cfg = AvellanedaStoikovConfig()
        assert cfg.gamma == 0.1
        assert cfg.sigma == 2.0
        assert cfg.kappa == 1.5
        assert cfg.order_size == 5.0


# ============================================================================
# Integration Test
# ============================================================================


class TestASIntegration:
    """Run AS baseline through the actual environment."""

    def test_run_episode(self):
        """AS baseline completes a full episode without errors."""
        from rcmm import EnvConfig, LimitOrderBookEnv
        from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig

        cfg = EnvConfig()
        cfg.max_steps = 50
        cfg.warmup_ticks = 30
        cfg.seed = 42

        env = LimitOrderBookEnv(config=cfg)
        agent = AvellanedaStoikovAgent(AvellanedaStoikovConfig(
            max_inventory=cfg.max_inventory,
        ))

        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _, _ = agent.select_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1
            assert np.all(np.isfinite(obs)), f"Non-finite obs at step {steps}"
            assert np.isfinite(reward), f"Non-finite reward at step {steps}"

        assert steps == 50  # max_steps reached


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_obs(norm_inv: float = 0.0, norm_pnl: float = 0.0) -> np.ndarray:
    """Create a minimal observation array with specified normalised inventory."""
    obs = np.zeros(22, dtype=np.float64)
    obs[-2] = norm_inv
    obs[-1] = norm_pnl
    return obs
