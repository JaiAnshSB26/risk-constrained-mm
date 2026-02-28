# ============================================================================
# risk-constrained-mm :: tests/python/test_cppo.py
# ============================================================================
"""
Rigorous pytest suite for Phase 8: Risk-Constrained RL (CPPO).

Covers:
  * CVaR mathematical computation with known arrays.
  * CVaR edge cases (uniform, all-same, single element).
  * Domain Randomization: regime sampling on reset.
  * Domain Randomization: Hawkes params actually change.
  * Domain Randomization: stationarity enforced.
  * CPPO network: shape and gradient flow.
  * CPPO update: returns metrics including CVaR diagnostics.
  * CPPO Lagrangian: λ increases when CVaR violates threshold.
  * CPPO short training loop: no crash, no NaN.
  * CPPO vs PPO: different loss components.
  * Evaluation helpers: max drawdown, Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Fresh LimitOrderBookEnv with short episodes."""
    from rcmm._rcmm_core import EnvConfig
    from rcmm.env import LimitOrderBookEnv

    cfg = EnvConfig()
    cfg.max_steps = 200
    cfg.ticks_per_step = 5
    cfg.warmup_ticks = 50
    cfg.seed = 42
    return LimitOrderBookEnv(config=cfg)


@pytest.fixture
def wrapped_env(env):
    """Env wrapped with regime randomization."""
    from rcmm.regime_wrapper import RegimeRandomizationWrapper

    return RegimeRandomizationWrapper(env, seed=42)


@pytest.fixture
def cppo_config():
    """CPPO config for fast testing."""
    from rcmm.cppo import CPPOConfig

    return CPPOConfig(
        hidden_dim=32,
        num_hidden=2,
        learning_rate=3e-4,
        rollout_steps=64,
        num_minibatches=4,
        update_epochs=2,
        cvar_alpha=0.05,
        cvar_threshold=-5.0,
        lagrange_lr=0.05,
        lagrange_init=0.1,
        dual_update=True,
        seed=42,
    )


@pytest.fixture
def cppo_agent(env, cppo_config):
    """CPPO agent initialised from env fixture."""
    from rcmm.cppo import CPPOAgent

    return CPPOAgent(env, cppo_config)


# ============================================================================
# CVaR Mathematical Tests
# ============================================================================


class TestCVaR:
    """Tests for the CVaR computation against known values."""

    def test_known_array_5pct(self):
        """
        Hand-computed CVaR for a known array.

        rewards = [-10, -5, -3, -1, 0, 1, 3, 5, 7, 10]  (10 elements)
        α = 0.05 → k = max(1, int(10 * 0.05)) = 1
        Sorted: [-10, -5, -3, -1, 0, 1, 3, 5, 7, 10]
        VaR_0.05 = sorted[0] = -10
        CVaR_0.05 = mean(rewards[rewards ≤ -10]) = -10
        """
        from rcmm.cppo import compute_cvar

        rewards = torch.tensor([-10, -5, -3, -1, 0, 1, 3, 5, 7, 10],
                               dtype=torch.float32)
        var_a, cvar_a = compute_cvar(rewards, alpha=0.05)

        assert abs(var_a.item() - (-10.0)) < 1e-5
        assert abs(cvar_a.item() - (-10.0)) < 1e-5

    def test_known_array_20pct(self):
        """
        α = 0.20 → k = max(1, int(10 * 0.20)) = 2
        Sorted: [-10, -5, -3, -1, 0, 1, 3, 5, 7, 10]
        VaR_0.20 = sorted[1] = -5
        CVaR_0.20 = mean(rewards[rewards ≤ -5]) = mean(-10, -5) = -7.5
        """
        from rcmm.cppo import compute_cvar

        rewards = torch.tensor([-10, -5, -3, -1, 0, 1, 3, 5, 7, 10],
                               dtype=torch.float32)
        var_a, cvar_a = compute_cvar(rewards, alpha=0.20)

        assert abs(var_a.item() - (-5.0)) < 1e-5
        assert abs(cvar_a.item() - (-7.5)) < 1e-5

    def test_known_array_50pct(self):
        """
        α = 0.50 → k = 5
        VaR = sorted[4] = 0
        CVaR = mean(-10, -5, -3, -1, 0) = -3.8
        """
        from rcmm.cppo import compute_cvar

        rewards = torch.tensor([-10, -5, -3, -1, 0, 1, 3, 5, 7, 10],
                               dtype=torch.float32)
        var_a, cvar_a = compute_cvar(rewards, alpha=0.50)

        assert abs(var_a.item() - 0.0) < 1e-5
        assert abs(cvar_a.item() - (-3.8)) < 1e-5

    def test_all_same_values(self):
        """When all rewards are identical, VaR = CVaR = that value."""
        from rcmm.cppo import compute_cvar

        rewards = torch.full((100,), 5.0)
        var_a, cvar_a = compute_cvar(rewards, alpha=0.05)

        assert abs(var_a.item() - 5.0) < 1e-5
        assert abs(cvar_a.item() - 5.0) < 1e-5

    def test_single_element(self):
        """CVaR of a single element = that element."""
        from rcmm.cppo import compute_cvar

        rewards = torch.tensor([42.0])
        var_a, cvar_a = compute_cvar(rewards, alpha=0.05)

        assert abs(var_a.item() - 42.0) < 1e-5
        assert abs(cvar_a.item() - 42.0) < 1e-5

    def test_cvar_leq_var(self):
        """CVaR ≤ VaR always (expected shortfall is more extreme)."""
        from rcmm.cppo import compute_cvar

        rng = np.random.default_rng(123)
        rewards = torch.tensor(rng.normal(0, 10, size=1000), dtype=torch.float32)
        var_a, cvar_a = compute_cvar(rewards, alpha=0.05)

        assert cvar_a.item() <= var_a.item() + 1e-6

    def test_cvar_leq_mean(self):
        """CVaR ≤ mean (tail average is worse than overall average)."""
        from rcmm.cppo import compute_cvar

        rng = np.random.default_rng(456)
        rewards = torch.tensor(rng.normal(0, 10, size=1000), dtype=torch.float32)
        _, cvar_a = compute_cvar(rewards, alpha=0.05)

        assert cvar_a.item() <= rewards.mean().item() + 1e-6

    def test_negative_skew_worse_cvar(self):
        """
        A distribution with negative skew should have a more negative CVaR
        than a symmetric one.
        """
        from rcmm.cppo import compute_cvar

        # Symmetric
        symmetric = torch.tensor([-5, -3, -1, 0, 1, 3, 5, 7, 9, 11],
                                 dtype=torch.float32)
        # Negatively skewed
        neg_skew = torch.tensor([-50, -20, -1, 0, 1, 3, 5, 7, 9, 11],
                                dtype=torch.float32)

        _, cvar_sym = compute_cvar(symmetric, alpha=0.20)
        _, cvar_neg = compute_cvar(neg_skew, alpha=0.20)

        assert cvar_neg.item() < cvar_sym.item()


# ============================================================================
# Domain Randomization Tests
# ============================================================================


class TestDomainRandomization:
    """Tests for the RegimeRandomizationWrapper."""

    def test_reset_returns_regime_info(self, wrapped_env):
        """Reset info dict contains regime metadata."""
        obs, info = wrapped_env.reset()
        assert "regime" in info
        assert info["regime"] in ("normal", "flash_crash")
        assert "hawkes_mu" in info
        assert "hawkes_alpha" in info
        assert "hawkes_beta" in info

    def test_hawkes_params_change_across_resets(self, wrapped_env):
        """Multiple resets should sample different Hawkes parameters."""
        params_seen = set()
        for _ in range(20):
            _obs, info = wrapped_env.reset()
            key = (round(info["hawkes_mu"], 6),
                   round(info["hawkes_alpha"], 6),
                   round(info["hawkes_beta"], 6))
            params_seen.add(key)

        # With 20 resets, we should see variation.
        assert len(params_seen) > 1, "Hawkes params never changed across resets"

    def test_stationarity_enforced(self, wrapped_env):
        """All sampled Hawkes params satisfy α < β (stationarity)."""
        for _ in range(50):
            _obs, info = wrapped_env.reset()
            alpha = info["hawkes_alpha"]
            beta = info["hawkes_beta"]
            assert alpha < beta, (
                f"Stationarity violated: alpha={alpha} >= beta={beta}"
            )

    def test_regime_distribution(self, wrapped_env):
        """
        With default 80/20 split, we should see both regimes
        in a sufficient number of resets.
        """
        regime_counts = {"normal": 0, "flash_crash": 0}
        for _ in range(100):
            _obs, info = wrapped_env.reset()
            regime_counts[info["regime"]] += 1

        assert regime_counts["normal"] > 0, "Normal regime never sampled"
        assert regime_counts["flash_crash"] > 0, "Flash crash never sampled"
        # With 80/20 split and 100 samples, normal should dominate.
        assert regime_counts["normal"] > regime_counts["flash_crash"]

    def test_flash_crash_regime_has_sell_pressure(self, wrapped_env):
        """Flash crash episodes should have buy_prob < 0.5."""
        found_crash = False
        for _ in range(50):
            _obs, info = wrapped_env.reset()
            if info["regime"] == "flash_crash":
                # The mark config should have been modified.
                inner = wrapped_env.env  # type: ignore
                assert inner._config.mark_config.buy_prob < 0.5
                found_crash = True
                break
        assert found_crash, "Flash crash regime not encountered in 50 resets"

    def test_step_works_after_randomized_reset(self, wrapped_env):
        """Step succeeds after a regime-randomized reset."""
        obs, info = wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs2, reward, term, trunc, info2 = wrapped_env.step(action)
        assert obs2.shape == obs.shape
        assert np.isfinite(reward)

    def test_custom_regime_spec(self, env):
        """Custom RegimeSpec can be provided."""
        from rcmm.regime_wrapper import RegimeSpec, RegimeRandomizationWrapper

        mild = RegimeSpec(
            name="mild",
            mu_mean=3.0, mu_std=0.1, mu_lo=2.0, mu_hi=5.0,
            alpha_mean=0.5, alpha_std=0.1, alpha_lo=0.1, alpha_hi=1.0,
            beta_mean=3.0, beta_std=0.1, beta_lo=2.0, beta_hi=5.0,
        )
        w = RegimeRandomizationWrapper(env, regimes=[(mild, 1.0)], seed=99)
        _obs, info = w.reset()
        assert info["regime"] == "mild"
        assert 2.0 <= info["hawkes_mu"] <= 5.0

    def test_seed_reproducibility(self, env):
        """Same seed produces same regime sequence."""
        from rcmm.regime_wrapper import RegimeRandomizationWrapper

        def get_regimes(seed: int) -> list[str]:
            w = RegimeRandomizationWrapper(env, seed=seed)
            return [w.reset()[1]["regime"] for _ in range(10)]

        r1 = get_regimes(42)
        r2 = get_regimes(42)
        assert r1 == r2


# ============================================================================
# CPPO Agent Tests
# ============================================================================


class TestCPPOAgent:
    """Tests for the CPPO agent training pipeline."""

    def test_action_selection_valid(self, env, cppo_agent):
        """Actions are within bounds."""
        obs, _ = env.reset()
        for _ in range(5):
            action, lp, v = cppo_agent.select_action(obs)
            assert np.all(action >= env.action_space.low)
            assert np.all(action <= env.action_space.high)
            assert np.isfinite(lp.item())
            assert np.isfinite(v.item())
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()

    def test_gradient_flow(self, cppo_agent):
        """Gradients flow through all CPPO network parameters."""
        obs = torch.randn(8, cppo_agent.obs_dim)
        action, lp, ent, val = cppo_agent.network.get_action_and_value(obs)
        loss = lp.mean() + val.mean() + ent.mean()
        loss.backward()
        for name, param in cppo_agent.network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad: {name}"

    def test_update_returns_cvar_metrics(self, env, cppo_agent):
        """CPPO update returns CVaR-specific metric keys."""
        obs, _ = env.reset()
        cppo_agent.collect_rollout(env, obs)
        metrics = cppo_agent.update()

        expected_keys = {
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_fraction", "cvar", "var", "cvar_penalty",
            "lagrange_lambda", "constraint_violation",
        }
        assert expected_keys.issubset(set(metrics.keys()))
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_lagrange_increases_on_violation(self):
        """λ increases when CVaR violates the threshold."""
        from rcmm._rcmm_core import EnvConfig
        from rcmm.cppo import CPPOAgent, CPPOConfig
        from rcmm.env import LimitOrderBookEnv

        cfg = EnvConfig()
        cfg.max_steps = 100
        cfg.warmup_ticks = 30
        cfg.seed = 42

        env = LimitOrderBookEnv(config=cfg)
        cppo_cfg = CPPOConfig(
            hidden_dim=16, num_hidden=1,
            rollout_steps=64, num_minibatches=4, update_epochs=2,
            cvar_alpha=0.10,               # 10% tail
            cvar_threshold=1000.0,         # impossibly high → always violated
            lagrange_lr=0.1,
            lagrange_init=0.0,
            dual_update=True,
        )

        agent = CPPOAgent(env, cppo_cfg)
        initial_lambda = agent.lagrange_multiplier

        obs, _ = env.reset()
        agent.collect_rollout(env, obs)
        agent.update()

        # λ should increase because CVaR < threshold (always violated).
        assert agent.lagrange_multiplier > initial_lambda, (
            f"λ did not increase: {initial_lambda} → {agent.lagrange_multiplier}"
        )

    def test_lagrange_stays_zero_no_violation(self):
        """λ stays at 0 when CVaR satisfies the threshold."""
        from rcmm._rcmm_core import EnvConfig
        from rcmm.cppo import CPPOAgent, CPPOConfig
        from rcmm.env import LimitOrderBookEnv

        cfg = EnvConfig()
        cfg.max_steps = 100
        cfg.warmup_ticks = 30
        cfg.seed = 42

        env = LimitOrderBookEnv(config=cfg)
        cppo_cfg = CPPOConfig(
            hidden_dim=16, num_hidden=1,
            rollout_steps=64, num_minibatches=4, update_epochs=2,
            cvar_alpha=0.05,
            cvar_threshold=-1e9,    # impossible to violate
            lagrange_lr=0.1,
            lagrange_init=0.0,
            dual_update=True,
        )

        agent = CPPOAgent(env, cppo_cfg)
        obs, _ = env.reset()
        agent.collect_rollout(env, obs)
        agent.update()

        assert agent.lagrange_multiplier == 0.0, (
            f"λ should be 0 but is {agent.lagrange_multiplier}"
        )

    def test_update_modifies_params(self, env, cppo_agent):
        """CPPO update actually changes network parameters."""
        obs, _ = env.reset()
        before = {
            name: param.clone()
            for name, param in cppo_agent.network.named_parameters()
        }
        cppo_agent.collect_rollout(env, obs)
        cppo_agent.update()

        any_changed = any(
            not torch.allclose(before[n], p)
            for n, p in cppo_agent.network.named_parameters()
        )
        assert any_changed, "No parameters changed after CPPO update"


# ============================================================================
# CPPO Training Tests
# ============================================================================


class TestCPPOTraining:
    """End-to-end CPPO training tests."""

    def test_short_training_no_crash(self):
        """Full CPPO training loop completes without NaN."""
        from rcmm._rcmm_core import EnvConfig
        from rcmm.cppo import CPPOAgent, CPPOConfig
        from rcmm.env import LimitOrderBookEnv

        cfg = EnvConfig()
        cfg.max_steps = 100
        cfg.warmup_ticks = 30
        cfg.seed = 99

        env = LimitOrderBookEnv(config=cfg)
        cppo_cfg = CPPOConfig(
            hidden_dim=32, num_hidden=2,
            rollout_steps=64, num_minibatches=4, update_epochs=2,
            cvar_alpha=0.05, cvar_threshold=-5.0,
            lagrange_lr=0.01, lagrange_init=0.1,
        )

        agent = CPPOAgent(env, cppo_cfg)
        metrics = agent.train(env, total_timesteps=256, verbose=False)

        assert len(metrics) >= 1
        for m in metrics:
            for k, v in m.items():
                assert np.isfinite(v), f"NaN/inf in {k}: {v}"

    def test_training_with_regime_wrapper(self):
        """CPPO trains through regime-randomized environment without crash."""
        from rcmm._rcmm_core import EnvConfig
        from rcmm.cppo import CPPOAgent, CPPOConfig
        from rcmm.env import LimitOrderBookEnv
        from rcmm.regime_wrapper import RegimeRandomizationWrapper

        cfg = EnvConfig()
        cfg.max_steps = 60
        cfg.warmup_ticks = 20
        cfg.seed = 77

        env = LimitOrderBookEnv(config=cfg)
        wrapped = RegimeRandomizationWrapper(env, seed=77)

        cppo_cfg = CPPOConfig(
            hidden_dim=16, num_hidden=1,
            rollout_steps=64, num_minibatches=4, update_epochs=2,
            cvar_alpha=0.05, cvar_threshold=-5.0,
        )

        agent = CPPOAgent(wrapped, cppo_cfg)
        metrics = agent.train(wrapped, total_timesteps=256, verbose=False)

        assert len(metrics) >= 1
        for m in metrics:
            for k, v in m.items():
                assert np.isfinite(v), f"NaN/inf in {k}: {v}"

    def test_lagrange_trajectory(self):
        """Lagrange multiplier evolves over training (not stuck)."""
        from rcmm._rcmm_core import EnvConfig
        from rcmm.cppo import CPPOAgent, CPPOConfig
        from rcmm.env import LimitOrderBookEnv

        cfg = EnvConfig()
        cfg.max_steps = 100
        cfg.warmup_ticks = 30
        cfg.seed = 42

        env = LimitOrderBookEnv(config=cfg)
        cppo_cfg = CPPOConfig(
            hidden_dim=16, num_hidden=1,
            rollout_steps=64, num_minibatches=4, update_epochs=2,
            cvar_alpha=0.10, cvar_threshold=-1.0,  # likely violated
            lagrange_lr=0.05, lagrange_init=0.0,
            dual_update=True,
        )

        agent = CPPOAgent(env, cppo_cfg)
        metrics = agent.train(env, total_timesteps=512, verbose=False)

        lambdas = [m["lagrange_lambda"] for m in metrics]
        # With threshold likely violated, λ should have moved from 0.
        assert max(lambdas) > 0.0, "Lagrange multiplier never increased"


# ============================================================================
# Evaluation Helper Tests
# ============================================================================


class TestEvaluationHelpers:
    """Tests for max drawdown and Sharpe ratio helpers."""

    def test_max_drawdown_known(self):
        """Known PnL curve: [10, 12, 8, 15, 6] → max dd = 15 - 6 = 9."""
        from scripts.evaluate_regimes import compute_max_drawdown

        dd = compute_max_drawdown([10, 12, 8, 15, 6])
        assert abs(dd - 9.0) < 1e-10

    def test_max_drawdown_monotonic_up(self):
        """Monotonically increasing curve → zero drawdown."""
        from scripts.evaluate_regimes import compute_max_drawdown

        dd = compute_max_drawdown([1, 2, 3, 4, 5])
        assert dd == 0.0

    def test_max_drawdown_monotonic_down(self):
        """Monotonically decreasing → drawdown = first - last."""
        from scripts.evaluate_regimes import compute_max_drawdown

        dd = compute_max_drawdown([100, 80, 60, 40, 20])
        assert abs(dd - 80.0) < 1e-10

    def test_max_drawdown_empty(self):
        """Empty curve → 0."""
        from scripts.evaluate_regimes import compute_max_drawdown

        assert compute_max_drawdown([]) == 0.0

    def test_sharpe_ratio_positive(self):
        """Positive returns → positive Sharpe."""
        from scripts.evaluate_regimes import compute_sharpe

        returns = [10.0, 12.0, 8.0, 11.0, 9.0]
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_ratio_zero_std(self):
        """Constant returns → Sharpe = 0 (no risk taken)."""
        from scripts.evaluate_regimes import compute_sharpe

        assert compute_sharpe([5.0, 5.0, 5.0]) == 0.0

    def test_sharpe_ratio_single(self):
        """Single return → 0."""
        from scripts.evaluate_regimes import compute_sharpe

        assert compute_sharpe([10.0]) == 0.0


# ============================================================================
# HawkesParams Binding Tests
# ============================================================================


class TestHawkesParamsBinding:
    """Tests for the pybind11 HawkesParams/MarkConfig exposure."""

    def test_hawkes_params_roundtrip(self):
        """Set and read HawkesParams fields."""
        from rcmm._rcmm_core import HawkesParams

        hp = HawkesParams()
        hp.mu = 7.0
        hp.alpha = 3.0
        hp.beta = 8.0

        assert hp.mu == 7.0
        assert hp.alpha == 3.0
        assert hp.beta == 8.0
        assert hp.is_stationary()
        assert abs(hp.branching_ratio() - 3.0 / 8.0) < 1e-10

    def test_hawkes_params_on_envconfig(self):
        """HawkesParams can be set on EnvConfig."""
        from rcmm._rcmm_core import EnvConfig, HawkesParams

        hp = HawkesParams()
        hp.mu = 10.0
        hp.alpha = 4.0
        hp.beta = 5.0

        cfg = EnvConfig()
        cfg.hawkes_params = hp

        # Verify it stuck.
        assert cfg.hawkes_params.mu == 10.0
        assert cfg.hawkes_params.alpha == 4.0

    def test_mark_config_roundtrip(self):
        """Set and read MarkConfig fields."""
        from rcmm._rcmm_core import EnvConfig

        cfg = EnvConfig()
        cfg.mark_config.buy_prob = 0.2
        assert abs(cfg.mark_config.buy_prob - 0.2) < 1e-10

    def test_hawkes_repr(self):
        """HawkesParams has a readable __repr__."""
        from rcmm._rcmm_core import HawkesParams

        hp = HawkesParams()
        hp.mu = 5.0
        hp.alpha = 1.5
        hp.beta = 5.0
        r = repr(hp)
        assert "5.0" in r
        assert "1.5" in r


# ============================================================================
# CPPOConfig Tests
# ============================================================================


class TestCPPOConfig:
    """Tests for CPPOConfig defaults."""

    def test_inherits_ppo_defaults(self):
        """CPPOConfig has all PPO defaults plus CVaR params."""
        from rcmm.cppo import CPPOConfig

        cfg = CPPOConfig()
        # PPO defaults.
        assert cfg.hidden_dim == 64
        assert cfg.clip_eps == 0.2
        assert cfg.gamma == 0.99
        # CPPO-specific.
        assert cfg.cvar_alpha == 0.05
        assert cfg.dual_update is True
        assert cfg.lagrange_init == 0.1
        assert cfg.lagrange_max == 10.0
