#!/usr/bin/env python3
# ============================================================================
# risk-constrained-mm :: python/scripts/evaluate_regimes.py
# ============================================================================
"""
Ablation / evaluation script: Baseline PPO vs. CPPO on Flash Crash regime.

Trains both agents for a short run, then evaluates exclusively on the
Flash Crash regime for N episodes.  Reports:
  - Cumulative PnL per episode
  - Max drawdown per episode
  - Sharpe ratio across episodes
  - CVaR of episode returns

Usage:
    python python/scripts/evaluate_regimes.py [--train-steps N] [--eval-episodes K]

Theory: The CPPO agent, trained with CVaR tail-risk penalty, should exhibit
smaller max drawdown and more stable returns under flash crash conditions
compared to the unconstrained baseline PPO.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def compute_max_drawdown(pnl_curve: list[float]) -> float:
    """
    Compute the maximum drawdown from a PnL curve.

    Max drawdown = max over t of (peak_up_to_t - curve[t]).
    """
    if not pnl_curve:
        return 0.0
    peak = pnl_curve[0]
    max_dd = 0.0
    for val in pnl_curve:
        peak = max(peak, val)
        dd = peak - val
        max_dd = max(max_dd, dd)
    return max_dd


def compute_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    """Sharpe ratio of episode returns."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free
    std = excess.std()
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std)


def evaluate_agent(agent, env, num_episodes: int, label: str) -> dict:
    """
    Evaluate an agent on `num_episodes` Flash Crash episodes.

    Returns a dict with per-episode PnL, max drawdown, and aggregate stats.
    """
    episode_returns: list[float] = []
    episode_drawdowns: list[float] = []
    all_step_rewards: list[float] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        pnl_curve = [0.0]
        ep_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            pnl_curve.append(pnl_curve[-1] + reward)
            all_step_rewards.append(reward)

        episode_returns.append(ep_reward)
        episode_drawdowns.append(compute_max_drawdown(pnl_curve))

    avg_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))
    avg_drawdown = float(np.mean(episode_drawdowns))
    max_drawdown = float(np.max(episode_drawdowns))
    sharpe = compute_sharpe(episode_returns)

    # CVaR of episode returns (bottom 20% given small sample).
    sorted_rets = sorted(episode_returns)
    tail_k = max(1, len(sorted_rets) // 5)
    cvar_episodes = float(np.mean(sorted_rets[:tail_k]))

    print(f"\n  [{label}] Evaluation Results ({num_episodes} episodes):")
    print(f"    Avg Return     : {avg_return:>12.2f} ± {std_return:.2f}")
    print(f"    Sharpe Ratio   : {sharpe:>12.4f}")
    print(f"    Avg Drawdown   : {avg_drawdown:>12.2f}")
    print(f"    Max Drawdown   : {max_drawdown:>12.2f}")
    print(f"    CVaR (20%)     : {cvar_episodes:>12.2f}")

    return {
        "label": label,
        "episode_returns": episode_returns,
        "episode_drawdowns": episode_drawdowns,
        "avg_return": avg_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
        "cvar": cvar_episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO vs CPPO on Flash Crash regime"
    )
    parser.add_argument(
        "--train-steps", type=int, default=5_000,
        help="Training timesteps per agent (default: 5000)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Flash Crash evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # ── Imports ──
    import torch

    from rcmm import (
        EnvConfig, HawkesParams, LimitOrderBookEnv,
        RegimeRandomizationWrapper,
        NORMAL_REGIME_SPEC, FLASH_CRASH_REGIME_SPEC,
    )
    from rcmm.ppo import PPOAgent, PPOConfig
    from rcmm.cppo import CPPOAgent, CPPOConfig

    print("=" * 70)
    print("  Phase 8 Evaluation: Baseline PPO vs. CPPO (Flash Crash)")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Training environments (with regime randomization) ──
    def make_train_env(seed: int) -> RegimeRandomizationWrapper:
        cfg = EnvConfig()
        cfg.max_steps = 500
        cfg.warmup_ticks = 100
        cfg.seed = seed
        cfg.inventory_aversion = 0.01
        env = LimitOrderBookEnv(config=cfg)
        return RegimeRandomizationWrapper(env, seed=seed)

    # ── Flash Crash evaluation environment (fixed regime) ──
    def make_flash_crash_env(seed: int) -> LimitOrderBookEnv:
        hp = HawkesParams()
        hp.mu = 5.0
        hp.alpha = 9.5
        hp.beta = 10.0

        cfg = EnvConfig()
        cfg.max_steps = 500
        cfg.warmup_ticks = 100
        cfg.seed = seed
        cfg.inventory_aversion = 0.01
        cfg.hawkes_params = hp

        # Flash crash: heavy sell pressure.
        cfg.mark_config.buy_prob = 0.15
        return LimitOrderBookEnv(config=cfg)

    rollout_steps = 256

    # ── Train Baseline PPO ──
    print("\n  Training Baseline PPO...")
    train_env_ppo = make_train_env(seed=args.seed)
    ppo_cfg = PPOConfig(
        hidden_dim=64, num_hidden=2,
        rollout_steps=rollout_steps,
        num_minibatches=8,
        update_epochs=4,
        seed=args.seed,
    )
    ppo_agent = PPOAgent(train_env_ppo, ppo_cfg)
    t0 = time.perf_counter()
    ppo_metrics = ppo_agent.train(
        train_env_ppo, total_timesteps=args.train_steps, verbose=False
    )
    ppo_time = time.perf_counter() - t0
    print(f"    PPO trained: {len(ppo_metrics)} updates in {ppo_time:.2f}s")

    # ── Train CPPO ──
    print("\n  Training CPPO (CVaR-constrained)...")
    train_env_cppo = make_train_env(seed=args.seed + 1000)
    cppo_cfg = CPPOConfig(
        hidden_dim=64, num_hidden=2,
        rollout_steps=rollout_steps,
        num_minibatches=8,
        update_epochs=4,
        seed=args.seed,
        cvar_alpha=0.05,
        cvar_threshold=-5.0,
        lagrange_lr=0.01,
        lagrange_init=0.1,
        dual_update=True,
    )
    cppo_agent = CPPOAgent(train_env_cppo, cppo_cfg)
    t0 = time.perf_counter()
    cppo_metrics = cppo_agent.train(
        train_env_cppo, total_timesteps=args.train_steps, verbose=False
    )
    cppo_time = time.perf_counter() - t0
    print(f"    CPPO trained: {len(cppo_metrics)} updates in {cppo_time:.2f}s")

    if cppo_metrics:
        last = cppo_metrics[-1]
        print(f"    Final λ={last['lagrange_lambda']:.4f}, "
              f"CVaR={last['cvar']:.2f}")

    # ── Evaluate both on Flash Crash ──
    print("\n" + "=" * 70)
    print("  Flash Crash Evaluation")
    print("=" * 70)

    eval_env = make_flash_crash_env(seed=args.seed + 5000)

    ppo_results = evaluate_agent(
        ppo_agent, eval_env, args.eval_episodes, "Baseline PPO"
    )

    cppo_results = evaluate_agent(
        cppo_agent, eval_env, args.eval_episodes, "CPPO"
    )

    # ── Comparison Summary ──
    print("\n" + "=" * 70)
    print("  Comparison Summary")
    print("=" * 70)
    print(f"  {'Metric':<20} {'PPO':>12} {'CPPO':>12} {'Delta':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    for key in ["avg_return", "sharpe", "avg_drawdown", "max_drawdown", "cvar"]:
        pv = ppo_results[key]
        cv = cppo_results[key]
        delta = cv - pv
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<20} {pv:>12.4f} {cv:>12.4f} {sign}{delta:>11.4f}")

    print("=" * 70)

    # ── Sanity checks ──
    for label, metrics in [("PPO", ppo_metrics), ("CPPO", cppo_metrics)]:
        for m in metrics:
            for k, v in m.items():
                if isinstance(v, float) and np.isnan(v):
                    print(f"  FAIL: NaN in {label} metric {k}")
                    return 1

    print("  All metrics finite. Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
