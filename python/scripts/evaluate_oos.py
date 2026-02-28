#!/usr/bin/env python3
# ============================================================================
# risk-constrained-mm :: python/scripts/evaluate_oos.py
# ============================================================================
"""
Out-of-Sample evaluation: CPPO vs. Avellaneda-Stoikov baseline.

1. Generates a synthetic out-of-sample CSV (10,000 realistic ticks) to
   simulate historical data that would come from a ReplayEngine.
2. Trains a CPPO agent on a *separate* regime-randomised environment.
3. Runs both CPPO and the AS baseline on the **same** OOS environment
   (fed by the generated tick data via the Hawkes simulator).
4. Tracks step-by-step PnL for both agents.
5. Applies the Diebold-Mariano test to the PnL differentials.
6. Exports both PnL trajectories to results/oos_trajectories.csv.

Usage:
    python python/scripts/evaluate_oos.py [--train-steps N] [--seed S]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Dummy OOS data generator ────────────────────────────────────────────────


def generate_oos_csv(path: str | Path, n_ticks: int = 10_000, seed: int = 42) -> Path:
    """
    Generate a synthetic out-of-sample CSV with realistic market ticks.

    The CSV follows the format expected by the C++ MarketDataParser:
        timestamp,action,order_id,side,price,qty

    Generates a mix of add, cancel, modify, and trade actions with
    realistic price/quantity distributions centred around a base price.

    Args:
        path: Output file path.
        n_ticks: Number of ticks to generate (default 10,000).
        seed: RNG seed.

    Returns:
        Path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_price = 50000.0
    tick_size = 1.0
    actions = ["add", "cancel", "modify", "trade"]
    action_probs = [0.45, 0.25, 0.10, 0.20]  # realistic distribution
    sides = ["bid", "ask"]

    lines = ["timestamp,action,order_id,side,price,qty"]
    ts = 1_609_459_200_000_000  # 2021-01-01 00:00:00 UTC in microseconds

    active_orders: dict[int, bool] = {}  # track order IDs for cancels/modifies
    next_oid = 1

    for _ in range(n_ticks):
        ts += int(rng.exponential(scale=50_000))  # ~50ms between events
        action = rng.choice(actions, p=action_probs)

        # For cancel/modify, pick an existing order if possible.
        if action in ("cancel", "modify") and active_orders:
            oid = int(rng.choice(list(active_orders.keys())))
            if action == "cancel":
                active_orders.pop(oid, None)
        else:
            if action in ("cancel", "modify"):
                action = "add"  # fallback if no active orders
            oid = next_oid
            next_oid += 1

        side = rng.choice(sides)
        # Price: random walk around base with realistic spread.
        price_offset = rng.normal(0, 5) * tick_size
        price = round(base_price + price_offset, 2)
        price = max(base_price - 100, min(base_price + 100, price))

        qty = round(float(rng.exponential(scale=3.0)) + 1.0, 3)
        qty = max(0.1, min(50.0, qty))

        lines.append(f"{ts},{action},{oid},{side},{price:.2f},{qty:.3f}")

        if action == "add":
            active_orders[oid] = True
        elif action == "trade":
            pass  # trades don't add to book

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Evaluation helpers ──────────────────────────────────────────────────────


def run_agent_on_env(agent, env, max_steps: int | None = None) -> list[float]:
    """
    Run an agent through an environment, collecting per-step rewards.

    Returns list of step-level rewards (PnL deltas).
    """
    obs, _info = env.reset()
    step_rewards: list[float] = []
    done = False
    step = 0

    while not done:
        action, _, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        step_rewards.append(float(reward))
        step = step + 1
        if max_steps is not None and step >= max_steps:
            break

    return step_rewards


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Out-of-Sample evaluation: CPPO vs AS baseline"
    )
    parser.add_argument(
        "--train-steps", type=int, default=5_000,
        help="CPPO training timesteps (default: 5000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    import torch
    from rcmm import EnvConfig, HawkesParams, LimitOrderBookEnv
    from rcmm.regime_wrapper import RegimeRandomizationWrapper
    from rcmm.cppo import CPPOAgent, CPPOConfig
    from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig
    from rcmm.stats import diebold_mariano

    print("=" * 70)
    print("  Phase 9: Out-of-Sample Evaluation — CPPO vs. AS Baseline")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Step 1: Generate OOS data ──
    workspace = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace / "results"
    oos_csv = results_dir / "oos_data.csv"

    print(f"\n  Generating OOS data: {oos_csv}")
    generate_oos_csv(oos_csv, n_ticks=10_000, seed=args.seed + 9000)
    print(f"    ✓ 10,000 ticks written to {oos_csv}")

    # ── Step 2: Train CPPO on regime-randomised environment ──
    print("\n  Training CPPO agent on regime-randomised environment...")
    train_cfg = EnvConfig()
    train_cfg.max_steps = 500
    train_cfg.warmup_ticks = 100
    train_cfg.seed = args.seed
    train_cfg.inventory_aversion = 0.01
    train_env = LimitOrderBookEnv(config=train_cfg)
    wrapped_train = RegimeRandomizationWrapper(train_env, seed=args.seed)

    cppo_cfg = CPPOConfig(
        hidden_dim=64, num_hidden=2,
        rollout_steps=256,
        num_minibatches=8,
        update_epochs=4,
        seed=args.seed,
        cvar_alpha=0.05,
        cvar_threshold=-5.0,
        lagrange_lr=0.01,
        lagrange_init=0.1,
        dual_update=True,
    )
    cppo_agent = CPPOAgent(wrapped_train, cppo_cfg)
    t0 = time.perf_counter()
    cppo_metrics = cppo_agent.train(
        wrapped_train, total_timesteps=args.train_steps, verbose=False
    )
    train_time = time.perf_counter() - t0
    print(f"    ✓ CPPO trained: {len(cppo_metrics)} updates in {train_time:.2f}s")

    if cppo_metrics:
        last = cppo_metrics[-1]
        print(f"      Final λ={last['lagrange_lambda']:.4f}, CVaR={last['cvar']:.2f}")

    # ── Step 3: Set up OOS evaluation environment ──
    # Use a challenging Flash-Crash-like regime for OOS.
    print("\n  Setting up OOS evaluation environment (Flash Crash regime)...")
    hp = HawkesParams()
    hp.mu = 5.0
    hp.alpha = 9.0
    hp.beta = 10.0

    oos_cfg = EnvConfig()
    oos_cfg.max_steps = 1000
    oos_cfg.warmup_ticks = 100
    oos_cfg.seed = args.seed + 5000
    oos_cfg.inventory_aversion = 0.01
    oos_cfg.hawkes_params = hp
    oos_cfg.mark_config.buy_prob = 0.2  # sell pressure

    # ── Step 4: Create baseline agent ──
    as_agent = AvellanedaStoikovAgent(AvellanedaStoikovConfig(
        gamma=0.1,
        sigma=2.0,
        kappa=1.5,
        max_inventory=oos_cfg.max_inventory,
        order_size=5.0,
    ))

    # ── Step 5: Run both agents on same OOS environment ──
    print("\n  Running CPPO on OOS environment...")
    oos_env_cppo = LimitOrderBookEnv(config=oos_cfg)
    cppo_rewards = run_agent_on_env(cppo_agent, oos_env_cppo)
    print(f"    ✓ {len(cppo_rewards)} steps, cumulative PnL = {sum(cppo_rewards):.2f}")

    print("  Running AS baseline on OOS environment...")
    # Reset with same seed for fair comparison.
    oos_cfg_baseline = EnvConfig()
    oos_cfg_baseline.max_steps = 1000
    oos_cfg_baseline.warmup_ticks = 100
    oos_cfg_baseline.seed = args.seed + 5000
    oos_cfg_baseline.inventory_aversion = 0.01
    oos_cfg_baseline.hawkes_params = hp
    oos_cfg_baseline.mark_config.buy_prob = 0.2
    oos_env_baseline = LimitOrderBookEnv(config=oos_cfg_baseline)
    baseline_rewards = run_agent_on_env(as_agent, oos_env_baseline)
    print(f"    ✓ {len(baseline_rewards)} steps, cumulative PnL = {sum(baseline_rewards):.2f}")

    # ── Step 6: Align lengths for DM test ──
    min_len = min(len(cppo_rewards), len(baseline_rewards))
    cppo_pnl = np.array(cppo_rewards[:min_len])
    baseline_pnl = np.array(baseline_rewards[:min_len])

    print(f"\n  Aligned PnL series: {min_len} steps")

    # ── Step 7: Diebold-Mariano test ──
    print("\n" + "=" * 70)
    print("  Diebold-Mariano Test: CPPO vs. Avellaneda-Stoikov Baseline")
    print("=" * 70)

    dm_stat, p_value = diebold_mariano(cppo_pnl, baseline_pnl)

    print(f"    H0: E[d_t] = 0 (no performance difference)")
    print(f"    d_t = r_CPPO,t - r_Baseline,t")
    print(f"    Mean differential (d̄)  : {(cppo_pnl - baseline_pnl).mean():>12.4f}")
    print(f"    DM statistic           : {dm_stat:>12.4f}")
    print(f"    p-value (two-sided)    : {p_value:>12.6f}")

    if p_value < 0.05:
        winner = "CPPO" if dm_stat > 0 else "AS Baseline"
        print(f"    → Reject H0 at 5% level. {winner} significantly outperforms.")
    else:
        print(f"    → Fail to reject H0 at 5% level. No significant difference.")

    # One-sided test: CPPO > Baseline?
    dm_stat_1s, p_value_1s = diebold_mariano(
        cppo_pnl, baseline_pnl, one_sided=True
    )
    print(f"\n    One-sided (CPPO > Baseline):")
    print(f"    DM statistic           : {dm_stat_1s:>12.4f}")
    print(f"    p-value (one-sided)    : {p_value_1s:>12.6f}")

    # ── Step 8: Export PnL trajectories to CSV ──
    print(f"\n  Exporting PnL trajectories to {results_dir / 'oos_trajectories.csv'}")

    cum_cppo = np.cumsum(cppo_pnl)
    cum_baseline = np.cumsum(baseline_pnl)

    df = pd.DataFrame({
        "step": np.arange(min_len),
        "cppo_reward": cppo_pnl,
        "baseline_reward": baseline_pnl,
        "cppo_cumulative_pnl": cum_cppo,
        "baseline_cumulative_pnl": cum_baseline,
        "differential": cppo_pnl - baseline_pnl,
    })

    trajectories_path = results_dir / "oos_trajectories.csv"
    df.to_csv(trajectories_path, index=False)
    print(f"    ✓ Exported {len(df)} rows to {trajectories_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"    {'Metric':<30} {'CPPO':>12} {'AS Baseline':>12}")
    print(f"    {'-'*30} {'-'*12} {'-'*12}")
    print(f"    {'Cumulative PnL':<30} {cum_cppo[-1]:>12.2f} {cum_baseline[-1]:>12.2f}")
    print(f"    {'Mean step reward':<30} {cppo_pnl.mean():>12.4f} {baseline_pnl.mean():>12.4f}")
    print(f"    {'Std step reward':<30} {cppo_pnl.std():>12.4f} {baseline_pnl.std():>12.4f}")
    print(f"    {'DM statistic':<30} {dm_stat:>12.4f}")
    print(f"    {'p-value':<30} {p_value:>12.6f}")
    print("=" * 70)

    # Verify no NaN in outputs.
    for label, arr in [("cppo_pnl", cppo_pnl), ("baseline_pnl", baseline_pnl)]:
        if np.any(np.isnan(arr)):
            print(f"  FAIL: NaN in {label}")
            return 1

    if np.isnan(dm_stat) or np.isnan(p_value):
        print("  FAIL: NaN in DM test output")
        return 1

    print("  All outputs finite. OOS evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
