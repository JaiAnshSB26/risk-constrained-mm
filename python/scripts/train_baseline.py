#!/usr/bin/env python3
# ============================================================================
# risk-constrained-mm :: python/scripts/train_baseline.py
# ============================================================================
"""
Baseline PPO training script for the market-making environment.

Usage:
    python python/scripts/train_baseline.py [--timesteps N] [--gamma G] [--seed S]

This script:
  1. Initialises a LimitOrderBookEnv with the "Normal" Hawkes regime.
  2. Creates a PPO agent with default hyperparameters.
  3. Trains for a short run (default 10,000 timesteps).
  4. Prints training metrics per update.
  5. Verifies no memory leak via tracemalloc.

This is a smoke-test / sanity-check script to ensure the full
forward/backward pipeline works end-to-end before longer training runs.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train baseline PPO agent on LOB environment"
    )
    parser.add_argument(
        "--timesteps", type=int, default=10_000,
        help="Total training timesteps (default: 10000)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.01,
        help="Inventory aversion γ (default: 0.01)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=256,
        help="Steps per rollout (default: 256)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Hidden layer dimension (default: 64)"
    )
    args = parser.parse_args()

    # ── Imports (after arg parsing for fast --help) ──
    import torch
    from rcmm import EnvConfig, LimitOrderBookEnv
    from rcmm.ppo import PPOAgent, PPOConfig

    print("=" * 70)
    print("  Baseline PPO Training — Risk-Constrained Market Making")
    print("=" * 70)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device          : cpu")
    print(f"  Timesteps       : {args.timesteps}")
    print(f"  Inventory γ     : {args.gamma}")
    print(f"  Seed            : {args.seed}")
    print(f"  Rollout steps   : {args.rollout_steps}")
    print(f"  Hidden dim      : {args.hidden_dim}")
    print("=" * 70)

    # ── Seed everything ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Environment ──
    env_cfg = EnvConfig()
    env_cfg.seed = args.seed
    env_cfg.max_steps = 5000  # shorter episodes for baseline training
    env_cfg.inventory_aversion = args.gamma

    env = LimitOrderBookEnv(config=env_cfg)
    print(f"\n  Observation dim : {env.obs_dim}")
    print(f"  Action dim      : {env.action_space.shape[0]}")
    print(f"  Action bounds   : low={env.action_space.low}, high={env.action_space.high}")

    # ── PPO Agent ──
    ppo_cfg = PPOConfig(
        hidden_dim=args.hidden_dim,
        num_hidden=2,
        learning_rate=3e-4,
        rollout_steps=args.rollout_steps,
        num_minibatches=min(8, args.rollout_steps),  # ensure valid batch size
        update_epochs=4,
        seed=args.seed,
    )

    agent = PPOAgent(env, ppo_cfg)

    param_count = sum(p.numel() for p in agent.network.parameters())
    print(f"  Network params  : {param_count:,}")
    print()

    # ── Memory tracking ──
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()

    # ── Training ──
    t0 = time.perf_counter()
    metrics = agent.train(
        env,
        total_timesteps=args.timesteps,
        log_interval=1,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    mem_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Summary ──
    total_steps = int(metrics[-1]["timesteps"]) if metrics else 0
    sps = total_steps / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print(f"  Total updates   : {len(metrics)}")
    print(f"  Total timesteps : {total_steps}")
    print(f"  Wall time       : {elapsed:.2f}s")
    print(f"  Steps/sec       : {sps:,.0f}")
    print(f"  Memory (before) : {mem_before[0] / 1024:.1f} KB current, "
          f"{mem_before[1] / 1024:.1f} KB peak")
    print(f"  Memory (after)  : {mem_after[0] / 1024:.1f} KB current, "
          f"{mem_after[1] / 1024:.1f} KB peak")
    mem_growth = (mem_after[0] - mem_before[0]) / 1024
    print(f"  Memory growth   : {mem_growth:.1f} KB")

    if metrics:
        last = metrics[-1]
        print(f"\n  Final metrics:")
        print(f"    Policy loss   : {last['policy_loss']:.6f}")
        print(f"    Value loss    : {last['value_loss']:.6f}")
        print(f"    Entropy       : {last['entropy']:.4f}")
        print(f"    Approx KL     : {last['approx_kl']:.6f}")
        print(f"    Clip fraction : {last['clip_fraction']:.4f}")

    # ── Sanity checks ──
    print("\n  Sanity checks:")

    # 1. No NaN in metrics
    all_ok = True
    for m in metrics:
        for k, v in m.items():
            if np.isnan(v):
                print(f"    FAIL: NaN detected in {k}")
                all_ok = False

    # 2. Memory growth should be bounded (< 50 MB for 10k steps)
    if mem_growth > 50_000:  # 50 MB
        print(f"    WARN: Memory growth {mem_growth:.0f} KB seems high")
        all_ok = False

    if all_ok:
        print("    All checks PASSED ✓")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
