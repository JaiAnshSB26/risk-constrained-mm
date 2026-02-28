#!/usr/bin/env python3
"""Publication-quality plots for the Risk-Constrained Market-Making paper.

Reads  ``results/oos_trajectories.csv``  (produced by evaluate_oos.py)
and writes three 300-DPI PNGs into ``results/plots/``:

1. **pnl_comparison.png**        – cumulative PnL time-series
2. **drawdown_comparison.png**   – underwater / drawdown fill-between
3. **reward_distribution.png**   – step-return histogram + KDE with CVaR
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.15,
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.edgecolor": "0.15",
        "grid.color": "0.9",
        "font.family": "serif",
    },
)

CPPO_COLOR = "#2563EB"      # blue-600
BASELINE_COLOR = "#DC2626"  # red-600
FILL_ALPHA = 0.30
DPI = 300

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "oos_trajectories.csv"
PLOT_DIR = ROOT / "results" / "plots"


def _load() -> pd.DataFrame:
    """Load the OOS trajectory CSV."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"OOS CSV not found at {CSV_PATH}.\n"
            "Run `python python/scripts/evaluate_oos.py` first."
        )
    return pd.read_csv(CSV_PATH)


def _thousands_fmt(x: float, _pos: int) -> str:
    """Format axis ticks as e.g. '10 k'."""
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


# ── Plot 1: Cumulative PnL ──────────────────────────────────────────
def plot_pnl(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(
        df["step"], df["cppo_cumulative_pnl"],
        color=CPPO_COLOR, linewidth=1.2, label="CPPO Agent",
    )
    ax.plot(
        df["step"], df["baseline_cumulative_pnl"],
        color=BASELINE_COLOR, linewidth=1.2, label="A-S Baseline",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative PnL")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands_fmt))
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6", loc="best")
    ax.set_title("Out-of-Sample Cumulative PnL", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = PLOT_DIR / "pnl_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


# ── Plot 2: Drawdown ────────────────────────────────────────────────
def _drawdown(cumulative: pd.Series) -> pd.Series:
    """Compute underwater curve: DD_t = PnL_t − max_{s≤t} PnL_s."""
    running_max = cumulative.cummax()
    return cumulative - running_max


def plot_drawdown(df: pd.DataFrame) -> None:
    cppo_dd = _drawdown(df["cppo_cumulative_pnl"])
    base_dd = _drawdown(df["baseline_cumulative_pnl"])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.fill_between(
        df["step"], cppo_dd, 0,
        color=CPPO_COLOR, alpha=FILL_ALPHA, label="CPPO",
    )
    ax.plot(df["step"], cppo_dd, color=CPPO_COLOR, linewidth=0.8)

    ax.fill_between(
        df["step"], base_dd, 0,
        color=BASELINE_COLOR, alpha=FILL_ALPHA, label="A-S Baseline",
    )
    ax.plot(df["step"], base_dd, color=BASELINE_COLOR, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands_fmt))
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6", loc="best")
    ax.set_title("Underwater / Drawdown Curve", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = PLOT_DIR / "drawdown_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


# ── Plot 3: Reward Distribution ─────────────────────────────────────
def plot_reward_dist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Histogram + KDE
    sns.histplot(
        df["cppo_reward"], bins=50, kde=True, stat="density",
        color=CPPO_COLOR, alpha=0.45, label="CPPO", ax=ax,
        edgecolor="white", linewidth=0.4,
    )
    sns.histplot(
        df["baseline_reward"], bins=50, kde=True, stat="density",
        color=BASELINE_COLOR, alpha=0.45, label="A-S Baseline", ax=ax,
        edgecolor="white", linewidth=0.4,
    )

    # CVaR-5 % vertical lines  (expected shortfall below 5th percentile)
    alpha = 0.05
    for rewards, color, name in [
        (df["cppo_reward"], CPPO_COLOR, "CPPO"),
        (df["baseline_reward"], BASELINE_COLOR, "Baseline"),
    ]:
        var = np.percentile(rewards, alpha * 100)
        cvar = rewards[rewards <= var].mean()
        ax.axvline(
            cvar, color=color, linestyle="--", linewidth=1.4,
            label=f"{name} CVaR₅% = {cvar:,.0f}",
        )

    ax.set_xlabel("Step Reward")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_thousands_fmt))
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6", fontsize=8, loc="best")
    ax.set_title("Reward Distribution with CVaR₅%", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = PLOT_DIR / "reward_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    print(f"Loaded {len(df)} OOS steps from {CSV_PATH.relative_to(ROOT)}")

    plot_pnl(df)
    plot_drawdown(df)
    plot_reward_dist(df)

    print("\nAll plots saved to results/plots/")


if __name__ == "__main__":
    main()
