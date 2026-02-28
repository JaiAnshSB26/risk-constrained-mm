# ============================================================================
# risk-constrained-mm :: python/rcmm/stats.py
# ============================================================================
"""
Statistical tests for comparing trading strategies.

Implements the **Diebold-Mariano (DM) test** for comparing predictive
accuracy (or trading performance) of two strategies.

Mathematical formulation:

    Let d_t = L(e_{1,t}) - L(e_{2,t}) be the loss differential at time t,
    where L is a loss function (e.g., negative PnL, squared error).

    For our use case: d_t = r_{CPPO,t} - r_{Baseline,t}
    (positive d_t means CPPO outperformed at step t)

    Under H0: E[d_t] = 0  (the two strategies perform equally)

    The DM statistic is:

        DM = d̄ / √(V̂(d̄))

    where:
        d̄   = (1/T) Σ d_t              (mean differential)
        V̂(d̄) = (1/T) Σ_{τ=-h}^{h} γ̂(τ)  (Newey-West HAC variance)
        γ̂(τ) = (1/T) Σ_{t=|τ|+1}^{T} (d_t - d̄)(d_{t-|τ|} - d̄)

    The truncation lag h controls how much autocorrelation is accounted for.
    Default: h = floor(T^{1/3}) (Diebold-Mariano recommendation).

    Under H0, DM ~ N(0,1) asymptotically.

    Two-sided p-value: p = 2 · Φ(-|DM|)

Reference:
    Diebold, F.X. & Mariano, R.S. (1995).
    "Comparing Predictive Accuracy."
    Journal of Business & Economic Statistics, 13(3), 253–263.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def diebold_mariano(
    pnl_1: np.ndarray | list[float],
    pnl_2: np.ndarray | list[float],
    *,
    h: int | None = None,
    one_sided: bool = False,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive / trading performance.

    Tests H0: E[d_t] = 0, where d_t = pnl_1[t] - pnl_2[t].
    A positive DM statistic means pnl_1 outperforms pnl_2 on average.

    Args:
        pnl_1: Step-level PnL (or returns) of strategy 1 (e.g. CPPO).
        pnl_2: Step-level PnL (or returns) of strategy 2 (e.g. baseline).
        h: Truncation lag for HAC variance estimator.
           Default: floor(T^{1/3}).
        one_sided: If True, test H1: E[d_t] > 0 (pnl_1 > pnl_2).
                   If False (default), two-sided test.

    Returns:
        (dm_stat, p_value): DM test statistic and p-value.

    Raises:
        ValueError: If inputs have different lengths or are too short.
    """
    d = np.asarray(pnl_1, dtype=np.float64) - np.asarray(pnl_2, dtype=np.float64)
    T = len(d)

    if T < 2:
        raise ValueError(f"Need at least 2 observations, got {T}")

    d_bar = d.mean()

    # Truncation lag.
    if h is None:
        h = max(1, int(np.floor(T ** (1.0 / 3.0))))

    # Compute autocovariance at lag 0..h.
    d_centered = d - d_bar

    def autocovariance(lag: int) -> float:
        """Autocovariance at the given lag."""
        if lag == 0:
            return float(np.dot(d_centered, d_centered) / T)
        return float(np.dot(d_centered[lag:], d_centered[:-lag]) / T)

    # Newey-West HAC variance of d̄:
    #   V̂(d̄) = (1/T) [γ̂(0) + 2 Σ_{τ=1}^{h} γ̂(τ)]
    gamma_0 = autocovariance(0)
    gamma_sum = sum(autocovariance(tau) for tau in range(1, h + 1))
    variance_d_bar = (gamma_0 + 2.0 * gamma_sum) / T

    # Guard against zero / negative variance (degenerate case).
    if variance_d_bar <= 0.0:
        # If variance is zero, all d_t are identical.
        if abs(d_bar) < 1e-15:
            return 0.0, 1.0  # identical series → no difference
        # Non-zero mean with zero variance → perfect signal.
        dm_stat = np.sign(d_bar) * np.inf
        return float(dm_stat), 0.0

    dm_stat = d_bar / np.sqrt(variance_d_bar)

    if one_sided:
        # H1: pnl_1 > pnl_2  ⟹  d̄ > 0  ⟹  right tail
        p_value = 1.0 - sp_stats.norm.cdf(dm_stat)
    else:
        p_value = 2.0 * sp_stats.norm.cdf(-abs(dm_stat))

    return float(dm_stat), float(p_value)
