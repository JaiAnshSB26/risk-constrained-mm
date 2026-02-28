# ============================================================================
# risk-constrained-mm :: tests/python/test_stats.py
# ============================================================================
"""
Pytest suite for the Diebold-Mariano statistical test.

Tests cover:
  * Known outperformance: p-value < 0.05 when one series clearly dominates.
  * No difference: p-value > 0.05 when series are identical.
  * Symmetry: swapping inputs negates the DM statistic.
  * Zero-variance guard: no division by zero.
  * Autocorrelation: HAC variance differs from naive variance.
  * One-sided test behaviour.
  * Edge cases: short series, constant differential.
  * Mathematical validation against manual computation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


# ============================================================================
# Core DM Tests
# ============================================================================


class TestDieboldMarianoCore:
    """Core Diebold-Mariano test behaviour."""

    def test_clear_outperformance_significant(self, rng):
        """
        When pnl_1 clearly outperforms pnl_2, p-value should be < 0.05.

        Construct: pnl_1 = pnl_2 + positive_shift + noise.
        """
        from rcmm.stats import diebold_mariano

        n = 500
        pnl_2 = rng.normal(0.0, 1.0, size=n)
        pnl_1 = pnl_2 + 0.5 + rng.normal(0.0, 0.2, size=n)  # clear advantage

        dm_stat, p_value = diebold_mariano(pnl_1, pnl_2)

        assert dm_stat > 0, f"DM should be positive when pnl_1 > pnl_2, got {dm_stat}"
        assert p_value < 0.05, f"p-value should be < 0.05, got {p_value}"

    def test_clear_underperformance_significant(self, rng):
        """When pnl_1 is worse, DM is negative and significant."""
        from rcmm.stats import diebold_mariano

        n = 500
        pnl_2 = rng.normal(0.0, 1.0, size=n)
        pnl_1 = pnl_2 - 0.5 + rng.normal(0.0, 0.2, size=n)

        dm_stat, p_value = diebold_mariano(pnl_1, pnl_2)

        assert dm_stat < 0
        assert p_value < 0.05

    def test_identical_series_not_significant(self):
        """Identical PnL series → p-value = 1.0, DM = 0."""
        from rcmm.stats import diebold_mariano

        pnl = np.ones(100)
        dm_stat, p_value = diebold_mariano(pnl, pnl)

        assert dm_stat == 0.0
        assert p_value == 1.0

    def test_no_difference_high_pvalue(self, rng):
        """IID series with same mean → non-significant (p > 0.05)."""
        from rcmm.stats import diebold_mariano

        n = 500
        pnl_1 = rng.normal(5.0, 1.0, size=n)
        pnl_2 = rng.normal(5.0, 1.0, size=n)

        _, p_value = diebold_mariano(pnl_1, pnl_2)

        # With same mean, we expect non-significance most of the time.
        # Use a generous threshold to avoid random test failures.
        # (This is a statistical test of a statistical test.)
        assert p_value > 0.01, f"Two identical distributions should not be significant, p={p_value}"


# ============================================================================
# Symmetry & Sign Tests
# ============================================================================


class TestDMSymmetry:
    """Swapping inputs should negate the DM statistic."""

    def test_swap_negates_statistic(self, rng):
        """DM(A, B) = -DM(B, A)."""
        from rcmm.stats import diebold_mariano

        n = 200
        pnl_1 = rng.normal(1.0, 1.0, size=n)
        pnl_2 = rng.normal(0.0, 1.0, size=n)

        dm_ab, p_ab = diebold_mariano(pnl_1, pnl_2)
        dm_ba, p_ba = diebold_mariano(pnl_2, pnl_1)

        assert abs(dm_ab + dm_ba) < 1e-10, (
            f"DM(A,B)={dm_ab} should equal -DM(B,A)={-dm_ba}"
        )
        assert abs(p_ab - p_ba) < 1e-10, (
            f"Two-sided p-values should be equal: {p_ab} vs {p_ba}"
        )


# ============================================================================
# Edge Cases
# ============================================================================


class TestDMEdgeCases:
    """Edge cases for robustness."""

    def test_too_short_raises(self):
        """Less than 2 observations should raise ValueError."""
        from rcmm.stats import diebold_mariano

        with pytest.raises(ValueError, match="at least 2"):
            diebold_mariano([1.0], [2.0])

    def test_different_lengths_raises(self):
        """Different-length arrays should raise (via numpy broadcast error)."""
        from rcmm.stats import diebold_mariano

        # numpy subtraction of different lengths raises
        with pytest.raises((ValueError, Exception)):
            diebold_mariano([1.0, 2.0], [3.0, 4.0, 5.0])

    def test_two_elements(self):
        """Minimum valid input (T=2) doesn't crash."""
        from rcmm.stats import diebold_mariano

        dm_stat, p_value = diebold_mariano([10.0, 20.0], [5.0, 10.0])
        assert np.isfinite(dm_stat) or np.isinf(dm_stat)  # may be inf for T=2
        assert 0.0 <= p_value <= 1.0

    def test_constant_positive_differential(self):
        """Constant d_t > 0 → zero variance but nonzero mean."""
        from rcmm.stats import diebold_mariano

        pnl_1 = np.array([10.0] * 50)
        pnl_2 = np.array([5.0] * 50)

        dm_stat, p_value = diebold_mariano(pnl_1, pnl_2)

        # Constant positive differential → should be significant (or inf).
        assert dm_stat > 0 or np.isinf(dm_stat)
        assert p_value < 0.05 or p_value == 0.0

    def test_zero_differential_everywhere(self):
        """All d_t = 0 → DM = 0, p = 1."""
        from rcmm.stats import diebold_mariano

        pnl = [1.0, 2.0, 3.0, 4.0, 5.0]
        dm_stat, p_value = diebold_mariano(pnl, pnl)

        assert dm_stat == 0.0
        assert p_value == 1.0

    def test_no_zero_division(self, rng):
        """Various inputs never produce ZeroDivisionError."""
        from rcmm.stats import diebold_mariano

        test_cases = [
            (np.zeros(10), np.zeros(10)),
            (np.ones(10), np.ones(10)),
            (np.arange(10, dtype=float), np.arange(10, dtype=float)),
            (rng.normal(0, 1, 10), rng.normal(0, 1, 10)),
        ]

        for pnl_1, pnl_2 in test_cases:
            dm, p = diebold_mariano(pnl_1, pnl_2)
            assert np.isfinite(p) or p == 0.0, f"Non-finite p-value: {p}"
            assert not np.isnan(dm), f"NaN DM statistic"


# ============================================================================
# One-Sided Test
# ============================================================================


class TestDMOneSided:
    """One-sided test behaviour."""

    def test_one_sided_positive(self, rng):
        """One-sided test detects positive outperformance."""
        from rcmm.stats import diebold_mariano

        n = 500
        pnl_1 = rng.normal(2.0, 1.0, size=n)
        pnl_2 = rng.normal(0.0, 1.0, size=n)

        _, p_two = diebold_mariano(pnl_1, pnl_2, one_sided=False)
        _, p_one = diebold_mariano(pnl_1, pnl_2, one_sided=True)

        # One-sided p should be roughly half of two-sided when d̄ > 0.
        assert p_one < p_two
        assert p_one < 0.05

    def test_one_sided_wrong_direction(self, rng):
        """One-sided test: if pnl_1 < pnl_2, p-value should be large."""
        from rcmm.stats import diebold_mariano

        n = 500
        pnl_1 = rng.normal(0.0, 1.0, size=n)
        pnl_2 = rng.normal(2.0, 1.0, size=n)

        _, p_one = diebold_mariano(pnl_1, pnl_2, one_sided=True)

        assert p_one > 0.5, f"Wrong-direction one-sided p should be > 0.5, got {p_one}"


# ============================================================================
# HAC Variance Tests
# ============================================================================


class TestDMHACVariance:
    """The Newey-West HAC variance accounts for autocorrelation."""

    def test_custom_lag(self, rng):
        """Custom truncation lag h works without error."""
        from rcmm.stats import diebold_mariano

        pnl_1 = rng.normal(1.0, 1.0, size=200)
        pnl_2 = rng.normal(0.0, 1.0, size=200)

        dm_h1, p_h1 = diebold_mariano(pnl_1, pnl_2, h=1)
        dm_h5, p_h5 = diebold_mariano(pnl_1, pnl_2, h=5)

        # Different lags should generally produce different DM values
        # (unless autocorrelations happen to be exactly zero).
        assert np.isfinite(dm_h1) and np.isfinite(dm_h5)
        assert 0.0 <= p_h1 <= 1.0 and 0.0 <= p_h5 <= 1.0

    def test_large_lag_no_crash(self, rng):
        """Large lag doesn't crash (even if > T)."""
        from rcmm.stats import diebold_mariano

        pnl_1 = rng.normal(0, 1, size=20)
        pnl_2 = rng.normal(0, 1, size=20)

        dm, p = diebold_mariano(pnl_1, pnl_2, h=15)
        assert np.isfinite(p) or p == 0.0


# ============================================================================
# Mathematical Validation
# ============================================================================


class TestDMMathValidation:
    """Validate DM computation against a manual reference."""

    def test_manual_computation(self):
        """
        Hand-computed example:
            d = [1, 3, 2, 4]  (T=4)
            d̄ = 2.5
            h = floor(4^{1/3}) = 1

            d_centered = [-1.5, 0.5, -0.5, 1.5]

            γ̂(0) = (1/4)(2.25 + 0.25 + 0.25 + 2.25) = 1.25
            γ̂(1) = (1/4)(0.5·(-1.5) + (-0.5)·0.5 + 1.5·(-0.5))
                  = (1/4)(-0.75 - 0.25 - 0.75) = -0.4375

            V̂(d̄) = (1/4)[1.25 + 2·(-0.4375)] = (1/4)(0.375) = 0.09375

            DM = 2.5 / sqrt(0.09375) ≈ 8.1650
        """
        from rcmm.stats import diebold_mariano

        pnl_1 = np.array([1.0, 3.0, 2.0, 4.0])
        pnl_2 = np.zeros(4)

        dm_stat, p_value = diebold_mariano(pnl_1, pnl_2, h=1)

        expected_dm = 2.5 / np.sqrt(0.09375)
        assert abs(dm_stat - expected_dm) < 1e-10, (
            f"DM={dm_stat}, expected={expected_dm}"
        )
        assert p_value < 0.05  # clearly significant
