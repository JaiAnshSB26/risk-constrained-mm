// ============================================================================
// risk-constrained-mm :: cpp/include/sim/hawkes_params.hpp
// ============================================================================
// Parameter struct and regime presets for the 1D Hawkes process simulator.
//
// The intensity function is:
//   lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
//
// Stationarity requires alpha < beta (branching ratio alpha/beta < 1).
// ============================================================================
#pragma once

#include <cassert>

namespace rcmm {

/// Parameters for a 1D exponential-kernel Hawkes process.
struct HawkesParams {
    double mu    = 1.0;   ///< Baseline intensity (events/sec).
    double alpha = 0.5;   ///< Jump size (excitation per event).
    double beta  = 1.0;   ///< Exponential decay rate.

    /// Branching ratio.  Must be < 1 for stationarity.
    [[nodiscard]] constexpr double branching_ratio() const noexcept {
        return alpha / beta;
    }

    /// Theoretical steady-state expected intensity.
    /// E[lambda] = mu / (1 - alpha/beta).
    [[nodiscard]] constexpr double expected_intensity() const noexcept {
        return mu / (1.0 - branching_ratio());
    }

    /// Returns true if parameters satisfy the stationarity condition.
    [[nodiscard]] constexpr bool is_stationary() const noexcept {
        return alpha < beta && mu > 0.0 && beta > 0.0;
    }
};

// ── Regime presets ──────────────────────────────────────────────────────────

/// Normal market: moderate clustering, branching ratio ~0.3.
inline constexpr HawkesParams NORMAL_REGIME{
    .mu = 5.0, .alpha = 1.5, .beta = 5.0};

/// Flash crash: intense clustering, branching ratio ~0.95.
inline constexpr HawkesParams FLASH_CRASH_REGIME{
    .mu = 5.0, .alpha = 9.5, .beta = 10.0};

// Compile-time verification.
static_assert(NORMAL_REGIME.is_stationary(),
              "NORMAL_REGIME must be stationary");
static_assert(FLASH_CRASH_REGIME.is_stationary(),
              "FLASH_CRASH_REGIME must be stationary");

}  // namespace rcmm
