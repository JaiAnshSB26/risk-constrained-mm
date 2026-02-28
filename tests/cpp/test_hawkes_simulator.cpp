// ============================================================================
// risk-constrained-mm :: tests/cpp/test_hawkes_simulator.cpp
// ============================================================================
// Catch2 test suite for the 1D Marked Hawkes Process simulator (Phase 5).
//
// Covers:
//   * HawkesParams: stationarity check, branching ratio, expected intensity
//   * Regime presets: NORMAL_REGIME, FLASH_CRASH_REGIME
//   * Ogata's thinning: 10k events, empirical vs theoretical intensity
//   * Strict monotonicity: no negative inter-arrival times
//   * Mark generation: side, price, qty, action distributions
//   * Two regimes: Normal and Flash Crash
//   * Reproducibility: same seed gives same output
//   * Tick struct validity
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sim/hawkes_params.hpp"
#include "sim/hawkes_simulator.hpp"
#include "data/tick.hpp"
#include "lob/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

using namespace rcmm;

// ═══════════════════════════════════════════════════════════════════════════
//  HawkesParams
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: params stationarity check",
          "[hawkes][params]") {
    HawkesParams good{.mu = 1.0, .alpha = 0.5, .beta = 1.0};
    CHECK(good.is_stationary());
    CHECK(good.branching_ratio() == 0.5);

    HawkesParams bad{.mu = 1.0, .alpha = 1.5, .beta = 1.0};
    CHECK_FALSE(bad.is_stationary());
}

TEST_CASE("Hawkes: expected intensity formula",
          "[hawkes][params]") {
    // E[lambda] = mu / (1 - alpha/beta)
    HawkesParams p{.mu = 5.0, .alpha = 2.5, .beta = 5.0};
    double expected = 5.0 / (1.0 - 0.5);  // = 10.0
    CHECK(p.expected_intensity() == expected);
}

TEST_CASE("Hawkes: branching ratio computation",
          "[hawkes][params]") {
    HawkesParams p{.mu = 5.0, .alpha = 1.5, .beta = 5.0};
    CHECK(p.branching_ratio() == Catch::Approx(0.3).epsilon(1e-12));

    HawkesParams fc{.mu = 5.0, .alpha = 9.5, .beta = 10.0};
    CHECK(fc.branching_ratio() == Catch::Approx(0.95).epsilon(1e-12));
}

TEST_CASE("Hawkes: NORMAL_REGIME preset is stationary",
          "[hawkes][params]") {
    CHECK(NORMAL_REGIME.is_stationary());
    CHECK(NORMAL_REGIME.branching_ratio() == Catch::Approx(0.3).epsilon(1e-12));
    // E[lambda] = 5 / (1 - 0.3) = 5 / 0.7 ~ 7.142857
    CHECK(NORMAL_REGIME.expected_intensity() ==
          Catch::Approx(5.0 / 0.7).epsilon(1e-9));
}

TEST_CASE("Hawkes: FLASH_CRASH_REGIME preset is stationary",
          "[hawkes][params]") {
    CHECK(FLASH_CRASH_REGIME.is_stationary());
    CHECK(FLASH_CRASH_REGIME.branching_ratio() ==
          Catch::Approx(0.95).epsilon(1e-12));
    // E[lambda] = 5 / (1 - 0.95) = 5 / 0.05 = 100
    CHECK(FLASH_CRASH_REGIME.expected_intensity() ==
          Catch::Approx(100.0).epsilon(1e-9));
}
