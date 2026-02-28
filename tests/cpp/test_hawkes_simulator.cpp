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

// ═══════════════════════════════════════════════════════════════════════════
//  Ogata's Thinning Algorithm — NORMAL_REGIME (10k events)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: Normal regime 10k events - empirical intensity matches theory",
          "[hawkes][ogata][normal]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 12345);
    auto ticks = sim.simulate(10000, 0);

    REQUIRE(ticks.size() == 10000u);

    // Compute total elapsed time (in seconds).
    double t_start = static_cast<double>(ticks.front().timestamp) * 1e-9;
    double t_end   = static_cast<double>(ticks.back().timestamp) * 1e-9;
    double duration = t_end - t_start;
    REQUIRE(duration > 0.0);

    // Empirical arrival rate = num_events / duration.
    double empirical = 10000.0 / duration;
    double theoretical = NORMAL_REGIME.expected_intensity();

    // Allow 10% relative tolerance for 10k samples.
    double rel_error = std::abs(empirical - theoretical) / theoretical;
    CHECK(rel_error < 0.10);

    INFO("Normal regime: empirical=" << empirical
         << " theoretical=" << theoretical
         << " rel_error=" << rel_error);
}

TEST_CASE("Hawkes: Normal regime - timestamps strictly increasing",
          "[hawkes][ogata][normal]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(10000, 0);

    REQUIRE(ticks.size() == 10000u);

    for (std::size_t i = 1; i < ticks.size(); ++i) {
        REQUIRE(ticks[i].timestamp > ticks[i - 1].timestamp);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Ogata's Thinning Algorithm — FLASH_CRASH_REGIME (10k events)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: Flash crash regime 10k events - empirical intensity matches theory",
          "[hawkes][ogata][flash]") {
    HawkesSimulator sim(FLASH_CRASH_REGIME, {}, 67890);
    auto ticks = sim.simulate(10000, 0);

    REQUIRE(ticks.size() == 10000u);

    double t_start = static_cast<double>(ticks.front().timestamp) * 1e-9;
    double t_end   = static_cast<double>(ticks.back().timestamp) * 1e-9;
    double duration = t_end - t_start;
    REQUIRE(duration > 0.0);

    double empirical = 10000.0 / duration;
    double theoretical = FLASH_CRASH_REGIME.expected_intensity();

    // Flash crash near-criticality (branching 0.95) causes very high
    // variance;  allow 30% tolerance with 10k samples.
    double rel_error = std::abs(empirical - theoretical) / theoretical;
    CHECK(rel_error < 0.30);

    INFO("Flash crash regime: empirical=" << empirical
         << " theoretical=" << theoretical
         << " rel_error=" << rel_error);
}

TEST_CASE("Hawkes: Flash crash regime - timestamps strictly increasing",
          "[hawkes][ogata][flash]") {
    HawkesSimulator sim(FLASH_CRASH_REGIME, {}, 11111);
    auto ticks = sim.simulate(10000, 0);

    REQUIRE(ticks.size() == 10000u);

    for (std::size_t i = 1; i < ticks.size(); ++i) {
        REQUIRE(ticks[i].timestamp > ticks[i - 1].timestamp);
    }
}

TEST_CASE("Hawkes: Flash crash arrivals are more clustered than normal",
          "[hawkes][ogata][regime]") {
    // Flash crash should produce events in a shorter time window
    // (higher intensity => same #events in less wall-clock time).
    HawkesSimulator sim_normal(NORMAL_REGIME, {}, 42);
    HawkesSimulator sim_flash(FLASH_CRASH_REGIME, {}, 42);

    auto normal_ticks = sim_normal.simulate(5000, 0);
    auto flash_ticks  = sim_flash.simulate(5000, 0);

    double normal_duration =
        static_cast<double>(normal_ticks.back().timestamp -
                            normal_ticks.front().timestamp) * 1e-9;
    double flash_duration =
        static_cast<double>(flash_ticks.back().timestamp -
                            flash_ticks.front().timestamp) * 1e-9;

    // Flash crash should finish in substantially less time.
    CHECK(flash_duration < normal_duration);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mark Generation
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: marks - order IDs are sequential starting from 1",
          "[hawkes][marks]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(100, 0);

    for (std::size_t i = 0; i < ticks.size(); ++i) {
        CHECK(ticks[i].order_id == static_cast<OrderId>(i + 1));
    }
}

TEST_CASE("Hawkes: marks - side distribution roughly 50/50",
          "[hawkes][marks]") {
    MarkConfig mc{};
    mc.buy_prob = 0.5;
    HawkesSimulator sim(NORMAL_REGIME, mc, 42);
    auto ticks = sim.simulate(10000, 0);

    std::size_t bid_count = 0;
    for (const auto& t : ticks) {
        if (t.side == Side::Bid) ++bid_count;
    }
    double bid_ratio = static_cast<double>(bid_count) / 10000.0;
    // 50% +/- 3%
    CHECK(bid_ratio > 0.47);
    CHECK(bid_ratio < 0.53);
}

TEST_CASE("Hawkes: marks - prices within configured range",
          "[hawkes][marks]") {
    MarkConfig mc{};
    mc.mid_price   = 50000;
    mc.half_spread = 10;
    HawkesSimulator sim(NORMAL_REGIME, mc, 42);
    auto ticks = sim.simulate(1000, 0);

    for (const auto& t : ticks) {
        CHECK(t.price >= mc.mid_price - mc.half_spread);
        CHECK(t.price <= mc.mid_price + mc.half_spread);
    }
}

TEST_CASE("Hawkes: marks - quantities within configured range",
          "[hawkes][marks]") {
    MarkConfig mc{};
    mc.min_qty = 1;
    mc.max_qty = 10;
    HawkesSimulator sim(NORMAL_REGIME, mc, 42);
    auto ticks = sim.simulate(1000, 0);

    for (const auto& t : ticks) {
        CHECK(t.qty >= mc.min_qty);
        CHECK(t.qty <= mc.max_qty);
    }
}

TEST_CASE("Hawkes: marks - action distribution roughly matches config",
          "[hawkes][marks]") {
    MarkConfig mc{};
    mc.add_prob    = 0.6;
    mc.cancel_prob = 0.2;
    mc.trade_prob  = 0.2;
    // modify = 1 - 0.6 - 0.2 - 0.2 = 0.0
    HawkesSimulator sim(NORMAL_REGIME, mc, 42);
    auto ticks = sim.simulate(10000, 0);

    std::size_t adds = 0, cancels = 0, trades = 0, modifies = 0;
    for (const auto& t : ticks) {
        switch (t.action) {
        case TickAction::Add:    ++adds;     break;
        case TickAction::Cancel: ++cancels;  break;
        case TickAction::Trade:  ++trades;   break;
        case TickAction::Modify: ++modifies; break;
        }
    }

    double add_r = static_cast<double>(adds) / 10000.0;
    double can_r = static_cast<double>(cancels) / 10000.0;
    double trd_r = static_cast<double>(trades) / 10000.0;

    // Allow 4% tolerance on 10k samples.
    CHECK(add_r > 0.56);
    CHECK(add_r < 0.64);
    CHECK(can_r > 0.16);
    CHECK(can_r < 0.24);
    CHECK(trd_r > 0.16);
    CHECK(trd_r < 0.24);
}

TEST_CASE("Hawkes: marks - is_trade flag matches action",
          "[hawkes][marks]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(1000, 0);

    for (const auto& t : ticks) {
        CHECK(t.is_trade == (t.action == TickAction::Trade));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Reproducibility & Edge Cases
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: same seed produces identical output",
          "[hawkes][repro]") {
    HawkesSimulator sim1(NORMAL_REGIME, {}, 42);
    HawkesSimulator sim2(NORMAL_REGIME, {}, 42);

    auto t1 = sim1.simulate(500, 1000);
    auto t2 = sim2.simulate(500, 1000);

    REQUIRE(t1.size() == t2.size());
    for (std::size_t i = 0; i < t1.size(); ++i) {
        CHECK(t1[i].timestamp == t2[i].timestamp);
        CHECK(t1[i].order_id  == t2[i].order_id);
        CHECK(t1[i].price     == t2[i].price);
        CHECK(t1[i].qty       == t2[i].qty);
        CHECK(t1[i].side      == t2[i].side);
        CHECK(t1[i].action    == t2[i].action);
    }
}

TEST_CASE("Hawkes: reseed produces different output",
          "[hawkes][repro]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto t1 = sim.simulate(100, 0);

    sim.seed(99);
    auto t2 = sim.simulate(100, 0);

    // Extremely unlikely for all 100 timestamps to match.
    bool all_same = true;
    for (std::size_t i = 0; i < t1.size(); ++i) {
        if (t1[i].timestamp != t2[i].timestamp) {
            all_same = false;
            break;
        }
    }
    CHECK_FALSE(all_same);
}

TEST_CASE("Hawkes: start_timestamp offset is applied",
          "[hawkes][edge]") {
    Timestamp offset = 1'000'000'000'000LL;  // 1000 seconds in ns
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(100, offset);

    REQUIRE(ticks.size() == 100u);
    CHECK(ticks[0].timestamp >= offset);
    CHECK(ticks[99].timestamp > ticks[0].timestamp);
}

TEST_CASE("Hawkes: small simulation (10 events) still works",
          "[hawkes][edge]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(10, 0);

    REQUIRE(ticks.size() == 10u);
    for (std::size_t i = 1; i < ticks.size(); ++i) {
        CHECK(ticks[i].timestamp > ticks[i - 1].timestamp);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Inter-arrival time statistics
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Hawkes: Normal regime - all inter-arrival times positive",
          "[hawkes][stats]") {
    HawkesSimulator sim(NORMAL_REGIME, {}, 42);
    auto ticks = sim.simulate(10000, 0);

    for (std::size_t i = 1; i < ticks.size(); ++i) {
        CHECK(ticks[i].timestamp - ticks[i - 1].timestamp > 0);
    }
}

TEST_CASE("Hawkes: Flash crash - all inter-arrival times positive",
          "[hawkes][stats]") {
    HawkesSimulator sim(FLASH_CRASH_REGIME, {}, 42);
    auto ticks = sim.simulate(10000, 0);

    for (std::size_t i = 1; i < ticks.size(); ++i) {
        CHECK(ticks[i].timestamp - ticks[i - 1].timestamp > 0);
    }
}

TEST_CASE("Hawkes: Flash crash mean inter-arrival < normal mean inter-arrival",
          "[hawkes][stats]") {
    HawkesSimulator sim_n(NORMAL_REGIME, {}, 42);
    HawkesSimulator sim_f(FLASH_CRASH_REGIME, {}, 42);

    auto tn = sim_n.simulate(10000, 0);
    auto tf = sim_f.simulate(10000, 0);

    // Mean inter-arrival in nanoseconds.
    double normal_mean = 0.0;
    for (std::size_t i = 1; i < tn.size(); ++i) {
        normal_mean += static_cast<double>(
            tn[i].timestamp - tn[i - 1].timestamp);
    }
    normal_mean /= static_cast<double>(tn.size() - 1u);

    double flash_mean = 0.0;
    for (std::size_t i = 1; i < tf.size(); ++i) {
        flash_mean += static_cast<double>(
            tf[i].timestamp - tf[i - 1].timestamp);
    }
    flash_mean /= static_cast<double>(tf.size() - 1u);

    // Flash crash should have smaller mean inter-arrival time.
    CHECK(flash_mean < normal_mean);
}
