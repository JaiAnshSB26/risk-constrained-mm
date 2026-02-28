// ============================================================================
// risk-constrained-mm :: cpp/include/sim/hawkes_simulator.hpp
// ============================================================================
// 1D Marked Hawkes Process simulator using Ogata's Modified Thinning
// Algorithm.
//
// The intensity function is:
//   lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
//
// Ogata's algorithm:
//   1. Set upper-bound intensity lambda_bar = lambda(t).
//   2. Draw candidate inter-arrival u ~ Exp(lambda_bar).
//   3. Advance t += u.  Update lambda(t) via exponential decay.
//   4. Accept with probability lambda(t) / lambda_bar (thinning).
//   5. If accepted, emit event, jump lambda by +alpha.
//   6. Repeat.
//
// Marks (Side, Price, Qty) are drawn from simple random distributions
// to produce well-formed Tick structs for downstream replay.
//
// Uses <random> exclusively — no external RNG libraries needed.
// ============================================================================
#pragma once

#include "data/tick.hpp"
#include "lob/types.hpp"
#include "sim/hawkes_params.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace rcmm {

/// Configuration for mark generation (synthetic order-flow properties).
struct MarkConfig {
    Price mid_price     = 50000;  ///< Centre price in tick units.
    Price half_spread   = 10;     ///< Half-spread: prices in [mid-hs, mid+hs].
    Qty   min_qty       = 1;      ///< Minimum order quantity.
    Qty   max_qty       = 10;     ///< Maximum order quantity.
    double buy_prob     = 0.5;    ///< Probability of Side::Bid.
    double add_prob     = 0.6;    ///< Probability of Add action.
    double cancel_prob  = 0.2;    ///< Probability of Cancel action.
    double trade_prob   = 0.2;    ///< Probability of Trade action.
    // Remaining probability goes to Modify.
};

/// 1D Marked Hawkes Process simulator — Ogata's modified thinning.
class HawkesSimulator {
public:
    // ── Construction ────────────────────────────────────────────────────────

    explicit HawkesSimulator(HawkesParams params,
                             MarkConfig marks = {},
                             std::uint64_t seed = 42) noexcept
        : params_(params), marks_(marks), rng_(seed) {
        assert(params_.is_stationary());
    }

    // ── Main API ────────────────────────────────────────────────────────────

    /// Simulate `num_events` Hawkes-process arrivals, starting at
    /// `start_timestamp` (in nanoseconds).
    ///
    /// Returns a pre-allocated vector of Tick structs with monotonically
    /// increasing timestamps, randomly assigned marks, and sequential
    /// order IDs.
    [[nodiscard]] std::vector<Tick> simulate(
            std::size_t num_events,
            Timestamp start_timestamp = 0) {

        std::vector<Tick> ticks;
        ticks.reserve(num_events);

        double t        = 0.0;   // continuous time (seconds from start)
        double lambda_t = params_.mu;   // current intensity

        OrderId next_id = 1;

        while (ticks.size() < num_events) {
            // 1. Upper-bound intensity for this step.
            double lambda_bar = lambda_t;

            // 2. Draw candidate inter-arrival time from Exp(lambda_bar).
            std::exponential_distribution<double> exp_dist(lambda_bar);
            double u = exp_dist(rng_);

            // 3. Advance time.  Decay intensity.
            t += u;
            lambda_t = params_.mu +
                       (lambda_t - params_.mu) *
                       std::exp(-params_.beta * u);

            // 4. Accept / reject (thinning).
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            double accept_p = lambda_t / lambda_bar;
            if (unif(rng_) > accept_p) {
                continue;   // rejected — no event at this time
            }

            // 5. Accepted — emit event and jump intensity by +alpha.
            lambda_t += params_.alpha;

            // Convert continuous seconds to integer nanoseconds.
            auto ts = start_timestamp +
                      static_cast<Timestamp>(t * 1e9);

            Tick tick = generate_mark(ts, next_id++);
            ticks.push_back(tick);
        }

        return ticks;
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] const HawkesParams& params() const noexcept {
        return params_;
    }

    [[nodiscard]] const MarkConfig& mark_config() const noexcept {
        return marks_;
    }

    /// Re-seed the RNG (useful for reproducible experiments).
    void seed(std::uint64_t s) noexcept { rng_.seed(s); }

private:
    // ── Mark generation ─────────────────────────────────────────────────────

    [[nodiscard]] Tick generate_mark(Timestamp ts, OrderId id) {
        Tick tick{};
        tick.timestamp = ts;
        tick.order_id  = id;

        // Side: bid/ask with configured probability.
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        tick.side = (u01(rng_) < marks_.buy_prob) ? Side::Bid : Side::Ask;

        // Price: uniform in [mid - half_spread, mid + half_spread].
        std::uniform_int_distribution<Price> price_dist(
            marks_.mid_price - marks_.half_spread,
            marks_.mid_price + marks_.half_spread);
        tick.price = price_dist(rng_);

        // Qty: uniform in [min_qty, max_qty].
        std::uniform_int_distribution<Qty> qty_dist(
            marks_.min_qty, marks_.max_qty);
        tick.qty = qty_dist(rng_);

        // Action: add / cancel / trade / modify by cumulative probability.
        double r = u01(rng_);
        if (r < marks_.add_prob) {
            tick.action = TickAction::Add;
        } else if (r < marks_.add_prob + marks_.cancel_prob) {
            tick.action = TickAction::Cancel;
        } else if (r < marks_.add_prob + marks_.cancel_prob +
                       marks_.trade_prob) {
            tick.action = TickAction::Trade;
        } else {
            tick.action = TickAction::Modify;
        }
        tick.is_trade = (tick.action == TickAction::Trade);

        return tick;
    }

    // ── Data ────────────────────────────────────────────────────────────────
    HawkesParams   params_;
    MarkConfig     marks_;
    std::mt19937_64 rng_;
};

}  // namespace rcmm
